//! High-level API: connect to a phone over USB and stream its camera to a V4L2 device.

use crate::adb::{self, AdbDevice};
use crate::convert::i420_to_yuyv;
use crate::decode::Decoder;
use crate::error::{Error, Result};
use crate::protocol;
use crate::sink::V4l2Sink;
use std::io::BufReader;
use std::net::TcpStream;
use std::path::Path;
use std::process::Child;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Facing {
    Front,
    Back,
}

impl Facing {
    fn as_arg(self) -> &'static str {
        match self {
            Facing::Front => "front",
            Facing::Back => "back",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectOptions {
    pub serial: Option<String>,
    pub facing: Facing,
    /// Requested camera capture size, e.g. `(1280, 720)`. `None` lets the phone pick.
    pub resolution: Option<(u32, u32)>,
    pub max_fps: Option<u32>,
    /// Target H.264 bitrate in bits per second. Higher means crisper detail (text,
    /// document edges) at the cost of bandwidth. `None` uses the server default.
    pub bitrate_bps: Option<u32>,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            serial: None,
            facing: Facing::Back,
            resolution: None,
            max_fps: None,
            bitrate_bps: None,
        }
    }
}

const SERVER_JAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scrcpy-server.jar"));

pub struct CameraSession {
    device: AdbDevice,
    server_process: Child,
    port: u16,
    socket: BufReader<TcpStream>,
    pub meta: protocol::CodecMeta,
}

impl CameraSession {
    /// Pushes the embedded scrcpy-server jar, starts it in camera mode, and connects
    /// to its video socket.
    pub fn connect(opts: ConnectOptions) -> Result<Self> {
        let device = match &opts.serial {
            Some(s) => AdbDevice::with_serial(s.clone()),
            None => AdbDevice::autodetect()?,
        };

        device.push_server_jar(SERVER_JAR)?;

        let scid = adb::random_scid_hex8();
        let port = device.forward(&scid)?;

        let mut server_args = vec![
            format!("scid={scid}"),
            "video_source=camera".to_string(),
            format!("camera_facing={}", opts.facing.as_arg()),
            "audio=false".to_string(),
            "control=false".to_string(),
            "cleanup=false".to_string(),
            "video_codec=h264".to_string(),
            // We reach the server through `adb forward`, so it must listen rather than
            // connect back to us (the default assumes `adb reverse`).
            "tunnel_forward=true".to_string(),
            "send_device_meta=false".to_string(),
            // In forward mode adb accepts our TCP connection before the server socket
            // exists; the dummy byte is how we know we reached the server itself.
            "send_dummy_byte=true".to_string(),
            "send_codec_meta=true".to_string(),
            "send_frame_meta=true".to_string(),
        ];
        if let Some((w, h)) = opts.resolution {
            server_args.push(format!("camera_size={w}x{h}"));
        }
        if let Some(fps) = opts.max_fps {
            server_args.push(format!("max_fps={fps}"));
        }
        if let Some(bps) = opts.bitrate_bps {
            server_args.push(format!("video_bit_rate={bps}"));
        }

        let mut server_process = match device.start_server(&server_args) {
            Ok(p) => p,
            Err(e) => {
                device.remove_forward(port);
                return Err(e);
            }
        };

        let socket = match connect_with_retry(port, Duration::from_secs(15)) {
            Ok(s) => s,
            Err(e) => return Err(abort(&device, &mut server_process, port, e)),
        };
        let mut socket = BufReader::new(socket);
        let meta = match protocol::read_codec_meta(&mut socket) {
            Ok(m) => m,
            Err(e) => return Err(abort(&device, &mut server_process, port, e)),
        };

        log::info!(
            "connected to camera stream: {}x{} from device {:?}",
            meta.width,
            meta.height,
            device.serial
        );

        Ok(Self {
            device,
            server_process,
            port,
            socket,
            meta,
        })
    }

    /// Blocks, decoding the camera stream and writing frames to `device` (a
    /// `/dev/videoN` v4l2loopback node) until the stream ends or an error occurs.
    pub fn run_to_v4l2(&mut self, device_path: &Path) -> Result<()> {
        let mut decoder = Decoder::new()?;
        let mut sink = V4l2Sink::open(device_path, self.meta.width, self.meta.height)?;
        let mut yuyv = vec![0u8; (self.meta.width * self.meta.height * 2) as usize];

        // A document camera may run for hours; a single corrupt packet (dropped
        // reference frame, bit error over USB) shouldn't kill the session. We skip
        // failed decodes and only give up if they never recover -- which is also how
        // a genuinely undecodable stream (e.g. a resolution openh264 can't handle)
        // surfaces as a clear error instead of an endless silent stall.
        const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 300;
        let mut consecutive_errors = 0u32;
        let mut decoded_any = false;

        while let Some(packet) = protocol::read_frame_packet(&mut self.socket)? {
            match decoder.decode(&packet.data) {
                Ok(Some(frame)) => {
                    decoded_any = true;
                    consecutive_errors = 0;
                    i420_to_yuyv(&frame, &mut yuyv);
                    sink.write_frame(&yuyv)?;
                }
                Ok(None) => {}
                Err(e) => {
                    consecutive_errors += 1;
                    if consecutive_errors >= MAX_CONSECUTIVE_DECODE_ERRORS {
                        if decoded_any {
                            return Err(e);
                        }
                        return Err(Error::Decode(format!(
                            "no frame decoded after {consecutive_errors} attempts \
                             at {}x{}; the phone's camera resolution may exceed what \
                             openh264 can decode (try --resolution 3840x2160 or lower). \
                             Last error: {e}",
                            self.meta.width, self.meta.height
                        )));
                    }
                    log::debug!("skipping undecodable packet ({e})");
                }
            }
        }

        Ok(())
    }
}

impl Drop for CameraSession {
    fn drop(&mut self) {
        adb::close_stdin(&mut self.server_process);
        let _ = self.server_process.kill();
        let _ = self.server_process.wait();
        self.device.remove_forward(self.port);
    }
}

/// Tears down a half-established session and folds the server's own output into the
/// error, since that's where the useful diagnostics (camera/encoder failures) end up.
fn abort(device: &AdbDevice, server: &mut Child, port: u16, err: Error) -> Error {
    adb::close_stdin(server);
    let _ = server.kill();
    let _ = server.wait();
    device.remove_forward(port);

    let mut output = String::new();
    if let Some(mut out) = server.stdout.take() {
        let _ = std::io::Read::read_to_string(&mut out, &mut output);
    }
    if let Some(mut err_out) = server.stderr.take() {
        let _ = std::io::Read::read_to_string(&mut err_out, &mut output);
    }
    let output = output.trim();
    if output.is_empty() {
        err
    } else {
        Error::Protocol(format!("{err}\nscrcpy server output:\n{output}"))
    }
}

/// Connects through the adb forward and keeps reconnecting until the server's dummy
/// byte arrives -- adb accepts the local TCP connection immediately and only then
/// tries the device-side socket, so an early connection reads EOF instead.
fn connect_with_retry(port: u16, timeout: Duration) -> Result<TcpStream> {
    use std::io::Read;

    let deadline = std::time::Instant::now() + timeout;
    let addr = format!("127.0.0.1:{port}");
    let mut last_err = String::new();
    while std::time::Instant::now() < deadline {
        match TcpStream::connect(&addr) {
            Ok(mut s) => {
                let mut dummy = [0u8; 1];
                match s.read(&mut dummy) {
                    Ok(1) => return Ok(s),
                    Ok(_) => last_err = "server not listening yet".to_string(),
                    Err(e) => last_err = e.to_string(),
                }
            }
            Err(e) => last_err = e.to_string(),
        }
        log::debug!("waiting for scrcpy server on {addr}: {last_err}");
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(Error::Protocol(format!(
        "could not reach scrcpy server on {addr} within {timeout:?}: {last_err}"
    )))
}
