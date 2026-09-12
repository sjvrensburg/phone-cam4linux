//! High-level API: connect to a phone (USB or TCP/IP ADB) and stream its camera to a
//! V4L2 device.

use crate::adb::{self, AdbDevice};
use crate::convert::i420_to_yuyv;
use crate::decode::{self, Decoder};
use crate::error::{Error, Result};
use crate::protocol;
use crate::sink::V4l2Sink;
use std::io::BufReader;
use std::io::Read;
use std::net::TcpStream;
use std::path::Path;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    /// ADB serial to use; `None` autodetects the sole attached device.
    pub serial: Option<String>,
    /// `host[:port]` of a phone in TCP/IP ADB mode. When set, `adb connect` is issued
    /// (again) on every `connect()`, so a reconnect loop recovers from Wi-Fi drops,
    /// and `serial` is ignored.
    pub tcp_address: Option<String>,
    pub facing: Facing,
    /// Requested camera capture size, e.g. `(1280, 720)`. `None` lets the phone pick.
    pub resolution: Option<(u32, u32)>,
    pub max_fps: Option<u32>,
    /// Target H.264 bitrate in bits per second. Higher means crisper detail (text,
    /// document edges) at the cost of bandwidth. `None` uses the server default.
    pub bitrate_bps: Option<u32>,
    /// Which H.264 decoder to use; see [`decode::Backend`].
    pub decoder: decode::Backend,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            serial: None,
            tcp_address: None,
            facing: Facing::Back,
            resolution: None,
            max_fps: None,
            bitrate_bps: None,
            decoder: decode::Backend::default(),
        }
    }
}

pub(crate) const SERVER_JAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scrcpy-server.jar"));

/// How long the stream may go without a single byte before the session is declared
/// dead. The camera produces frames continuously (even `--fps 1` is well inside
/// this), so silence means the phone, cable, or server went away.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(10);

pub struct CameraSession {
    decoder_backend: decode::Backend,
    frames_decoded: u64,
    device: AdbDevice,
    server_process: Child,
    server_log: ServerLog,
    port: u16,
    socket: BufReader<TcpStream>,
    pub meta: protocol::CodecMeta,
}

impl CameraSession {
    /// Pushes the embedded scrcpy-server jar, starts it in camera mode, and connects
    /// to its video socket.
    pub fn connect(opts: ConnectOptions) -> Result<Self> {
        Self::connect_with_stop(opts, &AtomicBool::new(false))
    }

    /// Like [`Self::connect`], but gives up (cleaning up the server and forward) as
    /// soon as `stop` is raised while waiting for the server to come up.
    pub fn connect_with_stop(opts: ConnectOptions, stop: &AtomicBool) -> Result<Self> {
        let device = match (&opts.tcp_address, &opts.serial) {
            (Some(addr), _) => AdbDevice::connect_tcp(addr)?,
            (None, Some(s)) => AdbDevice::with_serial(s.clone()),
            (None, None) => AdbDevice::autodetect()?,
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
        let server_log = ServerLog::attach(&mut server_process);

        let socket = match connect_with_retry(port, Duration::from_secs(15), stop) {
            Ok(s) => s,
            Err(e) => return Err(abort(&device, &mut server_process, &server_log, port, e)),
        };
        let mut socket = BufReader::new(socket);
        // The codec header only arrives once the camera and encoder are up, which can
        // fail silently on the phone; bound the wait and keep it interruptible.
        let meta = match read_codec_meta_with_stop(&mut socket, stop) {
            Ok(m) => m,
            Err(e) => return Err(abort(&device, &mut server_process, &server_log, port, e)),
        };

        log::info!(
            "connected to camera stream: {}x{} from device {:?}",
            meta.width,
            meta.height,
            device.serial
        );

        Ok(Self {
            decoder_backend: opts.decoder,
            frames_decoded: 0,
            device,
            server_process,
            server_log,
            port,
            socket,
            meta,
        })
    }

    /// Convenience for [`Self::run`]: opens `device_path` (a `/dev/videoN` v4l2loopback
    /// node) at the stream's size and runs until the stream ends or fails.
    pub fn run_to_v4l2(&mut self, device_path: &Path) -> Result<()> {
        let mut sink = V4l2Sink::open(device_path, self.meta.width, self.meta.height)?;
        self.run(&mut sink, &AtomicBool::new(false))
    }

    /// Frames decoded so far by [`Self::run`] -- lets a reconnect loop tell a session
    /// that actually worked from one that failed right after the handshake.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Blocks, decoding the camera stream and writing frames to `sink` (which must
    /// have been opened at [`Self::meta`]'s size) until:
    ///
    /// - `stop` becomes true (checked at least every 500 ms) -> `Ok(())`;
    /// - the phone closes the stream -> `Ok(())`;
    /// - nothing arrives for [`STALL_TIMEOUT`] -> [`Error::StreamStalled`];
    /// - any other error.
    pub fn run(&mut self, sink: &mut V4l2Sink, stop: &AtomicBool) -> Result<()> {
        let mut decoder = Decoder::with_backend(self.decoder_backend)?;
        log::debug!("decoding with {}", decoder.backend().name());
        let mut yuyv = vec![0u8; (self.meta.width * self.meta.height * 2) as usize];

        // A short socket timeout lets us notice `stop` and stalls between reads
        // without losing partial packets: `Interruptible` retries the *same* read.
        self.socket
            .get_ref()
            .set_read_timeout(Some(Duration::from_millis(500)))?;
        let mut reader = Interruptible {
            inner: &mut self.socket,
            stop,
            last_data: Instant::now(),
        };

        // A document camera may run for hours; a single corrupt packet (dropped
        // reference frame, bit error over USB) shouldn't kill the session. We skip
        // failed decodes and only give up if they never recover -- which is also how
        // a genuinely undecodable stream (e.g. a resolution openh264 can't handle)
        // surfaces as a clear error instead of an endless silent stall.
        const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 300;
        let mut consecutive_errors = 0u32;
        let mut decoded_any = false;

        let mut stats = StreamStats::default();
        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let packet = match protocol::read_frame_packet(&mut reader) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    log::info!("phone closed the video stream");
                    return Ok(());
                }
                Err(Error::Io(e)) if e.get_ref().is_some_and(|i| i.is::<StopRequested>()) => {
                    return Ok(());
                }
                Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::TimedOut => {
                    return Err(Error::StreamStalled(STALL_TIMEOUT));
                }
                Err(e) => return Err(e),
            };
            stats.packet(packet.data.len());
            match decoder.decode(&packet.data) {
                Ok(Some(frame)) => {
                    stats.frame();
                    self.frames_decoded += 1;
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
                             the {} decoder can handle (try --resolution max). \
                             Last error: {e}\nscrcpy server output:\n{}",
                            self.meta.width,
                            self.meta.height,
                            decoder.backend().name(),
                            self.server_log.collected()
                        )));
                    }
                    log::debug!("skipping undecodable packet ({e})");
                }
            }
        }
    }
}

/// Marker payload of the `io::Error` used to unwind out of a blocking read when the
/// caller's stop flag is raised.
#[derive(Debug)]
struct StopRequested;

impl std::fmt::Display for StopRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stop requested")
    }
}

impl std::error::Error for StopRequested {}

/// Wraps the socket so that read timeouts become opportunities to check the stop
/// flag and the stall deadline, transparently resuming the read otherwise.
struct Interruptible<'a, R: Read> {
    inner: &'a mut R,
    stop: &'a AtomicBool,
    last_data: Instant,
}

impl<R: Read> Read for Interruptible<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            match self.inner.read(buf) {
                Ok(n) => {
                    self.last_data = Instant::now();
                    return Ok(n);
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    if self.stop.load(Ordering::Relaxed) {
                        return Err(std::io::Error::other(StopRequested));
                    }
                    if self.last_data.elapsed() > STALL_TIMEOUT {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "stream stalled",
                        ));
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Periodic debug-level throughput report, so a stalled pipeline can be localised
/// (packets arriving but nothing decoding, vs. nothing arriving at all).
#[derive(Default)]
struct StreamStats {
    started: Option<std::time::Instant>,
    packets: u64,
    bytes: u64,
    frames: u64,
}

impl StreamStats {
    const INTERVAL: Duration = Duration::from_secs(5);

    fn packet(&mut self, len: usize) {
        self.packets += 1;
        self.bytes += len as u64;
        self.maybe_report();
    }

    fn frame(&mut self) {
        self.frames += 1;
    }

    fn maybe_report(&mut self) {
        let now = std::time::Instant::now();
        let Some(started) = self.started else {
            self.started = Some(now);
            return;
        };
        let elapsed = now.duration_since(started);
        if elapsed < Self::INTERVAL {
            return;
        }
        let secs = elapsed.as_secs_f64();
        log::debug!(
            "stream: {} packets ({:.1} Mbit/s), {} frames decoded ({:.1} fps)",
            self.packets,
            self.bytes as f64 * 8.0 / 1e6 / secs,
            self.frames,
            self.frames as f64 / secs
        );
        *self = Self::default();
        self.started = Some(now);
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
fn abort(
    device: &AdbDevice,
    server: &mut Child,
    server_log: &ServerLog,
    port: u16,
    err: Error,
) -> Error {
    adb::close_stdin(server);
    let _ = server.kill();
    let _ = server.wait();
    device.remove_forward(port);

    let output = server_log.collected();
    let output = output.trim();
    if output.is_empty() {
        err
    } else {
        Error::Protocol(format!("{err}\nscrcpy server output:\n{output}"))
    }
}

/// Relays the scrcpy server's stdout/stderr into our log as it happens (the server
/// reports camera and encoder failures there, and only there), while also keeping
/// a copy so a failed handshake can quote it.
#[derive(Clone, Default)]
struct ServerLog {
    lines: Arc<Mutex<Vec<String>>>,
}

impl ServerLog {
    fn attach(child: &mut Child) -> Self {
        let log = Self::default();
        for pipe in [
            child
                .stdout
                .take()
                .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
            child
                .stderr
                .take()
                .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let lines = Arc::clone(&log.lines);
            std::thread::spawn(move || {
                use std::io::BufRead;
                for line in BufReader::new(pipe)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    let line = line.trim_end().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    if line.contains("ERROR") || line.contains("Exception") {
                        log::warn!("scrcpy server: {line}");
                    } else {
                        log::debug!("scrcpy server: {line}");
                    }
                    lines.lock().unwrap().push(line);
                }
            });
        }
        log
    }

    fn collected(&self) -> String {
        // Give the reader threads a moment to drain what the dying server printed.
        std::thread::sleep(Duration::from_millis(200));
        self.lines.lock().unwrap().join("\n")
    }
}

/// Connects through the adb forward and keeps reconnecting until the server's dummy
/// byte arrives -- adb accepts the local TCP connection immediately and only then
/// tries the device-side socket, so an early connection reads EOF instead.
fn connect_with_retry(port: u16, timeout: Duration, stop: &AtomicBool) -> Result<TcpStream> {
    let deadline = Instant::now() + timeout;
    let addr = format!("127.0.0.1:{port}");
    let mut last_err = String::new();
    while Instant::now() < deadline {
        if stop.load(Ordering::Relaxed) {
            return Err(Error::Protocol(
                "interrupted while waiting for the server".into(),
            ));
        }
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

/// Reads the codec header with a bounded, `stop`-interruptible wait (the socket keeps
/// a 500 ms read timeout afterwards; `run` sets its own).
fn read_codec_meta_with_stop(
    socket: &mut BufReader<TcpStream>,
    stop: &AtomicBool,
) -> Result<protocol::CodecMeta> {
    socket
        .get_ref()
        .set_read_timeout(Some(Duration::from_millis(500)))?;
    let mut reader = Interruptible {
        inner: socket,
        stop,
        last_data: Instant::now(),
    };
    protocol::read_codec_meta(&mut reader).map_err(|e| match e {
        Error::Io(io) if io.get_ref().is_some_and(|i| i.is::<StopRequested>()) => {
            Error::Protocol("interrupted while waiting for the codec header".into())
        }
        Error::Io(io) if io.kind() == std::io::ErrorKind::TimedOut => Error::Protocol(format!(
            "no codec header from the server within {STALL_TIMEOUT:?} \
             (camera or encoder failed to start?)"
        )),
        other => other,
    })
}
