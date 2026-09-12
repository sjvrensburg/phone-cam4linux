//! pc4l-gui: use an Android phone as a document camera on the desktop -- live view,
//! drag a region to zoom, capture, save. The phone side and the decode pipeline are
//! the `phone-cam4linux` library; this crate is the window and the reconnect policy.

mod app;
#[cfg(feature = "local-model")]
mod local;
mod stream;
mod transcribe;

use anyhow::{Context, Result};
use clap::Parser;
use phone_cam4linux::cameras::is_usable_size;
use phone_cam4linux::convert::Rotation;
use phone_cam4linux::decode::Backend;
use phone_cam4linux::{ConnectOptions, Facing};
use std::path::PathBuf;
use stream::{Resolution, StreamConfig, Worker};

/// Live view, crop and capture an Android phone's camera.
#[derive(Parser, Debug)]
#[command(name = "pc4l-gui", version)]
struct Args {
    /// ADB serial of the device to use (autodetected if omitted and only one is attached).
    #[arg(long)]
    serial: Option<String>,

    /// Use the phone over Wi-Fi: `adb connect` to HOST[:PORT] instead of USB. See
    /// `pc4l --tcpip` for the one-time switch.
    #[arg(long, value_name = "HOST[:PORT]", conflicts_with = "serial")]
    connect: Option<String>,

    /// Which camera to start with (switchable in the window).
    #[arg(long, value_enum, default_value = "back")]
    facing: FacingArg,

    /// Capture resolution, e.g. 1280x720, or `max` for the largest the camera offers
    /// that the decoder can handle. Defaults to `max`: this is a document camera.
    #[arg(long, default_value = "max")]
    resolution: String,

    /// H.264 decoder; `ffmpeg` (if compiled in) lifts openh264's ~3840x2160 ceiling.
    #[arg(long, value_enum, default_value_t = DecoderArg::default())]
    decoder: DecoderArg,

    /// Requested max frame rate.
    #[arg(long)]
    fps: Option<u32>,

    /// H.264 bitrate in megabits per second.
    #[arg(long, default_value_t = 30)]
    bitrate: u32,

    /// Also write every frame to this v4l2loopback device (e.g. /dev/video10), so the
    /// same stream is a webcam for other apps while the window is open.
    #[arg(long, value_name = "/dev/videoN")]
    device: Option<PathBuf>,

    /// Turn the picture clockwise by this many degrees at startup (a phone on a
    /// stand is usually mounted sideways). Also changeable in the window.
    #[arg(long, value_parser = ["0", "90", "180", "270"], default_value = "0")]
    rotate: String,

    /// Development aid: start with this crop selected, in view pixels.
    #[arg(long, hide = true, value_name = "X,Y,W,H")]
    dev_crop: Option<String>,

    /// Where captures are saved. Defaults to ~/Pictures/pc4l.
    #[arg(long)]
    save_dir: Option<PathBuf>,

    /// Development aid: after this many seconds, write a PNG screenshot of the window
    /// to --screenshot-path and exit.
    #[arg(long, hide = true, requires = "screenshot_path")]
    screenshot_after: Option<f32>,

    #[arg(long, hide = true)]
    screenshot_path: Option<PathBuf>,

    /// Development aid: read the crop (or page) with the first backend as soon as a
    /// frame arrives.
    #[arg(long, hide = true)]
    dev_read: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum FacingArg {
    Front,
    Back,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DecoderArg {
    Openh264,
    #[cfg(feature = "ffmpeg")]
    Ffmpeg,
}

impl Default for DecoderArg {
    fn default() -> Self {
        match Backend::default() {
            Backend::Openh264 => DecoderArg::Openh264,
            #[cfg(feature = "ffmpeg")]
            Backend::Ffmpeg => DecoderArg::Ffmpeg,
        }
    }
}

impl From<DecoderArg> for Backend {
    fn from(d: DecoderArg) -> Self {
        match d {
            DecoderArg::Openh264 => Backend::Openh264,
            #[cfg(feature = "ffmpeg")]
            DecoderArg::Ffmpeg => Backend::Ffmpeg,
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let decoder = Backend::from(args.decoder);
    let resolution = match args.resolution.as_str() {
        "max" => Resolution::Max,
        "default" => Resolution::PhoneDefault,
        s => {
            let (w, h) = s
                .split_once('x')
                .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                .with_context(|| {
                    format!("--resolution {s:?} must be WIDTHxHEIGHT, max or default")
                })?;
            anyhow::ensure!(
                is_usable_size(w, h, decoder),
                "{w}x{h} is not usable with the {} decoder (see `pc4l --list-sizes`)",
                decoder.name()
            );
            Resolution::Fixed(w, h)
        }
    };
    let config = StreamConfig {
        options: ConnectOptions {
            serial: args.serial,
            tcp_address: args.connect,
            facing: match args.facing {
                FacingArg::Back => Facing::Back,
                FacingArg::Front => Facing::Front,
            },
            resolution: None,
            max_fps: args.fps,
            bitrate_bps: Some(args.bitrate.saturating_mul(1_000_000)),
            decoder,
        },
        resolution,
        tee_device: args.device,
    };
    let save_dir = args.save_dir.unwrap_or_else(|| {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default();
        home.join("Pictures").join("pc4l")
    });

    let rotation = match args.rotate.as_str() {
        "90" => Rotation::Cw90,
        "180" => Rotation::Cw180,
        "270" => Rotation::Cw270,
        _ => Rotation::None,
    };
    let dev_crop = args
        .dev_crop
        .as_deref()
        .map(|s| {
            let v: Vec<usize> = s
                .split(',')
                .map(|n| n.trim().parse())
                .collect::<Result<_, _>>()?;
            anyhow::ensure!(v.len() == 4, "--dev-crop wants X,Y,W,H");
            Ok(app::Crop {
                x: v[0],
                y: v[1],
                w: v[2],
                h: v[3],
            })
        })
        .transpose()?;
    let dev_read = args.dev_read;
    let screenshot = args
        .screenshot_after
        .zip(args.screenshot_path)
        .map(|(secs, path)| (std::time::Duration::from_secs_f32(secs), path));

    let backends: Vec<std::sync::Arc<dyn transcribe::Transcriber>> =
        match transcribe::Config::load_or_create() {
            Ok(config) => config
                .backends
                .iter()
                .filter_map(|b| b.build().map(Into::into))
                .collect(),
            Err(e) => {
                log::error!("{e:#}; no transcription backends available");
                Vec::new()
            }
        };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Phone camera")
            .with_inner_size([1400.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "pc4l-gui",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            let worker = Worker::start(config, move || ctx.request_repaint());
            let mut app = app::App::new(worker, save_dir, backends, screenshot);
            app.set_rotation(rotation);
            app.set_crop(dev_crop);
            app.set_dev_read(dev_read);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
