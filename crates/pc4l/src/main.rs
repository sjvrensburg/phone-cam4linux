use anyhow::{Context, Result};
use clap::Parser;
use phone_cam4linux::adb::AdbDevice;
use phone_cam4linux::cameras::is_usable_size;
use phone_cam4linux::decode::Backend;
use phone_cam4linux::{loopback, sink::V4l2Sink, CameraInfo, ConnectOptions, Facing};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Stream an Android phone's camera to a Linux V4L2 device.
#[derive(Parser, Debug)]
#[command(name = "pc4l", version)]
struct Args {
    /// ADB serial of the device to use (autodetected if omitted and only one is attached).
    #[arg(long)]
    serial: Option<String>,

    /// Which camera to use.
    #[arg(long, value_enum, default_value = "back")]
    facing: FacingArg,

    /// Target V4L2 device, e.g. /dev/video10. Created via v4l2loopback if it doesn't exist.
    #[arg(long, default_value = "/dev/video10")]
    device: PathBuf,

    /// Requested capture resolution, e.g. 1280x720, or `max` for the largest size the
    /// camera offers that the decoder can handle. Defaults to the phone's choice.
    /// See --list-sizes.
    #[arg(long)]
    resolution: Option<String>,

    /// List the phone's cameras and their supported capture sizes, then exit.
    #[arg(long)]
    list_sizes: bool,

    /// H.264 decoder. `openh264` is always available but limited to level-5.2 frame
    /// sizes (~3840x2160); `ffmpeg` (if compiled in with the `ffmpeg` feature) has no
    /// such limit. Defaults to ffmpeg when available.
    #[arg(long, value_enum, default_value_t = DecoderArg::default())]
    decoder: DecoderArg,

    /// Requested max frame rate.
    #[arg(long)]
    fps: Option<u32>,

    /// H.264 bitrate in megabits per second. Higher is crisper (better for reading
    /// text/documents) at the cost of bandwidth.
    #[arg(long, default_value_t = 30)]
    bitrate: u32,

    /// Exit when the phone disconnects or the stream fails, instead of waiting for it
    /// to come back and reconnecting (the default, so the virtual camera survives a
    /// cable wiggle or a phone reboot).
    #[arg(long)]
    no_reconnect: bool,

    /// Skip ADB/phone entirely and feed the V4L2 sink a synthetic color-bar pattern.
    /// Useful for testing the loopback/format/sink path without hardware attached.
    #[arg(long)]
    test_pattern: bool,
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
        Self::from(Backend::default())
    }
}

impl From<Backend> for DecoderArg {
    fn from(b: Backend) -> Self {
        match b {
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
    if args.list_sizes {
        return list_sizes(args.serial.as_deref(), decoder);
    }

    let video_nr = video_nr_from_path(&args.device)?;
    loopback::ensure_device(video_nr, "Android Cam")
        .context("could not prepare the v4l2loopback device")?;

    let stop = install_ctrlc_handler()?;

    if args.test_pattern {
        return run_test_pattern(&args.device, &stop);
    }

    let facing = match args.facing {
        FacingArg::Front => Facing::Front,
        FacingArg::Back => Facing::Back,
    };
    let resolution = match args.resolution.as_deref() {
        None => None,
        Some("max") => Some(largest_decodable_size(args.serial.as_deref(), facing, decoder)?),
        Some(s) => {
            let (w, h) = parse_resolution(s)?;
            anyhow::ensure!(
                decoder.fits(w, h),
                "{w}x{h} exceeds what the {} decoder can handle (e.g. 3840x2160); \
                 pick a smaller size from --list-sizes{}",
                decoder.name(),
                if cfg!(feature = "ffmpeg") {
                    " or use --decoder ffmpeg"
                } else {
                    ", or rebuild with `--features ffmpeg`"
                }
            );
            anyhow::ensure!(
                is_usable_size(w, h, decoder),
                "{w}x{h} is not a multiple of 8 in both dimensions; scrcpy would round it \
                 and the camera then rejects the size. Pick another from --list-sizes"
            );
            Some((w, h))
        }
    };

    let opts = ConnectOptions {
        serial: args.serial,
        facing,
        resolution,
        max_fps: args.fps,
        bitrate_bps: Some(args.bitrate.saturating_mul(1_000_000)),
        decoder,
    };

    stream_loop(&args.device, opts, !args.no_reconnect, &stop)
}

/// First Ctrl-C asks the pipeline to wind down cleanly (server stopped, adb forward
/// removed); a second one exits immediately in case the first is stuck.
fn install_ctrlc_handler() -> Result<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::Relaxed) {
            eprintln!("second interrupt, exiting immediately");
            std::process::exit(130);
        }
        log::info!("interrupted, shutting down");
    })
    .context("installing Ctrl-C handler")?;
    Ok(stop)
}

/// Connects and streams, reconnecting after failures (with backoff) when `reconnect`
/// is set, until `stop` is raised. The V4L2 sink is kept open across reconnects so
/// consumers (a browser, OBS) don't see the device disappear.
fn stream_loop(
    device: &std::path::Path,
    opts: ConnectOptions,
    reconnect: bool,
    stop: &AtomicBool,
) -> Result<()> {
    const MIN_BACKOFF: Duration = Duration::from_secs(1);
    const MAX_BACKOFF: Duration = Duration::from_secs(10);

    let mut sink: Option<V4l2Sink> = None;
    let mut backoff = MIN_BACKOFF;
    while !stop.load(Ordering::Relaxed) {
        let outcome = phone_cam4linux::CameraSession::connect(opts.clone())
            .context("connecting to phone camera")
            .and_then(|mut session| {
                let (w, h) = (session.meta.width, session.meta.height);
                if sink.as_ref().is_some_and(|s| s.size() != (w, h)) {
                    log::info!("stream size changed, reopening {}", device.display());
                    sink = None;
                }
                let sink = match &mut sink {
                    Some(s) => s,
                    None => {
                        log::info!("streaming {w}x{h} to {}", device.display());
                        sink.insert(V4l2Sink::open(device, w, h).context("opening V4L2 sink")?)
                    }
                };
                // A healthy connection resets the backoff for the *next* failure.
                backoff = MIN_BACKOFF;
                session
                    .run(sink, stop)
                    .context("streaming camera to V4L2 device")
            });

        match outcome {
            Ok(()) if stop.load(Ordering::Relaxed) => return Ok(()),
            Ok(()) => log::warn!("stream ended"),
            Err(e) if !reconnect => return Err(e),
            Err(e) => log::warn!("{e:#}"),
        }
        if !reconnect {
            return Ok(());
        }

        log::info!("reconnecting in {backoff:?} (Ctrl-C to quit)");
        sleep_unless_stopped(backoff, stop);
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
    Ok(())
}

fn sleep_unless_stopped(total: Duration, stop: &AtomicBool) {
    let deadline = std::time::Instant::now() + total;
    while !stop.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn video_nr_from_path(path: &std::path::Path) -> Result<u32> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("--device must look like /dev/videoN")?;
    name.strip_prefix("video")
        .and_then(|n| n.parse::<u32>().ok())
        .with_context(|| format!("--device {:?} must look like /dev/videoN", path))
}

fn select_device(serial: Option<&str>) -> Result<AdbDevice> {
    Ok(match serial {
        Some(s) => AdbDevice::with_serial(s),
        None => AdbDevice::autodetect()?,
    })
}

fn list_sizes(serial: Option<&str>, decoder: Backend) -> Result<()> {
    let device = select_device(serial)?;
    let cameras = phone_cam4linux::list_cameras(&device).context("listing cameras")?;
    for cam in &cameras {
        let facing = match cam.facing {
            Some(Facing::Back) => "back",
            Some(Facing::Front) => "front",
            None => "unknown facing",
        };
        let fps: Vec<String> = cam.fps.iter().map(u32::to_string).collect();
        println!("camera {} ({facing}, fps: {})", cam.id, fps.join("/"));
        for &(w, h) in &cam.sizes {
            let note = if !decoder.fits(w, h) {
                "   (exceeds decoder limit)"
            } else if !is_usable_size(w, h, decoder) {
                "   (not 8-aligned; rejected by the camera after encoder rounding)"
            } else {
                ""
            };
            println!("  {w}x{h}{note}");
        }
    }
    println!(
        "\nMarked sizes cannot be used with the {} decoder; \
         `--resolution max` picks the largest usable one.",
        decoder.name()
    );
    Ok(())
}

fn largest_decodable_size(
    serial: Option<&str>,
    facing: Facing,
    decoder: Backend,
) -> Result<(u32, u32)> {
    let device = select_device(serial)?;
    let cameras = phone_cam4linux::list_cameras(&device).context("listing cameras")?;
    let cam: &CameraInfo = cameras
        .iter()
        .find(|c| c.facing == Some(facing))
        .with_context(|| format!("phone reports no {facing:?}-facing camera"))?;
    let size = cam
        .largest_size(|w, h| is_usable_size(w, h, decoder))
        .context("camera offers no size the decoder can handle")?;
    log::info!("--resolution max resolved to {}x{}", size.0, size.1);
    Ok(size)
}

fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once('x')
        .with_context(|| format!("--resolution {s:?} must look like WIDTHxHEIGHT"))?;
    Ok((w.parse()?, h.parse()?))
}

/// Drives the V4L2 sink with a synthetic, cycling color-bar frame -- exercises the
/// loopback/format-negotiation/write path independently of ADB or a real phone.
fn run_test_pattern(device: &std::path::Path, stop: &AtomicBool) -> Result<()> {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    let mut sink = V4l2Sink::open(device, WIDTH, HEIGHT).context("opening V4L2 sink")?;
    let mut yuyv = vec![0u8; (WIDTH * HEIGHT * 2) as usize];

    log::info!(
        "writing {WIDTH}x{HEIGHT} test pattern to {} (Ctrl-C to stop)",
        device.display()
    );

    let bars: [(u8, u8, u8); 8] = [
        (235, 128, 128), // white
        (210, 16, 146),  // yellow
        (170, 166, 16),  // cyan
        (145, 54, 34),   // green
        (106, 202, 222), // magenta
        (81, 90, 240),   // red
        (41, 240, 110),  // blue
        (16, 128, 128),  // black
    ];

    let mut frame_idx: usize = 0;
    while !stop.load(Ordering::Relaxed) {
        let shift = frame_idx / 8;
        for row in 0..HEIGHT as usize {
            for pair in 0..(WIDTH as usize / 2) {
                let bar = ((pair * 2 * 8) / WIDTH as usize + shift) % bars.len();
                let (y, u, v) = bars[bar];
                let o = &mut yuyv[row * WIDTH as usize * 2 + pair * 4..][..4];
                o[0] = y;
                o[1] = u;
                o[2] = y;
                o[3] = v;
            }
        }
        sink.write_frame(&yuyv).context("writing test frame")?;
        frame_idx = frame_idx.wrapping_add(1);
        std::thread::sleep(Duration::from_millis(1000 / 30));
    }
    Ok(())
}
