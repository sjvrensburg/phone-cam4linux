use anyhow::{Context, Result};
use clap::Parser;
use phone_cam4linux::{loopback, sink::V4l2Sink, ConnectOptions, Facing};
use std::path::PathBuf;
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

    /// Requested capture resolution, e.g. 1280x720. Defaults to the phone's choice.
    #[arg(long)]
    resolution: Option<String>,

    /// Requested max frame rate.
    #[arg(long)]
    fps: Option<u32>,

    /// H.264 bitrate in megabits per second. Higher is crisper (better for reading
    /// text/documents) at the cost of bandwidth.
    #[arg(long, default_value_t = 30)]
    bitrate: u32,

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

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let video_nr = video_nr_from_path(&args.device)?;
    loopback::ensure_device(video_nr, "Android Cam")
        .context("could not prepare the v4l2loopback device")?;

    if args.test_pattern {
        return run_test_pattern(&args.device);
    }

    let resolution = args
        .resolution
        .as_deref()
        .map(parse_resolution)
        .transpose()?;

    let opts = ConnectOptions {
        serial: args.serial,
        facing: match args.facing {
            FacingArg::Front => Facing::Front,
            FacingArg::Back => Facing::Back,
        },
        resolution,
        max_fps: args.fps,
        bitrate_bps: Some(args.bitrate.saturating_mul(1_000_000)),
    };

    let mut session =
        phone_cam4linux::CameraSession::connect(opts).context("connecting to phone camera")?;
    log::info!("streaming to {}", args.device.display());
    session
        .run_to_v4l2(&args.device)
        .context("streaming camera to V4L2 device")?;
    Ok(())
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

fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once('x')
        .with_context(|| format!("--resolution {s:?} must look like WIDTHxHEIGHT"))?;
    Ok((w.parse()?, h.parse()?))
}

/// Drives the V4L2 sink with a synthetic, cycling color-bar frame -- exercises the
/// loopback/format-negotiation/write path independently of ADB or a real phone.
fn run_test_pattern(device: &std::path::Path) -> Result<()> {
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
    loop {
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
}
