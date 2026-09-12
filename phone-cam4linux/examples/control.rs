//! Drives the camera's zoom and torch live over scrcpy's control channel while
//! streaming to a V4L2 device:
//!
//! ```text
//! cargo run --release -p phone-cam4linux --example control -- /dev/video11
//! ```
//!
//! Streams for ~9 s: after 2 s it zooms in eight steps (x1.0625 each, about 1.6x),
//! at 5 s turns the torch on, at 7 s turns it off and zooms back out.

use phone_cam4linux::sink::{FrameSink, V4l2Sink};
use phone_cam4linux::{CameraSession, ConnectOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn main() -> phone_cam4linux::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let device = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/dev/video11".into());

    let opts = ConnectOptions {
        control: true,
        ..Default::default()
    };
    // The phone's default size may exceed the decoder (4000x3000 does for openh264).
    let device_handle = phone_cam4linux::adb::AdbDevice::autodetect()?;
    let resolution =
        phone_cam4linux::cameras::largest_usable_size(&device_handle, opts.facing, opts.decoder)?;
    let mut session = CameraSession::connect(ConnectOptions {
        resolution: Some(resolution),
        ..opts
    })?;
    let control = session.control().expect("control=true opens the channel");
    let (w, h) = (session.meta.width, session.meta.height);
    let mut sink = V4l2Sink::open(device.as_ref(), w, h)?;

    let stop = Arc::new(AtomicBool::new(false));
    let script = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || -> phone_cam4linux::Result<()> {
            std::thread::sleep(Duration::from_secs(2));
            for _ in 0..8 {
                control.zoom_in()?;
            }
            log::info!("zoomed in 8 steps");
            std::thread::sleep(Duration::from_secs(3));
            control.set_torch(true)?;
            log::info!("torch on");
            std::thread::sleep(Duration::from_secs(2));
            control.set_torch(false)?;
            for _ in 0..8 {
                control.zoom_out()?;
            }
            log::info!("torch off, zoomed back out");
            std::thread::sleep(Duration::from_secs(2));
            stop.store(true, Ordering::Relaxed);
            Ok(())
        })
    };

    let mut frames = 0u64;
    let mut tee = |frame: &phone_cam4linux::decode::YuvFrame| {
        frames += 1;
        sink.frame(frame)
    };
    session.run(&mut tee, &stop)?;
    script.join().expect("script thread")?;
    log::info!("{frames} frames");
    Ok(())
}
