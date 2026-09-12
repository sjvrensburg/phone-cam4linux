//! The camera worker thread: connects to the phone, decodes, and publishes the latest
//! frame for the UI, reconnecting with backoff when the stream drops (the same
//! policy as the `pc4l` CLI). Optionally tees every frame to a V4L2 device too.

use anyhow::{Context, Result};
use phone_cam4linux::cameras::largest_usable_size;
use phone_cam4linux::decode::YuvFrame;
use phone_cam4linux::sink::{FrameSink, V4l2Sink};
use phone_cam4linux::{adb::AdbDevice, CameraSession, ConnectOptions, Facing};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the worker is doing, for the status bar.
#[derive(Debug, Clone)]
pub enum Status {
    Connecting,
    Streaming {
        width: u32,
        height: u32,
    },
    /// The last attempt failed (or the stream ended); retrying after the backoff.
    Waiting {
        reason: String,
        retry_at: Instant,
    },
    Stopped,
}

#[derive(Debug, Clone, Copy)]
pub enum Resolution {
    PhoneDefault,
    Max,
    Fixed(u32, u32),
}

/// Everything the worker needs to (re)connect. Changing the facing takes effect on
/// the next connection, which [`Shared::restart`] forces.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub options: ConnectOptions,
    pub resolution: Resolution,
    /// Also write frames to this v4l2loopback device.
    pub tee_device: Option<PathBuf>,
}

pub struct Shared {
    stop: AtomicBool,
    restart: AtomicBool,
    frames: AtomicU64,
    latest: Mutex<Option<Arc<YuvFrame>>>,
    status: Mutex<Status>,
    config: Mutex<StreamConfig>,
}

impl Shared {
    /// The most recently decoded frame, if any.
    pub fn latest(&self) -> Option<Arc<YuvFrame>> {
        self.latest.lock().unwrap().clone()
    }

    pub fn status(&self) -> Status {
        self.status.lock().unwrap().clone()
    }

    /// Frames decoded since the worker started (all sessions).
    pub fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }

    pub fn facing(&self) -> Facing {
        self.config.lock().unwrap().options.facing
    }

    /// Switches camera: takes effect through a reconnect.
    pub fn set_facing(&self, facing: Facing) {
        let mut cfg = self.config.lock().unwrap();
        if cfg.options.facing != facing {
            cfg.options.facing = facing;
            drop(cfg);
            self.restart();
        }
    }

    /// Ends the current session (if any) and connects again with the current config.
    pub fn restart(&self) {
        self.restart.store(true, Ordering::Relaxed);
    }

    fn set_status(&self, status: Status) {
        *self.status.lock().unwrap() = status;
    }
}

pub struct Worker {
    pub shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Starts streaming in the background. `wake` is called after every frame and
    /// status change so the UI can repaint.
    pub fn start(config: StreamConfig, wake: impl Fn() + Send + 'static) -> Self {
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            restart: AtomicBool::new(false),
            frames: AtomicU64::new(0),
            latest: Mutex::new(None),
            status: Mutex::new(Status::Connecting),
            config: Mutex::new(config),
        });
        let thread_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("pc4l-stream".into())
            .spawn(move || run_loop(&thread_shared, &wake))
            .expect("spawning stream thread");
        Self {
            shared,
            handle: Some(handle),
        }
    }

    /// Asks the worker to stop and waits for it (which shuts the scrcpy server down
    /// and removes the adb forward).
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_loop(shared: &Arc<Shared>, wake: &dyn Fn()) {
    const MIN_BACKOFF: Duration = Duration::from_secs(1);
    const MAX_BACKOFF: Duration = Duration::from_secs(10);
    const RESTART_GRACE: Duration = Duration::from_millis(1500);

    let mut backoff = MIN_BACKOFF;
    let mut tee: Option<V4l2Sink> = None;
    while !shared.stop.load(Ordering::Relaxed) {
        shared.restart.store(false, Ordering::Relaxed);
        shared.set_status(Status::Connecting);
        wake();

        let config = shared.config.lock().unwrap().clone();
        let outcome = run_session(shared, &config, &mut tee, wake, &mut backoff);
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        if shared.restart.load(Ordering::Relaxed) {
            // The phone releases the camera a moment after the server goes; opening
            // the other camera straight away fails with "device is in the error state".
            shared.set_status(Status::Waiting {
                reason: "switching camera".to_string(),
                retry_at: Instant::now() + RESTART_GRACE,
            });
            wake();
            sleep_unless(RESTART_GRACE, || shared.stop.load(Ordering::Relaxed));
            continue;
        }
        let reason = match outcome {
            Ok(()) => "stream ended".to_string(),
            Err(e) => format!("{e:#}"),
        };
        log::warn!("{reason}; reconnecting in {backoff:?}");
        shared.set_status(Status::Waiting {
            reason,
            retry_at: Instant::now() + backoff,
        });
        wake();
        sleep_unless(backoff, || {
            shared.stop.load(Ordering::Relaxed) || shared.restart.load(Ordering::Relaxed)
        });
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
    shared.set_status(Status::Stopped);
    wake();
}

fn run_session(
    shared: &Arc<Shared>,
    config: &StreamConfig,
    tee: &mut Option<V4l2Sink>,
    wake: &dyn Fn(),
    backoff: &mut Duration,
) -> Result<()> {
    // The session's own stop flag: raised by the outer stop or by a restart request,
    // both of which end `run()` at the next packet.
    let session_stop = AtomicBool::new(false);
    let resolution = match config.resolution {
        Resolution::PhoneDefault => None,
        Resolution::Fixed(w, h) => Some((w, h)),
        Resolution::Max => {
            let device = select_device(&config.options)?;
            Some(
                largest_usable_size(&device, config.options.facing, config.options.decoder)
                    .context("resolving the maximum resolution")?,
            )
        }
    };
    let options = ConnectOptions {
        resolution,
        ..config.options.clone()
    };
    let mut session = CameraSession::connect_with_stop(options, &session_stop)
        .context("connecting to phone camera")?;
    let (w, h) = (session.meta.width, session.meta.height);
    log::info!("streaming {w}x{h}");

    if let Some(path) = &config.tee_device {
        if tee.as_ref().is_some_and(|s| s.size() != (w, h)) {
            *tee = None;
        }
        if tee.is_none() {
            *tee = Some(V4l2Sink::open(path, w, h).context("opening V4L2 device")?);
        }
    }

    shared.set_status(Status::Streaming {
        width: w,
        height: h,
    });
    let mut sink = |frame: &YuvFrame| -> phone_cam4linux::Result<()> {
        // A copy per frame (12 MB at 4K, well under a millisecond) keeps the decoder
        // free to overwrite its buffers while the UI reads this one.
        *shared.latest.lock().unwrap() = Some(Arc::new(frame.clone()));
        shared.frames.fetch_add(1, Ordering::Relaxed);
        if let Some(sink) = tee.as_mut() {
            sink.frame(frame)?;
        }
        if shared.stop.load(Ordering::Relaxed) || shared.restart.load(Ordering::Relaxed) {
            session_stop.store(true, Ordering::Relaxed);
        }
        wake();
        Ok(())
    };
    let result = session.run(&mut sink, &session_stop).context("streaming");
    // Only a session that delivered frames resets the backoff; one that fails right
    // after the handshake must keep backing off.
    if session.frames_decoded() > 0 {
        *backoff = Duration::from_secs(1);
    }
    result
}

fn select_device(options: &ConnectOptions) -> Result<AdbDevice> {
    Ok(match (&options.tcp_address, &options.serial) {
        (Some(addr), _) => AdbDevice::connect_tcp(addr).context("connecting over Wi-Fi")?,
        (None, Some(s)) => AdbDevice::with_serial(s),
        (None, None) => AdbDevice::autodetect()?,
    })
}

fn sleep_unless(total: Duration, done: impl Fn() -> bool) {
    let deadline = Instant::now() + total;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
}
