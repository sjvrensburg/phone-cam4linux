//! Creating the `v4l2loopback` device node when the target `/dev/videoN` is missing.
//!
//! Two situations need distinguishing:
//!
//! - the module isn't loaded at all: `modprobe v4l2loopback video_nr=N ...` creates it;
//! - the module is already loaded (OBS, a modprobe.d snippet, an earlier run with a
//!   different `video_nr`): `modprobe` is then a silent no-op, so the device has to be
//!   added at runtime via the `/dev/v4l2loopback` control node (`v4l2loopback-ctl add`,
//!   v0.12+).
//!
//! Both need root, obtained via `pkexec`, which prompts through the desktop's polkit
//! agent every time. To avoid the prompt entirely, create the device at boot instead
//! -- see `contrib/modprobe.d/` in the repository.

use crate::error::{Error, Result};
use std::path::Path;
use std::process::Command;

const CONTROL_NODE: &str = "/dev/v4l2loopback";

/// Ensures `/dev/video<video_nr>` exists, creating it via `pkexec` if it doesn't.
pub fn ensure_device(video_nr: u32, card_label: &str) -> Result<()> {
    let path = format!("/dev/video{video_nr}");
    if Path::new(&path).exists() {
        return Ok(());
    }

    let module_loaded = Path::new(CONTROL_NODE).exists();
    let argv: Vec<String> = if module_loaded {
        log::info!("{path} does not exist; adding it via pkexec v4l2loopback-ctl");
        vec![
            "v4l2loopback-ctl".into(),
            "add".into(),
            "-n".into(),
            card_label.into(),
            "-x".into(),
            "1".into(),
            path.clone(),
        ]
    } else {
        log::info!("{path} does not exist; loading v4l2loopback via pkexec modprobe");
        vec![
            "modprobe".into(),
            "v4l2loopback".into(),
            format!("video_nr={video_nr}"),
            format!("card_label={card_label}"),
            "exclusive_caps=1".into(),
        ]
    };
    let manual = manual_hint(video_nr, card_label, module_loaded);

    let status = Command::new("pkexec").args(&argv).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            return Err(Error::Loopback(format!(
                "`pkexec {}` exited with {s}; you can create the device manually with:\n{manual}",
                argv.join(" ")
            )));
        }
        Err(e) => {
            return Err(Error::Loopback(format!(
                "failed to run pkexec ({e}); is polkit installed? \
                 you can create the device manually with:\n{manual}"
            )));
        }
    }

    // udev creates the node asynchronously after the driver registers it, and applies
    // the `video` group permissions a moment after that -- so wait until we can
    // actually open it, not merely until it exists.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        {
            Ok(_) => return Ok(()),
            Err(e) if std::time::Instant::now() > deadline => {
                return Err(Error::Loopback(format!(
                    "`{}` succeeded but {path} is not usable: {e}",
                    argv.join(" ")
                )));
            }
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

fn manual_hint(video_nr: u32, card_label: &str, module_loaded: bool) -> String {
    if module_loaded {
        format!("sudo v4l2loopback-ctl add -n \"{card_label}\" -x 1 /dev/video{video_nr}")
    } else {
        format!(
            "sudo modprobe v4l2loopback video_nr={video_nr} card_label=\"{card_label}\" exclusive_caps=1"
        )
    }
}

/// Finds the lowest `/dev/videoN` (starting at `start`) that doesn't exist yet, so a
/// fresh v4l2loopback device can be created there without colliding with a real webcam.
pub fn next_free_video_nr(start: u32) -> u32 {
    let mut n = start;
    while Path::new(&format!("/dev/video{n}")).exists() {
        n += 1;
    }
    n
}
