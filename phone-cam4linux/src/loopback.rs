//! Auto-loading the `v4l2loopback` kernel module when the target device is missing.

use crate::error::{Error, Result};
use std::path::Path;
use std::process::Command;

/// Ensures `/dev/video<video_nr>` exists, loading `v4l2loopback` via `pkexec modprobe`
/// if it doesn't. If the module is already loaded but doesn't own `video_nr`, this
/// will fail (the caller should pick a free `video_nr`, e.g. via `next_free_video_nr`).
pub fn ensure_device(video_nr: u32, card_label: &str) -> Result<()> {
    let path = format!("/dev/video{video_nr}");
    if Path::new(&path).exists() {
        return Ok(());
    }

    log::info!("{path} does not exist; loading v4l2loopback via pkexec modprobe");
    let status = Command::new("pkexec")
        .args([
            "modprobe",
            "v4l2loopback",
            &format!("video_nr={video_nr}"),
            &format!("card_label={card_label}"),
            "exclusive_caps=1",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            return Err(Error::Loopback(format!(
                "`pkexec modprobe v4l2loopback` exited with {s}; \
                 you can load it manually with:\n\
                 sudo modprobe v4l2loopback video_nr={video_nr} card_label=\"{card_label}\" exclusive_caps=1"
            )));
        }
        Err(e) => {
            return Err(Error::Loopback(format!(
                "failed to run pkexec ({e}); is polkit installed? \
                 you can load the module manually with:\n\
                 sudo modprobe v4l2loopback video_nr={video_nr} card_label=\"{card_label}\" exclusive_caps=1"
            )));
        }
    }

    if !Path::new(&path).exists() {
        return Err(Error::Loopback(format!(
            "modprobe succeeded but {path} still doesn't exist"
        )));
    }
    Ok(())
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
