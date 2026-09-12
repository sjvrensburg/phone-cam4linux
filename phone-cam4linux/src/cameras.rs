//! Enumerating the phone's cameras and their supported capture sizes.
//!
//! scrcpy-server has no machine-readable listing; `list_camera_sizes=true` makes it
//! print a human-oriented report and exit. The parser here targets the format of the
//! pinned server version (see `SCRCPY_SERVER_VERSION` in `build.rs`):
//!
//! ```text
//! [server] INFO: List of cameras:
//!     --camera-id=0    (back, 5760x4312, fps=[15, 20, 24, 30])
//!         - 4000x3000
//!         - 3840x2160
//!       High speed capture (--camera-high-speed):
//!         - 1280x720 (fps=[120])
//!     --camera-id=1    (front, 4608x3456, fps=[15, 20, 24, 30])
//!         ...
//! ```
//!
//! High-speed sizes are ignored: this crate never enables `camera_high_speed`.

use crate::adb::AdbDevice;
use crate::decode::Backend;
use crate::error::{Error, Result};
use crate::session::Facing;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraInfo {
    /// Camera2 id, as accepted by scrcpy's `camera_id=`.
    pub id: String,
    pub facing: Option<Facing>,
    /// The sensor's full resolution as reported by scrcpy.
    pub sensor_size: Option<(u32, u32)>,
    pub fps: Vec<u32>,
    /// Supported capture sizes for normal (non-high-speed) capture, in the order the
    /// device reports them (largest first on every device seen so far).
    pub sizes: Vec<(u32, u32)>,
}

/// Whether this crate can stream a camera at `width`x`height` with `decoder`: the size
/// must survive scrcpy's encoder alignment (it rounds both dimensions down to a
/// multiple of 8 and then asks the camera for *that* size, which fails if it isn't a
/// supported mode -- e.g. 4000x2250 becomes 4000x2248 and the capture session never
/// configures) and the decoder must accept it.
pub fn is_usable_size(width: u32, height: u32, decoder: Backend) -> bool {
    width.is_multiple_of(8) && height.is_multiple_of(8) && decoder.fits(width, height)
}

impl CameraInfo {
    /// The largest supported size (by pixel count) that `accept` allows, e.g.
    /// [`is_usable_size`].
    pub fn largest_size(&self, accept: impl Fn(u32, u32) -> bool) -> Option<(u32, u32)> {
        self.sizes
            .iter()
            .copied()
            .filter(|&(w, h)| accept(w, h))
            .max_by_key(|&(w, h)| u64::from(w) * u64::from(h))
    }
}

/// The largest size the `facing` camera offers that `decoder` can stream
/// ([`is_usable_size`]): what a "maximum resolution" setting resolves to. Talks to
/// the phone, so call it per connection -- the phone may not be there yet, or a
/// different one may show up after a reconnect.
pub fn largest_usable_size(
    device: &AdbDevice,
    facing: Facing,
    decoder: Backend,
) -> Result<(u32, u32)> {
    let cameras = list_cameras(device)?;
    let cam = cameras
        .iter()
        .find(|c| c.facing == Some(facing))
        .ok_or_else(|| Error::Protocol(format!("phone reports no {facing:?}-facing camera")))?;
    let size = cam
        .largest_size(|w, h| is_usable_size(w, h, decoder))
        .ok_or_else(|| {
            Error::Protocol(format!(
                "camera {} offers no size the {} decoder can handle",
                cam.id,
                decoder.name()
            ))
        })?;
    log::info!("maximum resolution resolved to {}x{}", size.0, size.1);
    Ok(size)
}

/// Runs the embedded server in listing mode on `device` and parses its report.
pub fn list_cameras(device: &AdbDevice) -> Result<Vec<CameraInfo>> {
    device.push_server_jar(crate::session::SERVER_JAR)?;
    let output = device.run_server_once(&["list_camera_sizes=true".to_string()])?;
    let cameras = parse_camera_listing(&output);
    if cameras.is_empty() {
        return Err(Error::Protocol(format!(
            "no cameras found in scrcpy server output:\n{}",
            output.trim()
        )));
    }
    Ok(cameras)
}

pub fn parse_camera_listing(output: &str) -> Vec<CameraInfo> {
    let mut cameras: Vec<CameraInfo> = Vec::new();
    let mut in_high_speed = false;

    for raw in output.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("--camera-id=") {
            in_high_speed = false;
            cameras.push(parse_camera_header(rest));
        } else if line.starts_with("High speed capture") {
            in_high_speed = true;
        } else if let Some(rest) = line.strip_prefix("- ") {
            if in_high_speed {
                continue;
            }
            if let (Some(cam), Some(size)) = (cameras.last_mut(), parse_size(rest)) {
                cam.sizes.push(size);
            }
        }
    }
    cameras
}

/// Parses `0    (back, 5760x4312, fps=[15, 20, 24, 30])` (the part after `--camera-id=`).
fn parse_camera_header(rest: &str) -> CameraInfo {
    let (id, detail) = match rest.split_once('(') {
        Some((id, detail)) => (id.trim(), detail.trim_end_matches(')')),
        None => (rest.trim(), ""),
    };
    let mut info = CameraInfo {
        id: id.to_string(),
        facing: None,
        sensor_size: None,
        fps: Vec::new(),
        sizes: Vec::new(),
    };
    // The fps list contains commas itself, so split it off first.
    let (fields, fps) = match detail.split_once("fps=[") {
        Some((fields, fps)) => (fields, fps.trim_end_matches(']')),
        None => (detail, ""),
    };
    for field in fields.split(',').map(str::trim) {
        match field {
            "back" => info.facing = Some(Facing::Back),
            "front" => info.facing = Some(Facing::Front),
            _ => {
                if let Some(size) = parse_size(field) {
                    info.sensor_size = Some(size);
                }
            }
        }
    }
    info.fps = fps
        .split(',')
        .filter_map(|f| f.trim().parse().ok())
        .collect();
    info
}

/// Parses a leading `WxH`, ignoring anything after it.
fn parse_size(s: &str) -> Option<(u32, u32)> {
    let token = s.split_whitespace().next()?;
    let (w, h) = token.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
[server] INFO: Device: [samsung] samsung SM-A307FN (Android 13)
[server] INFO: List of cameras:
    --camera-id=0    (back, 5760x4312, fps=[15, 20, 24, 30])
        - 4000x3000
        - 3840x2160
        - 1920x1080
      High speed capture (--camera-high-speed):
        - 1280x720 (fps=[120])
    --camera-id=1    (front, 4608x3456, fps=[15, 20, 24, 30])
        - 4608x3456
        - 1280x720
";

    #[test]
    fn parses_server_listing() {
        let cams = parse_camera_listing(FIXTURE);
        assert_eq!(cams.len(), 2);

        let back = &cams[0];
        assert_eq!(back.id, "0");
        assert_eq!(back.facing, Some(Facing::Back));
        assert_eq!(back.sensor_size, Some((5760, 4312)));
        assert_eq!(back.fps, vec![15, 20, 24, 30]);
        // The high-speed 1280x720 must not leak into the normal list.
        assert_eq!(back.sizes, vec![(4000, 3000), (3840, 2160), (1920, 1080)]);

        let front = &cams[1];
        assert_eq!(front.id, "1");
        assert_eq!(front.facing, Some(Facing::Front));
        assert_eq!(front.sizes, vec![(4608, 3456), (1280, 720)]);
    }

    #[test]
    fn largest_size_respects_filter() {
        let cams = parse_camera_listing(FIXTURE);
        let back = &cams[0];
        assert_eq!(back.largest_size(|_, _| true), Some((4000, 3000)));
        assert_eq!(
            back.largest_size(|w, h| w * h <= 3840 * 2160),
            Some((3840, 2160))
        );
        assert_eq!(back.largest_size(|_, _| false), None);
    }

    #[test]
    fn usable_sizes() {
        let b = Backend::Openh264;
        assert!(is_usable_size(3840, 2160, b));
        assert!(is_usable_size(2992, 2992, b));
        assert!(!is_usable_size(4000, 2250, b), "not 8-aligned");
        assert!(!is_usable_size(4000, 3000, b), "too many macroblocks");
        #[cfg(feature = "ffmpeg")]
        assert!(is_usable_size(4000, 3000, Backend::Ffmpeg));
    }

    #[test]
    fn tolerates_garbage() {
        assert!(parse_camera_listing("").is_empty());
        assert!(parse_camera_listing("[server] ERROR: boom\n").is_empty());
    }
}
