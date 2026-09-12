//! Stream an Android phone's camera to a Linux V4L2 (`/dev/videoN`) device.
//!
//! The Android-side capture/encode is delegated to the real, upstream `scrcpy-server`
//! (embedded at build time, see `build.rs`) over ADB -- only the client-side protocol,
//! H.264 decode, and V4L2 sink are implemented here. See the crate's README for the
//! full architecture and scope.

pub mod adb;
pub mod cameras;
pub mod convert;
pub mod decode;
pub mod error;
pub mod loopback;
pub mod protocol;
pub mod session;
pub mod sink;

pub use cameras::{list_cameras, CameraInfo};
pub use error::{Error, Result};
pub use session::{CameraControl, CameraSession, ConnectOptions, Facing, ZOOM_STEP};
