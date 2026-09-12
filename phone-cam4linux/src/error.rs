use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("`adb` executable not found on PATH; install android-tools/platform-tools")]
    AdbNotFound,

    #[error("adb command failed: {0}")]
    AdbCommand(String),

    #[error("no Android device found (is USB debugging enabled and authorized, or the phone reachable over Wi-Fi?)")]
    NoDevice,

    #[error("unexpected scrcpy protocol data: {0}")]
    Protocol(String),

    #[error("video stream stalled: no data from the phone for {0:?}")]
    StreamStalled(std::time::Duration),

    #[error("H.264 decode error: {0}")]
    Decode(String),

    #[error("V4L2 sink error: {0}")]
    Sink(String),

    #[error("failed to load v4l2loopback module: {0}")]
    Loopback(String),

    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
