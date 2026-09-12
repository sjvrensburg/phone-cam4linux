//! V4L2 output sink, writing YUYV422 frames to a `v4l2loopback` device via the `v4l`
//! crate's safe mmap streaming API (rather than hand-rolled ioctls).

use crate::error::{Error, Result};
use std::path::Path;
use v4l::buffer::Type;
use v4l::device::Device;
use v4l::format::{Format, FourCC};
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::OutputStream;
use v4l::video::Output;

pub struct V4l2Sink {
    device: Device,
    stream: Option<MmapStream<'static>>,
    width: u32,
    height: u32,
}

impl V4l2Sink {
    /// Opens `path` and negotiates YUYV422 at `width`x`height`. The stream is created
    /// lazily on the first `write_frame` call (buffer allocation needs `&self`, and we
    /// want a clean error if the negotiated format doesn't match what was requested).
    pub fn open(path: &Path, width: u32, height: u32) -> Result<Self> {
        let device = Device::with_path(path)
            .map_err(|e| Error::Sink(format!("opening {}: {e}", path.display())))?;

        let fmt = Format::new(width, height, FourCC::new(b"YUYV"));
        let actual = Output::set_format(&device, &fmt)
            .map_err(|e| Error::Sink(format!("setting format on {}: {e}", path.display())))?;

        if actual.width != width || actual.height != height || actual.fourcc != fmt.fourcc {
            return Err(Error::Sink(format!(
                "{} negotiated {}x{} {} instead of requested {}x{} YUYV \
                 (is another process already using this /dev/videoN?)",
                path.display(),
                actual.width,
                actual.height,
                actual.fourcc,
                width,
                height
            )));
        }

        Ok(Self {
            device,
            stream: None,
            width,
            height,
        })
    }

    /// The `(width, height)` this sink was negotiated at.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn stream(&mut self) -> Result<&mut MmapStream<'static>> {
        if self.stream.is_none() {
            let stream = MmapStream::with_buffers(&self.device, Type::VideoOutput, 4)
                .map_err(|e| Error::Sink(format!("allocating output buffers: {e}")))?;
            self.stream = Some(stream);
        }
        Ok(self.stream.as_mut().unwrap())
    }

    /// Writes one packed YUYV422 frame (`width * height * 2` bytes) to the device.
    pub fn write_frame(&mut self, yuyv: &[u8]) -> Result<()> {
        let expected = (self.width * self.height * 2) as usize;
        if yuyv.len() != expected {
            return Err(Error::Sink(format!(
                "frame is {} bytes, expected {expected} for {}x{} YUYV",
                yuyv.len(),
                self.width,
                self.height
            )));
        }
        let stream = self.stream()?;
        let (buf, meta) = OutputStream::next(stream)
            .map_err(|e| Error::Sink(format!("dequeuing output buffer: {e}")))?;
        buf[..yuyv.len()].copy_from_slice(yuyv);
        meta.bytesused = yuyv.len() as u32;
        Ok(())
    }
}
