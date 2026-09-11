//! H.264 (Annex-B) decoding via `openh264`, statically compiled (`source` feature),
//! so no system FFmpeg/OpenH264 is required at build or run time.

use crate::error::{Error, Result};
use openh264::decoder::Decoder as H264Decoder;
use openh264::formats::YUVSource;

/// A decoded I420 (planar YUV 4:2:0) frame, tightly packed (row padding removed),
/// ready for pixel-format conversion.
pub struct YuvFrame {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

pub struct Decoder {
    inner: H264Decoder,
}

impl Decoder {
    pub fn new() -> Result<Self> {
        let inner = H264Decoder::new().map_err(|e| Error::Decode(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Feeds one Annex-B packet (as delivered by the scrcpy protocol layer -- config
    /// packets and slice packets alike) and returns a decoded frame if one was produced.
    /// Not every packet yields a frame (e.g. the initial SPS/PPS config packet doesn't).
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<YuvFrame>> {
        let yuv = self
            .inner
            .decode(packet)
            .map_err(|e| Error::Decode(e.to_string()))?;
        let Some(yuv) = yuv else {
            return Ok(None);
        };

        let (width, height) = yuv.dimensions();
        let (y_stride, uv_stride, _) = yuv.strides();
        let (uv_width, uv_height) = yuv.dimensions_uv();

        Ok(Some(YuvFrame {
            width,
            height,
            y: depad(yuv.y(), y_stride, width, height),
            u: depad(yuv.u(), uv_stride, uv_width, uv_height),
            v: depad(yuv.v(), uv_stride, uv_width, uv_height),
        }))
    }
}

/// Copies a possibly row-padded plane (stride > width) into a tightly packed buffer.
fn depad(plane: &[u8], stride: usize, width: usize, height: usize) -> Vec<u8> {
    if stride == width {
        return plane[..width * height].to_vec();
    }
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = row * stride;
        out.extend_from_slice(&plane[start..start + width]);
    }
    out
}
