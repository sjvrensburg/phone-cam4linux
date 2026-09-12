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

/// The largest frame the bundled openh264 build will decode, in 16x16 macroblocks.
///
/// openh264 hard-codes H.264 level 5.2 as its ceiling (`MaxFS` = 36864 MBs, i.e.
/// 3840x2160 = 32400 fits, 4000x3000 = 46875 doesn't) and rejects the SPS of anything
/// larger with `dsNoParamSets`. There is no runtime knob for this.
pub const MAX_MACROBLOCKS: u32 = 36864;

/// Whether a `width`x`height` stream is within [`MAX_MACROBLOCKS`].
pub fn fits_decoder(width: u32, height: u32) -> bool {
    let mbs = width.div_ceil(16) * height.div_ceil(16);
    mbs <= MAX_MACROBLOCKS
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

#[cfg(test)]
mod tests {
    use super::fits_decoder;

    #[test]
    fn decoder_ceiling() {
        assert!(fits_decoder(3840, 2160));
        assert!(fits_decoder(1920, 1080));
        assert!(!fits_decoder(4000, 3000));
        assert!(!fits_decoder(4608, 3456));
    }
}
