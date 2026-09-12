//! H.264 (Annex-B) decoding.
//!
//! Two backends:
//!
//! - **openh264** (always built, statically compiled via the `source` feature, so no
//!   system FFmpeg/OpenH264 is needed). Hard-limited to H.264 level 5.2 frame sizes.
//! - **FFmpeg** (`ffmpeg` cargo feature; links the system libavcodec) using the native
//!   `h264` decoder, which has no practical size limit. Selected via
//!   [`Backend::Ffmpeg`]; the default is whichever is "best available".

use crate::error::{Error, Result};
use openh264::decoder::Decoder as H264Decoder;
use openh264::formats::YUVSource;

/// The largest frame the bundled openh264 build will decode, in 16x16 macroblocks.
///
/// openh264 hard-codes H.264 level 5.2 as its ceiling (`MaxFS` = 36864 MBs, i.e.
/// 3840x2160 = 32400 fits, 4000x3000 = 46875 doesn't) and rejects the SPS of anything
/// larger with `dsNoParamSets`. There is no runtime knob for this.
pub const MAX_MACROBLOCKS: u32 = 36864;

/// Whether `width`x`height` is within openh264's [`MAX_MACROBLOCKS`].
pub fn fits_openh264(width: u32, height: u32) -> bool {
    let mbs = width.div_ceil(16) * height.div_ceil(16);
    mbs <= MAX_MACROBLOCKS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Openh264,
    #[cfg(feature = "ffmpeg")]
    Ffmpeg,
}

impl Default for Backend {
    /// FFmpeg when compiled in (no size ceiling), otherwise openh264.
    fn default() -> Self {
        #[cfg(feature = "ffmpeg")]
        {
            Backend::Ffmpeg
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            Backend::Openh264
        }
    }
}

impl Backend {
    /// Whether this backend can decode a `width`x`height` stream: [`MAX_MACROBLOCKS`]
    /// for openh264, no practical limit for FFmpeg.
    pub fn fits(self, width: u32, height: u32) -> bool {
        match self {
            Backend::Openh264 => fits_openh264(width, height),
            #[cfg(feature = "ffmpeg")]
            Backend::Ffmpeg => true,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Openh264 => "openh264",
            #[cfg(feature = "ffmpeg")]
            Backend::Ffmpeg => "ffmpeg",
        }
    }
}

/// A decoded I420 (planar YUV 4:2:0) frame, tightly packed (row padding removed),
/// ready for pixel-format conversion.
pub struct YuvFrame {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

pub enum Decoder {
    Openh264(H264Decoder),
    #[cfg(feature = "ffmpeg")]
    Ffmpeg(ffmpeg_backend::FfmpegDecoder),
}

impl Decoder {
    pub fn new() -> Result<Self> {
        Self::with_backend(Backend::default())
    }

    pub fn with_backend(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Openh264 => {
                let inner = H264Decoder::new().map_err(|e| Error::Decode(e.to_string()))?;
                Ok(Decoder::Openh264(inner))
            }
            #[cfg(feature = "ffmpeg")]
            Backend::Ffmpeg => Ok(Decoder::Ffmpeg(ffmpeg_backend::FfmpegDecoder::new()?)),
        }
    }

    pub fn backend(&self) -> Backend {
        match self {
            Decoder::Openh264(_) => Backend::Openh264,
            #[cfg(feature = "ffmpeg")]
            Decoder::Ffmpeg(_) => Backend::Ffmpeg,
        }
    }

    /// Feeds one Annex-B packet (as delivered by the scrcpy protocol layer -- config
    /// packets and slice packets alike) and returns a decoded frame if one was produced.
    /// Not every packet yields a frame (e.g. the initial SPS/PPS config packet doesn't).
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<YuvFrame>> {
        match self {
            Decoder::Openh264(inner) => decode_openh264(inner, packet),
            #[cfg(feature = "ffmpeg")]
            Decoder::Ffmpeg(inner) => inner.decode(packet),
        }
    }
}

fn decode_openh264(inner: &mut H264Decoder, packet: &[u8]) -> Result<Option<YuvFrame>> {
    let yuv = inner
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

#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend {
    use super::{depad, YuvFrame};
    use crate::error::{Error, Result};
    use ffmpeg_next as ffmpeg;
    use ffmpeg::codec::{decoder, packet::Packet, Id};
    use ffmpeg::format::Pixel;
    use ffmpeg::frame::Video;

    pub struct FfmpegDecoder {
        decoder: decoder::Video,
        frame: Video,
        scratch: Video,
        /// SPS/PPS config packet held back until the next packet, see `decode`.
        pending_config: Option<Vec<u8>>,
    }

    impl FfmpegDecoder {
        pub fn new() -> Result<Self> {
            ffmpeg::init().map_err(|e| Error::Decode(format!("ffmpeg init: {e}")))?;
            let codec = decoder::find(Id::H264)
                .ok_or_else(|| Error::Decode("libavcodec has no h264 decoder".into()))?;
            let mut ctx = ffmpeg::codec::Context::new_with_codec(codec);
            // Annex-B input arrives as complete access units; low-delay output so we
            // don't buffer frames behind the decoder's reorder window.
            ctx.set_flags(ffmpeg::codec::Flags::LOW_DELAY);
            let decoder = ctx
                .decoder()
                .video()
                .map_err(|e| Error::Decode(format!("opening h264 decoder: {e}")))?;
            Ok(Self {
                decoder,
                frame: Video::empty(),
                scratch: Video::empty(),
                pending_config: None,
            })
        }

        pub fn decode(&mut self, packet: &[u8]) -> Result<Option<YuvFrame>> {
            // libavcodec reports "Invalid data" for a packet holding only SPS/PPS
            // (it still records them, but that's a noisy error to surface). Like
            // scrcpy's own client, prepend such a config packet to the next packet.
            if is_config_only(packet) {
                self.pending_config = Some(packet.to_vec());
                return Ok(None);
            }
            let pkt = match self.pending_config.take() {
                Some(mut cfg) => {
                    cfg.extend_from_slice(packet);
                    Packet::copy(&cfg)
                }
                None => Packet::copy(packet),
            };
            self.decoder
                .send_packet(&pkt)
                .map_err(|e| Error::Decode(e.to_string()))?;

            // Drain everything available, keeping only the newest frame: a live
            // camera wants latency, not completeness. Note avcodec_receive_frame
            // unrefs its destination *before* reporting EAGAIN, so receive into a
            // scratch frame and swap the good ones in.
            let mut got = false;
            while self.decoder.receive_frame(&mut self.scratch).is_ok() {
                std::mem::swap(&mut self.frame, &mut self.scratch);
                got = true;
            }
            if !got {
                return Ok(None);
            }

            match self.frame.format() {
                Pixel::YUV420P | Pixel::YUVJ420P => {}
                other => {
                    return Err(Error::Decode(format!(
                        "unexpected decoded pixel format {other:?} (want yuv420p)"
                    )))
                }
            }
            let width = self.frame.width() as usize;
            let height = self.frame.height() as usize;
            let (uv_w, uv_h) = (width.div_ceil(2), height.div_ceil(2));
            Ok(Some(YuvFrame {
                width,
                height,
                y: depad(self.frame.data(0), self.frame.stride(0), width, height),
                u: depad(self.frame.data(1), self.frame.stride(1), uv_w, uv_h),
                v: depad(self.frame.data(2), self.frame.stride(2), uv_w, uv_h),
            }))
        }
    }

    /// True if every NAL unit in the Annex-B `packet` is a parameter set (SPS=7,
    /// PPS=8) -- i.e. it's scrcpy's codec-config packet, not a picture.
    pub(super) fn is_config_only(packet: &[u8]) -> bool {
        let mut saw_nal = false;
        let mut i = 0;
        while i + 3 < packet.len() {
            if packet[i] == 0 && packet[i + 1] == 0 && packet[i + 2] == 1 {
                let nal_type = packet[i + 3] & 0x1f;
                if !matches!(nal_type, 7 | 8) {
                    return false;
                }
                saw_nal = true;
                i += 4;
            } else {
                i += 1;
            }
        }
        saw_nal
    }
}

#[cfg(test)]
mod tests {
    use super::fits_openh264;

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn config_packet_detection() {
        use super::ffmpeg_backend::is_config_only;
        let sps_pps = [0, 0, 0, 1, 0x67, 0xaa, 0, 0, 0, 1, 0x68, 0xbb];
        let idr = [0, 0, 0, 1, 0x65, 0xcc, 0xdd];
        let mut both = sps_pps.to_vec();
        both.extend_from_slice(&idr);
        assert!(is_config_only(&sps_pps));
        assert!(!is_config_only(&idr));
        assert!(!is_config_only(&both));
        assert!(!is_config_only(&[]));
    }

    #[test]
    fn openh264_ceiling() {
        assert!(fits_openh264(3840, 2160));
        assert!(fits_openh264(1920, 1080));
        assert!(!fits_openh264(4000, 3000));
        assert!(!fits_openh264(4608, 3456));
    }
}
