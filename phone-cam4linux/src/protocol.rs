//! scrcpy's client-side video socket protocol.
//!
//! This is undocumented wire format, reverse-engineered from the scrcpy source
//! (`server/src/main/java/com/genymobile/scrcpy/video/VideoStreamer.java` and
//! `app/src/decoder.c`/`app/src/demuxer.c` in the pinned scrcpy release, see
//! `SCRCPY_SERVER_VERSION` in `build.rs`). It is pinned to that server version;
//! bumping the server version requires re-checking this module against the new
//! source before trusting it against a real device.
//!
//! With `send_device_meta=false send_dummy_byte=false send_codec_meta=true
//! send_frame_meta=true` (the options this crate always passes -- see
//! `session.rs`), the video socket carries, in order:
//!
//! 1. Codec metadata (12 bytes, sent once): `codec_id: u32BE`, `width: u32BE`,
//!    `height: u32BE`. `codec_id` for H.264 is the ASCII bytes `"h264"` read as a
//!    big-endian u32 (`0x68323634`).
//! 2. A stream of frame packets, each: a 12-byte header (`pts_and_flags: u64BE`,
//!    `packet_size: u32BE`) followed by `packet_size` bytes of Annex-B H.264 data.
//!    The top two bits of `pts_and_flags` are the config-packet and key-frame
//!    flags; the remaining 62 bits are the PTS in microseconds (meaningless for
//!    config packets).

use crate::error::{Error, Result};
use std::io::Read;

const CODEC_ID_H264: u32 = 0x6832_3634; // b"h264" as big-endian u32

const FLAG_CONFIG: u64 = 1 << 63;
const FLAG_KEY_FRAME: u64 = 1 << 62;
const PTS_MASK: u64 = !(FLAG_CONFIG | FLAG_KEY_FRAME);

#[derive(Debug, Clone, Copy)]
pub struct CodecMeta {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct FramePacket {
    pub is_config: bool,
    pub is_key_frame: bool,
    pub pts_us: u64,
    pub data: Vec<u8>,
}

/// Reads and validates the one-time codec metadata header at the start of the
/// video socket stream.
pub fn read_codec_meta<R: Read>(r: &mut R) -> Result<CodecMeta> {
    let mut buf = [0u8; 12];
    r.read_exact(&mut buf)
        .map_err(|e| Error::Protocol(format!("reading codec metadata: {e}")))?;
    let codec_id = u32::from_be_bytes(buf[0..4].try_into().unwrap());
    if codec_id != CODEC_ID_H264 {
        return Err(Error::Protocol(format!(
            "unsupported codec id 0x{codec_id:08x} (expected h264); \
             video_codec=h264 must be forced in server args"
        )));
    }
    let width = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    let height = u32::from_be_bytes(buf[8..12].try_into().unwrap());
    Ok(CodecMeta { width, height })
}

/// Reads one frame packet (header + payload). Returns `Ok(None)` on clean EOF
/// (socket closed between packets, i.e. the phone was disconnected / app killed).
pub fn read_frame_packet<R: Read>(r: &mut R) -> Result<Option<FramePacket>> {
    let mut header = [0u8; 12];
    if !read_exact_or_eof(r, &mut header)? {
        return Ok(None);
    }
    let pts_and_flags = u64::from_be_bytes(header[0..8].try_into().unwrap());
    let size = u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize;

    let mut data = vec![0u8; size];
    r.read_exact(&mut data)
        .map_err(|e| Error::Protocol(format!("reading frame payload ({size} bytes): {e}")))?;

    Ok(Some(FramePacket {
        is_config: pts_and_flags & FLAG_CONFIG != 0,
        is_key_frame: pts_and_flags & FLAG_KEY_FRAME != 0,
        pts_us: pts_and_flags & PTS_MASK,
        data,
    }))
}

/// Like `read_exact`, but returns `Ok(false)` instead of erroring if zero bytes
/// could be read before EOF (a clean disconnect between packets).
fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(Error::Protocol(
                    "connection closed mid-header".to_string(),
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn synth_codec_meta(width: u32, height: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&CODEC_ID_H264.to_be_bytes());
        v.extend_from_slice(&width.to_be_bytes());
        v.extend_from_slice(&height.to_be_bytes());
        v
    }

    fn synth_packet(flags: u64, pts_us: u64, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(flags | (pts_us & PTS_MASK)).to_be_bytes());
        v.extend_from_slice(&(data.len() as u32).to_be_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn parses_codec_meta() {
        let bytes = synth_codec_meta(1920, 1080);
        let meta = read_codec_meta(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
    }

    #[test]
    fn rejects_wrong_codec() {
        let mut bytes = synth_codec_meta(1920, 1080);
        bytes[3] = 0; // corrupt codec id
        assert!(read_codec_meta(&mut Cursor::new(bytes)).is_err());
    }

    #[test]
    fn parses_config_and_key_frame_packets() {
        let config = synth_packet(FLAG_CONFIG, 0, &[0, 0, 0, 1, 0x67]); // fake SPS NAL
        let key = synth_packet(FLAG_KEY_FRAME, 123_456, &[0, 0, 0, 1, 0x65]); // fake IDR

        let mut stream = Cursor::new([config, key].concat());

        let p1 = read_frame_packet(&mut stream).unwrap().unwrap();
        assert!(p1.is_config);
        assert!(!p1.is_key_frame);
        assert_eq!(p1.data, vec![0, 0, 0, 1, 0x67]);

        let p2 = read_frame_packet(&mut stream).unwrap().unwrap();
        assert!(!p2.is_config);
        assert!(p2.is_key_frame);
        assert_eq!(p2.pts_us, 123_456);

        assert!(read_frame_packet(&mut stream).unwrap().is_none());
    }

    #[test]
    fn clean_eof_between_packets_is_none() {
        let mut stream = Cursor::new(Vec::<u8>::new());
        assert!(read_frame_packet(&mut stream).unwrap().is_none());
    }
}
