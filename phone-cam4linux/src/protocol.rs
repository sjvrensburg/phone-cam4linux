//! scrcpy's client-side video socket protocol.
//!
//! This is undocumented wire format, reverse-engineered from the scrcpy source
//! (`server/src/main/java/com/genymobile/scrcpy/device/Streamer.java` and
//! `video/SurfaceEncoder.java` in the pinned scrcpy release, see
//! `SCRCPY_SERVER_VERSION` in `build.rs`). It is pinned to that server version;
//! bumping the server version requires re-checking this module against the new
//! source before trusting it against a real device.
//!
//! With `send_device_meta=false send_stream_meta=true send_frame_meta=true` (the
//! options this crate always passes -- see `session.rs`; the dummy byte is consumed
//! before this module sees the stream), the video socket carries, in order:
//!
//! 1. The codec id (4 bytes, sent once): `codec_id: u32BE`. For H.264 it is the
//!    ASCII bytes `"h264"` read as a big-endian u32 (`0x68323634`). The values `0`
//!    and `1` mean the server disabled the stream (1 = configuration error).
//! 2. A stream of packets, each starting with a 12-byte header. The first 8 bytes,
//!    read as `u64BE`, are `pts_and_flags`; their top three bits are flags:
//!    - bit 63: **session meta** -- the packet has no payload; bytes 4..8 and 8..12
//!      of the header are `width: u32BE` and `height: u32BE`. One is sent before
//!      the first frame of every encoder session (so at least once, at the start),
//!      and again if the encoder restarts at a new size;
//!    - bit 62: config packet (SPS/PPS), bit 61: key frame; for these and plain
//!      frames the low 61 bits are the PTS in microseconds (meaningless for config
//!      packets) and bytes 8..12 are `packet_size: u32BE`, followed by that many
//!      bytes of Annex-B H.264 data.
//!
//! scrcpy 3.x carried width/height in a 12-byte codec header instead and had no
//! session packets; that layout is gone.

use crate::error::{Error, Result};
use std::io::Read;

const CODEC_ID_H264: u32 = 0x6832_3634; // b"h264" as big-endian u32

const FLAG_SESSION: u64 = 1 << 63;
const FLAG_CONFIG: u64 = 1 << 62;
const FLAG_KEY_FRAME: u64 = 1 << 61;
const PTS_MASK: u64 = !(FLAG_SESSION | FLAG_CONFIG | FLAG_KEY_FRAME);

/// The stream's frame size, from a session-meta packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// One unit of the stream after the codec id.
#[derive(Debug, Clone)]
pub enum Packet {
    /// The encoder (re)started at this size.
    SessionMeta(CodecMeta),
    Frame(FramePacket),
}

fn io_or_protocol(e: std::io::Error, what: &str) -> Error {
    match e.kind() {
        // Keep interruptions/timeouts distinguishable for the caller (see session.rs).
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::Other => Error::Io(e),
        _ => Error::Protocol(format!("{what}: {e}")),
    }
}

/// Reads the start of the stream: the codec id, then the first session-meta packet
/// with the frame size. Fails on any other codec or on a disabled stream.
pub fn read_codec_meta<R: Read>(r: &mut R) -> Result<CodecMeta> {
    let mut id = [0u8; 4];
    r.read_exact(&mut id)
        .map_err(|e| io_or_protocol(e, "reading codec id"))?;
    let codec_id = u32::from_be_bytes(id);
    match codec_id {
        CODEC_ID_H264 => {}
        0 | 1 => {
            return Err(Error::Protocol(format!(
                "the server disabled the video stream (code {codec_id})"
            )))
        }
        _ => {
            return Err(Error::Protocol(format!(
                "unsupported codec id 0x{codec_id:08x} (expected h264); \
                 video_codec=h264 must be forced in server args"
            )))
        }
    }
    match read_packet(r)? {
        Some(Packet::SessionMeta(meta)) => Ok(meta),
        Some(Packet::Frame(_)) => Err(Error::Protocol(
            "frame data before the first session meta (send_stream_meta off?)".into(),
        )),
        None => Err(Error::Protocol(
            "connection closed before the first session meta".into(),
        )),
    }
}

/// Reads one packet (session meta, or header + payload). Returns `Ok(None)` on clean
/// EOF (socket closed between packets, i.e. the phone was disconnected / app killed).
pub fn read_packet<R: Read>(r: &mut R) -> Result<Option<Packet>> {
    let mut header = [0u8; 12];
    if !read_exact_or_eof(r, &mut header)? {
        return Ok(None);
    }
    let pts_and_flags = u64::from_be_bytes(header[0..8].try_into().unwrap());
    if pts_and_flags & FLAG_SESSION != 0 {
        return Ok(Some(Packet::SessionMeta(CodecMeta {
            width: u32::from_be_bytes(header[4..8].try_into().unwrap()),
            height: u32::from_be_bytes(header[8..12].try_into().unwrap()),
        })));
    }
    let size = u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize;

    let mut data = vec![0u8; size];
    r.read_exact(&mut data)
        .map_err(|e| io_or_protocol(e, &format!("reading frame payload ({size} bytes)")))?;

    Ok(Some(Packet::Frame(FramePacket {
        is_config: pts_and_flags & FLAG_CONFIG != 0,
        is_key_frame: pts_and_flags & FLAG_KEY_FRAME != 0,
        pts_us: pts_and_flags & PTS_MASK,
        data,
    })))
}

/// Like `read_exact`, but returns `Ok(false)` instead of erroring if zero bytes
/// could be read before EOF (a clean disconnect between packets).
fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(Error::Protocol("connection closed mid-header".to_string()));
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

    fn synth_session_meta(width: u32, height: u32) -> Vec<u8> {
        // The Java side writes the flags as an int (the high half of the u64) then
        // width and height; the low bit of that int marks a client-initiated resize.
        let mut v = Vec::new();
        v.extend_from_slice(&((FLAG_SESSION >> 32) as u32).to_be_bytes());
        v.extend_from_slice(&width.to_be_bytes());
        v.extend_from_slice(&height.to_be_bytes());
        v
    }

    fn synth_stream_start(width: u32, height: u32) -> Vec<u8> {
        let mut v = CODEC_ID_H264.to_be_bytes().to_vec();
        v.extend(synth_session_meta(width, height));
        v
    }

    fn synth_packet(flags: u64, pts_us: u64, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(flags | (pts_us & PTS_MASK)).to_be_bytes());
        v.extend_from_slice(&(data.len() as u32).to_be_bytes());
        v.extend_from_slice(data);
        v
    }

    fn frame(p: Option<Packet>) -> FramePacket {
        match p {
            Some(Packet::Frame(f)) => f,
            other => panic!("expected a frame, got {other:?}"),
        }
    }

    #[test]
    fn parses_codec_meta() {
        let meta = read_codec_meta(&mut Cursor::new(synth_stream_start(1920, 1080))).unwrap();
        assert_eq!(meta.width, 1920);
        assert_eq!(meta.height, 1080);
    }

    #[test]
    fn rejects_wrong_codec_and_disabled_stream() {
        let mut bytes = synth_stream_start(1920, 1080);
        bytes[3] = 0; // corrupt codec id
        assert!(read_codec_meta(&mut Cursor::new(bytes)).is_err());
        let err = read_codec_meta(&mut Cursor::new(1u32.to_be_bytes())).unwrap_err();
        assert!(err.to_string().contains("disabled"), "{err}");
    }

    #[test]
    fn frame_before_session_meta_is_an_error() {
        let mut bytes = CODEC_ID_H264.to_be_bytes().to_vec();
        bytes.extend(synth_packet(FLAG_CONFIG, 0, &[0, 0, 0, 1, 0x67]));
        assert!(read_codec_meta(&mut Cursor::new(bytes)).is_err());
    }

    #[test]
    fn parses_config_key_frame_and_mid_stream_session_packets() {
        let config = synth_packet(FLAG_CONFIG, 0, &[0, 0, 0, 1, 0x67]); // fake SPS NAL
        let key = synth_packet(FLAG_KEY_FRAME, 123_456, &[0, 0, 0, 1, 0x65]); // fake IDR
        let plain = synth_packet(0, 123_789, &[0, 0, 0, 1, 0x41]);
        let resize = synth_session_meta(1280, 720);

        let mut stream = Cursor::new([config, key, plain, resize].concat());

        let p1 = frame(read_packet(&mut stream).unwrap());
        assert!(p1.is_config);
        assert!(!p1.is_key_frame);
        assert_eq!(p1.data, vec![0, 0, 0, 1, 0x67]);

        let p2 = frame(read_packet(&mut stream).unwrap());
        assert!(!p2.is_config);
        assert!(p2.is_key_frame);
        assert_eq!(p2.pts_us, 123_456);

        let p3 = frame(read_packet(&mut stream).unwrap());
        assert!(!p3.is_config && !p3.is_key_frame);
        assert_eq!(p3.pts_us, 123_789);

        match read_packet(&mut stream).unwrap() {
            Some(Packet::SessionMeta(m)) => assert_eq!((m.width, m.height), (1280, 720)),
            other => panic!("expected session meta, got {other:?}"),
        }
        assert!(read_packet(&mut stream).unwrap().is_none());
    }

    #[test]
    fn clean_eof_between_packets_is_none() {
        let mut stream = Cursor::new(Vec::<u8>::new());
        assert!(read_packet(&mut stream).unwrap().is_none());
    }
}
