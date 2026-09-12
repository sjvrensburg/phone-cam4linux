# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust workspace that exposes an Android phone's camera as a Linux V4L2 (`/dev/videoN`)
webcam. The on-device capture/encode is **not** reimplemented: the real upstream
`scrcpy-server.jar` is fetched at build time, embedded, and pushed to the phone over
ADB. Everything from the socket down (scrcpy's client protocol, H.264 decode, pixel
conversion, V4L2 sink) is implemented here.

## Commands

```
cargo build --release                 # fetches scrcpy-server.jar on first build (needs network)
cargo build --release --features ffmpeg   # + system libavcodec decoder (needs full FFmpeg headers)
cargo test --workspace                 # unit tests (protocol parser, camera listing, pixel conversion)
cargo test -p phone-cam4linux protocol::tests::parses_codec_meta   # single test
cargo clippy --workspace --all-targets [--features ffmpeg]
cargo fmt --all -- --check             # CI enforces this and clippy -D warnings, both feature sets
```

Run against a phone (USB debugging authorized, Android 12+):
```
./target/release/pc4l --list-sizes
./target/release/pc4l --facing back --resolution max --bitrate 30 --device /dev/video10
```

Testing tips (no `ffmpeg` CLI needed): grab a frame from the loopback with
`gst-launch-1.0 -q v4l2src device=/dev/video10 num-buffers=1 ! videoconvert ! jpegenc ! filesink location=f.jpg`;
`RUST_LOG=debug` prints a 5 s throughput report and relays the scrcpy server's own log.
The phone's camera app must be closed (`adb shell input keyevent KEYCODE_HOME`) or the
server fails with `CAMERA_IN_USE`.

Exercise the whole V4L2 sink path with **no phone attached**:
```
./target/release/pc4l --test-pattern --device /dev/video10
```

## Build-time network dependency

`phone-cam4linux/build.rs` downloads `scrcpy-server-v<SCRCPY_VERSION>` from GitHub
releases and verifies it against a pinned `SERVER_SHA256`, writing it to `OUT_DIR`
for `include_bytes!`. Consequences:

- A clean build needs network access. For offline/CI builds, pre-fetch the jar and
  point `PHONE_CAM4LINUX_SERVER_JAR=/path/to/scrcpy-server.jar` at it.
- Bumping the scrcpy version means changing **both** `SCRCPY_VERSION` and
  `SERVER_SHA256` together, and re-checking `protocol.rs` (see below).

## Architecture

The pipeline, in data-flow order (all in `phone-cam4linux/src/`):

1. **`adb.rs`** — shells out to the system `adb` binary (not a Rust ADB lib; see the
   module doc for why). Pushes the jar, sets up `adb forward tcp:0 localabstract:scrcpy_<scid>`,
   and launches the server via `app_process`. Also `connect_tcp` / `enable_tcpip` for
   Wi-Fi (`adb connect` exits 0 even on failure -- the verdict is in its text).
2. **`session.rs`** — the orchestrator and public API (`CameraSession`, `ConnectOptions`,
   `Facing`). `connect()` starts the server with a fixed set of scrcpy options and
   completes the handshake; `connect_with_stop()`/`run(sink, stop)`; the latter is the blocking decode→sink loop
   (stop flag checked per packet / every 500 ms; `STALL_TIMEOUT` of silence →
   `Error::StreamStalled`). The server's stdout/stderr is relayed into `log`.
   **`cameras.rs`** parses the server's `list_camera_sizes=true` report.
3. **`protocol.rs`** — parses scrcpy's **undocumented** video-socket wire format
   (12-byte codec-meta header, then per-frame 12-byte header + Annex-B payload).
   Reverse-engineered against the pinned server version; unit-tested against
   hand-built fixtures.
4. **`decode.rs`** — Annex-B → I420 via `openh264` (statically linked via `source`
   feature) or, with the `ffmpeg` feature, system libavcodec (`Backend::Ffmpeg`).
5. **`convert.rs`** — I420 → packed YUYV422 (V4L2) and → RGBA8 (whole, cropped region, or
   decimated for a preview) for on-screen display.
6. **`sink.rs`** — the `FrameSink` trait `run()` feeds (implemented by `V4l2Sink` and by any
   `FnMut(&YuvFrame) -> Result<()>` closure, which is how a GUI gets frames), and
   `V4l2Sink`: `v4l` crate mmap output stream to `/dev/videoN`.
7. **`loopback.rs`** — auto-loads `v4l2loopback` via `pkexec modprobe` if the device
   node is missing.

`crates/pc4l` is a clap CLI over this library; it owns the policy bits: Ctrl-C/SIGTERM
handling, the reconnect-with-backoff loop (keeping the V4L2 sink open across sessions),
`--list-sizes` and `--resolution max`. `contrib/` has boot-time loopback config and a
systemd user unit.

### Non-obvious protocol details (hard-won, don't regress)

These are load-bearing and were each the cause of a real failure during bring-up:

- **`tunnel_forward=true` is required.** We reach the server through `adb forward`, so
  the server must *listen*; its default assumes `adb reverse` (connect-back).
- **`send_dummy_byte=true` + read that byte before the codec header.** With `adb forward`,
  the local TCP connect succeeds *before* the device-side socket exists, so an early
  read gets EOF. `connect_with_retry` reconnects until the dummy byte arrives.
- **`scid` must fit in a signed 32-bit int.** scrcpy parses it with `Integer.parseInt(v, 16)`,
  so the top bit must be 0 (first hex digit ≤ 7). `random_scid_hex8` masks with `0x7fffffff`.

### Resolution ceiling

openh264 hard-codes H.264 level 5.2 (`GetLevelLimits(52)`, 36864 macroblocks) in its SPS
parser: 3840x2160 and 2992x2992 fit, 4000x3000 does not (`dsNoParamSets`, native
error 16). `decode::MAX_MACROBLOCKS` / `Backend::fits` encode this so the CLI rejects
oversized modes up front. The `ffmpeg` feature has no such limit (4000x3000 verified).

Separately, camera sizes that aren't multiples of 8 (4000x2250) fail on the *phone*:
scrcpy rounds them for the encoder and the camera refuses the rounded size
(`onConfigureFailed`, stream connects but no packets ever arrive). `cameras::is_usable_size`
filters these.

libavcodec gotchas in the ffmpeg backend: `avcodec_receive_frame` unrefs its destination
before returning EAGAIN (so drain into a scratch frame and swap), and an SPS/PPS-only
packet is "invalid data" (so it is prepended to the next packet, as scrcpy does).

## Scope

Camera→V4L2 only (no audio, display mirroring, or input control). USB and TCP/IP ADB
(`--connect`); the one-time `--tcpip` switch needs USB. Cross-platform virtual-camera sinks (Windows/macOS) are explicitly out of
scope — V4L2 is Linux-only.
