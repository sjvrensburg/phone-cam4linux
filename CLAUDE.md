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
cargo test --workspace                 # unit tests (protocol parser, pixel conversion)
cargo test -p phone-cam4linux protocol::tests::parses_codec_meta   # single test
cargo clippy --workspace --all-targets
```

Run against a phone (USB debugging authorized, Android 12+):
```
./target/release/pc4l --facing back --resolution 3840x2160 --bitrate 30 --device /dev/video10
```

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
   and launches the server via `app_process`.
2. **`session.rs`** — the orchestrator and public API (`CameraSession`, `ConnectOptions`,
   `Facing`). `connect()` starts the server with a fixed set of scrcpy options and
   completes the handshake; `run_to_v4l2()` is the blocking decode→convert→write loop.
3. **`protocol.rs`** — parses scrcpy's **undocumented** video-socket wire format
   (12-byte codec-meta header, then per-frame 12-byte header + Annex-B payload).
   Reverse-engineered against the pinned server version; unit-tested against
   hand-built fixtures.
4. **`decode.rs`** — `openh264` (statically linked via `source` feature), Annex-B → I420.
5. **`convert.rs`** — I420 → packed YUYV422.
6. **`sink.rs`** — `v4l` crate mmap output stream to `/dev/videoN`.
7. **`loopback.rs`** — auto-loads `v4l2loopback` via `pkexec modprobe` if the device
   node is missing.

`crates/pc4l` is a thin clap CLI over this library.

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

The bundled `openh264` decoder handles up to **3840x2160**. Larger camera modes
(e.g. 4000x3000) make openh264 reject the SPS (`dsNoParamSets`, native error 16).
`run_to_v4l2` tolerates transient decode errors but surfaces a clear
"resolution may exceed openh264" error if nothing decodes at all.

## Scope

USB ADB only; camera→V4L2 only (no Wi-Fi/TCP ADB, audio, display mirroring, or input
control). Cross-platform virtual-camera sinks (Windows/macOS) are explicitly out of
scope — V4L2 is Linux-only.
