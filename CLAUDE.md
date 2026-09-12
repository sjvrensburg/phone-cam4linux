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
cargo build --release -p pc4l-gui --no-default-features   # GUI without the built-in ONNX models (no ort download) or Typst
cargo test --workspace                 # unit tests (protocol parser, camera listing, pixel conversion, GUI crop geometry, quad rectification)
cargo test -p phone-cam4linux protocol::tests::parses_codec_meta   # single test
cargo clippy --workspace --all-targets [--features ffmpeg]
cargo fmt --all -- --check             # CI enforces this and clippy -D warnings, both feature sets
```

`pc4l-gui --rotate 270 --screenshot-after 8 --screenshot-path /tmp/gui.png` renders the
window against the phone and writes a PNG of it; add `--dev-detect [--dev-read-all]`
to start in block mode (and read every block) unattended.

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

## Releases

`.github/workflows/release.yml` runs on a `v*` tag: builds both binaries on
ubuntu-24.04 (the prebuilt ONNX Runtime needs glibc 2.38), packages them with the
dereferenced `libwebgpu_dawn.so`, docs and `contrib/`, builds the models archive with
`pc4l-gui --fetch-model` (so it carries the in-binary checksums), and publishes a
GitHub release with `SHA256SUMS.txt`. Cut one with
`git tag vX.Y.Z && git push --tags` after bumping `[workspace.package].version`.
`LICENSE` is Apache-2.0 and `NOTICE` lists third-party terms -- keep it current when a
component is added (a model, a runtime, ported code).

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
   (4-byte codec id, then 12-byte packet headers: a *session meta* packet carrying
   width/height before each encoder session, else pts/flags + size + Annex-B payload;
   flag bits 63/62/61 = session/config/keyframe). Reverse-engineered against the pinned
   server version (4.1; 3.x had a 12-byte codec header and no session packets) and
   unit-tested against hand-built fixtures. A mid-stream session meta at a new size ends
   the session with `Error::StreamResized` so the reconnect loop reopens the sink.
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

`crates/pc4l-gui` is the egui document-camera window (`stream.rs`: worker thread with
the reconnect loop, publishing the latest `YuvFrame`; `app.rs`: preview, crop in
*view* (rotated) coordinates mapped back to the source frame, capture, save;
`transcribe.rs`: the `Transcriber` trait, the OpenAI-compatible and halo-workbench
`/hint/read` backends, and the `~/.config/pc4l/gui.toml` `Config` (backend list,
`[layout]`, `[prompts]`, `[ui] scale`) -- the default prompts are verbatim from
halo-workbench's `handwriting.py` and travel with each read (`Transcriber::read`
takes the prompt; the hint API ignores it),
readings are grouped and counted, never merged; `layout.rs`: the `BlockDetector`
trait, `Block`/`Quad` in view space, `Role` (the 25 classes folded into
text/formula/figure/other -- colour and prompt follow it, `Mode::Formula` for a
formula block) and the perspective `rectify` (imageproc) a
non-rectangular block goes through before it is shown or read -- feature-independent
so the window builds without a detector; `history.rs`: every finished read of the
session (`App::history`, appended alongside `results`, which only ever drops its
prefix -- "copy all" relies on that), Markdown export by capture; `settings.rs`: the Settings window editing
a draft `Config`, applied by `App::apply_config` (backends whose entry is unchanged
are kept, so the local model is not reloaded; the detector is rebuilt through the
`DetectorFactory` main.rs passes in; the scale is egui's zoom factor and the value
in force is authoritative -- `track_zoom` writes any change into config and draft
and saves it at once, so Save/Cancel never touch it); `mathtext.rs` (feature `math`): readings
typeset by Typst -- `$…$`/`$$…$$`/`\(…\)`/`\[…\]` segments converted by the `mitex`
crate and evaluated inside MiTeX's Typst scope (vendored under `assets/mitex/`, so
`\operatorname` and friends resolve), the rest escaped as markup, rasterised by
`typst-render` at the window's pixel density and cached per reading as a texture
(`app::Typeset`); a reading that fails to compile is shown as text. The `mitex`
crate's built-in spec predates Typst 0.15's symbol renames (`diff`→`partial`,
`sect`→`inter`, `plus.circle`→`plus.o`, …), so `modernise` rewrites its output by
the `RENAMES` table -- extend it when a reading fails with "unknown variable";
`local/`: the
built-in models --
`local/glmocr.rs` drives the onnx-community three-graph GLM-OCR export through `ort`
(vision encoder, embeddings, merged decoder with an explicit KV cache and the
undocumented scalar `num_logits_to_keep` input; preprocessing and MRoPE position ids
ported from oar-ocr-vl's Candle implementation), `local/layout.rs` is PP-DocLayoutV3
(official ONNX export: 800x800 stretched input, `[N,7]` boxes with a reading-order
column plus `[N,200,200]` instance masks; mask → largest contour → approxPolyDP →
min-area rect gives the quad, as PaddleX does; `suppress_overlaps` is the
cross-class NMS PaddleX also runs, since the model reports the same lines twice at
times), `local/models.rs` finds or downloads
each model's files (pinned HF revision + sha256 manifest; `$PC4L_MODEL_DIR/<name>/`,
exe-adjacent `models/<name>/`, then `~/.cache/pc4l/models/<name>/`), and
`local/mod.rs` holds the one process-wide `ort` environment (`ort` refuses a second)
and wraps GLM-OCR as a `Transcriber` that prepares on a thread, plus `RUNTIME`, the
one lock every session load and run takes: the WebGPU EP segfaults on concurrent
`run` across sessions (microsoft/onnxruntime#32561, open) -- keep it until the
pinned runtime has the fix; and `GPU_LOST`, set by `note_gpu_loss` when a run fails
with a lost device (with IO binding the loss surfaces as an ORT error, not a
segfault), after which `attempts()` yields the CPU only and both services drop their
model and `prepare()` again. In `app.rs` the crop
is a rectangle plus an optional quad (`Selection`); any hand edit of the crop drops
the quad (`set_rect`) except dragging a quad corner, which moves that corner and
refits the rectangle. Block mode (`block_mode`) re-runs the detector whenever the
shown frame is new (throttled to `LIVE_DETECT_INTERVAL` live, paused while a read
holds the GPU), so blocks follow zoom and aim; a fresh detection hands the selection
to the new block with the highest IoU (`follow_selection`, so tab keeps its place and
an untouched crop tracks its block); "read all" captures first, waits for the
capture's own detection, then drains a snapshot queue one read at a time. `ort` is pinned to a git commit because the published rc.13 has a different
API; its `download-binaries` fetches pyke's prebuilt ONNX Runtime at build time, and
the WebGPU provider is a separate `libwebgpu_dawn.so` that lands next to the binary
(as a symlink into `~/.cache/dfbin` -- copy the real file into a release tarball),
found via the `$ORIGIN` rpath from `build.rs`. `--no-default-features` builds without
any of this. The hidden `--screenshot-after SECS --screenshot-path FILE`,
`--dev-crop X,Y,W,H`, `--dev-read [--dev-second]`, `--dev-detect`, `--dev-read-all` and `--dev-settings` flags let you
drive it from a script (GNOME blocks external screenshots of the window);
`XDG_CONFIG_HOME` points it at a scratch backend config.

### Non-obvious protocol details (hard-won, don't regress)

These are load-bearing and were each the cause of a real failure during bring-up:

- **`tunnel_forward=true` is required.** We reach the server through `adb forward`, so
  the server must *listen*; its default assumes `adb reverse` (connect-back).
- **`send_dummy_byte=true` + read that byte before the codec header.** With `adb forward`,
  the local TCP connect succeeds *before* the device-side socket exists, so an early
  read gets EOF. `connect_with_retry` reconnects until the dummy byte arrives.
- **`send_stream_meta=true`** is the 4.x name of what 3.x called `send_codec_meta`;
  the server only warns about unknown options, so a stale name silently changes the
  stream layout.
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

## Camera controls

scrcpy 4.x exposes exactly two: zoom (`camera_zoom=` at start, Camera2
`CONTROL_ZOOM_RATIO`, Android 11+; the range comes from `--list-sizes` /
`CameraInfo::zoom_range`) and torch (`camera_torch=`). Both are also live control-socket
messages (TYPE_CAMERA_ZOOM_IN/OUT = 19/20, step ×1.0625; TYPE_CAMERA_SET_TORCH = 18)
-- `ConnectOptions::control` opens that second connection (made right after the video
socket's dummy byte; only the first connection gets one) and `CameraSession::control()`
hands out a cloneable `CameraControl` usable from any thread while `run()` blocks
(`phone-cam4linux/examples/control.rs` shows it). The phone never reports the zoom it
ends up at, so the GUI tracks the step count itself. No exposure, focus or
white-balance control exists at any version.

## Scope

Camera→V4L2 only (no audio, display mirroring, or input control). USB and TCP/IP ADB
(`--connect`); the one-time `--tcpip` switch needs USB. Cross-platform virtual-camera sinks (Windows/macOS) are explicitly out of
scope — V4L2 is Linux-only.
