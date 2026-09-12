# phone-cam4linux

[![CI](https://github.com/sjvrensburg/phone-cam4linux/actions/workflows/ci.yml/badge.svg)](https://github.com/sjvrensburg/phone-cam4linux/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Use an Android phone as a camera on Linux, without the full `scrcpy` client:

- **`pc4l`** -- a CLI that streams the phone's camera into a V4L2 (`/dev/videoN`)
  device, so browsers, OBS and any webcam app can use it.
- **`pc4l-gui`** -- a desktop document-camera window: live view, the phone's own zoom,
  a drag-to-zoom region, capture, save, and handwriting transcription with a built-in
  model (GLM-OCR on your GPU) or any OpenAI-compatible vision endpoint.
- **`phone-cam4linux`** -- the Rust library both are built on.

Linux only (V4L2 is Linux; the GUI is Linux-first). Android 12+ on the phone.

## How it works

The risky part of this problem -- driving Android's Camera2 API and H.264-encoding
the frames -- is not reimplemented. Instead, this crate embeds the real, upstream
[`scrcpy-server.jar`](https://github.com/Genymobile/scrcpy) (Apache-2.0) at build
time and pushes/runs it on the phone over ADB, exactly as `scrcpy` itself does. What
this crate implements natively in Rust is:

- the ADB plumbing to push and launch that server (via the system `adb` binary),
- scrcpy's client-side video-socket wire protocol (`src/protocol.rs`),
- H.264 decoding via `openh264` (statically linked, no system FFmpeg) or,
  optionally, the system FFmpeg (`--features ffmpeg`, no frame-size ceiling),
- I420 -> YUYV422 conversion, and
- a V4L2 sink that writes frames into a `v4l2loopback` device.

This is a from-scratch reimplementation of scrcpy's *camera -> V4L2* feature scoped
down to just that (see `doc/camera.md` and `doc/v4l2.md` in the scrcpy repo for the
equivalent `scrcpy --video-source=camera --v4l2-sink=...` invocation) -- no display
mirroring, audio, or input control.

## Requirements

- `adb` on `PATH` (`android-tools` / `platform-tools`), with the phone's USB
  debugging authorized.
- Android 12+ on the phone (camera-as-video-source requires it).
- `v4l2loopback` kernel module installed (`v4l2loopback-dkms` on most distros).
  If the target `/dev/videoN` is missing, `pc4l` creates it via `pkexec` (which
  prompts for your password each time). To avoid that permanently, create the device
  at boot instead:
  ```
  sudo contrib/install-system-config.sh    # modprobe.d + modules-load.d for /dev/video10
  ```
- Optional, for the `ffmpeg` feature: libavcodec/libavutil development headers from a
  *full* FFmpeg (on Fedora that's RPM Fusion's `ffmpeg-devel`; `ffmpeg-free` lacks the
  native `h264` decoder).

## Install

**Release archive** (x86_64 Linux with glibc 2.38 or newer -- Ubuntu 24.04, Fedora 39,
Debian 13 and later -- and a CPU with AVX2, both requirements of the built-in model's
prebuilt ONNX Runtime): download `pc4l-<version>-x86_64-linux.tar.gz` from the
[releases page](https://github.com/sjvrensburg/phone-cam4linux/releases) and extract
it anywhere; it contains `pc4l`, `pc4l-gui`, the `libwebgpu_dawn.so` the GUI's GPU path
needs (found next to the binary), and `contrib/`. The built-in models (~780 MB:
transcription and block detection) are fetched on first use, or extract
`pc4l-models-<version>.tar.gz` from the same release into the same directory to have
them offline (`models/` beside the binaries). `SHA256SUMS.txt` covers both archives.

**From source**:

```
cargo build --release                      # or: cargo build --release --features ffmpeg
```

needs a Rust toolchain, `nasm` (OpenH264 assembly), `libclang` (bindgen for the V4L2
bindings), and for the GUI `libxkbcommon` and `libwayland` development files. The first
build downloads the pinned `scrcpy-server` jar and (for the GUI) prebuilt ONNX Runtime
binaries; `cargo build --release -p pc4l-gui --no-default-features` skips the latter and
the built-in models. `pc4l-gui --fetch-model DIR` downloads the models into `DIR/` with
checksum verification, for machines that will be offline (set `HF_TOKEN` to a Hugging
Face token if anonymous downloads are being rate-limited; the files are public).

## Usage

```
./target/release/pc4l --list-sizes         # see what the phone offers
./target/release/pc4l --facing back --resolution max --bitrate 30 --device /dev/video10
```

Then point any V4L2-consuming app (browser, `ffplay`, OBS, etc.) at `/dev/video10`.
`pc4l` keeps running until Ctrl-C: if the phone disconnects, the server dies, or the
stream stalls, it reconnects with backoff while keeping the V4L2 device open, so
consumers don't lose the camera (`--no-reconnect` to exit instead).

Options:

- `--list-sizes` -- print each camera's supported capture sizes and exit, marking the
  ones the current decoder can't handle.
- `--facing front|back` -- which camera (default `back`).
- `--resolution WxH|max` -- capture size; must be one of the sizes from `--list-sizes`
  (and a multiple of 8 in both dimensions, see below). `max` picks the largest usable
  one. Defaults to the phone's choice.
- `--bitrate MBPS` -- H.264 bitrate in Mbit/s (default `30`). Higher is crisper for
  reading text/documents.
- `--fps N` -- cap the frame rate.
- `--zoom RATIO` -- the phone's own camera zoom (optical/sensor, not a crop of the
  stream), e.g. `2.5`; `--list-sizes` shows each camera's range. Android 11+.
- `--torch` -- keep the flash on while streaming.
- `--decoder openh264|ffmpeg` -- H.264 decoder (`ffmpeg` only with the feature; it's
  then the default).
- `--serial SERIAL` -- pick a device when more than one is attached.
- `--no-reconnect` -- exit on the first failure instead of retrying.

### Document-camera use / resolution ceiling

The bundled `openh264` decoder is hard-limited to H.264 level 5.2 frame sizes
(36864 macroblocks: 3840x2160 fits, and so does 2992x2992; 4000x3000 does not).
Build with `--features ffmpeg` to decode with the system libavcodec instead, which has
no such limit -- 4000x3000 (12 MP) at 25 fps has been verified that way.

Sizes that aren't a multiple of 8 in both dimensions (e.g. 4000x2250) are listed by the
phone but unusable: scrcpy rounds them for the encoder and the camera then refuses the
rounded size. `--list-sizes` marks these; `--resolution max` skips them.

### Desktop window (`pc4l-gui`)

`pc4l-gui` is a document-camera window over the same pipeline, with no V4L2 device
needed: a live view, a drag-to-select region shown at native pixels beside it (that
*is* the zoom), Capture to freeze the frame, and Save PNG for the region or the whole
frame at full resolution.

```
./target/release/pc4l-gui --rotate 270           # phone on a stand, mounted sideways
./target/release/pc4l-gui --device /dev/video10  # also feed the loopback device
./target/release/pc4l-gui --zoom 2               # start at 2x
```

When the phone reports a zoom range for the camera, the toolbar has a **Zoom** slider
and a **Torch** toggle that act live through scrcpy's control channel (the phone's own
zoom, in x1.0625 steps -- not a crop of the stream).

**Read it** sends the box (or the whole page) to a transcription backend and lists
every distinct answer with how many samples gave it -- several readings are shown as
several readings, never merged, and an empty answer is reported, not hidden. Backends
live in `~/.config/pc4l/gui.toml` (written with defaults on first run):

- `kind = "local"` -- the built-in model, [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
  (0.9B, handwriting and math) as the `onnx-community` q4f16 ONNX export, run through
  ONNX Runtime on the GPU via WebGPU/Vulkan, falling back to CPU (`device = "auto" |
  "webgpu" | "cpu"`). About 1 s per crop on a Radeon 8060S, 2 s on its CPU. Greedy, so
  one reading per request. Needs the default `local-model` cargo feature.
- `kind = "open-ai"` -- any OpenAI-compatible chat endpoint with image input
  (llama-server, Ollama, vLLM, OpenAI); `samples > 1` asks several times at
  `temperature` and shows the spread.
- `kind = "hint-api"` -- halo-workbench's `/hint/read`.

**Blocks** (`L`, a toggle) keeps the built-in layout model,
[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3), running over
the page as you aim (a few times a second live, once on a capture): every text block,
formula, figure and so on, numbered in the reading order the model predicts, drawn on
the preview. Click a block or `tab`/`shift+tab` through them to make it the region,
then drag its corners if the model's box is not quite what you want; **Read all
blocks** (`ctrl+enter`) reads them one after the other and lists the readings in page
order. The model predicts multi-point boxes, so on a curved or tilted page a block is a
quadrilateral, not a rectangle; such a block is perspective-rectified before it is
shown and read. About 0.1 s on the GPU, 0.3 s on the CPU. `[layout]` in `gui.toml`
turns it off, picks the device, and sets the score threshold (0.4: handwriting scores
lower than the printed pages it was trained on).

The built-in models' files (~658 MB for GLM-OCR, 130 MB for the layout model) are not
inside the binary. Each is looked for in `$PC4L_MODEL_DIR/<name>/`, then
`models/<name>/` next to the executable (how a release tarball can ship them), then
`~/.cache/pc4l/models/<name>/` (`glm-ocr-onnx-q4f16`, `pp-doclayoutv3-onnx`); if
none has it, it is downloaded there on first run from a pinned Hugging Face revision,
each file verified against a sha256 compiled into the app, with progress shown in the
window. Reads and detection are refused until the model is ready.

Keys: `space` capture/retake, `enter` read, `L` block mode on/off, `tab`/`shift+tab`
next/previous block, `ctrl+enter` read all blocks, `esc` clear the region (then
retake), `R`/`shift+R` rotate, `ctrl+S` save (to `~/Pictures/pc4l/`, or `--save-dir`).
Phone zoom: the slider, the wheel over the preview, `+`/`-`, `0` to reset. The
region: drag inside it to move it, drag a corner handle to reshape it, arrow keys to
nudge (`shift` for one pixel), `[`/`]` or the wheel over the zoomed view to shrink/grow
it. `--resolution` defaults to
`max`; `--facing`, `--connect`, `--serial`, `--bitrate`, `--fps` and `--decoder`
are as for `pc4l`. It reconnects with backoff like the CLI.

### Wireless (TCP/IP ADB)

Once, with the phone on USB and Wi-Fi:

```
pc4l --tcpip            # switches adbd to TCP mode, prints e.g. 192.168.1.53:5555
```

Then unplug and stream over Wi-Fi (the `adb connect` is re-issued on every reconnect,
so a Wi-Fi hiccup is recovered like a cable wiggle):

```
pc4l --connect 192.168.1.53 --resolution 1920x1080
```

TCP mode persists until the phone reboots; `adb usb` switches back. 1080p at 30 Mbit/s
streams fine over a decent Wi-Fi link; drop `--bitrate` if you see stalls.

### Running as a service

`contrib/systemd/pc4l.service` is a systemd *user* unit that keeps the camera exposed
whenever the phone is reachable (see the comments in the file for install steps). It
relies on the boot-time device from `contrib/install-system-config.sh` and on `pc4l`
being installed (`cargo install --path crates/pc4l [--features ffmpeg]`).

### Testing the V4L2 sink without a phone

```
./target/release/pc4l --test-pattern --device /dev/video10
```

Writes a synthetic cycling color-bar pattern instead of a real camera stream --
exercises the loopback/format-negotiation/write path independently of ADB/hardware.

## Status / caveats

- Verified end-to-end against a real device (Samsung SM-A307FN running Android 13
  via crDroid) at 1920x1080, 2992x2992 (openh264) and 4000x3000 (ffmpeg), including
  the GUI, live zoom/torch, and the built-in models on a Radeon 8060S (RADV) via WebGPU.
- The GUI's built-in models run on WebGPU (Vulkan on Linux), an execution provider ONNX
  Runtime still calls experimental; it falls back to the CPU if the provider cannot be
  set up, but a GPU driver fault mid-inference takes the process down. Images are
  capped at 2048 image tokens for that reason.
- **Protocol pinning**: `src/protocol.rs` implements scrcpy's undocumented
  video-socket wire format, reverse-engineered against the pinned server version in
  `build.rs` (`SCRCPY_VERSION`). Re-verify this module if you bump `SCRCPY_VERSION`.
- Wi-Fi works via TCP/IP ADB (`--tcpip` / `--connect`); the initial switch to TCP
  mode still needs the USB cable once per phone boot.
- No audio, display mirroring, or input control -- camera-to-V4L2 only.
- Decode ceiling is openh264's unless built with `--features ffmpeg`; see
  "resolution ceiling" above.

## License

Apache-2.0 (see `LICENSE`). `NOTICE` lists the third-party components this project
embeds, downloads or links -- notably the upstream `scrcpy-server` (Apache-2.0),
OpenH264 built from source (BSD-2-Clause; Cisco's H.264 royalty coverage applies only to
Cisco's own binaries), ONNX Runtime and Dawn (MIT / BSD-3-Clause), the GLM-OCR model
(MIT), the PP-DocLayoutV3 model and the post-processing ported from PaddleX (Apache-2.0),
and preprocessing code ported from oar-ocr (Apache-2.0).
