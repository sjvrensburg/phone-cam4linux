# Squigl

[![CI](https://github.com/sjvrensburg/squigl/actions/workflows/ci.yml/badge.svg)](https://github.com/sjvrensburg/squigl/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Use an Android phone as a document camera on Linux, without the full `scrcpy` client:

- **`squigl`** -- a desktop document-camera window: live view, the phone's own zoom,
  a drag-to-zoom region, capture, save, and handwriting transcription with a built-in
  model (GLM-OCR on your GPU) or any OpenAI-compatible vision endpoint.
- **`squigl-cli`** -- a headless CLI that streams the phone's camera into a V4L2
  (`/dev/videoN`) device, so browsers, OBS and any webcam app can use it too.
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
  debugging authorized (accept the RSA key prompt that appears on the phone the first
  time you connect).
- Android 12+ on the phone (camera-as-video-source requires it).
- `v4l2loopback` kernel module installed (`v4l2loopback-dkms` on most distros).
  If the target `/dev/videoN` is missing, `squigl-cli` creates it via `pkexec` (which
  prompts for your password each time). To avoid that permanently, create the device
  at boot instead:
  ```
  sudo contrib/install-system-config.sh    # modprobe.d + modules-load.d for /dev/video10
  ```
- Optional, for the `ffmpeg` feature: libavcodec/libavutil development headers from a
  *full* FFmpeg (on Fedora that's RPM Fusion's `ffmpeg-devel`; `ffmpeg-free` lacks the
  native `h264` decoder).
- Optional, for GPU acceleration of the GUI's built-in models: Vulkan drivers (e.g.
  `mesa-vulkan-drivers`, or your GPU vendor's package). Not required -- without them
  the models still work, just on the CPU (see Status/caveats).

**On a machine you don't fully control** (a managed/corporate laptop), two things can
block setup before you even reach the app, and are worth checking first:

- **Secure Boot** rejects an unsigned, DKMS-built `v4l2loopback.ko` on most distros.
  The package's install hook usually offers to enroll a MOK (`mokutil --import ...`,
  confirmed with a password at the next boot); if that's locked down by policy, you'll
  need IT to enroll it, install a distro that ships it pre-signed, or (if policy
  allows) disable Secure Boot.
- **No `sudo`/`pkexec`** blocks both the on-demand `pkexec` prompt and
  `install-system-config.sh`. If you don't have either, ask an admin to run
  `install-system-config.sh` once (it just drops two config files and reloads the
  module) rather than trying to work around it.

## Install

**AppImage** (x86_64 Linux with glibc 2.38 or newer -- Ubuntu 24.04, Fedora 39, Debian
13 and later -- and a CPU with AVX2, both requirements of the built-in models' prebuilt
ONNX Runtime; `adb` installed): download `squigl-<version>-x86_64.AppImage` from the
[releases page](https://github.com/sjvrensburg/squigl/releases), `chmod +x`,
run. It carries the GUI, the CLI and the GPU provider; the built-in models (~780 MB)
are fetched on first use into `~/.cache/squigl/models/`, or extract
`squigl-models-<version>.tar.gz` next to the AppImage (a `models/` directory beside it)
to have them offline. A desktop entry and icon are in `contrib/appimage/` (or let an
AppImage launcher such as AppImageLauncher integrate it).

**Release archive** (same requirements): `squigl-<version>-x86_64-linux.tar.gz` contains
`squigl`, `squigl-cli`, the `libwebgpu_dawn.so` the GUI's GPU path needs (found next to
the binary), and `contrib/`. Extract `squigl-models-<version>.tar.gz` into the same
directory for the models offline (`models/` beside the binaries). `SHA256SUMS.txt`
covers everything. `contrib/appimage/build.sh` builds the AppImage from a release build.

**From source**:

```
cargo build --release                      # or: cargo build --release --features ffmpeg
```

needs a Rust toolchain, `nasm` (OpenH264 assembly), `libclang` (bindgen for the V4L2
bindings), and for the GUI `libxkbcommon` and `libwayland` development files. The first
build downloads the pinned `scrcpy-server` jar and (for the GUI) prebuilt ONNX Runtime
binaries; `cargo build --release -p squigl --no-default-features` skips the latter,
the built-in models and the Typst typesetting. `squigl --fetch-model DIR` downloads the models into `DIR/` with
checksum verification, for machines that will be offline (set `HF_TOKEN` to a Hugging
Face token if anonymous downloads are being rate-limited; the files are public).

## Usage

```
./target/release/squigl-cli --list-sizes         # see what the phone offers
./target/release/squigl-cli --facing back --resolution max --bitrate 30 --device /dev/video10
```

Then point any V4L2-consuming app (browser, `ffplay`, OBS, etc.) at `/dev/video10`.
`squigl-cli` keeps running until Ctrl-C: if the phone disconnects, the server dies, or the
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

### Desktop window (`squigl`)

`squigl` is a document-camera window over the same pipeline, with no V4L2 device
needed: a live view, a drag-to-select region shown at native pixels beside it (that
*is* the zoom), Capture to freeze the frame, and Save PNG for the region or the whole
frame at full resolution.

```
./target/release/squigl --rotate 270           # phone on a stand, mounted sideways
./target/release/squigl --device /dev/video10  # also feed the loopback device
./target/release/squigl --zoom 2               # start at 2x
```

When the phone reports a zoom range for the camera, the toolbar has a **Zoom** slider
and a **Torch** toggle that act live through scrcpy's control channel (the phone's own
zoom, in x1.0625 steps -- not a crop of the stream).

**Read it** sends the box (or the whole page) to a transcription backend and lists
every distinct answer with how many samples gave it -- several readings are shown as
several readings, never merged, and an empty answer is reported, not hidden. Readings
are shown **typeset**: the LaTeX the models write for maths (`$\hat{y}_i \neq y_i$`)
is converted to Typst by [MiTeX](https://github.com/mitex-rs/mitex) and rendered by
[Typst](https://typst.app) with its embedded fonts, so a formula can be checked
against the ink at a glance; the `typeset` checkbox shows the raw text instead, `copy`
always copies the raw text, and anything that does not convert stays text. Every
reading of the session is kept: **history** (`H`) lists them all with the capture they
came from, **copy all** puts the current list on the clipboard in page order, and
**Save as Markdown** writes the session to `~/Pictures/squigl/squigl-readings-<time>.md`,
a section per capture. **2nd opinion** (`shift+enter`) reads the same thing again with
the next backend in the list and lists it alongside -- the two are never merged. The
readings' text size is a Settings value (`[ui] reading_size`, points) and scales with
the window. (The
`math` cargo feature, on by default; about 40 MB of the binary.) Backends live in
`~/.config/squigl/gui.toml` (written with defaults on first run) and are edited in the
window's **Settings** (`ctrl+,`) -- an OpenAI-compatible endpoint's URL, model and API
key go there -- along with the **prompts** sent to the OpenAI-compatible and built-in
backends (the defaults are the ones every model comparison was made with; the hint
API keeps the workbench's own), the block detector and the window **scale** (everything,
readings included; `ctrl+plus` / `ctrl+minus` / `ctrl+0` change it too and the value
is remembered):

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
the preview, coloured by kind: orange for text, violet for a formula, blue for a
figure or table, grey for page furniture. Click a block or `tab`/`shift+tab` through
them to make it the region (it follows the block as you aim until you adjust it), then
drag its corners if the model's box is not quite what you want. A formula block is read
with its own prompt (asking for LaTeX; new, and not part of the model comparison the
other two prompts come from -- adjust it in Settings). **Read all
blocks** (`ctrl+enter`) reads them one after the other and lists the readings in page
order. The model predicts multi-point boxes, so on a curved or tilted page a block is a
quadrilateral, not a rectangle; such a block is perspective-rectified before it is
shown and read. About 0.1 s on the GPU, 0.3 s on the CPU. `[layout]` in `gui.toml`
turns it off, picks the device, and sets the score threshold (0.4: handwriting scores
lower than the printed pages it was trained on).

The built-in models' files (~658 MB for GLM-OCR, 130 MB for the layout model) are not
inside the binary. Each is looked for in `$SQUIGL_MODEL_DIR/<name>/`, then `models/<name>/` next to the AppImage or the
executable (how a release can ship them), then
`~/.cache/squigl/models/<name>/` (`glm-ocr-onnx-q4f16`, `pp-doclayoutv3-onnx`); if
none has it, it is downloaded there on first run from a pinned Hugging Face revision,
each file verified against a sha256 compiled into the app, with progress shown in the
window. Reads and detection are refused until the model is ready.

Keys: `space` capture/retake, `enter` read, `L` block mode on/off, `tab`/`shift+tab`
next/previous block, `ctrl+enter` read all blocks, `esc` clear the region (then
retake), `R`/`shift+R` rotate, `ctrl+S` save (to `~/Pictures/squigl/`, or `--save-dir`),
`shift+enter` second opinion, `ctrl+,` settings, `H` reading history,
`ctrl+plus`/`ctrl+minus`/`ctrl+0` window scale.
Phone zoom: the slider, the wheel over the preview, `+`/`-`, `0` to reset. The
region: drag inside it to move it, drag a corner handle to reshape it, arrow keys to
nudge (`shift` for one pixel), `[`/`]` or the wheel over the zoomed view to shrink/grow
it. `--resolution` defaults to
`max`; `--facing`, `--connect`, `--serial`, `--bitrate`, `--fps` and `--decoder`
are as for `squigl-cli`. It reconnects with backoff like the CLI.

### Wireless (TCP/IP ADB)

Once, with the phone on USB and Wi-Fi:

```
squigl-cli --tcpip            # switches adbd to TCP mode, prints e.g. 192.168.1.53:5555
```

Then unplug and stream over Wi-Fi (the `adb connect` is re-issued on every reconnect,
so a Wi-Fi hiccup is recovered like a cable wiggle):

```
squigl-cli --connect 192.168.1.53 --resolution 1920x1080
```

TCP mode persists until the phone reboots; `adb usb` switches back. 1080p at 30 Mbit/s
streams fine over a decent Wi-Fi link; drop `--bitrate` if you see stalls.

### Running as a service

`contrib/systemd/squigl.service` is a systemd *user* unit that keeps the camera exposed
whenever the phone is reachable (see the comments in the file for install steps). It
relies on the boot-time device from `contrib/install-system-config.sh` and on `squigl-cli`
being installed (`cargo install --path crates/squigl-cli [--features ffmpeg]`).

### Testing the V4L2 sink without a phone

```
./target/release/squigl-cli --test-pattern --device /dev/video10
```

Writes a synthetic cycling color-bar pattern instead of a real camera stream --
exercises the loopback/format-negotiation/write path independently of ADB/hardware.

## Troubleshooting

- **`CAMERA_IN_USE`** (server fails immediately): the phone's own camera app is open
  and holding the camera. Close it (`adb shell input keyevent KEYCODE_HOME` sends the
  phone home) and retry.
- **`adb devices` shows nothing, or `unauthorized`**: check the USB cable actually
  carries data (some are charge-only), that USB debugging is on
  (Settings -> About phone -> tap Build number 7x -> Developer options -> USB
  debugging), and accept the RSA key prompt on the phone's screen -- it only appears
  once per host and is easy to miss.
- **`/dev/videoN` never appears** even after `install-system-config.sh` and a reboot:
  `v4l2loopback` likely failed to load. `journalctl -k | grep -i v4l2loopback` (or
  `dmesg`) usually shows why -- a Secure Boot signature rejection is the most common
  cause (see Requirements above).
- **The GUI's built-in models are slow / a message about the GPU being lost appears**:
  this is the WebGPU (Vulkan) execution provider, still experimental in ONNX Runtime;
  it falls back to the CPU automatically (both on missing Vulkan drivers and mid-run
  device loss) -- expected, not a bug, see Status/caveats below for the specifics.
- **"no transcription backends configured"**: `~/.config/squigl/gui.toml` is missing
  or has no working `[[backends]]` entry; delete it to regenerate the defaults, or
  check Settings (`ctrl+,`).

## Status / caveats

- Verified end-to-end against a real device (Samsung SM-A307FN running Android 13
  via crDroid) at 1920x1080, 2992x2992 (openh264) and 4000x3000 (ffmpeg), including
  the GUI, live zoom/torch, and the built-in models on a Radeon 8060S (RADV) via WebGPU.
- The GUI's built-in models run on WebGPU (Vulkan on Linux), an execution provider ONNX
  Runtime still calls experimental; they fall back to the CPU if the provider cannot be
  set up, and if the GPU is lost mid-inference (a driver reset, `VK_ERROR_DEVICE_LOST`
  -- a whole page at too large an image budget provokes one) the read fails with a
  message and both models reload on the CPU for the rest of the session; a restart gets
  the GPU back. Images are capped at 2048 image tokens by default for that reason.
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
preprocessing code ported from oar-ocr (Apache-2.0), Typst and MiTeX (Apache-2.0) with
the fonts Typst embeds (GUST, OFL and Bitstream Vera licences).
