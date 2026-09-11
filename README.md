# phone-cam4linux

Stream an Android phone's camera to a Linux V4L2 (`/dev/videoN`) device, as a Rust
library + CLI, without shelling out to the full `scrcpy` client.

## How it works

The risky part of this problem -- driving Android's Camera2 API and H.264-encoding
the frames -- is not reimplemented. Instead, this crate embeds the real, upstream
[`scrcpy-server.jar`](https://github.com/Genymobile/scrcpy) (Apache-2.0) at build
time and pushes/runs it on the phone over ADB, exactly as `scrcpy` itself does. What
this crate implements natively in Rust is:

- the ADB plumbing to push and launch that server (via the system `adb` binary),
- scrcpy's client-side video-socket wire protocol (`src/protocol.rs`),
- H.264 decoding via `openh264` (statically linked, no system FFmpeg),
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
  `pc4l` loads it automatically via `pkexec modprobe` if the target device is
  missing; you can also load it yourself:
  ```
  sudo modprobe v4l2loopback video_nr=10 card_label="Android Cam" exclusive_caps=1
  ```

## Usage

```
cargo build --release
./target/release/pc4l --facing back --resolution 3840x2160 --bitrate 30 --device /dev/video10
```

Then point any V4L2-consuming app (browser, `ffplay`, OBS, etc.) at `/dev/video10`.

Options:

- `--facing front|back` -- which camera (default `back`).
- `--resolution WxH` -- capture size; must be one of the camera's supported sizes.
  List them with `adb shell CLASSPATH=/data/local/tmp/scrcpy-server.jar app_process / com.genymobile.scrcpy.Server <ver> list_camera_sizes=true` (after a first run has pushed the jar). Defaults to the phone's choice.
- `--bitrate MBPS` -- H.264 bitrate in Mbit/s (default `30`). Higher is crisper for
  reading text/documents.
- `--fps N` -- cap the frame rate.
- `--serial SERIAL` -- pick a device when more than one is attached.

### Document-camera use / resolution ceiling

The bundled `openh264` decoder handles up to **3840x2160** (≈8.3 MP). Larger camera
modes such as 4000x3000 (12 MP) are rejected by openh264 and will fail with a decode
error suggesting a lower resolution. For a document camera, `--resolution 3840x2160
--bitrate 30` is the sharpest supported setting; the sensor's native 4:3 sizes below
the ceiling (e.g. 1440x1080) are also available if you need that aspect ratio.

### Testing the V4L2 sink without a phone

```
./target/release/pc4l --test-pattern --device /dev/video10
```

Writes a synthetic cycling color-bar pattern instead of a real camera stream --
exercises the loopback/format-negotiation/write path independently of ADB/hardware.

## Status / caveats

- Verified end-to-end against a real device (Samsung SM-A307FN running Android 13
  via crDroid) at 1920x1080 and 3840x2160.
- **Protocol pinning**: `src/protocol.rs` implements scrcpy's undocumented
  video-socket wire format, reverse-engineered against the pinned server version in
  `build.rs` (`SCRCPY_VERSION`). Re-verify this module if you bump `SCRCPY_VERSION`.
- USB ADB only for now (no Wi-Fi/TCP ADB, ratified as v1 scope).
- No audio, display mirroring, or input control -- camera-to-V4L2 only.
- Decode ceiling is openh264's; see "resolution ceiling" above.
