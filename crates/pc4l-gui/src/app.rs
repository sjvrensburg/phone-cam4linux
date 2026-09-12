//! The window: a live preview you drag a crop rectangle on, a zoomed view of that
//! region at native pixels, and Capture/Save. Modelled on halo-workbench's hint page:
//! aiming and pressing are one glance, the crop *is* the zoom, and what is shown is
//! always the real pixel size of what arrived.
//!
//! Everything the user sees is in *view* space: the frame turned by the chosen
//! [`Rotation`]. The crop is kept in view coordinates and mapped back to the source
//! frame only to fetch pixels ([`Crop::to_source`]).
//!
//! A block detector, when there is one, turns a captured page into a list of blocks
//! in reading order; picking one makes it the crop, and a block that is not a
//! rectangle (a curved or tilted page) is rectified before it is shown or read.

use crate::layout::{self, Block, BlockDetector, Quad};
use crate::stream::{Shared, Status, Worker};
use crate::transcribe::{Mode, Transcriber, Transcription};
use egui::{
    Color32, ColorImage, FontId, Key, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};
use phone_cam4linux::convert::{i420_region_to_rgba, region_size, rotate_rgba, Rotation};
use phone_cam4linux::decode::YuvFrame;
use phone_cam4linux::Facing;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Longest preview edge we bother converting: the preview is for aiming, the crop
/// view and the capture are the real pixels.
const PREVIEW_MAX_EDGE: usize = 1920;
/// Longest edge handed to the block detector (its own input is 800 px square).
const DETECT_MAX_EDGE: usize = 1600;
/// Drags smaller than this are a click, which clears the crop.
const MIN_CROP_PX: usize = 8;

/// A rectangle in pixels, in whichever space the context says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crop {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Crop {
    fn whole(w: usize, h: usize) -> Self {
        Self { x: 0, y: 0, w, h }
    }

    fn from_corners(a: (usize, usize), b: (usize, usize)) -> Self {
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    /// Clamps to a `width`x`height` space; `None` if nothing usable is left.
    pub fn clamped(self, width: usize, height: usize) -> Option<Self> {
        let x = self.x.min(width);
        let y = self.y.min(height);
        let w = self.w.min(width - x);
        let h = self.h.min(height - y);
        (w >= MIN_CROP_PX && h >= MIN_CROP_PX).then_some(Self { x, y, w, h })
    }

    /// Shifted by (`dx`, `dy`) and kept inside a `width`x`height` space.
    fn moved(self, dx: i64, dy: i64, width: usize, height: usize) -> Self {
        let x = (self.x as i64 + dx).clamp(0, (width - self.w) as i64) as usize;
        let y = (self.y as i64 + dy).clamp(0, (height - self.h) as i64) as usize;
        Self { x, y, ..self }
    }

    /// Scaled by `factor` about its centre, clamped to the space and to
    /// [`MIN_CROP_PX`]: the digital zoom in and out.
    fn scaled(self, factor: f32, width: usize, height: usize) -> Self {
        let (cx, cy) = (
            self.x as f32 + self.w as f32 / 2.0,
            self.y as f32 + self.h as f32 / 2.0,
        );
        let w = ((self.w as f32 * factor).round() as usize).clamp(MIN_CROP_PX, width);
        let h = ((self.h as f32 * factor).round() as usize).clamp(MIN_CROP_PX, height);
        let x = ((cx - w as f32 / 2.0).round().max(0.0) as usize).min(width - w);
        let y = ((cy - h as f32 / 2.0).round().max(0.0) as usize).min(height - h);
        Self { x, y, w, h }
    }

    fn contains(self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }

    fn area(self) -> usize {
        self.w * self.h
    }

    /// Maps a rectangle in view space (the source frame turned by `rotation`, so
    /// `view_w`x`view_h` pixels) back to the source frame.
    fn to_source(self, rotation: Rotation, view_w: usize, view_h: usize) -> Self {
        let Self { x, y, w, h } = self;
        match rotation {
            Rotation::None => self,
            Rotation::Cw90 => Self {
                x: y,
                y: view_w - x - w,
                w: h,
                h: w,
            },
            Rotation::Cw180 => Self {
                x: view_w - x - w,
                y: view_h - y - h,
                w,
                h,
            },
            Rotation::Cw270 => Self {
                x: view_h - y - h,
                y: x,
                w: h,
                h: w,
            },
        }
    }
}

/// Turns smoothed scroll deltas into whole wheel notches (positive = up/away).
fn wheel_notches(accum: &mut f32, delta: f32) -> i32 {
    const NOTCH: f32 = 40.0;
    *accum += delta;
    let notches = (*accum / NOTCH).trunc();
    *accum -= notches * NOTCH;
    notches as i32
}

/// What a results list belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResultsScope {
    /// One read of this selection (`None`: the whole page).
    One(Option<Selection>),
    /// Every block, in order.
    AllBlocks,
}

/// A drag in progress on the preview.
#[derive(Debug, Clone, Copy)]
enum Drag {
    Draw {
        origin: (usize, usize),
    },
    Move {
        /// Where the drag began, and the box as it was then.
        last: (usize, usize),
        box_at_start: Crop,
    },
}

/// Fetches a view-space region of `frame` as rotated RGBA, every `step`-th pixel.
/// Returns the pixels and their size.
fn render_region(
    frame: &YuvFrame,
    rotation: Rotation,
    region: Crop,
    step: usize,
) -> (Vec<u8>, usize, usize) {
    let (vw, vh) = rotation.rotated_size(frame.width, frame.height);
    let src = region.to_source(rotation, vw, vh);
    let (sw, sh) = region_size(src.w, src.h, step);
    let mut buf = vec![0u8; sw * sh * 4];
    i420_region_to_rgba(frame, src.x, src.y, src.w, src.h, step, &mut buf);
    let (ow, oh) = rotation.rotated_size(sw, sh);
    (rotate_rgba(&buf, sw, sh, rotation), ow, oh)
}

/// What is selected on the view: a rectangle, and -- when it came from a detected
/// block that is not rectangular -- the quad inside it that is the actual block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selection {
    pub rect: Crop,
    pub quad: Option<Quad>,
}

/// The pixels of a selection (or the whole view when there is none) at every
/// `step`-th pixel, rotated as shown and, for a quad, rectified. Returns the pixels
/// and their size.
fn render_selection(
    frame: &YuvFrame,
    rotation: Rotation,
    selection: Option<Selection>,
    step: usize,
) -> (Vec<u8>, usize, usize) {
    let (vw, vh) = rotation.rotated_size(frame.width, frame.height);
    let Some(sel) = selection else {
        return render_region(frame, rotation, Crop::whole(vw, vh), step);
    };
    let (rgba, w, h) = render_region(frame, rotation, sel.rect, step);
    let Some(quad) = sel.quad else {
        return (rgba, w, h);
    };
    // The quad in the rendered region's own pixels.
    let local: Quad = quad.map(|[x, y]| {
        [
            (x - sel.rect.x as f32) / step as f32,
            (y - sel.rect.y as f32) / step as f32,
        ]
    });
    let img = image::RgbaImage::from_raw(w as u32, h as u32, rgba).expect("buffer matches size");
    match layout::rectify(&img, &local) {
        Some(out) => {
            let (ow, oh) = (out.width() as usize, out.height() as usize);
            (out.into_raw(), ow, oh)
        }
        None => {
            let (w, h) = (img.width() as usize, img.height() as usize);
            (img.into_raw(), w, h)
        }
    }
}

/// A texture cached against the frame, region, step and rotation it was made from, so
/// a repaint with nothing new (a hover) costs no conversion or upload. Holds the
/// frame: comparing a bare pointer would misfire when the allocator hands a new
/// frame the old one's address.
struct View {
    texture: Option<TextureHandle>,
    name: &'static str,
    /// ... and the size of the texture it produced.
    key: Option<ViewKey>,
}

type ViewKey = (
    Arc<YuvFrame>,
    Option<Selection>,
    usize,
    Rotation,
    (usize, usize),
);

/// A read in flight: its result channel, when it started, which backend, and the
/// block label if it is one of a "read all".
type PendingRead = (
    Receiver<anyhow::Result<Transcription>>,
    Instant,
    String,
    Option<String>,
);

impl View {
    fn new(name: &'static str) -> Self {
        Self {
            texture: None,
            name,
            key: None,
        }
    }

    /// `selection` is `None` for the whole view.
    fn update(
        &mut self,
        ctx: &egui::Context,
        frame: &Arc<YuvFrame>,
        selection: Option<Selection>,
        step: usize,
        rotation: Rotation,
    ) -> (usize, usize) {
        if let Some((f, r, s, rot, size)) = &self.key {
            if Arc::ptr_eq(f, frame) && *r == selection && *s == step && *rot == rotation {
                return *size;
            }
        }
        let (rgba, w, h) = render_selection(frame, rotation, selection, step);
        let image = ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        match &mut self.texture {
            Some(t) => t.set(image, TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture(self.name, image, TextureOptions::LINEAR)),
        }
        self.key = Some((Arc::clone(frame), selection, step, rotation, (w, h)));
        (w, h)
    }
}

pub struct App {
    worker: Worker,
    preview: View,
    crop_view: View,
    /// A frozen frame; while set, it is shown instead of the live one.
    captured: Option<Arc<YuvFrame>>,
    rotation: Rotation,
    /// In view space.
    crop: Option<Crop>,
    /// The crop's quad when it came from a non-rectangular block; cleared by any
    /// hand edit of the crop.
    quad: Option<Quad>,
    drag: Option<Drag>,
    /// Scroll accumulators (egui smooths wheel input over frames).
    wheel_preview: f32,
    wheel_crop: f32,
    save_dir: PathBuf,
    message: Option<(String, Instant)>,
    fps: FpsCounter,
    /// View size the crop was drawn against; a different frame size (camera switch)
    /// invalidates it.
    crop_space: Option<(usize, usize)>,
    backends: Vec<Arc<dyn Transcriber>>,
    selected_backend: usize,
    /// For waking the UI from the read thread; set on the first frame.
    ctx: Option<egui::Context>,
    pending: Option<PendingRead>,
    /// Readings for the current capture, oldest first, each with its block label.
    results: Vec<(Option<String>, Result<Transcription, String>)>,
    /// What `results` were read from; they are dropped when a read of something
    /// else starts or the capture goes.
    results_key: Option<(Arc<YuvFrame>, ResultsScope)>,
    detector: Option<Arc<dyn BlockDetector>>,
    /// Detected blocks of the captured frame, in reading order, in view space.
    blocks: Vec<Block>,
    /// The frame and rotation `blocks` were found on; they go when it changes.
    blocks_key: Option<(Arc<YuvFrame>, Rotation)>,
    /// Which block the crop is, for tab to move on from.
    selected_block: Option<usize>,
    pending_detect: Option<(Receiver<anyhow::Result<Vec<Block>>>, Instant)>,
    /// Blocks still to read, for "read all".
    read_queue: VecDeque<usize>,
    /// Development aid: read once, as soon as a frame is available.
    dev_read: bool,
    /// Development aid: detect blocks once, as soon as the detector is ready, and
    /// then (second flag) read them all.
    dev_detect: (bool, bool),
    /// Development aid: a zoom to apply live, and when the first frame was seen.
    dev_zoom: Option<(f32, Option<Instant>)>,
    /// Development aid: write a screenshot of the window to this path after the
    /// delay, then quit.
    screenshot: Option<(Duration, PathBuf, Instant)>,
}

impl App {
    pub fn new(
        worker: Worker,
        save_dir: PathBuf,
        backends: Vec<Arc<dyn Transcriber>>,
        detector: Option<Arc<dyn BlockDetector>>,
        screenshot: Option<(Duration, PathBuf)>,
    ) -> Self {
        Self {
            worker,
            preview: View::new("preview"),
            crop_view: View::new("crop"),
            captured: None,
            rotation: Rotation::None,
            crop: None,
            quad: None,
            drag: None,
            wheel_preview: 0.0,
            wheel_crop: 0.0,
            save_dir,
            message: None,
            fps: FpsCounter::default(),
            crop_space: None,
            backends,
            selected_backend: 0,
            ctx: None,
            pending: None,
            results: Vec::new(),
            results_key: None,
            detector,
            blocks: Vec::new(),
            blocks_key: None,
            selected_block: None,
            pending_detect: None,
            read_queue: VecDeque::new(),
            dev_read: false,
            dev_detect: (false, false),
            dev_zoom: None,
            screenshot: screenshot.map(|(after, path)| (after, path, Instant::now())),
        }
    }

    pub fn set_dev_zoom(&mut self, zoom: Option<f32>) {
        self.dev_zoom = zoom.map(|z| (z, None));
    }

    pub fn set_dev_read(&mut self, on: bool) {
        self.dev_read = on;
    }

    pub fn set_dev_detect(&mut self, detect: bool, read_all: bool) {
        self.dev_detect = (detect, read_all);
    }

    pub fn set_rotation(&mut self, rotation: Rotation) {
        self.rotate(rotation);
    }

    /// Sets the crop in view space; it is clamped to the frame when first drawn.
    pub fn set_crop(&mut self, crop: Option<Crop>) {
        self.set_rect(crop);
    }

    /// A hand edit of the crop: any quad it carried no longer applies.
    fn set_rect(&mut self, crop: Option<Crop>) {
        self.crop = crop;
        self.quad = None;
        self.selected_block = None;
    }

    fn selection(&self) -> Option<Selection> {
        self.crop.map(|rect| Selection {
            rect,
            quad: self.quad,
        })
    }

    /// Makes block `i` the crop.
    fn select_block(&mut self, i: usize) {
        let Some(b) = self.blocks.get(i) else {
            return;
        };
        self.crop = Some(b.rect);
        self.quad = b.quad;
        self.selected_block = Some(i);
    }

    /// Tab: the next (or previous) block in reading order.
    fn step_block(&mut self, delta: i32) {
        if self.blocks.is_empty() {
            self.say("no blocks: detect them first [L]");
            return;
        }
        let n = self.blocks.len() as i32;
        let next = match self.selected_block {
            Some(i) => (i as i32 + delta).rem_euclid(n),
            None if delta < 0 => n - 1,
            None => 0,
        };
        self.select_block(next as usize);
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.worker.shared
    }

    /// The frame on screen: the capture if there is one, else the newest live frame.
    fn current_frame(&self) -> Option<Arc<YuvFrame>> {
        self.captured.clone().or_else(|| self.shared().latest())
    }

    fn say(&mut self, text: impl Into<String>) {
        self.message = Some((text.into(), Instant::now()));
    }

    fn capture(&mut self) {
        if self.captured.is_some() {
            self.captured = None;
            self.say("live again");
        } else if let Some(frame) = self.shared().latest() {
            self.say(format!("captured {}x{}", frame.width, frame.height));
            self.captured = Some(frame);
        }
    }

    /// Sends the crop (or the whole view) to the selected backend on a thread.
    /// Reading a live frame freezes it first, so the answer stays next to its ink.
    fn read(&mut self) {
        self.read_queue.clear();
        let selection = self.selection();
        self.read_selection(selection, ResultsScope::One(selection), None);
    }

    /// Reads every detected block in reading order, one after the other.
    fn read_all(&mut self) {
        if self.blocks.is_empty() {
            self.say("no blocks: detect them first [L]");
            return;
        }
        self.read_queue = (0..self.blocks.len()).collect();
        self.set_results_key(ResultsScope::AllBlocks);
        self.results.clear();
        self.next_queued_read();
    }

    /// Starts the next block of a "read all" once the previous one has landed.
    fn next_queued_read(&mut self) {
        if self.pending.is_some() {
            return;
        }
        while let Some(i) = self.read_queue.pop_front() {
            let Some(b) = self.blocks.get(i) else {
                continue;
            };
            let selection = Some(Selection {
                rect: b.rect,
                quad: b.quad,
            });
            let label = format!("#{} {}", i + 1, b.label);
            self.select_block(i);
            self.read_selection(selection, ResultsScope::AllBlocks, Some(label));
            return;
        }
    }

    fn read_selection(
        &mut self,
        selection: Option<Selection>,
        scope: ResultsScope,
        label: Option<String>,
    ) {
        if self.pending.is_some() {
            self.say("still reading the last one");
            return;
        }
        let Some(backend) = self.backends.get(self.selected_backend).cloned() else {
            self.say("no transcription backends configured (see ~/.config/pc4l/gui.toml)");
            return;
        };
        if self.captured.is_none() {
            self.capture();
        }
        let Some(frame) = self.captured.clone() else {
            self.say("nothing to read yet");
            return;
        };
        // Results belong to this frame and selection; a read of something else
        // starts a fresh list.
        self.set_results_key(scope);
        let (vw, vh) = self.rotation.rotated_size(frame.width, frame.height);
        let mode = if selection.is_some() {
            Mode::Crop
        } else {
            Mode::Page
        };
        let (rgba, w, h) = render_selection(&frame, self.rotation, selection, 1);
        let mut png = Vec::new();
        let encoded = image::RgbaImage::from_raw(w as u32, h as u32, rgba)
            .expect("buffer matches size")
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png);
        if let Err(e) = encoded {
            self.say(format!("encoding the crop failed: {e}"));
            return;
        }
        let (tx, rx) = mpsc::sync_channel(1);
        let ctx = self.ctx.clone();
        let name = backend.name().to_string();
        std::thread::Builder::new()
            .name("pc4l-read".into())
            .spawn(move || {
                let result = backend.read(&png, mode, (vw as u32, vh as u32));
                let _ = tx.send(result);
                if let Some(ctx) = ctx {
                    ctx.request_repaint();
                }
            })
            .expect("spawning read thread");
        self.pending = Some((rx, Instant::now(), name, label));
    }

    /// Readings are kept while they are of the captured frame and `scope`; anything
    /// else starts a fresh list.
    fn set_results_key(&mut self, scope: ResultsScope) {
        let key = self.captured.clone().map(|f| (f, scope));
        let same = match (&self.results_key, &key) {
            (Some((a, sa)), Some((b, sb))) => Arc::ptr_eq(a, b) && sa == sb,
            (None, None) => true,
            _ => false,
        };
        if !same {
            self.results.clear();
            self.results_key = key;
        }
    }

    /// Readings go with the capture they were made from.
    fn drop_stale_results(&mut self) {
        let stale = match (&self.results_key, &self.captured) {
            (Some((a, _)), Some(b)) => !Arc::ptr_eq(a, b),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if stale {
            self.results.clear();
            self.results_key = None;
            self.read_queue.clear();
        }
    }

    /// Collects a finished read, and a finished detection.
    fn poll_read(&mut self) {
        self.drop_stale_results();
        if let Some((rx, _, _, label)) = &self.pending {
            let done = match rx.try_recv() {
                Ok(result) => Some(result.map_err(|e| format!("{e:#}"))),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(Err("the read thread died".into())),
            };
            if let Some(result) = done {
                let label = label.clone();
                self.results.push((label, result));
                self.pending = None;
                self.next_queued_read();
            }
        }
        if let Some((rx, started)) = &self.pending_detect {
            let done = match rx.try_recv() {
                Ok(result) => Some(result),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    Some(Err(anyhow::anyhow!("the detection thread died")))
                }
            };
            if let Some(result) = done {
                let elapsed = started.elapsed();
                self.pending_detect = None;
                match result {
                    Ok(blocks) => {
                        self.say(format!(
                            "{} block{} in {:.2}s",
                            blocks.len(),
                            if blocks.len() == 1 { "" } else { "s" },
                            elapsed.as_secs_f32()
                        ));
                        self.blocks = blocks;
                        self.selected_block = None;
                    }
                    Err(e) => self.say(format!("block detection failed: {e:#}")),
                }
            }
        }
    }

    /// Runs the block detector over the captured frame (capturing first if live),
    /// on a thread. The result lands in `blocks` in view space.
    fn detect(&mut self) {
        let Some(detector) = self.detector.clone() else {
            self.say("no block detector in this build");
            return;
        };
        if self.pending_detect.is_some() {
            self.say("still detecting");
            return;
        }
        if !detector.ready() {
            self.say(
                detector
                    .status()
                    .unwrap_or_else(|| "block detector not ready".into()),
            );
            return;
        }
        if self.captured.is_none() {
            self.capture();
        }
        let Some(frame) = self.captured.clone() else {
            self.say("nothing to detect on yet");
            return;
        };
        let rotation = self.rotation;
        let (vw, vh) = rotation.rotated_size(frame.width, frame.height);
        let step = vw.max(vh).div_ceil(DETECT_MAX_EDGE).max(1);
        let (rgba, w, h) = render_region(&frame, rotation, Crop::whole(vw, vh), step);
        let img = layout::rgba_to_rgb(&rgba, w, h);
        self.blocks_key = Some((Arc::clone(&frame), rotation));
        let (tx, rx) = mpsc::sync_channel(1);
        let ctx = self.ctx.clone();
        std::thread::Builder::new()
            .name("pc4l-detect".into())
            .spawn(move || {
                // Back from the decimated image to view pixels.
                let result = detector.detect(&img).map(|blocks| {
                    blocks
                        .into_iter()
                        .filter_map(|mut b| {
                            b.rect = Crop {
                                x: b.rect.x * step,
                                y: b.rect.y * step,
                                w: b.rect.w * step,
                                h: b.rect.h * step,
                            }
                            .clamped(vw, vh)?;
                            b.quad = b
                                .quad
                                .map(|q| q.map(|[x, y]| [x * step as f32, y * step as f32]));
                            Some(b)
                        })
                        .collect()
                });
                let _ = tx.send(result);
                if let Some(ctx) = ctx {
                    ctx.request_repaint();
                }
            })
            .expect("spawning detection thread");
        self.pending_detect = Some((rx, Instant::now()));
    }

    /// Blocks belong to the frame and rotation they were found on.
    fn drop_stale_blocks(&mut self) {
        let stale = match (&self.blocks_key, &self.captured) {
            (Some((f, r)), Some(c)) => !Arc::ptr_eq(f, c) || *r != self.rotation,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if stale {
            self.blocks.clear();
            self.blocks_key = None;
            self.selected_block = None;
            self.read_queue.clear();
        }
    }

    /// Whether the selected backend can take a read now (a local model may still
    /// be downloading or loading).
    fn backend_ready(&self) -> bool {
        self.backends
            .get(self.selected_backend)
            .and_then(|b| b.status())
            .is_none_or(|s| s.starts_with("ready") || s.starts_with("unavailable"))
    }

    /// Any change to the camera itself (zoom) makes a frozen capture stale: drop it
    /// so the preview shows what the phone now sees.
    fn go_live(&mut self) {
        if self.captured.is_some() {
            self.captured = None;
            self.say("live again (zoom changed)");
        }
    }

    fn rotate(&mut self, rotation: Rotation) {
        if rotation != self.rotation {
            self.rotation = rotation;
            // The crop is in view space; rather than spin it, start over.
            self.set_rect(None);
        }
    }

    /// Writes the crop (or the whole view) at native resolution, rotated as shown.
    fn save(&mut self) {
        let Some(frame) = self.current_frame() else {
            self.say("nothing to save yet");
            return;
        };
        let (rgba, w, h) = render_selection(&frame, self.rotation, self.selection(), 1);
        let path = self.save_dir.join(format!(
            "pc4l-{}.png",
            chrono::Local::now().format("%Y%m%d-%H%M%S")
        ));
        let result = std::fs::create_dir_all(&self.save_dir).and_then(|()| {
            image::save_buffer(&path, &rgba, w as u32, h as u32, image::ColorType::Rgba8)
                .map_err(std::io::Error::other)
        });
        match result {
            Ok(()) => self.say(format!("saved {w}x{h} to {}", path.display())),
            Err(e) => self.say(format!("save failed: {e}")),
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let live = self.captured.is_none();
            let label = if live {
                "Capture  [space]"
            } else {
                "Retake  [space]"
            };
            if ui.button(label).clicked() {
                self.capture();
            }
            if ui.button("Save PNG  [ctrl+S]").clicked() {
                self.save();
            }
            ui.add_enabled_ui(self.crop.is_some(), |ui| {
                if ui.button("Clear crop  [esc]").clicked() {
                    self.set_rect(None);
                }
            });
            if let Some(detector) = &self.detector {
                let ready = detector.ready() && self.pending_detect.is_none();
                let button = ui.add_enabled(ready, egui::Button::new("Blocks  [L]"));
                let button = match detector.status() {
                    Some(status) => button.on_disabled_hover_text(status),
                    None => button
                        .on_hover_text("find the page's blocks; click one or tab through them"),
                };
                if button.clicked() {
                    self.detect();
                }
            }
            ui.separator();
            if ui.button("Rotate left  [shift+R]").clicked() {
                self.rotate(self.rotation.turned_ccw());
            }
            if ui.button("Rotate right  [R]").clicked() {
                self.rotate(self.rotation.turned_cw());
            }
            ui.separator();

            let mut facing = self.shared().facing();
            let before = facing;
            ui.label("Camera:");
            ui.selectable_value(&mut facing, Facing::Back, "back");
            ui.selectable_value(&mut facing, Facing::Front, "front");
            if facing != before {
                self.shared().set_facing(facing);
            }
            if ui.button("Reconnect").clicked() {
                self.shared().restart();
            }
            self.zoom_control(ui);
        });
    }

    /// The phone's own zoom and torch, when the phone reports a zoom range for this
    /// camera. Both apply live over the control channel.
    fn zoom_control(&mut self, ui: &mut egui::Ui) {
        let Some((lo, hi)) = self.shared().camera().and_then(|c| c.zoom_range) else {
            return;
        };
        ui.separator();
        if hi > lo {
            ui.label("Zoom:");
            let mut value = self.shared().zoom();
            let slider = ui.add(
                egui::Slider::new(&mut value, lo.max(1.0)..=hi)
                    .logarithmic(true)
                    .suffix("x")
                    .fixed_decimals(2),
            );
            if slider.changed() {
                self.go_live();
                self.shared().set_zoom(value);
            }
            if ui
                .button("1x  [0]")
                .on_hover_text("reset the phone's zoom")
                .clicked()
            {
                self.go_live();
                self.shared().set_zoom(1.0);
            }
        }
        let mut torch = self.shared().torch();
        if ui.checkbox(&mut torch, "Torch").changed() {
            self.shared().set_torch(torch);
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let status = self.shared().status();
            let text = match &status {
                Status::Connecting => "connecting to phone…".to_string(),
                Status::Streaming { width, height } => {
                    format!("streaming {width}x{height} at {:.0} fps", self.fps.rate)
                }
                Status::Waiting { reason, retry_at } => format!(
                    "{reason} — retrying in {}s",
                    retry_at.saturating_duration_since(Instant::now()).as_secs() + 1
                ),
                Status::Stopped => "stopped".to_string(),
            };
            let colour = match status {
                Status::Waiting { .. } => ui.visuals().warn_fg_color,
                _ => ui.visuals().text_color(),
            };
            ui.colored_label(colour, text);
            if self.rotation != Rotation::None {
                ui.separator();
                ui.label(format!("rotated {:?}", self.rotation));
            }
            let (want, have) = (self.shared().zoom(), self.shared().zoom_applied());
            if want > 1.0 || have > 1.0 {
                ui.separator();
                if (want - have).abs() > 0.01 {
                    ui.label(format!("zoom {have:.2}x → {want:.2}x"));
                } else {
                    ui.label(format!("zoom {have:.2}x"));
                }
            }
            if self.captured.is_some() {
                ui.separator();
                ui.strong("CAPTURED");
                ui.weak("preview frozen — space or esc goes back to live");
            }
            if self.pending_detect.is_some() {
                ui.separator();
                ui.spinner();
                ui.label("detecting blocks");
            } else if !self.blocks.is_empty() {
                ui.separator();
                ui.label(format!(
                    "{} blocks — click one, tab through them, ctrl+enter reads all",
                    self.blocks.len()
                ));
            } else if let Some(status) = self.detector.as_ref().and_then(|d| d.status()) {
                ui.separator();
                ui.weak(status);
            }
            if let Some((msg, at)) = &self.message {
                if at.elapsed().as_secs() < 8 {
                    ui.separator();
                    ui.label(msg);
                }
            }
        });
    }

    /// The live/captured image, fitted to the panel, with the crop rectangle drawn on
    /// it and drag-to-select. Coordinates here are view space.
    fn preview_panel(&mut self, ui: &mut egui::Ui, frame: &Arc<YuvFrame>) {
        let (vw, vh) = self.rotation.rotated_size(frame.width, frame.height);
        let step = vw.max(vh).div_ceil(PREVIEW_MAX_EDGE).max(1);
        self.preview
            .update(ui.ctx(), frame, None, step, self.rotation);
        let Some(texture) = &self.preview.texture else {
            return;
        };

        let avail = ui.available_size();
        let scale = (avail.x / vw as f32).min(avail.y / vh as f32);
        let size = Vec2::new(vw as f32 * scale, vh as f32 * scale);
        let response = ui
            .centered_and_justified(|ui| {
                ui.add(
                    egui::Image::from_texture(texture)
                        .fit_to_exact_size(size)
                        .sense(Sense::drag()),
                )
            })
            .inner;
        // The image is centred inside the response rect; map pointer positions
        // against where the pixels actually are.
        let image_rect = Rect::from_center_size(response.rect.center(), size);
        let to_view = |p: Pos2| -> (usize, usize) {
            let x = ((p.x - image_rect.min.x) / scale)
                .round()
                .clamp(0.0, vw as f32) as usize;
            let y = ((p.y - image_rect.min.y) / scale)
                .round()
                .clamp(0.0, vh as f32) as usize;
            (x, y)
        };

        if response.drag_started() {
            if let Some(start) = response.interact_pointer_pos().map(to_view) {
                // Inside the box: pan it. Outside: draw a new one.
                self.drag = Some(match self.crop {
                    Some(c) if c.contains(start.0, start.1) => Drag::Move {
                        last: start,
                        box_at_start: c,
                    },
                    _ => Drag::Draw { origin: start },
                });
            }
        }
        if let (Some(drag), Some(pos)) = (self.drag, response.interact_pointer_pos()) {
            if response.dragged() || response.drag_stopped() {
                let here = to_view(pos);
                match drag {
                    Drag::Draw { origin } => {
                        let drawn = Crop::from_corners(origin, here).clamped(vw, vh);
                        // A click (too small to be a box) on a detected block
                        // selects it; anywhere else it clears the crop.
                        match (drawn, response.drag_stopped()) {
                            (None, true) => match self.block_at(origin) {
                                Some(i) => self.select_block(i),
                                None => self.set_rect(None),
                            },
                            _ => self.set_rect(drawn),
                        }
                    }
                    Drag::Move { last, box_at_start } => {
                        let (dx, dy) =
                            (here.0 as i64 - last.0 as i64, here.1 as i64 - last.1 as i64);
                        self.set_rect(Some(box_at_start.moved(dx, dy, vw, vh)));
                    }
                }
            }
        }
        if response.drag_stopped() {
            self.drag = None;
        }
        // Wheel over the preview: the phone's zoom, one grid step per notch.
        if response.hovered() {
            let delta = ui.input(|i| i.smooth_scroll_delta.y);
            let notches = wheel_notches(&mut self.wheel_preview, delta);
            if notches != 0 {
                self.go_live();
                self.shared().step_zoom(notches);
            }
        }

        let to_screen =
            |x: f32, y: f32| Pos2::new(image_rect.min.x + x * scale, image_rect.min.y + y * scale);
        let painter = ui.painter_at(image_rect);
        // Detected blocks: their quads, numbered in reading order.
        for (i, b) in self.blocks.iter().enumerate() {
            let selected = self.selected_block == Some(i);
            let quad = b.quad.unwrap_or_else(|| layout::rect_quad(b.rect));
            let points: Vec<Pos2> = quad.iter().map(|[x, y]| to_screen(*x, *y)).collect();
            let colour = if selected {
                Color32::from_rgb(255, 196, 0)
            } else {
                Color32::from_rgb(80, 220, 120)
            };
            painter.add(Shape::closed_line(
                points.clone(),
                Stroke::new(if selected { 2.5 } else { 1.5 }, colour),
            ));
            let tag = format!("{}", i + 1);
            let font = FontId::proportional(13.0);
            let galley = painter.layout_no_wrap(tag, font, Color32::BLACK);
            let at = points[0];
            let bg = Rect::from_min_size(at, galley.size() + Vec2::splat(4.0));
            painter.rect_filled(bg, 2.0, colour);
            painter.galley(at + Vec2::splat(2.0), galley, Color32::BLACK);
        }

        if let Some(crop) = self.crop {
            let rect = Rect::from_min_max(
                to_screen(crop.x as f32, crop.y as f32),
                to_screen((crop.x + crop.w) as f32, (crop.y + crop.h) as f32),
            );
            let dim = Color32::from_black_alpha(110);
            // Dim everything outside the crop so the selection reads at a glance.
            for outside in [
                Rect::from_min_max(image_rect.min, Pos2::new(image_rect.max.x, rect.min.y)),
                Rect::from_min_max(Pos2::new(image_rect.min.x, rect.max.y), image_rect.max),
                Rect::from_min_max(
                    Pos2::new(image_rect.min.x, rect.min.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ),
                Rect::from_min_max(
                    Pos2::new(rect.max.x, rect.min.y),
                    Pos2::new(image_rect.max.x, rect.max.y),
                ),
            ] {
                painter.rect_filled(outside, 0.0, dim);
            }
            painter.rect_stroke(
                rect,
                0.0,
                Stroke::new(2.0, Color32::from_rgb(255, 196, 0)),
                StrokeKind::Outside,
            );
        }
    }

    /// The smallest detected block under a view point, if any.
    fn block_at(&self, (x, y): (usize, usize)) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.rect.contains(x, y))
            .min_by_key(|(_, b)| b.rect.area())
            .map(|(i, _)| i)
    }

    /// The crop at native pixels (decimated only if it is wider than the preview
    /// budget), scaled to the panel: this is the zoom.
    fn crop_panel(&mut self, ui: &mut egui::Ui, frame: &Arc<YuvFrame>) {
        // The zoomed region on top, the reading controls and results below it.
        let read_height = (ui.available_height() * 0.45).max(160.0);
        egui::Panel::bottom("read")
            .resizable(true)
            .default_size(read_height)
            .show(ui, |ui| self.read_section(ui));
        egui::CentralPanel::default().show(ui, |ui| self.crop_image(ui, frame));
    }

    fn crop_image(&mut self, ui: &mut egui::Ui, frame: &Arc<YuvFrame>) {
        let Some(selection) = self.selection() else {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label("Drag a box on the preview to zoom to a region.");
                ui.label(
                    "Drag inside the box or use the arrow keys to move it; [ and ] or the \
                     wheel over this panel resize it. The wheel over the preview, + / - \
                     and 0 drive the phone's own zoom.",
                );
                ui.label(
                    "Capture freezes the frame; Save writes the box (or the whole frame) \
                     as a PNG at full resolution; Read it sends it to the model.",
                );
                if self.detector.is_some() {
                    ui.label(
                        "Blocks [L] finds the page's text blocks in reading order: click \
                         one or tab through them to make it the box; ctrl+enter reads \
                         them all.",
                    );
                }
            });
            return;
        };
        let crop = selection.rect;
        let step = crop.w.max(crop.h).div_ceil(PREVIEW_MAX_EDGE).max(1);
        let (tw, th) = self
            .crop_view
            .update(ui.ctx(), frame, Some(selection), step, self.rotation);
        let Some(texture) = &self.crop_view.texture else {
            return;
        };
        // Native size of what is shown (a rectified quad is its own size).
        let (nw, nh) = (tw * step, th * step);
        let block = self
            .selected_block
            .and_then(|i| self.blocks.get(i))
            .map(|b| {
                format!(
                    " — block {} ({})",
                    self.selected_block.unwrap() + 1,
                    b.label
                )
            })
            .unwrap_or_default();
        ui.label(format!(
            "{nw}×{nh} px at ({}, {}){}{}{}",
            crop.x,
            crop.y,
            if selection.quad.is_some() {
                ", rectified"
            } else {
                ""
            },
            if step > 1 {
                format!(", shown at 1/{step}")
            } else {
                String::new()
            },
            block
        ));
        let avail = ui.available_size();
        let scale = (avail.x / nw as f32).min(avail.y / nh as f32);
        let size = Vec2::new(nw as f32 * scale, nh as f32 * scale);
        let response = ui
            .centered_and_justified(|ui| {
                ui.add(
                    egui::Image::from_texture(texture)
                        .fit_to_exact_size(size)
                        .sense(Sense::hover()),
                )
            })
            .inner;
        // Wheel over the zoomed view: grow or shrink the box about its centre.
        if response.hovered() {
            let delta = ui.input(|i| i.smooth_scroll_delta.y);
            let notches = wheel_notches(&mut self.wheel_crop, delta);
            if notches != 0 {
                let (vw, vh) = self.rotation.rotated_size(frame.width, frame.height);
                let factor = 1.1f32.powi(-notches);
                self.set_rect(Some(crop.scaled(factor, vw, vh)));
            }
        }
    }

    /// Backend picker, the Read button, and the readings so far.
    fn read_section(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let what = if self.crop.is_some() {
                "Read the box  [enter]"
            } else {
                "Read the page  [enter]"
            };
            ui.add_enabled_ui(self.pending.is_none() && !self.backends.is_empty(), |ui| {
                if ui.button(what).clicked() {
                    self.read();
                }
                if !self.blocks.is_empty() && ui.button("Read all blocks  [ctrl+enter]").clicked() {
                    self.read_all();
                }
            });
            let current = self
                .backends
                .get(self.selected_backend)
                .map(|b| b.name().to_string())
                .unwrap_or_else(|| "no backends".into());
            egui::ComboBox::from_id_salt("backend")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (i, b) in self.backends.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_backend, i, b.name());
                    }
                });
            if let Some((_, started, name, label)) = &self.pending {
                ui.spinner();
                let what = label.as_deref().unwrap_or("");
                ui.label(format!(
                    "{name}: {what} {:.0}s",
                    started.elapsed().as_secs_f32()
                ));
                if !self.read_queue.is_empty() {
                    ui.weak(format!("{} to go", self.read_queue.len()));
                }
            } else if let Some(status) = self
                .backends
                .get(self.selected_backend)
                .and_then(|b| b.status())
            {
                ui.weak(status);
            }
            if !self.results.is_empty() && ui.small_button("clear").clicked() {
                self.results.clear();
            }
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // One read at a time: newest on top. A "read all": in page order.
                let in_order = matches!(self.results_key, Some((_, ResultsScope::AllBlocks)));
                let mut ordered: Vec<_> = self.results.iter().collect();
                if !in_order {
                    ordered.reverse();
                }
                for (label, result) in ordered {
                    if let Some(label) = label {
                        ui.strong(label);
                    }
                    match result {
                        Ok(t) => show_transcription(ui, t),
                        Err(e) => {
                            ui.colored_label(ui.visuals().error_fg_color, e);
                        }
                    }
                    ui.separator();
                }
            });
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let (space, esc, save, rot_cw, rot_ccw, enter, read_all) = ctx.input(|i| {
            (
                i.key_pressed(Key::Space),
                i.key_pressed(Key::Escape),
                i.modifiers.command && i.key_pressed(Key::S),
                !i.modifiers.shift && i.key_pressed(Key::R),
                i.modifiers.shift && i.key_pressed(Key::R),
                !i.modifiers.command && i.key_pressed(Key::Enter),
                i.modifiers.command && i.key_pressed(Key::Enter),
            )
        });
        // Blocks: L detects, tab / shift+tab walk them.
        let (detect, tab) = ctx.input(|i| {
            (
                i.key_pressed(Key::L),
                if i.key_pressed(Key::Tab) {
                    if i.modifiers.shift {
                        -1
                    } else {
                        1
                    }
                } else {
                    0
                },
            )
        });
        if detect {
            self.detect();
        }
        if tab != 0 {
            self.step_block(tab);
        }
        // Phone zoom: + / - step, 0 resets.
        let (zoom_in, zoom_out, zoom_reset) = ctx.input(|i| {
            (
                i.key_pressed(Key::Plus) || i.key_pressed(Key::Equals),
                i.key_pressed(Key::Minus),
                i.key_pressed(Key::Num0),
            )
        });
        if zoom_in || zoom_out || zoom_reset {
            self.go_live();
        }
        if zoom_in {
            self.shared().step_zoom(2);
        }
        if zoom_out {
            self.shared().step_zoom(-2);
        }
        if zoom_reset {
            self.shared().set_zoom(1.0);
        }
        // Digital box: [ / ] shrink and grow, arrows pan (shift: finer).
        if let (Some(crop), Some((vw, vh))) = (self.crop, self.crop_space) {
            let (grow, shrink, dx, dy, fine) = ctx.input(|i| {
                (
                    i.key_pressed(Key::CloseBracket),
                    i.key_pressed(Key::OpenBracket),
                    i32::from(i.key_pressed(Key::ArrowRight))
                        - i32::from(i.key_pressed(Key::ArrowLeft)),
                    i32::from(i.key_pressed(Key::ArrowDown))
                        - i32::from(i.key_pressed(Key::ArrowUp)),
                    i.modifiers.shift,
                )
            });
            if grow {
                self.set_rect(Some(crop.scaled(1.25, vw, vh)));
            }
            if shrink {
                self.set_rect(Some(crop.scaled(0.8, vw, vh)));
            }
            if dx != 0 || dy != 0 {
                let unit = if fine {
                    1
                } else {
                    (crop.w.min(crop.h) / 5).max(1)
                } as i64;
                self.set_rect(Some(crop.moved(dx as i64 * unit, dy as i64 * unit, vw, vh)));
            }
        }
        if space {
            self.capture();
        }
        if enter {
            self.read();
        }
        if read_all {
            self.read_all();
        }
        if esc {
            // Back out one level: the box first, then the capture.
            if self.crop.is_some() {
                self.set_rect(None);
            } else if self.captured.is_some() {
                self.capture();
            }
        }
        if save {
            self.save();
        }
        if rot_cw {
            self.rotate(self.rotation.turned_cw());
        }
        if rot_ccw {
            self.rotate(self.rotation.turned_ccw());
        }
    }

    /// `--screenshot-after`: request the capture once the delay has passed, write
    /// it when it arrives, and close.
    fn handle_screenshot(&mut self, ctx: &egui::Context) {
        let Some((after, path, started)) = &self.screenshot else {
            return;
        };
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(Arc::clone(image)),
                _ => None,
            })
        });
        if let Some(image) = shot {
            let [w, h] = image.size;
            match image::save_buffer(
                path,
                image.as_raw(),
                w as u32,
                h as u32,
                image::ColorType::Rgba8,
            ) {
                Ok(()) => log::info!("screenshot written to {}", path.display()),
                Err(e) => log::error!("screenshot: {e}"),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if started.elapsed() >= *after {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.ctx.is_none() {
            self.ctx = Some(ui.ctx().clone());
        }
        self.fps.tick(self.shared().frames());
        self.handle_keys(ui.ctx());
        self.handle_screenshot(ui.ctx());
        self.drop_stale_blocks();
        self.poll_read();
        if self.pending.is_some() || self.pending_detect.is_some() {
            // Keep the elapsed counter moving even when no frames arrive.
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        let frame = self.current_frame();
        let Some(frame) = frame else {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| ui.label("waiting for the first frame…"));
            });
            return;
        };

        // The crop must fit the frame about to be drawn: frames change size when the
        // camera is switched, and a stale box would index outside the new frame.
        let view = self.rotation.rotated_size(frame.width, frame.height);
        if self.crop_space != Some(view) {
            if self.crop_space.is_some() {
                self.set_rect(None);
            }
            self.crop_space = Some(view);
        }
        let clamped = self.crop.and_then(|c| c.clamped(view.0, view.1));
        if clamped != self.crop {
            self.set_rect(clamped);
        }

        if let Some((zoom, seen)) = &mut self.dev_zoom {
            match seen {
                None => *seen = Some(Instant::now()),
                Some(at) if at.elapsed() > Duration::from_secs(3) => {
                    let zoom = *zoom;
                    self.dev_zoom = None;
                    self.shared().set_zoom(zoom);
                }
                _ => {
                    ui.ctx().request_repaint_after(Duration::from_millis(200));
                }
            }
        }
        if self.dev_read && self.backend_ready() {
            self.dev_read = false;
            self.read();
        }
        if self.dev_detect.0 && self.detector.as_ref().is_some_and(|d| d.ready()) {
            self.dev_detect.0 = false;
            self.detect();
        }
        if self.dev_detect.1 && !self.blocks.is_empty() && self.backend_ready() {
            self.dev_detect.1 = false;
            self.read_all();
        }

        let side = ui.available_width() * 0.4;
        egui::Panel::right("crop")
            .resizable(true)
            .default_size(side)
            .show(ui, |ui| self.crop_panel(ui, &frame));
        egui::CentralPanel::default().show(ui, |ui| self.preview_panel(ui, &frame));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.worker.stop();
    }
}

/// One backend's answer: every distinct reading with its support, and what did not
/// come back. Text is selectable, with a copy button, since the point is to use it.
fn show_transcription(ui: &mut egui::Ui, t: &Transcription) {
    ui.horizontal(|ui| {
        ui.strong(&t.backend);
        ui.weak(format!(
            "{} sample{} in {:.1}s",
            t.samples,
            if t.samples == 1 { "" } else { "s" },
            t.elapsed.as_secs_f32()
        ));
    });
    if t.readings.is_empty() {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "no answer: every sample came back empty",
        );
    }
    for r in &t.readings {
        ui.horizontal(|ui| {
            if t.samples > 1 {
                ui.weak(format!("{}/{}", r.count, t.samples));
            }
            if ui
                .small_button("copy")
                .on_hover_text("copy this reading")
                .clicked()
            {
                ui.ctx().copy_text(r.text.clone());
            }
            ui.add(egui::Label::new(egui::RichText::new(&r.text).size(18.0)).wrap());
        });
        if r.truncated {
            ui.colored_label(ui.visuals().warn_fg_color, "cut off at the token limit");
        }
    }
    if t.silent > 0 && !t.readings.is_empty() {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("{} of {} answers came back empty", t.silent, t.samples),
        );
    }
}

#[derive(Default)]
struct FpsCounter {
    rate: f32,
    last_frames: u64,
    last_at: Option<Instant>,
}

impl FpsCounter {
    fn tick(&mut self, frames: u64) {
        let now = Instant::now();
        match self.last_at {
            None => {
                self.last_at = Some(now);
                self.last_frames = frames;
            }
            Some(at) if now.duration_since(at).as_secs_f32() >= 1.0 => {
                let dt = now.duration_since(at).as_secs_f32();
                self.rate = (frames - self.last_frames) as f32 / dt;
                self.last_at = Some(now);
                self.last_frames = frames;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 6x4 source frame with luma = 16 + x + 10 y, so every pixel is identifiable.
    fn frame() -> YuvFrame {
        let (w, h) = (6, 4);
        let y = (0..h)
            .flat_map(|row| (0..w).map(move |col| (16 + col + 10 * row) as u8))
            .collect();
        YuvFrame {
            width: w,
            height: h,
            y,
            u: vec![128; 6],
            v: vec![128; 6],
        }
    }

    /// The red channel of a rendered gray pixel: monotonic in the source luma, so it
    /// identifies the source pixel.
    fn at(rgba: &[u8], w: usize, x: usize, y: usize) -> u8 {
        rgba[(y * w + x) * 4]
    }

    #[test]
    fn crop_maps_back_to_source_under_every_rotation() {
        let f = frame();
        // A 2x1 view-space region; whichever way the frame is turned, rendering it
        // must equal cutting it out of the rotated whole view.
        for rotation in [
            Rotation::None,
            Rotation::Cw90,
            Rotation::Cw180,
            Rotation::Cw270,
        ] {
            let (vw, vh) = rotation.rotated_size(f.width, f.height);
            let (whole, ww, _) = render_region(&f, rotation, Crop::whole(vw, vh), 1);
            let region = Crop {
                x: 1,
                y: 1,
                w: 2,
                h: 1,
            };
            let (part, pw, ph) = render_region(&f, rotation, region, 1);
            assert_eq!((pw, ph), (2, 1), "{rotation:?}");
            for x in 0..2 {
                assert_eq!(
                    at(&part, pw, x, 0),
                    at(&whole, ww, region.x + x, region.y),
                    "{rotation:?} pixel {x}"
                );
            }
        }
    }

    #[test]
    fn rotated_whole_view_has_the_source_corner_where_expected() {
        let f = frame();
        let (v, w, _) = render_region(&f, Rotation::None, Crop::whole(6, 4), 1);
        let src_top_left = at(&v, w, 0, 0);
        // Clockwise: the source's top-left corner is the view's top-right.
        let (v, w, h) = render_region(&f, Rotation::Cw90, Crop::whole(4, 6), 1);
        assert_eq!((w, h), (4, 6));
        assert_eq!(at(&v, w, 3, 0), src_top_left);
        // Counter-clockwise: bottom-left.
        let (v, w, _) = render_region(&f, Rotation::Cw270, Crop::whole(4, 6), 1);
        assert_eq!(at(&v, w, 0, 5), src_top_left);
    }

    #[test]
    fn small_drags_are_clicks() {
        assert!(Crop::from_corners((10, 10), (12, 40))
            .clamped(100, 100)
            .is_none());
        assert_eq!(
            Crop::from_corners((90, 5), (200, 20)).clamped(100, 100),
            Some(Crop {
                x: 90,
                y: 5,
                w: 10,
                h: 15
            })
        );
    }
}
