//! The window: a live preview you drag a crop rectangle on, a zoomed view of that
//! region at native pixels, and Capture/Save. Modelled on halo-workbench's hint page:
//! aiming and pressing are one glance, the crop *is* the zoom, and what is shown is
//! always the real pixel size of what arrived.
//!
//! Everything the user sees is in *view* space: the frame turned by the chosen
//! [`Rotation`]. The crop is kept in view coordinates and mapped back to the source
//! frame only to fetch pixels ([`Crop::to_source`]).
//!
//! A block detector, when there is one, runs over the page while block mode is on --
//! at a few Hz on the live preview, once on a capture -- and lists its blocks in
//! reading order; picking one makes it the crop, and a block that is not a
//! rectangle (a curved or tilted page) is rectified before it is shown or read. The
//! crop's corners can be dragged, so a block is a starting point, not a verdict.

use crate::layout::{self, Block, BlockDetector, Quad};
use crate::settings;
use crate::stream::{Shared, Status, Worker};
use crate::transcribe::{BackendConfig, Config, LayoutConfig, Mode, Transcriber, Transcription};
use egui::{
    Color32, ColorImage, FontId, Key, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};
use phone_cam4linux::convert::{i420_region_to_rgba, region_size, rotate_rgba, Rotation};
use phone_cam4linux::decode::YuvFrame;
use phone_cam4linux::Facing;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Longest preview edge we bother converting: the preview is for aiming, the crop
/// view and the capture are the real pixels.
const PREVIEW_MAX_EDGE: usize = 1920;
/// Longest edge handed to the block detector (its own input is 800 px square).
const DETECT_MAX_EDGE: usize = 1600;
/// How often, at most, the detector runs on the live preview.
const LIVE_DETECT_INTERVAL: Duration = Duration::from_millis(200);
/// How close (screen px) to a corner a drag must start to take the corner.
const HANDLE_PX: f32 = 10.0;
/// Drags smaller than this are a click, which clears the crop.
const MIN_CROP_PX: usize = 8;
const SELECTED_COLOUR: Color32 = Color32::from_rgb(255, 196, 0);
const BLOCK_COLOUR: Color32 = Color32::from_rgb(255, 90, 40);

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

    /// Intersection over union with `other`: how much the same box they are.
    fn iou(self, other: Self) -> f32 {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.w).min(other.x + other.w);
        let y1 = (self.y + self.h).min(other.y + other.h);
        if x1 <= x0 || y1 <= y0 {
            return 0.0;
        }
        let inter = ((x1 - x0) * (y1 - y0)) as f32;
        inter / ((self.area() + other.area()) as f32 - inter)
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

/// Typesets a reading (LaTeX maths and all) into pixels; `None` in a build without
/// one. Implemented by `mathtext::Renderer`.
pub trait Typesetter: Send + Sync {
    /// `width_pt` points wide, text `size_pt`, `scale` pixels per point, in `rgb`.
    fn render(
        &self,
        text: &str,
        width_pt: f32,
        size_pt: f32,
        scale: f32,
        rgb: [u8; 3],
    ) -> anyhow::Result<(image::RgbaImage, f32)>;
}

/// Width the readings are typeset at, in points; shown scaled down if the panel
/// is narrower.
const TYPESET_WIDTH_PT: f32 = 440.0;
const TYPESET_SIZE_PT: f32 = 15.0;

/// A reading typeset, or why not.
enum Typeset {
    Image { texture: TextureHandle, size: Vec2 },
    Failed,
}

/// One entry in the results list: its block label (in a "read all"), the answer,
/// and the typeset readings, made on first show.
struct ResultEntry {
    label: Option<String>,
    result: Result<Transcription, String>,
    typeset: HashMap<usize, Typeset>,
}

/// Builds the block detector for a layout config; `None` when the build has none or
/// it is disabled.
pub type DetectorFactory = Box<dyn Fn(&LayoutConfig) -> Option<Arc<dyn BlockDetector>>>;

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
    /// Dragging one corner of the selection (index into its quad, clockwise from
    /// the top-left).
    Corner(usize),
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
    /// The configuration in force, and the backends built from it (`backend_configs`
    /// says which config each one came from, so a Save can keep the unchanged ones
    /// -- the local model is not reloaded for an edit elsewhere).
    config: Config,
    backend_configs: Vec<BackendConfig>,
    backends: Vec<Arc<dyn Transcriber>>,
    selected_backend: usize,
    detector_factory: DetectorFactory,
    /// The Settings window's draft while it is open.
    draft: Option<Config>,
    settings_open: bool,
    /// For waking the UI from the read thread; set on the first frame.
    ctx: Option<egui::Context>,
    pending: Option<PendingRead>,
    /// Readings for the current capture, oldest first.
    results: Vec<ResultEntry>,
    typesetter: Option<Arc<dyn Typesetter>>,
    /// Show readings typeset (maths rendered) rather than as raw text.
    typeset_on: bool,
    /// The zoom factor the typeset textures were rendered for.
    last_zoom: Option<f32>,
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
    /// Block mode: the detector runs on whatever is shown and its blocks are drawn.
    block_mode: bool,
    pending_detect: Option<(Receiver<anyhow::Result<Vec<Block>>>, Instant)>,
    last_detect: Option<Instant>,
    /// Blocks still to read, for "read all": label and where, taken when it began
    /// so a live re-detection cannot reshuffle them.
    read_queue: VecDeque<(String, Selection)>,
    /// A "read all" waiting for the capture's own blocks to arrive.
    read_all_armed: bool,
    /// Development aid: read once, as soon as a frame is available.
    dev_read: bool,
    /// Development aid: read every block once there are some.
    dev_read_all: bool,
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
        config: Config,
        detector_factory: DetectorFactory,
        typesetter: Option<Arc<dyn Typesetter>>,
        screenshot: Option<(Duration, PathBuf)>,
    ) -> Self {
        let mut app = Self {
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
            config: Config {
                backends: Vec::new(),
                ..config.clone()
            },
            backend_configs: Vec::new(),
            backends: Vec::new(),
            selected_backend: 0,
            detector_factory,
            draft: None,
            settings_open: false,
            ctx: None,
            pending: None,
            results: Vec::new(),
            typesetter: typesetter.clone(),
            typeset_on: typesetter.is_some(),
            last_zoom: None,
            results_key: None,
            detector: None,
            blocks: Vec::new(),
            blocks_key: None,
            selected_block: None,
            block_mode: false,
            pending_detect: None,
            last_detect: None,
            read_queue: VecDeque::new(),
            read_all_armed: false,
            dev_read: false,
            dev_read_all: false,
            dev_zoom: None,
            screenshot: screenshot.map(|(after, path)| (after, path, Instant::now())),
        };
        app.apply_config(config, true);
        app
    }

    /// Puts `new` in force: backends whose entry did not change are kept (a local
    /// model stays loaded), the rest are built; the detector is rebuilt if its
    /// section changed; the scale is applied once the window exists.
    fn apply_config(&mut self, new: Config, first: bool) {
        let mut backends = Vec::new();
        let mut configs = Vec::new();
        for b in &new.backends {
            let existing = self
                .backend_configs
                .iter()
                .position(|c| c == b)
                .map(|i| Arc::clone(&self.backends[i]));
            let built = existing.or_else(|| b.build().map(Into::into));
            if let Some(t) = built {
                backends.push(t);
                configs.push(b.clone());
            }
        }
        let selected_name = self
            .backends
            .get(self.selected_backend)
            .map(|b| b.name().to_string());
        self.backends = backends;
        self.backend_configs = configs;
        self.selected_backend = selected_name
            .and_then(|n| self.backends.iter().position(|b| b.name() == n))
            .unwrap_or(0);
        if first || new.layout != self.config.layout {
            self.detector = (self.detector_factory)(&new.layout);
            if self.detector.is_none() {
                self.block_mode = false;
            }
        }
        if let Some(ctx) = &self.ctx {
            ctx.set_zoom_factor(new.ui.scale);
        }
        self.config = new;
    }

    /// The Settings window, when open. The draft's scale is applied once settled.
    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }
        let mut draft = self.draft.take().unwrap_or_else(|| self.config.clone());
        let outcome = settings::show(ctx, &mut self.settings_open, &mut draft);
        match outcome.action {
            settings::Action::None => {
                if outcome.apply_scale && (ctx.zoom_factor() - draft.ui.scale).abs() > 1e-3 {
                    ctx.set_zoom_factor(draft.ui.scale);
                }
                self.draft = Some(draft);
            }
            settings::Action::Save => {
                match draft.save() {
                    Ok(()) => self.say(format!("settings saved to {}", Config::path().display())),
                    Err(e) => self.say(format!("saving settings failed: {e:#}")),
                }
                self.apply_config(draft, false);
                self.draft = None;
                self.settings_open = false;
            }
            settings::Action::Cancel => {
                self.draft = None;
                self.settings_open = false;
            }
        }
    }

    /// The scale in force is the truth, whoever set it (the Settings slider, egui's
    /// own ctrl+plus / ctrl+minus / ctrl+0): a change is written into the config
    /// and the draft alike and saved at once, independent of Save/Cancel, so no
    /// later path can snap it back. The typeset textures, rendered for the old
    /// pixel density, are dropped.
    fn track_zoom(&mut self, ctx: &egui::Context) {
        let zoom = ctx.zoom_factor();
        // Only a change of the zoom itself counts: a draft mid-drag legitimately
        // differs from it.
        let changed = self.last_zoom.is_some_and(|z| (z - zoom).abs() > 1e-3);
        if changed {
            log::debug!("window scale now {zoom:.2}");
            if let Some(d) = &mut self.draft {
                d.ui.scale = zoom;
            }
            if (self.config.ui.scale - zoom).abs() > 1e-3 {
                self.config.ui.scale = zoom;
                if let Err(e) = self.config.save() {
                    log::warn!("saving the window scale: {e:#}");
                }
            }
            for entry in &mut self.results {
                entry.typeset.clear();
            }
        }
        self.last_zoom = Some(zoom);
    }

    pub fn set_dev_zoom(&mut self, zoom: Option<f32>) {
        self.dev_zoom = zoom.map(|z| (z, None));
    }

    pub fn set_dev_read(&mut self, on: bool) {
        self.dev_read = on;
    }

    pub fn set_settings_open(&mut self, open: bool) {
        self.settings_open = open;
    }

    /// Opens the Settings window, or closes it dropping unsaved edits (the scale
    /// is not an edit: it is already in force and saved).
    fn toggle_settings(&mut self) {
        self.settings_open = !self.settings_open;
        if !self.settings_open {
            self.draft = None;
        }
    }

    /// Starts in block mode, optionally reading every block once there are some.
    pub fn set_dev_detect(&mut self, detect: bool, read_all: bool) {
        self.block_mode = detect;
        self.dev_read_all = read_all;
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

    /// Reads every detected block in reading order, one after the other. Reads are
    /// of a capture, so a live view is captured first and its own detection awaited.
    fn read_all(&mut self) {
        if !self.block_mode {
            self.say("turn on Blocks [L] first");
            return;
        }
        if self.captured.is_none() {
            self.capture();
        }
        self.read_all_armed = true;
        self.start_read_all_if_ready();
    }

    /// The armed "read all" begins once the blocks are the capture's.
    fn start_read_all_if_ready(&mut self) {
        if !self.read_all_armed || self.pending_detect.is_some() {
            return;
        }
        let Some(captured) = &self.captured else {
            self.read_all_armed = false;
            return;
        };
        if !self
            .blocks_key
            .as_ref()
            .is_some_and(|(f, _)| Arc::ptr_eq(f, captured))
        {
            return;
        }
        self.read_all_armed = false;
        if self.blocks.is_empty() {
            self.say("no blocks found on this page");
            return;
        }
        self.read_queue = self
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                (
                    format!("#{} {}", i + 1, b.label),
                    Selection {
                        rect: b.rect,
                        quad: b.quad,
                    },
                )
            })
            .collect();
        self.set_results_key(ResultsScope::AllBlocks);
        self.results.clear();
        self.next_queued_read();
    }

    /// Starts the next block of a "read all" once the previous one has landed.
    fn next_queued_read(&mut self) {
        if self.pending.is_some() {
            return;
        }
        if let Some((label, selection)) = self.read_queue.pop_front() {
            self.crop = Some(selection.rect);
            self.quad = selection.quad;
            self.selected_block = None;
            self.read_selection(Some(selection), ResultsScope::AllBlocks, Some(label));
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
        let prompt = self.config.prompts.for_mode(mode).to_string();
        std::thread::Builder::new()
            .name("pc4l-read".into())
            .spawn(move || {
                let result = backend.read(&png, mode, &prompt, (vw as u32, vh as u32));
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
                self.results.push(ResultEntry {
                    label,
                    result,
                    typeset: HashMap::new(),
                });
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
                        if self.captured.is_some() {
                            self.say(format!(
                                "{} block{} in {:.2}s",
                                blocks.len(),
                                if blocks.len() == 1 { "" } else { "s" },
                                elapsed.as_secs_f32()
                            ));
                        }
                        self.follow_selection(&blocks);
                        self.blocks = blocks;
                    }
                    Err(e) => {
                        // Live mode would repeat the failure every tick.
                        self.block_mode = false;
                        self.say(format!("block detection failed: {e:#}"));
                    }
                }
                self.start_read_all_if_ready();
            }
        }
    }

    /// Block mode: `L` or the toolbar toggle.
    fn toggle_block_mode(&mut self) {
        let Some(detector) = &self.detector else {
            self.say("no block detector in this build");
            return;
        };
        if !self.block_mode {
            if let Some(status) = detector.status() {
                self.say(status);
                return;
            }
        }
        self.block_mode = !self.block_mode;
    }

    /// In block mode, runs the detector over whatever is shown whenever it is not
    /// already running, the frame is new, and (live) the interval has passed. A
    /// read in flight gets the GPU to itself.
    fn maybe_detect(&mut self, ctx: &egui::Context) {
        if !self.block_mode || self.pending_detect.is_some() || self.pending.is_some() {
            return;
        }
        let Some(detector) = self.detector.clone() else {
            return;
        };
        if !detector.ready() {
            return;
        }
        let Some(frame) = self.current_frame() else {
            return;
        };
        if self
            .blocks_key
            .as_ref()
            .is_some_and(|(f, r)| Arc::ptr_eq(f, &frame) && *r == self.rotation)
        {
            return;
        }
        if self.captured.is_none() {
            if let Some(at) = self.last_detect {
                let since = at.elapsed();
                if since < LIVE_DETECT_INTERVAL {
                    ctx.request_repaint_after(LIVE_DETECT_INTERVAL - since);
                    return;
                }
            }
        }
        self.detect(detector, frame);
    }

    /// Runs the block detector over `frame`, on a thread. The result lands in
    /// `blocks` in view space.
    fn detect(&mut self, detector: Arc<dyn BlockDetector>, frame: Arc<YuvFrame>) {
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
        self.last_detect = Some(Instant::now());
    }

    /// Carries the selected block over to a fresh detection: the new block that
    /// overlaps it most keeps the selection (so tab goes on from there), and if the
    /// crop was still exactly that block, the crop follows it.
    fn follow_selection(&mut self, new: &[Block]) {
        let Some(old) = self.selected_block.and_then(|i| self.blocks.get(i)) else {
            self.selected_block = None;
            return;
        };
        let untouched = self.crop == Some(old.rect) && self.quad == old.quad;
        let best = new
            .iter()
            .enumerate()
            .map(|(j, b)| (j, old.rect.iou(b.rect)))
            .filter(|(_, iou)| *iou > 0.3)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((j, _)) => {
                self.selected_block = Some(j);
                if untouched {
                    self.crop = Some(new[j].rect);
                    self.quad = new[j].quad;
                }
            }
            None => self.selected_block = None,
        }
    }

    /// Blocks stay until the next detection replaces them, unless the view they
    /// were found in is gone (rotation, block mode off; the caller handles a frame
    /// size change).
    fn drop_stale_blocks(&mut self) {
        let stale = !self.block_mode
            || self
                .blocks_key
                .as_ref()
                .is_some_and(|(_, r)| *r != self.rotation);
        if stale {
            self.clear_blocks();
        }
    }

    fn clear_blocks(&mut self) {
        self.blocks.clear();
        self.blocks_key = None;
        self.selected_block = None;
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
        // Wrapped: at a larger window scale the row overflows otherwise.
        ui.horizontal_wrapped(|ui| {
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
                let status = detector.status();
                let toggle = ui.add_enabled(
                    status.is_none(),
                    egui::Button::selectable(self.block_mode, "Blocks  [L]"),
                );
                let toggle = match status {
                    Some(status) => toggle.on_disabled_hover_text(status),
                    None => toggle.on_hover_text(
                        "find the page's blocks as you aim; click one or tab through them",
                    ),
                };
                if toggle.clicked() {
                    self.toggle_block_mode();
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
            ui.separator();
            if ui
                .button("Settings  [ctrl+,]")
                .on_hover_text("window scale, backends, prompts, block detector")
                .clicked()
            {
                self.toggle_settings();
            }
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
            // One unit, moved to the next row whole when the wrapped toolbar is
            // short of room (a nested row does not wrap by itself).
            if ui.available_rect_before_wrap().width() < 330.0 {
                ui.end_row();
            }
            ui.horizontal(|ui| {
                ui.label("Zoom:");
                ui.spacing_mut().slider_width = 120.0;
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
            });
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
            if self.block_mode {
                ui.separator();
                if self.captured.is_none() && self.pending_detect.is_some() {
                    ui.spinner();
                }
                ui.label(format!(
                    "{} block{} — click one, tab through them, ctrl+enter reads all",
                    self.blocks.len(),
                    if self.blocks.len() == 1 { "" } else { "s" }
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
        let to_screen =
            |x: f32, y: f32| Pos2::new(image_rect.min.x + x * scale, image_rect.min.y + y * scale);
        let to_view = |p: Pos2| -> (usize, usize) {
            let x = ((p.x - image_rect.min.x) / scale)
                .round()
                .clamp(0.0, vw as f32) as usize;
            let y = ((p.y - image_rect.min.y) / scale)
                .round()
                .clamp(0.0, vh as f32) as usize;
            (x, y)
        };

        // The selection's corners on screen, for the handles.
        let corners: Option<[Pos2; 4]> = self.selection().map(|sel| {
            sel.quad
                .unwrap_or_else(|| layout::rect_quad(sel.rect))
                .map(|[x, y]| to_screen(x, y))
        });
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                let start = to_view(pos);
                let handle =
                    corners.and_then(|c| (0..4).find(|&i| c[i].distance(pos) <= HANDLE_PX));
                // On a corner: drag it. Inside the box: pan it. Outside: draw a new one.
                self.drag = Some(match (handle, self.crop) {
                    (Some(i), _) => Drag::Corner(i),
                    (None, Some(c)) if c.contains(start.0, start.1) => Drag::Move {
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
                    Drag::Corner(i) => self.drag_corner(i, here, vw, vh),
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

        let painter = ui.painter_at(image_rect);
        // Detected blocks: their quads, numbered in reading order. Each line sits on
        // a dark underlay so it reads on white paper and on ink alike.
        for (i, b) in self.blocks.iter().enumerate() {
            let selected = self.selected_block == Some(i);
            let quad = b.quad.unwrap_or_else(|| layout::rect_quad(b.rect));
            let points: Vec<Pos2> = quad.iter().map(|[x, y]| to_screen(*x, *y)).collect();
            let colour = if selected {
                SELECTED_COLOUR
            } else {
                BLOCK_COLOUR
            };
            let width = if selected { 2.5 } else { 2.0 };
            painter.add(Shape::closed_line(
                points.clone(),
                Stroke::new(width + 2.0, Color32::from_black_alpha(200)),
            ));
            painter.add(Shape::closed_line(
                points.clone(),
                Stroke::new(width, colour),
            ));
            let tag = format!("{}", i + 1);
            let font = FontId::proportional(14.0);
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
                Stroke::new(2.0, SELECTED_COLOUR),
                StrokeKind::Outside,
            );
            // The quad, when the selection is one, and the corner handles.
            if let Some(corners) = corners {
                if self.quad.is_some() {
                    painter.add(Shape::closed_line(
                        corners.to_vec(),
                        Stroke::new(2.0, SELECTED_COLOUR),
                    ));
                }
                for c in corners {
                    painter.circle(c, 5.0, SELECTED_COLOUR, Stroke::new(1.5, Color32::BLACK));
                }
            }
        }
    }

    /// Moves corner `i` of the selection to `here`. A rectangle stays a rectangle
    /// (the opposite corner is fixed); a quad's corner moves on its own and the
    /// rectangle around it follows.
    fn drag_corner(&mut self, i: usize, here: (usize, usize), vw: usize, vh: usize) {
        let Some(sel) = self.selection() else {
            return;
        };
        match sel.quad {
            None => {
                let q = layout::rect_quad(sel.rect);
                let opposite = q[(i + 2) % 4];
                let opposite = (opposite[0] as usize, opposite[1] as usize);
                if let Some(rect) = Crop::from_corners(opposite, here).clamped(vw, vh) {
                    self.set_rect(Some(rect));
                }
            }
            Some(mut q) => {
                q[i] = [here.0 as f32, here.1 as f32];
                let (x0, x1) = q.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                    (lo.min(p[0]), hi.max(p[0]))
                });
                let (y0, y1) = q.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                    (lo.min(p[1]), hi.max(p[1]))
                });
                let rect = Crop {
                    x: x0.floor() as usize,
                    y: y0.floor() as usize,
                    w: (x1.ceil() - x0.floor()) as usize,
                    h: (y1.ceil() - y0.floor()) as usize,
                };
                if let Some(rect) = rect.clamped(vw, vh) {
                    self.crop = Some(rect);
                    self.quad = Some(q);
                }
            }
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
                        "Blocks [L] keeps finding the page's blocks in reading order as \
                         you aim: click one or tab through them to make it the box, drag \
                         its corners to adjust it; ctrl+enter reads them all.",
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
            if self.typesetter.is_some() {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.typeset_on, "typeset")
                        .on_hover_text("render the maths; off shows the raw text");
                });
            }
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // One read at a time: newest on top. A "read all": in page order.
                let in_order = matches!(self.results_key, Some((_, ResultsScope::AllBlocks)));
                let typesetter = self.typeset_on.then(|| self.typesetter.clone()).flatten();
                let mut ordered: Vec<_> = self.results.iter_mut().collect();
                if !in_order {
                    ordered.reverse();
                }
                for entry in ordered {
                    if let Some(label) = &entry.label {
                        ui.strong(label);
                    }
                    match &entry.result {
                        Ok(t) => {
                            show_transcription(ui, t, typesetter.as_deref(), &mut entry.typeset)
                        }
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
            self.toggle_block_mode();
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(Key::Comma)) {
            self.toggle_settings();
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
            ui.ctx().set_zoom_factor(self.config.ui.scale);
        }
        self.track_zoom(ui.ctx());
        self.settings_window(ui.ctx());
        self.fps.tick(self.shared().frames());
        self.handle_keys(ui.ctx());
        self.handle_screenshot(ui.ctx());
        self.drop_stale_blocks();
        self.poll_read();
        self.maybe_detect(ui.ctx());
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
                self.clear_blocks();
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
        if self.dev_read_all && !self.blocks.is_empty() && self.backend_ready() {
            self.dev_read_all = false;
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
/// With a typesetter, each reading is rendered on first show and cached in
/// `typeset` by its index; a reading that fails to render stays text.
fn show_transcription(
    ui: &mut egui::Ui,
    t: &Transcription,
    typesetter: Option<&dyn Typesetter>,
    typeset: &mut HashMap<usize, Typeset>,
) {
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
    for (i, r) in t.readings.iter().enumerate() {
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
            let rendered = typesetter.map(|ts| {
                typeset
                    .entry(i)
                    .or_insert_with(|| typeset_reading(ui, ts, &r.text))
            });
            match rendered {
                Some(Typeset::Image { texture, size }) => {
                    ui.add(
                        egui::Image::from_texture(&*texture)
                            .fit_to_exact_size(*size)
                            .max_width(ui.available_width()),
                    );
                }
                _ => {
                    ui.add(egui::Label::new(egui::RichText::new(&r.text).size(18.0)).wrap());
                }
            }
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

/// Renders one reading into a texture in the window's text colour.
fn typeset_reading(ui: &egui::Ui, ts: &dyn Typesetter, text: &str) -> Typeset {
    let colour = ui.visuals().text_color();
    let scale = ui.ctx().pixels_per_point() * 1.5;
    match ts.render(
        text,
        TYPESET_WIDTH_PT,
        TYPESET_SIZE_PT,
        scale,
        [colour.r(), colour.g(), colour.b()],
    ) {
        Ok((image, scale)) => {
            let (w, h) = (image.width() as usize, image.height() as usize);
            let texture = ui.ctx().load_texture(
                "reading",
                ColorImage::from_rgba_unmultiplied([w, h], image.as_raw()),
                TextureOptions::LINEAR,
            );
            Typeset::Image {
                texture,
                size: Vec2::new(w as f32 / scale, h as f32 / scale),
            }
        }
        Err(e) => {
            log::warn!("typesetting a reading failed, showing it as text: {e}");
            Typeset::Failed
        }
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
    fn iou_is_overlap_over_union() {
        let a = Crop {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Crop {
            x: 5,
            y: 0,
            w: 10,
            h: 10,
        };
        assert!((a.iou(a) - 1.0).abs() < 1e-6);
        assert!((a.iou(b) - 50.0 / 150.0).abs() < 1e-6);
        assert_eq!(a.iou(Crop { x: 20, ..a }), 0.0);
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
