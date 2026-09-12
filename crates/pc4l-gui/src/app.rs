//! The window: a live preview you drag a crop rectangle on, a zoomed view of that
//! region at native pixels, and Capture/Save. Modelled on halo-workbench's hint page:
//! aiming and pressing are one glance, the crop *is* the zoom, and what is shown is
//! always the real pixel size of what arrived.
//!
//! Everything the user sees is in *view* space: the frame turned by the chosen
//! [`Rotation`]. The crop is kept in view coordinates and mapped back to the source
//! frame only to fetch pixels ([`Crop::to_source`]).

use crate::stream::{Shared, Status, Worker};
use crate::transcribe::{Mode, Transcriber, Transcription};
use egui::{
    Color32, ColorImage, Key, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions,
    Vec2,
};
use phone_cam4linux::convert::{i420_region_to_rgba, region_size, rotate_rgba, Rotation};
use phone_cam4linux::decode::YuvFrame;
use phone_cam4linux::Facing;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Longest preview edge we bother converting: the preview is for aiming, the crop
/// view and the capture are the real pixels.
const PREVIEW_MAX_EDGE: usize = 1920;
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
    fn clamped(self, width: usize, height: usize) -> Option<Self> {
        let x = self.x.min(width);
        let y = self.y.min(height);
        let w = self.w.min(width - x);
        let h = self.h.min(height - y);
        (w >= MIN_CROP_PX && h >= MIN_CROP_PX).then_some(Self { x, y, w, h })
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

/// A texture cached against the frame, region, step and rotation it was made from, so
/// a repaint with nothing new (a hover) costs no conversion or upload. Holds the
/// frame: comparing a bare pointer would misfire when the allocator hands a new
/// frame the old one's address.
struct View {
    texture: Option<TextureHandle>,
    name: &'static str,
    key: Option<(Arc<YuvFrame>, Crop, usize, Rotation)>,
}

impl View {
    fn new(name: &'static str) -> Self {
        Self {
            texture: None,
            name,
            key: None,
        }
    }

    fn update(
        &mut self,
        ctx: &egui::Context,
        frame: &Arc<YuvFrame>,
        region: Crop,
        step: usize,
        rotation: Rotation,
    ) {
        if self.key.as_ref().is_some_and(|(f, r, s, rot)| {
            Arc::ptr_eq(f, frame) && *r == region && *s == step && *rot == rotation
        }) {
            return;
        }
        let (rgba, w, h) = render_region(frame, rotation, region, step);
        let image = ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        match &mut self.texture {
            Some(t) => t.set(image, TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture(self.name, image, TextureOptions::LINEAR)),
        }
        self.key = Some((Arc::clone(frame), region, step, rotation));
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
    drag_origin: Option<(usize, usize)>,
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
    /// A read in flight: its result channel, when it started, and which backend.
    pending: Option<(Receiver<anyhow::Result<Transcription>>, Instant, String)>,
    /// Readings for the current capture + crop, oldest first.
    results: Vec<Result<Transcription, String>>,
    /// What `results` were read from; they are dropped when it changes.
    results_key: Option<(Arc<YuvFrame>, Option<Crop>)>,
    /// Development aid: read once, as soon as a frame is available.
    dev_read: bool,
    /// Development aid: write a screenshot of the window to this path after the
    /// delay, then quit.
    screenshot: Option<(Duration, PathBuf, Instant)>,
}

impl App {
    pub fn new(
        worker: Worker,
        save_dir: PathBuf,
        backends: Vec<Arc<dyn Transcriber>>,
        screenshot: Option<(Duration, PathBuf)>,
    ) -> Self {
        Self {
            worker,
            preview: View::new("preview"),
            crop_view: View::new("crop"),
            captured: None,
            rotation: Rotation::None,
            crop: None,
            drag_origin: None,
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
            dev_read: false,
            screenshot: screenshot.map(|(after, path)| (after, path, Instant::now())),
        }
    }

    pub fn set_dev_read(&mut self, on: bool) {
        self.dev_read = on;
    }

    pub fn set_rotation(&mut self, rotation: Rotation) {
        self.rotate(rotation);
    }

    /// Sets the crop in view space; it is clamped to the frame when first drawn.
    pub fn set_crop(&mut self, crop: Option<Crop>) {
        self.crop = crop;
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
        let (vw, vh) = self.rotation.rotated_size(frame.width, frame.height);
        let (region, mode) = match self.crop {
            Some(c) => (c, Mode::Crop),
            None => (Crop::whole(vw, vh), Mode::Page),
        };
        let (rgba, w, h) = render_region(&frame, self.rotation, region, 1);
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
        self.pending = Some((rx, Instant::now(), name));
    }

    /// Collects a finished read, and drops readings that no longer belong to what is
    /// on screen.
    fn poll_read(&mut self) {
        if let Some((rx, _, _)) = &self.pending {
            match rx.try_recv() {
                Ok(result) => {
                    self.results.push(result.map_err(|e| format!("{e:#}")));
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.results.push(Err("the read thread died".into()));
                    self.pending = None;
                }
            }
        }
        let key = self.captured.clone().map(|f| (f, self.crop));
        let same = match (&self.results_key, &key) {
            (Some((a, ca)), Some((b, cb))) => Arc::ptr_eq(a, b) && ca == cb,
            (None, None) => true,
            _ => false,
        };
        if !same {
            self.results.clear();
            self.results_key = key;
        }
    }

    fn rotate(&mut self, rotation: Rotation) {
        if rotation != self.rotation {
            self.rotation = rotation;
            // The crop is in view space; rather than spin it, start over.
            self.crop = None;
        }
    }

    /// Writes the crop (or the whole view) at native resolution, rotated as shown.
    fn save(&mut self) {
        let Some(frame) = self.current_frame() else {
            self.say("nothing to save yet");
            return;
        };
        let (vw, vh) = self.rotation.rotated_size(frame.width, frame.height);
        let region = self.crop.unwrap_or(Crop::whole(vw, vh));
        let (rgba, w, h) = render_region(&frame, self.rotation, region, 1);
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
                    self.crop = None;
                }
            });
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
        });
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
            if self.captured.is_some() {
                ui.separator();
                ui.strong("CAPTURED");
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
            .update(ui.ctx(), frame, Crop::whole(vw, vh), step, self.rotation);
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
            self.drag_origin = response.interact_pointer_pos().map(to_view);
        }
        if let (Some(origin), Some(pos)) = (self.drag_origin, response.interact_pointer_pos()) {
            if response.dragged() || response.drag_stopped() {
                self.crop = Crop::from_corners(origin, to_view(pos)).clamped(vw, vh);
            }
        }
        if response.drag_stopped() {
            self.drag_origin = None;
        }

        if let Some(crop) = self.crop {
            let to_screen = |x: usize, y: usize| {
                Pos2::new(
                    image_rect.min.x + x as f32 * scale,
                    image_rect.min.y + y as f32 * scale,
                )
            };
            let rect = Rect::from_min_max(
                to_screen(crop.x, crop.y),
                to_screen(crop.x + crop.w, crop.y + crop.h),
            );
            let painter = ui.painter_at(image_rect);
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
        let Some(crop) = self.crop else {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label("Drag a box on the preview to zoom to a region.");
                ui.label(
                    "Capture freezes the frame; Save writes the box (or the whole frame) \
                     as a PNG at full resolution; Read it sends it to the model.",
                );
            });
            return;
        };
        let step = crop.w.max(crop.h).div_ceil(PREVIEW_MAX_EDGE).max(1);
        self.crop_view
            .update(ui.ctx(), frame, crop, step, self.rotation);
        let Some(texture) = &self.crop_view.texture else {
            return;
        };
        ui.label(format!(
            "{}×{} px at ({}, {}){}",
            crop.w,
            crop.h,
            crop.x,
            crop.y,
            if step > 1 {
                format!(", shown at 1/{step}")
            } else {
                String::new()
            }
        ));
        let avail = ui.available_size();
        let scale = (avail.x / crop.w as f32).min(avail.y / crop.h as f32);
        let size = Vec2::new(crop.w as f32 * scale, crop.h as f32 * scale);
        ui.centered_and_justified(|ui| {
            ui.add(egui::Image::from_texture(texture).fit_to_exact_size(size));
        });
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
            if let Some((_, started, name)) = &self.pending {
                ui.spinner();
                ui.label(format!("{name}: {:.0}s", started.elapsed().as_secs_f32()));
            }
            if !self.results.is_empty() && ui.small_button("clear").clicked() {
                self.results.clear();
            }
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for result in self.results.iter().rev() {
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
        let (space, esc, save, rot_cw, rot_ccw, enter) = ctx.input(|i| {
            (
                i.key_pressed(Key::Space),
                i.key_pressed(Key::Escape),
                i.modifiers.command && i.key_pressed(Key::S),
                !i.modifiers.shift && i.key_pressed(Key::R),
                i.modifiers.shift && i.key_pressed(Key::R),
                i.key_pressed(Key::Enter),
            )
        });
        if space {
            self.capture();
        }
        if enter {
            self.read();
        }
        if esc {
            self.crop = None;
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
        self.poll_read();
        if self.pending.is_some() {
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
                self.crop = None;
            }
            self.crop_space = Some(view);
        }
        self.crop = self.crop.and_then(|c| c.clamped(view.0, view.1));

        if std::mem::take(&mut self.dev_read) {
            self.read();
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
