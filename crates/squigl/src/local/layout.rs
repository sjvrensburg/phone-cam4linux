//! Block detection: PP-DocLayoutV3 (PaddlePaddle's official ONNX export) on ONNX
//! Runtime. One pass gives every layout element with a class, a score, a reading-order
//! rank and an instance mask; the mask is what makes the model useful on a page that
//! is not flat -- turned into a polygon and then a minimum-area quadrilateral, it
//! follows a skewed block where the axis-aligned box would not. The post-processing
//! (mask → polygon → quad) follows PaddleX's `layout_analysis` processors.

use super::{attempts, models, open_session, Device, DevicePref, GPU_LOST};
use crate::app::Crop;
use crate::layout::{self, BlockDetector};
use anyhow::{anyhow, bail, Result};
use image::{GrayImage, RgbImage};
use imageproc::contours::{find_contours, BorderType};
use imageproc::geometry::{approximate_polygon_dp, arc_length, contour_area, min_area_rect};
use imageproc::point::Point;
use ndarray::{Array, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The network's input edge; the image is stretched to it, not letterboxed.
const INPUT: u32 = 800;
/// The mask resolution, `INPUT / 4`.
const MASK: usize = 200;

/// `label_list` from the model's `inference.yml`, in class-id order.
pub const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// Four corners in image pixels, clockwise from the top-left.
pub type Quad = [[f32; 2]; 4];

/// One detected layout element, in the pixels of the image it was found in.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub label: &'static str,
    pub score: f32,
    /// Axis-aligned `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// Reading-order rank from the model (0 first).
    pub order: i32,
    /// The multi-point box: the minimum-area rectangle around the mask's outline.
    /// Equals the bbox corners when the mask gave nothing usable.
    pub quad: Quad,
}

/// The detector: one session.
pub struct Detector {
    session: Session,
    device: Device,
}

impl Detector {
    pub fn load(dir: &std::path::Path, device: Device) -> Result<Self> {
        let session = open_session(&dir.join("inference.onnx"), device)?;
        Ok(Self { session, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Detects blocks scoring at least `threshold`, sorted into reading order. Scores
    /// on handwriting run 0.4-0.8 (the model is trained on printed documents), so
    /// the configured default sits below PaddleX's 0.5.
    pub fn detect(&mut self, img: &RgbImage, threshold: f32) -> Result<Vec<Block>> {
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            bail!("empty image");
        }
        let t0 = Instant::now();
        let small =
            image::imageops::resize(img, INPUT, INPUT, image::imageops::FilterType::Triangle);
        let plane = (INPUT * INPUT) as usize;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = px.0[c] as f32 / 255.0;
            }
        }
        let image_t = Array::from_shape_vec(IxDyn(&[1, 3, INPUT as usize, INPUT as usize]), chw)?;
        let im_shape = Array::from_shape_vec(IxDyn(&[1, 2]), vec![INPUT as f32, INPUT as f32])?;
        // [h_scale, w_scale]: with these the boxes come back in source pixels.
        let scale = Array::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![INPUT as f32 / h as f32, INPUT as f32 / w as f32],
        )?;
        let out = self.session.run(ort::inputs![
            "im_shape" => Tensor::from_array(im_shape)?,
            "image" => Tensor::from_array(image_t)?,
            "scale_factor" => Tensor::from_array(scale)?,
        ])?;
        let boxes: ArrayD<f32> = out["fetch_name_0"].try_extract_array::<f32>()?.to_owned();
        let masks: ArrayD<i32> = out["fetch_name_2"].try_extract_array::<i32>()?.to_owned();
        drop(out);
        let t_run = t0.elapsed();

        let n = boxes.shape()[0];
        let cols = boxes.shape()[1];
        if cols < 7 || masks.shape() != [n, MASK, MASK] {
            bail!(
                "unexpected output shapes {:?} and {:?}",
                boxes.shape(),
                masks.shape()
            );
        }
        // Source px -> mask px.
        let (sx, sy) = (INPUT as f32 / w as f32 / 4.0, INPUT as f32 / h as f32 / 4.0);
        let mut blocks = Vec::new();
        for i in 0..n {
            let score = boxes[[i, 1]];
            if score < threshold {
                continue;
            }
            let bbox = [
                boxes[[i, 2]].clamp(0.0, w as f32),
                boxes[[i, 3]].clamp(0.0, h as f32),
                boxes[[i, 4]].clamp(0.0, w as f32),
                boxes[[i, 5]].clamp(0.0, h as f32),
            ];
            if bbox[2] - bbox[0] < 1.0 || bbox[3] - bbox[1] < 1.0 {
                continue;
            }
            let class = boxes[[i, 0]] as usize;
            blocks.push(Block {
                label: LABELS.get(class).copied().unwrap_or("?"),
                score,
                bbox,
                order: boxes[[i, 6]] as i32,
                quad: mask_quad(&masks, i, bbox, sx, sy).unwrap_or_else(|| rect_quad(bbox)),
            });
        }
        suppress_overlaps(&mut blocks);
        blocks.sort_by_key(|b| b.order);
        log::debug!(
            "layout: {} of {} blocks kept; run {:.0} ms, post {:.0} ms",
            blocks.len(),
            n,
            t_run.as_secs_f64() * 1e3,
            (t0.elapsed() - t_run).as_secs_f64() * 1e3
        );
        Ok(blocks)
    }
}

/// The model reports the same region more than once at times (a title and a text
/// block on the same lines, or a near-duplicate a class apart); PaddleX runs an NMS
/// over the classes for this. Keeps the higher score when two boxes mostly coincide,
/// or when one is almost entirely inside the other.
fn suppress_overlaps(blocks: &mut Vec<Block>) {
    blocks.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Block> = Vec::with_capacity(blocks.len());
    for b in blocks.drain(..) {
        let dup = kept.iter().any(|k| {
            let (inter, a1, a2) = overlap(k.bbox, b.bbox);
            inter / (a1 + a2 - inter) > 0.6 || inter / a1.min(a2) > 0.85
        });
        if !dup {
            kept.push(b);
        }
    }
    *blocks = kept;
}

/// Intersection area and the two areas.
fn overlap(a: [f32; 4], b: [f32; 4]) -> (f32, f32, f32) {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    (
        w * h,
        (a[2] - a[0]) * (a[3] - a[1]),
        (b[2] - b[0]) * (b[3] - b[1]),
    )
}

pub fn rect_quad(b: [f32; 4]) -> Quad {
    [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

/// PaddleX's `extract_polygon_points_by_masks` + `mask2polygon` +
/// `convert_polygon_to_quad`: crop the mask to the box, resize it to the box, take
/// the largest outer contour, simplify it, and wrap it in a minimum-area rectangle.
fn mask_quad(masks: &ArrayD<i32>, i: usize, bbox: [f32; 4], sx: f32, sy: f32) -> Option<Quad> {
    let [x1, y1, x2, y2] = bbox;
    let (bw, bh) = ((x2 - x1).round() as u32, (y2 - y1).round() as u32);
    if bw == 0 || bh == 0 {
        return None;
    }
    let mx0 = ((x1 * sx).round() as usize).min(MASK);
    let mx1 = ((x2 * sx).round() as usize).min(MASK);
    let my0 = ((y1 * sy).round() as usize).min(MASK);
    let my1 = ((y2 * sy).round() as usize).min(MASK);
    if mx1 <= mx0 || my1 <= my0 {
        return None;
    }
    let (cw, ch) = (mx1 - mx0, my1 - my0);
    let mut crop = GrayImage::new(cw as u32, ch as u32);
    let mut any = false;
    for y in 0..ch {
        for x in 0..cw {
            if masks[[i, my0 + y, mx0 + x]] != 0 {
                any = true;
                crop.put_pixel(x as u32, y as u32, image::Luma([255]));
            }
        }
    }
    if !any {
        return None;
    }
    let resized = image::imageops::resize(&crop, bw, bh, image::imageops::FilterType::Nearest);
    let contours = find_contours::<i32>(&resized);
    let largest = contours
        .iter()
        .filter(|c| c.border_type == BorderType::Outer)
        .max_by(|a, b| contour_area(&a.points).total_cmp(&contour_area(&b.points)))?;
    let pts: Vec<Point<f32>> = largest
        .points
        .iter()
        .map(|p| Point::new(p.x as f32, p.y as f32))
        .collect();
    let eps = 0.004 * arc_length(&pts, true);
    let approx = approximate_polygon_dp(&pts, eps, true);
    if approx.len() < 3 {
        return None;
    }
    let rect = min_area_rect(&approx);
    let mut quad = [[0f32; 2]; 4];
    for (q, p) in quad.iter_mut().zip(rect) {
        *q = [p.x + x1, p.y + y1];
    }
    Some(order_quad(quad))
}

/// Clockwise from the top-left corner, as PaddleX orders it.
pub fn order_quad(q: Quad) -> Quad {
    let cx = q.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = q.iter().map(|p| p[1]).sum::<f32>() / 4.0;
    let mut v = q.to_vec();
    // Screen y points down, so ascending angle is clockwise on screen.
    v.sort_by(|a, b| {
        (a[1] - cy)
            .atan2(a[0] - cx)
            .total_cmp(&(b[1] - cy).atan2(b[0] - cx))
    });
    let tl = (0..4)
        .min_by(|&a, &b| (v[a][0] + v[a][1]).total_cmp(&(v[b][0] + v[b][1])))
        .unwrap();
    v.rotate_left(tl);
    [v[0], v[1], v[2], v[3]]
}

// ---------------------------------------------------------------------------

enum State {
    Preparing(String),
    Ready(Box<Detector>),
    Failed(String),
}

/// The detector as the window uses it: preparation (download + load) starts on
/// construction, on a thread; `detect` refuses with the current status until it is
/// done.
pub struct LayoutService {
    state: Arc<Mutex<State>>,
    threshold: f32,
    device: DevicePref,
}

impl LayoutService {
    pub fn new(device: DevicePref, threshold: f32) -> Self {
        let service = Self {
            state: Arc::new(Mutex::new(State::Preparing("locating model".into()))),
            threshold,
            device,
        };
        service.prepare();
        service
    }

    /// Finds, downloads and loads the model on a thread; the state says how far.
    fn prepare(&self) {
        let worker_state = Arc::clone(&self.state);
        let device = self.device;
        std::thread::Builder::new()
            .name("squigl-layout".into())
            .spawn(move || {
                let set = |s: String| {
                    *worker_state.lock().unwrap() = State::Preparing(s);
                };
                let result = models::DOC_LAYOUT.ensure(&set).and_then(|dir| {
                    let mut last = None;
                    for &d in attempts(device) {
                        set(format!("loading on {}", d.name()));
                        match Detector::load(&dir, d) {
                            Ok(m) => return Ok(m),
                            Err(e) => {
                                log::warn!("PP-DocLayoutV3 on {}: {e:#}", d.name());
                                last = Some(e);
                            }
                        }
                    }
                    Err(last.unwrap_or_else(|| anyhow!("no device to try")))
                });
                *worker_state.lock().unwrap() = match result {
                    Ok(d) => {
                        log::info!("PP-DocLayoutV3 ready on {}", d.device().name());
                        State::Ready(Box::new(d))
                    }
                    Err(e) => {
                        log::error!("block detector unavailable: {e:#}");
                        State::Failed(format!("{e:#}"))
                    }
                };
            })
            .expect("spawning layout thread");
    }
}

impl BlockDetector for LayoutService {
    fn status(&self) -> Option<String> {
        match &*self.state.lock().unwrap() {
            State::Preparing(s) => Some(format!("block detector: {s}")),
            State::Ready(_) => None,
            State::Failed(e) => Some(format!("block detector unavailable: {e}")),
        }
    }

    fn ready(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), State::Ready(_))
    }

    fn detect(&self, img: &RgbImage) -> Result<Vec<layout::Block>> {
        let mut state = self.state.lock().unwrap();
        let blocks = match &mut *state {
            State::Ready(d) => {
                if d.device() == Device::WebGpu && GPU_LOST.load(Ordering::SeqCst) {
                    *state = State::Preparing("GPU lost — reloading on the CPU".into());
                    self.prepare();
                    bail!("the GPU was lost; the block detector is reloading on the CPU");
                }
                let _turn = super::runtime_turn();
                match d.detect(img, self.threshold) {
                    Ok(blocks) => blocks,
                    Err(e) if super::note_gpu_loss(&e) => {
                        *state = State::Preparing("GPU lost — reloading on the CPU".into());
                        self.prepare();
                        bail!("the GPU was lost; the block detector is reloading on the CPU");
                    }
                    Err(e) => return Err(e),
                }
            }
            State::Preparing(s) => bail!("block detector not ready yet: {s}"),
            State::Failed(e) => bail!("block detector unavailable: {e}"),
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        Ok(blocks
            .into_iter()
            .filter_map(|b| {
                let [x1, y1, x2, y2] = b.bbox;
                let rect = Crop {
                    x: x1.floor() as usize,
                    y: y1.floor() as usize,
                    w: (x2.ceil() - x1.floor()) as usize,
                    h: (y2.ceil() - y1.floor()) as usize,
                }
                .clamped(w, h)?;
                Some(layout::Block {
                    label: b.label,
                    score: b.score,
                    rect,
                    quad: (!layout::is_rectangular(&b.quad, rect)).then_some(b.quad),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quads_are_ordered_clockwise_from_top_left() {
        let q = order_quad([[10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]);
        assert_eq!(q, [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        // A tilted one keeps its shape.
        let tilted = order_quad([[6.0, 0.0], [0.0, 5.0], [14.0, 20.0], [20.0, 15.0]]);
        assert_eq!(tilted, [[0.0, 5.0], [6.0, 0.0], [20.0, 15.0], [14.0, 20.0]]);
    }

    #[test]
    fn overlapping_detections_keep_the_stronger() {
        let mk = |score, bbox, order| Block {
            label: "text",
            score,
            bbox,
            order,
            quad: rect_quad(bbox),
        };
        let mut blocks = vec![
            mk(0.5, [0.0, 0.0, 100.0, 50.0], 0),
            mk(0.7, [2.0, 1.0, 101.0, 52.0], 1),
            mk(0.6, [10.0, 10.0, 30.0, 20.0], 2),
            mk(0.9, [200.0, 0.0, 300.0, 50.0], 3),
        ];
        suppress_overlaps(&mut blocks);
        let mut orders: Vec<i32> = blocks.iter().map(|b| b.order).collect();
        orders.sort();
        assert_eq!(orders, vec![1, 3]);
    }

    #[test]
    fn labels_match_the_export() {
        assert_eq!(LABELS.len(), 25);
        assert_eq!(LABELS[22], "text");
    }
}
