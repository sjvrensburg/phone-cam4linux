//! Layout blocks: what a block detector reports, and the geometry the window needs
//! to use it -- independent of which detector (the built-in PP-DocLayoutV3 lives in
//! `local::layout`) so the window compiles without one.

use crate::app::Crop;
use anyhow::Result;
use image::{Rgb, RgbImage, RgbaImage};
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};

/// Four corners in view pixels, clockwise from the top-left. A block on a page that
/// is not flat is a quadrilateral, not a rectangle; the crop it becomes is
/// [`rectify`]d before anything reads it.
pub type Quad = [[f32; 2]; 4];

/// One detected layout element, in view space, as the window shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub label: &'static str,
    pub score: f32,
    /// Axis-aligned bounds, clamped to the view.
    pub rect: Crop,
    /// The multi-point box. `None` when it is just the rectangle.
    pub quad: Option<Quad>,
}

/// Something that finds blocks in a page image. Implementations run on a worker
/// thread and may block.
pub trait BlockDetector: Send + Sync {
    /// A line for the window about the detector's own state (downloading, loading,
    /// unavailable); `None` when it is ready.
    fn status(&self) -> Option<String>;
    fn ready(&self) -> bool;
    /// Blocks in `img`, in reading order, in `img`'s pixels.
    fn detect(&self, img: &RgbImage) -> Result<Vec<Block>>;
}

/// The corners of a rectangle as a quad.
pub fn rect_quad(r: Crop) -> Quad {
    let (x0, y0) = (r.x as f32, r.y as f32);
    let (x1, y1) = ((r.x + r.w) as f32, (r.y + r.h) as f32);
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

/// Whether a quad is (within a pixel) just its bounding rectangle.
#[cfg_attr(not(feature = "local-model"), allow(dead_code))]
pub fn is_rectangular(q: &Quad, r: Crop) -> bool {
    rect_quad(r)
        .iter()
        .zip(q)
        .all(|(a, b)| (a[0] - b[0]).abs() < 1.0 && (a[1] - b[1]).abs() < 1.0)
}

/// The size a rectified quad comes out at: its longer opposite edges.
pub fn rectified_size(q: &Quad) -> (u32, u32) {
    let len = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let w = len(q[0], q[1]).max(len(q[3], q[2])).round().max(1.0) as u32;
    let h = len(q[0], q[3]).max(len(q[1], q[2])).round().max(1.0) as u32;
    (w, h)
}

/// Warps the quad (in `img`'s pixels) to an upright rectangle sized by its edges, so
/// a block read off a curved or tilted page arrives straight. Edge pixels are
/// replicated where the quad reaches outside the image.
pub fn rectify(img: &RgbaImage, q: &Quad) -> Option<RgbaImage> {
    let (w, h) = rectified_size(q);
    let from = [
        (q[0][0], q[0][1]),
        (q[1][0], q[1][1]),
        (q[2][0], q[2][1]),
        (q[3][0], q[3][1]),
    ];
    let to = [
        (0.0, 0.0),
        (w as f32, 0.0),
        (w as f32, h as f32),
        (0.0, h as f32),
    ];
    let proj = Projection::from_control_points(from, to)?;
    let mut out = RgbaImage::new(w, h);
    warp_into(
        img,
        proj,
        Interpolation::Bilinear,
        Border::Replicate,
        &mut out,
    );
    Some(out)
}

/// Packed RGBA8 → RGB, for a detector.
pub fn rgba_to_rgb(rgba: &[u8], w: usize, h: usize) -> RgbImage {
    let mut out = RgbImage::new(w as u32, h as u32);
    for (dst, src) in out.pixels_mut().zip(rgba.chunks_exact(4)) {
        *dst = Rgb([src[0], src[1], src[2]]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rectangle_rectifies_to_itself() {
        let mut img = RgbaImage::new(20, 10);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba([x as u8 * 10, y as u8 * 20, 0, 255]);
        }
        let q = [[2.0, 1.0], [12.0, 1.0], [12.0, 6.0], [2.0, 6.0]];
        let out = rectify(&img, &q).unwrap();
        assert_eq!((out.width(), out.height()), (10, 5));
        // Sampling at pixel centres: out (0,0) is source (2,1) -- half a pixel in.
        let p = out.get_pixel(0, 0);
        assert!(
            (p[0] as i32 - 20).abs() <= 10 && (p[1] as i32 - 20).abs() <= 20,
            "{p:?}"
        );
        let p = out.get_pixel(9, 4);
        assert!(
            (p[0] as i32 - 110).abs() <= 10 && (p[1] as i32 - 100).abs() <= 20,
            "{p:?}"
        );
    }

    #[test]
    fn rect_quads_are_recognised() {
        let r = Crop {
            x: 3,
            y: 4,
            w: 10,
            h: 5,
        };
        assert!(is_rectangular(&rect_quad(r), r));
        let mut q = rect_quad(r);
        q[2][0] += 3.0;
        assert!(!is_rectangular(&q, r));
    }
}
