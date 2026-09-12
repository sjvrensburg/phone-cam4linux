//! Pixel-format conversions from the decoder's I420 (planar YUV 4:2:0).
//!
//! - [`i420_to_yuyv`]: packed YUYV422 for the V4L2 sink, because it's the most broadly
//!   compatible "raw webcam" format for consumers (browsers, most webcam apps) -- the
//!   real scrcpy V4L2 sink does the same conversion for the same reason.
//! - [`i420_to_rgba`], [`i420_crop_to_rgba`], [`i420_to_rgba_decimated`]: packed RGBA8
//!   for on-screen display (a GUI texture), whole, a region at native pixels, or
//!   every n-th pixel for a cheap preview of a large frame.

use crate::decode::YuvFrame;

/// Converts `frame` into `out` as packed YUYV422 (byte order Y0 U0 Y1 V0 per pixel
/// pair). `out` must be exactly `width * height * 2` bytes.
pub fn i420_to_yuyv(frame: &YuvFrame, out: &mut [u8]) {
    let (w, h) = (frame.width, frame.height);
    debug_assert_eq!(out.len(), w * h * 2);

    for row in 0..h {
        let y_row = &frame.y[row * w..row * w + w];
        let uv_row = row / 2;
        let uv_w = w / 2;
        let u_row = &frame.u[uv_row * uv_w..uv_row * uv_w + uv_w];
        let v_row = &frame.v[uv_row * uv_w..uv_row * uv_w + uv_w];

        let out_row = &mut out[row * w * 2..row * w * 2 + w * 2];
        for pair in 0..uv_w {
            let y0 = y_row[pair * 2];
            let y1 = y_row[pair * 2 + 1];
            let u = u_row[pair];
            let v = v_row[pair];
            let o = &mut out_row[pair * 4..pair * 4 + 4];
            o[0] = y0;
            o[1] = u;
            o[2] = y1;
            o[3] = v;
        }
    }
}

/// Converts the whole frame into `out` as packed RGBA8 (alpha 255). `out` must be
/// exactly `width * height * 4` bytes.
pub fn i420_to_rgba(frame: &YuvFrame, out: &mut [u8]) {
    convert_rgba(frame, 0, 0, frame.width, frame.height, 1, out);
}

/// Converts the `w`x`h` region whose top-left corner is (`x`, `y`) into `out` as
/// packed RGBA8 at native resolution -- the "zoom to region" view. The region must
/// lie within the frame; `out` must be exactly `w * h * 4` bytes.
pub fn i420_crop_to_rgba(frame: &YuvFrame, x: usize, y: usize, w: usize, h: usize, out: &mut [u8]) {
    assert!(
        x + w <= frame.width && y + h <= frame.height,
        "crop outside frame"
    );
    convert_rgba(frame, x, y, w, h, 1, out);
}

/// Size of the image [`i420_to_rgba_decimated`] produces for `frame` at `step`.
pub fn decimated_size(frame: &YuvFrame, step: usize) -> (usize, usize) {
    assert!(step >= 1);
    (frame.width.div_ceil(step), frame.height.div_ceil(step))
}

/// Converts every `step`-th pixel in each direction (nearest-neighbour downscale) into
/// `out` as packed RGBA8: a 4K frame at `step = 2` becomes a 1080p preview for a
/// quarter of the work. `out` must be exactly `w * h * 4` bytes for
/// [`decimated_size`]'s `(w, h)`.
pub fn i420_to_rgba_decimated(frame: &YuvFrame, step: usize, out: &mut [u8]) {
    let (w, h) = decimated_size(frame, step);
    convert_rgba(frame, 0, 0, w, h, step, out);
}

/// Writes `out_w` x `out_h` RGBA pixels, sampling the frame at
/// `(x0 + col * step, y0 + row * step)`. BT.601 limited range, the convention for
/// camera H.264 as produced by Android's encoder (and what YUYV consumers assume).
fn convert_rgba(
    frame: &YuvFrame,
    x0: usize,
    y0: usize,
    out_w: usize,
    out_h: usize,
    step: usize,
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), out_w * out_h * 4);
    let uv_w = frame.width / 2;
    for row in 0..out_h {
        let sy = y0 + row * step;
        let y_row = &frame.y[sy * frame.width..];
        let u_row = &frame.u[(sy / 2) * uv_w..];
        let v_row = &frame.v[(sy / 2) * uv_w..];
        let out_row = &mut out[row * out_w * 4..(row + 1) * out_w * 4];
        for (col, px) in out_row.chunks_exact_mut(4).enumerate() {
            let sx = x0 + col * step;
            let [r, g, b] = yuv_to_rgb(y_row[sx], u_row[sx / 2], v_row[sx / 2]);
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = 255;
        }
    }
}

/// BT.601 limited-range YCbCr -> RGB, fixed-point (x256):
/// R = 1.164 (Y-16) + 1.596 (V-128); G = 1.164 (Y-16) - 0.813 (V-128) - 0.391 (U-128);
/// B = 1.164 (Y-16) + 2.018 (U-128).
#[inline]
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let c = 298 * (y as i32 - 16) + 128;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let clamp = |x: i32| (x >> 8).clamp(0, 255) as u8;
    [
        clamp(c + 409 * e),
        clamp(c - 100 * d - 208 * e),
        clamp(c + 516 * d),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame whose luma is `x + 10 * y` (so positions are recognisable), flat chroma.
    fn gradient_frame(w: usize, h: usize) -> YuvFrame {
        let mut y = Vec::with_capacity(w * h);
        for row in 0..h {
            for col in 0..w {
                y.push((16 + col + 10 * row) as u8);
            }
        }
        YuvFrame {
            width: w,
            height: h,
            y,
            u: vec![128; (w / 2) * (h / 2)],
            v: vec![128; (w / 2) * (h / 2)],
        }
    }

    #[test]
    fn bt601_reference_colours() {
        assert_eq!(yuv_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(yuv_to_rgb(235, 128, 128), [255, 255, 255]);
        // Limited-range gray 128 maps to 130 in full range.
        assert_eq!(yuv_to_rgb(128, 128, 128), [130, 130, 130]);
        // BT.601 pure red is approximately (Y 81, U 90, V 240).
        let [r, g, b] = yuv_to_rgb(81, 90, 240);
        assert!(r >= 250 && g <= 5 && b <= 5, "got {r},{g},{b}");
        // Super-white / super-black input saturates instead of wrapping.
        assert_eq!(yuv_to_rgb(255, 128, 128), [255, 255, 255]);
        assert_eq!(yuv_to_rgb(0, 128, 128), [0, 0, 0]);
    }

    #[test]
    fn rgba_full_frame_alpha_and_gray() {
        let frame = gradient_frame(4, 2);
        let mut out = vec![0u8; 4 * 2 * 4];
        i420_to_rgba(&frame, &mut out);
        for px in out.chunks_exact(4) {
            assert_eq!(px[3], 255);
            assert_eq!(px[0], px[1]);
            assert_eq!(px[1], px[2]);
        }
        // Y=16 at (0,0) is black; luma rises along the row.
        assert_eq!(out[0], 0);
        assert!(out[4] > out[0] && out[8] > out[4]);
    }

    #[test]
    fn crop_samples_the_requested_region() {
        let frame = gradient_frame(8, 6);
        let mut full = vec![0u8; 8 * 6 * 4];
        i420_to_rgba(&frame, &mut full);
        let mut crop = vec![0u8; 3 * 2 * 4];
        i420_crop_to_rgba(&frame, 5, 3, 3, 2, &mut crop);
        for row in 0..2 {
            for col in 0..3 {
                let c = &crop[(row * 3 + col) * 4..][..4];
                let f = &full[((row + 3) * 8 + col + 5) * 4..][..4];
                assert_eq!(c, f, "crop pixel ({col},{row})");
            }
        }
    }

    #[test]
    fn decimation_picks_every_nth_pixel() {
        let frame = gradient_frame(8, 6);
        assert_eq!(decimated_size(&frame, 2), (4, 3));
        assert_eq!(decimated_size(&frame, 3), (3, 2));
        let mut full = vec![0u8; 8 * 6 * 4];
        i420_to_rgba(&frame, &mut full);
        let mut small = vec![0u8; 4 * 3 * 4];
        i420_to_rgba_decimated(&frame, 2, &mut small);
        for row in 0..3 {
            for col in 0..4 {
                let s = &small[(row * 4 + col) * 4..][..4];
                let f = &full[((row * 2) * 8 + col * 2) * 4..][..4];
                assert_eq!(s, f, "decimated pixel ({col},{row})");
            }
        }
    }

    #[test]
    fn converts_flat_gray_frame() {
        // 4x2 flat gray frame: Y=128, U=V=128 everywhere.
        let frame = YuvFrame {
            width: 4,
            height: 2,
            y: vec![128; 4 * 2],
            u: vec![128; 2],
            v: vec![128; 2],
        };
        let mut out = vec![0u8; 4 * 2 * 2];
        i420_to_yuyv(&frame, &mut out);
        assert!(out.iter().all(|&b| b == 128));
    }
}
