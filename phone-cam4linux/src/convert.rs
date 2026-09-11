//! I420 (planar YUV 4:2:0) -> YUYV422 (packed) conversion.
//!
//! YUYV is used for the V4L2 sink because it's the most broadly compatible "raw
//! webcam" format for consumers (browsers, most webcam apps) -- the real scrcpy
//! V4L2 sink does the same conversion for the same reason.

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

#[cfg(test)]
mod tests {
    use super::*;

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
