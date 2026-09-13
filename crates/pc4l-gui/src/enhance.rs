//! Ink enhancement for faint pencil and coloured pens (pink/purple): flatten uneven
//! paper lighting, then stretch contrast; `Ink` mode additionally desaturates to
//! whichever channel best carries the ink (green for pink/purple, red for blue,
//! any for pencil) before flattening.
//!
//! There is deliberately no auto-tuning here. Ambient light, phone angle and phone
//! distance change from shot to shot, so a fixed set of constants tuned in advance
//! cannot generalise; every parameter below is meant to be driven live by the user
//! while looking at the crop panel, not calibrated once and forgotten.

use image::RgbaImage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnhanceMode {
    /// No processing; the crop panel and the model see the raw pixels.
    Off,
    /// Flatten and stretch each colour channel independently; keeps colour.
    Auto,
    /// Desaturate to the ink channel first, then flatten and stretch; output is
    /// greyscale, meant for faint or coloured handwriting.
    Ink,
}

impl EnhanceMode {
    pub fn cycle(self) -> Self {
        match self {
            EnhanceMode::Off => EnhanceMode::Auto,
            EnhanceMode::Auto => EnhanceMode::Ink,
            EnhanceMode::Ink => EnhanceMode::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EnhanceMode::Off => "off",
            EnhanceMode::Auto => "auto",
            EnhanceMode::Ink => "ink",
        }
    }
}

/// Which channel carries the ink in `Ink` mode. `Min` (the darkest of r/g/b at each
/// pixel) suits pencil, where all three drop together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    Min,
    Red,
    Green,
    Blue,
}

impl Channel {
    pub const ALL: [Channel; 4] = [Channel::Min, Channel::Red, Channel::Green, Channel::Blue];

    pub fn label(self) -> &'static str {
        match self {
            Channel::Min => "darkest (pencil)",
            Channel::Red => "red (blue ink)",
            Channel::Green => "green (pink/purple ink)",
            Channel::Blue => "blue",
        }
    }
}

/// Every knob here is meant to be adjusted live -- see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhanceConfig {
    pub mode: EnhanceMode,
    pub channel: Channel,
    /// 0 = original image, 1 = fully flattened and stretched.
    pub strength: f32,
    /// Percentile mapped to black, 0..100.
    pub black_point: f32,
    /// Percentile mapped to white, 0..100.
    pub white_point: f32,
    pub gamma: f32,
}

impl Default for EnhanceConfig {
    fn default() -> Self {
        Self {
            mode: EnhanceMode::Off,
            channel: Channel::Min,
            strength: 0.7,
            black_point: 1.0,
            white_point: 60.0,
            gamma: 1.0,
        }
    }
}

/// Applies `cfg` to `img`, returning a new image. A no-op (cloning `img`) when the
/// mode is `Off` or the strength is zero.
pub fn apply(img: &RgbaImage, cfg: &EnhanceConfig) -> RgbaImage {
    if cfg.mode == EnhanceMode::Off || cfg.strength <= 0.0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }
    let mut out = img.clone();
    match cfg.mode {
        EnhanceMode::Off => {}
        EnhanceMode::Auto => {
            for channel in 0..3 {
                let vals: Vec<f32> = img.pixels().map(|p| p.0[channel] as f32).collect();
                let processed = process(&vals, w, h, cfg);
                for (p, v) in out.pixels_mut().zip(processed) {
                    p.0[channel] = v as u8;
                }
            }
        }
        EnhanceMode::Ink => {
            let vals: Vec<f32> = img.pixels().map(|p| ink_value(p.0, cfg.channel)).collect();
            let processed = process(&vals, w, h, cfg);
            for (p, v) in out.pixels_mut().zip(processed) {
                let v = v as u8;
                p.0[0] = v;
                p.0[1] = v;
                p.0[2] = v;
            }
        }
    }
    out
}

fn ink_value(rgba: [u8; 4], channel: Channel) -> f32 {
    let [r, g, b, _] = rgba;
    match channel {
        Channel::Min => r.min(g).min(b) as f32,
        Channel::Red => r as f32,
        Channel::Green => g as f32,
        Channel::Blue => b as f32,
    }
}

/// Background-flatten then contrast-stretch one channel's values, blended with the
/// original by `cfg.strength`.
fn process(vals: &[f32], w: u32, h: u32, cfg: &EnhanceConfig) -> Vec<f32> {
    let flattened = flatten(vals, w, h);
    let stretched = stretch(&flattened, cfg.black_point, cfg.white_point, cfg.gamma);
    vals.iter()
        .zip(stretched)
        .map(|(orig, s)| (orig * (1.0 - cfg.strength) + s * cfg.strength).clamp(0.0, 255.0))
        .collect()
}

/// Estimates an uneven background by averaging into a coarse grid (about 1/20th of
/// the shorter edge per cell) and dividing it out, so a shadowed or unevenly lit
/// page reads as flat.
fn flatten(vals: &[f32], w: u32, h: u32) -> Vec<f32> {
    let cell = (w.min(h) / 20).max(1);
    let gw = w.div_ceil(cell).max(1);
    let gh = h.div_ceil(cell).max(1);
    let mut sums = vec![0f64; (gw * gh) as usize];
    let mut counts = vec![0u32; (gw * gh) as usize];
    for y in 0..h {
        let gy = y / cell;
        for x in 0..w {
            let gx = x / cell;
            let idx = (gy * gw + gx) as usize;
            sums[idx] += vals[(y * w + x) as usize] as f64;
            counts[idx] += 1;
        }
    }
    let means: Vec<f32> = sums
        .iter()
        .zip(&counts)
        .map(|(sum, count)| {
            if *count == 0 {
                255.0
            } else {
                (*sum / *count as f64) as f32
            }
        })
        .collect();
    vals.iter()
        .enumerate()
        .map(|(i, v)| {
            let (x, y) = (i as u32 % w, i as u32 / w);
            let bg = means[((y / cell) * gw + x / cell) as usize].max(1.0);
            (v / bg * 200.0).clamp(0.0, 255.0)
        })
        .collect()
}

/// Linearly maps the `black_point`/`white_point` percentiles to 0/255, with an
/// optional gamma on the result.
fn stretch(vals: &[f32], black_point: f32, white_point: f32, gamma: f32) -> Vec<f32> {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let black = percentile(&sorted, black_point);
    let white = percentile(&sorted, white_point).max(black + 1.0);
    vals.iter()
        .map(|v| {
            let t = ((v - black) / (white - black)).clamp(0.0, 1.0);
            let t = if gamma != 1.0 { t.powf(1.0 / gamma) } else { t };
            t * 255.0
        })
        .collect()
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> RgbaImage {
        RgbaImage::from_fn(w, h, |_, _| image::Rgba([rgb[0], rgb[1], rgb[2], 255]))
    }

    #[test]
    fn off_is_a_no_op() {
        let img = solid(8, 8, [10, 20, 30]);
        let cfg = EnhanceConfig {
            mode: EnhanceMode::Off,
            ..EnhanceConfig::default()
        };
        assert_eq!(apply(&img, &cfg), img);
    }

    #[test]
    fn zero_strength_is_a_no_op() {
        let img = solid(8, 8, [10, 20, 30]);
        let cfg = EnhanceConfig {
            mode: EnhanceMode::Auto,
            strength: 0.0,
            ..EnhanceConfig::default()
        };
        assert_eq!(apply(&img, &cfg), img);
    }

    #[test]
    fn ink_mode_outputs_greyscale() {
        let img = solid(20, 20, [200, 40, 200]);
        let cfg = EnhanceConfig {
            mode: EnhanceMode::Ink,
            channel: Channel::Green,
            ..EnhanceConfig::default()
        };
        let out = apply(&img, &cfg);
        for p in out.pixels() {
            assert_eq!(p.0[0], p.0[1]);
            assert_eq!(p.0[1], p.0[2]);
        }
    }

    #[test]
    fn stretch_expands_a_flat_band_to_full_range() {
        // A checkerboard of 100 and 150 should stretch to roughly 0 and 255 once
        // black/white points bracket it tightly.
        let vals: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 100.0 } else { 150.0 })
            .collect();
        let out = stretch(&vals, 5.0, 95.0, 1.0);
        let lo = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(lo < 10.0, "low end should stretch near black: {lo}");
        assert!(hi > 245.0, "high end should stretch near white: {hi}");
    }

    #[test]
    fn cycle_visits_all_modes() {
        assert_eq!(EnhanceMode::Off.cycle(), EnhanceMode::Auto);
        assert_eq!(EnhanceMode::Auto.cycle(), EnhanceMode::Ink);
        assert_eq!(EnhanceMode::Ink.cycle(), EnhanceMode::Off);
    }
}
