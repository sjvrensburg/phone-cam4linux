//! GLM-OCR on ONNX Runtime: the `onnx-community/GLM-OCR-ONNX` three-graph export
//! (vision encoder, token embeddings, merged decoder with KV cache) driven by hand.
//! Greedy decoding only -- the export has no sampling -- so one call is one reading.
//!
//! Preprocessing and the MRoPE position ids follow `oar-ocr-vl`'s Candle port
//! (Apache-2.0), the reference implementation known to reproduce the model's
//! output; the graph interface (including the undocumented `num_logits_to_keep`
//! input) was read off the export itself.

use crate::transcribe::{Token, TokenAlt};
use anyhow::{anyhow, bail, Context, Result};
use image::{imageops::FilterType, RgbImage};
use ndarray::{Array, ArrayD, IxDyn};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use std::borrow::Cow;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Which execution provider to run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    WebGpu,
    Cpu,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::WebGpu => "WebGPU",
            Device::Cpu => "CPU",
        }
    }
}

// ---------------------------------------------------------------------------
// Config files shipped with the model

#[derive(serde::Deserialize)]
struct TextConfig {
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    eos_token_id: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct VisionConfig {
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(serde::Deserialize)]
struct ModelConfig {
    text_config: TextConfig,
    vision_config: VisionConfig,
    image_token_id: u32,
}

#[derive(serde::Deserialize)]
struct SizeCfg {
    shortest_edge: u32,
    longest_edge: u32,
}

#[derive(serde::Deserialize)]
struct PreprocessorConfig {
    size: SizeCfg,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

// ---------------------------------------------------------------------------
// Image preprocessing

/// The model's "smart resize": dimensions become multiples of `factor` (patch size x
/// merge size), scaled so the pixel budget lands between `min_pixels` and
/// `max_pixels`.
fn smart_resize(
    h: u32,
    w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
    t: usize,
) -> (u32, u32) {
    let (mut height, mut width) = (h as f64, w as f64);
    let f = factor as f64;
    if height < f {
        width = (width * f / height).round();
        height = f;
    }
    if width < f {
        height = (height * f / width).round();
        width = f;
    }
    let mut h_bar = (height / f).round() * f;
    let mut w_bar = (width / f).round() * f;
    let volume = t as f64 * h_bar * w_bar;
    if volume > max_pixels as f64 {
        let beta = ((t as f64 * height * width) / max_pixels as f64).sqrt();
        h_bar = ((height / beta) / f).floor().max(1.0) * f;
        w_bar = ((width / beta) / f).floor().max(1.0) * f;
    } else if volume < min_pixels as f64 {
        let beta = (min_pixels as f64 / (t as f64 * height * width)).sqrt();
        h_bar = ((height * beta) / f).ceil() * f;
        w_bar = ((width * beta) / f).ceil() * f;
    }
    (h_bar as u32, w_bar as u32)
}

struct ImageInputs {
    /// (num_patches, 3 * temporal * patch * patch)
    pixel_values: ArrayD<f32>,
    grid_thw: (usize, usize, usize),
    num_image_tokens: usize,
}

/// `max_image_tokens` caps the pixel budget: one image token per
/// `(patch * merge)^2` pixels. The model's own default (9.6 MP, ~6000 tokens) is
/// more than the GPU path survives -- a whole 2992x2992 page lost the Vulkan device
/// -- and 2048 is what the workbench's llama-server ran with.
fn preprocess(img: &RgbImage, pp: &PreprocessorConfig, max_image_tokens: usize) -> ImageInputs {
    let factor = (pp.patch_size * pp.merge_size) as u32;
    let t = pp.temporal_patch_size;
    let token_budget_px = (t * max_image_tokens * (factor as usize).pow(2)) as u32;
    let (rh, rw) = smart_resize(
        img.height(),
        img.width(),
        factor,
        pp.size.shortest_edge,
        pp.size.longest_edge.min(token_budget_px),
        t,
    );
    let resized = image::imageops::resize(img, rw, rh, FilterType::CatmullRom);

    let (h, w) = (rh as usize, rw as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            chw[c * h * w + i] = (px.0[c] as f32 / 255.0 - pp.image_mean[c]) / pp.image_std[c];
        }
    }

    let ps = pp.patch_size;
    let ms = pp.merge_size;
    let grid_h = h / ps;
    let grid_w = w / ps;
    let patch_dim = 3 * t * ps * ps;
    let num_patches = grid_h * grid_w;
    let mut flat = Vec::with_capacity(num_patches * patch_dim);
    // Merge-window-major patch order; within a patch: channel, temporal, row, col. A
    // still image is the same frame repeated `t` times.
    for hb in 0..grid_h / ms {
        for wb in 0..grid_w / ms {
            for hm in 0..ms {
                for wm in 0..ms {
                    let (pr, pc) = (hb * ms + hm, wb * ms + wm);
                    for c in 0..3 {
                        for _tp in 0..t {
                            for ph in 0..ps {
                                let row = c * h * w + (pr * ps + ph) * w;
                                for pw in 0..ps {
                                    flat.push(chw[row + pc * ps + pw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ImageInputs {
        pixel_values: Array::from_shape_vec(IxDyn(&[num_patches, patch_dim]), flat)
            .expect("patch buffer matches its shape"),
        grid_thw: (1, grid_h, grid_w),
        num_image_tokens: num_patches / (ms * ms),
    }
}

// ---------------------------------------------------------------------------
// MRoPE position ids: (3, 1, T) -- temporal, height, width -- and the max position.

fn position_ids(
    ids: &[u32],
    grid: (usize, usize, usize),
    merge: usize,
    image_tok: u32,
) -> Result<(ArrayD<i64>, i64)> {
    let (gt, gh, gw) = (grid.0, grid.1 / merge, grid.2 / merge);
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut max_pos: i64 = -1;
    let mut i = 0;
    while i < ids.len() {
        let is_img = ids[i] == image_tok;
        let mut j = i;
        while j < ids.len() && (ids[j] == image_tok) == is_img {
            j += 1;
        }
        let st = max_pos + 1;
        if is_img {
            anyhow::ensure!(
                j - i == gt * gh * gw,
                "image token count {} != grid {}x{}x{}",
                j - i,
                gt,
                gh,
                gw
            );
            for t in 0..gt {
                for h in 0..gh {
                    for w in 0..gw {
                        pt.push(st + t as i64);
                        ph.push(st + h as i64);
                        pw.push(st + w as i64);
                    }
                }
            }
            max_pos = st + gt.max(gh).max(gw) as i64 - 1;
        } else {
            for k in 0..(j - i) as i64 {
                pt.push(st + k);
                ph.push(st + k);
                pw.push(st + k);
            }
            max_pos = st + (j - i) as i64 - 1;
        }
        i = j;
    }
    let n = ids.len();
    let mut data = pt;
    data.extend(ph);
    data.extend(pw);
    Ok((Array::from_shape_vec(IxDyn(&[3, 1, n]), data)?, max_pos))
}

// ---------------------------------------------------------------------------
// dtype-agnostic tensor helpers: the q4f16 graphs declare f16 for some I/O.

fn input_type(session: &Session, name: &str) -> Result<TensorElementType> {
    let outlet = session
        .inputs()
        .iter()
        .find(|o| o.name() == name)
        .ok_or_else(|| anyhow!("graph has no input {name:?}"))?;
    match outlet.dtype() {
        ValueType::Tensor { ty, .. } => Ok(*ty),
        other => bail!("input {name} is not a tensor: {other:?}"),
    }
}

fn float_value(arr: ArrayD<f32>, ty: TensorElementType) -> Result<DynValue> {
    Ok(match ty {
        TensorElementType::Float32 => Tensor::from_array(arr)?.into_dyn(),
        TensorElementType::Float16 => Tensor::from_array(arr.mapv(half::f16::from_f32))?.into_dyn(),
        other => bail!("unsupported float input type {other:?}"),
    })
}

fn extract_f32(v: &DynValue) -> Result<ArrayD<f32>> {
    match v.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => Ok(v.try_extract_array::<f32>()?.to_owned()),
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => Ok(v.try_extract_array::<half::f16>()?.mapv(|x| x.to_f32())),
        other => bail!("unexpected output type {other:?}"),
    }
}

fn input<'a>(
    name: impl Into<Cow<'a, str>>,
    value: DynValue,
) -> (Cow<'a, str>, SessionInputValue<'a>) {
    (name.into(), SessionInputValue::from(value))
}

// ---------------------------------------------------------------------------

/// One reading, greedy.
pub struct Generated {
    pub text: String,
    /// Stopped at `max_tokens` rather than at an end-of-sequence token.
    pub truncated: bool,
    /// One entry per output token (not including the end-of-sequence token itself),
    /// with its probability and runner-up alternates -- a softmax over the same
    /// logits the greedy step already reads, so this costs nothing extra to produce.
    pub tokens: Vec<Token>,
}

/// The chosen token id and probability from one decode step, plus up to 3 runner-up
/// ids the model gave lower probability.
struct StepToken {
    id: u32,
    prob: f32,
    alternates: Vec<(u32, f32)>,
}

pub struct Model {
    vision: Session,
    embed: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    cfg: ModelConfig,
    pp: PreprocessorConfig,
    kv_type: TensorElementType,
    embeds_type: TensorElementType,
    pixel_type: TensorElementType,
    device: Device,
    max_image_tokens: usize,
}

/// Where the decoder's KV cache lives between steps: the execution provider's own
/// memory. (`MemoryInfo` is not `Send`, so it is made per step; it is a tiny handle.)
fn kv_memory(device: Device) -> Result<MemoryInfo<'static>> {
    let where_ = match device {
        Device::WebGpu => AllocationDevice::WEBGPU_BUFFER,
        Device::Cpu => AllocationDevice::CPU,
    };
    Ok(MemoryInfo::new(
        where_,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

fn host_memory() -> Result<MemoryInfo<'static>> {
    Ok(MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

impl Model {
    /// Loads the three graphs from `dir/onnx/*_{variant}.onnx` on `device`. A WebGPU
    /// request fails here (not later) if the provider cannot be registered.
    pub fn load(
        dir: &Path,
        variant: &str,
        device: Device,
        max_image_tokens: usize,
    ) -> Result<Self> {
        let open = |name: &str| -> Result<Session> {
            let path = dir.join("onnx").join(format!("{name}_{variant}.onnx"));
            super::open_session(&path, device)
        };
        let vision = open("vision_encoder")?;
        let embed = open("embed_tokens")?;
        let decoder = open("decoder_model_merged")?;

        let cfg: ModelConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .context("config.json")?;
        let pp: PreprocessorConfig =
            serde_json::from_slice(&std::fs::read(dir.join("preprocessor_config.json"))?)
                .context("preprocessor_config.json")?;
        anyhow::ensure!(pp.patch_size == cfg.vision_config.patch_size);
        anyhow::ensure!(pp.merge_size == cfg.vision_config.spatial_merge_size);
        anyhow::ensure!(pp.temporal_patch_size == cfg.vision_config.temporal_patch_size);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let kv_type = input_type(&decoder, "past_key_values.0.key")?;
        let embeds_type = input_type(&decoder, "inputs_embeds")?;
        let pixel_type = input_type(&vision, "pixel_values")?;
        Ok(Self {
            vision,
            embed,
            decoder,
            tokenizer,
            cfg,
            pp,
            kv_type,
            embeds_type,
            pixel_type,
            device,
            max_image_tokens: max_image_tokens.max(64),
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn generate(
        &mut self,
        img: &RgbImage,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<Generated> {
        let t0 = Instant::now();
        let inputs = preprocess(img, &self.pp, self.max_image_tokens);

        // Vision encoder.
        let (gt, gh, gw) = inputs.grid_thw;
        let grid = Array::from_shape_vec(IxDyn(&[1, 3]), vec![gt as i64, gh as i64, gw as i64])?;
        let out = self.vision.run(vec![
            input(
                "pixel_values",
                float_value(inputs.pixel_values, self.pixel_type)?,
            ),
            input("image_grid_thw", Tensor::from_array(grid)?.into_dyn()),
        ])?;
        let image_features = extract_f32(&out["image_features"])?;
        drop(out);
        let t_vision = t0.elapsed();

        // Prompt, with one image token per merged patch group.
        let image_tokens = "<|image|>".repeat(inputs.num_image_tokens);
        let text = format!(
            "[gMASK]<sop><|user|>\n<|begin_of_image|>{image_tokens}<|end_of_image|>{prompt}<|assistant|>\n"
        );
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let n = ids.len();
        let image_token_id = self.cfg.image_token_id;
        let n_img = ids.iter().filter(|&&i| i == image_token_id).count();
        anyhow::ensure!(
            n_img == inputs.num_image_tokens,
            "tokenizer produced {n_img} image tokens, expected {}",
            inputs.num_image_tokens
        );

        // Text embeddings with the image features spliced in at the image tokens.
        let mut embeds = self.embed_ids(&ids)?;
        let feats = image_features
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| anyhow!("image_features is not 2-D"))?;
        anyhow::ensure!(
            feats.shape()[0] == n_img,
            "image_features rows != image tokens"
        );
        let mut row = 0;
        for (i, &id) in ids.iter().enumerate() {
            if id == image_token_id {
                embeds
                    .index_axis_mut(ndarray::Axis(1), i)
                    .assign(&feats.row(row));
                row += 1;
            }
        }
        let (pos, max_pos) =
            position_ids(&ids, inputs.grid_thw, self.pp.merge_size, image_token_id)?;
        let rope_delta = max_pos + 1 - n as i64;

        // Prefill, then greedy decode one token at a time.
        let layers = self.cfg.text_config.num_hidden_layers;
        let eos = self.cfg.text_config.eos_token_id.clone();
        let empty = || -> Result<DynValue> {
            float_value(
                Array::zeros(IxDyn(&[
                    1,
                    self.cfg.text_config.num_key_value_heads,
                    0,
                    self.cfg.text_config.head_dim,
                ])),
                self.kv_type,
            )
        };
        let mut kv: Vec<DynValue> = (0..2 * layers).map(|_| empty()).collect::<Result<_>>()?;
        let mut past = 0usize;
        let (mut next, new_kv) = self.step(&embeds, &pos, past, &kv)?;
        kv = new_kv;
        past += n;
        let t_prefill = t0.elapsed();

        let mut generated: Vec<u32> = Vec::new();
        let mut steps: Vec<StepToken> = Vec::new();
        let mut truncated = false;
        loop {
            if eos.contains(&next.id) {
                break;
            }
            generated.push(next.id);
            if generated.len() >= max_tokens {
                truncated = true;
                break;
            }
            let e = self.embed_ids(&[next.id])?;
            let p = (past as i64) + rope_delta;
            let pos1 = Array::from_shape_vec(IxDyn(&[3, 1, 1]), vec![p, p, p])?;
            let (tok, new_kv) = self.step(&e, &pos1, past, &kv)?;
            kv = new_kv;
            past += 1;
            steps.push(next);
            next = tok;
        }
        let total = t0.elapsed();
        log::info!(
            "GLM-OCR ({}): {} image tokens, {} out tokens; vision {:.2}s, prefill {:.2}s, decode {:.2}s",
            self.device.name(),
            n_img,
            generated.len(),
            t_vision.as_secs_f64(),
            (t_prefill - t_vision).as_secs_f64(),
            (total - t_prefill).as_secs_f64()
        );
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        let tokens = steps
            .into_iter()
            .map(|s| self.token_with_text(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(Generated {
            text: text.trim().to_string(),
            truncated,
            tokens,
        })
    }

    /// Decodes a step's id and its alternates' ids to text. A lone id can decode
    /// slightly differently than it would inside the full sequence (BPE merges
    /// spanning a boundary); [`Reading::tokens_if_valid`] catches the rare case
    /// where this drifts from the whole-text decode.
    fn token_with_text(&self, step: StepToken) -> Result<Token> {
        let decode_one = |id: u32| -> Result<String> {
            self.tokenizer
                .decode(&[id], true)
                .map_err(|e| anyhow!("decode: {e}"))
        };
        Ok(Token {
            text: decode_one(step.id)?,
            prob: step.prob,
            alternates: step
                .alternates
                .into_iter()
                .map(|(id, prob)| -> Result<TokenAlt> {
                    Ok(TokenAlt {
                        text: decode_one(id)?,
                        prob,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }

    fn embed_ids(&mut self, ids: &[u32]) -> Result<ArrayD<f32>> {
        let arr = Array::from_shape_vec(
            IxDyn(&[1, ids.len()]),
            ids.iter().map(|&i| i as i64).collect(),
        )?;
        let out = self.embed.run(vec![input(
            "input_ids",
            Tensor::from_array(arr)?.into_dyn(),
        )])?;
        extract_f32(&out["inputs_embeds"])
    }

    /// One decoder call over `embeds` (1, T, hidden) with `past` cached positions.
    /// Returns the argmax token of the last position and the grown KV cache.
    ///
    /// The cache stays where the execution provider put it: outputs are bound to
    /// device memory and handed back as the next step's inputs, so a decode step
    /// moves only the new token in and the last logits out. Round-tripping the whole
    /// cache through host memory cost ~150 MB per token at page-sized contexts.
    fn step(
        &mut self,
        embeds: &ArrayD<f32>,
        pos: &ArrayD<i64>,
        past: usize,
        kv: &[DynValue],
    ) -> Result<(StepToken, Vec<DynValue>)> {
        let t = embeds.shape()[1];
        let mask: ArrayD<i64> = Array::ones(IxDyn(&[1, past + t]));
        let embeds = float_value(embeds.clone(), self.embeds_type)?;
        let mask = Tensor::from_array(mask)?.into_dyn();
        let pos = Tensor::from_array(pos.clone())?.into_dyn();
        // Only the last position's logits are wanted (scalar int64).
        let keep = Tensor::from_array(Array::from_elem(IxDyn(&[]), 1i64))?.into_dyn();

        let layers = self.cfg.text_config.num_hidden_layers;
        let mut binding = self.decoder.create_binding()?;
        binding.bind_input("inputs_embeds", &embeds)?;
        binding.bind_input("attention_mask", &mask)?;
        binding.bind_input("position_ids", &pos)?;
        binding.bind_input("num_logits_to_keep", &keep)?;
        for l in 0..layers {
            binding.bind_input(format!("past_key_values.{l}.key"), &kv[2 * l])?;
            binding.bind_input(format!("past_key_values.{l}.value"), &kv[2 * l + 1])?;
        }
        let (host, device) = (host_memory()?, kv_memory(self.device)?);
        binding.bind_output_to_device("logits", &host)?;
        for l in 0..layers {
            binding.bind_output_to_device(format!("present.{l}.key"), &device)?;
            binding.bind_output_to_device(format!("present.{l}.value"), &device)?;
        }
        let mut out = self.decoder.run_binding(&binding)?;
        let logits = extract_f32(&out["logits"])?;
        let last = logits.index_axis(ndarray::Axis(1), logits.shape()[1] - 1);
        let last = last.index_axis(ndarray::Axis(0), 0);
        // Top-4 by logit (the chosen token plus 3 runner-ups) found in one linear
        // pass, then turned into probabilities with a second pass for the softmax
        // normaliser -- avoids sorting the whole vocabulary just for the top few.
        let max_logit = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut top: [(usize, f32); 4] = [(0, f32::NEG_INFINITY); 4];
        for (i, &v) in last.iter().enumerate() {
            if v > top[3].1 {
                top[3] = (i, v);
                let mut j = 3;
                while j > 0 && top[j].1 > top[j - 1].1 {
                    top.swap(j, j - 1);
                    j -= 1;
                }
            }
        }
        let sum_exp: f32 = last.iter().map(|&v| (v - max_logit).exp()).sum();
        let prob_of = |logit: f32| (logit - max_logit).exp() / sum_exp;
        let token = StepToken {
            id: top[0].0 as u32,
            prob: prob_of(top[0].1),
            alternates: top[1..]
                .iter()
                .map(|&(i, v)| (i as u32, prob_of(v)))
                .collect(),
        };
        let mut new_kv = Vec::with_capacity(2 * layers);
        for l in 0..layers {
            for part in ["key", "value"] {
                let name = format!("present.{l}.{part}");
                new_kv.push(
                    out.remove(&name)
                        .ok_or_else(|| anyhow!("decoder produced no {name}"))?,
                );
            }
        }
        Ok((token, new_kv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_the_patch_grid() {
        // 28 = patch 14 x merge 2; a 900x260 crop becomes 896x252.
        assert_eq!(smart_resize(260, 900, 28, 12544, 9_633_792, 2), (252, 896));
        // Tiny inputs are scaled up to the minimum pixel budget.
        let (h, w) = smart_resize(10, 40, 28, 12544, 9_633_792, 2);
        assert!(h % 28 == 0 && w % 28 == 0 && 2 * h * w >= 12544);
    }

    #[test]
    fn position_ids_give_image_tokens_a_2d_grid() {
        // text, 4 image tokens (grid 1 x 4 x 2 with merge 2 -> 1x2x1... use a 2x2 group)
        let img = 7u32;
        let ids = [1, 2, img, img, img, img, 3];
        // grid (1, 4, 4) with merge 2 -> llm grid 1 x 2 x 2 = 4 tokens
        let (pos, max_pos) = position_ids(&ids, (1, 4, 4), 2, img).unwrap();
        assert_eq!(pos.shape(), &[3, 1, 7]);
        let row = |k: usize| -> Vec<i64> { (0..7).map(|i| pos[[k, 0, i]]).collect() };
        // temporal: text 0,1; image all at 2; then text continues after the group's
        // max extent (2 + 1).
        assert_eq!(row(0), vec![0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(row(1), vec![0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(row(2), vec![0, 1, 2, 3, 2, 3, 4]);
        assert_eq!(max_pos, 4);
        assert!(position_ids(&[img, img, img], (1, 4, 4), 2, img).is_err());
    }
}
