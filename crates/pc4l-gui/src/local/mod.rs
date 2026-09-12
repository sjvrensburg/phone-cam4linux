//! The bundled model: GLM-OCR as a [`Transcriber`], and where its files live.
//!
//! The ~650 MB of graphs are not embedded in the binary (that would relink them on
//! every build and put them in git); they are looked up, in order, in
//! `$PC4L_MODEL_DIR`, a `models/<name>/` directory next to the executable (how a
//! release tarball ships them), and `$XDG_CACHE_HOME/pc4l/models/<name>/`. If none
//! has them they are downloaded into the cache from a pinned Hugging Face revision,
//! each file checked against the sha256 recorded here before it is used.

mod glmocr;

use crate::transcribe::{Mode, Reading, Transcriber, Transcription};
use anyhow::{anyhow, bail, Context, Result};
use glmocr::{Device, Model};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// `onnx-community/GLM-OCR-ONNX`, q4f16 variant, at a pinned revision.
const MODEL_NAME: &str = "glm-ocr-onnx-q4f16";
const VARIANT: &str = "q4f16";
const REPO_URL: &str =
    "https://huggingface.co/onnx-community/GLM-OCR-ONNX/resolve/aea46198f09e3aa2b63422dd234f1cc66afffe52";

struct ModelFile {
    path: &'static str,
    size: u64,
    sha256: &'static str,
}

const FILES: &[ModelFile] = &[
    ModelFile {
        path: "config.json",
        size: 2022,
        sha256: "8bf81b89d42ae98917084dfeaca5a1dc20ac20d6145f088ec2aac8840c1e4b9f",
    },
    ModelFile {
        path: "preprocessor_config.json",
        size: 366,
        sha256: "0d4b3cf5190e6b5b53ac52b1880b111163888bfc975db396561ee78887942fe4",
    },
    ModelFile {
        path: "tokenizer.json",
        size: 5_420_559,
        sha256: "c3e229a66a06267194e62055f70cf5580af83b3582f46928fefee8cb9618f499",
    },
    ModelFile {
        path: "onnx/vision_encoder_q4f16.onnx",
        size: 474_559,
        sha256: "083d41123710a5ceb725d3f48fc8170f3e85d9289e7530454c86c1961254055b",
    },
    ModelFile {
        path: "onnx/vision_encoder_q4f16.onnx_data",
        size: 262_272_000,
        sha256: "b09ae1abca6bd2d229c63bc2dc4d09bba272c0627b270804f66267b0bac17ee1",
    },
    ModelFile {
        path: "onnx/embed_tokens_q4f16.onnx",
        size: 1060,
        sha256: "b56ef40c21191aa1fdd4e7251679347ed45dd8473605e9539caeed6b781e41f7",
    },
    ModelFile {
        path: "onnx/embed_tokens_q4f16.onnx_data",
        size: 52_740_096,
        sha256: "4b82f4062c1cf676e29126c6826c93d262872c1efad8e24fc78476be4245a966",
    },
    ModelFile {
        path: "onnx/decoder_model_merged_q4f16.onnx",
        size: 377_830,
        sha256: "6510318b0b3f1458c38a8678ebb2ca6868e83753cef92d72da8cb926aa82e0b8",
    },
    ModelFile {
        path: "onnx/decoder_model_merged_q4f16.onnx_data",
        size: 336_844_800,
        sha256: "82af470f508000dcc3914c36d102f60c39b12f4be0f016b333e5e78b2e865bc8",
    },
];

/// Sum of all file sizes: the download, and the "is it complete" test.
fn total_size() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

/// Every file present at its recorded size. Content is verified at download
/// time; hashing 650 MB on each start is not worth the seconds.
fn is_complete(dir: &Path) -> bool {
    FILES
        .iter()
        .all(|f| std::fs::metadata(dir.join(f.path)).is_ok_and(|m| m.len() == f.size))
}

/// The directories searched for the model, in order.
pub fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("PC4L_MODEL_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        dirs.push(exe_dir.join("models").join(MODEL_NAME));
    }
    dirs.push(cache_dir());
    dirs
}

/// Where a download lands.
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_default();
    base.join("pc4l").join("models").join(MODEL_NAME)
}

/// Finds the model, downloading it into the cache if no candidate has it.
/// `progress` is told what is happening, for the window.
fn ensure_model(progress: &dyn Fn(String)) -> Result<PathBuf> {
    for dir in candidate_dirs() {
        if is_complete(&dir) {
            log::info!("model found at {}", dir.display());
            return Ok(dir);
        }
    }
    let dir = cache_dir();
    download_into(&dir, progress)?;
    Ok(dir)
}

/// The directory name the model lives under (`models/<this>/` next to the binary).
pub fn model_dir_name() -> &'static str {
    MODEL_NAME
}

/// Downloads the model into `dir` (files already present at the right size are
/// kept), verifying every file's SHA-256. `pc4l-gui --fetch-model DIR` for scripts
/// and release packaging.
pub fn download_into(dir: &Path, progress: &dyn Fn(String)) -> Result<()> {
    log::info!(
        "downloading {} ({} MB) to {}",
        MODEL_NAME,
        total_size() / 1_000_000,
        dir.display()
    );
    let agent = ureq::Agent::config_builder()
        .timeout_global(None)
        .build()
        .new_agent();
    let mut done: u64 = 0;
    let total = total_size();
    for file in FILES {
        let target = dir.join(file.path);
        if std::fs::metadata(&target).is_ok_and(|m| m.len() == file.size) {
            done += file.size;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = format!("{REPO_URL}/{}", file.path);
        let part = dir.join(format!("{}.part", file.path));
        // Anonymous downloads are rate-limited (CI runners share addresses); a
        // Hugging Face token lifts that. The files themselves are public.
        let mut request = agent.get(&url);
        let token = std::env::var_os("HF_TOKEN").filter(|t| !t.is_empty());
        if let Some(token) = &token {
            request = request.header(
                "authorization",
                format!("Bearer {}", token.to_string_lossy()),
            );
        }
        // An invalid token is refused (401) even for public files, so say when
        // one was sent: a stale HF_TOKEN in the environment is the likely cause.
        let mut response = request.call().with_context(|| {
            if token.is_some() {
                format!(
                    "downloading {} (with HF_TOKEN from the environment)",
                    file.path
                )
            } else {
                format!("downloading {}", file.path)
            }
        })?;
        let mut reader = response.body_mut().as_reader();
        let mut out = std::fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut written: u64 = 0;
        let mut last_report = Instant::now();
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            written += n as u64;
            if last_report.elapsed().as_millis() > 200 {
                progress(format!(
                    "downloading model {}%",
                    (done + written) * 100 / total
                ));
                last_report = Instant::now();
            }
        }
        drop(out);
        let digest = format!("{:x}", hasher.finalize());
        if written != file.size || digest != file.sha256 {
            let _ = std::fs::remove_file(&part);
            bail!(
                "{} downloaded wrong: {written} bytes, sha256 {digest} (expected {} bytes, {})",
                file.path,
                file.size,
                file.sha256
            );
        }
        std::fs::rename(&part, &target)?;
        done += file.size;
    }
    Ok(())
}

// ---------------------------------------------------------------------------

/// Which device to try.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DevicePref {
    /// WebGPU, falling back to CPU if the provider cannot be set up.
    #[default]
    Auto,
    Webgpu,
    Cpu,
}

enum State {
    /// Downloading or loading; the string is shown in the window.
    Preparing(String),
    Ready(Box<Model>),
    Failed(String),
}

/// The bundled GLM-OCR. Preparation (download + load) starts on construction, on a
/// thread; reads are refused with the current status until it is done.
pub struct LocalBackend {
    name: String,
    max_tokens: usize,
    state: Arc<Mutex<State>>,
}

impl LocalBackend {
    pub fn new(name: String, device: DevicePref, max_tokens: u32, max_image_tokens: u32) -> Self {
        let state = Arc::new(Mutex::new(State::Preparing("locating model".into())));
        let worker_state = Arc::clone(&state);
        std::thread::Builder::new()
            .name("pc4l-model".into())
            .spawn(move || {
                let set = |s: String| {
                    *worker_state.lock().unwrap() = State::Preparing(s);
                };
                let result = ensure_model(&set)
                    .and_then(|dir| load(&dir, device, max_image_tokens as usize, &set));
                *worker_state.lock().unwrap() = match result {
                    Ok(model) => {
                        log::info!("GLM-OCR ready on {}", model.device().name());
                        State::Ready(Box::new(model))
                    }
                    Err(e) => {
                        log::error!("local model unavailable: {e:#}");
                        State::Failed(format!("{e:#}"))
                    }
                };
            })
            .expect("spawning model thread");
        Self {
            name,
            max_tokens: max_tokens as usize,
            state,
        }
    }
}

fn load(
    dir: &Path,
    device: DevicePref,
    max_image_tokens: usize,
    progress: &dyn Fn(String),
) -> Result<Model> {
    let attempts: &[Device] = match device {
        DevicePref::Auto => &[Device::WebGpu, Device::Cpu],
        DevicePref::Webgpu => &[Device::WebGpu],
        DevicePref::Cpu => &[Device::Cpu],
    };
    let mut last = None;
    for &d in attempts {
        progress(format!("loading model on {}", d.name()));
        match Model::load(dir, VARIANT, d, max_image_tokens) {
            Ok(m) => return Ok(m),
            Err(e) => {
                log::warn!("GLM-OCR on {}: {e:#}", d.name());
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("no device to try")))
}

impl Transcriber for LocalBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn status(&self) -> Option<String> {
        match &*self.state.lock().unwrap() {
            State::Preparing(s) => Some(s.clone()),
            State::Ready(m) => Some(format!("ready on {}", m.device().name())),
            State::Failed(e) => Some(format!("unavailable: {e}")),
        }
    }

    fn read(&self, png: &[u8], mode: Mode, _capture_px: (u32, u32)) -> Result<Transcription> {
        let started = Instant::now();
        let img = image::load_from_memory(png)
            .context("decoding the crop")?
            .to_rgb8();
        let mut state = self.state.lock().unwrap();
        let model = match &mut *state {
            State::Ready(m) => m,
            State::Preparing(s) => bail!("model not ready yet: {s}"),
            State::Failed(e) => bail!("model unavailable: {e}"),
        };
        let out = model.generate(&img, mode.prompt(), self.max_tokens)?;
        let (readings, silent) = if out.text.is_empty() {
            (Vec::new(), 1)
        } else {
            (
                vec![Reading {
                    text: out.text,
                    count: 1,
                    truncated: out.truncated,
                }],
                0,
            )
        };
        Ok(Transcription {
            backend: self.name.clone(),
            readings,
            silent,
            samples: 1,
            elapsed: started.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_consistent() {
        assert_eq!(FILES.len(), 9);
        assert!(FILES.iter().all(|f| f.sha256.len() == 64));
        assert_eq!(total_size(), 658_133_292);
        assert!(!is_complete(Path::new("/nonexistent")));
    }
}
