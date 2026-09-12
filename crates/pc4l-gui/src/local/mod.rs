//! The bundled models: GLM-OCR as a [`Transcriber`] and PP-DocLayoutV3 as the block
//! detector, both on ONNX Runtime. [`models`] finds or downloads their files.

mod glmocr;
pub mod layout;
pub mod models;

use crate::transcribe::{Mode, Reading, Transcriber, Transcription};
use anyhow::{anyhow, bail, Context, Result};
pub use glmocr::Device;
use glmocr::Model;
use ort::environment::Environment;
use ort::ep::{ExecutionProviderDispatch, WebGPU, CPU};
use ort::session::Session;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const VARIANT: &str = "q4f16";

/// The process-wide ONNX Runtime environment: `ort` allows exactly one, and both
/// models need it.
fn environment() -> Result<Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().build().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow!("initialising ONNX Runtime: {e}"))
}

/// One model at a time on the runtime. The WebGPU provider is not safe to `run`
/// from two threads at once, even on two sessions: a block detection overlapping a
/// read segfaulted inside a TopK kernel. This is
/// <https://github.com/microsoft/onnxruntime/issues/32561> (open; a fix is in
/// progress as PR 29851) -- sequential runs are fine, concurrent ones are a silent
/// SIGSEGV. Held around every `run` and, to be safe, every load; drop it once the
/// pinned ONNX Runtime carries the fix.
pub static RUNTIME: Mutex<()> = Mutex::new(());

/// Takes [`RUNTIME`], surviving a panic elsewhere.
pub fn runtime_turn() -> std::sync::MutexGuard<'static, ()> {
    RUNTIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// Opens one graph on `device`. A WebGPU request fails here (not later) if the
/// provider cannot be registered.
fn open_session(path: &Path, device: Device) -> Result<Session> {
    let _turn = runtime_turn();
    let providers: Vec<ExecutionProviderDispatch> = match device {
        Device::WebGpu => vec![WebGPU::default().build().error_on_failure()],
        Device::Cpu => vec![CPU::default().build()],
    };
    let t = Instant::now();
    let session = Session::builder(&environment()?)?
        .with_execution_providers(providers)
        .map_err(|e| anyhow!("registering the {} execution provider: {e}", device.name()))?
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))?;
    log::debug!(
        "loaded {} in {:.2}s",
        path.display(),
        t.elapsed().as_secs_f64()
    );
    Ok(session)
}

/// The devices to try for a preference, in order.
fn attempts(device: DevicePref) -> &'static [Device] {
    match device {
        DevicePref::Auto => &[Device::WebGpu, Device::Cpu],
        DevicePref::Webgpu => &[Device::WebGpu],
        DevicePref::Cpu => &[Device::Cpu],
    }
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
                let result = models::GLM_OCR
                    .ensure(&set)
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
    let mut last = None;
    for &d in attempts(device) {
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
        let out = {
            let _turn = runtime_turn();
            model.generate(&img, mode.prompt(), self.max_tokens)?
        };
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
