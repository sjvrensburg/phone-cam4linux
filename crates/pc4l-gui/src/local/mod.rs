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
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Set once the GPU device has been lost (a driver reset -- `VK_ERROR_DEVICE_LOST`
/// -- which a whole page at too large an image budget provokes). The WebGPU
/// provider does not recover the device, so from then on everything loads on the
/// CPU; a restart gets the GPU back.
pub static GPU_LOST: AtomicBool = AtomicBool::new(false);

/// Whether `e` is the GPU being lost. Records it if so.
pub(super) fn note_gpu_loss(e: &anyhow::Error) -> bool {
    let text = format!("{e:#}");
    let lost = text.contains("DEVICE_LOST")
        || (text.contains("lost") && (text.contains("Device") || text.contains("device")));
    if lost {
        GPU_LOST.store(true, Ordering::SeqCst);
        log::error!("the GPU device was lost; models reload on the CPU: {text}");
    }
    lost
}

/// The devices to try for a preference, in order -- the CPU alone once the GPU
/// has been lost.
fn attempts(device: DevicePref) -> &'static [Device] {
    if GPU_LOST.load(Ordering::SeqCst) {
        return &[Device::Cpu];
    }
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
/// thread; reads are refused with the current status until it is done. A lost GPU
/// mid-read reloads it on the CPU.
pub struct LocalBackend {
    name: String,
    max_tokens: usize,
    max_image_tokens: usize,
    device: DevicePref,
    state: Arc<Mutex<State>>,
}

impl LocalBackend {
    pub fn new(name: String, device: DevicePref, max_tokens: u32, max_image_tokens: u32) -> Self {
        let backend = Self {
            name,
            max_tokens: max_tokens as usize,
            max_image_tokens: max_image_tokens as usize,
            device,
            state: Arc::new(Mutex::new(State::Preparing("locating model".into()))),
        };
        backend.prepare();
        backend
    }

    /// Finds, downloads and loads the model on a thread; the state says how far.
    fn prepare(&self) {
        let worker_state = Arc::clone(&self.state);
        let (device, max_image_tokens) = (self.device, self.max_image_tokens);
        std::thread::Builder::new()
            .name("pc4l-model".into())
            .spawn(move || {
                let set = |s: String| {
                    *worker_state.lock().unwrap() = State::Preparing(s);
                };
                let result = models::GLM_OCR
                    .ensure(&set)
                    .and_then(|dir| load(&dir, device, max_image_tokens, &set));
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
    }

    /// Drops a model whose GPU is gone and reloads on the CPU. `state` is the
    /// caller's lock on the state.
    fn reload_after_gpu_loss(&self, state: &mut State) {
        *state = State::Preparing("GPU lost — reloading on the CPU".into());
        self.prepare();
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

    fn read(
        &self,
        png: &[u8],
        _mode: Mode,
        prompt: &str,
        _capture_px: (u32, u32),
    ) -> Result<Transcription> {
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
        // The other model may have lost the GPU under this one.
        if model.device() == Device::WebGpu && GPU_LOST.load(Ordering::SeqCst) {
            self.reload_after_gpu_loss(&mut state);
            bail!("the GPU was lost; the model is reloading on the CPU, try again shortly");
        }
        let out = {
            let _turn = runtime_turn();
            model.generate(&img, prompt, self.max_tokens)
        };
        let out = match out {
            Ok(out) => out,
            Err(e) if note_gpu_loss(&e) => {
                self.reload_after_gpu_loss(&mut state);
                bail!(
                    "the GPU was lost mid-read (a driver reset); the model is reloading on \
                     the CPU, try again shortly. A smaller region or image budget avoids it."
                );
            }
            Err(e) => return Err(e),
        };
        let (readings, silent) = if out.text.is_empty() {
            (Vec::new(), 1)
        } else {
            (
                vec![Reading {
                    text: out.text,
                    count: 1,
                    truncated: out.truncated,
                    tokens: (!out.tokens.is_empty()).then_some(out.tokens),
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
