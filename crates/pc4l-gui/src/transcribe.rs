//! Reading a crop: the [`Transcriber`] trait and the remote backends, plus the on-disk
//! backend list (`~/.config/pc4l/gui.toml`).
//!
//! The rules come from halo-workbench's hint tool, which this replaces: several
//! readings are shown as several readings (grouped by identical text, counted, never
//! merged or ranked across backends), an empty answer is reported rather than hidden,
//! and the prompts are the ones every measurement behind the model choice was taken
//! with -- change them and the comparison is void.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Verbatim from halo-workbench `app/handwriting.py`.
pub const CROP_PROMPT: &str =
    "This is a crop from a handwritten student's answer. Write out exactly what is \
     written in it, character for character. Do not correct spelling, do not complete \
     an abbreviation, and do not explain. If part of it is struck out, show that. \
     If you cannot read it, say UNREADABLE.";

/// Verbatim from halo-workbench `app/handwriting.py`.
pub const PAGE_PROMPT: &str =
    "Transcribe the handwritten text in the attached image of a student's answer \
     to a test question. Provide only the word for word transcription and do not \
     correct or infer meaning.";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One word or line the operator boxed: read it character for character.
    Crop,
    /// A whole page.
    Page,
}

impl Mode {
    pub fn prompt(self) -> &'static str {
        match self {
            Mode::Crop => CROP_PROMPT,
            Mode::Page => PAGE_PROMPT,
        }
    }

    fn api_name(self) -> &'static str {
        match self {
            Mode::Crop => "crop",
            Mode::Page => "page",
        }
    }
}

/// One distinct answer and how many of the samples gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reading {
    pub text: String,
    pub count: u32,
    /// The generation hit its token limit: the text may be cut short.
    pub truncated: bool,
}

/// What one backend said about one image.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub backend: String,
    pub readings: Vec<Reading>,
    /// Samples that came back with no text at all -- a reported result, not an error.
    pub silent: u32,
    pub samples: u32,
    pub elapsed: Duration,
}

/// Something that reads an image. `png` is the encoded crop; implementations run on
/// a worker thread and may block.
pub trait Transcriber: Send + Sync {
    fn name(&self) -> &str;
    /// A line for the window about the backend's own state (downloading, loading,
    /// unavailable); `None` when there is nothing to say.
    fn status(&self) -> Option<String> {
        None
    }
    /// `capture_px` is the size of the whole frame the crop was cut from, for
    /// backends that warn about low-resolution captures.
    fn read(&self, png: &[u8], mode: Mode, capture_px: (u32, u32)) -> Result<Transcription>;
}

// ---------------------------------------------------------------------------
// Configuration

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BackendConfig {
    /// Any OpenAI-compatible chat endpoint with image input: llama-server, Ollama,
    /// vLLM, OpenAI itself. `base_url` is everything before `/chat/completions`.
    OpenAi {
        name: String,
        base_url: String,
        model: String,
        #[serde(default)]
        api_key: Option<String>,
        /// How many times to ask; more than one samples at `temperature` to show
        /// how sure the model is.
        #[serde(default = "one")]
        samples: u32,
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
    },
    /// halo-workbench's `/hint/read`: it picks the model (`member`), starts it on
    /// demand, and does the sampling itself.
    HintApi {
        name: String,
        /// The workbench root, e.g. `http://127.0.0.1:8093`.
        base_url: String,
        member: String,
        #[serde(default = "default_hint_samples")]
        samples: u32,
    },
    /// The bundled GLM-OCR on ONNX Runtime (needs the `local-model` build feature).
    /// Greedy decoding: one reading per request.
    Local {
        name: String,
        #[serde(default)]
        device: LocalDevice,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        /// Ceiling on image tokens (the image is downscaled to fit). A whole page
        /// at the model's default budget lost the GPU; 2048 is the workbench's
        /// measured setting.
        #[serde(default = "default_max_image_tokens")]
        max_image_tokens: u32,
    },
}

#[cfg(feature = "local-model")]
pub type LocalDevice = crate::local::DevicePref;

/// Accepted and ignored when the feature is off, so one config file serves both builds.
#[cfg(not(feature = "local-model"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalDevice {
    #[default]
    Auto,
    Webgpu,
    Cpu,
}

fn one() -> u32 {
    1
}
fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> u32 {
    1024
}
fn default_hint_samples() -> u32 {
    3
}
fn default_max_image_tokens() -> u32 {
    2048
}

impl BackendConfig {
    /// `None` for a backend this build cannot provide.
    pub fn build(&self) -> Option<Box<dyn Transcriber>> {
        Some(match self.clone() {
            BackendConfig::OpenAi {
                name,
                base_url,
                model,
                api_key,
                samples,
                temperature,
                max_tokens,
            } => Box::new(OpenAiBackend {
                name,
                base_url: base_url.trim_end_matches('/').to_string(),
                model,
                api_key,
                samples: samples.max(1),
                temperature,
                max_tokens,
            }),
            BackendConfig::HintApi {
                name,
                base_url,
                member,
                samples,
            } => Box::new(HintApiBackend {
                name,
                base_url: base_url.trim_end_matches('/').to_string(),
                member,
                samples: samples.max(1),
            }),
            #[cfg(feature = "local-model")]
            BackendConfig::Local {
                name,
                device,
                max_tokens,
                max_image_tokens,
            } => Box::new(crate::local::LocalBackend::new(
                name,
                device,
                max_tokens,
                max_image_tokens,
            )),
            #[cfg(not(feature = "local-model"))]
            BackendConfig::Local { name, .. } => {
                log::warn!("backend {name:?} needs a build with the local-model feature");
                return None;
            }
        })
    }
}

/// The built-in block detector (PP-DocLayoutV3; needs the `local-model` build
/// feature).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LayoutConfig {
    pub enabled: bool,
    pub device: LocalDevice,
    /// Minimum detection score. Handwriting scores lower than the printed pages the
    /// model was trained on.
    pub threshold: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: LocalDevice::default(),
            threshold: 0.4,
        }
    }
}

/// The window itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UiConfig {
    /// Everything in the window scaled by this (1.0 = the desktop's own size).
    pub scale: f32,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
    #[serde(default)]
    pub layout: LayoutConfig,
    #[serde(default)]
    pub ui: UiConfig,
}

impl Config {
    pub fn path() -> PathBuf {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_default();
        base.join("pc4l").join("gui.toml")
    }

    /// The file's contents, or -- if there is no file -- the defaults, written out so
    /// there is something to edit.
    pub fn load_or_create() -> Result<Self> {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let config = Self::default();
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                std::fs::write(&path, Self::default_text())
                    .with_context(|| format!("writing {}", path.display()))?;
                log::info!("wrote default backends to {}", path.display());
                Ok(config)
            }
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Writes the file; the Settings window's Save.
    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, self.text()).with_context(|| format!("writing {}", path.display()))
    }

    fn default_text() -> String {
        Self::default().text()
    }

    fn text(&self) -> String {
        format!(
            "# pc4l-gui settings: edit here or in the window's Settings. Each [[backends]]\n\
             # entry is one choice in the window; the first is selected at startup.\n\
             # [layout] is the block detector, [ui] the window.\n\n{}",
            toml::to_string_pretty(self).expect("config serialises")
        )
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backends: vec![
                BackendConfig::Local {
                    name: "GLM-OCR (built in)".into(),
                    device: LocalDevice::default(),
                    max_tokens: 1024,
                    max_image_tokens: 2048,
                },
                BackendConfig::HintApi {
                    name: "workbench GLM-OCR".into(),
                    base_url: "http://127.0.0.1:8093".into(),
                    member: "glm-ocr".into(),
                    samples: 3,
                },
                BackendConfig::OpenAi {
                    name: "llama-server :8099".into(),
                    base_url: "http://127.0.0.1:8099/v1".into(),
                    model: "local".into(),
                    api_key: None,
                    samples: 1,
                    temperature: 0.0,
                    max_tokens: 1024,
                },
            ],
            layout: LayoutConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible

struct OpenAiBackend {
    name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    samples: u32,
    temperature: f32,
    max_tokens: u32,
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

fn data_url(png: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

impl Transcriber for OpenAiBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn read(&self, png: &[u8], mode: Mode, _capture_px: (u32, u32)) -> Result<Transcription> {
        let started = Instant::now();
        // One sample is a greedy read; spread is only asked for when sampling.
        let temperature = if self.samples == 1 {
            0.0
        } else {
            self.temperature
        };
        let body = serde_json::json!({
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(png)}},
                {"type": "text", "text": mode.prompt()},
            ]}],
        });
        let agent = agent();
        let url = format!("{}/chat/completions", self.base_url);
        let mut samples = Vec::new();
        for _ in 0..self.samples {
            let mut request = agent.post(&url);
            if let Some(key) = &self.api_key {
                request = request.header("authorization", format!("Bearer {key}"));
            }
            let mut response = request
                .send_json(&body)
                .with_context(|| format!("{} did not answer", self.base_url))?;
            let status = response.status().as_u16();
            let payload: serde_json::Value =
                response.body_mut().read_json().with_context(|| {
                    format!("{}: unreadable response (HTTP {status})", self.base_url)
                })?;
            if status >= 300 {
                let detail = payload
                    .pointer("/error/message")
                    .or_else(|| payload.get("error"))
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string())
                    })
                    .unwrap_or_default();
                bail!("{} answered HTTP {status} {detail}", self.base_url);
            }
            samples.push(parse_choice(&payload)?);
        }
        Ok(group(&self.name, samples, started.elapsed()))
    }
}

struct Sample {
    text: String,
    truncated: bool,
}

fn parse_choice(payload: &serde_json::Value) -> Result<Sample> {
    let choice = payload
        .pointer("/choices/0")
        .ok_or_else(|| anyhow!("response has no choices"))?;
    let text = choice
        .pointer("/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let truncated = choice.get("finish_reason").and_then(|f| f.as_str()) == Some("length");
    Ok(Sample { text, truncated })
}

/// Whitespace-insensitive key, as the workbench's `normalise()`: two samples that
/// differ only in spacing are one reading.
fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Distinct readings with their support, most-supported first, ties in first-seen
/// order; empty answers counted separately as `silent`.
fn group(backend: &str, samples: Vec<Sample>, elapsed: Duration) -> Transcription {
    let mut readings: Vec<Reading> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut silent = 0;
    let total = samples.len() as u32;
    for s in samples {
        if s.text.is_empty() {
            silent += 1;
            continue;
        }
        let key = normalise(&s.text);
        match keys.iter().position(|k| *k == key) {
            Some(i) => {
                readings[i].count += 1;
                readings[i].truncated |= s.truncated;
            }
            None => {
                keys.push(key);
                readings.push(Reading {
                    text: s.text,
                    count: 1,
                    truncated: s.truncated,
                });
            }
        }
    }
    // Stable sort keeps first-seen order among equals.
    readings.sort_by_key(|r| std::cmp::Reverse(r.count));
    Transcription {
        backend: backend.to_string(),
        readings,
        silent,
        samples: total,
        elapsed,
    }
}

// ---------------------------------------------------------------------------
// halo-workbench /hint/read

struct HintApiBackend {
    name: String,
    base_url: String,
    member: String,
    samples: u32,
}

impl Transcriber for HintApiBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn read(&self, png: &[u8], mode: Mode, capture_px: (u32, u32)) -> Result<Transcription> {
        let started = Instant::now();
        let body = serde_json::json!({
            "member": self.member,
            "mode": mode.api_name(),
            "samples": self.samples,
            "image": data_url(png),
            "capture_px": {"width": capture_px.0, "height": capture_px.1},
        });
        let url = format!("{}/hint/read", self.base_url);
        let mut response = agent()
            .post(&url)
            .send_json(&body)
            .with_context(|| format!("{} did not answer", self.base_url))?;
        let status = response.status().as_u16();
        let payload: serde_json::Value = response
            .body_mut()
            .read_json()
            .with_context(|| format!("{}: unreadable response (HTTP {status})", self.base_url))?;
        if status >= 300 {
            let detail = payload.get("error").and_then(|e| e.as_str()).unwrap_or("");
            bail!("{} answered HTTP {status}: {detail}", self.base_url);
        }
        parse_hint_response(&self.name, &payload, started.elapsed())
    }
}

fn parse_hint_response(
    backend: &str,
    payload: &serde_json::Value,
    elapsed: Duration,
) -> Result<Transcription> {
    let readings = payload
        .get("readings")
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("hint response has no readings"))?
        .iter()
        .map(|r| Reading {
            text: r
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            count: r.get("count").and_then(|c| c.as_u64()).unwrap_or(1) as u32,
            truncated: r
                .get("truncated")
                .and_then(|t| t.as_bool())
                .unwrap_or(false),
        })
        .collect();
    let samples = payload
        .get("samples")
        .and_then(|s| s.as_array())
        .map(|s| s.len() as u32)
        .unwrap_or(0);
    let silent = payload
        .get("silent_count")
        .and_then(|s| s.as_u64())
        .unwrap_or(0) as u32;
    Ok(Transcription {
        backend: backend.to_string(),
        readings,
        silent,
        samples,
        elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> Sample {
        Sample {
            text: text.into(),
            truncated: false,
        }
    }

    #[test]
    fn grouping_counts_support_and_keeps_first_seen_order() {
        let t = group(
            "b",
            vec![s("the"), s("tho"), s("the "), s(""), s("  the")],
            Duration::ZERO,
        );
        assert_eq!(t.samples, 5);
        assert_eq!(t.silent, 1);
        assert_eq!(
            t.readings,
            vec![
                Reading {
                    text: "the".into(),
                    count: 3,
                    truncated: false
                },
                Reading {
                    text: "tho".into(),
                    count: 1,
                    truncated: false
                },
            ]
        );
        // Ties stay in first-seen order rather than being ranked.
        let t = group("b", vec![s("b"), s("a")], Duration::ZERO);
        assert_eq!(t.readings[0].text, "b");
    }

    #[test]
    fn parses_openai_choice_and_length_stop() {
        let v = serde_json::json!({"choices": [{"message": {"content": " x^2 "}, "finish_reason": "length"}]});
        let c = parse_choice(&v).unwrap();
        assert_eq!(c.text, "x^2");
        assert!(c.truncated);
        assert!(parse_choice(&serde_json::json!({"choices": []})).is_err());
    }

    #[test]
    fn parses_hint_response() {
        let v = serde_json::json!({
            "readings": [{"text": "mitochondria", "count": 2, "truncated": false}],
            "samples": [{}, {}, {}],
            "silent_count": 1,
        });
        let t = parse_hint_response("w", &v, Duration::ZERO).unwrap();
        assert_eq!(t.readings.len(), 1);
        assert_eq!(t.readings[0].count, 2);
        assert_eq!((t.samples, t.silent), (3, 1));
    }

    #[test]
    fn default_config_round_trips_through_toml() {
        let text = Config::default_text();
        let parsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(parsed, Config::default());
        assert!(matches!(parsed.backends[0], BackendConfig::Local { .. }));
        assert_eq!(
            parsed.backends[1].build().unwrap().name(),
            "workbench GLM-OCR"
        );
        // Optional fields may be left out.
        let minimal: Config = toml::from_str(
            "[[backends]]\nkind = \"open-ai\"\nname = \"x\"\nbase_url = \"http://h/v1\"\nmodel = \"m\"\n",
        )
        .unwrap();
        assert!(matches!(
            minimal.backends[0],
            BackendConfig::OpenAi { samples: 1, .. }
        ));
    }
}
