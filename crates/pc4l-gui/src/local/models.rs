//! Where the bundled models' files live, and how they get there.
//!
//! The hundreds of megabytes of graphs are not embedded in the binary (that would
//! relink them on every build and put them in git); each model is looked up, in
//! order, in `$PC4L_MODEL_DIR/<name>/`, a `models/<name>/` directory next to the
//! executable (how a release tarball ships them), and
//! `$XDG_CACHE_HOME/pc4l/models/<name>/`. If none has it, it is downloaded into the
//! cache from a pinned Hugging Face revision, each file checked against the sha256
//! recorded here before it is used.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct ModelFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// One downloadable model: its directory name and its file manifest.
pub struct ModelSpec {
    /// The directory the files live under (`models/<name>/`).
    pub name: &'static str,
    /// `https://huggingface.co/<repo>/resolve/<revision>`.
    pub repo_url: &'static str,
    pub files: &'static [ModelFile],
}

/// `onnx-community/GLM-OCR-ONNX`, q4f16 variant, at a pinned revision.
pub const GLM_OCR: ModelSpec = ModelSpec {
    name: "glm-ocr-onnx-q4f16",
    repo_url: "https://huggingface.co/onnx-community/GLM-OCR-ONNX/resolve/aea46198f09e3aa2b63422dd234f1cc66afffe52",
    files: &[
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
    ],
};

/// PaddlePaddle's official ONNX export of PP-DocLayoutV3 (Apache-2.0), at a pinned
/// revision.
pub const DOC_LAYOUT: ModelSpec = ModelSpec {
    name: "pp-doclayoutv3-onnx",
    repo_url: "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/46bbdf188bb0a772c08aed74882ce7e51a8f1ea6",
    files: &[ModelFile {
        path: "inference.onnx",
        size: 130_502_049,
        sha256: "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba",
    }],
};

/// Everything `--fetch-model` fetches.
pub const ALL: &[&ModelSpec] = &[&GLM_OCR, &DOC_LAYOUT];

/// The directories a model named `name` is searched for in, in order.
fn candidate_dirs(name: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("PC4L_MODEL_DIR") {
        dirs.push(PathBuf::from(d).join(name));
    }
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        dirs.push(exe_dir.join("models").join(name));
    }
    dirs.push(cache_dir(name));
    dirs
}

/// Where a download lands.
fn cache_dir(name: &str) -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_default();
    base.join("pc4l").join("models").join(name)
}

impl ModelSpec {
    /// Sum of all file sizes: the download, and the "is it complete" test.
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Every file present at its recorded size. Content is verified at download
    /// time; hashing 650 MB on each start is not worth the seconds.
    fn is_complete(&self, dir: &Path) -> bool {
        self.files
            .iter()
            .all(|f| std::fs::metadata(dir.join(f.path)).is_ok_and(|m| m.len() == f.size))
    }

    /// Finds the model, downloading it into the cache if no candidate has it.
    /// `progress` is told what is happening, for the window.
    pub fn ensure(&self, progress: &dyn Fn(String)) -> Result<PathBuf> {
        for dir in candidate_dirs(self.name) {
            if self.is_complete(&dir) {
                log::info!("{} found at {}", self.name, dir.display());
                return Ok(dir);
            }
        }
        let dir = cache_dir(self.name);
        self.download_into(&dir, progress)?;
        Ok(dir)
    }

    /// Downloads the model into `dir` (files already present at the right size are
    /// kept), verifying every file's SHA-256. `pc4l-gui --fetch-model DIR` for
    /// scripts and release packaging.
    pub fn download_into(&self, dir: &Path, progress: &dyn Fn(String)) -> Result<()> {
        log::info!(
            "downloading {} ({} MB) to {}",
            self.name,
            self.total_size() / 1_000_000,
            dir.display()
        );
        let agent = ureq::Agent::config_builder()
            .timeout_global(None)
            .build()
            .new_agent();
        let mut done: u64 = 0;
        let total = self.total_size();
        for file in self.files {
            let target = dir.join(file.path);
            if std::fs::metadata(&target).is_ok_and(|m| m.len() == file.size) {
                done += file.size;
                continue;
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let url = format!("{}/{}", self.repo_url, file.path);
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
                        "downloading {} {}%",
                        self.name,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_are_consistent() {
        assert_eq!(GLM_OCR.files.len(), 9);
        assert_eq!(GLM_OCR.total_size(), 658_133_292);
        assert_eq!(DOC_LAYOUT.total_size(), 130_502_049);
        for spec in ALL {
            assert!(spec.files.iter().all(|f| f.sha256.len() == 64));
            assert!(!spec.is_complete(Path::new("/nonexistent")));
        }
    }
}
