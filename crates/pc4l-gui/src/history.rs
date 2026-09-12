//! The session's readings, kept after the results list moves on: what was read,
//! from which capture, by which backend, when. Copyable as text and saved as
//! Markdown, in page order for a "read all".

use crate::transcribe::Transcription;
use chrono::{DateTime, Local};
use std::path::{Path, PathBuf};

/// One finished read.
pub struct Entry {
    pub at: DateTime<Local>,
    /// Which capture of the session it came from (1-based).
    pub capture: u32,
    /// The block label in a "read all", else what was read ("box", "page").
    pub what: String,
    pub result: Result<Transcription, String>,
}

impl Entry {
    /// The readings as plain text: one per line, prefixed with support when the
    /// backend sampled more than once.
    pub fn text(&self) -> String {
        match &self.result {
            Ok(t) if t.readings.is_empty() => "(no answer)".into(),
            Ok(t) if t.samples == 1 => t.readings[0].text.clone(),
            Ok(t) => t
                .readings
                .iter()
                .map(|r| format!("[{}/{}] {}", r.count, t.samples, r.text))
                .collect::<Vec<_>>()
                .join("\n"),
            Err(e) => format!("(failed: {e})"),
        }
    }
}

#[derive(Default)]
pub struct History {
    pub entries: Vec<Entry>,
}

impl History {
    pub fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    /// The whole session as Markdown, a section per capture, entries in the order
    /// they were read (page order for a "read all").
    pub fn markdown(&self) -> String {
        let mut out = format!(
            "# pc4l readings — {}\n\n",
            Local::now().format("%Y-%m-%d %H:%M")
        );
        let mut capture = None;
        for e in &self.entries {
            if capture != Some(e.capture) {
                capture = Some(e.capture);
                out.push_str(&format!(
                    "## Capture {} ({})\n\n",
                    e.capture,
                    e.at.format("%Y-%m-%d %H:%M")
                ));
            }
            let backend = match &e.result {
                Ok(t) => t.backend.as_str(),
                Err(_) => "",
            };
            out.push_str(&format!("### {} — {}\n\n{}\n\n", e.what, backend, e.text()));
        }
        out
    }

    /// Writes the Markdown next to the captures; returns the path.
    pub fn save(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!(
            "pc4l-readings-{}.md",
            Local::now().format("%Y%m%d-%H%M%S")
        ));
        std::fs::write(&path, self.markdown())?;
        Ok(path)
    }
}

/// Entries joined as plain text, blank lines between; for the clipboard.
pub fn joined<'a>(entries: impl Iterator<Item = &'a Entry>) -> String {
    entries.map(|e| e.text()).collect::<Vec<_>>().join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::Reading;
    use std::time::Duration;

    fn entry(capture: u32, what: &str, text: &str) -> Entry {
        Entry {
            at: Local::now(),
            capture,
            what: what.into(),
            result: Ok(Transcription {
                backend: "test".into(),
                readings: vec![Reading {
                    text: text.into(),
                    count: 1,
                    truncated: false,
                }],
                silent: 0,
                samples: 1,
                elapsed: Duration::ZERO,
            }),
        }
    }

    #[test]
    fn markdown_groups_by_capture_in_order() {
        let mut h = History::default();
        h.push(entry(1, "#1 text", "first"));
        h.push(entry(1, "#2 text", "second"));
        h.push(entry(2, "box", "third"));
        let md = h.markdown();
        let i1 = md.find("## Capture 1").unwrap();
        let i2 = md.find("## Capture 2").unwrap();
        assert!(i1 < md.find("first").unwrap() && md.find("first") < md.find("second"));
        assert!(md.find("second").unwrap() < i2 && i2 < md.find("third").unwrap());
        assert_eq!(joined(h.entries.iter()), "first\n\nsecond\n\nthird");
    }
}
