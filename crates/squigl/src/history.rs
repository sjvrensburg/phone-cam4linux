//! The session's readings, kept after the results list moves on: what was read,
//! from which capture, by which backend, when. Copyable as text and saved as
//! Markdown, in page order for a "read all".

use crate::transcribe::{Confidence, Reading, Transcription, UiConfig};
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

/// Wraps `word` (the non-whitespace part of a token's text) in `wrap` on both
/// sides, leaving any leading whitespace outside it so the markers land next to
/// the word as Markdown emphasis requires.
fn mark(text: &str, wrap: &str) -> String {
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return text.to_string();
    }
    let (ws, word) = text.split_at(text.len() - trimmed.len());
    format!("{ws}{wrap}{word}{wrap}")
}

/// A reading's text with wavering tokens in `*italics*` and hesitant tokens in
/// `**bold**`, so the flag survives a plain-text copy or Markdown export; falls
/// back to the plain text when the backend reported no (validated) tokens.
fn marked_text(r: &Reading, ui_cfg: &UiConfig) -> String {
    let Some(tokens) = r.tokens_if_valid() else {
        return r.text.clone();
    };
    tokens
        .iter()
        .map(|t| match t.confidence(ui_cfg) {
            Confidence::Steady => t.text.clone(),
            Confidence::Wavering => mark(&t.text, "*"),
            Confidence::Hesitant => mark(&t.text, "**"),
        })
        .collect()
}

impl Entry {
    /// The readings as plain text: one per line, prefixed with support when the
    /// backend sampled more than once.
    pub fn text(&self, ui_cfg: &UiConfig) -> String {
        match &self.result {
            Ok(t) if t.readings.is_empty() => "(no answer)".into(),
            Ok(t) if t.samples == 1 => marked_text(&t.readings[0], ui_cfg),
            Ok(t) => t
                .readings
                .iter()
                .map(|r| format!("[{}/{}] {}", r.count, t.samples, marked_text(r, ui_cfg)))
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
    pub fn markdown(&self, ui_cfg: &UiConfig) -> String {
        let mut out = format!(
            "# squigl readings — {}\n\n",
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
            out.push_str(&format!(
                "### {} — {}\n\n{}\n\n",
                e.what,
                backend,
                e.text(ui_cfg)
            ));
        }
        out
    }

    /// Writes the Markdown next to the captures; returns the path.
    pub fn save(&self, dir: &Path, ui_cfg: &UiConfig) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!(
            "squigl-readings-{}.md",
            Local::now().format("%Y%m%d-%H%M%S")
        ));
        std::fs::write(&path, self.markdown(ui_cfg))?;
        Ok(path)
    }
}

/// Entries joined as plain text, blank lines between; for the clipboard.
pub fn joined<'a>(entries: impl Iterator<Item = &'a Entry>, ui_cfg: &UiConfig) -> String {
    entries
        .map(|e| e.text(ui_cfg))
        .collect::<Vec<_>>()
        .join("\n\n")
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
                    tokens: None,
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
        let ui_cfg = UiConfig::default();
        let md = h.markdown(&ui_cfg);
        let i1 = md.find("## Capture 1").unwrap();
        let i2 = md.find("## Capture 2").unwrap();
        assert!(i1 < md.find("first").unwrap() && md.find("first") < md.find("second"));
        assert!(md.find("second").unwrap() < i2 && i2 < md.find("third").unwrap());
        assert_eq!(
            joined(h.entries.iter(), &ui_cfg),
            "first\n\nsecond\n\nthird"
        );
    }

    #[test]
    fn hesitant_tokens_are_marked_in_exported_text() {
        use crate::transcribe::{Token, TokenAlt};
        let ui_cfg = UiConfig::default();
        let mut e = entry(1, "box", "the cat");
        let Ok(t) = &mut e.result else { unreachable!() };
        t.readings[0].tokens = Some(vec![
            Token {
                text: "the".into(),
                prob: 0.99,
                alternates: vec![],
            },
            Token {
                text: " cat".into(),
                prob: 0.3,
                alternates: vec![TokenAlt {
                    text: " cot".into(),
                    prob: 0.2,
                }],
            },
        ]);
        assert_eq!(e.text(&ui_cfg), "the **cat**");
    }
}
