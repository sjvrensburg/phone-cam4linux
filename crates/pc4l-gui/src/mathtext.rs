//! Readings typeset: the models write maths as LaTeX (`$\hat{y}_i \neq y_i$`), which
//! is hard to check against the ink as raw text. Each `$…$` / `$$…$$` (and `\(…\)`,
//! `\[…\]`) segment is converted to Typst math by MiTeX and the whole reading is
//! compiled by Typst and rasterised. Anything that does not convert or compile is
//! left as the text it was: a rendering is a convenience, the reading is the text.

use anyhow::{anyhow, Result};
use image::RgbaImage;
use std::time::Instant;
use typst::foundations::{Dict, IntoValue};
use typst_as_lib::typst_kit_options::TypstKitFontOptions;
use typst_as_lib::{TypstEngine, TypstTemplateCollection};
use typst_layout::PagedDocument;

/// The page: as wide as asked, as tall as needed, no background, text in the
/// window's colour, and the reading evaluated as markup inside MiTeX's scope so the
/// converted commands (`operatorname`, …) resolve.
const TEMPLATE: &str = r#"
#import sys: inputs
#import "specs/mod.typ": mitex-scope
#set page(width: inputs.width * 1pt, height: auto, margin: 3pt, fill: none)
#set text(size: inputs.size * 1pt, fill: rgb(inputs.color))
#set par(leading: 0.5em)
#let compat = (hbar: symbol("ℏ"))
#eval(inputs.src, mode: "markup", scope: mitex-scope + compat)
"#;

/// Symbol names the `mitex` crate's built-in spec (older than Typst 0.15) emits
/// that Typst has since renamed, with what they are called now. Applied to whole
/// dotted tokens: an entry matches the token or a prefix of it up to a dot. Found
/// by pushing every command in MiTeX's spec through both.
const RENAMES: &[(&str, &str)] = &[
    ("diff", "partial"),
    ("sect", "inter"),
    ("ohm", "Omega"),
    ("dot.circle", "dot.o"),
    ("plus.circle", "plus.o"),
    ("times.circle", "times.o"),
    ("minus.circle", "minus.o"),
    ("ast.circle", "convolve.o"),
    ("dash.circle", "dash.o"),
    ("circle.nested", "compose.o"),
    ("planck.reduce", "hbar"),
    ("angle.l", "chevron.l"),
    ("angle.r", "chevron.r"),
    ("bracket.l.double", "bracket.l.stroked"),
    ("bracket.r.double", "bracket.r.stroked"),
    ("arrow.l.dash", "arrow.l.dashed"),
    ("arrow.r.dash", "arrow.r.dashed"),
];

/// Rewrites renamed symbols in converted math, token by token (a token is a run of
/// letters, digits and dots).
fn modernise(math: &str) -> String {
    let mut out = String::with_capacity(math.len());
    let mut token = String::new();
    let flush = |token: &mut String, out: &mut String| {
        if token.is_empty() {
            return;
        }
        let mut replaced = None;
        for (old, new) in RENAMES {
            if token == old {
                replaced = Some(new.to_string());
                break;
            }
            if let Some(rest) = token.strip_prefix(old) {
                if rest.starts_with('.') {
                    replaced = Some(format!("{new}{rest}"));
                    break;
                }
            }
        }
        out.push_str(&replaced.unwrap_or_else(|| token.clone()));
        token.clear();
    };
    for c in math.chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            token.push(c);
        } else {
            flush(&mut token, &mut out);
            out.push(c);
        }
    }
    flush(&mut token, &mut out);
    out
}

pub struct Renderer {
    engine: TypstEngine<TypstTemplateCollection>,
}

/// A rasterised reading.
pub struct Rendered {
    pub image: RgbaImage,
    /// Pixels per typographic point it was rendered at, to show it at true size.
    pub scale: f32,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        let engine = TypstEngine::builder()
            .with_static_source_file_resolver([
                ("main.typ", TEMPLATE),
                (
                    "specs/mod.typ",
                    include_str!("../assets/mitex/specs/mod.typ"),
                ),
                (
                    "specs/prelude.typ",
                    include_str!("../assets/mitex/specs/prelude.typ"),
                ),
                (
                    "specs/latex/standard.typ",
                    include_str!("../assets/mitex/specs/latex/standard.typ"),
                ),
            ])
            .search_fonts_with(
                TypstKitFontOptions::new()
                    .include_system_fonts(false)
                    .include_embedded_fonts(true),
            )
            .build();
        Self { engine }
    }

    /// Typesets `text` `width_pt` points wide at `size_pt`, `scale` pixels per
    /// point, in colour `rgb`. Errors when the text does not compile.
    pub fn render(
        &self,
        text: &str,
        width_pt: f32,
        size_pt: f32,
        scale: f32,
        rgb: [u8; 3],
    ) -> Result<Rendered> {
        let started = Instant::now();
        let src = to_typst(text);
        let mut inputs = Dict::new();
        inputs.insert("src".into(), src.into_value());
        inputs.insert("width".into(), f64::from(width_pt).into_value());
        inputs.insert("size".into(), f64::from(size_pt).into_value());
        inputs.insert(
            "color".into(),
            format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]).into_value(),
        );
        let doc: PagedDocument = self
            .engine
            .compile_with_input("main.typ", inputs)
            .output
            .map_err(|e| anyhow!("{e:?}"))?;
        let page = doc.pages().first().ok_or_else(|| anyhow!("no page"))?;
        let pix = typst_render::render(
            page,
            &typst_render::RenderOptions {
                pixel_per_pt: f64::from(scale).into(),
                ..Default::default()
            },
        );
        let (w, h) = (pix.width(), pix.height());
        // tiny-skia pixmaps are premultiplied RGBA.
        let mut data = pix.take();
        for px in data.as_chunks_mut::<4>().0 {
            let a = px[3] as u32;
            if a > 0 && a < 255 {
                for c in &mut px[..3] {
                    *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
                }
            }
        }
        let image = RgbaImage::from_raw(w, h, data).ok_or_else(|| anyhow!("pixmap size"))?;
        log::debug!(
            "typeset {} chars to {w}x{h} in {:.1} ms",
            text.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        Ok(Rendered { image, scale })
    }
}

/// The reading as Typst markup: maths converted (a `$$` block may span lines), the
/// rest escaped with its line breaks kept.
fn to_typst(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    let mut at_line_start = true;
    while !rest.is_empty() {
        let Some((start, open, close, display)) = next_math(rest) else {
            escape(rest, at_line_start, &mut out);
            break;
        };
        escape(&rest[..start], at_line_start, &mut out);
        at_line_start = false;
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => {
                let tex = &after[..end];
                // Newlines inside the converted maths are only layout.
                let converted = mitex::convert_math(tex, None).map(|m| {
                    modernise(&m)
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                });
                match converted {
                    Ok(math) if !tex.trim().is_empty() => {
                        if display {
                            out.push_str(&format!("$ {math} $"));
                        } else {
                            out.push_str(&format!("${math}$"));
                        }
                    }
                    _ => escape(
                        &rest[start..start + open.len() + end + close.len()],
                        false,
                        &mut out,
                    ),
                }
                rest = &after[end + close.len()..];
                // A display block on its own line: the newline after it is its own.
                if display {
                    if let Some(r) = rest.strip_prefix('\n') {
                        rest = r;
                        at_line_start = true;
                    }
                }
            }
            None => {
                escape(&rest[start..], false, &mut out);
                break;
            }
        }
    }
    out
}

/// The next maths opener in `s`: its byte offset, the opener, its closer, and
/// whether it is display maths.
fn next_math(s: &str) -> Option<(usize, &'static str, &'static str, bool)> {
    let candidates: [(&str, &str, bool); 4] = [
        ("$$", "$$", true),
        ("\\[", "\\]", true),
        ("\\(", "\\)", false),
        ("$", "$", false),
    ];
    let mut best: Option<(usize, &'static str, &'static str, bool)> = None;
    for (open, close, display) in candidates {
        if let Some(i) = s.find(open) {
            if best.is_none_or(|b| i < b.0) {
                best = Some((i, open, close, display));
            }
        }
    }
    best
}

/// Plain text as Typst markup: everything that would be syntax is backslashed,
/// and the list/heading markers only matter at the start of a line.
fn escape(s: &str, at_line_start: bool, out: &mut String) {
    let mut first = at_line_start;
    for (i, line) in s.split('\n').enumerate() {
        if i > 0 {
            // A blank line is a paragraph break, any other a line break.
            out.push_str(if line.trim().is_empty() {
                "\n\n"
            } else {
                " \\\n"
            });
            first = true;
        }
        for c in line.chars() {
            let special =
                "#$*_`<>@\\[]~".contains(c) || (first && !c.is_whitespace() && "-+/=".contains(c));
            if special {
                out.push('\\');
            }
            out.push(c);
            if !c.is_whitespace() {
                first = false;
            }
        }
    }
}

impl crate::app::Typesetter for Renderer {
    fn render(
        &self,
        text: &str,
        width_pt: f32,
        size_pt: f32,
        scale: f32,
        rgb: [u8; 3],
    ) -> Result<(RgbaImage, f32)> {
        let r = Renderer::render(self, text, width_pt, size_pt, scale, rgb)?;
        Ok((r.image, r.scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_maths_and_escapes_the_rest() {
        let t = to_typst("where $\\hat{y}_i \\neq y_i$ (see #3)\nnext line");
        assert!(t.starts_with("where $hat("), "{t}");
        assert!(t.contains("!="), "{t}");
        assert!(t.contains("\\#3"), "{t}");
        assert!(t.contains(" \\\nnext line"), "{t}");
    }

    #[test]
    fn display_maths_spans_lines() {
        let t = to_typst("func\n$$\n\\ln x+b 1\n$$\nnext");
        assert!(t.contains("$ ln"), "{t}");
        assert!(!t.contains("\\$"), "{t}");
        assert!(t.ends_with("next"), "{t}");
    }

    #[test]
    fn unclosed_maths_stays_text() {
        let t = to_typst("cost $5 and more");
        assert_eq!(t, "cost \\$5 and more");
    }

    #[test]
    fn list_markers_are_escaped_only_at_line_start() {
        let t = to_typst("- a - b\n= c");
        assert_eq!(t, "\\- a - b \\\n\\= c");
    }

    #[test]
    fn renamed_symbols_are_modernised() {
        assert_eq!(
            modernise("frac(diff l ,diff w )"),
            "frac(partial l ,partial w )"
        );
        assert_eq!(modernise("A sect.big B"), "A inter.big B");
        assert_eq!(modernise("differential"), "differential");
        assert_eq!(modernise("x plus.circle.big y"), "x plus.o.big y");
    }

    #[test]
    fn a_page_with_partials_cases_and_operators_renders() {
        let text = "1) cases when $y_i \\neq y_j$\n$$\\frac{\\partial l}{\\partial w} = \\begin{cases} -x & \\text{if } wx+b < 0 \\\\ x & \\text{if } wx+b > 0 \\end{cases}$$\n$$w_{i.e.} = \\pi \\operatorname{sign}(wx+b)x; \\hbar \\oplus \\langle x \\rangle$$";
        let r = Renderer::new();
        let out = r.render(text, 300.0, 12.0, 1.0, [0, 0, 0]).unwrap();
        assert!(out.image.height() > 40);
    }

    #[test]
    fn renders_something() {
        let r = Renderer::new();
        let out = r
            .render(
                "The entropy is $E = -\\sum_i p_i \\log_2 p_i$",
                300.0,
                12.0,
                1.0,
                [0, 0, 0],
            )
            .unwrap();
        assert!(out.image.width() > 250 && out.image.height() > 10);
        assert!(out.image.pixels().any(|p| p[3] > 0));
    }
}
