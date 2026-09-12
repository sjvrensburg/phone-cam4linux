//! The Settings window: everything in `~/.config/pc4l/gui.toml`, editable in place --
//! the window's scale, the transcription backends (an OpenAI-compatible endpoint's
//! URL, model and key go here), and the block detector. Edits are made to a draft;
//! Save writes the file and applies it, Cancel drops the draft.

use crate::transcribe::{BackendConfig, Config, LocalDevice, Mode};
use egui::{ComboBox, DragValue, Slider, TextEdit};

/// What the window asked for this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    /// Save the draft and apply it.
    Save,
    /// Drop the draft.
    Cancel,
}

/// The window's requests this frame.
pub struct Outcome {
    pub action: Action,
    /// The draft's scale is settled (slider released, value typed, button
    /// pressed): apply it now. Not while the slider is being dragged -- the window
    /// rescaling under the pointer makes the slider jump.
    pub apply_scale: bool,
}

/// Draws the window; `draft` is edited in place.
pub fn show(ctx: &egui::Context, open: &mut bool, draft: &mut Config) -> Outcome {
    let mut action = Action::None;
    let mut apply_scale = false;
    let mut stay_open = *open;
    egui::Window::new("Settings")
        .open(&mut stay_open)
        .default_width(520.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height(ui.available_height() - 40.0)
                .show(ui, |ui| {
                    ui.heading("Window");
                    ui.horizontal(|ui| {
                        ui.label("Scale");
                        if ui.button("−").clicked() {
                            draft.ui.scale = (draft.ui.scale - 0.1).max(0.75);
                            apply_scale = true;
                        }
                        let slider = ui.add(
                            Slider::new(&mut draft.ui.scale, 0.75..=2.5)
                                .step_by(0.05)
                                .fixed_decimals(2)
                                .suffix("×"),
                        )
                        .on_hover_text("applies when released; also ctrl+plus / ctrl+minus / ctrl+0");
                        if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                            apply_scale = true;
                        }
                        if ui.button("+").clicked() {
                            draft.ui.scale = (draft.ui.scale + 0.1).min(2.5);
                            apply_scale = true;
                        }
                    });
                    ui.add_space(8.0);

                    ui.heading("Backends");
                    ui.weak("The first is selected at startup. Readings are never merged across them.");
                    let mut remove = None;
                    let mut swap = None;
                    let n = draft.backends.len();
                    for (i, b) in draft.backends.iter_mut().enumerate() {
                        ui.push_id(i, |ui| {
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.strong(kind_name(b));
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("remove").clicked() {
                                                remove = Some(i);
                                            }
                                            if i + 1 < n && ui.small_button("down").clicked() {
                                                swap = Some((i, i + 1));
                                            }
                                            if i > 0 && ui.small_button("up").clicked() {
                                                swap = Some((i - 1, i));
                                            }
                                        },
                                    );
                                });
                                backend_fields(ui, b);
                            });
                        });
                    }
                    if let Some(i) = remove {
                        draft.backends.remove(i);
                    }
                    if let Some((a, b)) = swap {
                        draft.backends.swap(a, b);
                    }
                    ui.horizontal(|ui| {
                        ui.label("Add:");
                        if ui.button("OpenAI-compatible").clicked() {
                            draft.backends.push(BackendConfig::OpenAi {
                                name: "new endpoint".into(),
                                base_url: "http://127.0.0.1:8080/v1".into(),
                                model: "".into(),
                                api_key: None,
                                samples: 1,
                                temperature: 0.0,
                                max_tokens: 1024,
                            });
                        }
                        if ui.button("workbench hint API").clicked() {
                            draft.backends.push(BackendConfig::HintApi {
                                name: "workbench".into(),
                                base_url: "http://127.0.0.1:8093".into(),
                                member: "glm-ocr".into(),
                                samples: 3,
                            });
                        }
                        if ui.button("built-in GLM-OCR").clicked() {
                            draft.backends.push(BackendConfig::Local {
                                name: "GLM-OCR (built in)".into(),
                                device: LocalDevice::default(),
                                max_tokens: 1024,
                                max_image_tokens: 2048,
                            });
                        }
                    });
                    ui.add_space(8.0);

                    ui.heading("Prompts");
                    ui.weak(
                        "What the OpenAI-compatible and built-in backends are asked. The hint \
                         API sets its own on the workbench side. The defaults are the ones \
                         every model comparison was made with.",
                    );
                    prompt_row(ui, "For a boxed region", &mut draft.prompts.crop, Mode::Crop);
                    prompt_row(
                        ui,
                        "For a detected formula block",
                        &mut draft.prompts.formula,
                        Mode::Formula,
                    );
                    prompt_row(ui, "For a whole page", &mut draft.prompts.page, Mode::Page);
                    ui.add_space(8.0);

                    ui.heading("Block detector");
                    ui.checkbox(&mut draft.layout.enabled, "Enabled (PP-DocLayoutV3)");
                    ui.horizontal(|ui| {
                        ui.label("Device");
                        device_combo(ui, "layout-device", &mut draft.layout.device);
                        ui.label("Score threshold");
                        ui.add(
                            Slider::new(&mut draft.layout.threshold, 0.1..=0.9)
                                .step_by(0.05)
                                .fixed_decimals(2),
                        )
                        .on_hover_text("handwriting scores lower than the printed pages the model was trained on; 0.4 suits it");
                    });
                    ui.add_space(8.0);
                    ui.weak(format!("Saved to {}", Config::path().display()));
                });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    action = Action::Save;
                }
                if ui.button("Cancel").clicked() {
                    action = Action::Cancel;
                }
            });
        });
    if !stay_open && *open {
        // Closed with the window's own X: same as Cancel.
        action = Action::Cancel;
    }
    *open = stay_open && action == Action::None;
    Outcome {
        action,
        apply_scale,
    }
}

fn prompt_row(ui: &mut egui::Ui, label: &str, value: &mut String, mode: Mode) {
    ui.horizontal(|ui| {
        ui.label(label);
        if *value != mode.default_prompt() && ui.small_button("reset to default").clicked() {
            *value = mode.default_prompt().to_string();
        }
    });
    ui.add(
        TextEdit::multiline(value)
            .desired_rows(3)
            .desired_width(f32::INFINITY),
    );
}

fn kind_name(b: &BackendConfig) -> &'static str {
    match b {
        BackendConfig::OpenAi { .. } => "OpenAI-compatible endpoint",
        BackendConfig::HintApi { .. } => "halo-workbench hint API",
        BackendConfig::Local { .. } => "Built-in GLM-OCR",
    }
}

fn text_row(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            TextEdit::singleline(value)
                .hint_text(hint)
                .desired_width(f32::INFINITY),
        );
    });
}

fn device_combo(ui: &mut egui::Ui, id: &str, device: &mut LocalDevice) {
    ComboBox::from_id_salt(id)
        .selected_text(match device {
            LocalDevice::Auto => "auto (GPU, else CPU)",
            LocalDevice::Webgpu => "GPU (WebGPU)",
            LocalDevice::Cpu => "CPU",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(device, LocalDevice::Auto, "auto (GPU, else CPU)");
            ui.selectable_value(device, LocalDevice::Webgpu, "GPU (WebGPU)");
            ui.selectable_value(device, LocalDevice::Cpu, "CPU");
        });
}

fn backend_fields(ui: &mut egui::Ui, b: &mut BackendConfig) {
    match b {
        BackendConfig::OpenAi {
            name,
            base_url,
            model,
            api_key,
            samples,
            temperature,
            max_tokens,
        } => {
            text_row(ui, "Name", name, "shown in the window");
            text_row(
                ui,
                "Base URL",
                base_url,
                "everything before /chat/completions, e.g. https://api.openai.com/v1",
            );
            text_row(ui, "Model", model, "e.g. gpt-4o");
            ui.horizontal(|ui| {
                ui.label("API key");
                let mut key = api_key.clone().unwrap_or_default();
                if ui
                    .add(
                        TextEdit::singleline(&mut key)
                            .password(true)
                            .hint_text("empty for none")
                            .desired_width(f32::INFINITY),
                    )
                    .changed()
                {
                    *api_key = (!key.is_empty()).then_some(key);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Samples");
                ui.add(DragValue::new(samples).range(1..=16));
                ui.label("Temperature");
                ui.add(DragValue::new(temperature).range(0.0..=2.0).speed(0.05));
                ui.label("Max tokens");
                ui.add(DragValue::new(max_tokens).range(16..=32768));
            });
        }
        BackendConfig::HintApi {
            name,
            base_url,
            member,
            samples,
        } => {
            text_row(ui, "Name", name, "shown in the window");
            text_row(
                ui,
                "Base URL",
                base_url,
                "the workbench root, e.g. http://127.0.0.1:8093",
            );
            text_row(ui, "Member", member, "the workbench model name");
            ui.horizontal(|ui| {
                ui.label("Samples");
                ui.add(DragValue::new(samples).range(1..=16));
            });
        }
        BackendConfig::Local {
            name,
            device,
            max_tokens,
            max_image_tokens,
        } => {
            text_row(ui, "Name", name, "shown in the window");
            ui.horizontal(|ui| {
                ui.label("Device");
                device_combo(ui, "local-device", device);
                ui.label("Max tokens");
                ui.add(DragValue::new(max_tokens).range(16..=8192));
                ui.label("Max image tokens");
                ui.add(DragValue::new(max_image_tokens).range(64..=6000))
                    .on_hover_text(
                        "the image is downscaled to fit; a whole page above ~2048 has lost the GPU",
                    );
            });
        }
    }
}
