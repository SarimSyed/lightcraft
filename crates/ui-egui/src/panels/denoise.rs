//! Focused, single-photo enhancement preview; temporary amount changes never touch history.
use crate::i18n::tr;
use crate::{LightcraftApp, state::Dialog, theme::Tokens, widgets::register};
use egui::{Align2, Color32, Id, RichText, TextureOptions, vec2};
use serde_json::{Value, json};
use std::sync::Arc;
type PreviewTexture = ((u64, u64, bool), egui::TextureHandle, Arc<egui::ColorImage>);
pub(crate) fn displayed_before(ctx: &egui::Context) -> Option<bool> {
    ctx.data(|d| d.get_temp::<PreviewTexture>(Id::new("denoise-preview-texture"))).map(|cache| cache.0.2)
}
pub(crate) fn cpu_texture(ctx: &egui::Context) -> Option<(egui::TextureId, crate::softpaint::CpuTexture)> {
    let (_, texture, image) = ctx.data(|d| d.get_temp::<PreviewTexture>(Id::new("denoise-preview-texture")))?;
    Some((
        texture.id(),
        crate::softpaint::CpuTexture { image, magnification: egui::TextureFilter::Nearest, minification: egui::TextureFilter::Nearest },
    ))
}

pub fn open(app: &mut LightcraftApp) -> Result<Value, String> {
    if cfg!(target_arch = "wasm32") {
        return Err("AI denoise requires native LightCraft".into());
    }
    let id = app.session.active().ok_or("Select one photo for AI Denoise")?;
    let photo = app.session.catalog.photo(id).ok_or("Photo no longer exists")?;
    let amount = if photo.develop.enhance.model.is_some() { photo.develop.enhance.denoise } else { 50.0 };
    app.session.enhancer.cancel();
    app.session.enhancer.preview = None;
    app.ui.dialog = Some(Dialog::Denoise { photo: id.0, amount });
    if app.session.denoise_model_status()["installed"] == true || photo.develop.enhance.model.is_some() {
        prepare(app, id.0, amount)?;
    }
    Ok(Value::Null)
}
fn prepare(app: &mut LightcraftApp, photo: u64, amount: f64) -> Result<Value, String> {
    let p = app.session.catalog.photo(lightcraft_engine::catalog::PhotoId(photo)).ok_or("Photo no longer exists")?;
    let (w, h) = (p.width.max(1) as usize, p.height.max(1) as usize);
    let (width, height) = (w.min(512), h.min(320));
    app.run(
        "enhance.denoise.preview",
        json!({"photo":photo,"amount":amount,"wait":false,
        "region":{"x":(w-width)/2,"y":(h-height)/2,"width":width,"height":height}}),
    )
}
pub fn show(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(Dialog::Denoise { photo, mut amount }) = app.ui.dialog.clone() else {
        return;
    };
    // Space is the dialog's Before gesture, even when Apply has keyboard focus.
    ctx.input_mut(|i| {
        i.consume_key(egui::Modifiers::NONE, egui::Key::Space);
    });
    let status = app.session.enhancer.status();
    if status.applying && status.state == "success" {
        app.ui.dialog = None;
        return;
    }
    let running = matches!(status.state.as_str(), "preparing" | "processing");
    if running {
        ctx.request_repaint_after(std::time::Duration::from_millis(40));
    }
    let model = app.session.denoise_model_status();
    let installed = model["installed"] == true;
    let download_running = model["download"]["state"] == "downloading";
    let t = Tokens::get(ctx);
    let mut close = false;
    let mut apply = false;
    let mut retry = false;
    egui::Area::new(Id::new("dialog-dim")).order(egui::Order::Middle).fixed_pos(ctx.content_rect().min).interactable(false).show(ctx, |ui| {
        ui.painter().rect_filled(ctx.content_rect(), 0.0, Color32::from_black_alpha(140));
    });
    egui::Window::new(tr("AI Denoise"))
        .id(Id::new("lightcraft-dialog"))
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(560.0)
        .frame(egui::Frame::window(&ctx.global_style()).inner_margin(egui::Margin::symmetric(16, 12)))
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 10.0;
            if let Some(p) = app.session.catalog.photo(lightcraft_engine::catalog::PhotoId(photo)) {
                ui.label(&p.file_name);
            }
            if !installed && app.session.enhancer.preview.is_none() && !running {
                ui.label(tr("Install NAFNet to prepare the preview."));
                ui.label(tr("Model size: 116.7 MB. Inference runs locally."));
                ui.hyperlink_to(tr("Model licence and notices"), "https://github.com/megvii-research/NAFNet/blob/main/LICENSE");
                ui.label(
                    RichText::new(tr("Download is awaiting checkpoint redistribution approval. Offline installation is available."))
                        .color(t.text_dim),
                );
                ui.label(app.session.enhancer.model_dir.to_string_lossy());
                ui.label(RichText::new(tr("See docs/denoise.md for verified offline installation.")).color(t.text_dim));
                let r = ui.add_enabled(model["redistributionVerified"] == true && !download_running, egui::Button::new(tr("Download Model")));
                register(ctx, "denoise:download", r.rect);
                if r.clicked() {
                    let _ = app.run("enhance.model.download", json!({}));
                }
                let r = ui.button(tr("Check Installation"));
                register(ctx, "denoise:checkInstallation", r.rect);
                if r.clicked() && installed {
                    retry = true;
                }
            }
            if download_running {
                let done = model["download"]["done"].as_f64().unwrap_or(0.0);
                let total = model["download"]["total"].as_f64().unwrap_or(1.0).max(1.0);
                ui.add(egui::ProgressBar::new((done / total) as f32).show_percentage());
                let r = ui.button(tr("Cancel Download"));
                register(ctx, "denoise:cancelDownload", r.rect);
                if r.clicked() {
                    let _ = app.run("enhance.model.cancel", json!({}));
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            if let Some(error) = model["download"]["error"].as_str() {
                ui.label(RichText::new(error).color(Color32::from_rgb(230, 90, 80)));
            }
            if running {
                ui.label(tr(if status.applying { "Processing AI Denoise…" } else { "Preparing preview…" }));
                let fraction = status.done as f32 / status.total.max(1) as f32;
                let r = ui.add(egui::ProgressBar::new(fraction).show_percentage());
                register(ctx, "denoise:progress", r.rect);
            }
            let before_button = ui.add_enabled(app.session.enhancer.preview.is_some() && !running, egui::Button::new(tr("Hold Before")));
            register(ctx, "denoise:before", before_button.rect);
            let before = ctx
                .data(|d| d.get_temp::<bool>(Id::new("denoise-before-override")))
                .unwrap_or_else(|| before_button.is_pointer_button_down_on() || ctx.input(|i| i.key_down(egui::Key::Space)));
            if let Some(preview) = &app.session.enhancer.preview {
                let key = (preview.operation, amount.to_bits(), before);
                let texture_id = Id::new("denoise-preview-texture");
                let stored = ctx.data(|d| d.get_temp::<PreviewTexture>(texture_id));
                let texture = match stored {
                    Some((old, texture, _)) if old == key => Some(texture),
                    _ => match preview.render(amount, before) {
                        Ok(image) => {
                            let data: Vec<u8> = image.data.iter().flatten().copied().collect();
                            let pixels = Arc::new(egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &data));
                            let texture = ctx.load_texture("AI denoise preview", (*pixels).clone(), TextureOptions::NEAREST);
                            ctx.data_mut(|d| d.insert_temp(texture_id, (key, texture.clone(), pixels)));
                            Some(texture)
                        }
                        Err(error) => {
                            ui.label(error);
                            None
                        }
                    },
                };
                if let Some(texture) = texture {
                    let size = vec2(texture.size()[0] as f32, texture.size()[1] as f32) / ctx.pixels_per_point();
                    egui::ScrollArea::both().max_height(320.0).show(ui, |ui| {
                        let r = ui.image((texture.id(), size));
                        register(ctx, "denoise:preview", r.rect);
                    });
                }
                ui.label(RichText::new(tr("100% preview · Hold Before or Space to compare")).color(t.text_dim));
            }
            if let Some(spec) = lightcraft_develop::controls::find("enhance.denoise") {
                let out = crate::widgets::slider(ui, spec, amount, !running && app.session.enhancer.preview.is_some(), None);
                if let Some(value) = out.value {
                    amount = value;
                }
                if out.reset {
                    amount = 50.0;
                }
            }
            if let Some(error) = &status.error {
                ui.label(RichText::new(error).color(Color32::from_rgb(230, 90, 80)));
            }
            if !status.backend.is_empty() {
                ui.label(RichText::new(format!("{}: {}", tr("Processing device"), status.backend)).color(t.text_dim));
            }
            if status.cpu_fallback {
                ui.label(tr("GPU was unavailable. Processing completed on CPU."))
                    .on_hover_text(status.fallback_reason.as_deref().unwrap_or_default());
            }
            ui.horizontal(|ui| {
                let r = ui.button(tr("Cancel"));
                register(ctx, "denoise:cancel", r.rect);
                close |= r.clicked();
                if installed && !running && app.session.enhancer.preview.is_none() {
                    let r = ui.button(tr("Prepare Preview"));
                    register(ctx, "denoise:retry", r.rect);
                    retry |= r.clicked();
                }
                let r = ui.add_enabled(!running && app.session.enhancer.preview.is_some(), egui::Button::new(tr("Apply")));
                register(ctx, "denoise:apply", r.rect);
                apply |= r.clicked();
            });
        });
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        close = true;
    }
    if close {
        let _ = app.run("enhance.denoise.cancel", json!({}));
        app.ui.dialog = None;
    } else if apply {
        match app.run("enhance.denoise.apply", json!({"photo":photo,"amount":amount,"wait":false})) {
            Ok(_) => app.ui.dialog = Some(Dialog::Denoise { photo, amount }),
            Err(e) => app.ui.status = e,
        }
    } else {
        app.ui.dialog = Some(Dialog::Denoise { photo, amount });
        if retry {
            let _ = prepare(app, photo, amount);
        }
    }
}
