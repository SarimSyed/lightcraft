//! Focused, single-photo enhancement preview; temporary amount changes never touch history.
use crate::i18n::tr;
use crate::{
    LightcraftApp,
    state::{DenoiseView, Dialog},
    theme::Tokens,
    widgets::register,
};
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
    app.ui.denoise_due = None;
    app.ui.dialog = Some(Dialog::Denoise { photo: id.0, amount, view: Default::default() });
    if app.session.denoise_model_status()["installed"] == true || photo.develop.enhance.model.is_some() {
        prepare(app, id.0, amount, DenoiseView::default())?;
    }
    Ok(Value::Null)
}
fn region(app: &LightcraftApp, photo: u64, view: DenoiseView) -> Result<lightcraft_engine::enhance::Region, String> {
    let p = app.session.catalog.photo(lightcraft_engine::catalog::PhotoId(photo)).ok_or("Photo no longer exists")?;
    let (w, h) = (p.width.max(1) as usize, p.height.max(1) as usize);
    let (width, height) = (w.min(51200 / view.zoom.max(50) as usize), h.min(32000 / view.zoom.max(50) as usize));
    Ok(lightcraft_engine::enhance::Region {
        x: (view.center[0] * w as f32 - width as f32 / 2.0).round().clamp(0.0, (w - width) as f32) as usize,
        y: (view.center[1] * h as f32 - height as f32 / 2.0).round().clamp(0.0, (h - height) as f32) as usize,
        width,
        height,
    })
}
fn prepare(app: &mut LightcraftApp, photo: u64, amount: f64, view: DenoiseView) -> Result<Value, String> {
    let region = region(app, photo, view)?;
    if app.session.enhancer.preview.as_ref().is_some_and(|p| {
        p.region == region && app.session.develop_of(lightcraft_catalog::PhotoId(photo)).is_some_and(|s| s.hash64() == p.settings.hash64())
    }) {
        return Ok(Value::Null);
    }
    app.run("enhance.denoise.preview", json!({"photo":photo,"amount":amount,"wait":false,"region":region}))
}
pub fn navigate(app: &mut LightcraftApp, params: &Value) -> Result<Value, String> {
    let Some(Dialog::Denoise { photo, amount, view }) = app.ui.dialog.clone() else {
        return Err("Open AI Denoise first".into());
    };
    if app.session.enhancer.status().applying {
        return Err("Wait for Apply or cancel processing before navigating".into());
    }
    let mut next = view;
    if let Some(value) = params.get("center") {
        let a = value.as_array().filter(|a| a.len() == 2).ok_or("center needs two normalized numbers")?;
        let n = |i| {
            a.get(i).and_then(Value::as_f64).filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).map(|v| v as f32).ok_or("center must be in 0..1")
        };
        next.center = [n(0)?, n(1)?];
    }
    if let Some(value) = params.get("zoom") {
        next.zoom = value.as_u64().filter(|v| (50..=800).contains(v)).ok_or("preview zoom must be 50–800 percent")? as u32;
    }
    if next != view {
        app.session.enhancer.cancel();
        app.ui.denoise_due = Some(std::time::Instant::now() + std::time::Duration::from_millis(150));
        app.ui.dialog = Some(Dialog::Denoise { photo, amount, view: next });
    }
    Ok(json!(next))
}
pub fn apply(app: &mut LightcraftApp, photo: u64, amount: f64, view: DenoiseView) -> Result<Value, String> {
    let status = app.session.enhancer.status();
    let expected = region(app, photo, view)?;
    if app.ui.denoise_due.is_some()
        || matches!(status.state.as_str(), "preparing" | "processing")
        || app.session.enhancer.preview.as_ref().is_none_or(|p| p.region != expected)
    {
        return Err("Wait for the current AI denoise preview before applying".into());
    }
    app.run("enhance.denoise.apply", json!({"photo":photo,"amount":amount,"wait":false}))
}
pub fn show(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(Dialog::Denoise { photo, mut amount, mut view }) = app.ui.dialog.clone() else {
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
    let mut next_view = view;
    let pending = app.ui.denoise_due.is_some();
    let stale = app
        .session
        .enhancer
        .preview
        .as_ref()
        .is_some_and(|p| app.session.develop_of(lightcraft_catalog::PhotoId(photo)).is_some_and(|s| s.hash64() != p.settings.hash64()));
    if stale && !running && !status.applying && !pending {
        if let Err(error) = prepare(app, photo, amount, view) {
            app.ui.status = error;
        }
        ctx.request_repaint();
    }
    if pending {
        ctx.request_repaint_after(std::time::Duration::from_millis(30));
    }
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
            ui.label(RichText::new(tr("RGB-stage denoise · Higher amounts remove more noise and may soften fine texture.")).color(t.text_dim));
            ui.label(RichText::new(tr("Current colour and Detail edits · Preview before geometry.")).color(t.text_dim));
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
            if running || app.session.enhancer.preview.is_some() {
                let fraction = status.done as f32 / status.total.max(1) as f32;
                let label = if status.applying {
                    "Processing AI Denoise…"
                } else if running || pending {
                    "Preparing preview…"
                } else {
                    "Preview ready"
                };
                let r = ui.add(egui::ProgressBar::new(fraction).text(tr(label)));
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
                    let target = region(app, photo, view).unwrap_or(preview.region);
                    // Keep the viewport steady while its old pixels wait for a new source crop.
                    let size = vec2(target.width as f32, target.height as f32) * view.zoom as f32 / 100.0 / ctx.pixels_per_point();
                    ui.scope(|ui| {
                        let r = ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click_and_drag()));
                        register(ctx, "denoise:preview", r.rect);
                        if !status.applying {
                            if let Some((q, factor)) = crate::navigation::wheel(ctx, r.rect, app.ui.settings.navigation.wheel_modifier) {
                                let zoom = (view.zoom as f32 * factor).round().clamp(50.0, 800.0) as u32;
                                let p = target;
                                let source = app.session.catalog.photo(lightcraft_catalog::PhotoId(photo));
                                if let Some(source) = source {
                                    let anchor = (q - r.rect.min) / r.rect.size();
                                    let ratio = view.zoom as f32 / zoom as f32;
                                    next_view.center = [
                                        ((p.x as f32 + anchor.x * p.width as f32 + (0.5 - anchor.x) * p.width as f32 * ratio)
                                            / source.width.max(1) as f32)
                                            .clamp(0.0, 1.0),
                                        ((p.y as f32 + anchor.y * p.height as f32 + (0.5 - anchor.y) * p.height as f32 * ratio)
                                            / source.height.max(1) as f32)
                                            .clamp(0.0, 1.0),
                                    ];
                                }
                                next_view.zoom = zoom;
                            }
                            if crate::navigation::pan_requested(ctx, &r, &app.ui.settings.navigation, false) {
                                let d = r.drag_delta() * 100.0 / view.zoom as f32 * ctx.pixels_per_point();
                                if let Some(source) = app.session.catalog.photo(lightcraft_catalog::PhotoId(photo)) {
                                    next_view.center = [
                                        (view.center[0] - d.x / source.width.max(1) as f32).clamp(0.0, 1.0),
                                        (view.center[1] - d.y / source.height.max(1) as f32).clamp(0.0, 1.0),
                                    ];
                                }
                                ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                            }
                        }
                    });
                }
                ui.label(RichText::new(format!("{}% · {}", view.zoom, tr("Hold Before or Space to compare"))).color(t.text_dim));
            }
            if app.session.enhancer.preview.is_some() {
                navigator(app, ui, photo, &mut next_view, !status.applying);
            }
            ui.horizontal(|ui| {
                let r = ui.add_enabled(!status.applying, egui::Button::new(tr("100%")));
                register(ctx, "denoise:zoom100", r.rect);
                if r.clicked() {
                    next_view.zoom = 100;
                }
                ui.label(tr("Use the crop navigator to inspect another area. Arrow keys move the crop."));
            });
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
                let r = ui
                    .add_enabled(!running && !pending && next_view == view && app.session.enhancer.preview.is_some(), egui::Button::new(tr("Apply")));
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
        app.ui.denoise_due = None;
    } else if apply {
        match self::apply(app, photo, amount, view) {
            Ok(_) => app.ui.dialog = Some(Dialog::Denoise { photo, amount, view }),
            Err(e) => app.ui.status = e,
        }
    } else {
        if !status.applying {
            let mut shift = [0.0, 0.0];
            ctx.input_mut(|i| {
                for (key, x, y) in [
                    (egui::Key::ArrowLeft, -1.0, 0.0),
                    (egui::Key::ArrowRight, 1.0, 0.0),
                    (egui::Key::ArrowUp, 0.0, -1.0),
                    (egui::Key::ArrowDown, 0.0, 1.0),
                ] {
                    if i.consume_key(egui::Modifiers::NONE, key) {
                        shift[0] += x;
                        shift[1] += y;
                    }
                }
            });
            if let Some(p) = app.session.catalog.photo(lightcraft_catalog::PhotoId(photo)) {
                next_view.center[0] = (next_view.center[0] + shift[0] * 5120.0 / view.zoom as f32 / p.width.max(1) as f32).clamp(0.0, 1.0);
                next_view.center[1] = (next_view.center[1] + shift[1] * 3200.0 / view.zoom as f32 / p.height.max(1) as f32).clamp(0.0, 1.0);
            }
        }
        if next_view != view {
            let _ = navigate(app, &json!(next_view));
            view = next_view;
        }
        app.ui.dialog = Some(Dialog::Denoise { photo, amount, view });
        if retry {
            let _ = prepare(app, photo, amount, view);
        } else if app.ui.denoise_due.is_some_and(|at| std::time::Instant::now() >= at) && !ctx.input(|i| i.pointer.any_down()) {
            app.ui.denoise_due = None;
            if let Err(error) = prepare(app, photo, amount, view) {
                app.ui.status = error;
            }
        }
    }
}

fn navigator(app: &mut LightcraftApp, ui: &mut egui::Ui, photo: u64, view: &mut DenoiseView, enabled: bool) {
    use crate::render::Slot;
    let id = lightcraft_catalog::PhotoId(photo);
    if let Some(job) = app.session.render_job(id, 180, 110, true, false) {
        app.renderer.request(Slot::DenoiseNavigator, job, 110);
    }
    let Some(source) = app.session.catalog.photo(id) else {
        return;
    };
    let size = vec2(source.width.max(1) as f32, source.height.max(1) as f32);
    let display = size * (64.0 / size.max_elem());
    ui.horizontal(|ui| {
        let (rect, resp) = ui.allocate_exact_size(display, egui::Sense::click_and_drag());
        register(ui.ctx(), "denoise:navigator", rect);
        if let Some(t) = app.renderer.textures.get(&Slot::DenoiseNavigator).filter(|t| t.photo == id) {
            ui.painter().image(t.tex.id(), rect, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        if enabled
            && (resp.clicked() || resp.dragged())
            && let Some(q) = resp.interact_pointer_pos()
        {
            let n = (q - rect.min) / rect.size();
            view.center = [n.x.clamp(0.0, 1.0), n.y.clamp(0.0, 1.0)];
        }
        if let Ok(region) = region(app, photo, *view) {
            let a = rect.min + vec2(region.x as f32, region.y as f32) / size * rect.size();
            let b = a + vec2(region.width as f32, region.height as f32) / size * rect.size();
            ui.painter().rect_stroke(egui::Rect::from_min_max(a, b), 0.0, egui::Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Inside);
        }
        ui.label(tr("Crop navigator"));
    });
}
