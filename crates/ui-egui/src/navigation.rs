//! Photo navigation shared by loupe, comparisons and the enhancement preview.
use crate::{LightcraftApp, state::Zoom};
use egui::{Pos2, Rect};
use serde_json::{Value, json};

pub const KEY_COMMANDS: &[&str] = &["view.zoomIn", "view.zoomOut", "view.zoomFit", "view.zoom100", "view.zoomToggle"];
pub fn shortcut<'a>(app: &'a LightcraftApp, id: &str, default: Option<&'a str>) -> Option<&'a str> {
    crate::shortcuts::binding(&app.ui.settings.keymap, id, default)
}

/// Fold the fork's earlier zoom bindings into the general editor without replacing newer choices.
pub fn migrate(settings: &mut crate::state::AppSettings) {
    for (id, key) in std::mem::take(&mut settings.navigation.keys) {
        if KEY_COMMANDS.contains(&id.as_str()) && (key.is_empty() || crate::shortcuts::parse(&key).is_some()) {
            settings.keymap.entry(id).or_insert(key);
        }
    }
}

pub fn binding(app: &mut LightcraftApp, p: &Value) -> Result<Value, String> {
    let id = p.get("command").and_then(Value::as_str).filter(|id| KEY_COMMANDS.contains(id)).ok_or("Choose a photo navigation command")?;
    if p.get("reset") == Some(&Value::Bool(true)) {
        app.ui.settings.navigation.keys.remove(id);
        crate::shortcuts::reset(&mut app.ui.settings.keymap, id)?;
        return Ok(Value::Null);
    }
    let key = p.get("shortcut").and_then(Value::as_str).filter(|s| s.len() <= 64).ok_or("shortcut string required (empty disables the key)")?;
    if !key.is_empty() {
        let normalize = |s: &str| {
            let (mut m, k) = crate::shortcuts::parse(s)?;
            if !cfg!(target_os = "macos") && m.ctrl {
                m.command = true;
                m.ctrl = false;
            }
            Some((m, k))
        };
        let parts: Vec<_> = key.split('+').collect();
        let keys = parts.iter().filter(|part| !["Ctrl", "Cmd", "Alt", "Shift"].contains(part)).count();
        if keys != 1 || parts.iter().any(|part| part.is_empty()) || normalize(key).is_none() {
            return Err("Use one key with optional Ctrl, Cmd, Alt or Shift modifiers".into());
        }
        let wanted = normalize(key);
        let reserved = crate::menus::ui_commands()
            .filter(|c| c.0 != id)
            .filter_map(|(other, _, default, _)| shortcut(app, other, *default))
            .chain(lightcraft_engine::command_specs().iter().filter_map(|c| c.shortcut))
            .chain(crate::shortcuts::ALIASES.iter().filter(|(_, command, _)| *command != id).map(|(s, _, _)| *s));
        if reserved.into_iter().any(|s| normalize(s) == wanted)
            || (0..=9).any(|n| [n.to_string(), format!("Shift+{n}")].iter().any(|s| normalize(s) == wanted))
        {
            return Err("That shortcut already has an action. Choose another binding.".into());
        }
    }
    crate::shortcuts::assign(&mut app.ui.settings.keymap, id, Some(key))?;
    Ok(json!({"command":id,"shortcut":key}))
}

pub fn wheel(ctx: &egui::Context, area: Rect, binding: crate::state::WheelModifier) -> Option<(Pos2, f32)> {
    ctx.input(|i| {
        let q = i.pointer.hover_pos().filter(|q| area.contains(*q))?;
        let delta: f32 = i
            .events
            .iter()
            .filter_map(|e| match e {
                egui::Event::MouseWheel { delta, unit, modifiers, .. } if binding.matches(*modifiers) => {
                    let scale = match unit {
                        egui::MouseWheelUnit::Point => 1.0,
                        egui::MouseWheelUnit::Line => 40.0,
                        egui::MouseWheelUnit::Page => 300.0,
                    };
                    Some(delta.y * scale)
                }
                _ => None,
            })
            .sum();
        (delta.is_finite() && delta != 0.0).then_some((q, (delta / 300.0).clamp(-1.0, 1.0).exp()))
    })
}

pub fn zoom_at(app: &mut LightcraftApp, p: &Value) -> Result<Value, String> {
    let number = |key: &str| {
        p.get(key)
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite())
            .map(|n| n as f32)
            .filter(|n| n.is_finite())
            .ok_or_else(|| format!("view.zoomAt: finite {key} required"))
    };
    let factor = number("factor")?;
    if !(0.1..=10.0).contains(&factor) {
        return Err("view.zoomAt: factor must be 0.1–10".into());
    }
    let q = egui::pos2(number("x")?, number("y")?);
    if q.x.abs() > 1e7 || q.y.abs() > 1e7 {
        return Err("view.zoomAt: pointer is out of bounds".into());
    }
    let rectangle = |key: &str, fallback: Option<Rect>| -> Result<Rect, String> {
        let Some(value) = p.get(key) else {
            return fallback.ok_or_else(|| format!("No {key} on screen"));
        };
        let a = value.as_array().filter(|a| a.len() == 4).ok_or("Rectangle needs four numbers")?;
        let n =
            |i| a.get(i).and_then(Value::as_f64).filter(|n| n.is_finite() && n.abs() <= 1e7).map(|n| n as f32).ok_or("Invalid rectangle coordinate");
        let (x, y, w, h) = (n(0)?, n(1)?, n(2)?, n(3)?);
        if w < 0.01 || h < 0.01 {
            return Err("Empty rectangle".into());
        }
        Ok(Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h)))
    };
    let img = rectangle("image", app.image_rect)?;
    let area = rectangle("viewport", app.canvas_rect.map(|r| r.shrink(24.0)))?;
    let id = p.get("photo").and_then(Value::as_u64).map(lightcraft_catalog::PhotoId).or_else(|| app.session.active());
    let photo = id.and_then(|id| app.session.catalog.photo(id)).ok_or("Select a photo")?;
    let ppp = p.get("pixelsPerPoint").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    if !ppp.is_finite() || !(0.1..=16.0).contains(&ppp) {
        return Err("view.zoomAt: invalid pixelsPerPoint".into());
    }
    let cur = img.width() * ppp / photo.width.max(1) as f32 * 100.0;
    if !cur.is_finite() || cur <= 0.0 {
        return Err("view.zoomAt: invalid image size".into());
    }
    let percent = (cur * factor).clamp(6.0, 1600.0);
    let new_size = img.size() * (percent / cur.max(0.001));
    if !new_size.x.is_finite() || !new_size.y.is_finite() {
        return Err("view.zoomAt: invalid zoom size".into());
    }
    let anchor = (q - img.min) / img.size().max(egui::vec2(1.0, 1.0));
    let center = q + (egui::vec2(0.5, 0.5) - anchor) * new_size;
    app.ui.pan = (
        (0.5 + (area.center().x - center.x) / new_size.x.max(1.0)).clamp(0.0, 1.0),
        (0.5 + (area.center().y - center.y) / new_size.y.max(1.0)).clamp(0.0, 1.0),
    );
    app.ui.zoom = Zoom::Percent(percent);
    app.ui.zoom_anim = false;
    Ok(json!({"zoom":app.ui.zoom,"pan":app.ui.pan}))
}

pub fn image_wheel(app: &mut LightcraftApp, ctx: &egui::Context, area: Rect, img: Rect, photo: lightcraft_catalog::PhotoId) {
    if app.ui.dialog.is_none()
        && let Some((q, factor)) = wheel(ctx, area, app.ui.settings.navigation.wheel_modifier)
    {
        let _ = app.run("view.zoomAt", json!({"photo":photo.0,"x":q.x,"y":q.y,"factor":factor,"pixelsPerPoint":ctx.pixels_per_point(),"viewport":[area.left(),area.top(),area.width(),area.height()],"image":[img.left(),img.top(),img.width(),img.height()]}));
    }
}

/// Navigation takes precedence over editing only for an explicit navigation gesture.
pub fn pan_requested(ctx: &egui::Context, resp: &egui::Response, settings: &crate::state::NavigationSettings, tool: bool) -> bool {
    let button = settings.pan_button.button();
    let held = ctx.input(|i| settings.pan_modifier.matches(i.modifiers));
    if resp.dragged_by(egui::PointerButton::Middle) || resp.drag_started_by(egui::PointerButton::Middle) {
        return true;
    }
    let gesture = resp.dragged_by(button) || resp.drag_started_by(button);
    gesture && (!tool || button == egui::PointerButton::Middle || held)
}

pub fn pan(app: &mut LightcraftApp, ctx: &egui::Context, resp: &egui::Response, img: Rect) {
    if pan_requested(ctx, resp, &app.ui.settings.navigation, false) {
        let delta = resp.drag_delta() / img.size().max(egui::vec2(1.0, 1.0));
        let _ = app.run("view.pan", json!({"dx":-delta.x,"dy":-delta.y}));
        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
    }
}

pub fn help(app: &LightcraftApp, ui: &mut egui::Ui) -> bool {
    use crate::i18n::tr;
    let prefs = &app.ui.settings.navigation;
    ui.heading(tr("Photo Navigation"));
    ui.label(tr("Move the pointer over the photo to zoom. The point under it stays in place."));
    egui::Grid::new("navigation-help").spacing([20.0, 8.0]).show(ui, |ui| {
        ui.label(tr("Zoom with wheel"));
        ui.label(match prefs.wheel_modifier {
            crate::state::WheelModifier::Disabled => tr("Disabled").to_string(),
            crate::state::WheelModifier::None => tr("Scroll").to_string(),
            value => crate::i18n::tr_format!("{} + scroll", tr(value.label())),
        });
        ui.end_row();
        ui.label(tr("Pan while zoomed"));
        ui.label(tr(prefs.pan_button.label()));
        ui.end_row();
        ui.label(tr("Pan with an editing tool"));
        ui.label(if prefs.pan_button == crate::state::PanButton::Middle || prefs.pan_modifier == crate::state::WheelModifier::Disabled {
            tr("Middle drag").to_string()
        } else {
            crate::i18n::tr_format!("{} + left drag", tr(prefs.pan_modifier.label()))
        });
        ui.end_row();
        for (id, label, default, _) in crate::menus::ui_commands().filter(|c| KEY_COMMANDS.contains(&c.0)) {
            let key = shortcut(app, id, *default).unwrap_or("Disabled");
            ui.label(tr(label));
            ui.label(if cfg!(target_os = "macos") { key.to_string() } else { key.replace("Cmd", "Ctrl") });
            ui.end_row();
        }
    });
    ui.label(tr("The Navigator moves to another part of the photo. AI Denoise has its own crop navigator; moving it prepares a new preview without applying edits."));
    let r = ui.button(tr("Change Navigation Settings…"));
    crate::widgets::register(ui.ctx(), "help:navigation", r.rect);
    r.clicked()
}
