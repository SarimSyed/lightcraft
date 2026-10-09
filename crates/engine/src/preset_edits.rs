//! Keep preset-free settings alongside the ordinary photo history. Never undo another edit to
//! change a preset amount. Old catalogs can recover provenance only from retained snapshots.
use std::sync::Arc;

use lightcraft_catalog::{HistoryStep, Op, Photo, PhotoId, PresetEdit};
use lightcraft_develop::DevelopSettings;

use crate::{Session, json_delta};

pub(crate) const INCOMPLETE: &str =
    "The settings before this older preset are missing. Restore a known pre-preset version or undo the preset; manual edits were left unchanged.";

pub(crate) fn carry(before: &DevelopSettings, after: &DevelopSettings, without: &DevelopSettings) -> DevelopSettings {
    let (before, after, base) = (before.to_json(), after.to_json(), without.to_json());
    let Some(mut patch) = json_delta(&before, &after) else { return without.clone() };
    // An added manual mask must not bring every unchanged preset mask along with it. List
    // entries have stable IDs; edited entries are manual overrides, unchanged ones stay absent.
    for key in ["masks", "spots", "red_eye", "point_colors"] {
        let (Some(old), Some(new), Some(original)) = (before[key].as_array(), after[key].as_array(), base[key].as_array()) else { continue };
        if old == new || !old.iter().chain(new).chain(original).all(|v| v.get("id").is_some()) {
            continue;
        }
        let mut entries = original.clone();
        entries.retain(|v| !old.iter().any(|o| o["id"] == v["id"]) || new.iter().any(|n| n["id"] == v["id"]));
        for value in new {
            let previous = old.iter().find(|v| v["id"] == value["id"]);
            if previous == Some(value) {
                continue;
            }
            if let Some(existing) = entries.iter_mut().find(|v| v["id"] == value["id"]) {
                let delta = previous.and_then(|v| json_delta(v, value)).unwrap_or_else(|| value.clone());
                lightcraft_develop::presets::deep_merge(existing, &delta);
            } else {
                entries.push(value.clone());
            }
        }
        if let Some(object) = patch.as_object_mut() {
            object.insert(key.into(), entries.into());
        }
    }
    without.merged(&patch).unwrap_or_else(|_| without.clone())
}

pub(crate) fn state(photo: &Photo) -> Option<PresetEdit> {
    if let Some(step) = photo.history.last() {
        if let Some(preset) = &step.preset {
            return Some(preset.clone());
        }
        if step.label == "Remove Preset Effects" {
            return None;
        }
    }
    let start = photo.history.iter().rposition(|h| h.label == "Reset" || h.label == "Remove Preset Effects").unwrap_or(0);
    let history = photo.history.get(start..).unwrap_or_default();
    let mut without = photo.camera_defaults();
    let mut previous = photo.import_defaults();
    let mut found = photo.import_look.is_some();
    if history.first().is_some_and(|h| h.label == "Remove Preset Effects") {
        found = false;
    }
    if !found && !history.iter().any(|h| h.label.starts_with("Preset: ")) {
        return None;
    }
    for (index, step) in history.iter().enumerate() {
        if step.label.starts_with("Preset: ") {
            if index == 0 {
                return Some(PresetEdit { without: photo.develop.clone(), last: None, incomplete: true });
            }
            found = true;
        } else {
            without = carry(&previous, &step.settings, &without);
        }
        previous = (*step.settings).clone();
    }
    without = carry(&previous, &photo.develop, &without);
    found.then(|| PresetEdit { without: Arc::new(without), last: None, incomplete: false })
}

/// Override the history metadata for a preset operation; the ordinary develop op would
/// otherwise interpret its changed settings as manual edits.
pub(crate) fn op(s: &Session, id: PhotoId, settings: DevelopSettings, label: &str, preset: Option<PresetEdit>) -> Option<Op> {
    let mut op = s.develop_op(id, settings, label)?;
    if let Op::Batch { ops } = &mut op {
        for op in ops {
            if let Op::PushHistory { step, .. } = op {
                step.preset = preset.clone();
            }
        }
    }
    Some(op)
}

pub(crate) fn manual_step(photo: &Photo, settings: Arc<DevelopSettings>, label: &str) -> HistoryStep {
    let preset = if label == "Reset" {
        photo.import_look.as_ref().map(|_| PresetEdit { without: Arc::new(photo.camera_defaults()), last: None, incomplete: false })
    } else {
        state(photo).map(|mut p| {
            p.without = Arc::new(carry(&photo.develop, &settings, &p.without));
            p
        })
    };
    HistoryStep { label: label.into(), settings, preset }
}
