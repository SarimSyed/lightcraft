use lightcraft_engine::Session;
use serde_json::json;

#[test]
fn remove_preset_effects_preserves_manual_edits_and_is_undoable() {
    let mut s = Session::with_demo();
    s.execute("develop.set", &json!({"values":{"light.exposure":1.0,"light.contrast":40}})).unwrap();
    let preset = s.execute("preset.create", &json!({"name":"Test light","groups":["light"]})).unwrap()["id"].clone();
    s.execute("develop.reset", &json!({})).unwrap();
    s.execute("develop.set", &json!({"values":{"light.exposure":0.25,"effects.clarity":5}})).unwrap();
    let photo = s.active().unwrap();
    s.execute("preset.apply", &json!({"id":preset})).unwrap();
    s.execute("develop.set", &json!({"values":{"light.exposure":1.7,"wb.tint":12}})).unwrap();
    let applied = s.develop_of(photo).unwrap();
    s.execute("preset.remove", &json!({})).unwrap();
    let removed = s.develop_of(photo).unwrap();
    assert_eq!(removed.light.exposure, 1.7);
    assert_eq!(removed.light.contrast, 0.0);
    assert_eq!(removed.effects.clarity, 5.0);
    assert_eq!(removed.wb.tint, 12.0);
    s.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(s.develop_of(photo).unwrap(), applied);
    s.execute("edit.redo", &json!({})).unwrap();
    assert_eq!(s.develop_of(photo).unwrap(), removed);
}

#[test]
fn preset_amount_is_photo_local_and_keeps_later_manual_edits() {
    let mut s = Session::with_demo();
    s.execute("develop.set", &json!({"values":{"light.exposure":1.0,"light.contrast":40}})).unwrap();
    let preset = s.execute("preset.create", &json!({"name":"Test light","groups":["light"]})).unwrap()["id"].clone();
    s.execute("develop.reset", &json!({})).unwrap();
    s.execute("develop.set", &json!({"control":"light.exposure","value":0.25})).unwrap();
    let photo = s.active().unwrap();
    s.execute("preset.apply", &json!({"id":preset})).unwrap();
    s.execute("develop.set", &json!({"control":"light.exposure","value":1.7})).unwrap();
    let other = s.catalog.photos().find(|p| p.id != photo).unwrap().id;
    s.execute("library.select", &json!({"ids":[other.0],"active":other.0})).unwrap();
    s.execute("develop.set", &json!({"control":"effects.clarity","value":23})).unwrap();
    let other_settings = s.develop_of(other).unwrap();
    assert!(s.execute("preset.amount", &json!({"amount":50})).is_err(), "no preset on the other photo");
    s.execute("library.select", &json!({"ids":[photo.0,other.0],"active":photo.0})).unwrap();
    s.execute("preset.amount", &json!({"amount":50})).unwrap();
    assert_eq!(s.develop_of(photo).unwrap().light.exposure, 1.7);
    assert_eq!(s.develop_of(photo).unwrap().light.contrast, 20.0);
    assert_eq!(s.develop_of(other).unwrap(), other_settings);
    s.execute("preset.amount", &json!({"amount":25})).unwrap();
    assert_eq!(s.develop_of(photo).unwrap().light.contrast, 10.0);
    s.execute("preset.remove", &json!({})).unwrap();
    assert_eq!(s.develop_of(photo).unwrap().light.exposure, 1.7);
    assert_eq!(s.develop_of(photo).unwrap().light.contrast, 0.0);
    assert_eq!(s.develop_of(other).unwrap(), other_settings);
}

#[test]
fn removal_survives_reopening_and_history_clear() {
    let store = lightcraft_engine::catalog::MemStore::new();
    let mut s = Session::with_demo();
    // The public journal snapshot models reopening without an undo stack.
    s.execute("preset.apply", &json!({"id":"lc.bw-high-contrast"})).unwrap();
    s.execute("develop.set", &json!({"control":"effects.clarity","value":23})).unwrap();
    s.execute("history.clear", &json!({})).unwrap();
    let id = s.active().unwrap();
    let (mut journal, _, _) = lightcraft_engine::catalog::Journal::open(Box::new(store.clone())).unwrap();
    journal.snapshot(&s.catalog).unwrap();
    drop(journal);
    let (_, catalog, _) = lightcraft_engine::catalog::Journal::open(Box::new(store)).unwrap();
    let mut reopened = Session::new();
    reopened.catalog = catalog;
    reopened.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    reopened.execute("preset.remove", &json!({})).unwrap();
    let result = reopened.develop_of(id).unwrap();
    assert_eq!(result.treatment, lightcraft_develop::Treatment::Color);
    assert_eq!(result.light.contrast, 0.0);
    assert_eq!(result.effects.clarity, 23.0);
    assert_eq!(reopened.execute("preset.status", &json!({})).unwrap()["removable"], false);
}

#[test]
fn remove_all_presets_keeps_manual_masks() {
    let mut s = Session::with_demo();
    s.execute("mask.add", &json!({"kind":"radial","name":"Preset mask"})).unwrap();
    s.execute("mask.adjust", &json!({"values":{"exposure":1.0}})).unwrap();
    let preset = s.execute("preset.create", &json!({"name":"Test mask","groups":["masks"]})).unwrap()["id"].clone();
    s.execute("develop.reset", &json!({})).unwrap();
    s.execute("preset.apply", &json!({"id":preset})).unwrap();
    s.execute("preset.apply", &json!({"id":"lc.bw-high-contrast"})).unwrap();
    s.execute("mask.add", &json!({"kind":"radial","name":"Manual mask"})).unwrap();
    s.execute("mask.adjust", &json!({"values":{"exposure":0.5}})).unwrap();
    s.execute("preset.remove", &json!({})).unwrap();
    let result = s.develop_of(s.active().unwrap()).unwrap();
    assert_eq!(result.treatment, lightcraft_develop::Treatment::Color);
    assert_eq!(result.masks.len(), 1);
    assert_eq!(result.masks[0].name, "Manual mask");
    assert_eq!(result.masks[0].adjust.exposure, 0.5);
}

#[test]
fn import_default_removal_keeps_manual_edits() {
    use lightcraft_engine::catalog::{Op, Photo, PhotoId, Source};
    use std::sync::Arc;
    let mut s = Session::new();
    let mut photo = Photo::new(PhotoId(1), Source::File { path: "photo.nef".into() }, "photo.nef", "NEF", 24, 16, "");
    let mut look = photo.camera_defaults();
    look.light.contrast = 40.0;
    look.effects.clarity = 30.0;
    photo.develop = Arc::new(look);
    photo.import_look = Some(photo.develop.clone());
    s.commit("Import", Op::AddPhoto { photo: Box::new(photo) }).unwrap();
    s.execute("library.select", &json!({"ids":[1]})).unwrap();
    s.execute("develop.set", &json!({"control":"light.exposure","value":1.0})).unwrap();
    s.execute("preset.remove", &json!({})).unwrap();
    let result = s.develop_of(PhotoId(1)).unwrap();
    assert_eq!(result.light.exposure, 1.0);
    assert_eq!(result.light.contrast, 0.0);
    assert_eq!(result.effects.clarity, 0.0);
    s.execute("develop.set", &json!({"control":"effects.clarity","value":15.0})).unwrap();
    assert_eq!(s.execute("preset.status", &json!({})).unwrap()["removable"], false);
}

#[test]
fn old_presets_need_retained_pre_preset_settings() {
    use lightcraft_engine::catalog::{HistoryStep, Op};
    let mut s = Session::with_demo();
    let id = s.active().unwrap();
    let baseline = s.develop_of(id).unwrap();
    s.execute("preset.apply", &json!({"id":"lc.bw-high-contrast"})).unwrap();
    let applied = s.develop_of(id).unwrap();
    // A v2 catalog: labels and snapshots, without the new provenance field.
    s.commit(
        "Old history",
        Op::SetHistory { id, history: vec![HistoryStep { label: "Preset: High Contrast B&W".into(), settings: applied.clone(), preset: None }] },
    )
    .unwrap();
    assert!(s.execute("preset.remove", &json!({})).unwrap_err().to_string().contains("missing"));
    assert_eq!(s.develop_of(id).unwrap(), applied, "never guess an absent baseline");
    s.commit(
        "Old complete history",
        Op::SetHistory {
            id,
            history: vec![
                HistoryStep { label: "Exposure".into(), settings: baseline, preset: None },
                HistoryStep { label: "Preset: High Contrast B&W".into(), settings: applied, preset: None },
            ],
        },
    )
    .unwrap();
    s.execute("develop.set", &json!({"control":"effects.clarity","value":23})).unwrap();
    s.execute("preset.remove", &json!({})).unwrap();
    let result = s.develop_of(id).unwrap();
    assert_eq!(result.treatment, lightcraft_develop::Treatment::Color);
    assert_eq!(result.effects.clarity, 23.0);
}

#[test]
fn incomplete_legacy_history_does_not_block_applying_new_presets_or_copying() {
    use lightcraft_engine::catalog::{HistoryStep, Op};
    let mut s = Session::with_demo();
    let id = s.active().unwrap();
    s.execute("preset.apply", &json!({"id":"lc.bw-high-contrast"})).unwrap();
    let applied = s.develop_of(id).unwrap();
    s.commit(
        "Old history",
        Op::SetHistory { id, history: vec![HistoryStep { label: "Preset: High Contrast B&W".into(), settings: applied, preset: None }] },
    )
    .unwrap();
    s.execute("preset.apply", &json!({"id":"lc.warm-glow"})).unwrap();
    s.execute("preset.amount", &json!({"amount":50})).unwrap();
    let current = s.develop_of(id).unwrap();
    assert!(s.execute("preset.remove", &json!({})).is_err());
    assert_eq!(s.develop_of(id).unwrap(), current);
    s.execute("history.clear", &json!({})).unwrap();
    assert!(s.execute("preset.remove", &json!({})).is_err());
    s.execute("photo.virtualCopy", &json!({})).unwrap();
}

#[test]
fn reopening_either_v3_schema_preserves_preset_removal_and_album_order() {
    use lightcraft_engine::{
        catalog::{MemStore, journal::SNAPSHOT},
        library::LibraryStores,
    };
    // Both forks used v3; exercise each serialized schema through the library/command boundary.
    for schema in ["fork", "upstream"] {
        let mut s = Session::with_demo();
        let id = s.active().unwrap();
        s.execute("preset.apply", &json!({"id":"lc.bw-high-contrast"})).unwrap();
        s.execute("develop.set", &json!({"control":"effects.clarity","value":23})).unwrap();
        let a = s.execute("album.create", &json!({"name":"A"})).unwrap()["id"].as_u64().unwrap();
        let b = s.execute("album.create", &json!({"name":"B"})).unwrap()["id"].as_u64().unwrap();
        s.execute("album.reorder", &json!({"id":b,"before":a})).unwrap();
        let mut catalog: serde_json::Value = serde_json::from_str(&s.catalog.to_snapshot()).unwrap();
        fn remove_key(v: &mut serde_json::Value, key: &str) {
            match v {
                serde_json::Value::Object(o) => {
                    o.remove(key);
                    for child in o.values_mut() {
                        remove_key(child, key);
                    }
                }
                serde_json::Value::Array(a) => {
                    for child in a {
                        remove_key(child, key);
                    }
                }
                _ => {}
            }
        }
        remove_key(&mut catalog, if schema == "fork" { "order" } else { "preset" });
        let store = MemStore::new();
        store.set(SNAPSHOT, serde_json::to_vec(&json!({"format":"lightcraft-catalog","version":3,"seq":0,"catalog":catalog})).unwrap());
        let mut reopened = Session::new();
        let report = reopened
            .open_library_in(
                LibraryStores { dir: "v3-migration-test".into(), catalog: Box::new(store.clone()), files: Box::new(MemStore::new()), on_disk: false },
                false,
            )
            .unwrap();
        assert_eq!(report.upgraded_from, Some(3), "{schema} schema must be upgraded before edits");
        let snap: serde_json::Value = serde_json::from_slice(&store.get(SNAPSHOT).unwrap()).unwrap();
        assert_eq!(snap["version"], 4, "older divergent builds must refuse the merged catalog");
        reopened.execute("library.select", &json!({"ids":[id.0]})).unwrap();
        if schema == "fork" {
            reopened.execute("preset.remove", &json!({})).unwrap();
            let result = reopened.develop_of(id).unwrap();
            assert_eq!(result.light.contrast, 0.0);
            assert_eq!(result.effects.clarity, 23.0);
        } else {
            let albums = reopened.catalog.album_children(None);
            assert!(albums.iter().position(|x| x.id.0 == b).unwrap() < albums.iter().position(|x| x.id.0 == a).unwrap());
        }
        reopened.close_library().unwrap();
    }
}

#[test]
fn a_preset_applied_at_import_keeps_its_amount_and_removal_provenance() {
    let dir = std::env::temp_dir().join(format!("lc-merge-import-preset-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scene.png");
    let pixels = vec![[80, 120, 160, 255]; 16 * 12];
    let image = lightcraft_raster::Rgba8 { width: 16, height: 12, data: pixels };
    let bytes = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&image), &Default::default()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let mut s = Session::new().with_fs();
    s.execute("library.import", &json!({"paths":[path],"preset":"lc.bw-high-contrast"})).unwrap();
    let id = s.catalog.photos().next().unwrap().id;
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    assert_eq!(s.execute("preset.status", &json!({})).unwrap()["amount"], 100.0);
    s.execute("develop.set", &json!({"control":"effects.clarity","value":23})).unwrap();
    s.execute("preset.amount", &json!({"amount":50})).unwrap();
    s.execute("history.clear", &json!({})).unwrap();
    s.execute("preset.remove", &json!({})).unwrap();
    let d = s.develop_of(id).unwrap();
    assert_eq!(d.treatment, lightcraft_develop::Treatment::Color);
    assert_eq!(d.light.contrast, 0.0);
    assert_eq!(d.effects.clarity, 23.0);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    std::fs::remove_dir_all(dir).unwrap();
}
