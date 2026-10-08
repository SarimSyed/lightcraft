//! Real CLI exports and control-channel interactions for noise reduction.
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_lightcraft-cli");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lightcraft-nr-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_png(path: &Path) -> lightcraft_raster::Rgba8 {
    lightcraft_codecs::decode(&std::fs::read(path).unwrap(), Default::default()).unwrap().to_srgb8()
}

#[test]
fn lossless_exports_apply_noise_contrast_and_preserve_the_input() {
    let dir = scratch("exports");
    let input = dir.join("noise.png");
    let pixels = lightcraft_raster::Rgba8::from_fn(128, 128, |x, y| {
        let value = (120.0 + (x as f32 * std::f32::consts::TAU / 16.0).sin() * 24.0) as u8;
        let value = value.saturating_add(((x * 7919 + y * 104729) % 3) as u8);
        [value, value, value, 255]
    });
    let bytes = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&pixels), &Default::default()).unwrap();
    std::fs::write(&input, &bytes).unwrap();
    let mut exports = Vec::new();
    for (name, contrast) in [("smooth", "detail.nrContrast=0"), ("contrast", "detail.nrContrast=100")] {
        let output = dir.join(format!("{name}.png"));
        let result = Command::new(BIN)
            .env("LIGHTCRAFT_GPU", "0")
            .args(["render", input.to_str().unwrap(), "-o", output.to_str().unwrap(), "--set", "detail.nrLuminance=80", "--set", contrast])
            .output()
            .unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
        let image = read_png(&output);
        assert_eq!((image.width, image.height), (128, 128));
        exports.push(image);
    }
    assert!(exports[0].data != exports[1].data, "Contrast must affect lossless exported pixels");
    assert!(std::fs::read(&input).unwrap() == bytes, "input remains intact");
    std::fs::remove_dir_all(dir).unwrap();
}

fn execute(command: &str, params: Value) -> Value {
    json!({"method": "engine.execute", "params": {"command": command, "params": params}})
}

fn control_session(dir: &Path, name: &str, requests: &[Value]) -> Vec<Value> {
    let script = dir.join(format!("{name}.jsonl"));
    let text = requests.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("\n");
    std::fs::write(&script, text).unwrap();
    let result = Command::new(BIN)
        .env("LIGHTCRAFT_GPU", "0")
        .env("LIGHTCRAFT_NO_PREFS", "1")
        .args([
            "snapshot",
            "--demo",
            "--script",
            script.to_str().unwrap(),
            "-o",
            dir.join(format!("{name}.png")).to_str().unwrap(),
            "--size",
            "1600x1200",
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let replies: Vec<Value> = String::from_utf8_lossy(&result.stdout).lines().map(|s| serde_json::from_str(s).unwrap()).collect();
    assert_eq!(replies.len(), requests.len());
    assert!(replies.iter().all(|r| r["ok"] == true), "{replies:?}");
    replies
}

fn setup() -> Vec<Value> {
    let mut requests = vec![
        execute("library.select", json!({"ids": [1], "active": 1})),
        json!({"method": "ui.set", "params": {"view": "detail", "right": "edit", "openSections": ["detail"]}}),
    ];
    for (id, value) in [
        ("detail.nrLuminance", 0),
        ("detail.nrColor", 0),
        ("detail.nrDetail", 73),
        ("detail.nrContrast", 42),
        ("detail.nrColorDetail", 77),
        ("detail.nrColorSmoothness", 65),
    ] {
        requests.push(execute("develop.set", json!({"control": id, "value": value})));
    }
    requests
}

#[test]
fn inactive_nr_controls_ignore_pointer_keyboard_and_label_reset() {
    let dir = scratch("inactive");
    let mut probe = setup();
    probe.push(json!({"method": "ui.widgets", "params": {"filter": "slider:detail.nr"}}));
    let replies = control_session(&dir, "probe", &probe);
    let widgets = replies.last().unwrap()["result"].as_array().unwrap();
    let mut requests = setup();
    requests.push(execute("develop.get", json!({})));
    let before = requests.len() - 1;
    let rect = widgets.iter().find(|w| w["id"] == "slider:detail.nrDetail").unwrap()["rect"].as_array().unwrap();
    requests.push(json!({"method": "ui.click", "params": {
        "x": rect[0].as_f64().unwrap() + rect[2].as_f64().unwrap() * 0.5,
        "y": rect[1].as_f64().unwrap() - 9.0, "count": 2
    }}));
    requests.push(execute("develop.get", json!({})));
    let label = requests.len() - 1;
    for id in ["detail.nrDetail", "detail.nrContrast", "detail.nrColorDetail", "detail.nrColorSmoothness"] {
        let widget = format!("slider:{id}");
        requests.push(json!({"method": "ui.dragWidget", "params": {"id": widget, "dx": 50, "steps": 5}}));
        requests.push(json!({"method": "ui.clickWidget", "params": {"id": widget, "count": 2}}));
        requests.push(json!({"method": "ui.hoverWidget", "params": {"id": widget}}));
        requests.push(json!({"method": "ui.key", "params": {"key": "ArrowUp"}}));
        // Label is immediately above its track in the incumbent panel. Use discovered widget
        // geometry rather than window coordinates, and dispatch clicks through the public channel.
        let rect = widgets.iter().find(|w| w["id"] == widget).unwrap()["rect"].as_array().unwrap();
        let x = rect[0].as_f64().unwrap() + rect[2].as_f64().unwrap() * 0.5;
        let y = rect[1].as_f64().unwrap() - 9.0;
        requests.push(json!({"method": "ui.click", "params": {"x": x, "y": y, "count": 2}}));
    }
    requests.push(execute("develop.get", json!({})));
    let replies = control_session(&dir, "disabled", &requests);
    assert_eq!(replies[label]["result"]["detail"]["nr_detail"].as_f64(), Some(73.0), "inactive label cannot reset the saved Detail value");
    assert_eq!(replies[before]["result"]["detail"], replies.last().unwrap()["result"]["detail"], "inactive controls must preserve values");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn enabled_noise_slider_drag_is_one_undo_step_and_label_reset_is_undoable() {
    let dir = scratch("enabled");
    let mut probe = setup();
    probe.push(execute("develop.set", json!({"control": "detail.nrLuminance", "value": 80})));
    probe.push(json!({"method": "ui.widgets", "params": {"filter": "slider:detail.nrContrast"}}));
    let replies = control_session(&dir, "probe", &probe);
    let rect = replies.last().unwrap()["result"][0]["rect"].as_array().unwrap();
    let x = rect[0].as_f64().unwrap() + rect[2].as_f64().unwrap() * 0.5;
    let y = rect[1].as_f64().unwrap() - 9.0;
    let mut requests = setup();
    requests.push(execute("develop.set", json!({"control": "detail.nrLuminance", "value": 80})));
    requests.push(json!({"method": "ui.dragWidget", "params": {"id": "slider:detail.nrContrast", "dx": 50, "steps": 8}}));
    requests.push(execute("develop.get", json!({})));
    let changed = requests.len() - 1;
    requests.push(execute("edit.undo", json!({})));
    requests.push(execute("develop.get", json!({})));
    let undone = requests.len() - 1;
    requests.push(json!({"method": "ui.click", "params": {"x": x, "y": y, "count": 2}}));
    requests.push(execute("develop.get", json!({})));
    let reset = requests.len() - 1;
    requests.push(execute("edit.undo", json!({})));
    requests.push(execute("develop.get", json!({})));
    let replies = control_session(&dir, "enabled", &requests);
    let contrast = |i: usize| replies[i]["result"]["detail"]["nr_contrast"].as_f64().unwrap();
    assert_ne!(contrast(changed), 42.0, "drag changes Contrast");
    assert_eq!(contrast(undone), 42.0, "one undo restores the entire drag");
    assert_eq!(contrast(reset), 0.0, "double-click on the label resets Contrast");
    assert_eq!(replies.last().unwrap()["result"]["detail"]["nr_contrast"].as_f64(), Some(42.0), "reset is undoable");
    std::fs::remove_dir_all(dir).unwrap();
}
