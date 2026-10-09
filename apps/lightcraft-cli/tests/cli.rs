//! End-to-end tests of the `lightcraft-cli` binary: `mcp` over real stdio pipes, `render`, `commands`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_lightcraft-cli");

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lightcraft-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn gradient_png(path: &std::path::Path) {
    let img = lightcraft_raster::Rgba8::from_fn(120, 80, |x, y| [(x * 2) as u8, (y * 3) as u8, 100, 255]);
    let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &Default::default()).unwrap();
    std::fs::write(path, png).unwrap();
}

fn solid_png(path: &std::path::Path, width: usize, height: usize) {
    let img = lightcraft_raster::Rgba8::from_fn(width, height, |_, _| [80, 120, 180, 255]);
    let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &Default::default()).unwrap();
    std::fs::write(path, png).unwrap();
}

#[test]
fn calibrate_reads_raf_inputs_and_reports_bad_files() {
    let input = tmp("invalid.RAF");
    std::fs::write(&input, b"FUJIFILMCCD-RAW").unwrap();
    let out = tmp("raf-profiles");
    let result = Command::new(BIN).args(["calibrate", "--out"]).arg(&out).arg(&input).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("1 of 1 raw files"), "RAF was silently ignored: {stderr}");
    assert!(stderr.contains("invalid.RAF") && stderr.contains("no profile written"), "{stderr}");
    assert!(!out.join("X-T4.json").exists(), "bad input must not produce a profile");
}

fn mean(path: &std::path::Path) -> f64 {
    let d = lightcraft_codecs::decode(&std::fs::read(path).unwrap(), Default::default()).unwrap();
    let img = d.to_srgb8();
    img.data.iter().map(|p| (p[0] as u32 + p[1] as u32 + p[2] as u32) as f64).sum::<f64>() / (3 * img.data.len()) as f64
}

#[test]
fn mcp_over_stdio_sets_exposure_and_renders() {
    let input = tmp("in.png");
    gradient_png(&input);
    let mut child =
        Command::new(BIN).args(["mcp", input.to_str().unwrap()]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut rpc = |id: u64, method: &str, params: Value| -> Value {
        writeln!(stdin, "{}", json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["id"], id);
        v
    };
    let init = rpc(1, "initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "cli-test", "version": "0"}}));
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    let tools = rpc(2, "tools/list", json!({}));
    assert!(tools["result"]["tools"].as_array().unwrap().len() > 50);
    let base = tmp("base.png");
    let bright = tmp("bright.png");
    let r = rpc(3, "tools/call", json!({"name": "render_photo", "arguments": {"path": base.to_str().unwrap()}}));
    assert_eq!(r["result"]["isError"], false, "{r}");
    let r = rpc(4, "tools/call", json!({"name": "set_develop", "arguments": {"values": {"light.exposure": 1.0}}}));
    assert_eq!(r["result"]["isError"], false, "{r}");
    let r = rpc(5, "tools/call", json!({"name": "render_photo", "arguments": {"path": bright.to_str().unwrap()}}));
    assert_eq!(r["result"]["isError"], false, "{r}");
    assert_eq!(r["result"]["content"][0]["type"], "image");
    drop(stdin);
    assert!(child.wait().unwrap().success());
    let (a, b) = (mean(&base), mean(&bright));
    println!("cli mcp: mean sRGB exposure 0 → {a:.1}, +1 → {b:.1}");
    assert!(b > a + 10.0, "{a} → {b}");
}

#[test]
fn render_subcommand() {
    let input = tmp("r-in.png");
    gradient_png(&input);
    let plain = tmp("r-plain.png");
    let out = tmp("r-out.jpg");
    let st = Command::new(BIN).args(["render", input.to_str().unwrap(), "-o", plain.to_str().unwrap()]).status().unwrap();
    assert!(st.success());
    let st = Command::new(BIN)
        .args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap(), "--set", "light.exposure=1", "--size", "60"])
        .status()
        .unwrap();
    assert!(st.success());
    let d = lightcraft_codecs::decode(&std::fs::read(&out).unwrap(), Default::default()).unwrap();
    assert_eq!((d.width, d.height), (60, 40));
    let d = lightcraft_codecs::decode(&std::fs::read(&plain).unwrap(), Default::default()).unwrap();
    assert_eq!((d.width, d.height), (120, 80));
    assert!(mean(&out) > mean(&plain) + 10.0);
    // Bad control id fails cleanly.
    let o = Command::new(BIN).args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap(), "--set", "nope=1"]).output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown control"));
}

/// Issue #93: `render IMG -o IMG` replaced its own input with the render.
#[test]
fn render_refuses_to_overwrite_its_input() {
    let input = tmp("self-in.png");
    gradient_png(&input);
    let before = std::fs::read(&input).unwrap();
    let dir = input.parent().unwrap();
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let other_spelling = dir.join("sub/../self-in.png");
    for out in [&input, &other_spelling] {
        let o =
            Command::new(BIN).args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap(), "--set", "light.exposure=1"]).output().unwrap();
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains("never writes over an original"), "{}", String::from_utf8_lossy(&o.stderr));
    }
    assert_eq!(std::fs::read(&input).unwrap(), before, "the input is untouched");
}

#[test]
fn render_export_options() {
    let input = tmp("o-in.png");
    gradient_png(&input);
    let out = tmp("o-out.tif");
    let st = Command::new(BIN)
        .args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap()])
        .args(["--opt", "shortEdge=40", "--opt", "colorSpace=displayP3", "--opt", "bitDepth=16"])
        .status()
        .unwrap();
    assert!(st.success());
    let d = lightcraft_codecs::decode(&std::fs::read(&out).unwrap(), Default::default()).unwrap();
    assert_eq!((d.width, d.height), (60, 40));
    assert_eq!(d.space.named, Some(lightcraft_codecs::NamedSpace::DisplayP3));
    // original + sidecar named after the output
    let out = tmp("o-copy.png");
    let st = Command::new(BIN)
        .args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap(), "--set", "light.exposure=1", "--opt", "format=original"])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&input).unwrap());
    let xmp = std::fs::read_to_string(out.with_extension("xmp")).unwrap();
    assert!(xmp.contains("exposure"), "{xmp}");
    // unknown extension fails with a hint
    let o = Command::new(BIN).args(["render", input.to_str().unwrap(), "-o", tmp("x.bmp").to_str().unwrap()]).output().unwrap();
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("unknown extension"));
}

#[test]
fn commands_subcommand_lists_registry() {
    let o = Command::new(BIN).args(["commands", "--json"]).output().unwrap();
    assert!(o.status.success());
    let v: Value = serde_json::from_slice(&o.stdout).unwrap();
    let ids: Vec<&str> = v.as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
    for id in ["develop.set", "photo.rate", "library.import", "edit.undo", "app.export"] {
        assert!(ids.contains(&id), "{id}");
    }
    let o = Command::new(BIN).arg("controls").output().unwrap();
    assert!(String::from_utf8_lossy(&o.stdout).contains("light.exposure"));
}

#[test]
fn snapshot_subcommand_renders_the_ui_headlessly() {
    let script = tmp("snap.jsonl");
    let a = tmp("snap-grid.png");
    let b = tmp("snap-export.png");
    std::fs::write(
        &script,
        format!(
            "# comment\n{}\n{}\n{}\n{}\n{}\n{}\n",
            json!({"method": "ui.set", "params": {"view": "photoGrid"}}),
            // (settled: under load a grid shot taken before its thumbnails is darker than the
            // dimmed export dialog over them, and the comparison below fails)
            json!({"method": "ui.settle", "params": {"timeoutMs": 20000}}),
            json!({"method": "ui.screenshot"}),
            json!({"method": "engine.execute", "params": {"command": "dialog.export"}}),
            json!({"method": "ui.settle", "params": {"timeoutMs": 20000}}),
            json!({"method": "ui.screenshot", "params": {"path": b.to_str().unwrap()}}),
        ),
    )
    .unwrap();
    let o = Command::new(BIN)
        .args(["snapshot", "--demo", "--script", script.to_str().unwrap(), "-o", a.to_str().unwrap(), "--size", "640x400"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let replies: Vec<Value> = String::from_utf8_lossy(&o.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(replies.len(), 6);
    assert!(replies.iter().all(|r| r["ok"] == true), "{replies:?}");
    for (p, dimmed) in [(&a, false), (&b, true)] {
        let d = lightcraft_codecs::decode(&std::fs::read(p).unwrap(), Default::default()).unwrap();
        assert_eq!((d.width, d.height), (640, 400));
        // the export dialog dims everything around it
        assert_eq!(mean(p) < mean(&a) - 2.0, dimmed, "{}", p.display());
    }
    // no script: one settled screenshot, at 2× scale
    let o = Command::new(BIN).args(["snapshot", "-o", a.to_str().unwrap(), "--size", "320x240", "--scale", "2"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let d = lightcraft_codecs::decode(&std::fs::read(&a).unwrap(), Default::default()).unwrap();
    assert_eq!((d.width, d.height), (640, 480));
}

/// Issue #166: a scripted request that fails makes `snapshot` exit non-zero. The failed reply is
/// still printed, the later lines still run, and the final `-o` screenshot is still written.
#[test]
fn snapshot_script_failure_exits_non_zero() {
    let script = tmp("snap-fail.jsonl");
    let out = tmp("snap-fail.png");
    std::fs::write(
        &script,
        format!(
            "{}\n{}\n",
            json!({"method": "ui.set", "params": {"view": "grid"}}), // not a view: `photoGrid`, `detail`, …
            json!({"method": "ui.set", "params": {"view": "photoGrid"}}),
        ),
    )
    .unwrap();
    let _ = std::fs::remove_file(&out);
    let o = Command::new(BIN)
        .args(["snapshot", "--demo", "--script", script.to_str().unwrap(), "-o", out.to_str().unwrap(), "--size", "320x200"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "a failed scripted request must fail the run\n{stderr}");
    assert!(stderr.contains("1 scripted request(s) failed"), "{stderr}");
    let replies: Vec<Value> = String::from_utf8_lossy(&o.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(replies.len(), 2, "{replies:?}");
    assert_eq!(replies[0]["ok"], false, "{replies:?}");
    assert!(replies[0]["error"].as_str().unwrap_or("").contains("grid"), "{replies:?}");
    assert_eq!(replies[1]["ok"], true, "{replies:?}");
    // continue-and-report: the final screenshot is still written
    let d = lightcraft_codecs::decode(&std::fs::read(&out).unwrap(), Default::default()).unwrap();
    assert_eq!((d.width, d.height), (320, 200));
}

#[test]
fn snapshot_ui_zoom_keeps_requested_pixel_dimensions() {
    let script = tmp("snap-ui-zoom.jsonl");
    std::fs::write(
        &script,
        format!("{}\n{}\n", json!({"method": "ui.key", "params": {"key": "Plus", "cmd": true}}), json!({"method": "ui.inspect"})),
    )
    .unwrap();
    for (size, scale, expected) in [("400x240", "1", (400, 240)), ("345x200", "0.9", (311, 180))] {
        let out = tmp(&format!("snap-ui-zoom-{scale}.png"));
        let o = Command::new(BIN)
            .args(["snapshot", "--script", script.to_str().unwrap(), "-o", out.to_str().unwrap(), "--size", size, "--scale", scale])
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let replies: Vec<Value> = String::from_utf8_lossy(&o.stdout).lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert!(replies.iter().all(|reply| reply["ok"] == true), "{replies:?}");
        assert!(replies[1]["result"]["pixelsPerPoint"].as_f64().unwrap() > scale.parse::<f64>().unwrap());
        let decoded = lightcraft_codecs::decode(&std::fs::read(out).unwrap(), Default::default()).unwrap();
        assert_eq!((decoded.width, decoded.height), expected);
    }
}

/// Headless snapshots default to CPU photo rendering as well as CPU UI rasterization.
#[test]
fn snapshot_defaults_to_cpu_without_gpu_environment_overrides() {
    let script = tmp("default-cpu.jsonl");
    std::fs::write(
        &script,
        format!("{}\n{}\n", json!({"method": "engine.execute", "params": {"command": "app.gpu"}}), json!({"method": "ui.screenshot"})),
    )
    .unwrap();
    let image = tmp("default-cpu.png");
    let output = Command::new(BIN)
        .env_remove("LIGHTCRAFT_GPU")
        .env_remove("LIGHTCRAFT_GPU_BACKEND")
        .env_remove("WGPU_BACKEND")
        .args(["snapshot", "--demo", "--script", script.to_str().unwrap(), "-o", image.to_str().unwrap(), "--size", "480x320"])
        .output()
        .unwrap();
    assert!(output.status.success(), "status {}: {}", output.status, String::from_utf8_lossy(&output.stderr));
    let replies: Vec<Value> = String::from_utf8_lossy(&output.stdout).lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert!(replies.iter().all(|reply| reply["ok"] == true), "{replies:?}");
    let gpu = &replies[0]["result"];
    assert_eq!(gpu["enabled"], false, "{gpu}");
    assert_eq!(gpu["available"], false, "{gpu}");
    assert_eq!(gpu["adapter"], Value::Null, "no device is created: {gpu}");
    assert!(gpu["reason"].as_str().is_some_and(|reason| reason.contains("preference")), "{gpu}");
    let decoded = lightcraft_codecs::decode(&std::fs::read(image).unwrap(), Default::default()).unwrap();
    assert_eq!((decoded.width, decoded.height), (480, 320));
}

/// Issue #136: with the GPU switched off from the environment the UI starts and renders on the CPU,
/// without creating a GPU device (no driver is loaded), and says why.
#[test]
fn snapshot_starts_without_a_gpu() {
    let script = tmp("nogpu.jsonl");
    std::fs::write(
        &script,
        format!(
            "{}\n{}\n{}\n",
            json!({"method": "ui.set", "params": {"view": "photoGrid"}}),
            json!({"method": "engine.execute", "params": {"command": "app.gpu"}}),
            json!({"method": "ui.screenshot"}),
        ),
    )
    .unwrap();
    for (var, value, reason) in [("LIGHTCRAFT_GPU", "0", "LIGHTCRAFT_GPU=0"), ("LIGHTCRAFT_GPU_BACKEND", "off", "LIGHTCRAFT_GPU_BACKEND=off")] {
        let out = tmp(&format!("nogpu-{var}.png"));
        let o = Command::new(BIN)
            .env_remove("LIGHTCRAFT_GPU")
            .env_remove("LIGHTCRAFT_GPU_BACKEND")
            .env(var, value)
            .args(["snapshot", "--demo", "--script", script.to_str().unwrap(), "-o", out.to_str().unwrap(), "--size", "480x320"])
            .output()
            .unwrap();
        assert!(o.status.success(), "{var}: {}", String::from_utf8_lossy(&o.stderr));
        let replies: Vec<Value> = String::from_utf8_lossy(&o.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert!(replies.iter().all(|r| r["ok"] == true), "{replies:?}");
        let gpu = &replies[1]["result"];
        assert_eq!(gpu["available"], false, "{var}: {gpu}");
        assert_eq!(gpu["adapter"], Value::Null, "{var}: no device was created: {gpu}");
        assert!(gpu["reason"].as_str().is_some_and(|r| r.contains(reason)), "{var}: {gpu}");
        let d = lightcraft_codecs::decode(&std::fs::read(&out).unwrap(), Default::default()).unwrap();
        assert_eq!((d.width, d.height), (480, 320));
    }
}

fn run_cli(args: &[&str], stdin: Option<&str>) -> (bool, Vec<Value>, String) {
    let mut child = Command::new(BIN).arg("run").args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    if let Some(text) = stdin {
        child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    }
    drop(child.stdin.take());
    let o = child.wait_with_output().unwrap();
    let lines = String::from_utf8_lossy(&o.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    (o.status.success(), lines, String::from_utf8_lossy(&o.stderr).to_string())
}

/// Crash/stall regressions run out of process. Files avoid pipe-buffer deadlocks while waiting.
fn run_cli_bounded(tag: &str, args: &[&str]) -> (bool, Vec<Value>, String) {
    let (ok, out, err) = run_process_bounded(tag, &[&["run"], args].concat());
    let lines = out.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    (ok, lines, err)
}

fn run_process_bounded(tag: &str, args: &[&str]) -> (bool, String, String) {
    let stdout = tmp(&format!("{tag}.stdout"));
    let stderr = tmp(&format!("{tag}.stderr"));
    let mut child = Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&stdout).unwrap())
        .stderr(std::fs::File::create(&stderr).unwrap())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() >= std::time::Duration::from_secs(10) {
            timed_out = true;
            let _ = child.kill();
            break child.wait().unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let out = std::fs::read_to_string(&stdout).unwrap();
    let err = std::fs::read_to_string(&stderr).unwrap();
    std::fs::remove_file(stdout).unwrap();
    std::fs::remove_file(stderr).unwrap();
    assert!(!timed_out, "{tag} exceeded 10 s and was killed/reaped; stdout: {out}; stderr: {err}");
    (status.success(), out, err)
}

/// CIE76, D65: an independent comparison of sRGB output against the camera JPEG, not a render snapshot.
fn srgb_lab(p: [u8; 4]) -> [f64; 3] {
    let [r, g, b] = [p[0], p[1], p[2]].map(|v| {
        let v = v as f64 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    });
    let [x, y, z] = [
        (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047,
        0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
        (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883,
    ]
    .map(|v| if v > (6.0f64 / 29.0).powi(3) { v.cbrt() } else { v / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0 });
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}

#[test]
fn corpus_d7100_cli_renders_plausible_colour_and_keeps_exposure_editable() {
    use lightcraft_raster::resample::{Filter, resize};
    let dir = std::env::var_os("LIGHTCRAFT_CORPUS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus"))
        .join("raw");
    for mode in ["lossless12", "lossless14", "lossy12", "lossy14"] {
        let path = dir.join(format!("nef-nikon-d7100-{mode}.nef"));
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skip: {} absent", path.display());
            continue;
        };
        let out = tmp(&format!("d7100-{mode}.png"));
        let (ok, stdout, stderr) = run_process_bounded(mode, &["render", path.to_str().unwrap(), "-o", out.to_str().unwrap(), "--size", "1200"]);
        assert!(ok, "D7100 {mode}: {stdout}; {stderr}");
        let decoded = lightcraft_codecs::decode(&std::fs::read(&out).unwrap(), Default::default()).unwrap();
        assert_eq!((decoded.width, decoded.height), (1200, 799));
        let render = decoded.to_srgb8();
        let reference = lightcraft_engine::files::embedded_preview_srgb(&bytes, 384).unwrap();
        let small = |img: &lightcraft_raster::Rgba8| resize(&img.to_linear(), 200, 133, Filter::Box).to_srgb8();
        let (a, b) = (small(&render), small(&reference));
        let (mut de, mut dl) = (0.0, 0.0);
        for (x, y) in a.data.iter().zip(&b.data) {
            let (x, y) = (srgb_lab(*x), srgb_lab(*y));
            de += x.iter().zip(y).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
            dl += x[0] - y[0];
        }
        let (de, dl) = (de / a.data.len() as f64, dl / a.data.len() as f64);
        eprintln!("D7100 {mode}: mean CIE76 {de:.2}, signed lightness {dl:.2}");
        // A plausibility/regression ceiling, not a claim of calibrated colour or Lightroom parity.
        assert!(de < 15.0 && dl.abs() < 4.0, "D7100 {mode}: colour/lightness regression ({de}, {dl})");
        if mode == "lossless14" {
            let edited = tmp("d7100-edited.png");
            let (ok, stdout, stderr) = run_process_bounded(
                "d7100-exposure",
                &["render", path.to_str().unwrap(), "-o", edited.to_str().unwrap(), "--size", "1200", "--set", "light.exposure=1"],
            );
            assert!(ok, "{stdout}; {stderr}");
            assert!(mean(&edited) > mean(&out) + 8.0, "exposure must change exported pixels");
            std::fs::remove_file(edited).unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "D7100 {mode}: original changed");
        std::fs::remove_file(out).unwrap();
    }
}

#[test]
fn preset_import_rejects_deep_nesting_without_aborting() {
    let path = tmp("deep.lrtemplate");
    std::fs::write(&path, format!("return {}{}", "{".repeat(30_000), "}".repeat(30_000))).unwrap();
    let paths = format!("paths={}", json!([path.to_string_lossy()]));
    let (ok, lines, err) = run_cli_bounded("deep-preset", &["preset.import", &paths, "dryRun=true"]);
    assert!(ok, "the process must survive: {err}");
    assert_eq!(lines[0]["result"]["imported"], json!([]));
    let error = lines[0]["result"]["failed"][0][1].as_str().unwrap();
    assert!(error.contains("64") && error.contains("at byte"), "{error}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn preset_import_bounds_recursive_wrappers_and_continues_the_batch() {
    let dir = tmp("preset-nesting");
    std::fs::create_dir_all(&dir).unwrap();
    let valid = dir.join("valid.lrtemplate");
    std::fs::write(&valid, r#"return { title = "Safe", value = { settings = { Exposure2012 = 0.5 } } }"#).unwrap();
    let bad = dir.join("bad.lrtemplate");
    let paths = format!("paths={}", json!([bad.to_string_lossy(), valid.to_string_lossy()]));
    for (tag, text) in [
        ("negation", format!("return {}1", "- ".repeat(30_000))),
        ("calls", format!("return {}1{}", "ZSTR(".repeat(30_000), ")".repeat(30_000))),
        ("keys", format!("return {}1{}", "{[".repeat(30_000), "]=1}".repeat(30_000))),
    ] {
        std::fs::write(&bad, text).unwrap();
        let (ok, lines, err) = run_cli_bounded(tag, &["presets.list", "preset.import", &paths, "dryRun=true", "presets.list"]);
        assert!(ok, "{tag}: {err}");
        assert_eq!(lines[1]["result"]["imported"].as_array().unwrap().len(), 1);
        let error = lines[1]["result"]["failed"][0][1].as_str().unwrap();
        assert!(error.contains("64") && error.contains("at byte"), "{tag}: {error}");
        assert_eq!(lines[0]["result"], lines[2]["result"], "dry run leaves presets unchanged");
    }
    for depth in [63, 64] {
        std::fs::write(&bad, format!("return {}1", "- ".repeat(depth))).unwrap();
        let (ok, lines, err) = run_cli_bounded("depth-boundary", &["preset.import", &paths, "dryRun=true"]);
        assert!(ok, "{err}");
        let error = lines[0]["result"]["failed"][0][1].as_str().unwrap();
        assert_eq!(error.contains("nesting exceeds 64"), depth == 64, "{depth}: {error}");
    }
    let (ok, lines, err) = run_cli_bounded("preset-batch", &["preset.import", &paths, "presets.list"]);
    assert!(ok, "{err}");
    assert_eq!(lines[0]["result"]["imported"][0]["name"], "Safe");
    assert_eq!(lines[0]["result"]["failed"].as_array().unwrap().len(), 1);
    assert!(lines[1]["result"].as_array().unwrap().iter().any(|p| p["name"] == "Safe"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn folder_imports_terminate_on_directory_symlink_cycles() {
    let dir = tmp("cycle-imports");
    std::fs::create_dir_all(&dir).unwrap();
    std::os::unix::fs::symlink(".", dir.join("a")).unwrap();
    std::os::unix::fs::symlink(".", dir.join("b")).unwrap();
    gradient_png(&dir.join("photo.png"));
    std::fs::write(dir.join("look.lrtemplate"), r#"return { title = "Cycle Look", value = { settings = { Exposure2012 = 0.5 } } }"#).unwrap();
    let path = dir.to_str().unwrap();
    let paths = format!("paths={}", json!([path]));
    for (tag, args) in [
        ("cycle-cli", vec!["--import", path, "library.info"]),
        ("cycle-engine", vec!["library.import", &paths, "library.info"]),
        ("cycle-presets", vec!["preset.import", &paths, "dryRun=true"]),
    ] {
        let (ok, lines, err) = run_cli_bounded(tag, &args);
        assert!(ok, "{tag}: {err}");
        let result = &lines.last().unwrap()["result"];
        if tag == "cycle-presets" {
            assert_eq!(result["imported"].as_array().unwrap().len(), 1, "{result}");
            assert_eq!(result["failed"], json!([]));
        } else {
            assert_eq!(result["photos"], 1, "{result}");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn directory_symlinks_remain_importable_and_library_aliases_are_skipped() {
    let dir = tmp("linked-imports");
    let real = dir.join("real");
    let alias = dir.join("alias");
    std::fs::create_dir_all(&real).unwrap();
    std::os::unix::fs::symlink("real", &alias).unwrap();
    gradient_png(&real.join("photo.png"));
    std::fs::write(real.join("look.lrtemplate"), r#"return { title = "Linked Look", value = { settings = { Exposure2012 = 0.5 } } }"#).unwrap();
    let paths = format!("paths={}", json!([alias.to_string_lossy(), real.to_string_lossy()]));
    let (ok, lines, err) = run_cli_bounded("linked-cli", &["--import", alias.to_str().unwrap(), "library.info"]);
    assert!(ok, "{err}");
    assert_eq!(lines[0]["result"]["photos"], 1);
    let (ok, lines, err) = run_cli_bounded("linked-engine", &["library.import", &paths, "library.info"]);
    assert!(ok, "{err}");
    assert_eq!(lines[0]["result"]["imported"].as_array().unwrap().len(), 1);
    assert_eq!(lines[0]["result"]["duplicates"], json!([]));
    assert_eq!(lines[1]["result"]["photos"], 1);
    let (ok, lines, err) = run_cli_bounded("linked-presets", &["preset.import", &paths, "dryRun=true"]);
    assert!(ok, "{err}");
    let imported = lines[0]["result"]["imported"].as_array().unwrap();
    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0]["file"], alias.join("look.lrtemplate").to_string_lossy().as_ref(), "keep the caller's path spelling");

    let library = dir.join("library");
    std::fs::create_dir_all(&library).unwrap();
    solid_png(&library.join("must-not-import.png"), 2, 2);
    std::os::unix::fs::symlink("library", dir.join("a-library-alias")).unwrap();
    let paths = format!("paths={}", json!([dir.to_string_lossy()]));
    let (ok, lines, err) = run_cli_bounded("skip-library", &["--library", library.to_str().unwrap(), "library.import", &paths, "library.info"]);
    assert!(ok, "{err}");
    assert_eq!(lines[1]["result"]["photos"], 1, "neither the library nor its alias is imported");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn extreme_aspect_ratios_export_at_the_requested_size() {
    for (tag, width, height, expected) in [("wide", 70_000, 1, (3000, 1)), ("tall", 1, 70_000, (1, 3000))] {
        let input = tmp(&format!("{tag}.png"));
        let output = tmp(&format!("{tag}-resized.jpg"));
        solid_png(&input, width, height);
        let path = format!("path={}", output.display());
        let (ok, lines, err) = run_cli_bounded(tag, &["--import", input.to_str().unwrap(), "app.export", &path, "longEdge=3000"]);
        assert!(ok, "{tag} export must survive: {err}");
        assert_eq!((lines[0]["result"]["width"].as_u64().unwrap(), lines[0]["result"]["height"].as_u64().unwrap()), expected);
        let decoded = lightcraft_codecs::decode(&std::fs::read(&output).unwrap(), Default::default()).unwrap();
        assert_eq!((decoded.width as u64, decoded.height as u64), expected);
        let (ok, lines, err) =
            run_cli_bounded(tag, &["--import", input.to_str().unwrap(), "app.export", &path, "longEdge=100000", "dontEnlarge=false"]);
        assert!(ok, "{tag}: {err}");
        let capped = if width > height { (65_535, 1) } else { (1, 65_535) };
        assert_eq!((lines[0]["result"]["width"].as_u64().unwrap(), lines[0]["result"]["height"].as_u64().unwrap()), capped);
        let decoded = lightcraft_codecs::decode(&std::fs::read(&output).unwrap(), Default::default()).unwrap();
        assert_eq!((decoded.width as u64, decoded.height as u64), capped);
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn run_subcommand_chains_commands_and_persists_a_library() {
    let input = tmp("run-in.png");
    gradient_png(&input);
    let lib = tmp("run-lib");
    let _ = std::fs::remove_dir_all(&lib);
    let out = tmp("run-out.jpg");
    let (lib_s, in_s, out_s) = (lib.to_str().unwrap(), input.to_str().unwrap(), out.to_str().unwrap());
    // import into a library, edit, export, in one invocation
    let (ok, lines, err) = run_cli(
        &[
            "--library",
            lib_s,
            "--import",
            in_s,
            "develop.set",
            "control=light.exposure",
            "value=0.75",
            "app.export",
            &format!("path={out_s}"),
            "width=60",
        ],
        None,
    );
    assert!(ok, "{err}");
    assert_eq!(lines.len(), 2);
    assert!(lines.iter().all(|l| l["ok"] == true), "{lines:?}");
    assert_eq!(lines[1]["result"]["width"], 60);
    assert!(out.exists());
    // a second invocation sees the saved edit (same file: selected again, not re-imported)
    let (ok, lines, err) = run_cli(&["--library", lib_s, "--import", in_s, "develop.get"], None);
    assert!(ok, "{err}");
    assert_eq!(lines[0]["result"]["light"]["exposure"], 0.75);
    // JSON-lines script from stdin, a failing command stops the run with a non-zero status
    let script = "{\"command\": \"library.info\"}\n# comment\n{\"command\": \"no.such\"}\n{\"command\": \"library.info\"}\n";
    let (ok, lines, _) = run_cli(&["--demo", "--script", "-"], Some(script));
    assert!(!ok);
    assert_eq!(lines.len(), 2, "stopped after the failure: {lines:?}");
    assert_eq!(lines[1]["ok"], false);
    let (ok, lines, _) = run_cli(&["--demo", "--keep-going", "--script", "-"], Some(script));
    assert!(!ok);
    assert_eq!(lines.len(), 3);
    // control-protocol methods work headlessly too; bad usage is reported
    let (ok, lines, _) = run_cli(&["--demo", "engine.commands"], None);
    assert!(ok && lines[0]["result"].as_array().is_some_and(|a| a.len() > 50));
    let (ok, _, err) = run_cli(&["--demo", "x=1"], None);
    assert!(!ok && err.contains("before any command"), "{err}");
}

#[test]
fn devices_are_listed_and_imported_from() {
    let base = tmp("devices");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("CARD/DCIM/100TEST")).unwrap();
    gradient_png(&base.join("CARD/DCIM/100TEST/IMG_0001.png"));
    let o = Command::new(BIN).env("LIGHTCRAFT_DEVICE_ROOTS", &base).args(["run", "--demo", "library.devices"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let line: Value = serde_json::from_slice(o.stdout.split(|b| *b == b'\n').next().unwrap()).unwrap();
    let dev = &line["result"][0];
    assert_eq!(dev["name"], "CARD", "{line}");
    // the import review of its DCIM folder finds the photo
    let dcim = dev["path"].as_str().unwrap().to_string();
    // JSON-escaped: a Windows path has backslashes
    let paths = serde_json::to_string(&[&dcim]).unwrap();
    let o = Command::new(BIN).args(["run", "library.importPreview", &format!("paths={paths}")]).output().unwrap();
    let line: Value = serde_json::from_slice(o.stdout.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert_eq!(line["result"]["candidates"].as_array().map(Vec::len), Some(1), "{line}");
}

/// Issue #99: a library open in one process (here `mcp --library`) is refused by a second one,
/// with who has it and how to drive the running app instead; free again once the first exits.
#[test]
fn a_library_open_in_another_process_is_refused() {
    let lib = tmp("locked-lib");
    let _ = std::fs::remove_dir_all(&lib);
    let lib_s = lib.to_str().unwrap();
    let mut holder =
        Command::new(BIN).args(["mcp", "--library", lib_s]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let mut err = BufReader::new(holder.stderr.take().unwrap());
    let mut line = String::new();
    while !line.contains("opened library") {
        line.clear();
        assert!(err.read_line(&mut line).unwrap() > 0, "mcp exited before opening the library");
    }
    let (ok, _, stderr) = run_cli(&["--library", lib_s, "library.info"], None);
    assert!(!ok);
    assert!(stderr.contains("already open in lightcraft-cli") && stderr.contains(&format!("process {}", holder.id())), "{stderr}");
    assert!(stderr.contains("mcp --connect"), "{stderr}");

    drop(holder.stdin.take()); // EOF: the server exits and lets go of the library
    assert!(holder.wait().unwrap().success());
    let (ok, lines, stderr) = run_cli(&["--library", lib_s, "library.info"], None);
    assert!(ok, "{stderr}");
    assert_eq!(lines[0]["ok"], true);
    let _ = std::fs::remove_dir_all(&lib);
}

/// Issue #167: when `--connect ADDR` cannot reach the app, the recovery hint must name the port
/// ADDR actually names — following the old fixed "7980" left the retry connecting elsewhere.
#[test]
fn connect_failure_hint_names_the_attempted_port() {
    let (ok, _lines, err) = run_cli(&["--connect", "127.0.0.1:18437", "ui.inspect"], None);
    assert!(!ok);
    assert!(err.contains("--control 18437"), "the hint must name the port it tried: {err}");
    assert!(!err.contains("--control 7980"), "{err}");
}

/// The CLI logs warnings on stderr (#168): with `RUST_LOG=warn` an unknown GPU backend name is
/// reported, and `RUST_LOG=off` silences it. Before, no logger was installed and nothing appeared.
#[test]
fn warnings_are_logged_on_stderr() {
    let input = tmp("log-in.png");
    gradient_png(&input);
    let out = tmp("log-out.jpg");
    let run = |level: &str| {
        let o = Command::new(BIN)
            .args(["render", input.to_str().unwrap(), "-o", out.to_str().unwrap()])
            .env("LIGHTCRAFT_GPU_BACKEND", "bogus")
            .env("RUST_LOG", level)
            .env_remove("LIGHTCRAFT_LOG")
            .env_remove("LIGHTCRAFT_GPU")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(o.status.success(), "{stderr}");
        stderr
    };
    let warn = run("warn");
    assert!(warn.contains("LIGHTCRAFT_GPU_BACKEND=bogus names no known backend"), "{warn}");
    let off = run("off");
    assert!(!off.contains("names no known backend"), "{off}");
}

#[test]
fn snapshot_rejects_invalid_dimensions_before_creating_an_image() {
    for (index, (size, scale)) in
        [("NaNx100", "1"), ("100x100", "inf"), ("9000x9000", "1"), ("9000x100", "2"), ("100x100", "0.001")].into_iter().enumerate()
    {
        let image = tmp(&format!("invalid-viewport-{index}.png"));
        let output = Command::new(BIN).args(["snapshot", "--size", size, "--scale", scale, "-o"]).arg(&image).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("snapshot: bad --size/--scale"), "{}", String::from_utf8_lossy(&output.stderr));
        assert!(!image.exists());
    }
}

#[test]
fn snapshot_preserves_fractional_scale_rounding() {
    let image = tmp("fractional-scale-rounding.png");
    let output =
        Command::new(BIN).env("LIGHTCRAFT_GPU", "0").args(["snapshot", "--size", "345x200", "--scale", "0.9", "-o"]).arg(&image).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let decoded = lightcraft_codecs::decode(&std::fs::read(image).unwrap(), Default::default()).unwrap();
    assert_eq!((decoded.width, decoded.height), (311, 180));
}

/// Snapshot's CPU setting must never replace the desktop preference, including with --library.
#[cfg(target_os = "linux")]
#[test]
fn snapshot_leaves_desktop_gpu_preferences_untouched() {
    let dir = tmp("snapshot-desktop-preferences");
    let library = dir.join("library");
    let mut session = lightcraft_engine::Session::new().with_fs();
    session.open_library(&library, true).unwrap();
    drop(session);
    let config = dir.join("config");
    let app_config = config.join("lightcraft");
    std::fs::create_dir_all(&app_config).unwrap();
    let ui_path = app_config.join("ui.json");
    let mut ui = lightcraft_ui_egui::UiState::default();
    ui.settings.gpu = true;
    let saved = serde_json::to_vec_pretty(&ui).unwrap();
    std::fs::write(&ui_path, &saved).unwrap();
    let script = dir.join("preferences.jsonl");
    std::fs::write(
        &script,
        format!("{}\n{}\n", json!({"method": "engine.execute", "params": {"command": "app.gpu"}}), json!({"method": "ui.inspect"})),
    )
    .unwrap();
    for gpu_environment in [None, Some("1")] {
        let image = dir.join(format!("{}.png", gpu_environment.unwrap_or("default")));
        let mut cmd = Command::new(BIN);
        cmd.env("XDG_CONFIG_HOME", &config).env_remove("LIGHTCRAFT_GPU").env_remove("LIGHTCRAFT_GPU_BACKEND").env_remove("WGPU_BACKEND");
        if let Some(value) = gpu_environment {
            cmd.env("LIGHTCRAFT_GPU", value);
        }
        let output = cmd
            .args(["snapshot", "--library"])
            .arg(&library)
            .args(["--script"])
            .arg(&script)
            .args(["-o"])
            .arg(&image)
            .args(["--size", "480x320"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let replies: Vec<Value> = String::from_utf8_lossy(&output.stdout).lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert!(replies.iter().all(|reply| reply["ok"] == true), "{replies:?}");
        assert_eq!(replies[0]["result"]["enabled"], false);
        assert_eq!(replies[0]["result"]["available"], false);
        assert_eq!(replies[0]["result"]["adapter"], Value::Null);
        assert_eq!(replies[1]["result"]["ui"]["settings"]["gpu"], false);
        assert_eq!(std::fs::read(&ui_path).unwrap(), saved, "snapshot changed the desktop preferences");
        assert!(!app_config.join("ui.json.tmp").exists());
        assert!(image.exists());
    }
}
