use crate::{LightcraftApp, Services, headless::Headless};
use serde_json::json;
use std::time::Duration;

#[test]
fn first_use_denoise_dialog_shows_installation_and_escape_preserves_the_edit() {
    let mut session = lightcraft_engine::Session::with_demo();
    session.enhancer.model_dir = std::env::temp_dir().join(format!("lc-denoise-ui-absent-{}", std::process::id()));
    let app = LightcraftApp::new(session, Services::default());
    let mut h = Headless::new(app, [1200.0, 800.0], 1.0);
    let id = h.app.session.active().unwrap();
    let before = h.app.session.develop_of(id).unwrap();
    let reply = h.request("engine.execute", json!({"command":"dialog.denoise"}), Duration::from_secs(20));
    assert_eq!(reply["ok"], true, "{reply}");
    let inspect = h.request("ui.widgets", json!({}), Duration::from_secs(20));
    let text = inspect.to_string();
    assert!(text.contains("denoise:download"), "{text}");
    h.request("ui.key", json!({"key":"Escape"}), Duration::from_secs(20));
    assert!(h.app.ui.dialog.is_none());
    assert_eq!(h.app.session.develop_of(id).unwrap(), before);
}

#[cfg(feature = "reference-validation")]
#[test]
fn denoise_preview_can_navigate_the_source_without_committing() {
    let dir = std::env::temp_dir().join(format!("lc-denoise-navigation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.tif");
    let pixels: Vec<f32> = (0..640 * 480).flat_map(|i| [0.2 + (i % 29) as f32 * 0.002, 0.3, 0.4]).collect();
    std::fs::write(
        &source,
        lightcraft_codecs::encode_tiff(
            &lightcraft_codecs::EncodeImage::new(640, 480, 3, lightcraft_codecs::Samples::F32(&pixels)),
            lightcraft_codecs::TiffCompression::Deflate,
            &Default::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut s = lightcraft_engine::Session::new().with_fs();
    s.enhancer.model_dir = std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent").into();
    s.execute("library.import", &json!({"paths":[source]})).unwrap();
    let id = s.catalog.photos().next().unwrap().id;
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    let original = s.develop_of(id).unwrap();
    let mut h = Headless::new(LightcraftApp::new(s, Services::default()), [1200.0, 900.0], 1.0);
    let timeout = Duration::from_secs(60);
    assert_eq!(h.request("engine.execute", json!({"command":"dialog.denoise"}), timeout)["ok"], true);
    assert!(h.step_until(timeout, |h| h.app.session.enhancer.preview.is_some()));
    h.step();
    let region = h.app.session.enhancer.preview.as_ref().unwrap().region;
    let reply = h.request("ui.clickWidget", json!({"id":"denoise:navigator","fx":0.95,"fy":0.9}), timeout);
    assert_eq!(reply["ok"], true, "{reply}");
    assert!(
        h.step_until(timeout, |h| h.app.session.enhancer.preview.as_ref().is_some_and(|p| p.region != region)),
        "{}",
        json!(h.app.session.enhancer.status())
    );
    let moved = h.app.session.enhancer.preview.as_ref().unwrap().region;
    assert!(moved.x > region.x && moved.y > region.y);
    h.settle(timeout);
    let image_width = h.request("ui.widgets", json!({"filter":"denoise:preview"}), timeout)["result"][0]["rect"][2].as_f64().unwrap();
    let hovered = h.request("ui.hoverWidget", json!({"id":"denoise:preview"}), timeout);
    assert_eq!(hovered["ok"], true, "{hovered}");
    h.request("ui.scroll", json!({"dy":120.0,"ctrl":true}), timeout);
    h.step();
    let preparing_width = h.request("ui.widgets", json!({"filter":"denoise:preview"}), timeout)["result"][0]["rect"][2].as_f64().unwrap();
    assert!(
        (preparing_width - image_width).abs() < 2.0,
        "the old preview must not stretch the dialog while zooming: {image_width} → {preparing_width}"
    );
    assert!(
        matches!(h.app.ui.dialog,Some(crate::state::Dialog::Denoise { view,.. }) if view.zoom>100),
        "wheel {:?} {:?}",
        h.app.ui.dialog,
        h.view.ctx.input(|i| i.pointer.hover_pos())
    );
    assert!(
        h.step_until(timeout, |h| h.app.session.enhancer.preview.as_ref().is_some_and(|p| p.region.width < moved.width)),
        "{:?} {}",
        h.app.ui.dialog,
        json!(h.app.session.enhancer.status())
    );
    assert_eq!(h.app.session.develop_of(id).unwrap(), original);
    h.request("ui.dialog.cancel", json!({}), timeout);
    assert_eq!(h.app.session.develop_of(id).unwrap(), original);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "reference-validation")]
#[test]
fn denoise_amount_preview_and_before_leave_history_untouched_until_apply() {
    use lightcraft_engine::catalog::{Op, Photo, Source};
    let dir = std::env::temp_dir().join(format!("lc-denoise-ui-preview-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("source.tif");
    let pixels: Vec<f32> = (0..32 * 24)
        .flat_map(|i| {
            let n = ((i * 17) % 23) as f32 * 0.002;
            [0.2 + n, 0.3 + n, 0.4 + n]
        })
        .collect();
    let bytes = lightcraft_codecs::encode_tiff(
        &lightcraft_codecs::EncodeImage::new(32, 24, 3, lightcraft_codecs::Samples::F32(&pixels)),
        lightcraft_codecs::TiffCompression::Deflate,
        &Default::default(),
    )
    .unwrap();
    std::fs::write(&path, bytes).unwrap();
    let mut s = lightcraft_engine::Session::new().with_fs();
    s.enhancer.model_dir = std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent").into();
    let id = s.catalog.alloc_photo_id();
    s.commit(
        "Import",
        Op::AddPhoto {
            photo: Box::new(Photo::new(id, Source::File { path: path.to_string_lossy().into() }, "source.tif", "TIFF", 32, 24, "2026-10-08")),
        },
    )
    .unwrap();
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    let before = s.develop_of(id).unwrap();
    let mut h = Headless::new(LightcraftApp::new(s, Services::default()), [1200.0, 800.0], 1.0);
    let timeout = Duration::from_secs(30);
    assert_eq!(h.request("engine.execute", json!({"command":"dialog.denoise"}), timeout)["ok"], true);
    h.step_until(timeout, |h| h.app.session.enhancer.preview.is_some());
    let status = h.request("engine.execute", json!({"command":"enhance.denoise.status"}), timeout);
    assert!(status["result"]["backend"].as_str().is_some_and(|name| !name.is_empty()));
    if std::env::var_os("LIGHTCRAFT_REQUIRE_DENOISE_GPU").is_some() {
        assert!(status["result"]["backend"].as_str().is_some_and(|name| name.contains("Vulkan")), "{status}");
        assert_eq!(status["result"]["cpu_fallback"], false);
    }
    let capture = h.app.headless_screenshot(&h.view.ctx, true).unwrap();
    let preview_widget = h.request("ui.widgets", json!({"filter":"denoise:preview"}), timeout);
    let rect = preview_widget["result"][0]["rect"].as_array().unwrap();
    let (x, y) = (rect[0].as_f64().unwrap() as usize + 5, rect[1].as_f64().unwrap() as usize + 5);
    assert!(capture.pixels[y * capture.size[0] + x].r() < 250, "native headless capture must contain preview pixels");
    let widgets = h.request("ui.widgets", json!({"filter":"denoise"}), timeout).to_string();
    assert!(widgets.contains("denoise:before"));
    assert!(widgets.contains("denoise:preview"));
    assert_eq!(h.request("ui.clickWidget", json!({"id":"slider:enhance.denoise","fx":0.8}), timeout)["ok"], true);
    assert_eq!(h.app.session.develop_of(id).unwrap(), before);
    h.request("ui.key", json!({"key":"space","pressed":true}), timeout);
    assert!(h.view.ctx.input(|i| i.key_down(egui::Key::Space)));
    h.request("ui.key", json!({"key":"space","pressed":false}), timeout);
    assert_eq!(h.app.session.develop_of(id).unwrap(), before);
    assert_eq!(h.request("ui.dialog.confirm", json!({}), timeout)["ok"], true);
    h.step_until(timeout, |h| h.app.session.develop_of(id).is_some_and(|d| d.enhance.model.is_some()));
    assert!(h.app.session.develop_of(id).unwrap().enhance.denoise > 50.0);
    h.app.session.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(h.app.session.develop_of(id).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}
