use lightcraft_engine::Session;
use serde_json::json;

#[test]
fn preview_missing_model_is_actionable_and_does_not_commit() {
    let mut s = Session::with_demo();
    s.enhancer.model_dir = std::env::temp_dir().join(format!("lc-denoise-absent-{}", std::process::id()));
    let id = s.catalog.photos().next().unwrap().id;
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    let before = s.develop_of(s.active().unwrap()).unwrap();
    assert!(s.execute("enhance.denoise.preview", &json!({"photo":-1})).err().unwrap().to_string().contains("photo must"));
    assert!(s.execute("enhance.denoise.apply", &json!({"wait":"yes"})).err().unwrap().to_string().contains("wait must"));
    let error = s.execute("enhance.denoise.preview", &json!({})).unwrap_err();
    assert!(error.to_string().contains("install"), "{error}");
    assert_eq!(s.develop_of(s.active().unwrap()).unwrap(), before);
    assert!(s.execute("enhance.model.status", &json!({})).unwrap()["installed"].is_boolean());
}

#[cfg(feature = "reference-validation")]
#[test]
fn apply_is_one_undoable_edit_and_cached_amounts_render_without_weights() {
    use lightcraft_engine::catalog::{Op, Photo, Source};
    let dir = std::env::temp_dir().join(format!("lc-denoise-engine-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent"));
    let mut s = Session::new().with_fs();
    s.open_library(&dir, false).unwrap();
    s.enhancer.model_dir = model.clone();
    let id = s.catalog.alloc_photo_id();
    let path = dir.join("original.tif");
    let pixels: Vec<f32> = (0..32 * 24)
        .flat_map(|i| {
            let n = ((i * 17) % 23) as f32 * 0.001;
            [0.2 + n, 0.3 + n, 0.4 + n]
        })
        .collect();
    let encoded = lightcraft_codecs::encode_tiff(
        &lightcraft_codecs::EncodeImage::new(32, 24, 3, lightcraft_codecs::Samples::F32(&pixels)),
        lightcraft_codecs::TiffCompression::Deflate,
        &Default::default(),
    )
    .unwrap();
    std::fs::write(&path, &encoded).unwrap();
    let photo = Photo::new(id, Source::File { path: path.to_string_lossy().into() }, "original.tif", "TIFF", 32, 24, "2026-10-08");
    s.commit("Import", Op::AddPhoto { photo: Box::new(photo) }).unwrap();
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    let before = s.develop_of(id).unwrap();
    let baseline = s.render_job(id, 32, 24, false, true).unwrap().run().rendered.unwrap().image;
    s.execute("enhance.denoise.preview", &json!({"wait":true})).unwrap();
    assert_eq!(s.develop_of(id).unwrap(), before);
    let status = s.execute("enhance.denoise.status", &json!({})).unwrap();
    assert!(status["backend"].as_str().is_some_and(|name| !name.is_empty()));
    assert_eq!(status["cpu_fallback"].as_bool(), Some(status["fallback_reason"].is_string()));
    if std::env::var_os("LIGHTCRAFT_REQUIRE_DENOISE_GPU").is_some() {
        assert!(status["backend"].as_str().is_some_and(|name| name.contains("Vulkan")), "{status}");
        assert_eq!(status["cpu_fallback"], false);
    }
    s.execute("enhance.denoise.apply", &json!({"wait":true,"amount":50})).unwrap();
    assert!(s.develop_of(id).unwrap().enhance.model.is_some());
    assert_eq!(std::fs::read(&path).unwrap(), encoded);
    let enhanced = s.render_job(id, 32, 24, false, true).unwrap().run().rendered.unwrap().image;
    assert_ne!(enhanced, baseline, "enhancement must reach the public render boundary");
    s.execute("edit.undo", &json!({})).unwrap();
    assert_eq!(s.develop_of(id).unwrap(), before);
    s.execute("edit.redo", &json!({})).unwrap();
    s.close_library().unwrap();
    drop(s);
    // A photo can be imported from the enhancement folder. Regeneration must never
    // overwrite it, even when its filename collides with the content-derived cache key.
    let cache_path = std::fs::read_dir(dir.join("enhancements"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|e| e == "tif"))
        .unwrap();
    let saved_cache = std::fs::read(&cache_path).unwrap();
    std::fs::write(&cache_path, &encoded).unwrap();
    let mut collision = Session::new().with_fs();
    collision.open_library(&dir, false).unwrap();
    collision.enhancer.model_dir = model;
    let collision_id = collision.catalog.alloc_photo_id();
    collision
        .commit(
            "Import",
            Op::AddPhoto {
                photo: Box::new(Photo::new(
                    collision_id,
                    Source::File { path: cache_path.to_string_lossy().into() },
                    "cache-source.tif",
                    "TIFF",
                    32,
                    24,
                    "2026-10-08",
                )),
            },
        )
        .unwrap();
    let error = collision.execute("enhance.denoise.apply", &json!({"photo":collision_id.0,"wait":true})).err().unwrap();
    assert!(error.to_string().contains("overwrite original"), "{error}");
    assert_eq!(std::fs::read(&cache_path).unwrap(), encoded);
    assert!(collision.develop_of(collision_id).unwrap().enhance.model.is_none());
    collision.close_library().unwrap();
    drop(collision);
    std::fs::write(&cache_path, saved_cache).unwrap();
    let mut reopened = Session::new().with_fs();
    reopened.open_library(&dir, false).unwrap();
    reopened.enhancer.model_dir = dir.join("absent-model");
    assert_eq!(reopened.render_job(id, 32, 24, false, true).unwrap().run().rendered.unwrap().image, enhanced);
    reopened.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    reopened.execute("develop.set", &json!({"control":"enhance.denoise","value":100})).unwrap();
    let full = reopened.render_job(id, 32, 24, false, true).unwrap().run().rendered.unwrap().image;
    assert_ne!(full, enhanced, "amount changes reuse the saved correction without installed weights");
    reopened.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    reopened.execute("develop.set", &json!({"control":"enhance.denoise","value":0})).unwrap();
    let zero = reopened.render_job(id, 32, 24, false, true).unwrap().run().rendered.unwrap().image;
    assert_eq!(zero, reopened.render_job(id, 32, 24, true, true).unwrap().run().rendered.unwrap().image);
    // Copying settings cannot reuse another photo's cached pixels.
    reopened.execute("develop.set", &json!({"control":"enhance.denoise","value":50})).unwrap();
    reopened.execute("develop.copy", &json!({"groups":["detail"]})).unwrap();
    let destination = reopened.catalog.alloc_photo_id();
    let destination_path = dir.join("destination.tif");
    let destination_pixels = vec![0.6f32; 32 * 24 * 3];
    let destination_bytes = lightcraft_codecs::encode_tiff(
        &lightcraft_codecs::EncodeImage::new(32, 24, 3, lightcraft_codecs::Samples::F32(&destination_pixels)),
        lightcraft_codecs::TiffCompression::Deflate,
        &Default::default(),
    )
    .unwrap();
    std::fs::write(&destination_path, destination_bytes).unwrap();
    reopened
        .commit(
            "Import",
            Op::AddPhoto {
                photo: Box::new(Photo::new(
                    destination,
                    Source::File { path: destination_path.to_string_lossy().into() },
                    "destination.tif",
                    "TIFF",
                    32,
                    24,
                    "2026-10-08",
                )),
            },
        )
        .unwrap();
    reopened.execute("develop.paste", &json!({"ids":[destination.0]})).unwrap();
    assert!(reopened.develop_of(destination).unwrap().enhance.model.is_some());
    assert!(reopened.render_job(destination, 32, 24, false, true).unwrap().run().rendered.err().unwrap().contains("install"));
    reopened.close_library().unwrap();
    drop(reopened);
    std::fs::remove_dir_all(dir.join("enhancements")).unwrap();
    let mut missing = Session::new().with_fs();
    missing.open_library(&dir, false).unwrap();
    missing.enhancer.model_dir = dir.join("absent-model");
    missing.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    missing.execute("develop.set", &json!({"control":"enhance.denoise","value":50})).unwrap();
    assert!(lightcraft_engine::export::export_photo(&mut missing, id, &Default::default(), 1).err().unwrap().contains("install"));
    missing.close_library().unwrap();
    drop(missing);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "reference-validation")]
#[test]
fn stale_and_cancelled_workers_never_replace_conflicting_edits() {
    use lightcraft_engine::catalog::{Op, Photo, Source};
    let dir = std::env::temp_dir().join(format!("lc-denoise-stale-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("source.tif");
    let pixels = vec![0.25f32; 32 * 24 * 3];
    let bytes = lightcraft_codecs::encode_tiff(
        &lightcraft_codecs::EncodeImage::new(32, 24, 3, lightcraft_codecs::Samples::F32(&pixels)),
        lightcraft_codecs::TiffCompression::Deflate,
        &Default::default(),
    )
    .unwrap();
    std::fs::write(&path, bytes).unwrap();
    let mut s = Session::new().with_fs();
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
    s.execute("enhance.denoise.apply", &json!({"wait":false})).unwrap();
    s.execute("develop.set", &json!({"control":"light.exposure","value":1.25})).unwrap();
    let start = std::time::Instant::now();
    while matches!(s.enhancer.status().state.as_str(), "preparing" | "processing") {
        let _ = s.execute("enhance.denoise.status", &json!({}));
        assert!(start.elapsed() < std::time::Duration::from_secs(30));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(s.enhancer.status().error.unwrap().contains("stale"));
    assert!(s.develop_of(id).unwrap().enhance.model.is_none());
    assert_eq!(s.develop_of(id).unwrap().light.exposure, 1.25);
    s.execute("enhance.denoise.apply", &json!({"wait":false})).unwrap();
    s.execute("enhance.denoise.cancel", &json!({})).unwrap();
    assert_eq!(s.execute("enhance.denoise.status", &json!({})).unwrap()["state"], "cancelled");
    assert!(s.develop_of(id).unwrap().enhance.model.is_none());
    s.execute("enhance.denoise.apply", &json!({"wait":false})).unwrap();
    s.execute("photo.delete", &json!({"ids":[id.0]})).unwrap();
    let start = std::time::Instant::now();
    while matches!(s.enhancer.status().state.as_str(), "preparing" | "processing") {
        s.execute("enhance.denoise.status", &json!({})).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(30));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(s.enhancer.status().error.as_ref().is_some_and(|error| error.contains("deleted")));
    assert!(s.develop_of(id).unwrap().enhance.model.is_none());
    s.commit("Restore fixture", Op::SetDeleted { id, deleted: false }).unwrap();
    s.execute("library.select", &json!({"ids":[id.0]})).unwrap();
    s.execute("enhance.denoise.apply", &json!({"wait":false})).unwrap();
    let replacement = lightcraft_codecs::encode_tiff(
        &lightcraft_codecs::EncodeImage::new(32, 24, 3, lightcraft_codecs::Samples::F32(&vec![0.75; 32 * 24 * 3])),
        lightcraft_codecs::TiffCompression::Deflate,
        &Default::default(),
    )
    .unwrap();
    std::fs::write(&path, replacement).unwrap();
    let start = std::time::Instant::now();
    while matches!(s.enhancer.status().state.as_str(), "preparing" | "processing") {
        s.execute("enhance.denoise.status", &json!({})).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(30));
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(s.enhancer.status().error.as_ref().is_some_and(|e| e.contains("changed") || e.contains("stale")));
    assert!(s.develop_of(id).unwrap().enhance.model.is_none());
    // A replacement must invalidate the old result even when its dimensions are identical.
    s.enhancer.model_dir = dir.join("absent-model");
    assert!(s.execute("enhance.denoise.apply", &json!({"wait":true})).err().unwrap().to_string().contains("install"));
    s.enhancer.model_dir = std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").unwrap().into();
    s.execute("enhance.denoise.apply", &json!({"wait":false})).unwrap();
    s.open_library(&dir.join("switched"), false).unwrap();
    assert_eq!(s.execute("enhance.denoise.status", &json!({})).unwrap()["state"], "cancelled");
    assert!(s.catalog.photos().all(|p| p.develop.enhance.model.is_none()));
    s.close_library().unwrap();
    drop(s);
    std::fs::remove_dir_all(dir).unwrap();
}
