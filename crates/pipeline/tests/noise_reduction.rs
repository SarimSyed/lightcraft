//! Noise reduction is tested through the public renderer, against independently clean scenes.
use lightcraft_develop::DevelopSettings;
use lightcraft_pipeline::{DeepSamples, OutputDepth, RenderRequest, SourceInfo, StageCache, render, render_cached};
use lightcraft_raster::Rgb32f;
use std::sync::Arc;

fn linear(src: &Rgb32f, settings: &DevelopSettings, size: usize) -> Vec<f32> {
    let request = RenderRequest { depth: OutputDepth::F32Linear, ..RenderRequest::fit(size, size) };
    let result = render(src, &SourceInfo::default(), settings, &request);
    match result.deep.expect("linear output").samples {
        DeepSamples::F32(pixels) => pixels,
        _ => panic!("expected linear samples"),
    }
}

fn texture() -> Rgb32f {
    Rgb32f::from_fn(128, 128, |x, y| {
        let structure = (x as f32 * std::f32::consts::TAU / 16.0).sin() * 0.06;
        let noise = (((x * 7919 + y * 104729) % 97) as f32 / 97.0 - 0.5) * 0.01;
        [0.2 + structure + noise; 3]
    })
}

fn amplitude(pixels: &[f32]) -> f32 {
    pixels.as_chunks::<3>().0.iter().enumerate().map(|(i, p)| p[1] * ((i % 128) as f32 * std::f32::consts::TAU / 16.0).sin()).sum::<f32>().abs() * 2.0
        / (pixels.len() / 3) as f32
}

#[test]
fn luminance_contrast_preserves_structure_without_disabling_noise_reduction() {
    let source = texture();
    let mut settings = DevelopSettings::default();
    let original = amplitude(&linear(&source, &settings, 128));
    settings.detail.nr_luminance = 80.0;
    let smooth = amplitude(&linear(&source, &settings, 128));
    settings.detail.nr_contrast = 100.0;
    let preserved = amplitude(&linear(&source, &settings, 128));
    assert!(preserved > smooth * 1.05 && preserved < original, "original {original}, smooth {smooth}, contrast {preserved}");
}

#[test]
fn contrast_change_refreshes_the_cached_preview() {
    let source = Arc::new(texture());
    let mut settings = DevelopSettings::default();
    settings.detail.nr_luminance = 80.0;
    let cache = StageCache::default();
    let request = RenderRequest::fit(128, 128);
    let info = SourceInfo::default();
    let before = render_cached(&source, &info, &settings, &request, &cache).image;
    settings.detail.nr_contrast = 100.0;
    let after = render_cached(&source, &info, &settings, &request, &cache).image;
    let fresh = render(&source, &info, &settings, &request).image;
    assert!(before.data != fresh.data, "Contrast must change the render");
    assert!(after.data == fresh.data, "cached preview must reflect Contrast changes");
}

#[test]
fn luminance_nr_removes_flat_noise_without_changing_brightness() {
    let source = Rgb32f::from_fn(128, 128, |x, y| {
        let noise = (((x * 7919 + y * 104729) % 97) as f32 / 97.0 - 0.5) * 0.06;
        [0.2 + noise; 3]
    });
    let stats = |pixels: &[f32]| {
        let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
        let variance = pixels.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pixels.len() as f32;
        (mean, variance)
    };
    let mut settings = DevelopSettings::default();
    let (mean, variance) = stats(&linear(&source, &settings, 128));
    settings.detail.nr_luminance = 80.0;
    for contrast in [0.0, 100.0] {
        settings.detail.nr_contrast = contrast;
        let (filtered_mean, filtered_variance) = stats(&linear(&source, &settings, 128));
        assert!(
            filtered_variance <= variance * 0.5 && (filtered_mean - mean).abs() <= mean * 0.01,
            "contrast {contrast}: mean {mean} -> {filtered_mean}, variance {variance} -> {filtered_variance}"
        );
    }
}

#[test]
fn dependent_controls_do_not_change_pixels_when_noise_reduction_is_off() {
    let source = texture();
    let mut settings = DevelopSettings::default();
    let plain = linear(&source, &settings, 128);
    settings.detail.nr_detail = 100.0;
    settings.detail.nr_contrast = 100.0;
    settings.detail.nr_color_detail = 100.0;
    settings.detail.nr_color_smoothness = 100.0;
    assert!(linear(&source, &settings, 128) == plain);
}

fn colour_scene(noisy: bool) -> Rgb32f {
    colour_scene_at(noisy, 64)
}

fn colour_scene_at(noisy: bool, edge: usize) -> Rgb32f {
    // These two independently specified colours have equal Rec.2020 luminance (~0.20508).
    Rgb32f::from_fn(128, 128, |x, y| {
        let mut colour = if x < edge { [0.5, 0.1, 0.1] } else { [0.1, 0.25498524, 0.1] };
        if noisy {
            let noise = (((x * 7919 + y * 104729) % 97) as f32 / 97.0 - 0.5) * 0.02;
            colour[0] += noise;
            colour[1] -= noise * (0.2627 / 0.6780);
        }
        colour
    })
}

fn colour_step(pixels: &[f32], size: usize) -> f32 {
    colour_step_at(pixels, size, size / 2)
}

fn colour_step_at(pixels: &[f32], size: usize, edge: usize) -> f32 {
    let pixels = pixels.as_chunks::<3>().0;
    let mut step = 0.0;
    for y in 8..size - 8 {
        step += pixels[y * size + edge - 1][0] - pixels[y * size + edge][0];
    }
    step / (size - 16) as f32
}

#[test]
fn colour_noise_reduction_preserves_equal_luminance_boundaries() {
    let mut settings = DevelopSettings::default();
    for size in [64, 128] {
        let reference = linear(&colour_scene(false), &settings, size);
        settings.detail.nr_color = 80.0;
        let filtered = linear(&colour_scene(true), &settings, size);
        let ratio = colour_step(&filtered, size) / colour_step(&reference, size);
        assert!(ratio >= 0.95, "{size}px colour-edge retention {ratio}");
        settings.detail.nr_color = 0.0;
    }
}

#[test]
fn colour_edges_are_preserved_between_coefficient_samples() {
    for edge in [31, 63, 65, 97] {
        let mut settings = DevelopSettings::default();
        let reference = linear(&colour_scene_at(false, edge), &settings, 128);
        settings.detail.nr_color = 80.0;
        let filtered = linear(&colour_scene_at(true, edge), &settings, 128);
        let ratio = colour_step_at(&filtered, 128, edge) / colour_step_at(&reference, 128, edge);
        assert!(ratio >= 0.95 - 1e-5, "edge {edge}: retention {ratio}");
    }
}

#[test]
fn colour_nr_removes_speckles_preserves_luminance_and_detail() {
    let clean = Rgb32f::filled(128, 128, [0.2; 3]);
    let noisy = Rgb32f::from_fn(128, 128, |x, y| {
        let n = (((x * 7919 + y * 104729) % 97) as f32 / 97.0 - 0.5) * 0.02;
        [0.2 + n, 0.2 - n * (0.2627 / 0.6780), 0.2]
    });
    let mut settings = DevelopSettings::default();
    for size in [64, 128] {
        settings.detail.nr_color = 0.0;
        let reference = linear(&clean, &settings, size);
        let original = linear(&noisy, &settings, size);
        settings.detail.nr_color = 80.0;
        settings.detail.nr_color_detail = 50.0;
        let filtered = linear(&noisy, &settings, size);
        let error = |pixels: &[f32]| {
            pixels
                .as_chunks::<3>()
                .0
                .iter()
                .zip(reference.as_chunks::<3>().0)
                .enumerate()
                .filter(|(i, _)| {
                    let x = i % size;
                    x > 8 && x < size - 8 && x.abs_diff(size / 2) > 12
                })
                .map(|(_, (p, r))| (p[0] - r[0]).abs())
                .sum::<f32>()
        };
        assert!(error(&filtered) <= error(&original) * 0.6, "{size}px error {} -> {}", error(&original), error(&filtered));
        let luminance = |p: &[f32; 3]| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722;
        let max_brightness_error =
            filtered.as_chunks::<3>().0.iter().zip(original.as_chunks::<3>().0).map(|(p, r)| (luminance(p) - luminance(r)).abs()).fold(0.0, f32::max);
        assert!(max_brightness_error < 0.0001, "luminance drift {max_brightness_error}");
        settings.detail.nr_color_detail = 100.0;
        let detailed = linear(&colour_scene(true), &settings, size);
        settings.detail.nr_color_detail = 50.0;
        let edges = linear(&colour_scene(true), &settings, size);
        assert!(colour_step(&detailed, size) > colour_step(&edges, size), "higher Detail must preserve colour edges");
    }
}

#[test]
fn nr_preserves_constant_black_and_singleton_images() {
    let mut settings = DevelopSettings::default();
    for (w, h) in [(1, 1), (1, 16), (16, 1), (16, 16)] {
        for colour in [[0.0; 3], [0.2; 3], [0.05, 0.2, 0.1], [1e-9; 3]] {
            let source = Rgb32f::filled(w, h, colour);
            settings.detail.nr_luminance = 0.0;
            settings.detail.nr_color = 0.0;
            let reference = linear(&source, &settings, 16);
            settings.detail.nr_luminance = 80.0;
            settings.detail.nr_color = 80.0;
            let filtered = linear(&source, &settings, 16);
            assert!(filtered.iter().zip(&reference).all(|(v, r)| v.is_finite() && (v - r).abs() < 0.00001), "{w}x{h} {colour:?}");
        }
    }
}
