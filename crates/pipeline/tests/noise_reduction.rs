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
