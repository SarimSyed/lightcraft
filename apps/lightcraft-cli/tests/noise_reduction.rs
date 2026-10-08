//! Real CLI exports and control-channel interactions for noise reduction.
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
