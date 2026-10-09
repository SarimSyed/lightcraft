use lightcraft_nafnet::NafNet;
use lightcraft_nafnet::{Cancellation, SrgbRgb, model_input, restore_working};
use lightcraft_raster::Rgb32f;

#[test]
fn missing_and_corrupt_checkpoints_return_actionable_errors() {
    let dir = std::env::temp_dir().join(format!("lc-denoise-api-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    assert!(NafNet::load(&dir.join("missing.safetensors")).err().unwrap().to_string().contains("checkpoint"));
    let corrupt = dir.join("corrupt.safetensors");
    std::fs::write(&corrupt, u64::MAX.to_le_bytes()).unwrap();
    assert!(NafNet::load(&corrupt).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rgb_contract_rejects_invalid_buffers_and_keeps_unclamped_highlights() {
    for image in [
        Rgb32f::new(0, 0),
        Rgb32f { width: usize::MAX, height: 2, data: vec![] },
        Rgb32f::filled(1, 1, [f32::NAN; 3]),
        Rgb32f::filled(1, 1, [2.0; 3]),
    ] {
        assert!(SrgbRgb::new(image).is_err());
    }
    let original = Rgb32f::filled(2, 1, [4.0, -0.5, 0.25]);
    let input = model_input(&original).unwrap();
    assert!(input.image().data.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
    let recovered = restore_working(&original, input.image(), None).unwrap();
    for (a, b) in recovered.data.iter().flatten().zip(original.data.iter().flatten()) {
        assert!((a - b).abs() < 1e-6);
    }
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(cancel.check().is_err());
}
