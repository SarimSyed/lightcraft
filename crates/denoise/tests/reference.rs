#![cfg(feature = "reference-validation")]
use lightcraft_denoise::{Cancellation, Denoiser, NafNet, Region, SrgbRgb};
use lightcraft_raster::Rgb32f;

#[test]
fn author_checkpoint_matches_independent_pytorch_padding_and_overlap() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required reference bundle absent"));
    let model = NafNet::load(&dir.join("nafnet-sidd-width32-v1.safetensors")).unwrap();
    let refs = candle_core::safetensors::load(dir.join("reference.safetensors"), &candle_core::Device::Cpu).unwrap();
    for name in ["padding", "tiles"] {
        let input = &refs[&format!("{name}.input")];
        let (_, _, h, w) = input.dims4().unwrap();
        let data = input.permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let image = Rgb32f { width: w, height: h, data: data.as_chunks::<3>().0.to_vec() };
        let actual = if name == "padding" {
            model.infer_patch(&SrgbRgb::new(image.clone()).unwrap(), &Cancellation::default()).unwrap()
        } else {
            model.infer(&SrgbRgb::new(image.clone()).unwrap(), None, &Cancellation::default(), &|_, _| {}).unwrap().image
        };
        let expected = refs[&format!("{name}.output")].permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let maximum = actual.data.iter().flatten().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{name}: maximum reference error {maximum}");
        assert!(maximum < 0.0001, "{name}: maximum error {maximum}");
        if name == "tiles" {
            let clean = refs["tiles.clean"].permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let noisy: Vec<f32> = image.data.iter().flatten().copied().collect();
            let mse = |values: &[f32]| values.iter().zip(&clean).map(|(a, b)| (a - b).powi(2) as f64).sum::<f64>() / clean.len() as f64;
            let output: Vec<f32> = actual.data.iter().flatten().copied().collect();
            println!("MSE: noisy {}, denoised {}", mse(&noisy), mse(&output));
            assert!(mse(&output) < mse(&noisy) * 0.2, "denoising must measurably improve the procedural scene");
            let brightness = output.iter().zip(&clean).map(|(a, b)| (a - b) as f64).sum::<f64>() / clean.len() as f64;
            assert!(brightness.abs() < 0.005, "brightness bias {brightness}");
            let cancel = Cancellation::default();
            cancel.cancel();
            assert!(model.infer(&SrgbRgb::new(image.clone()).unwrap(), None, &cancel, &|_, _| {}).is_err());
            let between_tiles = Cancellation::default();
            assert!(
                model
                    .infer(&SrgbRgb::new(image.clone()).unwrap(), None, &between_tiles, &|done, _| {
                        if done == 1 {
                            between_tiles.cancel();
                        }
                    })
                    .is_err(),
                "cancellation after a completed tile must stop inference"
            );
            let crop = Region { x: 190, y: 150, width: 60, height: 40 };
            let part = model.infer(&SrgbRgb::new(image).unwrap(), Some(crop), &Cancellation::default(), &|_, _| {}).unwrap();
            for y in 0..40 {
                for x in 0..60 {
                    assert_eq!(part.image.get(x, y), actual.get(x + 190, y + 150));
                }
            }
        }
    }
}
