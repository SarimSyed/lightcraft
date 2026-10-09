#![cfg(feature = "reference-validation")]
use lightcraft_denoise::{Cancellation, Denoiser, NafNet, Region, SrgbRgb};
use lightcraft_raster::Rgb32f;

/// Opt-in, hardware-specific performance gate; normal CI has no wall-clock assertions.
#[test]
#[ignore = "requires an explicit CPU tile latency budget and real checkpoint"]
fn cpu_tile_inference_meets_explicit_latency_budget() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model bundle absent"));
    let budget_ms: f64 = std::env::var("LIGHTCRAFT_NAFNET_TILE_BUDGET_MS").expect("explicit latency budget required").parse().unwrap();
    assert!(budget_ms.is_finite() && budget_ms > 0.0);
    let model = NafNet::load(&dir.join("nafnet-sidd-width32-v1.safetensors")).unwrap();
    let mut image = Rgb32f::new(256, 256);
    for (i, pixel) in image.data.iter_mut().enumerate() {
        *pixel = [0.3 + (i % 23) as f32 * 0.004, 0.5, 0.7];
    }
    let input = SrgbRgb::new(image).unwrap();
    model.infer_patch(&input, &Cancellation::default()).unwrap(); // warm up the CPU kernels
    let mut timings = Vec::new();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        model.infer_patch(&input, &Cancellation::default()).unwrap();
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    println!("CPU 256px tile inference milliseconds: {timings:?}");
    assert!(timings.iter().all(|ms| *ms <= budget_ms), "CPU tile exceeds {budget_ms} ms budget: {timings:?}");
}

#[test]
fn author_checkpoint_matches_independent_pytorch_padding_and_overlap() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required reference bundle absent"));
    let model = NafNet::load(&dir.join("nafnet-sidd-width32-v1.safetensors")).unwrap();
    let refs = candle_core::safetensors::load(dir.join("reference.safetensors"), &candle_core::Device::Cpu).unwrap();
    for name in ["tiny", "padding", "tiles"] {
        let input = &refs[&format!("{name}.input")];
        let (_, _, h, w) = input.dims4().unwrap();
        let data = input.permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let image = Rgb32f { width: w, height: h, data: data.as_chunks::<3>().0.to_vec() };
        let actual = if name != "tiles" {
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

/// Explicit hardware gate: it must fail, rather than pass on CPU, without a GPU.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a physical Vulkan GPU and real checkpoint"]
fn vulkan_checkpoint_matches_reference_and_reports_gpu_use() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent"));
    let model = vulkan_model(&dir);
    assert!(model.backend().contains("Vulkan"), "expected hardware Vulkan: {}", model.backend());
    let refs = candle_core::safetensors::load(dir.join("reference.safetensors"), &candle_core::Device::Cpu).unwrap();
    for name in ["tiny", "padding", "tiles"] {
        let input = &refs[&format!("{name}.input")];
        let (_, _, h, w) = input.dims4().unwrap();
        let data = input.permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let image = SrgbRgb::new(Rgb32f { width: w, height: h, data: data.as_chunks::<3>().0.to_vec() }).unwrap();
        let started = std::time::Instant::now();
        let actual = if name == "tiles" {
            model.infer(&image, None, &Cancellation::default(), &|_, _| {}).unwrap().image
        } else {
            model.infer_patch(&image, &Cancellation::default()).unwrap()
        };
        let expected = refs[&format!("{name}.output")].permute((0, 2, 3, 1)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let maximum = actual.data.iter().flatten().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{} {name}: max error {maximum}, elapsed {:?}", model.backend(), started.elapsed());
        assert!(maximum < 0.0001, "{name}: reference error {maximum}");
        assert!(model.backend().contains("Vulkan"), "GPU must not silently fall back");
        if name == "tiles" {
            let crop = Region { x: 190, y: 150, width: 60, height: 40 };
            let part = model.infer(&image, Some(crop), &Cancellation::default(), &|_, _| {}).unwrap();
            assert!(!part.cpu_fallback);
            for y in 0..40 {
                for x in 0..60 {
                    assert_eq!(part.image.get(x, y), actual.get(x + 190, y + 150));
                }
            }
            let cancel = Cancellation::default();
            assert!(
                model
                    .infer(&image, None, &cancel, &|done, _| {
                        if done == 1 {
                            cancel.cancel();
                        }
                    })
                    .is_err()
            );
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a physical Vulkan GPU and real checkpoint"]
fn gpu_disabled_mid_operation_retries_cpu_with_reported_reason() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent"));
    let model = vulkan_model(&dir);
    assert!(model.backend().contains("Vulkan"));
    let image = SrgbRgb::new(Rgb32f { width: 1, height: 1, data: vec![[0.3, 0.5, 0.7]] }).unwrap();
    model.infer_patch(&image, &Cancellation::default()).unwrap();
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            GPU_DISABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let _reset = Reset;
    GPU_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    let output = model.infer(&image, None, &Cancellation::default(), &|_, _| {}).unwrap();
    assert_eq!(output.backend, "CPU");
    assert!(output.cpu_fallback);
    assert!(output.fallback_reason.as_ref().is_some_and(|reason| reason.contains("disabled")));
    assert!(output.image.data.iter().flatten().all(|value| value.is_finite()));
}

#[cfg(target_os = "linux")]
static GPU_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(target_os = "linux")]
fn vulkan_model(dir: &std::path::Path) -> NafNet {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
    )
    .expect("physical GPU required");
    let info = adapter.get_info();
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu, "hardware GPU required");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() })).unwrap();
    NafNet::load_vulkan(&dir.join("nafnet-sidd-width32-v1.safetensors"), Ok((device, queue, info)), || {
        GPU_DISABLED.load(std::sync::atomic::Ordering::Relaxed).then(|| "GPU disabled by preference".into())
    })
    .unwrap()
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires an explicit GPU tile latency budget and real checkpoint"]
fn vulkan_tile_inference_meets_explicit_latency_budget() {
    let dir = std::path::PathBuf::from(std::env::var("LIGHTCRAFT_NAFNET_REFERENCE").expect("required model absent"));
    let budget: f64 = std::env::var("LIGHTCRAFT_NAFNET_GPU_TILE_BUDGET_MS").expect("explicit GPU budget required").parse().unwrap();
    assert!(budget.is_finite() && budget > 0.0);
    let model = vulkan_model(&dir);
    let image =
        SrgbRgb::new(Rgb32f { width: 256, height: 256, data: (0..256 * 256).map(|i| [0.3 + (i % 23) as f32 * 0.004, 0.5, 0.7]).collect() }).unwrap();
    // Warm driver compilation and the reusable activation/binding cache separately.
    for _ in 0..2 {
        model.infer_patch(&image, &Cancellation::default()).unwrap();
    }
    let mut timings = Vec::new();
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let result = model.infer_patch(&image, &Cancellation::default()).unwrap();
        assert!(result.data.iter().flatten().all(|v| v.is_finite()));
        assert!(model.backend().contains("Vulkan"), "CPU fallback cannot pass the GPU gate");
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    println!("GPU 256px tile milliseconds: {timings:?}");
    assert!(timings.iter().all(|ms| *ms <= budget), "GPU tile exceeds {budget} ms budget: {timings:?}");
}
