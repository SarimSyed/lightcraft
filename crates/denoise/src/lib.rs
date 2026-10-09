//! RGB-stage image restoration. Sensor-domain models require a separate input contract.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use lightcraft_raster::Rgb32f;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
mod network;
#[cfg(target_os = "linux")]
mod vulkan;

pub const MODEL_ID: &str = "nafnet-sidd-width32";
pub const PROCESSING_REVISION: &str = "rec2020-correction-v1-tiles256-overlap64-global-pool";
pub const MAX_PIXELS: usize = 100_000_000;
pub const TILE: usize = 256;
pub const OVERLAP: usize = 64;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelDescription {
    pub id: String,
    pub checkpoint_digest: String,
    pub input_domain: &'static str,
    pub licence: &'static str,
}

/// A validated, interleaved float sRGB-encoded RGB image, never sensor data.
pub struct SrgbRgb(Rgb32f);
impl SrgbRgb {
    pub fn new(image: Rgb32f) -> Result<Self> {
        validate(&image)?;
        if image.data.iter().flatten().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(Error::Input("model input must be in 0..1".into()));
        }
        Ok(Self(image))
    }
    pub fn image(&self) -> &Rgb32f {
        &self.0
    }
}
pub fn validate(image: &Rgb32f) -> Result<()> {
    let n = image.width.checked_mul(image.height).ok_or_else(|| Error::Input("dimensions overflow".into()))?;
    if n == 0
        || n > MAX_PIXELS
        || image.width > 32768
        || image.height > 32768
        || n != image.data.len()
        || image.data.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(Error::Input("invalid dimensions, buffer length or non-finite samples".into()));
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Region {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}
impl Region {
    fn validate(self, image: &Rgb32f) -> Result<Self> {
        if self.width == 0
            || self.height == 0
            || self.x.checked_add(self.width).is_none_or(|e| e > image.width)
            || self.y.checked_add(self.height).is_none_or(|e| e > image.height)
        {
            return Err(Error::Input("preview region lies outside the source".into()));
        }
        Ok(self)
    }
}
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Relaxed) { Err(Error::Cancelled) } else { Ok(()) }
    }
}
pub struct Inference {
    pub image: Rgb32f,
    pub region: Region,
    pub cpu_fallback: bool,
    pub backend: String,
    pub gpu_buffer_bytes: u64,
    pub fallback_reason: Option<String>,
}
pub trait Denoiser: Send + Sync {
    fn description(&self) -> ModelDescription;
    fn infer(&self, input: &SrgbRgb, region: Option<Region>, cancel: &Cancellation, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<Inference>;
}

/// Versioned input adapter. Only this model input is clamped; the source is untouched.
pub fn model_input(source: &Rgb32f) -> Result<SrgbRgb> {
    validate(source)?;
    let matrix = lightcraft_color::WORKING.to_space(&lightcraft_color::SRGB);
    let image = Rgb32f {
        width: source.width,
        height: source.height,
        data: source.data.iter().map(|p| matrix.apply_f32(*p).map(|v| lightcraft_color::transfer::linear_to_srgb(v.clamp(0.0, 1.0)))).collect(),
    };
    SrgbRgb::new(image)
}

/// Add the model's linear-light correction to the unclamped Rec.2020 source.
pub fn restore_working(source: &Rgb32f, prediction: &Rgb32f, region: Option<Region>) -> Result<Rgb32f> {
    validate(source)?;
    validate(prediction)?;
    let r = region.unwrap_or(Region { x: 0, y: 0, width: source.width, height: source.height }).validate(source)?;
    if prediction.width != r.width || prediction.height != r.height {
        return Err(Error::Input("prediction dimensions differ from region".into()));
    }
    let to_srgb = lightcraft_color::WORKING.to_space(&lightcraft_color::SRGB);
    let to_working = lightcraft_color::SRGB.to_space(&lightcraft_color::WORKING);
    let mut data = Vec::with_capacity(prediction.data.len());
    for (i, p) in prediction.data.iter().enumerate() {
        let original = source
            .data
            .get((r.y + i / r.width) * source.width + r.x + i % r.width)
            .ok_or_else(|| Error::Input("invalid correction extent".into()))?;
        let clipped = to_srgb.apply_f32(*original).map(|v| v.clamp(0.0, 1.0));
        let corrected = p.map(lightcraft_color::transfer::srgb_to_linear);
        let delta = to_working.apply_f32(std::array::from_fn(|c| corrected[c] - clipped[c]));
        data.push(std::array::from_fn(|c| original[c] + delta[c]));
    }
    let image = Rgb32f { width: r.width, height: r.height, data };
    validate(&image)?;
    Ok(image)
}

pub fn blend(original: &Rgb32f, enhanced: &Rgb32f, amount: f64) -> Result<Rgb32f> {
    validate(original)?;
    validate(enhanced)?;
    if !amount.is_finite() || !(0.0..=100.0).contains(&amount) || original.width != enhanced.width || original.height != enhanced.height {
        return Err(Error::Input("invalid amount or enhanced dimensions".into()));
    }
    if amount == 0.0 {
        return Ok(original.clone());
    }
    let factor = (amount / 100.0) as f32;
    let image = Rgb32f {
        width: original.width,
        height: original.height,
        data: original.data.iter().zip(&enhanced.data).map(|(a, b)| std::array::from_fn(|c| a[c] + factor * (b[c] - a[c]))).collect(),
    };
    validate(&image)?;
    Ok(image)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    #[error("invalid RGB input: {0}")]
    Input(String),
    #[error("AI denoise cancelled")]
    Cancelled,
    #[error("AI denoise inference: {0}")]
    Inference(String),
}
pub type Result<T> = std::result::Result<T, Error>;

pub struct NafNet {
    #[cfg(not(target_arch = "wasm32"))]
    weight_bytes: usize,
    #[cfg(not(target_arch = "wasm32"))]
    fallback_reason: std::sync::Mutex<Option<String>>,
    #[cfg(target_os = "linux")]
    vulkan: Option<vulkan::Network>,
    pub checkpoint_digest: String,
    #[cfg(not(target_arch = "wasm32"))]
    network: network::Network,
    #[cfg(not(target_arch = "wasm32"))]
    device: candle_core::Device,
    #[cfg(target_os = "macos")]
    metal: Option<(network::Network, candle_core::Device)>,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    weights: std::collections::HashMap<String, candle_core::Tensor>,
    #[cfg(not(target_arch = "wasm32"))]
    fallback: AtomicBool,
}

impl NafNet {
    /// Retained host weights/workspace and device buffers, excluding allocator/driver overhead.
    /// Reading this does not wait for active inference.
    pub fn resident_bytes(&self) -> (usize, u64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            #[cfg(target_os = "linux")]
            if let Some(gpu) = &self.vulkan {
                return (self.weight_bytes.saturating_add(gpu.host_bytes()), gpu.buffer_bytes());
            }
            #[cfg(target_os = "macos")]
            if self.metal.is_some() {
                return (self.weight_bytes, self.weight_bytes as u64);
            }
            (self.weight_bytes, 0)
        }
        #[cfg(target_arch = "wasm32")]
        (0, 0)
    }

    /// The inference backend in use; after a recoverable GPU failure this reports CPU.
    pub fn backend(&self) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        if !self.fallback.load(Ordering::Relaxed) {
            #[cfg(target_os = "linux")]
            if let Some(gpu) = &self.vulkan {
                return gpu.name.clone();
            }
            #[cfg(target_os = "macos")]
            if self.metal.is_some() {
                return "Metal".into();
            }
        }
        "CPU".into()
    }
    pub fn load_accelerated(path: &Path) -> Result<Self> {
        let model = Self::load(path)?;
        #[cfg(target_os = "macos")]
        {
            let mut model = model;
            let accelerated = (|| -> candle_core::Result<_> {
                let device = candle_core::Device::new_metal(0)?;
                let weights =
                    model.weights.iter().map(|(name, tensor)| Ok((name.clone(), tensor.to_device(&device)?))).collect::<candle_core::Result<_>>()?;
                let network = network::Network::new(candle_nn::VarBuilder::from_tensors(weights, candle_core::DType::F32, &device))?;
                Ok((network, device))
            })();
            match accelerated {
                Ok(metal) => model.metal = Some(metal),
                Err(error) => {
                    *model.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.to_string());
                    model.fallback.store(true, Ordering::Relaxed);
                }
            }
            Ok(model)
        }
        #[cfg(not(target_os = "macos"))]
        Ok(model)
    }
    /// Linux inference using a device supplied by the host. The availability
    /// callback respects host preferences/device loss without depending on its renderer.
    #[cfg(target_os = "linux")]
    pub fn load_vulkan(
        path: &Path,
        device: std::result::Result<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo), String>,
        unavailable: fn() -> Option<String>,
    ) -> Result<Self> {
        let mut model = Self::load(path)?;
        match device.and_then(|device| vulkan::Network::new(&model.weights, device, unavailable)) {
            Ok(gpu) => model.vulkan = Some(gpu),
            Err(error) => {
                *model.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                model.fallback.store(true, Ordering::Relaxed);
            }
        }
        Ok(model)
    }
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Checkpoint(format!("{}: {e}; install NAFNet SIDD width-32", path.display())))?;
        let size = file.metadata().map_err(|e| Error::Checkpoint(e.to_string()))?.len();
        if !(8..=128 << 20).contains(&size) {
            return Err(Error::Checkpoint("file size exceeds the width-32 checkpoint limit".into()));
        }
        use std::io::Read;
        let mut bytes = Vec::new();
        file.take((128 << 20) + 1).read_to_end(&mut bytes).map_err(|e| Error::Checkpoint(e.to_string()))?;
        if bytes.len() > 128 << 20 {
            return Err(Error::Checkpoint("file grew beyond the size limit".into()));
        }
        let prefix: [u8; 8] = bytes.get(..8).and_then(|s| s.try_into().ok()).ok_or_else(|| Error::Checkpoint("truncated header".into()))?;
        let len = usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| Error::Checkpoint("header too large".into()))?;
        if len > 1 << 20 {
            return Err(Error::Checkpoint("header too large".into()));
        }
        let end = 8usize.checked_add(len).ok_or_else(|| Error::Checkpoint("header too large".into()))?;
        let header = bytes.get(8..end).ok_or_else(|| Error::Checkpoint("truncated header".into()))?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| Error::Checkpoint(e.to_string()))?;
        if header.is_empty() || header.len() > 1024 {
            return Err(Error::Checkpoint("invalid tensor count".into()));
        }
        for (name, value) in header.iter().filter(|(name, _)| name.as_str() != "__metadata__") {
            let shape = value.get("shape").and_then(|v| v.as_array()).ok_or_else(|| Error::Checkpoint(format!("{name}: missing shape")))?;
            if shape.is_empty() || shape.len() > 4 {
                return Err(Error::Checkpoint(format!("{name}: invalid rank")));
            }
            let size = shape
                .iter()
                .try_fold(4usize, |n, v| {
                    let d = usize::try_from(v.as_u64()?).ok()?;
                    if d == 0 || d > 1024 {
                        return None;
                    }
                    n.checked_mul(d)
                })
                .ok_or_else(|| Error::Checkpoint(format!("{name}: invalid dimensions")))?;
            let offsets =
                value.get("data_offsets").and_then(|v| v.as_array()).ok_or_else(|| Error::Checkpoint(format!("{name}: missing offsets")))?;
            let start = offsets.first().and_then(|v| v.as_u64());
            let stop = offsets.get(1).and_then(|v| v.as_u64());
            if offsets.len() != 2
                || value.get("dtype").and_then(|v| v.as_str()) != Some("F32")
                || start.zip(stop).is_none_or(|(a, b)| b.checked_sub(a) != Some(size as u64) || b > bytes.len().saturating_sub(end) as u64)
            {
                return Err(Error::Checkpoint(format!("{name}: invalid tensor extent or type")));
            }
            let (Some(start), Some(stop)) = (start, stop) else {
                return Err(Error::Checkpoint("invalid offsets".into()));
            };
            let data = bytes.get(end + start as usize..end + stop as usize).ok_or_else(|| Error::Checkpoint("invalid tensor extent".into()))?;
            if data.as_chunks::<4>().0.iter().any(|v| !f32::from_le_bytes(*v).is_finite()) {
                return Err(Error::Checkpoint(format!("{name}: non-finite weights")));
            }
        }
        let checkpoint_digest = format!("{:x}", Sha256::digest(&bytes));
        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = candle_core::Device::Cpu;
            let tensors = candle_core::safetensors::load_buffer(&bytes, &device).map_err(|e| Error::Checkpoint(e.to_string()))?;
            let vb = candle_nn::VarBuilder::from_tensors(tensors.clone(), candle_core::DType::F32, &device);
            let network = network::Network::new(vb).map_err(|e| Error::Checkpoint(e.to_string()))?;
            Ok(Self {
                weight_bytes: tensors.values().map(|t| t.elem_count().saturating_mul(4)).sum(),
                checkpoint_digest,
                fallback_reason: std::sync::Mutex::new(None),
                network,
                device,
                fallback: AtomicBool::new(false),
                #[cfg(target_os = "macos")]
                metal: None,
                #[cfg(any(target_os = "macos", target_os = "linux"))]
                weights: tensors,
                #[cfg(target_os = "linux")]
                vulkan: None,
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = checkpoint_digest;
            Err(Error::Inference("native inference required".into()))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn infer_patch(&self, input: &SrgbRgb, cancel: &Cancellation) -> Result<Rgb32f> {
        cancel.check()?;
        let image = input.image();
        if image.width > TILE || image.height > TILE {
            return Err(Error::Input("patch exceeds 256 pixels".into()));
        }
        let run = |network: &network::Network, device: &candle_core::Device| -> candle_core::Result<_> {
            let data: Vec<f32> = image.data.iter().flatten().copied().collect();
            let tensor = candle_core::Tensor::from_slice(&data, (1, image.height, image.width, 3), device)?.permute((0, 3, 1, 2))?.contiguous()?;
            network.forward(&tensor)?.permute((0, 2, 3, 1))?.contiguous()?.flatten_all()?.to_vec1::<f32>()
        };
        #[cfg(target_os = "macos")]
        let result = if !self.fallback.load(Ordering::Relaxed)
            && let Some((network, device)) = &self.metal
        {
            match run(network, device) {
                Ok(output) => Ok(output),
                Err(error) => {
                    *self.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.to_string());
                    self.fallback.store(true, Ordering::Relaxed);
                    run(&self.network, &self.device)
                }
            }
        } else {
            run(&self.network, &self.device)
        };
        #[cfg(target_os = "linux")]
        let result = if !self.fallback.load(Ordering::Relaxed)
            && let Some(gpu) = &self.vulkan
        {
            match gpu.forward(image) {
                Ok(output) => Ok(output),
                Err(error) => {
                    *self.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(error.to_string());
                    self.fallback.store(true, Ordering::Relaxed);
                    run(&self.network, &self.device)
                }
            }
        } else {
            run(&self.network, &self.device)
        };
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let result = run(&self.network, &self.device);
        let result = result.map_err(|e| Error::Inference(e.to_string()))?;
        cancel.check()?;
        let output = Rgb32f { width: image.width, height: image.height, data: result.as_chunks::<3>().0.to_vec() };
        validate(&output)?;
        Ok(output)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn infer_tiles(
        &self,
        source: &Rgb32f,
        tiles: &[(usize, usize)],
        cancel: &Cancellation,
        consume: &mut impl FnMut(usize, Rgb32f) -> Result<()>,
    ) -> Result<()> {
        let prepare = |index: usize| -> Result<SrgbRgb> {
            cancel.check()?;
            let (sx, sy) = *tiles.get(index).ok_or_else(|| Error::Inference("invalid tile index".into()))?;
            let mut tile = Rgb32f::new(TILE, TILE);
            for y in 0..TILE.min(source.height - sy) {
                for x in 0..TILE.min(source.width - sx) {
                    let from = source.data.get((sy + y) * source.width + sx + x).ok_or_else(|| Error::Input("invalid source tile extent".into()))?;
                    let to = tile.data.get_mut(y * TILE + x).ok_or_else(|| Error::Input("invalid tile extent".into()))?;
                    *to = *from;
                }
            }
            Ok(SrgbRgb(tile))
        };
        #[cfg(target_os = "linux")]
        if let Some(gpu) = &self.vulkan
            && !self.fallback.load(Ordering::Relaxed)
        {
            for (batch, locations) in tiles.chunks(2).enumerate() {
                let inputs = (0..locations.len()).map(|offset| prepare(batch * 2 + offset)).collect::<Result<Vec<_>>>()?;
                let outputs = if !self.fallback.load(Ordering::Relaxed) {
                    let result = gpu.forward_tiles(&inputs.iter().map(SrgbRgb::image).collect::<Vec<_>>());
                    cancel.check()?;
                    match result {
                        Ok(raw) => Some(raw),
                        Err(reason) => {
                            *self.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason);
                            self.fallback.store(true, Ordering::Relaxed);
                            None
                        }
                    }
                } else {
                    None
                };
                if let Some(outputs) = outputs {
                    if outputs.len() != inputs.len() {
                        return Err(Error::Inference("invalid tile result count".into()));
                    }
                    for (offset, (raw, input)) in outputs.into_iter().zip(&inputs).enumerate() {
                        cancel.check()?;
                        consume(batch * 2 + offset, Rgb32f { width: input.0.width, height: input.0.height, data: raw.as_chunks::<3>().0.to_vec() })?;
                    }
                } else {
                    for (offset, input) in inputs.iter().enumerate() {
                        consume(batch * 2 + offset, self.infer_patch(input, cancel)?)?;
                    }
                }
            }
            return Ok(());
        }
        for index in 0..tiles.len() {
            consume(index, self.infer_patch(&prepare(index)?, cancel)?)?;
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Denoiser for NafNet {
    fn description(&self) -> ModelDescription {
        ModelDescription {
            id: MODEL_ID.into(),
            checkpoint_digest: self.checkpoint_digest.clone(),
            input_domain: "float-srgb-encoded-rgb",
            licence: "MIT + BasicSR Apache-2.0 (code); checkpoint redistribution requires verification",
        }
    }
    fn infer(&self, input: &SrgbRgb, region: Option<Region>, cancel: &Cancellation, progress: &(dyn Fn(usize, usize) + Sync)) -> Result<Inference> {
        cancel.check()?;
        let source = input.image();
        let r = region.unwrap_or(Region { x: 0, y: 0, width: source.width, height: source.height }).validate(source)?;
        let tiles: Vec<_> = (0..source.height)
            .step_by(TILE - OVERLAP)
            .flat_map(|y| (0..source.width).step_by(TILE - OVERLAP).map(move |x| (x, y)))
            .filter(|(x, y)| *x < r.x + r.width && *x + TILE > r.x && *y < r.y + r.height && *y + TILE > r.y)
            .collect();
        let mut image = Rgb32f::new(r.width, r.height);
        let mut sums = vec![0f32; r.width * r.height];
        let taper = |i: usize| ((i as f32 + 0.5) / OVERLAP as f32).min((TILE as f32 - 0.5 - i as f32) / OVERLAP as f32).min(1.0);
        self.infer_tiles(source, &tiles, cancel, &mut |done, output| {
            let (sx, sy) = *tiles.get(done).ok_or_else(|| Error::Inference("invalid tile index".into()))?;
            for y in sy.max(r.y)..(sy + TILE).min(r.y + r.height) {
                for x in sx.max(r.x)..(sx + TILE).min(r.x + r.width) {
                    let i = (y - r.y) * r.width + x - r.x;
                    let weight = taper(x - sx) * taper(y - sy);
                    let value = output.data.get((y - sy) * TILE + x - sx).ok_or_else(|| Error::Inference("invalid tile extent".into()))?;
                    let dst = image.data.get_mut(i).ok_or_else(|| Error::Inference("invalid destination extent".into()))?;
                    for (to, from) in dst.iter_mut().zip(value) {
                        *to += from * weight;
                    }
                    let sum = sums.get_mut(i).ok_or_else(|| Error::Inference("invalid weight extent".into()))?;
                    *sum += weight;
                }
            }
            progress(done + 1, tiles.len());
            Ok(())
        })?;
        for (pixel, sum) in image.data.iter_mut().zip(sums) {
            if sum <= 0.0 {
                return Err(Error::Inference("uncovered pixel".into()));
            }
            for v in pixel {
                *v /= sum;
            }
        }
        validate(&image)?;
        Ok(Inference {
            image,
            region: r,
            cpu_fallback: self.fallback.load(Ordering::Relaxed),
            backend: self.backend(),
            gpu_buffer_bytes: {
                #[cfg(target_os = "linux")]
                {
                    self.vulkan.as_ref().map_or(0, vulkan::Network::buffer_bytes)
                }
                #[cfg(not(target_os = "linux"))]
                {
                    0
                }
            },
            fallback_reason: self.fallback_reason.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        })
    }
}
