//! Linux NAFNet on the existing wgpu compute device. All intermediate tensors stay
//! on the GPU; only the input and final tile cross the CPU/GPU boundary.
use lightcraft_raster::Rgb32f;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use wgpu::util::DeviceExt;

type Result<T> = std::result::Result<T, String>;
#[derive(Clone)]
struct Image {
    buffer: Arc<wgpu::Buffer>,
    c: usize,
    h: usize,
    w: usize,
}
struct Weight {
    buffer: wgpu::Buffer,
    len: u32,
    bias: bool,
}
struct Arena {
    buffers: Vec<Arc<wgpu::Buffer>>,
    bytes: u64,
}
pub(crate) struct Network {
    pub name: String,
    unavailable: fn() -> Option<String>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    kernels: HashMap<&'static str, wgpu::ComputePipeline>,
    weights: HashMap<String, Weight>,
    dummy: wgpu::Buffer,
    unwritten: wgpu::Buffer,
    arena: Mutex<Arena>,
}

// Scopes are per-thread in wgpu. Always pop them, including after a caught panic.
fn guarded<T>(device: &wgpu::Device, run: impl FnOnce() -> Result<T>) -> Result<T> {
    let scopes =
        [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation].map(|filter| device.push_error_scope(filter));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|_| Err("GPU denoise worker failed".into()));
    let mut error = None;
    for scope in scopes.into_iter().rev() {
        if let Some(e) = pollster::block_on(scope.pop()) {
            error = Some(e.to_string());
        }
    }
    match error {
        Some(e) => Err(e),
        None => result,
    }
}
impl Network {
    pub fn new(
        tensors: &HashMap<String, candle_core::Tensor>,
        device: (wgpu::Device, wgpu::Queue, wgpu::AdapterInfo),
        unavailable: fn() -> Option<String>,
    ) -> Result<Self> {
        let (device, queue, info) = device;
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err("software GPU adapter skipped".into());
        }
        if let Some(reason) = unavailable() {
            return Err(reason);
        }
        guarded(&device, || {
            if device.limits().max_storage_buffer_binding_size < 64 * 256 * 256 * 4 {
                return Err("GPU storage limit is too small for NAFNet tiles".into());
            }
            let entries: Vec<_> = (0..5)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: binding != 4 },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("NAFNet"), entries: &entries });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NAFNet"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("NAFNet"),
                source: wgpu::ShaderSource::Wgsl(include_str!("nafnet.wgsl").into()),
            });
            let kernels = ["pointwise", "spatial", "depthwise", "norm", "pool", "map"]
                .into_iter()
                .map(|name| {
                    (
                        name,
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(name),
                            layout: Some(&pipeline_layout),
                            module: &shader,
                            entry_point: Some(name),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                    )
                })
                .collect();
            let mut weights = HashMap::new();
            // The CPU constructor has already verified every required tensor's shape.
            for (name, tensor) in tensors {
                if name.ends_with(".bias") {
                    continue;
                }
                let mut data = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
                let len = u32::try_from(data.len()).map_err(|_| "GPU weight count overflow")?;
                let bias = if let Some(base) = name.strip_suffix(".weight") {
                    if let Some(t) = tensors.get(&format!("{base}.bias")) {
                        data.extend(t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if data.len() as u64 * 4 > device.limits().max_storage_buffer_binding_size {
                    return Err("GPU weight buffer exceeds device limit".into());
                }
                weights.insert(name.clone(), Weight { buffer: upload(&device, &data), len, bias });
            }
            Ok(Self {
                unavailable,
                name: format!("{} ({:?})", info.name, info.backend),
                device: device.clone(),
                queue,
                layout,
                kernels,
                weights,
                dummy: upload(&device, &[0.0]),
                unwritten: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NAFNet unwritten sentinel"),
                    contents: bytemuck::cast_slice(&vec![f32::NAN; 3 * 256 * 256]),
                    usage: wgpu::BufferUsages::COPY_SRC,
                }),
                arena: Mutex::new(Arena { buffers: Vec::new(), bytes: 0 }),
            })
        })
    }
    /// Model + reusable activation buffers, excluding driver/command bookkeeping.
    pub fn buffer_bytes(&self) -> u64 {
        self.weights.values().map(|w| w.buffer.size()).sum::<u64>()
            + self.unwritten.size()
            + self.dummy.size()
            + self.arena.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }
    pub fn forward(&self, input: &Rgb32f) -> Result<Vec<f32>> {
        if let Some(reason) = (self.unavailable)() {
            return Err(reason);
        }
        guarded(&self.device, || self.run(input))
    }
    fn run(&self, input: &Rgb32f) -> Result<Vec<f32>> {
        let mut arena = self.arena.lock().unwrap_or_else(|e| e.into_inner());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("NAFNet tile") });
        let h = input.height.div_ceil(16) * 16;
        let w = input.width.div_ceil(16) * 16;
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return Err("invalid NAFNet GPU tile".into());
        }
        let original = self.image(&mut arena, 3, h, w)?;
        let mut data = vec![0.0f32; 3 * h * w];
        for (i, pixel) in input.data.iter().enumerate() {
            let y = i / input.width;
            let x = i % input.width;
            for (ch, value) in pixel.iter().enumerate() {
                *data.get_mut(ch * h * w + y * w + x).ok_or("GPU input extent")? = *value;
            }
        }
        self.queue.write_buffer(&original.buffer, 0, bytemuck::cast_slice(&data));
        let mut x = self.conv(&mut arena, &mut encoder, &original, "intro", 32, 3, 1, 1)?;
        let mut skips = Vec::new();
        for (i, count) in [2, 2, 4, 8].into_iter().enumerate() {
            x = self.blocks(&mut arena, &mut encoder, x, &format!("encoders.{i}"), count)?;
            skips.push(x.clone());
            x = self.conv(&mut arena, &mut encoder, &x, &format!("downs.{i}"), x.c * 2, 2, 2, 0)?;
        }
        x = self.blocks(&mut arena, &mut encoder, x, "middle_blks", 12)?;
        for (i, skip) in skips.into_iter().rev().enumerate() {
            let up = self.conv(&mut arena, &mut encoder, &x, &format!("ups.{i}.0"), x.c * 2, 1, 1, 0)?;
            let shuffled = self.map(&mut arena, &mut encoder, &up, None, None, 4, up.c / 4, up.h * 2, up.w * 2)?;
            x = self.map(&mut arena, &mut encoder, &shuffled, Some(&skip), None, 3, skip.c, skip.h, skip.w)?;
            x = self.blocks(&mut arena, &mut encoder, x, &format!("decoders.{i}"), 2)?;
        }
        let ending = self.conv(&mut arena, &mut encoder, &x, "ending", 3, 3, 1, 1)?;
        let output = self.image(&mut arena, 3, h, w)?;
        // Clear after earlier uses of recycled buffers: unwritten results must never
        // look like a successful black/undenoised image after a driver failure.
        encoder.copy_buffer_to_buffer(&self.unwritten, 0, &output.buffer, 0, (3 * h * w * 4) as u64);
        self.dispatch(
            &mut encoder,
            "map",
            [3, h as u32, w as u32, 3, 0, 0, 0, 3, 0, 0, 0, 0],
            &ending,
            Some(&original),
            None,
            &output,
            groups(3 * h * w),
        )?;
        let bytes = (3 * h * w * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NAFNet readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &readback, 0, bytes);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(60)) })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(std::time::Duration::from_secs(1)).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = readback.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
        let values: &[f32] = bytemuck::try_cast_slice(&mapped).map_err(|e| e.to_string())?;
        let mut result = Vec::with_capacity(input.width * input.height * 3);
        for y in 0..input.height {
            for x in 0..input.width {
                for ch in 0..3 {
                    let value = *values.get(ch * h * w + y * w + x).ok_or("GPU output extent")?;
                    if !value.is_finite() {
                        return Err("nonfinite GPU output".into());
                    }
                    result.push(value);
                }
            }
        }
        drop(mapped);
        readback.unmap();
        Ok(result)
    }
    fn image(&self, arena: &mut Arena, c: usize, h: usize, w: usize) -> Result<Image> {
        let bytes = c
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n > 0 && *n <= 64 * 256 * 256 * 4)
            .ok_or("NAFNet GPU allocation limit")? as u64;
        let buffer = if let Some(b) = arena.buffers.iter().filter(|b| Arc::strong_count(b) == 1 && b.size() >= bytes).min_by_key(|b| b.size()) {
            b.clone()
        } else {
            if arena.bytes.saturating_add(bytes) > 128 << 20 {
                return Err("NAFNet GPU workspace limit".into());
            }
            let b = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NAFNet activation"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            arena.bytes += bytes;
            arena.buffers.push(b.clone());
            b
        };
        Ok(Image { buffer, c, h, w })
    }
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &str,
        p: [u32; 12],
        a: &Image,
        b: Option<&Image>,
        weight: Option<&Weight>,
        out: &Image,
        groups: [u32; 3],
    ) -> Result<()> {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("NAFNet params"),
            contents: bytemuck::cast_slice(&p),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffers = [&params, &a.buffer, b.map_or(&self.dummy, |b| &b.buffer), weight.map_or(&self.dummy, |w| &w.buffer), &out.buffer];
        let entries: Vec<_> =
            buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("NAFNet"), layout: &self.layout, entries: &entries });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(kernel), timestamp_writes: None });
        pass.set_pipeline(self.kernels.get(kernel).ok_or("missing NAFNet kernel")?);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }
    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights.get(name).ok_or_else(|| format!("missing GPU weight {name}"))
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        arena: &mut Arena,
        encoder: &mut wgpu::CommandEncoder,
        x: &Image,
        name: &str,
        c: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Image> {
        let out = self.image(arena, c, (x.h + 2 * pad - k) / stride + 1, (x.w + 2 * pad - k) / stride + 1)?;
        let weight = self.weight(&format!("{name}.weight"))?;
        let p = [x.c as u32, x.h as u32, x.w as u32, c as u32, k as u32, stride as u32, pad as u32, 0, weight.len, u32::from(weight.bias), 0, 0];
        let (kernel, groups) = if k == 1 {
            ("pointwise", [(out.h * out.w).div_ceil(16) as u32, c.div_ceil(16) as u32, 1])
        } else {
            ("spatial", groups(c * out.h * out.w))
        };
        self.dispatch(encoder, kernel, p, x, None, Some(weight), &out, groups)?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)]
    fn map(
        &self,
        arena: &mut Arena,
        encoder: &mut wgpu::CommandEncoder,
        a: &Image,
        b: Option<&Image>,
        weight: Option<&Weight>,
        mode: u32,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Image> {
        let out = self.image(arena, c, h, w)?;
        self.dispatch(encoder, "map", [a.c as u32, h as u32, w as u32, c as u32, 0, 0, 0, mode, 0, 0, 0, 0], a, b, weight, &out, groups(c * h * w))?;
        Ok(out)
    }
    fn norm(&self, arena: &mut Arena, encoder: &mut wgpu::CommandEncoder, x: &Image, name: &str) -> Result<Image> {
        let out = self.image(arena, x.c, x.h, x.w)?;
        self.dispatch(
            encoder,
            "norm",
            [x.c as u32, x.h as u32, x.w as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            x,
            None,
            Some(self.weight(&format!("{name}.weight"))?),
            &out,
            groups(x.h * x.w),
        )?;
        Ok(out)
    }
    fn blocks(&self, arena: &mut Arena, encoder: &mut wgpu::CommandEncoder, mut input: Image, name: &str, count: usize) -> Result<Image> {
        for i in 0..count {
            let name = format!("{name}.{i}");
            let c = input.c;
            let h = input.h;
            let w = input.w;
            let norm = self.norm(arena, encoder, &input, &format!("{name}.norm1"))?;
            let expanded = self.conv(arena, encoder, &norm, &format!("{name}.conv1"), c * 2, 1, 1, 0)?;
            drop(norm);
            let depth = self.image(arena, c * 2, h, w)?;
            let weight = self.weight(&format!("{name}.conv2.weight"))?;
            self.dispatch(
                encoder,
                "depthwise",
                [c as u32 * 2, h as u32, w as u32, 0, 0, 0, 0, 0, weight.len, 0, 0, 0],
                &expanded,
                None,
                Some(weight),
                &depth,
                groups(c * 2 * h * w),
            )?;
            drop(expanded);
            let gated = self.map(arena, encoder, &depth, None, None, 0, c, h, w)?;
            drop(depth);
            let pooled = self.image(arena, c, 1, 1)?;
            self.dispatch(encoder, "pool", [c as u32, h as u32, w as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0], &gated, None, None, &pooled, [c as u32, 1, 1])?;
            let attention = self.conv(arena, encoder, &pooled, &format!("{name}.sca.1"), c, 1, 1, 0)?;
            drop(pooled);
            let scaled = self.map(arena, encoder, &gated, Some(&attention), None, 1, c, h, w)?;
            drop(gated);
            drop(attention);
            let reduced = self.conv(arena, encoder, &scaled, &format!("{name}.conv3"), c, 1, 1, 0)?;
            drop(scaled);
            let y = self.map(arena, encoder, &input, Some(&reduced), Some(self.weight(&format!("{name}.beta"))?), 2, c, h, w)?;
            drop(reduced);
            drop(input);
            let norm = self.norm(arena, encoder, &y, &format!("{name}.norm2"))?;
            let expanded = self.conv(arena, encoder, &norm, &format!("{name}.conv4"), c * 2, 1, 1, 0)?;
            drop(norm);
            let gated = self.map(arena, encoder, &expanded, None, None, 0, c, h, w)?;
            drop(expanded);
            let reduced = self.conv(arena, encoder, &gated, &format!("{name}.conv5"), c, 1, 1, 0)?;
            drop(gated);
            input = self.map(arena, encoder, &y, Some(&reduced), Some(self.weight(&format!("{name}.gamma"))?), 2, c, h, w)?;
        }
        Ok(input)
    }
}
fn upload(device: &wgpu::Device, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("NAFNet weights"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn groups(n: usize) -> [u32; 3] {
    let count = n.div_ceil(64);
    [count.min(32768) as u32, count.div_ceil(32768) as u32, 1]
}
