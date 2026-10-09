//! Linux NAFNet on the existing wgpu compute device. All intermediate tensors stay
//! on the GPU; only the input and final tile cross the CPU/GPU boundary.
use lightcraft_raster::Rgb32f;
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
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
    residual: u32,
}
struct Arena {
    buffers: Vec<Arc<wgpu::Buffer>>,
    bytes: u64,
    input: Vec<f32>,
    upload: wgpu::util::StagingBelt,
    readbacks: Vec<wgpu::Buffer>,
}
#[derive(PartialEq, Eq, Hash)]
struct BindingKey {
    params: [u32; 12],
    buffers: [wgpu::Buffer; 5],
}
struct PendingTile {
    buffer: wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    ready: std::sync::mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    bytes: u64,
    width: usize,
    height: usize,
    h: usize,
    w: usize,
}
impl Drop for PendingTile {
    fn drop(&mut self) {
        self.buffer.unmap();
    }
}
pub(crate) struct Network {
    pub name: String,
    allocated: AtomicU64,
    host_bytes: AtomicU64,
    unavailable: fn() -> Option<String>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    kernels: HashMap<&'static str, wgpu::ComputePipeline>,
    weights: HashMap<String, Weight>,
    dummy: wgpu::Buffer,
    unwritten: wgpu::Buffer,
    arena: Mutex<Arena>,
    bindings: Mutex<HashMap<BindingKey, wgpu::BindGroup>>,
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
            let entries: Vec<_> = (0..6)
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
            let kernels = ["pointwise", "pointwise_gate", "spatial", "depthwise_gate", "norm", "norm_channels", "pool", "map"]
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
                // Append learned residual scales to their convolution's weight buffer.
                let residual = if let Some(base) = name.strip_suffix(".weight").and_then(|base| {
                    base.strip_suffix(".conv3").map(|n| format!("{n}.beta")).or_else(|| base.strip_suffix(".conv5").map(|n| format!("{n}.gamma")))
                }) {
                    let offset = u32::try_from(data.len()).map_err(|_| "GPU residual offset overflow")?;
                    let tensor = tensors.get(&base).ok_or("missing residual scale")?;
                    data.extend(tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?);
                    offset
                } else {
                    0
                };
                if data.len() as u64 * 4 > device.limits().max_storage_buffer_binding_size {
                    return Err("GPU weight buffer exceeds device limit".into());
                }
                weights.insert(name.clone(), Weight { buffer: upload(&device, &data), len, bias, residual });
            }
            let allocated = weights.values().map(|w| w.buffer.size()).sum::<u64>() + (3 * 256 * 256 * 4) as u64 + 4;
            Ok(Self {
                allocated: AtomicU64::new(allocated),
                host_bytes: AtomicU64::new(0),
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
                arena: Mutex::new(Arena {
                    buffers: Vec::new(),
                    bytes: 0,
                    input: Vec::new(),
                    readbacks: Vec::new(),
                    upload: wgpu::util::StagingBelt::new(device.clone(), (3 * 256 * 256 * 4) as u64),
                }),
                bindings: Mutex::new(HashMap::new()),
            })
        })
    }
    /// Model + reusable activation buffers, excluding driver/command bookkeeping.
    pub fn buffer_bytes(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }
    pub fn host_bytes(&self) -> usize {
        self.host_bytes.load(Ordering::Relaxed) as usize
    }
    fn record_memory(&self, arena: &Arena) {
        // Upload chunks are reused; two submissions bound their pooled capacity.
        let bytes = self.weights.values().map(|w| w.buffer.size()).sum::<u64>()
            + self.unwritten.size()
            + self.dummy.size()
            + arena.bytes
            + arena.readbacks.iter().map(wgpu::Buffer::size).sum::<u64>()
            + arena.readbacks.len() as u64 * (3 * 256 * 256 * 4) as u64
            + self.bindings.lock().unwrap_or_else(|e| e.into_inner()).len() as u64 * 48;
        self.allocated.store(bytes, Ordering::Relaxed);
        self.host_bytes.store(arena.input.capacity() as u64 * 4, Ordering::Relaxed);
    }
    pub fn forward(&self, input: &Rgb32f) -> Result<Vec<f32>> {
        self.forward_tiles(&[input])?.pop().ok_or_else(|| "missing GPU tile output".into())
    }
    pub fn forward_tiles(&self, inputs: &[&Rgb32f]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err("GPU batch must contain one or two tiles".into());
        }
        if let Some(reason) = (self.unavailable)() {
            return Err(reason);
        }
        guarded(&self.device, || {
            let mut arena = self.arena.lock().unwrap_or_else(|e| e.into_inner());
            // Queue the next tile before reading the previous one. Queue ordering
            // safely reuses activations; readback slots remain independent.
            let pending = inputs.iter().enumerate().map(|(slot, input)| self.submit(&mut arena, input, slot)).collect::<Result<Vec<_>>>()?;
            let mut outputs = Vec::new();
            let mut error = None;
            for tile in pending {
                match self.read(&tile) {
                    Ok(output) => outputs.push(output),
                    Err(e) => {
                        error = Some(e);
                    }
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok(outputs),
            }
        })
    }
    fn submit(&self, arena: &mut Arena, input: &Rgb32f, slot: usize) -> Result<PendingTile> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("NAFNet tile") });
        let h = input.height.div_ceil(16) * 16;
        let w = input.width.div_ceil(16) * 16;
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return Err("invalid NAFNet GPU tile".into());
        }
        let original = self.image(arena, 3, h, w)?;
        arena.input.resize(3 * h * w, 0.0);
        arena.input.fill(0.0);
        for (i, pixel) in input.data.iter().enumerate() {
            let y = i / input.width;
            let x = i % input.width;
            for (ch, value) in pixel.iter().enumerate() {
                *arena.input.get_mut(ch * h * w + y * w + x).ok_or("GPU input extent")? = *value;
            }
        }
        let upload_bytes = wgpu::BufferSize::new((arena.input.len() * 4) as u64).ok_or("empty GPU upload")?;
        arena.upload.write_buffer(&mut encoder, &original.buffer, 0, upload_bytes).copy_from_slice(bytemuck::cast_slice(&arena.input));
        let mut x = self.conv(arena, &mut encoder, &original, "intro", 32, 3, 1, 1)?;
        let mut skips = Vec::new();
        for (i, count) in [2, 2, 4, 8].into_iter().enumerate() {
            x = self.blocks(arena, &mut encoder, x, &format!("encoders.{i}"), count)?;
            skips.push(x.clone());
            x = self.conv(arena, &mut encoder, &x, &format!("downs.{i}"), x.c * 2, 2, 2, 0)?;
        }
        x = self.blocks(arena, &mut encoder, x, "middle_blks", 12)?;
        for (i, skip) in skips.into_iter().rev().enumerate() {
            let up = self.conv(arena, &mut encoder, &x, &format!("ups.{i}.0"), x.c * 2, 1, 1, 0)?;
            let shuffled = self.map(arena, &mut encoder, &up, None, None, 4, up.c / 4, up.h * 2, up.w * 2)?;
            x = self.map(arena, &mut encoder, &shuffled, Some(&skip), None, 3, skip.c, skip.h, skip.w)?;
            x = self.blocks(arena, &mut encoder, x, &format!("decoders.{i}"), 2)?;
        }
        let ending = self.conv(arena, &mut encoder, &x, "ending", 3, 3, 1, 1)?;
        let output = self.image(arena, 3, h, w)?;
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
            None,
        )?;
        let bytes = (3 * h * w * 4) as u64;
        while arena.readbacks.len() <= slot {
            arena.readbacks.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NAFNet reusable readback"),
                size: (3 * 256 * 256 * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let readback = arena.readbacks.get(slot).ok_or("invalid GPU readback slot")?.clone();
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &readback, 0, bytes);
        arena.upload.finish_and_recall_on_submit(&encoder);
        self.record_memory(arena);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        readback.slice(..bytes).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        Ok(PendingTile { buffer: readback, submission, ready: rx, bytes, width: input.width, height: input.height, h, w })
    }
    fn read(&self, tile: &PendingTile) -> Result<Vec<f32>> {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(tile.submission.clone()), timeout: Some(std::time::Duration::from_secs(60)) })
            .map_err(|e| e.to_string())?;
        tile.ready.recv_timeout(std::time::Duration::from_secs(1)).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = tile.buffer.slice(..tile.bytes).get_mapped_range().map_err(|e| e.to_string())?;
        let values: &[f32] = bytemuck::try_cast_slice(&mapped).map_err(|e| e.to_string())?;
        let mut result = Vec::with_capacity(tile.width * tile.height * 3);
        for y in 0..tile.height {
            for x in 0..tile.width {
                for ch in 0..3 {
                    let value = *values.get(ch * tile.h * tile.w + y * tile.w + x).ok_or("GPU output extent")?;
                    if !value.is_finite() {
                        return Err("nonfinite GPU output".into());
                    }
                    result.push(value);
                }
            }
        }
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
        residual: Option<&Image>,
    ) -> Result<()> {
        let key = BindingKey {
            params: p,
            buffers: [
                (*a.buffer).clone(),
                b.map_or(&self.dummy, |b| &b.buffer).clone(),
                weight.map_or(&self.dummy, |w| &w.buffer).clone(),
                (*out.buffer).clone(),
                residual.map_or(&self.dummy, |r| &r.buffer).clone(),
            ],
        };
        let mut bindings = self.bindings.lock().unwrap_or_else(|e| e.into_inner());
        // Fixed tiles reuse the same bindings. Bound the cache for edge-tile shapes.
        if bindings.len() >= 2048 {
            bindings.clear();
        }
        let group = bindings
            .entry(key)
            .or_insert_with_key(|key| {
                let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NAFNet params"),
                    contents: bytemuck::cast_slice(&key.params),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buffers = [&params, &key.buffers[0], &key.buffers[1], &key.buffers[2], &key.buffers[3], &key.buffers[4]];
                let entries: Vec<_> =
                    buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("NAFNet"), layout: &self.layout, entries: &entries })
            })
            .clone();
        drop(bindings);
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
        self.conv_fused(arena, encoder, x, name, c, k, stride, pad, None, None, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn conv_fused(
        &self,
        arena: &mut Arena,
        encoder: &mut wgpu::CommandEncoder,
        x: &Image,
        name: &str,
        c: usize,
        k: usize,
        stride: usize,
        pad: usize,
        attention: Option<&Image>,
        residual: Option<&Image>,
        gate: bool,
    ) -> Result<Image> {
        let out = self.image(arena, c, (x.h + 2 * pad - k) / stride + 1, (x.w + 2 * pad - k) / stride + 1)?;
        let weight = self.weight(&format!("{name}.weight"))?;
        let p = [
            x.c as u32,
            x.h as u32,
            x.w as u32,
            c as u32,
            k as u32,
            stride as u32,
            pad as u32,
            0,
            weight.len,
            u32::from(weight.bias),
            u32::from(attention.is_some()),
            if residual.is_some() { weight.residual } else { 0 },
        ];
        let (kernel, groups) = if k == 1 || k == 2 {
            (if gate { "pointwise_gate" } else { "pointwise" }, [(out.h * out.w).div_ceil(32) as u32, c.div_ceil(16) as u32, 1])
        } else {
            ("spatial", groups(c * out.h * out.w))
        };
        self.dispatch(encoder, kernel, p, x, attention, Some(weight), &out, groups, residual)?;
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
        self.dispatch(
            encoder,
            "map",
            [a.c as u32, h as u32, w as u32, c as u32, 0, 0, 0, mode, 0, 0, 0, 0],
            a,
            b,
            weight,
            &out,
            groups(c * h * w),
            None,
        )?;
        Ok(out)
    }
    fn norm(&self, arena: &mut Arena, encoder: &mut wgpu::CommandEncoder, x: &Image, name: &str) -> Result<Image> {
        let out = self.image(arena, x.c, x.h, x.w)?;
        self.dispatch(
            encoder,
            if x.c >= 128 { "norm_channels" } else { "norm" },
            [x.c as u32, x.h as u32, x.w as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            x,
            None,
            Some(self.weight(&format!("{name}.weight"))?),
            &out,
            if x.c >= 128 { [(x.h * x.w).div_ceil(8) as u32, 1, 1] } else { groups(x.h * x.w) },
            None,
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
            let gated = self.image(arena, c, h, w)?;
            let weight = self.weight(&format!("{name}.conv2.weight"))?;
            self.dispatch(
                encoder,
                "depthwise_gate",
                [c as u32 * 2, h as u32, w as u32, c as u32, 0, 0, 0, 0, weight.len, 0, 0, 0],
                &expanded,
                None,
                Some(weight),
                &gated,
                groups(c * h * w),
                None,
            )?;
            drop(expanded);
            let pooled = self.image(arena, c, 1, 1)?;
            self.dispatch(
                encoder,
                "pool",
                [c as u32, h as u32, w as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &gated,
                None,
                None,
                &pooled,
                [c as u32, 1, 1],
                None,
            )?;
            let attention = self.conv(arena, encoder, &pooled, &format!("{name}.sca.1"), c, 1, 1, 0)?;
            drop(pooled);
            let y = self.conv_fused(arena, encoder, &gated, &format!("{name}.conv3"), c, 1, 1, 0, Some(&attention), Some(&input), false)?;
            drop(gated);
            drop(attention);
            drop(input);
            let norm = self.norm(arena, encoder, &y, &format!("{name}.norm2"))?;
            let gated = self.conv_fused(arena, encoder, &norm, &format!("{name}.conv4"), c, 1, 1, 0, None, None, true)?;
            drop(norm);
            input = self.conv_fused(arena, encoder, &gated, &format!("{name}.conv5"), c, 1, 1, 0, None, Some(&y), false)?;
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
