//! NAFNet SIDD width-32, ordinary adaptive global pooling (not NAFNetLocal/TLC).
//! Architecture: megvii-research/NAFNet, revision 2b4af71; see docs/denoise.md.
//! Copyright 2022 megvii-model (MIT); normalization follows BasicSR (Apache-2.0).
//! Modified by LightCraft contributors, 2026: Rust/Candle port and bounded tiled inference.
//! The Rust port is Apache-2.0; upstream notices are retained in ../NOTICE.
use candle_core::{CpuStorage, CustomOp1, D, Layout, Shape, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d, conv2d_no_bias};

type Result<T> = candle_core::Result<T>;

fn conv(input: usize, output: usize, kernel: usize, padding: usize, stride: usize, groups: usize, vb: VarBuilder<'_>) -> Result<Conv2d> {
    conv2d(input, output, kernel, Conv2dConfig { padding, stride, groups, ..Default::default() }, vb)
}

// A 1x1 convolution is a matrix product. Candle's CPU convolution copies the entire
// input first; these batch-one NCHW activations can be reshaped without that copy.
fn pointwise(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return conv.forward(x);
    }
    let (_, c, h, w) = x.dims4()?;
    let (out, _, _, _) = conv.weight().dims4()?;
    let output = conv.weight().reshape((out, c))?.matmul(&x.reshape((c, h * w))?)?.reshape((1, out, h, w))?;
    match conv.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, out, 1, 1))?),
        None => Ok(output),
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
    cpu: Option<CpuNorm>,
}
impl Norm {
    fn new(c: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get(c, "weight")?;
        let bias = vb.get(c, "bias")?;
        let cpu = if weight.device().is_cpu() { Some(CpuNorm { weight: weight.to_vec1::<f32>()?, bias: bias.to_vec1::<f32>()? }) } else { None };
        Ok(Self { weight: weight.reshape((1, c, 1, 1))?, bias: bias.reshape((1, c, 1, 1))?, cpu })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(cpu) = &self.cpu {
            return x.contiguous()?.apply_op1_no_bwd(cpu);
        }
        let centered = x.broadcast_sub(&x.mean_keepdim(1)?)?;
        centered.broadcast_div(&(centered.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?)?.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// Channel normalization in three contiguous passes, rather than allocating a tensor
/// for each subtraction, square, reduction, division and affine operation.
struct CpuNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl CustomOp1 for CpuNorm {
    fn name(&self) -> &'static str {
        "nafnet-channel-norm"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let c = self.weight.len();
        if c != self.bias.len() {
            candle_core::bail!("invalid NAFNet normalization weights");
        }
        let (h, w, input) = cpu_input(storage, layout, c)?;
        let spatial = h * w;
        let mut mean = vec![0.0f32; spatial];
        for plane in input.chunks_exact(spatial) {
            for (mean, value) in mean.iter_mut().zip(plane) {
                *mean += value;
            }
        }
        for value in &mut mean {
            *value /= c as f32;
        }
        let mut scale = vec![0.0f32; spatial];
        for plane in input.chunks_exact(spatial) {
            for ((variance, mean), value) in scale.iter_mut().zip(&mean).zip(plane) {
                *variance += (value - mean).powi(2);
            }
        }
        for variance in &mut scale {
            *variance = (*variance / c as f32 + 1e-6).sqrt();
        }
        let mut output = vec![0.0f32; input.len()];
        for ((source, dest), (weight, bias)) in
            input.chunks_exact(spatial).zip(output.chunks_exact_mut(spatial)).zip(self.weight.iter().zip(&self.bias))
        {
            for (((to, from), mean), scale) in dest.iter_mut().zip(source).zip(&mean).zip(&scale) {
                *to = (from - mean) / scale * weight + bias;
            }
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

fn cpu_input<'a>(storage: &'a CpuStorage, layout: &Layout, channels: usize) -> Result<(usize, usize, &'a [f32])> {
    let (b, c, h, w) = layout.shape().dims4()?;
    if b != 1 || c == 0 || c != channels || h == 0 || w == 0 || h > 256 || w > 256 || !layout.is_contiguous() {
        candle_core::bail!("invalid NAFNet CPU layout");
    }
    let count = c
        .checked_mul(h)
        .and_then(|n| n.checked_mul(w))
        .filter(|n| *n <= 64 * 256 * 256)
        .ok_or_else(|| candle_core::Error::Msg("NAFNet CPU allocation limit".into()))?;
    let end = layout.start_offset().checked_add(count).ok_or_else(|| candle_core::Error::Msg("NAFNet CPU offset overflow".into()))?;
    let CpuStorage::F32(data) = storage else { candle_core::bail!("NAFNet CPU operation requires F32") };
    let input = data.get(layout.start_offset()..end).ok_or_else(|| candle_core::Error::Msg("invalid NAFNet CPU input extent".into()))?;
    Ok((h, w, input))
}
fn gate(x: Tensor) -> Result<Tensor> {
    let half = x.dim(1)? / 2;
    x.narrow(1, 0, half)?.mul(&x.narrow(1, half, half)?)
}

struct Depthwise {
    convolution: Conv2d,
    cpu: Option<CpuDepthwise>,
}
impl Depthwise {
    fn new(c: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let convolution = conv(c, c, 3, 1, 1, c, vb)?;
        let cpu = if convolution.weight().device().is_cpu() {
            let weights = convolution.weight().flatten_all()?.to_vec1::<f32>()?;
            let kernels = weights.as_chunks::<9>().0.to_vec();
            let bias = convolution.bias().ok_or_else(|| candle_core::Error::Msg("missing depthwise bias".into()))?.to_vec1::<f32>()?;
            Some(CpuDepthwise { kernels, bias })
        } else {
            None
        };
        Ok(Self { convolution, cpu })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.cpu {
            Some(cpu) => x.contiguous()?.apply_op1_no_bwd(cpu),
            None => self.convolution.forward(x),
        }
    }
}

/// Candle's grouped CPU convolution splits every channel into a separate convolution.
/// NAFNet needs only a 3x3, stride-one, zero-padded stencil: accumulate contiguous rows
/// directly, without im2col matrices, channel concatenation or tiny GEMM launches.
struct CpuDepthwise {
    kernels: Vec<[f32; 9]>,
    bias: Vec<f32>,
}
impl CustomOp1 for CpuDepthwise {
    fn name(&self) -> &'static str {
        "nafnet-depthwise3x3"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if self.kernels.len() != self.bias.len() {
            candle_core::bail!("invalid NAFNet depthwise weights");
        }
        let (h, w, input) = cpu_input(storage, layout, self.kernels.len())?;
        let spatial = h * w;
        let mut output = vec![0.0f32; input.len()];
        for ((source, dest), (kernel, bias)) in
            input.chunks_exact(spatial).zip(output.chunks_exact_mut(spatial)).zip(self.kernels.iter().zip(&self.bias))
        {
            for (i, weight) in kernel.iter().enumerate() {
                let (ky, kx) = (i / 3, i % 3);
                let (sy, sx) = (ky.saturating_sub(1), kx.saturating_sub(1));
                let (dy, dx) = (1usize.saturating_sub(ky), 1usize.saturating_sub(kx));
                let (rows, columns) = (h.saturating_sub(ky.abs_diff(1)), w.saturating_sub(kx.abs_diff(1)));
                for y in 0..rows {
                    let (from, to) = ((sy + y) * w + sx, (dy + y) * w + dx);
                    let source = source.get(from..from + columns).ok_or_else(|| candle_core::Error::Msg("invalid depthwise row".into()))?;
                    let dest = dest.get_mut(to..to + columns).ok_or_else(|| candle_core::Error::Msg("invalid depthwise output row".into()))?;
                    for (to, from) in dest.iter_mut().zip(source) {
                        *to += from * weight;
                    }
                }
            }
            for value in dest {
                *value += bias;
            }
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    c1: Conv2d,
    c2: Depthwise,
    c3: Conv2d,
    c4: Conv2d,
    c5: Conv2d,
    attention: Conv2d,
    beta: Tensor,
    gamma: Tensor,
}
impl Block {
    fn new(c: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm1: Norm::new(c, vb.pp("norm1"))?,
            norm2: Norm::new(c, vb.pp("norm2"))?,
            c1: conv(c, c * 2, 1, 0, 1, 1, vb.pp("conv1"))?,
            c2: Depthwise::new(c * 2, vb.pp("conv2"))?,
            c3: conv(c, c, 1, 0, 1, 1, vb.pp("conv3"))?,
            c4: conv(c, c * 2, 1, 0, 1, 1, vb.pp("conv4"))?,
            c5: conv(c, c, 1, 0, 1, 1, vb.pp("conv5"))?,
            attention: conv(c, c, 1, 0, 1, 1, vb.pp("sca.1"))?,
            beta: vb.get((1, c, 1, 1), "beta")?,
            gamma: vb.get((1, c, 1, 1), "gamma")?,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = gate(self.c2.forward(&pointwise(&self.c1, &self.norm1.forward(input)?)?)?)?;
        let attention = self.attention.forward(&x.mean_keepdim((2, 3))?)?;
        let y = (input + pointwise(&self.c3, &x.broadcast_mul(&attention)?)?.broadcast_mul(&self.beta)?)?;
        let x = pointwise(&self.c5, &gate(pointwise(&self.c4, &self.norm2.forward(&y)?)?)?)?;
        y + x.broadcast_mul(&self.gamma)?
    }
}
fn blocks(c: usize, count: usize, vb: VarBuilder<'_>) -> Result<Vec<Block>> {
    (0..count).map(|i| Block::new(c, vb.pp(i))).collect()
}
fn forward_blocks(blocks: &[Block], mut x: Tensor) -> Result<Tensor> {
    for block in blocks {
        x = block.forward(&x)?;
    }
    Ok(x)
}

pub(crate) struct Network {
    intro: Conv2d,
    ending: Conv2d,
    encoders: Vec<Vec<Block>>,
    downs: Vec<Conv2d>,
    middle: Vec<Block>,
    decoders: Vec<Vec<Block>>,
    ups: Vec<Conv2d>,
}
impl Network {
    pub(crate) fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let mut encoders = Vec::new();
        let mut downs = Vec::new();
        let mut c = 32;
        for (i, n) in [2, 2, 4, 8].into_iter().enumerate() {
            encoders.push(blocks(c, n, vb.pp(format!("encoders.{i}")))?);
            downs.push(conv(c, c * 2, 2, 0, 2, 1, vb.pp(format!("downs.{i}")))?);
            c *= 2;
        }
        let middle = blocks(c, 12, vb.pp("middle_blks"))?;
        let mut decoders = Vec::new();
        let mut ups = Vec::new();
        for i in 0..4 {
            ups.push(conv2d_no_bias(c, c * 2, 1, Default::default(), vb.pp(format!("ups.{i}.0")))?);
            c /= 2;
            decoders.push(blocks(c, 2, vb.pp(format!("decoders.{i}")))?);
        }
        Ok(Self {
            intro: conv(3, 32, 3, 1, 1, 1, vb.pp("intro"))?,
            ending: conv(32, 3, 3, 1, 1, 1, vb.pp("ending"))?,
            encoders,
            downs,
            middle,
            decoders,
            ups,
        })
    }
    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = input.dims4()?;
        let input = input.pad_with_zeros(D::Minus1, 0, (16 - w % 16) % 16)?.pad_with_zeros(D::Minus2, 0, (16 - h % 16) % 16)?;
        let mut x = self.intro.forward(&input)?;
        let mut skips = Vec::new();
        for (encoder, down) in self.encoders.iter().zip(&self.downs) {
            x = forward_blocks(encoder, x)?;
            skips.push(x.clone());
            x = down.forward(&convolution_layout(&x)?)?;
        }
        x = forward_blocks(&self.middle, x)?;
        for ((decoder, up), skip) in self.decoders.iter().zip(&self.ups).zip(skips.iter().rev()) {
            let t = pointwise(up, &x)?;
            let (b, c, h, w) = t.dims4()?;
            let shuffled = t.reshape((b, c / 4, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.contiguous()?.reshape((b, c / 4, h * 2, w * 2))?;
            x = forward_blocks(decoder, (shuffled + skip)?)?;
        }
        (self.ending.forward(&convolution_layout(&x)?)? + input)?.narrow(2, 0, h)?.narrow(3, 0, w)
    }
}

// Candle 0.9.2's CPU convolution shortcut can confuse NCHW with NHWC when
// C == H == W. Make the physical layout unambiguous before spatial convolutions.
fn convolution_layout(x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 1))?.contiguous()?.permute((0, 3, 1, 2))
}
