//! NAFNet SIDD width-32, ordinary adaptive global pooling (not NAFNetLocal/TLC).
//! Architecture: megvii-research/NAFNet, revision 2b4af71; see docs/denoise.md.
//! Copyright 2022 megvii-model (MIT); normalization follows BasicSR (Apache-2.0).
//! Modified by LightCraft contributors, 2026: Rust/Candle port and bounded tiled inference.
//! The Rust port is Apache-2.0; upstream notices are retained in ../NOTICE.
use candle_core::{D, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d, conv2d_no_bias};

type Result<T> = candle_core::Result<T>;

fn conv(input: usize, output: usize, kernel: usize, padding: usize, stride: usize, groups: usize, vb: VarBuilder<'_>) -> Result<Conv2d> {
    conv2d(input, output, kernel, Conv2dConfig { padding, stride, groups, ..Default::default() }, vb)
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}
impl Norm {
    fn new(c: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self { weight: vb.get(c, "weight")?.reshape((1, c, 1, 1))?, bias: vb.get(c, "bias")?.reshape((1, c, 1, 1))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let centered = x.broadcast_sub(&x.mean_keepdim(1)?)?;
        centered.broadcast_div(&(centered.sqr()?.mean_keepdim(1)? + 1e-6)?.sqrt()?)?.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}
fn gate(x: Tensor) -> Result<Tensor> {
    let half = x.dim(1)? / 2;
    x.narrow(1, 0, half)?.mul(&x.narrow(1, half, half)?)
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    c1: Conv2d,
    c2: Conv2d,
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
            c2: conv(c * 2, c * 2, 3, 1, 1, c * 2, vb.pp("conv2"))?,
            c3: conv(c, c, 1, 0, 1, 1, vb.pp("conv3"))?,
            c4: conv(c, c * 2, 1, 0, 1, 1, vb.pp("conv4"))?,
            c5: conv(c, c, 1, 0, 1, 1, vb.pp("conv5"))?,
            attention: conv(c, c, 1, 0, 1, 1, vb.pp("sca.1"))?,
            beta: vb.get((1, c, 1, 1), "beta")?,
            gamma: vb.get((1, c, 1, 1), "gamma")?,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = gate(self.c2.forward(&self.c1.forward(&self.norm1.forward(input)?)?)?)?;
        let attention = self.attention.forward(&x.mean_keepdim((2, 3))?)?;
        let y = (input + self.c3.forward(&x.broadcast_mul(&attention)?)?.broadcast_mul(&self.beta)?)?;
        let x = self.c5.forward(&gate(self.c4.forward(&self.norm2.forward(&y)?)?)?)?;
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
            let t = up.forward(&x)?;
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
