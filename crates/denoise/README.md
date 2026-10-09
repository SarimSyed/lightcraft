# lightcraft-denoise (L3)

UI-independent RGB-stage restoration. `Denoiser` accepts only validated float,
sRGB-encoded RGB via `SrgbRgb`; sensor-domain models need a separate typed contract.
The sole implementation is the authors' NAFNet SIDD width-32 network with ordinary
global pooling. Native CPU inference uses Candle 0.9.2; macOS can use Metal and
retry on CPU. The browser cannot run inference.

The CPU network uses fused, bounded F32 channel normalization and direct depthwise
3×3 stencils, plus zero-copy matrix products for its 1×1 convolutions. These avoid
Candle's per-channel convolution launches and intermediate normalization tensors.
The architecture, weights, global pooling and tiling remain unchanged. No additional
native library or GPU driver is needed on Linux.

`model_input` / `restore_working` implement the versioned Rec.2020 correction
adapter. `infer` uses full-source-anchored 256-pixel tiles, 64-pixel overlap and
normalized tapered blending. A region evaluates exactly the same contributing
tiles and produces the same values as cropping a full inference. Cancellation
is checked between tiles and before/after each network operation.

Normal API tests need no weights or network. Required reference validation:

```sh
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/reference-bundle \
  cargo test -p lightcraft-denoise --features reference-validation --test reference
```

This deliberately fails if its model or independent references are missing.
See [denoise documentation](../../docs/denoise.md) for provenance and delivery gates.
