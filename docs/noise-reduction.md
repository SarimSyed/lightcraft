# Noise reduction

LightCraft's Detail controls use classical filters, with the same parameters on CPU and GPU.
No model download is involved. Original files and saved settings are unchanged; photos with NR
enabled render with the improved colour filter, including existing saved edits.

## Controls and rendering

- Luminance NR self-guides log luminance. Detail controls its regularization; Contrast reduces
  that regularization by up to 75%, preserving more structure while NR remains active. Contrast
  is included in the shared denoised-stage cache key.
- Colour NR independently self-guides each component of `RGB/Y`. Detail sets regularization
  to `0.0025 + 0.0375 × (1 − detail/100)²`, as well as retaining its existing blend-strength
  contribution. Smoothness controls the window radius. After blending, nonnegative chromaticity
  is normalized before restoring the original luminance; invalid normalization retains the pixel.
- Coefficients use half resolution for windows of at least 1.5 px; output guidance remains at
  full resolution. CPU RGB blur passes evaluate the three scalar filters together without
  cross-channel covariance. GPU filters use scalar planes and never six-channel blur buffers.
  Larger subsampling steps lost too much colour-edge detail in the procedural regression.

## Validation

Public render tests compare output against independently clean, procedurally generated images:

- Luminance NR 80 halves flat-region noise variance and keeps mean brightness within 1%, both
  at Contrast 0 and 100. Contrast 100 retains more texture without bypassing NR.
- Colour NR 80 reduces smooth-region chroma error by at least 40%, retains approximately 95%
  of a strong equal-luminance colour step, and preserves luminance. Preview/export sizes and
  boundaries between coefficient samples are covered, with a 0.00001 numerical tolerance on
  edge-retention ratios. Higher Colour Detail preserves more edge detail.
- NR-off, black, constant-colour, near-black and singleton images remain finite and correct.
  A Contrast change invalidates cached previews. Real CLI tests verify lossless exports and
  preservation of the input file.
- GPU equivalence requires mean RGB difference below 0.5 LSB and maximum difference at most
  3 LSB. Hardware tests run on AMD Radeon RX 9060 XT / Vulkan; tests explicitly skip without
  an adapter. This is not verification of all drivers.

These tests establish synthetic quality criteria, not Lightroom parity or a measured Nikon noise
model. The D7100 sample is used for performance and native Fit/100% smoke checks; high-ISO
camera noise and comparison with Lightroom remain part of the render-fidelity gap.

## Performance, 2026-10-08

`cargo xtask bench corpus/raw/nef-nikon-d7100-lossless14.nef`, three runs per scenario, CPU
rendering of the same 6036 × 4020 CC0 raw on this workstation. CPU time sums work across threads;
wall time is elapsed time. Recorded results live in the gitignored `target/bench/history.jsonl`.

| Scenario | Before CPU time | Improved CPU time | Change |
|---|---:|---:|---:|
| 1920 × 1280 cold preview | 659.6 ms | 727.0 ms | +10.2% |
| 1152 × 768 cold draft | 192.0 ms | 209.0 ms | +8.9% |
| 1920 × 1280 NR drag | 590.5 ms | 609.0 ms | +3.1% |
| 1152 × 768 NR drag | 161.9 ms | 181.1 ms | +11.9% |
| 24 MP export render | 6914.9 ms | 7955.7 ms | +15.1% |

Full export render wall time was 735.8 → 871.3 ms. JPEG encoding is measured separately.
Every recorded CPU metric stays within the 20% regression gate relative to the original baseline.
