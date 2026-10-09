# RGB-stage AI Denoise

NAFNet SIDD width-32 runs locally in native LightCraft, CLI and MCP. It consumes
full-resolution, EXIF-oriented decoded RGB before user geometry, conventional
noise reduction, sharpening, tone and grain. Supported RAWs are demosaiced first:
this is **RGB-stage AI denoise**, not Lightroom sensor-level RAW-denoise parity.
Unsupported RAW variants that only expose an embedded JPEG are rejected.

Detail → AI Denoise… and Photo → AI Denoise… open a focused 100% crop preview.
Hold Before or Space to compare. Amount (default 50) blends the same correction
without re-running inference or changing history. Apply processes the full image
and saves one undoable edit after its result has been written successfully.
Cancel, failures and stale completions do not change the saved edit.

## Model and delivery status

The Rust implementation and local/offline workflow are available. **Public model
download is release-blocked**: publishing on this fork requires working GitHub
authentication, a published asset, and documented redistribution permission for
the externally hosted author checkpoint. The upstream repository supplies an MIT
code licence and BasicSR Apache-2.0 notices; an explicit standalone checkpoint
permission has not yet been established. The application explains this and does
not pretend that a non-working URL is an available download.

The prepared release target is
`https://github.com/SarimSyed/lightcraft/releases/download/nafnet-sidd-width32-v1/nafnet-sidd-width32-v1.safetensors`.
Do not enable `REDISTRIBUTION_VERIFIED` until the actual asset has been downloaded
through `lightcraft-fetch`, its checksum verified, and permission evidence
recorded here. Include the upstream notices and provenance with the release.
`lightcraft-fetch` supplies progress, cancellation, resume, SHA-256 verification
and atomic installation. No weights or large reference fixtures are committed.

## Verified offline installation and provenance

Use only the author's **SIDD width-32** checkpoint. Do not substitute GoPro or
width-64 weights. Development-only conversion requires PyTorch, NumPy and
safetensors; none is a product dependency.

| Item | Pinned value |
|---|---|
| Upstream | https://github.com/megvii-research/NAFNet |
| Revision | `2b4af71ebe098a92a75910c233a3965a3e93ede4` |
| Author checkpoint | https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view |
| Original SHA-256 | `89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4` |
| Converted file | `nafnet-sidd-width32-v1.safetensors` |
| Bundle SHA-256 | `c6ae62717e6a8388e376302a592ada0d84333836fef32c25c25a130d72d6159a` |
| Size | 116,701,252 bytes (116.7 MB) |
| Network | width 32; encoder 2/2/4/8; middle 12; decoder 2/2/2/2 |
| Pooling | ordinary adaptive global average pooling, not NAFNetLocal/TLC |
| Runtime | F32; Candle 0.9.2 CPU/macOS Metal; Linux wgpu Vulkan with CPU fallback |

Fetch `NAFNet_arch.py` and `arch_util.py` at the revision above, then:

```sh
python tools/nafnet_reference.py original.pth NAFNet_arch.py arch_util.py bundle
sha256sum bundle/nafnet-sidd-width32-v1.safetensors
LIGHTCRAFT_NAFNET_DIR=/absolute/path/to/bundle cargo run -p lightcraft
```

The tool checks all three source hashes, extracts `params` without changing names
or float values, sorts tensor names and JSON keys deterministically, and records
conversion metadata. It also runs the pinned author's architecture to generate
independent padding and overlap reference tensors. A manually installed bundle
is checked against the pinned digest before inference. The normal persistent
model folder is `<LightCraft config>/models/nafnet`, shown by `enhance.model.status`.
The upstream notices are retained in `crates/denoise/NOTICE`.

## Commands and persistence

All frontends dispatch the same engine commands. Native UI calls are asynchronous;
CLI/MCP callers wait by default, or explicitly use `wait: true/false`.

| Command | Parameters / behavior |
|---|---|
| `enhance.denoise.preview` | `photo?`, `amount?` (50), `region?: {x,y,width,height}` in EXIF-oriented source pixels, `wait?`; no edit |
| `enhance.denoise.apply` | `photo?`, `amount?` (50), `wait?`; full-resolution result, one undo step |
| `enhance.denoise.status` | completion/progress/error/backend/CPU fallback and reason |
| `enhance.denoise.cancel` | cancel current inference |
| `enhance.model.status` | folder, exact size/digest, licence and installation/download state |
| `enhance.model.download` | explicit download; requires the release gate above |
| `enhance.model.cancel` | cancel installation |

Existing `Enhance.denoise` stores amount 0–100. An optional serialized model
selection records ID, checkpoint and processing revision. Legacy settings with
no model are inactive. Amount zero uses the baseline source exactly. Copy/paste,
presets, reset, history and versions use normal develop settings, never another
photo's result pixels.

Full-strength results live in `<library>/enhancements` as lossless F32 RGB TIFFs
and JSON manifests; temporary sessions use private session scratch. The TIFF is
explicitly **linear Rec.2020/D65**, read without guessed ICC/transfer conversion.
Atomic TIFF and manifest writes precede committing the adjustment. Keys include
original content, decoder/calibration revision, checkpoint and processing
revision. Exposure, crop and amount are excluded. Original source files are never
written. Render jobs resolve enhanced sources centrally for thumbnails, loupe,
comparison, variants and exports.

Missing/damaged results regenerate when verified weights are available. Otherwise
render/export returns an actionable error. Cached results work without weights.
Library changes, source replacement, deletion and conflicting edits reject stale
worker completions. A per-session gate prevents concurrent cache regeneration.

## Colour, tiling and limits

The versioned adapter transforms linear Rec.2020 to linear sRGB, clamps only the
model input, and applies the sRGB transfer function. The prediction's linear-light
correction relative to that clamped input is transformed back to Rec.2020 and
added to the original unclamped source. This retains out-of-range components;
photographic highlight/colour quality remains a separate validation gate.

Tiles are fixed 256 × 256 with 64-pixel overlap (stride 192), anchored at pixel
zero of the full source. Zero padding and normalized separable linear tapers are
deterministic. Crop previews compute all contributing full-grid tiles. Checkpoint
files are capped at 128 MiB, headers at 1 MiB, tensor ranks/shapes/offsets and
finite weights are checked. Sources are bounded at 100 megapixels, 32768 pixels
per edge and 512 MiB encoded data. Worker panics become errors through the existing
guard. CPU throughput on large images is a practical limitation.

### CPU execution

On Linux (and other CPU backends), NAFNet uses bounded pure-Rust fused channel
normalization and direct 3×3 depthwise stencils. Its 1×1 convolutions reshape the
activation into Candle matrix products without an extra input copy. This replaces
per-channel convolution launches and intermediate normalization tensors, while
retaining the same F32 weights, epsilon, global pooling, padding and tile order.
Metal retains the existing tensor path. Linux uses the Vulkan path below when the existing GPU preference is enabled.

The denoiser crate is optimized at level 3 in development builds as well as release
builds, so the fused loops receive compiler vectorization during native editing.
No additional product dependency or machine-specific CPU instruction requirement
is introduced. Existing cached corrections remain compatible; Amount still reuses
them without inference.

An opt-in public API latency test takes an explicit hardware-specific budget:

```sh
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle \
LIGHTCRAFT_NAFNET_TILE_BUDGET_MS=500 \
  cargo test -p lightcraft-denoise --features reference-validation --test reference \
    cpu_tile_inference_meets_explicit_latency_budget -- --ignored --nocapture --test-threads=1
```

Run timing tests on an otherwise idle machine. Normal CI has no wall-clock gate;
the required real-checkpoint job checks independent PyTorch outputs, including
a 1×1 input whose padded network bottleneck is also 1×1.

## Validation and remaining scope

Normal CI is network-independent. `.github/workflows/denoise-validation.yml` adds
a real-checkpoint job intended as a required branch-protection check that prefetches by SHA-256, generates independent
PyTorch references, and fails if its required files are absent. Numerical error
must stay below 1e-4. The procedural scene requires at least 80% MSE reduction and
mean brightness error below 0.005, plus exact crop/full overlap consistency.
Engine tests cover preview/no commit, one-step Apply/undo/redo, unchanged originals
(including sources whose paths collide with cache artifacts),
reopening without weights, amount reuse/zero, and missing-result export failures.
Headless UI tests exercise first use, hold-Space Before, temporary Amount, Apply/undo, native headless preview pixels and keyboard cancellation. Engine tests also cover pasted destination settings, conflicting edits, deleted photos, source replacement and library switches.

Browser inference, batch processing, automatic noise estimation, sensor-level
RAW models, Restormer and PMRID are deferred. The small `Denoiser` contract permits
other RGB models; a sensor model needs a distinct typed input and adapter.
macOS Metal execution, broad high-ISO RAW quality, detailed seam/artifact inspection
and platform-specific performance must be reported from actual validation, not
inferred from reference agreement or the presence of a button.

### Measured local validation (2026-10-08)

Rust inference matched the pinned author's PyTorch outputs with maximum absolute
error **2.38e-7** for non-multiple-of-16 padding and **1.07e-5** for overlapping
tiles. Procedural noise MSE fell from **0.0007040 to 0.00006050** (91.4%); mean
brightness bias was 0.000519. Cancellation after a completed tile was exercised.

Photographic validation used two CC0 samples, with no binaries committed:

- Nikon D7100 ISO 200 lossless NEF (6036 × 4020), already pinned in
  `docs/camera-preview-colour.md`: 512 × 320 source crop, 12 contributing tiles,
  19.07 s, process peak RSS 1,185,056 KiB (full decode plus crop inference).
- Canon PowerShot SX100 IS ISO 800 CHDK DNG (3264 × 2448),
  [raw.pixls.us sample 3823](https://raw.pixls.us/getfile.php/3823/nice/Canon%20-%20PowerShot%20SX100%20IS%20-%2010bit%2010bit%20CHDK%20ver.%201.4.1%20%284:3%29.DNG),
  SHA-256 `5c5e2d15b81778d1b48b44c53f4f0b454e4bdde5d683532828328a7842508400`:
  512 × 320 source crop, 12 tiles, 17.32 s, peak RSS 642,644 KiB. The crop
  contained 5,405 original out-of-range channel samples; output stayed finite.
  Mean linear correction was -0.00127. Full Apply, reusing the decoded original,
  took **359.62 s**; total process peak RSS **713,796 KiB**. Retained enhancement
  originals/results/previews occupied **195,698,688 bytes**. Cached Amount
  50/100/50 renders at 1024 pixels took **50.0 / 57.3 / 70.9 ms**; baseline
  Amount zero used its separate normal decode path (581 ms).

These are single runs on a Ryzen 5 9600X, Linux, CPU, development build; other
workspace validation ran concurrently. They are observations, not release
performance budgets. CPU full-image processing is slow. Native control inspection
and headless screenshots checked installation, progress, 100% preview, Before,
Amount, Apply and English/Portuguese/Japanese layouts. A native-headless preview
texture regression was found by looking at the screenshots and fixed test-first.

The ISO 800 crop visibly loses noise while retaining the main edges, with stronger
smoothing at Amount 100 and residual noise in saturated red/highlight areas. The
unclamped-source correction adapter intentionally retains those out-of-range
components. No obvious grid seams appeared in the inspected crop; this limited
inspection does not prove artifact-free output for every scene or camera. There
is no paired clean ground truth for these RAWs, and no Lightroom comparison.
Broader high-ISO, highlight, colour and fine-detail validation remains open.

### Linux CPU optimization (2026-10-09)

Matched runs of the previous and optimized implementations used the same pinned
Canon ISO 800 DNG above (3264×2448, 221 full-image tiles), on the same Ryzen 5 9600X,
Linux, native development builds, default CPU thread settings. The full runs were
sequential with no other validation build running. These remain local observations,
not a universal latency guarantee or a Linux GPU benchmark.

| Public operation / process measurement | Previous | Optimized |
|---|---:|---:|
| 512×320 source-crop preview, including decode (12 tiles) | 16.93 s | 4.50 s |
| Full Apply, decoded original already retained | 303.47 s | 75.19 s |
| Peak process RSS, decode + preview + Apply + cached renders | 781,548 KiB | 731,092 KiB |
| Cached Amount 50 / 100 / 50, 1024px renders | 40.3 / 41.0 / 40.1 ms | 40.1 / 38.6 / 39.2 ms |

Full Apply is about 4.0× faster; retained enhancement memory remains 195,698,688
bytes. Three warmed public 256px tile calls fell from 1,406–1,426 ms to 370–374 ms,
passing the explicit 500 ms local budget. The budget test is opt-in, not normal CI.
PyTorch maximum errors were 5.96e-8 (1×1 input), 1.19e-7 (padding) and 1.052e-5
(overlapping tiles). Procedural denoising MSE and brightness requirements still pass;
crop/full inference remains exactly equal and cancellation still returns an error.

The saved 1024px full renders differ from the previous implementation by at most
one 8-bit level (57 changed channel samples at Amount 50, 42 at Amount 100).
Amount zero and Before are pixel-identical. The photographic crop remains finite,
including its 5,405 original out-of-range channel samples. This preserves the
existing result's quality; it does not close the broader photographic validation
gaps. Large 24 MP photos still require hundreds of tiles.

Real-checkpoint engine tests and headless UI tests pass, covering safe Apply,
undo/cache reuse, stale results, cancellation and Before. An English/Japanese
headless dialog screenshot review completed. The current native control probe
timed out before inspection; native window responsiveness remains unverified by
this optimization run.

Reproduce photographic crops and optional full processing with development tools:

```sh
cargo run -p lightcraft-engine --example denoise_validate -- \
  corpus/raw/dng-canon-sx100-high-iso.dng /path/to/model /tmp/denoise-review full
```

The example uses public engine commands, saves Before/Amount previews and reports
elapsed time and retained cache memory. Use `/usr/bin/time -v` on the built
example binary for process peak RSS; timing `cargo` also counts compilation.


### Linux Vulkan inference (2026-10-09)

Native Linux AI denoise now shares the existing `lightcraft-gpu` compute device,
supplied by the engine to the independent denoiser crate.
The GPU preference and `LIGHTCRAFT_GPU=0` / `LIGHTCRAFT_GPU_BACKEND=off` apply to
inference too. No new runtime or GPU driver dependency is added: wgpu/WGSL uses
the system Vulkan driver. Software adapters are rejected. Windows remains on
CPU; macOS retains Candle Metal. Browser inference remains deferred.

All convolutions, channel normalization, SimpleGate, global pooling, attention,
residual scales and pixel shuffle run in F32 on the GPU. One model and its bounded
workspace stay resident per engine session, including between preview and Apply.
Weight-file identity and GPU availability changes reload it. Activations are reused
between layers and tiles.
Only input/final tiles cross the CPU/GPU boundary. Allocation/workgroup limits,
error scopes, a bounded readback wait and an unwritten-output sentinel prevent
invalid GPU results from being accepted. A recoverable GPU failure retries on
CPU and reports the reason. Cancellation is checked between tiles. The model,
processing revision and cached corrections remain compatible.

The dialog displays the processing device after preview. The shared
`enhance.denoise.status` command reports `backend`, `cpu_fallback`,
`fallback_reason` and `gpu_buffer_bytes` (Linux accounting, zero where unreported); `backend: "cache"` means an existing
result was reused. GPU buffer bytes count model/activation/sentinel/parameter/readback
buffers plus the bounded upload pool, excluding driver and command bookkeeping,
and are zero for a cached result. `library.memory` and `ui.inspect` account for
the resident model/workspace even after an operation finishes; CPU weights are
in `enhancements`, device buffers in `gpu.allocated`. Inspection does not wait
on the inference workspace lock.

On this machine's Radeon RX 9060 XT / RADV Vulkan, the independent PyTorch
reference maximum errors were 4.47e-8 (1×1), 1.79e-7 (padding) and 2.38e-7
(overlapping tiles), below the required 1e-4. Crop/full pixels matched exactly;
between-tile cancellation and reported CPU retry after GPU disable passed.
Hardware tests explicitly require Vulkan, so CPU fallback cannot pass them:

```sh
LIGHTCRAFT_GPU_BACKEND=vulkan LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle \
  cargo test -p lightcraft-denoise --features reference-validation --test reference \
    gpu_ -- --ignored --nocapture --test-threads=1
LIGHTCRAFT_REQUIRE_DENOISE_GPU=1 LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle \
  cargo test -p lightcraft-engine --features reference-validation --test denoise
LIGHTCRAFT_REQUIRE_DENOISE_GPU=1 LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle \
  cargo test -p lightcraft-ui-egui --features reference-validation tests_denoise --lib
```

These hardware gates are opt-in, not silently skipped required tests; ordinary
CI remains network-independent and the required real-checkpoint CPU job remains
unchanged. Missing weights or GPU fail an explicitly requested hardware gate.

| Local native development-build observation | CPU | Vulkan |
|---|---:|---:|
| Canon 8 MP ISO 800, 512×320 crop including decode | 4.50 s | 1.22 s |
| Same Canon, full Apply (221 tiles) | 75.19 s | 12.59 s |
| Same Canon, peak process RSS for validation workflow | 731,092 KiB | 843,656 KiB |
| Same Canon, cached Amount 50 / 100 / 50 at 1024px | 40.1 / 38.6 / 39.2 ms | 33.4 / 32.8 / 32.8 ms |
| Private Nikon D7100 ISO 6400 NEF, 4020×6036 full Apply | not timed | 40.33 s |

A repeat after adding buffer accounting took 12.76 s for full Apply, with
219,403,152 bytes of model/reusable activation/sentinel GPU buffers.
The Canon full operation is approximately 6× faster than the optimized CPU path.
The Nikon run overlapped an application build; its timing is an observation,
not an isolated benchmark or general guarantee. Its crop remained finite, with
mean linear correction 0.000279 and RMS correction 0.012799. Visual inspection
showed reduced noise and retained bark texture at Amount 50; no paired clean
reference or sensor-level parity is established. The native control channel
confirmed asynchronous preparing/processing/success with Vulkan and no fallback.
An English/Japanese screenshot review confirmed the device label fits the dialog.
Public model redistribution/delivery and broader photographic validation remain
release gates. GPU inference has been measured on this AMD device only.


The standard rendering benchmark passed with the matching Nikon lossless-12
fixture. A fresh build of the unchanged HEAD was also compared with the new
build on that same fixture: no CPU-time metric regressed by more than 20%.
Full 24 MP render CPU time was 8,188 versus 7,997 ms, with Vulkan/CPU export
agreement of max 1 LSB and mean 0.0025 LSB. An initial comparison against an
older run flagged CPU-time variation; the fresh matched baseline ruled out a
regression from this change.


### Linux Vulkan kernel optimization (2026-10-09)

GPU timestamps on the Radeon RX 9060 XT identified dense matrix products and
channel normalization as roughly 75% of dispatch time. The matrix kernel now
reuses each weight across two adjacent output pixels, with a fixed unrolled inner
reduction. Stride-two 2×2 downsampling uses the same matrix kernel with direct
coordinate gathering, avoiding im2col buffers. Deep normalization reduces 32
channel lanes across eight adjacent pixels. A bounded 2,048-entry binding cache
reuses immutable parameters/bindings between tiles. No feature flag, dependency,
precision change, new model, tile-size change or processing revision is introduced.

Sequential matched native development-build runs used the same private Nikon
D7100 ISO 6400 NEF (4020×6036), retained decoded source and Vulkan device. No
other build or validation ran during either full-image measurement:

| Operation / process measurement | Previous Vulkan | Optimized Vulkan |
|---|---:|---:|
| Full Apply, including lossless result storage | 36.99 s | 25.31 s |
| 512×320 crop including source decode (12 tiles) | 2.53 s | 2.27 s |
| Peak RSS, decode/preview/Apply/cached renders | 1,442,888 KiB | 1,441,404 KiB |
| Cached Amount 50 / 100 / 50 at 1024px | 108.4 / 105.7 / 103.2 ms | 110.5 / 123.7 / 110.5 ms |

Full Apply is 32% faster (1.46×). Cached editing still reuses the completed
correction and shows ordinary timing variation; this change targets inference.
Retained enhancement memory remains 586,285,440 bytes for this validation workflow.
The completed crop reported 219,443,616 bytes of GPU model/activation/sentinel/
parameter buffers. Parameter buffers add at most 98,304 bytes to the existing
bounded GPU workspace; driver and bind-group bookkeeping are excluded.

Five warmed public 256px tile calls took 32.08–33.01 ms, versus 57.65–58.60 ms
before optimization. The opt-in 35 ms local budget fails on the original kernel.
Two untimed calls warm driver compilation and the reusable binding cache; this
is a steady-tile budget, not a guarantee for model installation/loading or the
first tile. Reproduce on an otherwise idle physical GPU:

```sh
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle \
LIGHTCRAFT_NAFNET_GPU_TILE_BUDGET_MS=35 \
  cargo test -p lightcraft-denoise --features reference-validation --test reference \
    vulkan_tile_inference_meets_explicit_latency_budget -- --ignored --nocapture --test-threads=1
```

Independent PyTorch maximum errors remained 4.47e-8 (1×1), 1.49e-7 (padding)
and 2.38e-7 (overlapping tiles), below 1e-4. Crop/full consistency stayed exact,
and cancellation and reported CPU fallback passed. Before and Amount zero were
pixel-identical to the previous Vulkan implementation. At 1024px, full renders
changed only 17 channel samples at Amount 50 and 12 at Amount 100, by at most one
8-bit level. Both original NEF checksums remained unchanged. These measurements
preserve the existing tested photographic behavior; broader camera/high-ISO
quality and other GPU hardware remain validation gaps.

`cargo xtask ci` passed all seven gates. Real-checkpoint engine and headless UI
tests also passed with Vulkan explicitly required. The rebuilt native app completed
both preview (12 tiles) and asynchronous Apply (221 tiles) on the Canon corpus
photo with no CPU fallback. During Apply, status calls returned in 0.46–32.52 ms;
the final inspection reported a 0.36 ms frame and an 8.49 ms maximum update.
A native dialog screenshot review and one confirmation pass checked the completed
preview/Apply state. The strict ordinary-render benchmark passed with the same
Nikon lossless-12 corpus fixture: export CPU time was 8,145 versus 7,944 ms,
GPU export 327 versus 326 ms, and CPU/GPU output agreement stayed at max 1 LSB
and mean 0.0025 LSB.


### Fused kernels, paired submissions and model reuse (2026-10-09)

A fresh GPU timestamp profile of the previous optimized path put dense matrix
products at roughly 70% of dispatch time. A four-value register tile now reuses
operands across two channels and two pixels. Depthwise convolution feeds
SimpleGate directly; attention multiplication, learned residual scales and
residual addition are fused into their pointwise convolutions. This removes
180 intermediate dispatches per tile and reduces reusable activation storage.
F32 precision, normalization epsilon, global pooling, fixed tiles/overlap,
zero padding and the processing revision remain unchanged.

Linux queues at most two complete tile submissions before waiting for readback.
Queue ordering safely reuses activation buffers; two independent readback slots
and a staging belt reuse transfer storage. Progress and blending stay in source
tile order. Cancellation is checked before preparation and between consumed
tiles; one additional tile may already be queued. Failures retry the batch on
CPU with a reported reason, and cancellation/failure cannot commit an edit.
CPU and Metal keep serial tile processing.

The engine retains one model across previews, Apply and photo changes; only
weights/workspace are shared. Each photo still supplies its own source pixels
and correction/cache key. Changed or missing weight files and GPU preference
changes invalidate the model. Failed Linux GPU models can retry initialization on the
next operation when the host permits acceleration. Loading and memory inspection
use a short cache lock, with no GPU wait while that lock is held.

Matched, isolated native development-build runs on the same Radeon RX 9060 XT
and Nikon D7100 ISO 6400 NEF (4020×6036), using the public engine commands:

| Operation / process measurement | Fresh previous baseline | Fused + paired + resident |
|---|---:|---:|
| Full Apply, including lossless result storage | 24.32 s | 12.58 s |
| 512×320 crop including decode, cold model (12 tiles) | 2.35 s | 2.25 s |
| Peak RSS for decode/preview/Apply/cached renders | 1,442,916 KiB | 1,442,364 KiB |
| Cached Amount 50 / 100 / 50 at 1024px | 109.6 / 103.9 / 107.4 ms | 108.4 / 107.4 / 108.5 ms |
| Retained enhancement CPU bytes | 586,285,440 | 703,710,732 |
| Denoiser GPU buffers after preview | 219,443,616 | 203,782,000 |

Full Apply improved by 48% (1.93×). Retaining the model costs 117,425,292 additional
host bytes (weights plus the input workspace) and keeps approximately 204 MB of
GPU buffers live. The upload pool allowance is bounded to two tile-sized chunks;
allocator, bind-group and driver overhead are excluded. Peak process RSS was
roughly unchanged, since full-image processing dominates the peak. These are
local observations, not a universal hardware budget.

Five warmed public 256px tile calls took **23.01–23.35 ms**, versus 31.75–33.50 ms
before this change, passing an explicit 25 ms local gate. Paired submissions
reduced a 768×512/12-tile test from 285.12–288.32 ms after fusion alone to
267.51–271.08 ms, passing an explicit 275 ms gate. Repeated 32×24 engine crop
previews fell from 127.45 ms with repeated loading to 23–26 ms with model reuse,
passing a 100 ms gate. The engine test also checks memory accounting, no preview
edit, GPU off/on recovery and corrupt replacement rejection. These opt-in tests
require weights and a physical GPU and never accept a CPU fallback as a pass:

```sh
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle LIGHTCRAFT_NAFNET_GPU_TILE_BUDGET_MS=25 \
  cargo test -p lightcraft-denoise --features reference-validation --test reference \
    vulkan_tile_inference -- --ignored --nocapture --test-threads=1
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle LIGHTCRAFT_NAFNET_GPU_IMAGE_BUDGET_MS=275 \
  cargo test -p lightcraft-denoise --features reference-validation --test reference \
    vulkan_image_inference -- --ignored --nocapture --test-threads=1
LIGHTCRAFT_NAFNET_REFERENCE=/path/to/bundle LIGHTCRAFT_NAFNET_PREVIEW_BUDGET_MS=100 \
  cargo test -p lightcraft-engine --features reference-validation --test denoise \
    repeated_preview -- --ignored --nocapture --test-threads=1
```

Independent PyTorch maximum errors remain 4.47e-8 (1×1), 1.19e-7 (padding) and
2.38e-7 (overlapping tiles), below 1e-4. Crop/full pixels match exactly; cancellation
and reported CPU fallback pass. Against the previous Vulkan photographic renders,
Before and Amount zero are identical. Amount 50/100 at 1024px differ in only 15/18
channel samples by at most one 8-bit level; the 100% crop changes one channel
sample at Amount 100. Crop/full visual inspection found no new tile boundaries.
Both private NEF checksums remain unchanged. Photographic quality, model delivery
permission and broader GPU coverage remain the existing validation gates.


All seven `cargo xtask ci` gates passed, together with required CPU references,
explicit GPU references, engine behavior and headless UI tests. The rebuilt
release app completed crop preview (12 tiles) and asynchronous Apply (221 tiles)
on the CC0 Canon DNG with Vulkan and no CPU fallback. During Apply, 11 status
round trips took 0.49–19.40 ms; the final native inspection reported a 0.39 ms
frame and a 9.07 ms maximum update. One batched preview/hold-Space screenshot
review and one Apply confirmation pass checked the native result and retained
model accounting. Test app processes were terminated and reaped afterward.

The initial three-sample strict ordinary-render check flagged four GPU host-time
metrics (three 1.4→1.7 ms changes and one 2.9→4.1 ms change). The nine-sample
strict gate passed, although a direct baseline comparison still flagged four
different small host timings. Focused, matched 30-sample runs of the previous
binary, rebuilt binary and unchanged-binary control found no CPU-time regressions
above 20%. No renderer source or benchmark threshold changed. Highlights/clarity
GPU host time was 1.1 ms in all three focused runs; full loupe NR was
4.7→4.1→4.0 ms, with GPU wall time 24.0→23.8→23.5 ms. This resolves the flags as
non-repeatable timing variation in this experiment.

On the same Nikon lossless-12 corpus fixture, the nine-sample full runs measured
CPU export at 8,274→7,656 ms and GPU export wall time at 328.2→327.5 ms.
CPU/GPU agreement stayed at max 1 LSB and mean 0.0025 LSB. These renderer
measurements are separate from NAFNet inference and cached enhanced editing.
