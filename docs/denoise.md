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
| Runtime | Candle 0.9.2, F32, CPU; optional macOS Metal with CPU fallback |

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
| `enhance.denoise.status` | completion/progress/error/CPU fallback |
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

Reproduce photographic crops and optional full processing with development tools:

```sh
cargo run -p lightcraft-engine --example denoise_validate -- \
  corpus/raw/dng-canon-sx100-high-iso.dng /path/to/model /tmp/denoise-review full
```

The example uses public engine commands, saves Before/Amount previews and reports
elapsed time and retained cache memory. Use `/usr/bin/time -v` on the built
example binary for process peak RSS; timing `cargo` also counts compilation.
