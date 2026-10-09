//! Single-photo RGB-stage enhancement. Workers prepare artifacts; the session commits edits.
#[cfg(not(target_arch = "wasm32"))]
mod cache;
#[cfg(not(target_arch = "wasm32"))]
use crate::media::{SourceLevel, SourceRef};
use crate::{Result, Session, media::DecodedSource};
use lightcraft_catalog::{PhotoId, Source};
use lightcraft_denoise::{Cancellation, Region};
use lightcraft_develop::{DenoiseModel, DevelopSettings};
use lightcraft_raster::{Rgb32f, Rgba8};
use serde::Serialize;
use serde_json::{Value, json};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
    mpsc,
};

pub const MODEL_FILE: &str = "nafnet-sidd-width32-v1.safetensors";
pub const MODEL_DIGEST: &str = "c6ae62717e6a8388e376302a592ada0d84333836fef32c25c25a130d72d6159a";
pub const MODEL_BYTES: u64 = 116_701_252;
pub const RELEASE_URL: &str = "https://github.com/SarimSyed/lightcraft/releases/download/nafnet-sidd-width32-v1";
// A code licence alone does not establish redistribution permission for external weights.
pub const REDISTRIBUTION_VERIFIED: bool = false;
static SCRATCH_ID: AtomicU64 = AtomicU64::new(0);
type SharedResult = Arc<Mutex<Option<(String, Arc<Rgb32f>)>>>;
#[cfg(not(target_arch = "wasm32"))]
type SharedModel = Arc<Mutex<Option<(String, Arc<lightcraft_denoise::NafNet>)>>>;
type SharedOriginal = Arc<Mutex<Option<(String, DecodedSource, String)>>>;

fn selection() -> DenoiseModel {
    DenoiseModel {
        id: lightcraft_denoise::MODEL_ID.into(),
        checkpoint: MODEL_DIGEST.into(),
        processing_revision: lightcraft_denoise::PROCESSING_REVISION.into(),
    }
}
fn validate_selection(model: &DenoiseModel) -> std::result::Result<(), String> {
    if *model != selection() {
        return Err("Unsupported AI denoise checkpoint or processing revision; restore the recorded model/revision".into());
    }
    Ok(())
}
fn stamp(origin: &Source) -> String {
    match origin {
        Source::File { path } => {
            let meta = std::fs::metadata(path).ok();
            #[cfg(unix)]
            let identity = {
                use std::os::unix::fs::MetadataExt;
                meta.as_ref().map(|m| (m.dev(), m.ino(), m.ctime(), m.ctime_nsec()))
            };
            #[cfg(not(unix))]
            let identity = ();
            format!("{path}:{:?}:{:?}:{identity:?}", meta.as_ref().map(|m| m.len()), meta.and_then(|m| m.modified().ok()))
        }
        _ => serde_json::to_string(origin).unwrap_or_default(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct SourceResolver {
    source: SourceRef,
    origin: Source,
    cache_dir: PathBuf,
    model_dir: PathBuf,
    model: SharedModel,
    calibration: u64,
    memory: SharedResult,
    original: SharedOriginal,
    gate: Arc<Mutex<()>>,
}
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
pub struct SourceResolver;
impl SourceResolver {
    #[cfg(not(target_arch = "wasm32"))]
    fn process(
        &self,
        model: &DenoiseModel,
        region: Option<Region>,
        cancel: &Cancellation,
        progress: &(dyn Fn(usize, usize) + Sync),
    ) -> std::result::Result<Completed, String> {
        use lightcraft_denoise::Denoiser;
        validate_selection(model)?;
        cancel.check().map_err(|e| e.to_string())?;
        // ponytail: one enhancement at a time per session, including cache regeneration.
        let _gate = self.gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        cancel.check().map_err(|e| e.to_string())?;
        let initial = stamp(&self.origin);
        let previous = self
            .original
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .filter(|(stamp, _, _)| stamp == &initial)
            .map(|(_, source, _)| source.clone());
        let source = match previous {
            Some(source) => source,
            None => {
                let source = match &self.origin {
                    Source::File { path } => {
                        let (image, info) = crate::files::load_enhancement(path)?;
                        DecodedSource::new(Arc::new(image), Some(info))
                    }
                    _ => self.source.load_source()?,
                };
                lightcraft_denoise::validate(&source.image).map_err(|e| e.to_string())?;
                let digest = self.content_digest(&source.image)?;
                *self.original.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((initial.clone(), source.clone(), digest));
                source
            }
        };
        let key = self.key(&source.image, model)?;
        // A corrupt/missing artifact can be regenerated, but never silently bypassed.
        let cached = self.cached(&key, &source.image, model).ok().flatten();
        let was_cached = cached.is_some();
        let (image, r, fallback, backend, fallback_reason, gpu_buffer_bytes) = if let Some(cached) = cached {
            let r = region.unwrap_or(Region { x: 0, y: 0, width: cached.width, height: cached.height });
            (if region.is_none() { cached } else { Arc::new(crop(&cached, r)?) }, r, false, "cache".into(), None, 0)
        } else {
            let network = self.network(model)?;
            let input = lightcraft_denoise::model_input(&source.image).map_err(|e| e.to_string())?;
            let prediction = network.infer(&input, region, cancel, progress).map_err(|e| e.to_string())?;
            let image = lightcraft_denoise::restore_working(&source.image, &prediction.image, Some(prediction.region)).map_err(|e| e.to_string())?;
            (Arc::new(image), prediction.region, prediction.cpu_fallback, prediction.backend, prediction.fallback_reason, prediction.gpu_buffer_bytes)
        };
        cancel.check().map_err(|e| e.to_string())?;
        if stamp(&self.origin) != initial || self.key(&source.image, model)? != key {
            return Err("Original changed during AI denoise; retry".into());
        }
        if region.is_none() && !was_cached {
            self.save(&key, model, &image)?;
        }
        cancel.check().map_err(|e| e.to_string())?;
        Ok(Completed { original: source, enhanced: image, region: r, fallback, backend, fallback_reason, gpu_buffer_bytes })
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn network(&self, selection: &DenoiseModel) -> std::result::Result<Arc<lightcraft_denoise::NafNet>, String> {
        let path = self.model_dir.join(MODEL_FILE);
        let origin = Source::File { path: path.to_string_lossy().into() };
        #[cfg(target_os = "linux")]
        let availability = lightcraft_gpu::unavailable_reason();
        #[cfg(not(target_os = "linux"))]
        let availability: Option<String> = None;
        let identity = format!("{}:{availability:?}", stamp(&origin));
        let cached = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .filter(|(key, network)| key == &identity && network.checkpoint_digest == selection.checkpoint)
            .map(|(_, network)| network.clone());
        if let Some(network) = cached {
            #[cfg(target_os = "linux")]
            let retry_gpu = network.backend() == "CPU" && availability.is_none();
            #[cfg(not(target_os = "linux"))]
            let retry_gpu = false;
            if !retry_gpu {
                return Ok(network);
            }
        }
        // Loading/uploading and dropping GPU resources must not hold the inspection lock.
        let previous = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        drop(previous);
        #[cfg(target_os = "linux")]
        let network = lightcraft_denoise::NafNet::load_vulkan(&path, lightcraft_gpu::compute_device(), lightcraft_gpu::unavailable_reason);
        #[cfg(not(target_os = "linux"))]
        let network = lightcraft_denoise::NafNet::load_accelerated(&path);
        let network = network.map_err(|e| format!("{e}; install or restore the verified model to regenerate this photo's result"))?;
        if network.checkpoint_digest != selection.checkpoint {
            return Err("Checkpoint checksum mismatch; reinstall NAFNet SIDD width-32".into());
        }
        if format!("{}:{availability:?}", stamp(&origin)) != identity {
            return Err("Checkpoint changed while loading; reinstall or retry NAFNet".into());
        }
        let network = Arc::new(network);
        let previous = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner).replace((identity, network.clone()));
        drop(previous);
        Ok(network)
    }
    #[cfg(target_arch = "wasm32")]
    fn process(
        &self,
        _: &DenoiseModel,
        _: Option<Region>,
        _: &Cancellation,
        _: &(dyn Fn(usize, usize) + Sync),
    ) -> std::result::Result<Completed, String> {
        Err("AI denoise requires native LightCraft".into())
    }
    /// Central render source. Errors propagate through render and export.
    pub fn resolve(&self, settings: &DevelopSettings) -> std::result::Result<DecodedSource, String> {
        let model = settings.enhance.model.as_ref().ok_or("no denoise model selected")?;
        let result = crate::guard::catch("AI denoise source", || self.process(model, None, &Cancellation::default(), &|_, _| {}))??;
        let image = lightcraft_denoise::blend(&result.original.image, &result.enhanced, settings.enhance.denoise).map_err(|e| e.to_string())?;
        Ok(DecodedSource { image: Arc::new(image), info: result.original.info, camera_tone: result.original.camera_tone })
    }
}
fn crop(image: &Rgb32f, region: Region) -> std::result::Result<Rgb32f, String> {
    if region.width == 0
        || region.height == 0
        || region.x.checked_add(region.width).is_none_or(|e| e > image.width)
        || region.y.checked_add(region.height).is_none_or(|e| e > image.height)
    {
        return Err("Preview region lies outside the full source".into());
    }
    let mut data = Vec::with_capacity(region.width * region.height);
    for y in region.y..region.y + region.height {
        data.extend_from_slice(
            image.data.get(y * image.width + region.x..y * image.width + region.x + region.width).ok_or("invalid preview extent")?,
        );
    }
    Ok(Rgb32f { width: region.width, height: region.height, data })
}
struct Completed {
    original: DecodedSource,
    enhanced: Arc<Rgb32f>,
    region: Region,
    fallback: bool,
    backend: String,
    gpu_buffer_bytes: u64,
    fallback_reason: Option<String>,
}
#[derive(Clone, Default, Serialize)]
pub struct Status {
    pub state: String,
    pub done: usize,
    pub total: usize,
    pub error: Option<String>,
    pub cpu_fallback: bool,
    pub backend: String,
    pub gpu_buffer_bytes: u64,
    pub fallback_reason: Option<String>,
    pub operation: u64,
    pub applying: bool,
}
struct Pending {
    rx: mpsc::Receiver<std::result::Result<Completed, String>>,
    cancel: Cancellation,
    photo: PhotoId,
    settings: u64,
    origin_stamp: String,
    library: u64,
    apply: bool,
    amount: f64,
    model: DenoiseModel,
}
pub struct Preview {
    pub original: Arc<Rgb32f>,
    pub enhanced: Arc<Rgb32f>,
    pub region: Region,
    pub info: lightcraft_pipeline::SourceInfo,
    pub operation: u64,
}
impl Preview {
    pub fn render(&self, amount: f64, before: bool) -> std::result::Result<Rgba8, String> {
        let image = if before {
            (*self.original).clone()
        } else {
            lightcraft_denoise::blend(&self.original, &self.enhanced, amount).map_err(|e| e.to_string())?
        };
        let req = lightcraft_pipeline::RenderRequest { apply_crop: false, ..lightcraft_pipeline::RenderRequest::fit(image.width, image.height) };
        Ok(lightcraft_pipeline::render(&image, &self.info, &DevelopSettings::default(), &req).image)
    }
}
pub struct Enhancer {
    #[cfg(not(target_arch = "wasm32"))]
    model: SharedModel,
    pub model_dir: PathBuf,
    pub background: bool,
    pub preview: Option<Preview>,
    status: Arc<Mutex<Status>>,
    pending: Option<Pending>,
    scratch: PathBuf,
    memory: SharedResult,
    epoch: u64,
    sequence: u64,
    original: SharedOriginal,
    gate: Arc<Mutex<()>>,
    #[cfg(not(target_arch = "wasm32"))]
    download_cancel: Arc<std::sync::atomic::AtomicBool>,
    model_status: Arc<Mutex<Status>>,
}
impl Default for Enhancer {
    fn default() -> Self {
        Self {
            model_dir: std::env::var_os("LIGHTCRAFT_NAFNET_DIR")
                .map(PathBuf::from)
                .or_else(|| crate::camera_profiles::config_dir().map(|d| d.join("models").join("nafnet")))
                .unwrap_or_else(|| std::env::temp_dir().join("lightcraft-models").join("nafnet")),
            #[cfg(not(target_arch = "wasm32"))]
            model: Default::default(),
            background: false,
            preview: None,
            status: Arc::new(Mutex::new(Status::default())),
            pending: None,
            scratch: std::env::temp_dir().join(format!("lightcraft-enhance-{}-{}", std::process::id(), SCRATCH_ID.fetch_add(1, Ordering::Relaxed))),
            memory: Default::default(),
            epoch: 0,
            sequence: 0,
            original: Default::default(),
            gate: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            download_cancel: Default::default(),
            model_status: Default::default(),
        }
    }
}
impl Drop for Enhancer {
    fn drop(&mut self) {
        self.invalidate();
        self.cancel_download();
    }
}
impl Enhancer {
    pub(crate) fn usage(&self) -> crate::memory::Usage {
        let mut images: Vec<Arc<Rgb32f>> = Vec::new();
        if let Some((_, source, _)) = &*self.original.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
            images.push(source.image.clone());
        }
        if let Some((_, result)) = &*self.memory.lock().unwrap_or_else(std::sync::PoisonError::into_inner) {
            images.push(result.clone());
        }
        if let Some(preview) = &self.preview {
            for image in [&preview.original, &preview.enhanced] {
                if !images.iter().any(|other| Arc::ptr_eq(other, image)) {
                    images.push(image.clone());
                }
            }
        }
        let mut usage = crate::memory::Usage::new(images.len(), images.iter().map(|image| image.data.len().saturating_mul(12)).sum());
        let (cpu, _) = self.model_bytes();
        if cpu > 0 {
            usage.count += 1;
            usage.bytes = usage.bytes.saturating_add(cpu);
        }
        usage
    }
    pub(crate) fn model_bytes(&self) -> (usize, u64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().map_or((0, 0), |(_, model)| model.resident_bytes())
        }
        #[cfg(target_arch = "wasm32")]
        (0, 0)
    }
    pub fn status(&self) -> Status {
        self.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
    pub fn invalidate(&mut self) {
        self.cancel();
        self.epoch = self.epoch.wrapping_add(1);
        self.preview = None;
        self.memory = Default::default();
        self.original = Default::default();
        self.gate = Default::default();
    }
    pub fn cancel(&mut self) {
        if let Some(p) = self.pending.take() {
            p.cancel.cancel();
            self.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).state = "cancelled".into();
        }
    }
    fn cancel_download(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.download_cancel.store(true, Ordering::Relaxed);
    }
}
impl Session {
    pub fn enhancement_source(&mut self, photo: &lightcraft_catalog::Photo) -> SourceResolver {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = photo;
            SourceResolver
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let cache_dir =
                self.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.join("enhancements")).unwrap_or_else(|| self.enhancer.scratch.clone());
            SourceResolver {
                source: self.media.origin_ref(&photo.source, SourceLevel::Full.max_edge()),
                origin: photo.source.clone(),
                cache_dir,
                model_dir: self.enhancer.model_dir.clone(),
                model: self.enhancer.model.clone(),
                calibration: crate::camera_profiles::cache_key(),
                memory: self.enhancer.memory.clone(),
                original: self.enhancer.original.clone(),
                gate: self.enhancer.gate.clone(),
            }
        }
    }
    pub fn denoise_preview(&mut self, params: &Value) -> Result<Value> {
        self.denoise_start(params, false)
    }
    pub fn denoise_apply(&mut self, params: &Value) -> Result<Value> {
        self.denoise_start(params, true)
    }
    fn denoise_start(&mut self, params: &Value, apply: bool) -> Result<Value> {
        let c = if apply { "enhance.denoise.apply" } else { "enhance.denoise.preview" };
        if cfg!(target_arch = "wasm32") {
            return Err(crate::cmd::bad(c, "AI denoise requires native LightCraft"));
        }
        let named = params
            .get("photo")
            .map(|value| value.as_u64().map(PhotoId).ok_or_else(|| crate::cmd::bad(c, "photo must be an unsigned photo ID")))
            .transpose()?;
        let id = named.or(self.active()).ok_or_else(|| crate::cmd::bad(c, "no photo selected"))?;
        let wait = params
            .get("wait")
            .map(|value| value.as_bool().ok_or_else(|| crate::cmd::bad(c, "wait must be a boolean")))
            .transpose()?
            .unwrap_or(!self.enhancer.background);
        let photo = self.catalog.photo(id).filter(|photo| !photo.deleted).cloned().ok_or_else(|| crate::cmd::bad(c, "photo no longer exists"))?;
        if (photo.width as usize).checked_mul(photo.height as usize).is_none_or(|n| n == 0 || n > lightcraft_denoise::MAX_PIXELS)
            || photo.width > 32768
            || photo.height > 32768
        {
            return Err(crate::cmd::bad(c, "source exceeds the enhancement dimension limit (100 megapixels, 32768 pixels per edge)"));
        }
        if self.interaction.is_some() {
            return Err(crate::cmd::bad(c, "finish the current edit before AI denoise"));
        }
        let amount = params.get("amount").map_or(Some(50.0), Value::as_f64).ok_or_else(|| crate::cmd::bad(c, "amount must be a number"))?;
        if !amount.is_finite() || !(0.0..=100.0).contains(&amount) {
            return Err(crate::cmd::bad(c, "amount must be in 0..100"));
        }
        let region = if apply {
            None
        } else {
            params
                .get("region")
                .map(|r| {
                    serde_json::from_value::<Region>(r.clone()).map_err(|e| crate::cmd::bad(c, format!("region uses source pixel coordinates: {e}")))
                })
                .transpose()?
        };
        let model = photo.develop.enhance.model.clone().unwrap_or_else(selection);
        validate_selection(&model).map_err(|e| crate::cmd::bad(c, e))?;
        self.enhancer.cancel();
        self.enhancer.sequence = self.enhancer.sequence.wrapping_add(1);
        if !apply {
            self.enhancer.preview = None;
        }
        *self.enhancer.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Status { state: "preparing".into(), operation: self.enhancer.sequence, applying: apply, ..Default::default() };
        let cancel = Cancellation::default();
        let token = cancel.clone();
        let progress = self.enhancer.status.clone();
        let resolver = self.enhancement_source(&photo);
        let request_model = model.clone();
        let origin_stamp = stamp(&photo.source);
        let operation = self.enhancer.sequence;
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("ai-denoise".into())
            .spawn(move || {
                let result = crate::guard::catch("AI denoise worker", || {
                    resolver.process(&request_model, region, &token, &|done, total| {
                        if token.check().is_err() {
                            return;
                        }
                        let mut p = progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        if p.operation != operation || token.check().is_err() {
                            return;
                        }
                        p.state = "processing".into();
                        p.done = done;
                        p.total = total;
                    })
                })
                .and_then(|r| r);
                let _ = tx.send(result);
            })
            .map_err(|e| {
                let mut state = self.enhancer.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                state.state = "failure".into();
                state.error = Some(e.to_string());
                crate::cmd::bad(c, e.to_string())
            })?;
        self.enhancer.pending = Some(Pending {
            rx,
            cancel,
            photo: id,
            settings: photo.develop.hash64(),
            origin_stamp,
            library: self.enhancer.epoch,
            apply,
            amount,
            model,
        });
        if wait {
            let result = self.enhancer.pending.as_ref().ok_or_else(|| crate::cmd::bad(c, "operation disappeared"))?.rx.recv();
            match result {
                Ok(result) => self.denoise_complete(result)?,
                Err(error) => {
                    self.enhancer.cancel();
                    return Err(crate::cmd::bad(c, format!("AI denoise wait: {error}; operation cancelled")));
                }
            }
        }
        Ok(serde_json::to_value(self.enhancer.status()).unwrap_or(Value::Null))
    }
    fn denoise_complete(&mut self, result: std::result::Result<Completed, String>) -> Result<()> {
        let Some(pending) = self.enhancer.pending.take() else {
            return Ok(());
        };
        let result = result.and_then(|result| {
            pending.cancel.check().map_err(|e| e.to_string())?;
            let photo = self.catalog.photo(pending.photo).filter(|photo| !photo.deleted).ok_or("photo was deleted during AI denoise")?;
            if pending.library != self.enhancer.epoch
                || pending.settings != photo.develop.hash64()
                || pending.origin_stamp != stamp(&photo.source)
                || self.interaction.is_some()
            {
                return Err("AI denoise result is stale after a source, library or edit change; retry".into());
            }
            Ok(result)
        });
        match result {
            Ok(result) => {
                if pending.apply {
                    let mut settings = self
                        .develop_of(pending.photo)
                        .ok_or_else(|| crate::cmd::bad("enhance.denoise.apply", "photo no longer exists"))?
                        .as_ref()
                        .clone();
                    settings.enhance.model = Some(pending.model);
                    settings.enhance.denoise = pending.amount;
                    // Enhance is a single-photo operation even when Auto Sync is enabled.
                    let committed = self.execute_fn("enhance.denoise.apply", |s| {
                        let op = s
                            .develop_op(pending.photo, settings, "AI Denoise")
                            .ok_or_else(|| crate::cmd::bad("enhance.denoise.apply", "photo no longer exists"))?;
                        s.commit("AI Denoise", op)?;
                        Ok(Value::Null)
                    });
                    if let Err(error) = committed {
                        let mut state = self.enhancer.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        state.state = "failure".into();
                        state.error = Some(error.to_string());
                        return Err(error);
                    }
                }
                // Apply must not retain a second full-resolution original just for the dialog.
                let preview_region = if pending.apply {
                    let width = result.region.width.min(512);
                    let height = result.region.height.min(320);
                    Region { x: (result.region.width - width) / 2, y: (result.region.height - height) / 2, width, height }
                } else {
                    result.region
                };
                let enhanced = if preview_region == result.region {
                    result.enhanced
                } else {
                    Arc::new(crop(&result.enhanced, preview_region).map_err(|e| crate::cmd::bad("enhance.denoise.preview", e))?)
                };
                self.enhancer.preview = Some(Preview {
                    original: Arc::new(crop(&result.original.image, preview_region).map_err(|e| crate::cmd::bad("enhance.denoise.preview", e))?),
                    enhanced,
                    region: preview_region,
                    info: result.original.info.unwrap_or_default(),
                    operation: self.enhancer.sequence,
                });
                let mut status = self.enhancer.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                status.state = "success".into();
                status.cpu_fallback = result.fallback;
                status.backend = result.backend;
                status.gpu_buffer_bytes = result.gpu_buffer_bytes;
                status.fallback_reason = result.fallback_reason;
                Ok(())
            }
            Err(error) => {
                let mut status = self.enhancer.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                status.state = "failure".into();
                status.error = Some(error.clone());
                Err(crate::cmd::bad("enhance.denoise", error))
            }
        }
    }
    pub fn denoise_poll(&mut self) -> Result<bool> {
        match self.enhancer.pending.as_ref().map(|p| p.rx.try_recv()) {
            Some(Ok(result)) => {
                self.denoise_complete(result)?;
                Ok(true)
            }
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                self.denoise_complete(Err("AI denoise worker disconnected; retry".into()))?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    pub fn denoise_model_status(&self) -> Value {
        let installed = std::fs::metadata(self.enhancer.model_dir.join(MODEL_FILE)).is_ok_and(|m| m.len() == MODEL_BYTES);
        json!({"installed":installed,"model":lightcraft_denoise::MODEL_ID,"sha256":MODEL_DIGEST,"bytes":MODEL_BYTES,"directory":self.enhancer.model_dir,
            "licence":"MIT + BasicSR Apache-2.0 (code)","licenceUrl":"https://github.com/megvii-research/NAFNet/blob/main/LICENSE","redistributionVerified":REDISTRIBUTION_VERIFIED,"releaseUrl":RELEASE_URL,"download":self.enhancer.model_status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()})
    }
    pub fn denoise_model_cancel(&mut self) {
        self.enhancer.cancel_download();
    }
    pub fn denoise_model_download(&mut self) -> Result<Value> {
        if !REDISTRIBUTION_VERIFIED {
            return Err(crate::cmd::bad(
                "enhance.model.download",
                "Model download is not released: checkpoint redistribution permission and the published fork release must be verified. Manual offline installation is available; see docs/denoise.md.",
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut state = self.enhancer.model_status.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.state == "downloading" {
                return Err(crate::cmd::bad("enhance.model.download", "model download already running"));
            }
            *state = Status { state: "downloading".into(), ..Default::default() };
            drop(state);
            self.enhancer.download_cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let cancel = self.enhancer.download_cancel.clone();
            let state = self.enhancer.model_status.clone();
            let dir = self.enhancer.model_dir.clone();
            std::thread::Builder::new()
                .name("nafnet-download".into())
                .spawn(move || {
                    let result = crate::guard::catch("NAFNet download", || {
                        lightcraft_fetch::download(
                            &[lightcraft_fetch::FileSpec { name: MODEL_FILE, size: Some(MODEL_BYTES), sha256: Some(MODEL_DIGEST), max: 128 << 20 }],
                            &[RELEASE_URL.into()],
                            &dir,
                            &Default::default(),
                            &cancel,
                            &mut |p| {
                                let mut state = state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                                state.done = p.done as usize;
                                state.total = p.total as usize;
                            },
                        )
                        .map_err(|e| e.to_string())
                    })
                    .and_then(|r| r);
                    let mut state = state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    match result {
                        Ok(()) => state.state = "success".into(),
                        Err(e) => {
                            state.state = if cancel.load(Ordering::Relaxed) { "cancelled" } else { "failure" }.into();
                            state.error = Some(e);
                        }
                    }
                })
                .map_err(|e| crate::cmd::bad("enhance.model.download", e.to_string()))?;
            Ok(self.denoise_model_status())
        }
        #[cfg(target_arch = "wasm32")]
        Err(crate::cmd::bad("enhance.model.download", "native LightCraft required"))
    }
}
