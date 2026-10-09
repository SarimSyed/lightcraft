//! Atomic, lossless working-space result artifacts. The manifest is the commit marker.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;

fn bounded_read(path: &Path, max: u64) -> std::result::Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if file.metadata().map_err(|e| e.to_string())?.len() > max {
        return Err("enhancement artifact exceeds the size limit".into());
    }
    let mut data = Vec::new();
    file.take(max + 1).read_to_end(&mut data).map_err(|e| e.to_string())?;
    if data.len() as u64 > max {
        return Err("enhancement artifact grew beyond the size limit".into());
    }
    Ok(data)
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    key: String,
    width: usize,
    height: usize,
    space: String,
    digest: String,
    model: DenoiseModel,
}
impl SourceResolver {
    pub(super) fn content_digest(&self, original: &Rgb32f) -> std::result::Result<String, String> {
        let mut hash = Sha256::new();
        match &self.origin {
            Source::File { path } => {
                let mut file = std::fs::File::open(path).map_err(|e| format!("original unavailable: {path}: {e}"))?;
                let mut buf = [0u8; 64 * 1024];
                let mut total = 0u64;
                loop {
                    let n = file.read(&mut buf).map_err(|e| e.to_string())?;
                    if n == 0 {
                        break;
                    }
                    total = total.saturating_add(n as u64);
                    if total > 2 << 30 {
                        return Err("original exceeds 2 GiB enhancement limit".into());
                    }
                    hash.update(buf.get(..n).ok_or("invalid read extent")?);
                }
            }
            _ => {
                hash.update(serde_json::to_vec(&self.origin).map_err(|e| e.to_string())?);
                hash.update(bytemuck::cast_slice(&original.data));
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    pub(super) fn key(&self, original: &Rgb32f, model: &DenoiseModel) -> std::result::Result<String, String> {
        let stamp = stamp(&self.origin);
        let cached = self
            .original
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .filter(|(s, _, _)| s == &stamp)
            .map(|(_, _, digest)| digest.clone());
        let digest = match cached {
            Some(digest) => digest,
            None => self.content_digest(original)?,
        };
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(digest, "oriented-full-rec2020-decoder-v12", self.calibration, model)).map_err(|e| e.to_string())?)
        ))
    }
    fn paths(&self, key: &str) -> (PathBuf, PathBuf) {
        (self.cache_dir.join(format!("{key}.tif")), self.cache_dir.join(format!("{key}.json")))
    }
    pub(super) fn cached(&self, key: &str, original: &Rgb32f, model: &DenoiseModel) -> std::result::Result<Option<Arc<Rgb32f>>, String> {
        if let Some((stored, result)) = &*self.memory.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            && stored == key
        {
            return Ok(Some(result.clone()));
        }
        let (image, manifest) = self.paths(key);
        if !manifest.exists() {
            return Ok(None);
        }
        let meta: Manifest = serde_json::from_slice(&bounded_read(&manifest, 4096)?).map_err(|e| format!("invalid enhancement manifest: {e}"))?;
        if meta.key != key
            || meta.space != "linear-rec2020-d65-f32"
            || meta.model != *model
            || meta.width != original.width
            || meta.height != original.height
        {
            return Err("enhancement manifest differs from this source; regenerate AI denoise".into());
        }
        let bytes = bounded_read(&image, lightcraft_nafnet::MAX_PIXELS as u64 * 12 + 32 * 1024 * 1024)?;
        if format!("{:x}", Sha256::digest(&bytes)) != meta.digest {
            return Err("damaged enhancement TIFF; regenerate AI denoise".into());
        }
        // Read raw F32 samples: internal caches are Rec.2020, never guessed sRGB/ICC.
        let mut limits = tiff::decoder::Limits::default();
        limits.decoding_buffer_size = original.data.len().saturating_mul(12);
        limits.ifd_value_size = 4096;
        let mut decoder = tiff::decoder::Decoder::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?.with_limits(limits);
        if decoder.dimensions().map_err(|e| e.to_string())? != (meta.width as u32, meta.height as u32)
            || decoder.colortype().map_err(|e| e.to_string())? != tiff::ColorType::RGB(32)
        {
            return Err("invalid enhancement TIFF dimensions or sample type".into());
        }
        let tiff::decoder::DecodingResult::F32(values) = decoder.read_image().map_err(|e| e.to_string())? else {
            return Err("enhancement TIFF needs 32-bit float samples".into());
        };
        if values.len() != original.data.len() * 3 {
            return Err("invalid enhancement TIFF sample count".into());
        }
        let image = Rgb32f { width: meta.width, height: meta.height, data: values.as_chunks::<3>().0.to_vec() };
        lightcraft_nafnet::validate(&image).map_err(|e| e.to_string())?;
        let image = Arc::new(image);
        *self.memory.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((key.into(), image.clone()));
        Ok(Some(image))
    }
    pub(super) fn save(&self, key: &str, model: &DenoiseModel, image: &Rgb32f) -> std::result::Result<(), String> {
        lightcraft_nafnet::validate(image).map_err(|e| e.to_string())?;
        let (path, manifest) = self.paths(key);
        if let Source::File { path: original } = &self.origin {
            let original = std::fs::canonicalize(original).map_err(|e| format!("original unavailable: {e}"))?;
            for destination in [&path, &manifest] {
                match std::fs::canonicalize(destination) {
                    Ok(existing) if existing == original => {
                        return Err(
                            "AI denoise cache would overwrite original; move the source outside the enhancement cache or use another library".into(),
                        );
                    }
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(format!("cannot check enhancement cache destination: {e}")),
                }
            }
        }
        let bytes = lightcraft_codecs::encode_tiff(
            &lightcraft_codecs::EncodeImage::new(
                image.width as u32,
                image.height as u32,
                3,
                lightcraft_codecs::Samples::F32(bytemuck::cast_slice(&image.data)),
            ),
            lightcraft_codecs::TiffCompression::Deflate,
            &Default::default(),
        )
        .map_err(|e| e.to_string())?;
        let meta = Manifest {
            key: key.into(),
            width: image.width,
            height: image.height,
            space: "linear-rec2020-d65-f32".into(),
            digest: format!("{:x}", Sha256::digest(&bytes)),
            model: model.clone(),
        };
        crate::export::write_file_durable(&path.to_string_lossy(), &bytes)?;
        crate::export::write_file_durable(&manifest.to_string_lossy(), &serde_json::to_vec(&meta).map_err(|e| e.to_string())?)?;
        *self.memory.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((key.into(), Arc::new(image.clone())));
        Ok(())
    }
}
