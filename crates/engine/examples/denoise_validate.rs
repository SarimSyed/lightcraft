//! Development-only photographic crop validation: SOURCE MODEL_DIRECTORY OUTPUT_DIRECTORY.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
use lightcraft_engine::Session;
use serde_json::json;
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = args.first().ok_or("SOURCE MODEL_DIRECTORY OUTPUT_DIRECTORY required")?;
    let model = args.get(1).ok_or("MODEL_DIRECTORY required")?;
    let output = std::path::PathBuf::from(args.get(2).ok_or("OUTPUT_DIRECTORY required")?);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    // Development-only inputs for independently comparing model context/pooling. No weights
    // or private photographs are bundled with the product or committed to the repository.
    if args.get(3).is_some_and(|mode| mode == "input") {
        let (image, _) = lightcraft_engine::files::load_enhancement(source)?;
        let input = lightcraft_denoise::model_input(&image).map_err(|e| e.to_string())?;
        std::fs::write(output.join("input.f32"), bytemuck::cast_slice(&input.image().data)).map_err(|e| e.to_string())?;
        println!("{}", json!({"source":source,"width":image.width,"height":image.height,"encoding":"interleaved native-endian F32 sRGB"}));
        return Ok(());
    }
    let mut session = Session::new().with_fs();
    session.enhancer.model_dir = model.into();
    session.execute("library.import", &json!({"paths":[source]})).map_err(|e| e.to_string())?;
    let photo = session.catalog.photos().next().cloned().ok_or("import produced no photo")?;
    let (w, h) = (photo.width as usize, photo.height as usize);
    let (width, height) = (w.min(512), h.min(320));
    let t = std::time::Instant::now();
    session
        .execute(
            "enhance.denoise.preview",
            &json!({"photo":photo.id.0,"wait":true,"region":{"x":(w-width)/2,"y":(h-height)/2,"width":width,"height":height}}),
        )
        .map_err(|e| e.to_string())?;
    let elapsed = t.elapsed().as_secs_f64();
    let preview = session.enhancer.preview.as_ref().ok_or("preview absent")?;
    let mut delta = 0.0f64;
    let mut squared = 0.0f64;
    let mut outside = 0usize;
    for (before, after) in preview.original.data.iter().zip(&preview.enhanced.data) {
        for (a, b) in before.iter().zip(after) {
            let d = f64::from(b - a);
            delta += d;
            squared += d * d;
            outside += usize::from(*a < 0.0 || *a > 1.0);
            if !b.is_finite() {
                return Err("nonfinite output".into());
            }
        }
    }
    for (name, before, amount) in [("before", true, 100.0), ("after50", false, 50.0), ("after100", false, 100.0)] {
        let image = preview.render(amount, before)?;
        let bytes = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&image), &Default::default()).map_err(|e| e.to_string())?;
        std::fs::write(output.join(format!("{name}.png")), bytes).map_err(|e| e.to_string())?;
    }
    let n = preview.original.data.len() as f64 * 3.0;
    println!(
        "{}",
        json!({"source":source,"camera":photo.meta.camera,"iso":photo.meta.iso,"dimensions":[w,h],"region":preview.region,"seconds":elapsed,"meanLinearCorrection":delta/n,"rmsLinearCorrection":(squared/n).sqrt(),"originalOutOfRangeSamples":outside,"status":session.enhancer.status()})
    );
    if args.get(3).is_some_and(|mode| mode == "full") {
        let t = std::time::Instant::now();
        session.execute("enhance.denoise.apply", &json!({"photo":photo.id.0,"wait":true})).map_err(|e| e.to_string())?;
        let full_seconds = t.elapsed().as_secs_f64();
        session.execute("library.select", &json!({"ids":[photo.id.0]})).map_err(|e| e.to_string())?;
        let mut cached_ms = Vec::new();
        for amount in [0, 50, 100, 50] {
            session.execute("develop.set", &json!({"control":"enhance.denoise","value":amount})).map_err(|e| e.to_string())?;
            let t = std::time::Instant::now();
            let rendered = session.render_job(photo.id, 1024, 1024, false, true).ok_or("render job absent")?.run().rendered?;
            cached_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            let bytes = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&rendered.image), &Default::default())
                .map_err(|e| e.to_string())?;
            std::fs::write(output.join(format!("full-{amount}.png")), bytes).map_err(|e| e.to_string())?;
        }
        println!("{}", json!({"fullApplySeconds":full_seconds,"cachedAmountRenderMs":cached_ms,"memory":session.memory_report()}));
    }
    Ok(())
}
