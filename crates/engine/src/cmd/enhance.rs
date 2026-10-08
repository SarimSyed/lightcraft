use super::{CommandSpec, always, cmd};
pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "enhance.model.status", "AI Denoise Model Status", [], None, "{}", always, |s,_| Ok(s.denoise_model_status())),
        cmd!(query "enhance.denoise.preview", "Preview AI Denoise", [], None, "{photo?: id, amount?: 0..100, region?: {x,y,width,height} in source pixels, wait?: bool}", always, |s,p| s.denoise_preview(p)),
        cmd!("enhance.denoise.apply", "Apply AI Denoise", [], None, "{photo?: id, amount?: 0..100, wait?: bool}", always, |s, p| s.denoise_apply(p)),
        cmd!(query "enhance.denoise.status", "AI Denoise Status", [], None, "{}", always, |s,_| { let _ = s.denoise_poll(); Ok(serde_json::to_value(s.enhancer.status()).unwrap_or_default()) }),
        cmd!(query "enhance.denoise.cancel", "Cancel AI Denoise", [], None, "{}", always, |s,_| { s.enhancer.cancel(); Ok(serde_json::Value::Null) }),
        cmd!(query "enhance.model.download", "Download AI Denoise Model", [], None, "{} — explicit user download", always, |s,_| s.denoise_model_download()),
        cmd!(query "enhance.model.cancel", "Cancel AI Denoise Model Download", [], None, "{}", always, |s,_| { s.denoise_model_cancel(); Ok(serde_json::Value::Null) }),
    ]
}
