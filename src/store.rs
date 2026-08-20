//! Safe `Engine` over the canonical `seraph` crate (the idiomatic Rust wrapper
//! around the SERAPH cdylib, `seraph-wrapper/src/lib.rs`).
//!
//! This module used to hand-transcribe the C ABI from `FFI_API.md`. It now
//! depends on the `seraph` crate directly, so the binding surface can never
//! drift from the engine again (the warp-JSON envelope bug came from exactly
//! that drift). `Engine` owns a `seraph::Store` plus a single, lazily-created
//! `seraph::Encoder`; all text is encoded through that one encoder and fed to
//! the by-vector store API, so the model is loaded once.
//!
//! `Engine` is `Send` (its fields are), and access is serialized by a `Mutex`
//! in `main`; it is intentionally not used concurrently.

use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use seraph::{Encoder, Store, StoreConfig, TemporalOrder};

pub struct Engine {
    // `encoder` is declared before `store` so it drops first (mirrors the old
    // explicit Drop order: close the encoder, then the store).
    encoder: Option<Encoder>,
    store: Store,
    model_id: String,
    dim: usize,
}

// --------------------------------------------------------------------------- //
// small helpers
// --------------------------------------------------------------------------- //

/// Parse a JSON string returned by the engine into a `Value`.
fn parse_value(s: &str) -> Result<Value> {
    serde_json::from_str(s).map_err(|e| anyhow!("invalid JSON from engine: {e}; raw: {s}"))
}

/// Parse an optional JSON string; `None` (engine returned no data) is an error.
fn parse_opt(label: &str, s: Option<String>) -> Result<Value> {
    match s {
        Some(s) => parse_value(&s),
        None => Err(anyhow!("{label} returned no data")),
    }
}

/// Named warp magnitudes (the by-vector warp API is float-level; names are
/// resolved here before building the warp JSON).
pub fn named_magnitude(name: &str) -> Option<f32> {
    match name {
        "action_floor" => Some(0.05),
        "routing" => Some(0.20),
        "sub_saturation" => Some(0.80),
        "saturation" => Some(1.50),
        _ => None,
    }
}

/// Resolve a tool-supplied magnitude (number or named string) to a float.
pub fn resolve_mag(v: &Value) -> f32 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.20) as f32,
        Value::String(s) => named_magnitude(s).unwrap_or(0.20),
        _ => 0.20,
    }
}

/// A single search hit enriched with a content snippet.
/// A frame's write time as an RFC 3339 string, decoded from its own id.
///
/// Frame ids are UUIDv7: the leading 48 bits are unix milliseconds, which is
/// the same creation ordering the engine's recency sort follows. The time is
/// therefore already in hand for every hit — no engine round trip needed.
fn frame_timestamp_iso(frame_id: &str) -> Option<String> {
    let hex: String = frame_id.chars().filter(|c| *c != '-').take(12).collect();
    if hex.len() < 12 {
        return None;
    }
    let ms = u64::from_str_radix(&hex, 16).ok()?;
    chrono::DateTime::from_timestamp_millis(ms as i64).map(|dt| dt.to_rfc3339())
}

#[cfg(test)]
mod timestamp_tests {
    use super::frame_timestamp_iso;

    #[test]
    fn decodes_write_time_from_the_v7_id() {
        // A real frame id from the store: leading 48 bits are unix millis.
        let iso = frame_timestamp_iso("019f97ad-194b-7ec9-929c-73cb3177d002").unwrap();
        assert!(iso.starts_with("2026-07-25T"), "got {iso}");
    }

    #[test]
    fn rejects_ids_that_are_not_v7() {
        assert!(frame_timestamp_iso("").is_none());
        assert!(frame_timestamp_iso("not-a-uuid").is_none());
        assert!(frame_timestamp_iso("zzzzzzzz-zzzz-7000-8000-000000000000").is_none());
    }
}

fn hit_to_value(frame_id: String, score: f32, confidence: f32, path_length: u32, snippet: Option<String>) -> Value {
    json!({
        "frame_id": frame_id,
        "score": (score * 10000.0).round() / 10000.0,
        "confidence": (confidence * 10000.0).round() / 10000.0,
        "path_length": path_length,
        "snippet": snippet,
    })
}

// --------------------------------------------------------------------------- //
// lifecycle
// --------------------------------------------------------------------------- //

impl Engine {
    /// Open an existing store, or create it (with `default_model`) if missing.
    pub fn open_or_create(path: &Path, default_model: &str) -> Result<Engine> {
        let mut store = if path.exists() {
            Store::open(path)?
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Store::create(path, Some(StoreConfig::new().model_id(default_model)))?
        };

        let model_id = {
            let m = store.model_id();
            if m.is_empty() { default_model.to_string() } else { m }
        };
        let dim = store.dim();

        // Self-tuning warp: as the single-writer daemon, calibrate the store's
        // own encoder + corpus once on load and persist it (a _seraph_warp_cal:
        // System sentinel). Geometry-gated — a no-op on a fresh/undeveloped
        // store — and re-persists only if a named magnitude drifted > 0.1.
        // Named-magnitude warp resolves against this; non-fatal on error.
        match store.fixup_warp() {
            Ok(true) => eprintln!("seraph: warp (re)calibrated on load"),
            Ok(false) => {}
            Err(e) => eprintln!("seraph: warp fixup skipped: {e}"),
        }

        Ok(Engine { encoder: None, store, model_id, dim })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Lazily create (once) and borrow the engine's encoder.
    fn encoder(&mut self) -> Result<&mut Encoder> {
        if self.encoder.is_none() {
            self.encoder = Some(Encoder::from_pretrained(&self.model_id)?);
        }
        Ok(self.encoder.as_mut().expect("encoder just set"))
    }

    /// Encode one text into an owned embedding vector.
    fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.encoder()?.encode(text)?)
    }

    /// Encode many texts in a single batched forward pass.
    fn encode_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.encoder()?.encode_batch(&refs)?)
    }
}

// --------------------------------------------------------------------------- //
// write
// --------------------------------------------------------------------------- //

/// The engine FFI requires a JSON *object* string; anything else parses to an
/// empty map with no error. Normalize the caller's value so nothing is
/// silently lost: unwrap a JSON-object-in-a-string (clients facing a typeless
/// schema stringify arguments), and wrap any other non-object as
/// `{"value": ...}`.
fn normalize_metadata(m: &Value) -> String {
    match m {
        Value::Object(_) => m.to_string(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Object(_)) => v.to_string(),
            _ => json!({ "value": s }).to_string(),
        },
        Value::Null => "{}".to_string(),
        other => json!({ "value": other }).to_string(),
    }
}

impl Engine {
    pub fn remember(&mut self, content: &str, metadata: Option<&Value>) -> Result<Value> {
        let emb = self.encode(content)?;
        let meta_str = metadata.map(normalize_metadata).unwrap_or_else(|| "{}".to_string());
        let fid = self.store.put_with_metadata(content.as_bytes(), &emb, &meta_str)?;
        Ok(json!({ "frame_id": fid }))
    }

    pub fn remember_batch(&mut self, contents: &[String]) -> Result<Value> {
        let embs = self.encode_batch(contents)?;
        let mut ids = Vec::with_capacity(contents.len());
        for (content, emb) in contents.iter().zip(embs.iter()) {
            let fid = self.store.put(content.as_bytes(), emb)?;
            ids.push(fid);
        }
        Ok(json!({ "frame_ids": ids, "count": ids.len() }))
    }

    pub fn add_seed(&mut self, name: &str) -> Result<Value> {
        let emb = self.encode(name)?;
        let fid = self.store.add_seed(name, &emb, None)?;
        Ok(json!({ "frame_id": fid, "seed_name": name }))
    }

    pub fn promote(&mut self, frame_id: &str) -> Result<Value> {
        self.store.promote(frame_id)?;
        Ok(json!({ "promoted": frame_id }))
    }
}

// --------------------------------------------------------------------------- //
// search & retrieval
// --------------------------------------------------------------------------- //

impl Engine {
    /// Full decoded content of a frame (raw bytes, no base64 round-trip).
    fn content_full(&self, frame_id: &str) -> Option<String> {
        self.store
            .get_content(frame_id)
            .ok()
            .flatten()
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    fn content_snippet(&self, frame_id: &str) -> Option<String> {
        let mut s = self.content_full(frame_id)?;
        if s.chars().count() > 240 {
            s = s.chars().take(240).collect::<String>() + "…";
        }
        Some(s)
    }

    /// Semantic recall, optionally scoped to a time window and re-ordered by
    /// recency. The window gates which frames may occupy a result slot and the
    /// order is applied after selection — neither changes what the geometry
    /// retrieves. When a temporal option is in play each hit also carries its
    /// timestamp, since a recency-ordered list is not readable without one.
    pub fn recall(
        &mut self,
        query: &str,
        top_k: usize,
        tau: f32,
        include_superseded: bool,
        order: TemporalOrder,
        after_us: Option<u64>,
        before_us: Option<u64>,
    ) -> Result<Value> {
        let emb = self.encode(query)?;
        let temporal =
            order != TemporalOrder::Relevance || after_us.is_some() || before_us.is_some();
        let hits = self.store.search_temporal(
            &emb,
            top_k,
            tau,
            include_superseded,
            order,
            after_us,
            before_us,
        )?;
        let results: Vec<Value> = hits
            .into_iter()
            .map(|h| {
                let snip = self.content_snippet(&h.frame_id);
                let ts = if temporal {
                    frame_timestamp_iso(&h.frame_id)
                } else {
                    None
                };
                let mut v = hit_to_value(h.frame_id, h.score, h.confidence, h.path_length, snip);
                if let (Some(obj), Some(ts)) = (v.as_object_mut(), ts) {
                    obj.insert("timestamp".to_string(), json!(ts));
                }
                v
            })
            .collect();
        Ok(json!(results))
    }

    /// Soft-forget a frame: mark it superseded so it stops surfacing in recall
    /// and chain waypoints, while its structural chain link is preserved. Use
    /// `recall(include_superseded=true)` to retrieve it for audit.
    pub fn supersede(&mut self, frame_id: &str) -> Result<Value> {
        let id = self.store.supersede(frame_id)?;
        Ok(json!({ "superseded": id }))
    }

    pub fn get_frame(&self, frame_id: &str) -> Result<Value> {
        let Some(raw) = self.store.get_frame_json(frame_id) else {
            return Ok(json!({ "error": format!("frame not found: {frame_id}") }));
        };
        let mut v = parse_value(&raw)?;
        // Strip the bulky/binary fields and surface plain full content instead of
        // the base64 blob — callers want the text, not the vector.
        if let Some(obj) = v.as_object_mut() {
            for k in ["embedding", "content_b64", "quantized", "quantized_fine"] {
                obj.remove(k);
            }
            if let Some(content) = self.content_full(frame_id) {
                obj.insert("content".to_string(), json!(content));
            }
        }
        Ok(v)
    }

    pub fn eigenframes(&self) -> Result<Value> {
        parse_value(&self.store.eigenframes_json())
    }
}

// --------------------------------------------------------------------------- //
// chains, neighborhood, lineage
// --------------------------------------------------------------------------- //

impl Engine {
    pub fn chain(&self, from_id: &str, to_id: &str) -> Result<Value> {
        parse_value(&self.store.chain_json(from_id, to_id)?)
    }

    pub fn chain_to(&mut self, query: &str) -> Result<Value> {
        let emb = self.encode(query)?;
        parse_value(&self.store.chain_to_json(&emb)?)
    }

    pub fn chain_between(&mut self, query_a: &str, query_b: &str) -> Result<Value> {
        let ea = self.encode(query_a)?;
        let eb = self.encode(query_b)?;
        parse_value(&self.store.chain_between_json(&ea, &eb)?)
    }

    pub fn neighborhood(&self, frame_id: &str, tau: Option<f32>, hops: usize) -> Result<Value> {
        let tau = tau.unwrap_or_else(|| self.store.resolve_tau());
        parse_value(&self.store.neighborhood_json(frame_id, tau, hops)?)
    }

    pub fn lineage(&self, frame_id: &str) -> Result<Value> {
        Ok(json!(self.store.lineage(frame_id)))
    }

    pub fn evolution(&self, seed_frame_id: &str) -> Result<Value> {
        parse_opt("evolution", self.store.evolution_json(seed_frame_id))
    }
}

// --------------------------------------------------------------------------- //
// warp
// --------------------------------------------------------------------------- //

impl Engine {
    /// Encode each `(text, magnitude)` pair into the canonical warp JSON
    /// `[{"embedding": [...], "magnitude": f32}, ...]` consumed by the engine's
    /// by-vector warp API (see `seraph::Store::search_with_warp`).
    fn warp_points_json(&mut self, items: &[(String, f32)]) -> Result<String> {
        let mut arr: Vec<Value> = Vec::with_capacity(items.len());
        for (text, mag) in items {
            let emb = self.encode(text)?;
            arr.push(json!({ "embedding": emb, "magnitude": mag }));
        }
        Ok(Value::Array(arr).to_string())
    }

    pub fn recall_with_warp(
        &mut self,
        query: &str,
        targets: &[(String, f32)],
        suppress: &[(String, f32)],
        profile: &str,
        bound_k: f32,
        top_k: usize,
        tau: f32,
    ) -> Result<Value> {
        let emb = self.encode(query)?;
        let tjson = self.warp_points_json(targets)?;
        let sjson = self.warp_points_json(suppress)?;
        let hits = self
            .store
            .search_with_warp(&emb, &tjson, &sjson, profile, bound_k, top_k, tau)?;
        let results: Vec<Value> = hits
            .into_iter()
            .map(|h| {
                let snip = self.content_snippet(&h.frame_id);
                hit_to_value(h.frame_id, h.score, h.confidence, h.path_length, snip)
            })
            .collect();
        Ok(json!({ "profile": profile, "results": results }))
    }

    /// Apply a warp field to an embedding (pure math, no search).
    fn apply_warp(
        &mut self,
        emb: &[f32],
        targets: &[(String, f32)],
        suppress: &[(String, f32)],
        profile: &str,
        bound_k: f32,
    ) -> Result<Vec<f32>> {
        let tjson = self.warp_points_json(targets)?;
        let sjson = self.warp_points_json(suppress)?;
        Ok(seraph::apply_warp(emb, &tjson, &sjson, profile, bound_k)?)
    }

    pub fn warp_chain_to(
        &mut self,
        query: &str,
        targets: &[(String, f32)],
        suppress: &[(String, f32)],
        profile: &str,
        bound_k: f32,
    ) -> Result<Value> {
        let emb = self.encode(query)?;
        let warped = self.apply_warp(&emb, targets, suppress, profile, bound_k)?;
        parse_value(&self.store.chain_to_json(&warped)?)
    }

    pub fn warp_chain_between(
        &mut self,
        query_a: &str,
        query_b: &str,
        targets_a: &[(String, f32)],
        targets_b: &[(String, f32)],
        profile: &str,
        bound_k: f32,
    ) -> Result<Value> {
        let ea = self.encode(query_a)?;
        let eb = self.encode(query_b)?;
        let wa = if targets_a.is_empty() { ea } else { self.apply_warp(&ea, targets_a, &[], profile, bound_k)? };
        let wb = if targets_b.is_empty() { eb } else { self.apply_warp(&eb, targets_b, &[], profile, bound_k)? };
        parse_value(&self.store.chain_between_json(&wa, &wb)?)
    }

    pub fn seed_margin(&self, frame_id: &str) -> Result<Value> {
        let m = self.store.seed_margin(frame_id);
        Ok(json!({ "frame_id": frame_id, "seed_margin": (m * 10000.0).round() / 10000.0 }))
    }

    pub fn steerability(&mut self, query: &str, target_id: &str) -> Result<Value> {
        let emb = self.encode(query)?;
        let d = self.store.steerability(&emb, target_id);
        Ok(json!({ "target_id": target_id, "displacement": (d * 1_000_000.0).round() / 1_000_000.0 }))
    }

    pub fn calibrate_warp(&self, n_queries: usize) -> Result<Value> {
        parse_value(&self.store.calibrate_warp(n_queries, 3, None, 10, 42)?)
    }
}

// --------------------------------------------------------------------------- //
// typed structural edges
// --------------------------------------------------------------------------- //

impl Engine {
    pub fn declare_relation(&mut self, name: &str, flags: u32) -> Result<Value> {
        let slot = self.store.declare_relation(name, flags)?;
        Ok(json!({ "relation": name, "slot": slot, "flags": flags }))
    }

    pub fn list_relations(&self) -> Result<Value> {
        parse_opt("list_relations", self.store.list_relations_json())
    }

    pub fn assert_relation(&mut self, from: &str, to: &str, relation: &str, weight: f32) -> Result<Value> {
        self.store.assert_relation(from, to, relation, weight)?;
        Ok(json!({ "from": from, "to": to, "relation": relation, "weight": weight }))
    }

    pub fn retract_relation(&mut self, from: &str, to: &str, relation: &str) -> Result<Value> {
        self.store.retract_relation(from, to, relation)?;
        Ok(json!({ "retracted": { "from": from, "to": to, "relation": relation } }))
    }

    pub fn structural_neighbors(
        &self,
        frame_id: &str,
        relations: Option<&[String]>,
        direction: &str,
    ) -> Result<Value> {
        // The engine treats an empty JSON array as "no relation filter" (all
        // relations); a null pointer is an error, so default to "[]".
        let rels = relations
            .map(|r| Value::Array(r.iter().map(|s| json!(s)).collect()).to_string())
            .unwrap_or_else(|| "[]".to_string());
        parse_value(&self.store.structural_neighbors_json(frame_id, &rels, direction)?)
    }
}

// --------------------------------------------------------------------------- //
// diagnostics
// --------------------------------------------------------------------------- //

impl Engine {
    pub fn stats(&self) -> Result<Value> {
        let st = self.store.stats()?;
        let sim = self
            .store
            .similarity_stats_json()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        Ok(json!({
            "total_frames": st.total_frames,
            "active_frames": st.active_frames,
            "eigenframes": st.eigenframes,
            "superseded": st.superseded,
            "redirects": st.redirects,
            "model_id": self.model_id,
            "dim": self.dim,
            "similarity_distribution": sim,
        }))
    }

    pub fn contradictions(&self, min_divergence: f32) -> Result<Value> {
        parse_opt("contradictions", self.store.contradictions_json(min_divergence))
    }

    pub fn density(&self, frame_id: &str) -> Result<Value> {
        let s = self.store.density_score(frame_id);
        Ok(json!({ "frame_id": frame_id, "density_score": (s * 10000.0).round() / 10000.0 }))
    }

    pub fn region_health(&self, eigenframe_id: &str) -> Result<Value> {
        parse_opt("region_health", self.store.region_health_json(eigenframe_id))
    }

    pub fn drift_report(&self, window: usize, threshold: f32) -> Result<Value> {
        parse_opt("drift_report", self.store.drift_report_json(window, threshold))
    }

    pub fn verify(&self) -> Result<Value> {
        let (ok, failures) = self.store.verify()?;
        Ok(json!({
            "valid": ok,
            "failure_count": failures.len(),
            "failing_frames": failures.into_iter().take(20).collect::<Vec<_>>(),
        }))
    }
}
