//! Safe `Engine` over the canonical `seraph` crate (the idiomatic Rust wrapper
//! around the SERAPH cdylib, `seraph-wrapper/src/lib.rs`).
//!
//! `Engine` owns a `seraph::Store` plus a single, lazily-created
//! `seraph::Encoder`; all text is encoded through that one encoder and fed to
//! the by-vector store API, so the model is loaded once. Every call goes
//! through the store's options forms (`put_with`, `put_batch`, `search_with`,
//! `chain_with`, ...), which take everything a call can vary in one place.
//!
//! `Engine` is `Send` (its fields are), and access is serialized by a `Mutex`
//! in `main`; it is intentionally not used concurrently.

use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

use seraph::{
    ChainResult, DoorCause, Encoder, Endpoint, EndpointFit, HistoryOptions, Magnitude,
    NeighborhoodOptions, PutItem, PutOptions, Retirement, RouteOptions, SearchOptions, Store,
    StoreConfig, StrandOptions, Tau, TauSpec, TemporalOrder, TimeFilter, TimeWindow, WarpInput,
    WarpProfile, WarpSpec,
};

pub struct Engine {
    // `encoder` is declared before `store` so it drops first (mirrors the old
    // explicit Drop order: close the encoder, then the store).
    encoder: Option<Encoder>,
    store: Store,
    model_id: String,
    dim: usize,
}

/// A warp direction as a tool supplies it: text to encode and a magnitude,
/// either a number or a level name resolved from the store's calibration.
pub struct WarpPoint {
    pub text: String,
    pub magnitude: Magnitude,
}

/// One frame of a batch as a tool supplies it.
pub struct BatchItem {
    pub content: String,
    pub metadata: Option<Value>,
    pub occurred_at: Option<i64>,
}

/// A similarity threshold as a tool supplies it: a number, or a level of the
/// store's own background similarity (how alike unrelated frames are), which
/// moves with the corpus as it densifies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TauLevel {
    Value(f32),
    /// The background's 75th percentile: the default.
    Loose,
    /// Its 95th percentile.
    Background,
    /// Its 99th percentile.
    Strict,
}

/// A threshold and the offset added to it once it is resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauArg {
    pub level: TauLevel,
    pub offset: f32,
}

/// Which time `recent` reads frames by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecentBy {
    /// When they were written.
    Written,
    /// When their content happened (their write time when they give none).
    Occurred,
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

/// The threshold a search walks at when the caller's is higher. The engine's
/// tau gates the walk as well as the hits: a branch whose door scores below it
/// is never entered, so raising it can lose the best match (measured on the
/// memory store: a query whose best frame scores 0.634 returns nothing at tau
/// 0.62). Reads walk at this gate at most and filter their hits at the
/// caller's threshold, so a higher threshold trims the list instead.
const WALK_TAU: f32 = 0.5;

/// Values for the named warp levels, used only when the store has no warp
/// calibration of its own to resolve a name against.
fn fallback_magnitude(name: &str) -> Option<f32> {
    match name {
        "action_floor" => Some(0.05),
        "routing" => Some(0.20),
        "sub_saturation" => Some(0.80),
        "saturation" => Some(1.50),
        _ => None,
    }
}

/// Read a tool-supplied magnitude: a number, or a level name.
pub fn parse_magnitude(v: Option<&Value>) -> Result<Magnitude> {
    match v {
        None | Some(Value::Null) => Ok(Magnitude::Named("routing".into())),
        Some(Value::Number(n)) => Ok(Magnitude::Value(n.as_f64().unwrap_or(0.20) as f32)),
        Some(Value::String(s)) => match s.trim().parse::<f32>() {
            Ok(x) => Ok(Magnitude::Value(x)),
            Err(_) if fallback_magnitude(s).is_some() => Ok(Magnitude::Named(s.clone())),
            Err(_) => Err(anyhow!(
                "unknown magnitude '{s}' — use a number or one of action_floor, routing, sub_saturation, saturation"
            )),
        },
        Some(other) => Err(anyhow!("magnitude must be a number or a level name, got {other}")),
    }
}

/// Read a tool-supplied threshold: a number (or a numeric string), or a level
/// name. Absent means the store's loose level.
pub fn parse_tau(v: Option<&Value>) -> Result<TauLevel> {
    match v {
        None | Some(Value::Null) => Ok(TauLevel::Loose),
        Some(Value::Number(n)) => Ok(TauLevel::Value(n.as_f64().unwrap_or(0.5) as f32)),
        Some(Value::String(s)) => {
            let t = s.trim().to_ascii_lowercase();
            if let Ok(x) = t.parse::<f32>() {
                return Ok(TauLevel::Value(x));
            }
            match t.as_str() {
                "" | "loose" | "p75" => Ok(TauLevel::Loose),
                "background" | "auto" | "p95" => Ok(TauLevel::Background),
                "strict" | "p99" => Ok(TauLevel::Strict),
                _ => Err(anyhow!("unknown tau '{s}' — use a number 0.0-1.0 or one of loose, background, strict")),
            }
        }
        Some(other) => Err(anyhow!("tau must be a number or a level name, got {other}")),
    }
}

/// Parse a tool-supplied warp profile.
pub fn parse_profile(s: Option<&str>) -> Result<WarpProfile> {
    match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("bounded") => Ok(WarpProfile::Bounded),
        Some("routing") => Ok(WarpProfile::Routing),
        Some("saturation") => Ok(WarpProfile::Saturation),
        Some(other) => Err(anyhow!("unknown warp profile '{other}' — use bounded, routing, or saturation")),
    }
}

/// Microseconds since the epoch as an RFC 3339 string.
fn iso(us: i64) -> Option<String> {
    chrono::DateTime::from_timestamp_micros(us).map(|dt| dt.to_rfc3339())
}

fn round4(x: f32) -> f64 {
    (x as f64 * 10000.0).round() / 10000.0
}

fn chain_value(c: ChainResult) -> Value {
    json!({
        "frame_ids": c.frame_ids,
        "primitive": c.primitive,
        "reached": c.reached,
        "coherence": {
            "mean_gradient": round4(c.coherence.mean_gradient),
            "min_gradient": round4(c.coherence.min_gradient),
            "fragmentation_events": c.coherence.fragmentation_events,
            "pass_strict": c.coherence.pass_strict,
            "pass_loose": c.coherence.pass_loose,
            "stretch": if c.coherence.stretch.is_finite() { json!(round4(c.coherence.stretch)) } else { Value::Null },
        },
    })
}

/// One end of a time read as a tool gives it: an existing frame, or text that
/// stands for the frames that fit it best.
enum End {
    Frame(String),
    Text(Vec<f32>),
}

impl End {
    fn endpoint(&self) -> Endpoint<'_> {
        match self {
            End::Frame(f) => Endpoint::Frame(f),
            End::Text(e) => Endpoint::Query(e),
        }
    }
}

/// Metadata as the engine takes it: a JSON object. A JSON object sent as a
/// string (clients facing a typeless schema stringify arguments) is unwrapped;
/// any other value is kept under `value`, so nothing a caller sent is lost.
fn metadata_map(m: Option<&Value>) -> Map<String, Value> {
    match m {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(o)) => o.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => o,
            _ => Map::from_iter([("value".to_string(), json!(s))]),
        },
        Some(other) => Map::from_iter([("value".to_string(), other.clone())]),
    }
}

/// Turn the engine's refusal of an event time on an older store into advice.
fn explain_format(e: seraph::Error) -> anyhow::Error {
    let msg = e.to_string();
    if msg.contains("format_version") || matches!(e, seraph::Error::Format(_)) {
        anyhow!("{msg} — this memory store predates event time; migrate it to the current format to record when things happened")
    } else {
        anyhow!(msg)
    }
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
        // own encoder + corpus once on load and persist it. Named warp levels
        // resolve against this calibration. Non-fatal on error.
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

impl Engine {
    /// Commit one text. `occurred_at` records when the content happened
    /// (microseconds since the epoch), a claim bound into the frame's record.
    pub fn remember(&mut self, content: &str, metadata: Option<&Value>, occurred_at: Option<i64>) -> Result<Value> {
        let emb = self.encode(content)?;
        let options = PutOptions { metadata: metadata_map(metadata), occurred_at, ..Default::default() };
        let fid = self.store.put_with(content.as_bytes(), &emb, &options).map_err(explain_format)?;
        Ok(json!({ "frame_id": fid }))
    }

    /// Commit many texts in one batched encode and one batch commit; each item
    /// keeps its own metadata and event time.
    pub fn remember_batch(&mut self, items: Vec<BatchItem>) -> Result<Value> {
        let texts: Vec<String> = items.iter().map(|i| i.content.clone()).collect();
        let embs = self.encode_batch(&texts)?;
        let batch: Vec<PutItem> = items
            .into_iter()
            .zip(embs)
            .map(|(item, embedding)| PutItem {
                content: item.content.into_bytes(),
                embedding,
                options: PutOptions {
                    metadata: metadata_map(item.metadata.as_ref()),
                    occurred_at: item.occurred_at,
                    ..Default::default()
                },
            })
            .collect();
        let ids = match self.store.put_batch(&batch) {
            Ok(ids) => ids,
            Err(seraph::Error::BatchIncomplete { committed, message }) => {
                self.store.sync().ok();
                return Err(anyhow!("batch stopped after {} frame(s) ({message}); committed: {committed:?}", committed.len()));
            }
            Err(e) => return Err(explain_format(e)),
        };
        // A batch leaves the store buffered; this daemon commits each call.
        self.store.sync()?;
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

/// What `recall` asks: how many hits, at what threshold, from which frames.
pub struct RecallQuery {
    pub query: String,
    pub top_k: usize,
    pub tau: TauArg,
    pub include_superseded: bool,
    pub order: TemporalOrder,
    /// On when frames were written (the engine's commit time).
    pub written: TimeWindow,
    /// On when their content happened, for frames that record it.
    pub occurred: TimeWindow,
}

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

    /// A threshold resolved against this store's background similarity, plus
    /// its offset, held to 0..=1.
    fn tau(&self, t: TauArg) -> f32 {
        let base = match t.level {
            TauLevel::Value(v) => v,
            TauLevel::Loose => self.store.resolve_tau_spec(TauSpec::Loose),
            TauLevel::Background => self.store.resolve_tau_spec(TauSpec::Auto),
            TauLevel::Strict => self.store.resolve_tau_spec(TauSpec::Strict),
        };
        (base + t.offset).clamp(0.0, 1.0)
    }

    /// A frame as a time read shows it: its snippet, when it was written, and
    /// when its content happened where it records that.
    fn brief(&self, frame_id: &str) -> Value {
        let mut v = json!({ "frame_id": frame_id, "snippet": self.content_snippet(frame_id) });
        if let Ok(Some(l)) = self.store.lifecycle(frame_id) {
            let obj = v.as_object_mut().expect("object");
            if let Some(t) = iso(l.committed.committed_at) {
                obj.insert("timestamp".into(), json!(t));
            }
            if let Some(t) = l.occurred_at.and_then(iso) {
                obj.insert("occurred_at".into(), json!(t));
            }
        }
        v
    }

    /// A route with each of its frames shown in place of their IDs.
    fn route_value(&self, c: ChainResult) -> Value {
        let frames: Vec<Value> = c.frame_ids.iter().map(|f| self.brief(f)).collect();
        let mut v = chain_value(c);
        let obj = v.as_object_mut().expect("object");
        obj.remove("frame_ids");
        obj.insert("frames".into(), json!(frames));
        v
    }

    /// A frame's place in the record: the order frames were written in.
    fn position(&self, frame_id: &str) -> Option<u32> {
        self.store.lifecycle(frame_id).ok().flatten().map(|l| l.committed.position)
    }

    /// One end of a time read: an existing frame's ID as given, else text.
    fn end(&mut self, s: &str) -> Result<End> {
        let t = s.trim();
        if self.store.borrow_frame(t).is_some() {
            return Ok(End::Frame(t.to_string()));
        }
        Ok(End::Text(self.encode(t)?))
    }

    /// The warp a `toward` point compiles to, if one is given.
    fn toward(&mut self, toward: Option<&WarpPoint>) -> Result<Option<seraph::Warp>> {
        let Some(p) = toward else { return Ok(None) };
        let spec = self.warp_spec(std::slice::from_ref(p), &[], WarpProfile::Bounded, 1.5)?;
        Ok(Some(self.compile(spec)?.1))
    }

    /// A hit with its snippet. Its write time is included when `with_time` is
    /// set (a time-scoped or recency-ordered recall), and the time its content
    /// happened whenever the frame records one.
    fn hit_value(&self, h: seraph::Hit, with_time: bool) -> Value {
        let mut v = json!({
            "frame_id": h.frame_id,
            "score": round4(h.score),
            "confidence": round4(h.confidence),
            "path_length": h.path_length,
            "snippet": self.content_snippet(&h.frame_id),
        });
        let obj = v.as_object_mut().expect("object");
        if with_time {
            if let Some(t) = iso(h.timestamp) {
                obj.insert("timestamp".into(), json!(t));
            }
        }
        if let Some(t) = h.occurred_at.and_then(iso) {
            obj.insert("occurred_at".into(), json!(t));
        }
        v
    }

    /// Semantic recall, optionally scoped to when frames were written and when
    /// their content happened, and re-ordered by time. The windows decide which
    /// frames may occupy a result slot and the order is applied after
    /// selection; neither changes what the geometry retrieves.
    pub fn recall(&mut self, q: RecallQuery) -> Result<Value> {
        let emb = self.encode(&q.query)?;
        let with_time = q.order != TemporalOrder::Relevance
            || q.written != TimeWindow::default()
            || q.occurred != TimeWindow::default();
        let tau = self.tau(q.tau);
        let options = SearchOptions {
            top_k: q.top_k,
            tau: Tau::Value(tau.min(WALK_TAU)),
            include_superseded: q.include_superseded,
            time: TimeFilter { committed: q.written, occurred: q.occurred, ..Default::default() },
            order: q.order,
            ..Default::default()
        };
        let hits = self.store.search_with(&emb, &options)?.hits;
        let results: Vec<Value> =
            hits.into_iter().filter(|h| h.score >= tau).map(|h| self.hit_value(h, with_time)).collect();
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
            for k in ["embedding", "content_b64", "codes_hex", "quantized", "quantized_fine"] {
                obj.remove(k);
            }
            if let Some(content) = self.content_full(frame_id) {
                obj.insert("content".to_string(), json!(content));
            }
            for k in ["timestamp", "occurred_at"] {
                if let Some(t) = obj.get(k).and_then(|x| x.as_i64()).and_then(iso) {
                    obj.insert(format!("{k}_iso"), json!(t));
                }
            }
        }
        Ok(v)
    }

    /// The record's account of a frame: when it was written, when it became a
    /// landmark (eigenframe) and how, and when it was retired and how.
    pub fn lifecycle(&self, frame_id: &str) -> Result<Value> {
        let Some(l) = self.store.lifecycle(frame_id)? else {
            return Ok(json!({ "error": format!("frame not found: {frame_id}") }));
        };
        let event = |e: seraph::RecordEvent| json!({ "position": e.position, "at": iso(e.committed_at) });
        Ok(json!({
            "frame_id": l.frame_id,
            "written": event(l.committed),
            "occurred_at": l.occurred_at.and_then(iso),
            "landmark": l.door.map(|(e, cause)| {
                let mut v = event(e);
                v["cause"] = json!(match cause {
                    DoorCause::Genesis => "genesis",
                    DoorCause::Seed => "seed",
                    DoorCause::Threshold => "threshold",
                    DoorCause::Manual => "manual",
                });
                v
            }),
            "retired": l.retired.map(|(e, how)| {
                let mut v = event(e);
                match how {
                    Retirement::Superseded => { v["how"] = json!("superseded"); }
                    Retirement::Consolidated { into } => { v["how"] = json!("consolidated"); v["into"] = json!(into); }
                }
                v
            }),
        }))
    }

    pub fn eigenframes(&self) -> Result<Value> {
        parse_value(&self.store.eigenframes_json())
    }
}

// --------------------------------------------------------------------------- //
// chains, neighborhood, lineage, strands
// --------------------------------------------------------------------------- //

impl Engine {
    pub fn chain(&self, from_id: &str, to_id: &str) -> Result<Value> {
        let c = self.store.chain_with(Endpoint::Frame(from_id), Endpoint::Frame(to_id), &RouteOptions::default())?;
        Ok(chain_value(c))
    }

    pub fn chain_to(&mut self, query: &str) -> Result<Value> {
        let emb = self.encode(query)?;
        let c = self.store.chain_with(Endpoint::Entry, Endpoint::Query(&emb), &RouteOptions::default())?;
        Ok(chain_value(c))
    }

    pub fn chain_between(&mut self, query_a: &str, query_b: &str) -> Result<Value> {
        let ea = self.encode(query_a)?;
        let eb = self.encode(query_b)?;
        let c = self.store.chain_with(Endpoint::Query(&ea), Endpoint::Query(&eb), &RouteOptions::default())?;
        Ok(chain_value(c))
    }

    pub fn neighborhood(&self, frame_id: &str, tau: Option<f32>, hops: usize) -> Result<Value> {
        let options = NeighborhoodOptions { tau: tau.map_or(Tau::Background, Tau::Value), hops, warp: None };
        let n = self.store.neighborhood_with(frame_id, &options)?;
        let mut scores: Vec<(String, f32)> = n.scores.into_iter().collect();
        scores.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(json!({
            "consensus": n.consensus,
            "geometric": n.geometric,
            "lineage": n.lineage,
            "tau_used": round4(n.tau_used),
            "hops_used": n.hops_used,
            "scores": scores.into_iter().map(|(k, v)| (k, json!(round4(v)))).collect::<Map<String, Value>>(),
        }))
    }

    pub fn lineage(&self, frame_id: &str) -> Result<Value> {
        Ok(json!(self.store.lineage(frame_id)))
    }

    pub fn evolution(&self, seed_frame_id: &str) -> Result<Value> {
        parse_opt("evolution", self.store.evolution_json(seed_frame_id))
    }

}

// --------------------------------------------------------------------------- //
// time reads: strands, history, routes forward, recent
// --------------------------------------------------------------------------- //

/// What `history` asks: a topic, how much of its past, and from which frames.
pub struct HistoryQuery {
    pub query: String,
    pub top_k: usize,
    pub anchors: usize,
    pub toward: Option<WarpPoint>,
    pub written: TimeWindow,
    pub occurred: TimeWindow,
}

impl Engine {
    /// What a frame was built on (`forward = false`) or what was later built on
    /// it (`forward = true`): a strand through the frames it attached to when
    /// it was written, or that attached to it. `tau = None` stops at the
    /// frame's own background; `toward` steers which strand is followed;
    /// `as_of` reads the record as it stood then.
    pub fn strand(
        &mut self,
        frame_id: &str,
        forward: bool,
        tau: Option<f32>,
        toward: Option<&WarpPoint>,
        as_of: Option<i64>,
    ) -> Result<Value> {
        let warp = self.toward(toward)?;
        let options = StrandOptions {
            tau: tau.map_or(Tau::QueryBackground, Tau::Value),
            warp: warp.as_ref(),
            as_of,
            ..Default::default()
        };
        let c = if forward {
            self.store.influence_with(frame_id, &options)?
        } else {
            self.store.provenance_with(frame_id, &options)?
        };
        Ok(self.route_value(c))
    }

    /// How the store came to hold what a query finds: the query's top
    /// `anchors` hits choose the trees, and the best `top_k` frames along them
    /// come back oldest first (by when their content happened, else when they
    /// were written).
    pub fn history(&mut self, q: HistoryQuery) -> Result<Value> {
        let emb = self.encode(&q.query)?;
        let warp = self.toward(q.toward.as_ref())?;
        let options = HistoryOptions {
            top_k: q.top_k,
            tau: Tau::Value(WALK_TAU),
            anchors: q.anchors,
            warp: warp.as_ref(),
            time: TimeFilter { committed: q.written, occurred: q.occurred, ..Default::default() },
        };
        let h = self.store.history_with(&emb, &options)?;
        let hits: Vec<Value> = h
            .hits
            .into_iter()
            .map(|x| {
                let mut v = json!({
                    "frame_id": x.frame_id,
                    "score": round4(x.score),
                    "depth": x.depth,
                    "timestamp": iso(x.timestamp),
                    "snippet": self.content_snippet(&x.frame_id),
                });
                if let Some(t) = x.occurred_at.and_then(iso) {
                    v["occurred_at"] = json!(t);
                }
                v
            })
            .collect();
        Ok(json!({ "anchors": h.anchors, "hits": hits }))
    }

    /// How `from` led to `to`: the route forward in record order between them.
    /// Each end is a frame ID or text; text stands for its top 10 hits, and
    /// the best-fitting earlier/later pair a route joins is used.
    pub fn led_to(&mut self, from: &str, to: &str, as_of: Option<i64>) -> Result<Value> {
        let (a, b) = (self.end(from)?, self.end(to)?);
        let options = RouteOptions {
            forward: true,
            fit: Some(EndpointFit { top_k: 10, tau: WALK_TAU }),
            as_of,
            ..Default::default()
        };
        let c = self.store.chain_with(a.endpoint(), b.endpoint(), &options)?;
        let reached = c.reached;
        let mut v = self.route_value(c);
        if let (false, End::Frame(fa), End::Frame(fb)) = (reached, &a, &b) {
            if self.position(fa) > self.position(fb) {
                v["note"] = json!("`from` was written after `to`; a route forward in time starts at the earlier frame, so swap them");
            }
        }
        Ok(v)
    }

    /// The frame a route end stands for: an existing frame's ID as given, or
    /// the best hit for text in the record as it stood at `as_of`.
    fn resolve_frame(&mut self, s: &str, as_of: Option<i64>) -> Result<String> {
        match self.end(s)? {
            End::Frame(f) => Ok(f),
            End::Text(emb) => {
                let options = SearchOptions {
                    top_k: 1,
                    tau: Tau::Value(WALK_TAU),
                    time: TimeFilter { committed: TimeWindow { after_us: None, before_us: as_of }, ..Default::default() },
                    ..Default::default()
                };
                let hit = self.store.search_with(&emb, &options)?.hits.into_iter().next();
                hit.map(|h| h.frame_id)
                    .ok_or_else(|| anyhow!("nothing in memory fits '{s}'; rephrase it or give a frame ID"))
            }
        }
    }

    /// The distinct ways two frames connect: up to `max_routes` routes whose
    /// interiors share no frame, shortest first, each with its bridge (the
    /// interior frame closest to both ends). With `forward`, only routes that
    /// run forward in record order, from the earlier end to the later.
    pub fn connections(
        &mut self,
        a: &str,
        b: &str,
        max_routes: usize,
        forward: bool,
        as_of: Option<i64>,
    ) -> Result<Value> {
        let mut from = self.resolve_frame(a, as_of)?;
        let mut to = self.resolve_frame(b, as_of)?;
        if from == to {
            return Ok(json!({
                "from": self.brief(&from),
                "to": self.brief(&to),
                "routes": [],
                "note": "both ends resolve to the same frame; give more distinct ends or frame IDs",
            }));
        }
        let swapped = forward && self.position(&from) > self.position(&to);
        if swapped {
            std::mem::swap(&mut from, &mut to);
        }
        let options = RouteOptions { forward, as_of, ..Default::default() };
        let conns = self.store.connections_with(Endpoint::Frame(&from), Endpoint::Frame(&to), &options, max_routes)?;
        let routes: Vec<Value> = conns
            .into_iter()
            .map(|c| {
                let ids = &c.chain.frame_ids;
                let via: Vec<Value> = ids.iter().skip(1).take(ids.len().saturating_sub(2)).map(|f| self.brief(f)).collect();
                let stretch = c.chain.coherence.stretch;
                json!({
                    "bridge": c.bridge,
                    "via": via,
                    "reached": c.chain.reached,
                    "stretch": if stretch.is_finite() { json!(round4(stretch)) } else { Value::Null },
                })
            })
            .collect();
        let mut v = json!({ "from": self.brief(&from), "to": self.brief(&to), "forward": forward, "routes": routes });
        if swapped {
            v["swapped"] = json!(true);
        }
        Ok(v)
    }

    /// What the store has gained lately, newest first, with no query: the
    /// latest `limit` frames, or those written (or, `by` occurred, that
    /// happened) since `since`, with how many there are in all.
    pub fn recent(&self, since: Option<i64>, limit: usize, by: RecentBy, include_superseded: bool) -> Result<Value> {
        // Genesis, redirect and system frames are the store's own records, not memories.
        let shown = |status: u8| match status {
            0 | 5 | 6 => false,
            4 => include_superseded,
            _ => true,
        };
        let ids = self.store.frame_ids();
        let mut picked: Vec<(i64, &String)> = Vec::new();
        match by {
            RecentBy::Written => {
                // The record is in write order, so read it back from the end
                // and stop at the first frame written before `since`.
                for id in ids.iter().rev() {
                    let Some(f) = self.store.borrow_frame(id) else { continue };
                    let (status, t) = (f.status(), f.timestamp());
                    drop(f);
                    if since.is_some_and(|s| t < s) {
                        break;
                    }
                    if shown(status) {
                        picked.push((t, id));
                        if since.is_none() && picked.len() >= limit {
                            break;
                        }
                    }
                }
            }
            RecentBy::Occurred => {
                // Event time follows no order in the record: read every frame.
                for id in &ids {
                    let Some(status) = self.store.borrow_frame(id).map(|f| f.status()) else { continue };
                    if !shown(status) {
                        continue;
                    }
                    let Some(l) = self.store.lifecycle(id)? else { continue };
                    let t = l.occurred_at.unwrap_or(l.committed.committed_at);
                    if since.map_or(true, |s| t >= s) {
                        picked.push((t, id));
                    }
                }
                picked.sort_by(|x, y| y.0.cmp(&x.0));
            }
        }
        let total = picked.len();
        let frames: Vec<Value> = picked
            .into_iter()
            .take(limit)
            .map(|(_, id)| {
                let mut v = self.brief(id);
                if self.store.borrow_frame(id).is_some_and(|f| f.status() == 4) {
                    v["superseded"] = json!(true);
                }
                v
            })
            .collect();
        let mut v = json!({
            "by": match by { RecentBy::Written => "written", RecentBy::Occurred => "occurred" },
            "frames": frames,
        });
        if let Some(s) = since {
            v["since"] = json!(iso(s));
            v["total"] = json!(total);
        }
        Ok(v)
    }
}

// --------------------------------------------------------------------------- //
// warp
// --------------------------------------------------------------------------- //

impl Engine {
    /// A warp spec from text points. Named levels resolve against the store's
    /// own calibration; a store with none falls back to fixed values.
    fn warp_spec(&mut self, targets: &[WarpPoint], suppress: &[WarpPoint], profile: WarpProfile, bound_k: f32) -> Result<WarpSpec> {
        let mut side = |points: &[WarpPoint]| -> Result<Vec<(WarpInput, Magnitude)>> {
            points.iter().map(|p| Ok((WarpInput::Embedding(self.encode(&p.text)?), p.magnitude.clone()))).collect()
        };
        let (t, s) = (side(targets)?, side(suppress)?);
        Ok(WarpSpec { targets: t, suppress: s, profile, bound_k, ..Default::default() })
    }

    /// Compile a spec against the store, returning the spec as compiled. If a
    /// named level does not resolve (the store has no calibration yet), the
    /// fixed values stand in for it.
    fn compile(&self, spec: WarpSpec) -> Result<(WarpSpec, seraph::Warp)> {
        match self.store.compile_warp_spec(&spec) {
            Ok(w) => Ok((spec, w)),
            Err(e) if e.to_string().contains("named magnitude") => {
                let fixed = |side: &[(WarpInput, Magnitude)]| -> Vec<(WarpInput, Magnitude)> {
                    side.iter().map(|(i, m)| (i.clone(), match m {
                        Magnitude::Named(n) => Magnitude::Value(fallback_magnitude(n).unwrap_or(0.20)),
                        v => v.clone(),
                    })).collect()
                };
                let spec = WarpSpec { targets: fixed(&spec.targets), suppress: fixed(&spec.suppress), ..spec };
                let w = self.store.compile_warp_spec(&spec)?;
                Ok((spec, w))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn recall_with_warp(
        &mut self,
        query: &str,
        targets: &[WarpPoint],
        suppress: &[WarpPoint],
        profile: WarpProfile,
        bound_k: f32,
        top_k: usize,
        tau: TauArg,
    ) -> Result<Value> {
        let emb = self.encode(query)?;
        let spec = self.warp_spec(targets, suppress, profile, bound_k)?;
        let (_, warp) = self.compile(spec)?;
        let tau = self.tau(tau);
        let options = SearchOptions { top_k, tau: Tau::Value(tau.min(WALK_TAU)), warp: Some(&warp), ..Default::default() };
        let hits = self.store.search_with(&emb, &options)?.hits;
        let results: Vec<Value> =
            hits.into_iter().filter(|h| h.score >= tau).map(|h| self.hit_value(h, false)).collect();
        Ok(json!({ "profile": format!("{profile:?}").to_ascii_lowercase(), "results": results }))
    }

    pub fn warp_chain_to(
        &mut self,
        query: &str,
        targets: &[WarpPoint],
        suppress: &[WarpPoint],
        profile: WarpProfile,
        bound_k: f32,
    ) -> Result<Value> {
        let emb = self.encode(query)?;
        let spec = self.warp_spec(targets, suppress, profile, bound_k)?;
        let (_, warp) = self.compile(spec)?;
        let options = RouteOptions { warp: Some(&warp), ..Default::default() };
        Ok(chain_value(self.store.chain_with(Endpoint::Entry, Endpoint::Query(&emb), &options)?))
    }

    /// Route between two concepts, each end warped by its own targets.
    pub fn warp_chain_between(
        &mut self,
        query_a: &str,
        query_b: &str,
        targets_a: &[WarpPoint],
        targets_b: &[WarpPoint],
        profile: WarpProfile,
        bound_k: f32,
    ) -> Result<Value> {
        let ea = self.encode(query_a)?;
        let eb = self.encode(query_b)?;
        let mut warped = |emb: Vec<f32>, targets: &[WarpPoint]| -> Result<Vec<f32>> {
            if targets.is_empty() {
                return Ok(emb);
            }
            let spec = self.warp_spec(targets, &[], profile, bound_k)?;
            let (spec, _) = self.compile(spec)?;
            Ok(self.store.warp_query(&emb, &spec.to_json())?)
        };
        let wa = warped(ea, targets_a)?;
        let wb = warped(eb, targets_b)?;
        Ok(chain_value(self.store.chain_with(Endpoint::Query(&wa), Endpoint::Query(&wb), &RouteOptions::default())?))
    }

    pub fn seed_margin(&self, frame_id: &str) -> Result<Value> {
        let m = self.store.seed_margin(frame_id);
        Ok(json!({ "frame_id": frame_id, "seed_margin": round4(m) }))
    }

    pub fn steerability(&mut self, query: &str, target_id: &str) -> Result<Value> {
        let emb = self.encode(query)?;
        let d = self.store.steerability(&emb, target_id);
        Ok(json!({ "target_id": target_id, "displacement": (d as f64 * 1_000_000.0).round() / 1_000_000.0 }))
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
        // An empty JSON array means "no relation filter" (all relations).
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
        Ok(json!({ "frame_id": frame_id, "density_score": round4(s) }))
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
