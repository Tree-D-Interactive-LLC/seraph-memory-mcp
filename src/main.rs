//! seraph-memory-rs — a global MCP memory server over the SERAPH C FFI.
//!
//! One global store (auto-created at `%APPDATA%\seraph-memory\memory.sfg`, or
//! `SERAPH_MEMORY_STORE` / `--store <path>`), the engine's candle encoder, and a
//! curated set of substrate tools: remember/recall, reasoning chains,
//! neighborhoods, contradictions, lineage, warp steering, declared typed
//! relations, and diagnostics.
//!
//! Transport: stdio. stdout is the MCP channel — all logging goes to stderr.

mod store;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, ServerHandler, ServiceExt,
};
use serde_json::Value;

use store::{
    parse_magnitude, parse_profile, parse_tau, BatchItem, Engine, HistoryQuery, RecallQuery, RecentBy,
    TauArg, WarpPoint,
};

const DEFAULT_MODEL: &str = "BAAI/bge-small-en-v1.5";

// --------------------------------------------------------------------------- //
// time and order parsing — the application-layer half of temporal recall
// --------------------------------------------------------------------------- //

/// Parse a caller-supplied time expression into microseconds since the Unix
/// epoch, which is what the engine's time windows and event times take.
///
/// Accepted forms, in the order they are tried:
/// - relative age — `30m`, `24h`, `7d`, `2w`: that long *before now*
/// - raw epoch microseconds — a bare integer, for programmatic callers
/// - RFC 3339 timestamp — `2026-07-01T14:30:00Z`
/// - plain date — `2026-07-01`, read as midnight UTC
fn parse_time_us(s: &str) -> Result<i64, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty time value".into());
    }

    // Relative age: digits followed by a unit suffix. Parsed unsigned, so a
    // negative span is rejected rather than resolving into the future.
    if let Some(split) = t.find(|c: char| c.is_ascii_alphabetic()) {
        if split > 0 {
            if let Ok(n) = t[..split].parse::<u64>() {
                let unit = t[split..].to_ascii_lowercase();
                let secs: u64 = match unit.as_str() {
                    "s" | "sec" | "secs" | "second" | "seconds" => 1,
                    "m" | "min" | "mins" | "minute" | "minutes" => 60,
                    "h" | "hr" | "hrs" | "hour" | "hours" => 3_600,
                    "d" | "day" | "days" => 86_400,
                    "w" | "wk" | "wks" | "week" | "weeks" => 604_800,
                    other => {
                        return Err(format!(
                            "unknown time unit '{other}' — use s, m, h, d, or w"
                        ))
                    }
                };
                let total = n.saturating_mul(secs).min(i64::MAX as u64) as i64;
                let delta = chrono::Duration::try_seconds(total)
                    .ok_or_else(|| format!("time span '{t}' is out of range"))?;
                let then = chrono::Utc::now() - delta;
                return Ok(then.timestamp_micros());
            }
        }
    }

    // Raw epoch microseconds.
    if t.chars().all(|c| c.is_ascii_digit()) {
        return t
            .parse::<i64>()
            .map_err(|_| format!("'{t}' is not a valid epoch-microsecond value"));
    }

    // Absolute timestamps.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(t) {
        return Ok(dt.timestamp_micros());
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(t, "%Y-%m-%d") {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Ok(dt.and_utc().timestamp_micros());
        }
    }

    Err(format!(
        "unrecognized time '{t}' — use a relative age (7d), a date (2026-07-01), \
         an RFC 3339 timestamp, or epoch microseconds"
    ))
}

/// Parse an optional time argument, reporting a bad one as invalid params.
fn opt_time(s: Option<&str>) -> Result<Option<i64>, McpError> {
    s.map(parse_time_us).transpose().map_err(|e| McpError::invalid_params(e, None))
}

/// Parse the caller's `order` string into the engine's presentation order.
fn parse_order(s: Option<&str>) -> Result<seraph::TemporalOrder, String> {
    match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("relevance") | Some("score") => Ok(seraph::TemporalOrder::Relevance),
        Some("most_recent") | Some("recent") | Some("newest") => {
            Ok(seraph::TemporalOrder::MostRecent)
        }
        Some("oldest") | Some("earliest") => Ok(seraph::TemporalOrder::Oldest),
        Some(other) => Err(format!(
            "unknown order '{other}' — use relevance, most_recent, or oldest"
        )),
    }
}

// --------------------------------------------------------------------------- //
// request shapes
// --------------------------------------------------------------------------- //

/// Schema override for `metadata`. schemars renders a bare `Option<Value>`
/// with NO `type` in the published schema, and at least one MCP client
/// (Claude Code) serializes arguments for typeless properties as JSON
/// *strings* — which the engine FFI silently parsed to an empty map. Publish
/// a typed object so schema-driven clients send a real one, while
/// deserialization stays lenient (`Value`): `Engine::remember` normalizes a
/// stringified object rather than erroring on it.
fn metadata_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["object", "null"],
        "additionalProperties": true,
        "description": "Optional metadata object stored alongside the frame."
    })
}

/// Schema for `tau`: a number or a level name. Typed for the same reason as
/// `metadata` — a typeless property reaches the server as a string.
fn tau_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["number", "string", "null"],
        "description": "Minimum score of a returned memory: a number 0.0-1.0, or a level of this \
            store's own background similarity (how alike unrelated memories are) — \"loose\" \
            (p75, the default), \"background\" (p95) or \"strict\" (p99). Levels move with the \
            store as it grows. Raising it trims the list; it never hides a better match. An empty \
            result means nothing in memory stands out for the query."
    })
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberReq {
    #[schemars(description = "Text content to commit to memory.")]
    pub content: String,
    #[schemars(schema_with = "metadata_schema")]
    pub metadata: Option<Value>,
    #[schemars(description = "When the content happened, if that is not now — the date of a \
        meeting, a decision, an event. Same formats as recall's `after`. The time of writing is \
        recorded regardless; this is a separate claim, kept with the frame.")]
    pub occurred_at: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BatchItemReq {
    #[schemars(description = "Text content to commit.")]
    pub content: String,
    #[schemars(schema_with = "metadata_schema")]
    pub metadata: Option<Value>,
    #[schemars(description = "When the content happened, if that is not now.")]
    pub occurred_at: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberBatchReq {
    #[schemars(description = "Texts to commit, without metadata. Pass this or `items`.")]
    pub contents: Option<Vec<String>>,
    #[schemars(description = "Frames to commit, each with its own metadata and event time. Pass \
        this or `contents`.")]
    pub items: Option<Vec<BatchItemReq>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct NameReq {
    #[schemars(description = "Seed name (also used as the embedding text). Seeds are eigenframes from birth and anchor a domain.")]
    pub name: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FrameIdReq {
    #[schemars(description = "Frame ID.")]
    pub frame_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallReq {
    #[schemars(description = "Search query text.")]
    pub query: String,
    #[schemars(description = "Max results (default 10).")]
    pub top_k: Option<u32>,
    #[schemars(schema_with = "tau_schema")]
    pub tau: Option<Value>,
    #[schemars(description = "Added to the resolved threshold (default 0): -0.05 widens a vague \
        query, +0.05 keeps only close matches.")]
    pub tau_offset: Option<f32>,
    #[schemars(description = "Include soft-forgotten (superseded) frames in results (default false). \
        Off keeps recall clean; on is for audit — retrieving what was superseded.")]
    pub include_superseded: Option<bool>,
    #[schemars(description = "Order of the returned hits: \"relevance\" (default, by meaning), \
        \"most_recent\" (newest first), or \"oldest\". This sorts the SAME relevant hits by write \
        time — it does not search for newer material, so a recent frame that was not relevant \
        enough to make top_k will not appear. Raise top_k for more time depth, or use `after` to \
        scope the search to a window.")]
    pub order: Option<String>,
    #[schemars(description = "Only return frames written at or after this time. Accepts a \
        relative age (\"30m\", \"24h\", \"7d\", \"2w\" — that long ago), a date (\"2026-07-01\", \
        UTC midnight), an RFC 3339 timestamp, or raw epoch microseconds. Use this for \
        \"what changed since ...\" questions.")]
    pub after: Option<String>,
    #[schemars(description = "Only return frames written at or before this time. Same formats as \
        `after`. Also scopes the search to the memory as it stood then.")]
    pub before: Option<String>,
    #[schemars(description = "Only return frames whose content happened at or after this time — \
        the `occurred_at` given when they were remembered, or their write time if none was given. \
        Same formats as `after`.")]
    pub occurred_after: Option<String>,
    #[schemars(description = "Only return frames whose content happened at or before this time. \
        Same formats as `after`.")]
    pub occurred_before: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChainReq {
    #[schemars(description = "Starting frame ID.")]
    pub from_id: String,
    #[schemars(description = "Target frame ID.")]
    pub to_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct QueryReq {
    #[schemars(description = "Query text.")]
    pub query: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct TwoQueryReq {
    #[schemars(description = "First concept query.")]
    pub query_a: String,
    #[schemars(description = "Second concept query.")]
    pub query_b: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct NeighborhoodReq {
    #[schemars(description = "Center frame ID.")]
    pub frame_id: String,
    #[schemars(description = "Similarity threshold (omit to auto-resolve p95).")]
    pub tau: Option<f32>,
    #[schemars(description = "Parent-tree hop distance (default 2).")]
    pub hops: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StrandReq {
    #[schemars(description = "Frame ID.")]
    pub frame_id: String,
    #[schemars(description = "Similarity floor for each step (omit to stop at the frame's own \
        background similarity).")]
    pub tau: Option<f32>,
    #[schemars(description = "Text to steer the walk toward: of the strands, follow the one about \
        this (e.g. \"licensing\" for why a decision was made as far as licensing goes). A step may \
        not move against it.")]
    pub toward: Option<String>,
    #[schemars(description = "Strength of `toward`: a float or a named level (action_floor, \
        routing, sub_saturation, saturation). Default sub_saturation.")]
    pub magnitude: Option<Value>,
    #[schemars(description = "Read the record as it stood at this time: frames written later are \
        not steps. Same formats as recall's `after`.")]
    pub as_of: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct HistoryReq {
    #[schemars(description = "The topic whose history to read.")]
    pub query: String,
    #[schemars(description = "Frames to return (default 10).")]
    pub top_k: Option<u32>,
    #[schemars(description = "How many of the topic's best search hits choose the trees the walk \
        reads (default 3). More anchors read a broader history.")]
    pub anchors: Option<u32>,
    #[schemars(description = "Text to steer toward: which strand of the history to read.")]
    pub toward: Option<String>,
    #[schemars(description = "Strength of `toward`: a float or a named level. Default sub_saturation.")]
    pub magnitude: Option<Value>,
    #[schemars(description = "Only frames written at or after this time. Same formats as recall.")]
    pub after: Option<String>,
    #[schemars(description = "Only frames written at or before this time; the walk reads the record \
        as it stood then.")]
    pub before: Option<String>,
    #[schemars(description = "Only frames whose content happened at or after this time.")]
    pub occurred_after: Option<String>,
    #[schemars(description = "Only frames whose content happened at or before this time.")]
    pub occurred_before: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LedToReq {
    #[schemars(description = "The earlier idea: text, or a frame ID.")]
    pub from: String,
    #[schemars(description = "The later idea: text, or a frame ID.")]
    pub to: String,
    #[schemars(description = "Read the record as it stood at this time. Same formats as recall's `after`.")]
    pub as_of: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ConnectionsReq {
    #[schemars(description = "One end: text (resolved to its best-matching memory), or a frame ID.")]
    pub from: String,
    #[schemars(description = "The other end: text, or a frame ID.")]
    pub to: String,
    #[schemars(description = "Most routes to return (default 4).")]
    pub max_routes: Option<u32>,
    #[schemars(description = "Only routes that run forward in the order memories were written, \
        from the earlier end to the later (default false).")]
    pub forward: Option<bool>,
    #[schemars(description = "Read the record as it stood at this time. Same formats as recall's `after`.")]
    pub as_of: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecentReq {
    #[schemars(description = "Only memories since this time (\"24h\", \"7d\", a date, an RFC 3339 \
        timestamp); the result then says how many there are in all. Omit for the latest `limit`.")]
    pub since: Option<String>,
    #[schemars(description = "Most memories to return (default 20).")]
    pub limit: Option<u32>,
    #[schemars(description = "\"written\" (default): by when memories were written — what was \
        learned lately. \"occurred\": by when their content happened. A store migrated from an \
        older format counts every earlier memory as written at the migration; read those by \
        \"occurred\".")]
    pub by: Option<String>,
    #[schemars(description = "Include soft-forgotten (superseded) memories (default false).")]
    pub include_superseded: Option<bool>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WarpPointReq {
    #[schemars(description = "Text to encode as a warp direction (or an existing frame's text).")]
    pub input: String,
    #[schemars(description = "Magnitude: a float, or a named level resolved from this store's \
        calibration: action_floor, routing, sub_saturation, saturation. Default routing.")]
    pub magnitude: Option<Value>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallWarpReq {
    #[schemars(description = "Search query text.")]
    pub query: String,
    #[schemars(description = "Warp targets to pull the query toward.")]
    pub targets: Option<Vec<WarpPointReq>>,
    #[schemars(description = "Warp suppressions to push the query away from.")]
    pub suppress: Option<Vec<WarpPointReq>>,
    #[schemars(description = "Profile: bounded (default), routing, or saturation.")]
    pub profile: Option<String>,
    #[schemars(description = "Bound parameter (default 1.5).")]
    pub bound_k: Option<f32>,
    #[schemars(description = "Max results (default 10).")]
    pub top_k: Option<u32>,
    #[schemars(schema_with = "tau_schema")]
    pub tau: Option<Value>,
    #[schemars(description = "Added to the resolved threshold (default 0).")]
    pub tau_offset: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WarpChainToReq {
    pub query: String,
    pub targets: Option<Vec<WarpPointReq>>,
    pub suppress: Option<Vec<WarpPointReq>>,
    #[schemars(description = "Profile: bounded (default), routing, or saturation.")]
    pub profile: Option<String>,
    pub bound_k: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WarpChainBetweenReq {
    pub query_a: String,
    pub query_b: String,
    pub targets_a: Option<Vec<WarpPointReq>>,
    pub targets_b: Option<Vec<WarpPointReq>>,
    #[schemars(description = "Profile: bounded (default), routing, or saturation.")]
    pub profile: Option<String>,
    pub bound_k: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SteerReq {
    #[schemars(description = "Query text.")]
    pub query: String,
    #[schemars(description = "Target frame ID to steer toward.")]
    pub target_id: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CalibrateReq {
    #[schemars(description = "Number of probe queries for the sweep (default 30).")]
    pub n_queries: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeclareRelationReq {
    #[schemars(description = "Relation name, e.g. implements, supersedes, supports, refutes, derived_from.")]
    pub name: String,
    #[schemars(description = "Flag bitfield: Active=1, Deprecated=2, Bidirectional=4, AsymmetricInverse=8. Default 1 (Active).")]
    pub flags: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AssertRelationReq {
    pub from_id: String,
    pub to_id: String,
    #[schemars(description = "Relation name (must be declared first).")]
    pub relation: String,
    #[schemars(description = "Edge weight (default 1.0).")]
    pub weight: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RetractRelationReq {
    pub from_id: String,
    pub to_id: String,
    pub relation: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StructuralNeighborsReq {
    pub frame_id: String,
    #[schemars(description = "Relation names to traverse (omit for all).")]
    pub relations: Option<Vec<String>>,
    #[schemars(description = "Direction: outgoing (default), incoming, or both.")]
    pub direction: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ContradictionsReq {
    #[schemars(description = "Minimum content divergence to flag (default 1.2).")]
    pub min_divergence: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DriftReq {
    #[schemars(description = "Trailing window of children to assess (default 20).")]
    pub window: Option<u32>,
    #[schemars(description = "Drift threshold (default 0.3).")]
    pub threshold: Option<f32>,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn to_points(v: Option<Vec<WarpPointReq>>) -> Result<Vec<store::WarpPoint>, McpError> {
    v.unwrap_or_default()
        .into_iter()
        .map(|p| {
            let magnitude = parse_magnitude(p.magnitude.as_ref())
                .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
            Ok(store::WarpPoint { text: p.input, magnitude })
        })
        .collect()
}

fn to_profile(s: Option<&str>) -> Result<seraph::WarpProfile, McpError> {
    parse_profile(s).map_err(|e| McpError::invalid_params(e.to_string(), None))
}

fn to_tau(tau: Option<&Value>, offset: Option<f32>) -> Result<TauArg, McpError> {
    let level = parse_tau(tau).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
    Ok(TauArg { level, offset: offset.unwrap_or(0.0) })
}

/// A `toward` steer as a warp point, at `sub_saturation` unless a magnitude is given.
fn to_toward(text: Option<String>, magnitude: Option<&Value>) -> Result<Option<WarpPoint>, McpError> {
    let Some(text) = text.filter(|t| !t.trim().is_empty()) else { return Ok(None) };
    let magnitude = match magnitude {
        None | Some(Value::Null) => seraph::Magnitude::Named("sub_saturation".into()),
        m => parse_magnitude(m).map_err(|e| McpError::invalid_params(e.to_string(), None))?,
    };
    Ok(Some(WarpPoint { text, magnitude }))
}

fn parse_recent_by(s: Option<&str>) -> Result<RecentBy, String> {
    match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("written") | Some("write") => Ok(RecentBy::Written),
        Some("occurred") | Some("happened") | Some("event") => Ok(RecentBy::Occurred),
        Some(other) => Err(format!("unknown by '{other}' — use written or occurred")),
    }
}

fn ok_json(v: Value) -> Result<CallToolResult, McpError> {
    let s = serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string());
    Ok(CallToolResult::success(vec![Content::text(s)]))
}

// --------------------------------------------------------------------------- //
// server
// --------------------------------------------------------------------------- //

/// The engine behind the tools: open, or closed after an idle spell (hosted
/// mode) and reopened by the next call that needs it.
pub struct EngineSlot {
    engine: Option<Engine>,
    path: PathBuf,
    last_used: std::time::Instant,
}

impl EngineSlot {
    /// The open engine, reopening the store if it was closed.
    fn get(&mut self) -> anyhow::Result<&mut Engine> {
        if self.engine.is_none() {
            self.engine = Some(open_engine(&self.path)?);
            tracing::info!("store reopened");
        }
        Ok(self.engine.as_mut().expect("engine just opened"))
    }
}

#[derive(Clone)]
pub struct SeraphMemory {
    engine: Arc<Mutex<EngineSlot>>,
    store_path: String,
    #[allow(dead_code)]
    tool_router: ToolRouter<SeraphMemory>,
}

impl SeraphMemory {
    fn new(engine: Engine, store_path: String) -> Self {
        let slot = EngineSlot {
            engine: Some(engine),
            path: PathBuf::from(&store_path),
            last_used: std::time::Instant::now(),
        };
        Self {
            engine: Arc::new(Mutex::new(slot)),
            store_path,
            tool_router: Self::tool_router(),
        }
    }

    /// Run a blocking engine operation off the async runtime, under the mutex.
    async fn run<F>(&self, f: F) -> Result<CallToolResult, McpError>
    where
        F: FnOnce(&mut Engine) -> anyhow::Result<Value> + Send + 'static,
    {
        let engine = self.engine.clone();
        let res = tokio::task::spawn_blocking(move || {
            let mut guard = engine
                .lock()
                .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?;
            let result = f(guard.get()?);
            guard.last_used = std::time::Instant::now();
            result
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task join error: {e}"), None))?;

        match res {
            Ok(v) => ok_json(v),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }
}

#[tool_router]
impl SeraphMemory {
    // ---- write ----
    #[tool(description = "Commit a piece of text to memory. The store places it geometrically by \
        similarity to everything already present and returns its frame ID. Give occurred_at when \
        the content describes something that happened at another time than now.")]
    async fn remember(&self, Parameters(r): Parameters<RememberReq>) -> Result<CallToolResult, McpError> {
        let occurred_at = opt_time(r.occurred_at.as_deref())?;
        self.run(move |e| e.remember(&r.content, r.metadata.as_ref(), occurred_at)).await
    }

    #[tool(description = "Commit many texts at once (batched encode, one GPU pass), in the order \
        given. Pass `contents` for plain texts or `items` for texts with their own metadata and \
        occurred_at. Returns frame IDs.")]
    async fn remember_batch(&self, Parameters(r): Parameters<RememberBatchReq>) -> Result<CallToolResult, McpError> {
        let items: Vec<BatchItem> = match (r.contents, r.items) {
            (Some(contents), None) => contents
                .into_iter()
                .map(|content| BatchItem { content, metadata: None, occurred_at: None })
                .collect(),
            (None, Some(items)) => items
                .into_iter()
                .map(|i| {
                    let occurred_at = opt_time(i.occurred_at.as_deref())?;
                    Ok(BatchItem { content: i.content, metadata: i.metadata, occurred_at })
                })
                .collect::<Result<_, McpError>>()?,
            _ => return Err(McpError::invalid_params("pass either `contents` or `items`", None)),
        };
        self.run(move |e| e.remember_batch(items)).await
    }

    #[tool(description = "Add a domain seed — an eigenframe from birth that anchors a region of the \
        store. Use a few seeds to partition broad domains before ingesting.")]
    async fn add_seed(&self, Parameters(r): Parameters<NameReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.add_seed(&r.name)).await
    }

    #[tool(description = "Promote a frame to eigenframe status unconditionally (make it a landmark).")]
    async fn promote_frame(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.promote(&r.frame_id)).await
    }

    // ---- recall ----
    #[tool(description = "Semantic recall: search memory by meaning. Returns ranked frames with \
        score, confidence, and a content snippet. Set include_superseded=true to also retrieve \
        soft-forgotten frames (audit). Temporal options: order=\"most_recent\" returns the same \
        relevant hits sorted newest-first (it reorders, it does not search further back in \
        relevance — raise top_k for more depth); after/before scope which frames may occupy a \
        result slot, taking a relative age (\"7d\"), a date (\"2026-07-01\"), or an RFC 3339 \
        timestamp — use after for \"what changed since ...\"; before also reads the memory as it \
        stood at that time. occurred_after/occurred_before scope by when the content happened \
        (the occurred_at given at remember time, else the write time). When any time option is \
        set, each hit carries its write timestamp; a hit whose frame records occurred_at always \
        carries it.")]
    async fn recall(&self, Parameters(r): Parameters<RecallReq>) -> Result<CallToolResult, McpError> {
        let top_k = r.top_k.unwrap_or(10) as usize;
        let tau = to_tau(r.tau.as_ref(), r.tau_offset)?;
        let include_superseded = r.include_superseded.unwrap_or(false);
        let order = parse_order(r.order.as_deref())
            .map_err(|e| McpError::invalid_params(e, None))?;
        let q = RecallQuery {
            query: r.query,
            top_k,
            tau,
            include_superseded,
            order,
            written: seraph::TimeWindow {
                after_us: opt_time(r.after.as_deref())?,
                before_us: opt_time(r.before.as_deref())?,
            },
            occurred: seraph::TimeWindow {
                after_us: opt_time(r.occurred_after.as_deref())?,
                before_us: opt_time(r.occurred_before.as_deref())?,
            },
        };
        self.run(move |e| e.recall(q)).await
    }

    #[tool(description = "Soft-forget a frame: mark it superseded so it stops surfacing in recall \
        and reasoning chains, while its structural chain link is preserved (not deleted). Use for \
        stale/corrected knowledge you don't want polluting results. Retrieve later with \
        recall(include_superseded=true).")]
    async fn supersede(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.supersede(&r.frame_id)).await
    }

    #[tool(description = "Fetch a single frame by ID (content, status, parent, metadata, in-degree).")]
    async fn get_frame(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.get_frame(&r.frame_id)).await
    }

    #[tool(description = "A frame's history in the record: when it was written, when (and how) it \
        became a landmark, when (and how) it was retired by supersede or consolidation, and the \
        occurred_at it claims, if any.")]
    async fn lifecycle(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.lifecycle(&r.frame_id)).await
    }

    #[tool(description = "List eigenframes — the store's emergent table of contents (landmark regions).")]
    async fn eigenframes_list(&self) -> Result<CallToolResult, McpError> {
        self.run(|e| e.eigenframes()).await
    }

    // ---- chains ----
    #[tool(description = "Reasoning chain between two frames: parent-tree walk if same subtree, \
        similarity walk otherwise. Returns the chain plus coherence metrics.")]
    async fn chain(&self, Parameters(r): Parameters<ChainReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.chain(&r.from_id, &r.to_id)).await
    }

    #[tool(description = "Reasoning chain from the store's geometry to the knowledge matching a query.")]
    async fn chain_to(&self, Parameters(r): Parameters<QueryReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.chain_to(&r.query)).await
    }

    #[tool(description = "Reasoning chain between two concepts — how concept A relates to B through \
        the store's geometry. The intermediate frames are the explanation.")]
    async fn chain_between(&self, Parameters(r): Parameters<TwoQueryReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.chain_between(&r.query_a, &r.query_b)).await
    }

    // ---- neighborhood / provenance ----
    // Populations: consensus = cosine AND lineage agree (solid neighbors); geometric = cosine-only
    // (cross-domain analogies OR noise); lineage = subtree-only (structural kin, drift candidates).
    // TAU CAVEAT (verified 2026-06-25): omitting tau auto-resolves to ~p95 of the global similarity
    // distribution (~0.875 in this store, vs mean 0.76) — stricter than almost every pair, so EVERY
    // frame tends to come back consensus:[] geometric:[] lineage:[subtree]. That looks like geometric
    // isolation but is just the threshold. Pass an explicit tau (~0.6) to actually populate geometric/
    // consensus. CAUTION: geometric-only neighbors at low tau on this small encoder (bge-small) are
    // frequently SURFACE/REGISTER collisions (shared technical vocabulary), not true relational
    // analogies — for genuine cross-domain bridging use recall_with_warp, not raw neighborhood geometry.
    #[tool(description = "Three-population neighborhood of a frame: consensus (cosine+lineage agree), \
        geometric (cosine-only — analogies/noise), lineage (subtree-only — drift). Divergence is a signal. \
        Omit tau and it auto-resolves to ~p95 (very strict) — most frames then return lineage-only; \
        pass an explicit tau (~0.6) to surface geometric/consensus neighbors.")]
    async fn neighborhood(&self, Parameters(r): Parameters<NeighborhoodReq>) -> Result<CallToolResult, McpError> {
        let hops = r.hops.unwrap_or(2) as usize;
        self.run(move |e| e.neighborhood(&r.frame_id, r.tau, hops)).await
    }

    #[tool(description = "Walk a frame's ingestion parent-tree back to genesis — the audit trail of where \
        it came from structurally. Each parent is the landmark the frame attached to at commit time (the \
        nearest region), NOT the previous commit — so chains are short (a handful of hops) and may cross \
        domains. Works from ANY frame. High geometric distance along a real lineage edge is the divergence signal.")]
    async fn lineage(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.lineage(&r.frame_id)).await
    }

    #[tool(description = "Provenance: what a frame was built on — why a conclusion was reached. \
        Follows the frames it connected to when it was written, back through earlier frames, and \
        returns the strand (each frame with its snippet and times) with coherence metrics. Unlike \
        lineage (the parent tree), this follows similarity links too. `toward` steers which strand \
        is followed; `as_of` reads the record as it stood then.")]
    async fn provenance(&self, Parameters(r): Parameters<StrandReq>) -> Result<CallToolResult, McpError> {
        let toward = to_toward(r.toward, r.magnitude.as_ref())?;
        let as_of = opt_time(r.as_of.as_deref())?;
        self.run(move |e| e.strand(&r.frame_id, false, r.tau, toward.as_ref(), as_of)).await
    }

    #[tool(description = "Influence: what was later built on a frame — what a finding went on to \
        shape. Follows the frames written after it that connected to it, forward in time. The \
        counterpart of provenance; takes the same `toward` and `as_of`.")]
    async fn influence(&self, Parameters(r): Parameters<StrandReq>) -> Result<CallToolResult, McpError> {
        let toward = to_toward(r.toward, r.magnitude.as_ref())?;
        let as_of = opt_time(r.as_of.as_deref())?;
        self.run(move |e| e.strand(&r.frame_id, true, r.tau, toward.as_ref(), as_of)).await
    }

    // ---- time reads ----
    #[tool(description = "The backstory of a topic, in order: what memory holds about it, oldest \
        first (by when the content happened, else when it was written). The topic's best search \
        hits choose the trees a walk reads, so this returns the arc that led to where things \
        stand rather than a relevance-ranked list. Use it to catch up on a topic before working \
        on it. `toward` steers which strand of the history is read; time windows work as in recall.")]
    async fn history(&self, Parameters(r): Parameters<HistoryReq>) -> Result<CallToolResult, McpError> {
        let q = HistoryQuery {
            query: r.query,
            top_k: r.top_k.unwrap_or(10).max(1) as usize,
            anchors: r.anchors.unwrap_or(3).max(1) as usize,
            toward: to_toward(r.toward, r.magnitude.as_ref())?,
            written: seraph::TimeWindow {
                after_us: opt_time(r.after.as_deref())?,
                before_us: opt_time(r.before.as_deref())?,
            },
            occurred: seraph::TimeWindow {
                after_us: opt_time(r.occurred_after.as_deref())?,
                before_us: opt_time(r.occurred_before.as_deref())?,
            },
        };
        self.run(move |e| e.history(q)).await
    }

    #[tool(description = "What memory has gained lately, with no query: the newest memories, newest \
        first, each with its snippet and times. `since` limits it to memories since a time and \
        reports how many there are in all; `by=\"occurred\"` reads by when content happened instead \
        of when it was written. Use it at the start of a session to see what other sessions have \
        learned.")]
    async fn recent(&self, Parameters(r): Parameters<RecentReq>) -> Result<CallToolResult, McpError> {
        let since = opt_time(r.since.as_deref())?;
        let limit = r.limit.unwrap_or(20).max(1) as usize;
        let by = parse_recent_by(r.by.as_deref()).map_err(|e| McpError::invalid_params(e, None))?;
        let include_superseded = r.include_superseded.unwrap_or(false);
        self.run(move |e| e.recent(since, limit, by, include_superseded)).await
    }

    #[tool(description = "How A led to B: a route forward in time, each step a memory written after \
        the one before and connected to it when it was written. Use it to trace how a decision or \
        finding came about from an earlier one. Each end is text (the memories that best fit it) \
        or a frame ID. reached=false means the record holds no forward route. Time here is the \
        order memories were written, not when their content happened.")]
    async fn led_to(&self, Parameters(r): Parameters<LedToReq>) -> Result<CallToolResult, McpError> {
        let as_of = opt_time(r.as_of.as_deref())?;
        self.run(move |e| e.led_to(&r.from, &r.to, as_of)).await
    }

    #[tool(description = "The distinct ways two things relate: up to max_routes routes between them \
        that share no intermediate memory, shortest first, each with its bridge — the memory \
        closest to both ends. Each end is text (resolved to its best-matching memory) or a frame \
        ID. forward=true keeps only routes that run forward in the order memories were written.")]
    async fn connections(&self, Parameters(r): Parameters<ConnectionsReq>) -> Result<CallToolResult, McpError> {
        let as_of = opt_time(r.as_of.as_deref())?;
        let max_routes = r.max_routes.unwrap_or(4).max(1) as usize;
        let forward = r.forward.unwrap_or(false);
        self.run(move |e| e.connections(&r.from, &r.to, max_routes, forward, as_of)).await
    }

    // The forward counterpart to `lineage`: it descends the same ingestion tree from the seed's region.
    // EMPIRICAL CAVEAT (engine seraph.dll, FFI shape undocumented/PLACEHOLDER): in practice this only
    // resolves from the GENESIS / region-root seed. Content frames — and even eigenframes — that
    // provably have children in the tree (per `lineage`) still return []. similarity_to_seed is also
    // degenerate (0.0) for a genesis seed, which carries no semantic vector. For "where did this concept
    // come from", prefer `lineage` (works from any frame).
    #[tool(description = "Trace a region's growth forward through the ingestion tree from a seed frame (inverse \
        of `lineage`). NOTE: empirically resolves only from the genesis/region-root seed; most content frames \
        return [] even when they have children. Use `lineage` for per-concept provenance.")]
    async fn evolution(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.evolution(&r.frame_id)).await
    }

    // ---- warp ----
    #[tool(description = "Warped recall — reshape the query vector before retrieval to bias toward \
        targets and away from suppressions. Enables cross-domain bridging and anti-repetition. \
        Named magnitudes resolve against this store's own calibration.")]
    async fn recall_with_warp(&self, Parameters(r): Parameters<RecallWarpReq>) -> Result<CallToolResult, McpError> {
        let targets = to_points(r.targets)?;
        let suppress = to_points(r.suppress)?;
        let profile = to_profile(r.profile.as_deref())?;
        let bound_k = r.bound_k.unwrap_or(1.5);
        let top_k = r.top_k.unwrap_or(10) as usize;
        let tau = to_tau(r.tau.as_ref(), r.tau_offset)?;
        let query = r.query;
        self.run(move |e| e.recall_with_warp(&query, &targets, &suppress, profile, bound_k, top_k, tau))
            .await
    }

    #[tool(description = "Warped chain_to — chain to the knowledge matching a query after steering \
        the query vector toward targets / away from suppressions.")]
    async fn warp_chain_to(&self, Parameters(r): Parameters<WarpChainToReq>) -> Result<CallToolResult, McpError> {
        let targets = to_points(r.targets)?;
        let suppress = to_points(r.suppress)?;
        let profile = to_profile(r.profile.as_deref())?;
        let bound_k = r.bound_k.unwrap_or(1.5);
        let query = r.query;
        self.run(move |e| e.warp_chain_to(&query, &targets, &suppress, profile, bound_k)).await
    }

    #[tool(description = "Warped chain_between — chain between two concepts with per-endpoint warp steering.")]
    async fn warp_chain_between(&self, Parameters(r): Parameters<WarpChainBetweenReq>) -> Result<CallToolResult, McpError> {
        let ta = to_points(r.targets_a)?;
        let tb = to_points(r.targets_b)?;
        let profile = to_profile(r.profile.as_deref())?;
        let bound_k = r.bound_k.unwrap_or(1.5);
        let qa = r.query_a;
        let qb = r.query_b;
        self.run(move |e| e.warp_chain_between(&qa, &qb, &ta, &tb, profile, bound_k)).await
    }

    #[tool(description = "Seed margin: geometric distinctness of a frame vs the corpus centroid. \
        Higher margin = stronger warp effect per unit magnitude.")]
    async fn warp_seed_margin(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.seed_margin(&r.frame_id)).await
    }

    #[tool(description = "Steerability: predicted warp displacement of a query toward a target frame. \
        Near 0 means structurally unsteerable — use multi-seed composition instead of more magnitude.")]
    async fn warp_steerability(&self, Parameters(r): Parameters<SteerReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.steerability(&r.query, &r.target_id)).await
    }

    #[tool(description = "Calibrate warp magnitudes against this store's actual corpus and encoder \
        (dimension-correct regime boundaries). The store recalibrates itself on load when its \
        corpus has moved; run this to recalibrate now, or to inspect the levels.")]
    async fn warp_calibration(&self, Parameters(r): Parameters<CalibrateReq>) -> Result<CallToolResult, McpError> {
        let n = r.n_queries.unwrap_or(30) as usize;
        self.run(move |e| e.calibrate_warp(n)).await
    }

    // ---- declared structure ----
    #[tool(description = "Declare a typed relation (e.g. implements, supersedes, supports, refutes, \
        derived_from). Declared edges overlay the geometry without polluting it. Returns the slot.")]
    async fn declare_relation(&self, Parameters(r): Parameters<DeclareRelationReq>) -> Result<CallToolResult, McpError> {
        let flags = r.flags.unwrap_or(1);
        self.run(move |e| e.declare_relation(&r.name, flags)).await
    }

    #[tool(description = "List all declared relation types and their flags/state.")]
    async fn list_relations(&self) -> Result<CallToolResult, McpError> {
        self.run(|e| e.list_relations()).await
    }

    #[tool(description = "Assert a typed edge between two frames along a declared relation.")]
    async fn assert_relation(&self, Parameters(r): Parameters<AssertRelationReq>) -> Result<CallToolResult, McpError> {
        let weight = r.weight.unwrap_or(1.0);
        self.run(move |e| e.assert_relation(&r.from_id, &r.to_id, &r.relation, weight)).await
    }

    #[tool(description = "Retract a typed edge (recorded as an event, not silently deleted).")]
    async fn retract_relation(&self, Parameters(r): Parameters<RetractRelationReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.retract_relation(&r.from_id, &r.to_id, &r.relation)).await
    }

    #[tool(description = "Traverse typed edges from a frame: forward/backward along declared relations.")]
    async fn structural_neighbors(&self, Parameters(r): Parameters<StructuralNeighborsReq>) -> Result<CallToolResult, McpError> {
        let dir = r.direction.unwrap_or_else(|| "outgoing".into());
        self.run(move |e| e.structural_neighbors(&r.frame_id, r.relations.as_deref(), &dir)).await
    }

    // ---- diagnostics ----
    #[tool(description = "Store statistics: frame/eigenframe/superseded counts, model, and the global \
        similarity distribution.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        self.run(|e| e.stats()).await
    }

    // min_divergence is a content-divergence floor for pairs that are ALSO geometrically close.
    // Calibration (verified 2026-06-25): the default 1.2 is high — a consistent store returns [] (no
    // genuine close-but-divergent pairs), which is the healthy/expected result, not a failure. Lowering
    // the floor toward ~0.3 floods the output with thousands of ambient neighbor pairs (noise), NOT
    // newly-discovered contradictions. Treat [] at default as "clean"; only lower the floor to probe a
    // specific suspected tension, and expect noise once you go much below ~0.85.
    #[tool(description = "Detect contradictions — frames close in geometry but divergent in content. \
        min_divergence (default 1.2) is a high floor: [] means the store is consistent (expected). \
        Lowering it much below ~0.85 floods with ambient noise, not real contradictions.")]
    async fn contradictions(&self, Parameters(r): Parameters<ContradictionsReq>) -> Result<CallToolResult, McpError> {
        let min_div = r.min_divergence.unwrap_or(1.2);
        self.run(move |e| e.contradictions(min_div)).await
    }

    #[tool(description = "Local density score for a frame (tighter cluster = higher).")]
    async fn density(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.density(&r.frame_id)).await
    }

    #[tool(description = "Health of a concept region around an eigenframe: stable, consolidating, or fragmenting.")]
    async fn region_health(&self, Parameters(r): Parameters<FrameIdReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.region_health(&r.frame_id)).await
    }

    #[tool(description = "Scan eigenframes for semantic drift away from their children.")]
    async fn drift_report(&self, Parameters(r): Parameters<DriftReq>) -> Result<CallToolResult, McpError> {
        let window = r.window.unwrap_or(20) as usize;
        let threshold = r.threshold.unwrap_or(0.3);
        self.run(move |e| e.drift_report(window, threshold)).await
    }

    #[tool(description = "Verify watermark-chain integrity across all frames (tamper evidence).")]
    async fn verify(&self) -> Result<CallToolResult, McpError> {
        self.run(|e| e.verify()).await
    }
}

#[tool_handler]
impl ServerHandler for SeraphMemory {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_instructions(format!(
            "SERAPH memory substrate (Rust, over the C FFI). A self-organizing geometric knowledge \
             store with structural provenance, reasoning chains, and contradiction detection. One \
             global store is open at {}. Use `remember` to commit knowledge and `recall` to search \
             by meaning; `chain`/`chain_between` to explain how things connect; `neighborhood` for \
             three-population context; declared relations (`declare_relation`/`assert_relation`/\
             `structural_neighbors`) for exact structure; `provenance`/`influence` for what a \
             frame was built on and what was built on it; `lifecycle` for a frame's history. \
             TIME: `remember` takes occurred_at for content about another time; `recall` scopes \
             by write time (after/before) or by when content happened (occurred_after/\
             occurred_before). `recent` lists what memory gained lately with no query — use it \
             to catch up at the start of a session; `history` reads a topic's backstory oldest \
             first; `led_to` traces how one idea led to a later one; `connections` lists the \
             distinct ways two things relate, each with its bridge. THRESHOLDS: tau defaults to \
             the store's own loose level, so an empty recall means nothing in memory stands out; \
             tau_offset shifts it. WARP: named magnitudes resolve against the store's own \
             calibration. Use `routing` magnitude for gentle steering, `saturation` only for \
             cross-domain bridging; keep profile=bounded.",
            self.store_path
        ))
    }
}

// --------------------------------------------------------------------------- //
// startup
// --------------------------------------------------------------------------- //

fn resolve_store_path() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--store") {
        if let Some(p) = args.get(i + 1) {
            return PathBuf::from(p);
        }
    }
    if let Ok(p) = std::env::var("SERAPH_MEMORY_STORE") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("seraph-memory").join("memory.sfg")
}

/// Open (or first-run create) the store, logging the result.
fn open_engine(path: &Path) -> anyhow::Result<Engine> {
    let engine = Engine::open_or_create(path, DEFAULT_MODEL)?;
    tracing::info!("store ready: model={}, dim={}", engine.model_id(), engine.dim());
    Ok(engine)
}

/// Open the store or print first-run guidance and exit.
fn open_engine_or_exit(path: &Path) -> Engine {
    match open_engine(path) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("failed to open/create store: {e}");
            eprintln!(
                "seraph-memory-rs: could not open/create the store at {}.\n\
                 If this is first run, store creation needs a SERAPH license at \
                 %APPDATA%\\SERAPH\\license.key (run `seraph license activate`).\n\
                 Underlying error: {e}",
                path.display()
            );
            std::process::exit(1);
        }
    }
}

/// Degraded path: open the store directly and serve this one session over stdio.
/// Used when no shared daemon can be reached/spawned (and on non-Windows).
async fn run_inprocess(path: &Path) -> anyhow::Result<()> {
    let engine = open_engine_or_exit(path);
    let service = SeraphMemory::new(engine, path.to_string_lossy().to_string())
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("serving error: {e:?}"))?;
    service.waiting().await?;
    Ok(())
}

// --------------------------------------------------------------------------- //
// shared-daemon bridge (Windows)
//
// SERAPH is single-writer by design, but Claude Code launches one stdio server
// process per session. To avoid N processes each opening the same `.sfg`, the
// first session spawns a detached **daemon** that owns the store; every session
// process then runs as a thin **client** that relays its stdio to the daemon
// over a named pipe. The daemon serves the same rmcp handler over each pipe
// connection, sharing one `Arc<Mutex<Engine>>`, so all sessions see one warm,
// consistent store and writes stay serialized. The daemon self-exits after an
// idle period (default 60 min, `SERAPH_MEMORY_IDLE_SECS`).
// --------------------------------------------------------------------------- //
#[cfg(windows)]
mod bridge {
    use super::*;
    use std::hash::{Hash, Hasher};
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient, ServerOptions};

    // CreateProcess flags: detach from the launching session, no console.
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    /// Pipe name unique to a store path so different stores get distinct daemons.
    /// The hash only needs to be stable across processes of *this* binary, which
    /// `DefaultHasher` satisfies within a single build.
    pub fn pipe_name_for(path: &Path) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        path.to_string_lossy().to_lowercase().hash(&mut h);
        format!(r"\\.\pipe\seraph-memory-{:016x}", h.finish())
    }

    fn idle_timeout() -> Duration {
        let secs = std::env::var("SERAPH_MEMORY_IDLE_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(3600);
        Duration::from_secs(secs)
    }

    /// A log sink for the detached daemon (its stderr is otherwise discarded).
    fn daemon_log() -> Stdio {
        if let Some(dir) = dirs::data_dir().map(|d| d.join("seraph-memory")) {
            let _ = std::fs::create_dir_all(&dir);
            if let Ok(f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("daemon.log"))
            {
                return Stdio::from(f);
            }
        }
        Stdio::null()
    }

    /// Daemon: own the store, accept session connections, serve rmcp over each.
    pub async fn run_daemon(path: &Path, pipe_name: &str) -> anyhow::Result<()> {
        // Single-instance guard: only one daemon may hold the first pipe instance.
        let mut server = match ServerOptions::new().first_pipe_instance(true).create(pipe_name) {
            Ok(s) => s,
            Err(e) => {
                tracing::info!("another daemon already owns {pipe_name} ({e}); exiting");
                return Ok(());
            }
        };

        let engine = open_engine_or_exit(path);
        let handler = SeraphMemory::new(engine, path.to_string_lossy().to_string());
        let idle = idle_timeout();
        let active = Arc::new(AtomicUsize::new(0));
        tracing::info!("daemon listening on {pipe_name} (idle timeout {idle:?})");

        loop {
            match tokio::time::timeout(idle, server.connect()).await {
                Ok(Ok(())) => {
                    // `server` is now connected to a client; hand it off and
                    // arm a fresh instance for the next session.
                    let next = ServerOptions::new().create(pipe_name)?;
                    let connected = std::mem::replace(&mut server, next);
                    active.fetch_add(1, Ordering::SeqCst);
                    let h = handler.clone();
                    let a = active.clone();
                    tokio::spawn(async move {
                        match h.serve(connected).await {
                            Ok(svc) => {
                                let _ = svc.waiting().await;
                            }
                            Err(e) => tracing::warn!("session serve error: {e:?}"),
                        }
                        a.fetch_sub(1, Ordering::SeqCst);
                    });
                }
                Ok(Err(e)) => tracing::warn!("pipe connect error: {e}"),
                Err(_elapsed) => {
                    if active.load(Ordering::SeqCst) == 0 {
                        tracing::info!("idle {idle:?} with no sessions; daemon exiting");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Client: connect to (or spawn) the daemon and relay this session's stdio.
    /// Falls back to in-process serving if the daemon cannot be reached.
    pub async fn run_client(path: &Path, pipe_name: &str) -> anyhow::Result<()> {
        match connect_or_spawn(path, pipe_name).await {
            Some(client) => {
                tracing::info!("connected to shared daemon; relaying session");
                relay(client).await
            }
            None => {
                tracing::warn!(
                    "could not reach a shared daemon; opening the store in-process \
                     (degraded — no cross-session sharing)"
                );
                run_inprocess(path).await
            }
        }
    }

    fn try_connect(pipe_name: &str) -> Option<NamedPipeClient> {
        ClientOptions::new().open(pipe_name).ok()
    }

    async fn connect_or_spawn(path: &Path, pipe_name: &str) -> Option<NamedPipeClient> {
        if let Some(c) = try_connect(pipe_name) {
            return Some(c);
        }
        spawn_daemon(path);
        // The daemon opens the store (~100ms) then starts listening; retry briefly.
        for _ in 0..150 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if let Some(c) = try_connect(pipe_name) {
                return Some(c);
            }
        }
        None
    }

    fn spawn_daemon(path: &Path) {
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("current_exe failed, cannot spawn daemon: {e}");
                return;
            }
        };
        let mut cmd = Command::new(exe);
        cmd.arg("--daemon")
            .arg("--store")
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(daemon_log())
            .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        match cmd.spawn() {
            Ok(_) => tracing::info!("spawned shared daemon"),
            Err(e) => tracing::error!("failed to spawn daemon: {e}"),
        }
    }

    /// Pump bytes both ways between this process's stdio and the daemon pipe.
    /// Either side closing (session ends, or daemon exits) tears the relay down.
    async fn relay(client: NamedPipeClient) -> anyhow::Result<()> {
        let (mut pipe_rd, mut pipe_wr) = tokio::io::split(client);
        let mut stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        tokio::select! {
            r = tokio::io::copy(&mut stdin, &mut pipe_wr) => {
                let _ = r;
                let _ = pipe_wr.shutdown().await;
            }
            r = tokio::io::copy(&mut pipe_rd, &mut stdout) => {
                let _ = r;
                let _ = stdout.flush().await;
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// hosted mode (`--http <addr>`)
//
// Serves the same tools over MCP's streamable HTTP transport, so sessions on
// other machines share this one store: the server is the store's single
// writer-of-record, as the daemon is for the sessions of one machine. It runs
// stateless with JSON responses — every request stands alone — so a restart or
// a proxy in front never strands a session. `/mcp` and `/backup` require
// `Authorization: Bearer <SERAPH_MEMORY_TOKEN>`; `/health` is open.
// `SERAPH_MEMORY_HOSTS` lists the host names the server answers to (comma
// separated; loopback only by default, which guards a local server against DNS
// rebinding). After `SERAPH_MEMORY_IDLE_SECS` (default 900; 0 never) without a
// tool call it closes the store and drops the encoder, handing their memory
// back to the OS; the next call reopens them.
// --------------------------------------------------------------------------- //
mod http {
    use super::*;
    use axum::{
        body::Body,
        extract::{Request, State},
        http::{header, StatusCode},
        middleware::{self, Next},
        response::{IntoResponse, Response},
        routing::get,
        Router,
    };
    use rmcp::transport::streamable_http_server::{
        session::never::NeverSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };
    use std::time::Duration;

    pub async fn run(path: &Path, addr: &str) -> anyhow::Result<()> {
        let token = std::env::var("SERAPH_MEMORY_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("--http needs SERAPH_MEMORY_TOKEN, the bearer token every request must carry"))?;
        let hosts: Vec<String> = std::env::var("SERAPH_MEMORY_HOSTS")
            .ok()
            .map(|v| v.split(',').map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect())
            .filter(|v: &Vec<String>| !v.is_empty())
            .unwrap_or_else(|| vec!["localhost".into(), "127.0.0.1".into(), "::1".into()]);

        let engine = open_engine_or_exit(path);
        let handler = SeraphMemory::new(engine, path.to_string_lossy().to_string());
        let config = StreamableHttpServerConfig::default()
            .with_stateful_mode(false)
            .with_json_response(true)
            .with_allowed_hosts(hosts.clone());
        let session = handler.clone();
        let mcp = StreamableHttpService::new(
            move || Ok(session.clone()),
            Arc::new(NeverSessionManager::default()),
            config,
        );

        let backup_route = {
            let engine = handler.engine.clone();
            let path = path.to_path_buf();
            move || backup(engine.clone(), path.clone())
        };
        let gated = Router::new()
            .route_service("/mcp", mcp)
            .route("/backup", get(backup_route))
            .layer(middleware::from_fn_with_state(Arc::new(token), require_token));
        let app = Router::new().route("/health", get(|| async { "ok" })).merge(gated);

        let idle = std::env::var("SERAPH_MEMORY_IDLE_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(900);
        if idle > 0 {
            spawn_idle_closer(handler.engine.clone(), Duration::from_secs(idle));
        }

        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("serving MCP over HTTP on {addr} for hosts {hosts:?}; idle close after {idle}s");
        axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
        // Dropping the last handle closes the store cleanly before exit.
        drop(handler);
        tracing::info!("shut down");
        Ok(())
    }

    /// Ctrl-C, or the SIGTERM a container platform sends before stopping us.
    async fn shutdown_signal() {
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        #[cfg(unix)]
        let term = async {
            if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                s.recv().await;
            }
        };
        #[cfg(not(unix))]
        let term = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => {}
            _ = term => {}
        }
    }

    /// Admit a request only with the bearer token, compared in constant time.
    async fn require_token(State(token): State<Arc<String>>, req: Request, next: Next) -> Response {
        let given = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let ok = given.is_some_and(|g| {
            g.len() == token.len() && g.bytes().zip(token.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
        });
        if ok {
            next.run(req).await
        } else {
            (StatusCode::UNAUTHORIZED, [(header::WWW_AUTHENTICATE, "Bearer")], "missing or wrong bearer token\n")
                .into_response()
        }
    }

    /// Close the store once no tool call has used it for `idle`. The check
    /// takes the engine lock, so a close never lands mid-call.
    fn spawn_idle_closer(slot: Arc<Mutex<EngineSlot>>, idle: Duration) {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(idle.min(Duration::from_secs(30)));
            loop {
                tick.tick().await;
                let slot = slot.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    let Ok(mut guard) = slot.lock() else { return };
                    if guard.engine.is_none() || guard.last_used.elapsed() < idle {
                        return;
                    }
                    let before = rss_mb();
                    // Dropping the engine closes the encoder, then the store.
                    drop(guard.engine.take());
                    release_memory(&guard.path);
                    match (before, rss_mb()) {
                        (Some(b), Some(a)) => tracing::info!("idle {idle:?}: store closed; RSS {b:.0} MB -> {a:.0} MB"),
                        _ => tracing::info!("idle {idle:?}: store closed"),
                    }
                })
                .await;
            }
        });
    }

    /// The process's resident memory, where the OS reports it.
    fn rss_mb() -> Option<f64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let kb: f64 = status.lines().find(|l| l.starts_with("VmRSS:"))?.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / 1024.0)
    }

    /// Hand freed memory back to the OS. glibc keeps freed heap in its arenas
    /// (the engine allocates from many worker threads), so trim them; and drop
    /// the store's pages from the page cache, which counts against a container.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn release_memory(path: &Path) {
        use std::os::fd::AsRawFd;
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        const POSIX_FADV_DONTNEED: i32 = 4;
        unsafe { malloc_trim(0) };
        let gidx = PathBuf::from(format!("{}.gidx", path.display()));
        for p in [path.to_path_buf(), gidx] {
            if let Ok(f) = std::fs::File::open(&p) {
                unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
            }
        }
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    fn release_memory(_path: &Path) {}

    /// The store file as it stands: written to disk under the engine lock, so
    /// no write lands while it is read. A copy of this is a complete backup.
    /// A closed store is already whole on disk, so it is read as it is.
    async fn backup(slot: Arc<Mutex<EngineSlot>>, path: PathBuf) -> Response {
        let read = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
            let mut guard = slot.lock().map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?;
            if let Some(engine) = guard.engine.as_mut() {
                engine.sync()?;
            }
            Ok(std::fs::read(&path)?)
        })
        .await;
        match read {
            Ok(Ok(bytes)) => {
                let name = format!("memory-{}.sfg", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
                Response::builder()
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{name}\""))
                    .body(Body::from(bytes))
                    .expect("valid response")
            }
            Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, format!("backup failed: {e}\n")).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("backup task failed: {e}\n")).into_response(),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let path = resolve_store_path();
    tracing::info!("SERAPH memory store: {}", path.display());

    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--http") {
        let addr = args.get(i + 1).map(String::as_str).unwrap_or("127.0.0.1:8765");
        return http::run(&path, addr).await;
    }

    #[cfg(windows)]
    {
        let is_daemon = std::env::args().any(|a| a == "--daemon");
        let pipe_name = bridge::pipe_name_for(&path);
        if is_daemon {
            bridge::run_daemon(&path, &pipe_name).await
        } else {
            bridge::run_client(&path, &pipe_name).await
        }
    }
    #[cfg(not(windows))]
    {
        run_inprocess(&path).await
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_order, parse_recent_by, parse_time_us};
    use crate::store::{parse_tau, RecentBy, TauLevel};
    use seraph::TemporalOrder;
    use serde_json::json;

    fn now_us() -> i64 {
        chrono::Utc::now().timestamp_micros()
    }

    #[test]
    fn relative_ages_resolve_into_the_past() {
        let now = now_us();
        for (expr, secs) in [("90s", 90i64), ("30m", 1_800), ("24h", 86_400), ("7d", 604_800), ("2w", 1_209_600)] {
            let t = parse_time_us(expr).expect(expr);
            let delta = now - t;
            let expected = secs * 1_000_000;
            // Generous tolerance — the clock advances between the two reads.
            assert!(
                delta.abs_diff(expected) < 5_000_000,
                "{expr}: expected ~{expected}us ago, got {delta}us"
            );
        }
    }

    #[test]
    fn unit_spellings_are_accepted() {
        let a = parse_time_us("1d").unwrap();
        for alias in ["1day", "1days", "24h", "24hours"] {
            let b = parse_time_us(alias).unwrap();
            assert!(a.abs_diff(b) < 5_000_000, "{alias} should match 1d");
        }
    }

    #[test]
    fn absolute_forms_parse_exactly() {
        // RFC 3339, and the same instant as a plain date at UTC midnight.
        assert_eq!(
            parse_time_us("2026-07-01T00:00:00Z").unwrap(),
            parse_time_us("2026-07-01").unwrap()
        );
        // Raw epoch microseconds pass through untouched.
        assert_eq!(parse_time_us("1751328000000000").unwrap(), 1_751_328_000_000_000);
        // Dates before the epoch are event times like any other.
        assert_eq!(parse_time_us("1969-07-20").unwrap(), -14_256_000_000_000);
    }

    #[test]
    fn malformed_time_is_rejected() {
        for bad in ["", "   ", "yesterday", "7y", "2026-13-45", "abc", "-5d"] {
            assert!(parse_time_us(bad).is_err(), "'{bad}' should be rejected");
        }
    }

    #[test]
    fn order_parses_with_aliases_and_rejects_junk() {
        assert_eq!(parse_order(None).unwrap(), TemporalOrder::Relevance);
        assert_eq!(parse_order(Some("relevance")).unwrap(), TemporalOrder::Relevance);
        assert_eq!(parse_order(Some("MOST_RECENT")).unwrap(), TemporalOrder::MostRecent);
        assert_eq!(parse_order(Some(" newest ")).unwrap(), TemporalOrder::MostRecent);
        assert_eq!(parse_order(Some("oldest")).unwrap(), TemporalOrder::Oldest);
        assert!(parse_order(Some("sideways")).is_err());
    }

    #[test]
    fn tau_reads_numbers_and_levels_and_defaults_to_loose() {
        assert_eq!(parse_tau(None).unwrap(), TauLevel::Loose);
        assert_eq!(parse_tau(Some(&json!(null))).unwrap(), TauLevel::Loose);
        assert_eq!(parse_tau(Some(&json!(0.6))).unwrap(), TauLevel::Value(0.6));
        // A stringified number, as a client facing a loose schema may send it.
        assert_eq!(parse_tau(Some(&json!("0.6"))).unwrap(), TauLevel::Value(0.6));
        assert_eq!(parse_tau(Some(&json!("Strict"))).unwrap(), TauLevel::Strict);
        assert_eq!(parse_tau(Some(&json!("p95"))).unwrap(), TauLevel::Background);
        assert!(parse_tau(Some(&json!("tight"))).is_err());
        assert!(parse_tau(Some(&json!([0.5]))).is_err());
    }

    #[test]
    fn recent_by_parses_and_rejects_junk() {
        assert_eq!(parse_recent_by(None).unwrap(), RecentBy::Written);
        assert_eq!(parse_recent_by(Some("OCCURRED")).unwrap(), RecentBy::Occurred);
        assert!(parse_recent_by(Some("sideways")).is_err());
    }
}
