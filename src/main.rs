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

use store::{resolve_mag, Engine};

const DEFAULT_MODEL: &str = "BAAI/bge-small-en-v1.5";

// --------------------------------------------------------------------------- //
// time and order parsing — the application-layer half of temporal recall
// --------------------------------------------------------------------------- //

/// Parse a caller-supplied time expression into microseconds since the Unix
/// epoch, which is what the engine's temporal window takes.
///
/// Accepted forms, in the order they are tried:
/// - relative age — `30m`, `24h`, `7d`, `2w`: that long *before now*
/// - raw epoch microseconds — a bare integer, for programmatic callers
/// - RFC 3339 timestamp — `2026-07-01T14:30:00Z`
/// - plain date — `2026-07-01`, read as midnight UTC
fn parse_time_us(s: &str) -> Result<u64, String> {
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
                return Ok(then.timestamp_micros().max(0) as u64);
            }
        }
    }

    // Raw epoch microseconds.
    if t.chars().all(|c| c.is_ascii_digit()) {
        return t
            .parse::<u64>()
            .map_err(|_| format!("'{t}' is not a valid epoch-microsecond value"));
    }

    // Absolute timestamps.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(t) {
        return Ok(dt.timestamp_micros().max(0) as u64);
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(t, "%Y-%m-%d") {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Ok(dt.and_utc().timestamp_micros().max(0) as u64);
        }
    }

    Err(format!(
        "unrecognized time '{t}' — use a relative age (7d), a date (2026-07-01), \
         an RFC 3339 timestamp, or epoch microseconds"
    ))
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

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberReq {
    #[schemars(description = "Text content to commit to memory.")]
    pub content: String,
    #[schemars(description = "Optional metadata object stored alongside the frame.")]
    pub metadata: Option<Value>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberBatchReq {
    #[schemars(description = "List of texts to commit; encoded in one batched pass.")]
    pub contents: Vec<String>,
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
    #[schemars(description = "Similarity threshold 0.0-1.0 (default 0.5).")]
    pub tau: Option<f32>,
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
        `after`.")]
    pub before: Option<String>,
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
pub struct WarpPoint {
    #[schemars(description = "Text to encode as a warp direction (or an existing frame's text).")]
    pub input: String,
    #[schemars(description = "Magnitude: a float, or a named level: action_floor(0.05), routing(0.20), sub_saturation(0.80), saturation(1.50). Default routing.")]
    pub magnitude: Option<Value>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallWarpReq {
    #[schemars(description = "Search query text.")]
    pub query: String,
    #[schemars(description = "Warp targets to pull the query toward.")]
    pub targets: Option<Vec<WarpPoint>>,
    #[schemars(description = "Warp suppressions to push the query away from.")]
    pub suppress: Option<Vec<WarpPoint>>,
    #[schemars(description = "Profile: bounded (default), routing, or saturation.")]
    pub profile: Option<String>,
    #[schemars(description = "Bound parameter (default 1.5).")]
    pub bound_k: Option<f32>,
    #[schemars(description = "Max results (default 10).")]
    pub top_k: Option<u32>,
    #[schemars(description = "Similarity threshold (default 0.5).")]
    pub tau: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WarpChainToReq {
    pub query: String,
    pub targets: Option<Vec<WarpPoint>>,
    pub suppress: Option<Vec<WarpPoint>>,
    #[schemars(description = "Profile: bounded (default), routing, or saturation.")]
    pub profile: Option<String>,
    pub bound_k: Option<f32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct WarpChainBetweenReq {
    pub query_a: String,
    pub query_b: String,
    pub targets_a: Option<Vec<WarpPoint>>,
    pub targets_b: Option<Vec<WarpPoint>>,
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

fn to_points(v: Option<Vec<WarpPoint>>) -> Vec<(String, f32)> {
    v.unwrap_or_default()
        .into_iter()
        .map(|p| {
            let mag = p.magnitude.as_ref().map(resolve_mag).unwrap_or(0.20);
            (p.input, mag)
        })
        .collect()
}

fn ok_json(v: Value) -> Result<CallToolResult, McpError> {
    let s = serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string());
    Ok(CallToolResult::success(vec![Content::text(s)]))
}

// --------------------------------------------------------------------------- //
// server
// --------------------------------------------------------------------------- //

#[derive(Clone)]
pub struct SeraphMemory {
    engine: Arc<Mutex<Engine>>,
    store_path: String,
    #[allow(dead_code)]
    tool_router: ToolRouter<SeraphMemory>,
}

impl SeraphMemory {
    fn new(engine: Engine, store_path: String) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
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
            f(&mut guard)
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
        similarity to everything already present and returns its frame ID.")]
    async fn remember(&self, Parameters(r): Parameters<RememberReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.remember(&r.content, r.metadata.as_ref())).await
    }

    #[tool(description = "Commit many texts at once (batched encode, one GPU pass). Returns frame IDs.")]
    async fn remember_batch(&self, Parameters(r): Parameters<RememberBatchReq>) -> Result<CallToolResult, McpError> {
        self.run(move |e| e.remember_batch(&r.contents)).await
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
        timestamp — use after for \"what changed since ...\". When either is set, each hit also \
        carries its RFC 3339 timestamp.")]
    async fn recall(&self, Parameters(r): Parameters<RecallReq>) -> Result<CallToolResult, McpError> {
        let top_k = r.top_k.unwrap_or(10) as usize;
        let tau = r.tau.unwrap_or(0.5);
        let include_superseded = r.include_superseded.unwrap_or(false);
        let order = parse_order(r.order.as_deref())
            .map_err(|e| McpError::invalid_params(e, None))?;
        let after_us = match r.after.as_deref() {
            Some(s) => Some(parse_time_us(s).map_err(|e| McpError::invalid_params(e, None))?),
            None => None,
        };
        let before_us = match r.before.as_deref() {
            Some(s) => Some(parse_time_us(s).map_err(|e| McpError::invalid_params(e, None))?),
            None => None,
        };
        self.run(move |e| {
            e.recall(&r.query, top_k, tau, include_superseded, order, after_us, before_us)
        })
        .await
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
        Run warp_calibration once per session before using named magnitudes (defaults assume 1024d).")]
    async fn recall_with_warp(&self, Parameters(r): Parameters<RecallWarpReq>) -> Result<CallToolResult, McpError> {
        let targets = to_points(r.targets);
        let suppress = to_points(r.suppress);
        let profile = r.profile.unwrap_or_else(|| "bounded".into());
        let bound_k = r.bound_k.unwrap_or(1.5);
        let top_k = r.top_k.unwrap_or(10) as usize;
        let tau = r.tau.unwrap_or(0.5);
        let query = r.query;
        self.run(move |e| e.recall_with_warp(&query, &targets, &suppress, &profile, bound_k, top_k, tau))
            .await
    }

    #[tool(description = "Warped chain_to — chain to the knowledge matching a query after steering \
        the query vector toward targets / away from suppressions.")]
    async fn warp_chain_to(&self, Parameters(r): Parameters<WarpChainToReq>) -> Result<CallToolResult, McpError> {
        let targets = to_points(r.targets);
        let suppress = to_points(r.suppress);
        let profile = r.profile.unwrap_or_else(|| "bounded".into());
        let bound_k = r.bound_k.unwrap_or(1.5);
        let query = r.query;
        self.run(move |e| e.warp_chain_to(&query, &targets, &suppress, &profile, bound_k)).await
    }

    #[tool(description = "Warped chain_between — chain between two concepts with per-endpoint warp steering.")]
    async fn warp_chain_between(&self, Parameters(r): Parameters<WarpChainBetweenReq>) -> Result<CallToolResult, McpError> {
        let ta = to_points(r.targets_a);
        let tb = to_points(r.targets_b);
        let profile = r.profile.unwrap_or_else(|| "bounded".into());
        let bound_k = r.bound_k.unwrap_or(1.5);
        let qa = r.query_a;
        let qb = r.query_b;
        self.run(move |e| e.warp_chain_between(&qa, &qb, &ta, &tb, &profile, bound_k)).await
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
        (dimension-correct regime boundaries). Run once per session before using named magnitudes.")]
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
             `structural_neighbors`) for exact structure. WARP: run `warp_calibration` once per \
             session before named magnitudes (defaults assume 1024d). Use `routing` magnitude for \
             gentle steering, `saturation` only for cross-domain bridging; keep profile=bounded.",
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
    use super::{parse_order, parse_time_us};
    use seraph::TemporalOrder;

    fn now_us() -> u64 {
        chrono::Utc::now().timestamp_micros() as u64
    }

    #[test]
    fn relative_ages_resolve_into_the_past() {
        let now = now_us();
        for (expr, secs) in [("90s", 90u64), ("30m", 1_800), ("24h", 86_400), ("7d", 604_800), ("2w", 1_209_600)] {
            let t = parse_time_us(expr).expect(expr);
            let delta = now.saturating_sub(t);
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
}
