# Seraph Memory MCP

A standalone **MCP server** that gives an AI assistant (Claude Code, or any MCP
client) a persistent, self-organizing memory backed by the **SERAPH** geometric
knowledge substrate — talking to the engine over its **C ABI** through the
canonical [`seraph`](https://github.com/Tree-D-Interactive-LLC/seraph) safe
wrapper crate. No Python runtime, no PyO3.

One global store, the engine's own on-device encoder, and a curated set of
substrate tools: *remember / recall*, reads through **time** (a topic's
**history**, what is **recent**, how one idea **led to** another, the
**connections** between two), reasoning **chains**, three-population
**neighborhoods**, **contradiction** detection, **lineage**, **provenance** and
**influence**, each frame's **lifecycle**, **warp** steering, declared **typed
relations**, and diagnostics.

## Tools

| Group | Tools |
|-------|-------|
| Write | `remember`, `remember_batch`, `add_seed`, `promote_frame` |
| Recall | `recall`, `get_frame`, `lifecycle`, `eigenframes_list` |
| Time | `history`, `recent`, `led_to`, `connections` |
| Chains | `chain`, `chain_to`, `chain_between` |
| Context / provenance | `neighborhood`, `lineage`, `provenance`, `influence`, `evolution` |
| Warp (steering) | `recall_with_warp`, `warp_chain_to`, `warp_chain_between`, `warp_seed_margin`, `warp_steerability`, `warp_calibration` |
| Declared structure | `declare_relation`, `list_relations`, `assert_relation`, `retract_relation`, `structural_neighbors` |
| Diagnostics | `stats`, `contradictions`, `density`, `region_health`, `drift_report`, `verify` |

**Warp note:** named magnitudes (`action_floor`, `routing`, `sub_saturation`,
`saturation`) resolve against the store's own calibration, which the server
refreshes on load when the corpus has moved; `warp_calibration` recalibrates now
and shows the levels. Keep `profile=bounded`.

### Threshold

`recall`'s `tau` is the lowest score a returned memory may have. By default it
is the store's own **loose** level: the 75th percentile of how alike unrelated
memories are, so it moves with the corpus as it grows. Pass a number, or
`background` (p95) or `strict` (p99), and shift any of them with `tau_offset`
(`-0.05` to widen a vague query, `+0.05` to keep only close matches). An empty
result means nothing in memory stands out for the query.

The engine's threshold also decides which branches a search walks into, so a
high one can lose the best match outright: a query whose best memory scores
0.634 returns nothing at 0.62. The server therefore walks at 0.5 at most and
applies the threshold to the hits, so raising it trims the list rather than
hiding a better match. `recall_with_warp` takes the same `tau` and `tau_offset`.

### Time

A memory has two times: when it was **written** (recorded by the engine) and,
optionally, when its content **happened** (`occurred_at`, given at `remember`
time — the date of a meeting, a decision, an event). `remember_batch` takes
`items`, each with its own `metadata` and `occurred_at`, or plain `contents`.

`recall` is primarily a search by *meaning*; these optional arguments scope or
re-present its hits in time:

| Argument | Effect |
|----------|--------|
| `after` | Only frames written at or after this time |
| `before` | Only frames written at or before this time; the memory as it stood then |
| `occurred_after` | Only frames whose content happened at or after this time (their write time when they give none) |
| `occurred_before` | Only frames whose content happened at or before this time |
| `order` | `relevance` (default), `most_recent`, or `oldest` |

Times accept a **relative age** (`30m`, `24h`, `7d`, `2w` — that long ago), a
**plain date** (`2026-07-01`, read as midnight UTC), an **RFC 3339 timestamp**
(`2026-07-01T14:30:00Z`), or raw **epoch microseconds**. When any time option is
set, every hit carries its write timestamp; a hit whose frame gives
`occurred_at` always carries it.

Event time needs a store at the current format. On an older store `remember`
refuses `occurred_at` with that advice, and the `occurred_*` windows fall back
to write time. Migrating an older store records each memory's original write
time as its `occurred_at`; from then on `after`/`before` see the migration as
every memory's write time, and `occurred_after`/`occurred_before` answer about
its history.

> **`order` reorders, it does not search further back.** `most_recent` sorts the
> *same* relevant hits newest-first, so a recent frame that wasn't relevant enough
> to make `top_k` still won't appear. For more time depth, raise `top_k` — or use
> `after` to scope the search to a window, which frees result slots that
> higher-scoring older frames would otherwise occupy.

### Reads through time

Beyond recall, these tools read the record by when things were written and what
was built on what. Each returns its memories with their snippet and both times.

| Tool | Answers |
|------|---------|
| `history` | *Catch me up on X.* The memories about a topic, oldest first by when their content happened. The topic's best hits (`anchors`, default 3) choose the trees a walk reads, so this is the arc that led to where things stand, not a ranked list. |
| `recent` | *What has memory gained lately?* No query: the newest memories, newest first; `since` scopes it and reports the total; `by="occurred"` reads by when content happened. |
| `led_to` | *How did A lead to B?* A route forward in time, each step a memory written after the one before and connected to it when it was written. Each end is text or a frame ID. |
| `connections` | *In what different ways does X relate to Y?* Up to `max_routes` routes that share no intermediate memory, each with its bridge: the memory closest to both ends. `forward=true` keeps routes that run forward in time. |
| `provenance` / `influence` | *Why was this concluded? What did it go on to shape?* The strand a memory was built on, or that was built on it. |

`history`, `provenance` and `influence` take `toward`: text that chooses which
strand to follow ("why was this decided, as far as licensing goes"), at
`sub_saturation` unless `magnitude` says otherwise. A step may not move against
it, so a direction no strand runs in stops at the first memory. `provenance`,
`influence`, `led_to` and `connections` take `as_of`, which reads the record as
it stood at a time.

"Forward in time" for `led_to`, `connections` and the strands is the order the
store learned things, not the order they happened: a batch of old notes imported
today follows everything written before it. `history` orders by when content
happened.

> **On a migrated store, write time starts at the migration.** Migration keeps
> every memory's place in the record, so `led_to`, `connections` and the strands
> read the whole history. But it stamps one write time on every memory that
> predates it, so `as_of` and `after`/`before` see nothing earlier, and `recent`
> counts all of them as written at the migration. Read those memories by when
> they happened: `occurred_after`/`occurred_before`, `recent(by="occurred")`,
> and `history`.

## Requirements

- **Windows x86_64** with the **MSVC toolchain** (`x86_64-pc-windows-msvc`). The
  engine ships an MSVC import library, so the GNU toolchain won't link.
- **Rust** (stable) and network access on first build (see below).
- A **SERAPH license** at `%APPDATA%\SERAPH\license.key` is required only to
  **create** a new store — activate once with `seraph license activate` (the same
  license the pip package uses). Opening/reading an existing store needs no
  license.

## Build

```powershell
cargo build --release
```

`build.rs` obtains the engine on its own — no vendored binary, no source access:

1. If `SERAPH_LIB_DIR` is set, it links the `seraph.dll` + `seraph.dll.lib` in
   that directory (use this for offline builds or an unreleased engine).
2. Otherwise it downloads, from the GitHub release of
   [`Tree-D-Interactive-LLC/seraph`](https://github.com/Tree-D-Interactive-LLC/seraph)
   that the bindings come from, the self-contained candle-GPU FFI build
   (`*-ffi-windows-x86_64-gpu.zip`), and caches it under `target/` keyed by
   release tag. CUDA is loaded at runtime via the driver, with automatic CPU
   fallback — the build has zero native GPU dependencies.

The safe Rust surface (`seraph` crate) is pulled as a **git dependency** from the
same engine repo, unpinned: its default branch carries the latest release.
`Cargo.lock` records the commit a build uses, and `build.rs` fetches the DLL of
the release the lock names, so the bindings and the engine always come from the
same release. To move to a newer engine release:

```powershell
cargo update -p seraph
cargo build --release
```

To hold a release instead, add `tag = "vX.Y.Z"` to the `seraph` line in
`Cargo.toml`; the DLL follows the tag.

> **Linking note:** if the build fails with a `link.exe` error, Git's
> `C:\Program Files\Git\usr\bin\link.exe` is shadowing the MSVC linker — build
> from a *Developer PowerShell for VS*, or remove the Git `usr\bin` entry from
> `PATH` for the build.

The server binary is `target\release\seraph-memory-rs.exe`, with `seraph.dll`
copied alongside it — keep the two together if you move the binary.

## Store location

Resolved in this order:

1. `--store <path>` argument
2. `SERAPH_MEMORY_STORE` environment variable
3. default: `%APPDATA%\seraph-memory\memory.sfg`

A missing store is **auto-created** with the default model
`BAAI/bge-small-en-v1.5` (384-d), which requires a license as noted above.

## Register with Claude Code (user scope)

Point the MCP registration at the built executable:

```powershell
claude mcp add seraph-memory -s user -- C:\path\to\seraph-memory-rs.exe
```

Or add it by hand to your **user** MCP config (`~/.claude.json`):

```json
{
  "mcpServers": {
    "seraph-memory": {
      "command": "C:\\path\\to\\seraph-memory-rs.exe",
      "env": { "SERAPH_MEMORY_STORE": "C:\\Users\\you\\AppData\\Roaming\\seraph-memory\\memory.sfg" }
    }
  }
}
```

User scope makes the tools available in **every** project.

## Process model — one shared store across sessions

SERAPH is **single-writer** by design, but Claude Code launches one stdio server
process per session. To avoid N processes each opening the same `.sfg`, the
server runs a **shared-daemon** model (Windows):

- The first session to start spawns a **detached daemon** that opens the store
  once (the sole writer, warm encoder, one shared in-memory state).
- Every session process — including the first — then runs as a thin **client**
  that relays its stdio to the daemon over a named pipe
  (`\\.\pipe\seraph-memory-<store-hash>`). The daemon serves the same tool set
  over each connection, serializing all access on one engine mutex.
- The daemon **self-exits after an idle period** (no connected sessions),
  default **3600s** — override with `SERAPH_MEMORY_IDLE_SECS`.
- Daemon logs go to `%APPDATA%\seraph-memory\daemon.log` (clients log to stderr).
- If the daemon can't be reached or spawned, a session **falls back** to opening
  the store in-process (degraded — no cross-session sharing).

No configuration change is needed: the registered command is the same exe; it
decides client-vs-daemon at startup.

## License

The MCP server in this repository is licensed **MIT** (see [LICENSE](LICENSE)) —
use it, fork it, build on it.

The **SERAPH engine** it downloads and links (`seraph.dll`) is a separate,
proprietary component under the SERAPH Software License Agreement; MIT covers
this server code only, not the engine binary.
