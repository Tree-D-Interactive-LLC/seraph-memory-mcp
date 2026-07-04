# Seraph Memory MCP

A standalone **MCP server** that gives an AI assistant (Claude Code, or any MCP
client) a persistent, self-organizing memory backed by the **SERAPH** geometric
knowledge substrate — talking to the engine over its **C ABI** through the
canonical [`seraph`](https://github.com/Tree-D-Interactive-LLC/seraph) safe
wrapper crate. No Python runtime, no PyO3.

One global store, the engine's own on-device encoder, and a curated set of
substrate tools: *remember / recall*, reasoning **chains**, three-population
**neighborhoods**, **contradiction** detection, **lineage** / provenance,
**warp** steering, declared **typed relations**, and diagnostics.

## Tools

| Group | Tools |
|-------|-------|
| Write | `remember`, `remember_batch`, `add_seed`, `promote_frame` |
| Recall | `recall`, `get_frame`, `eigenframes_list` |
| Chains | `chain`, `chain_to`, `chain_between` |
| Context / provenance | `neighborhood`, `lineage`, `evolution` |
| Warp (steering) | `recall_with_warp`, `warp_chain_to`, `warp_chain_between`, `warp_seed_margin`, `warp_steerability`, `warp_calibration` |
| Declared structure | `declare_relation`, `list_relations`, `assert_relation`, `retract_relation`, `structural_neighbors` |
| Diagnostics | `stats`, `contradictions`, `density`, `region_health`, `drift_report`, `verify` |

**Warp note:** run `warp_calibration` once per session before using named
magnitudes (`routing`, `saturation`, …) — defaults assume 1024-d and are wrong on
the 384-d default encoder. Keep `profile=bounded`.

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
2. Otherwise it queries the **latest GitHub release** of
   [`Tree-D-Interactive-LLC/seraph`](https://github.com/Tree-D-Interactive-LLC/seraph),
   downloads the self-contained candle-GPU FFI build
   (`*-ffi-windows-x86_64-gpu.zip`), and caches it under `target/` keyed by
   release tag. CUDA is loaded at runtime via the driver, with automatic CPU
   fallback — the build has zero native GPU dependencies.

The safe Rust surface (`seraph` crate) is pulled as a **git dependency** from the
same engine repo, so the binding layer can't drift out of sync with the engine.

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
