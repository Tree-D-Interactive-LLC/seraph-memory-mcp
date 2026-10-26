//! Build script: obtain the SERAPH engine library, link it, and place it next
//! to the produced binary so it is found at load time.
//!
//! Windows: the MSVC linker expects the bare name `seraph.lib`, but the release
//! ships `seraph.dll.lib`, so we copy it under that name into OUT_DIR and link
//! from there, then copy `seraph.dll` into the target profile dir (e.g.
//! `target/release/`) so the .exe can load it as a sibling. Linux: we link
//! `libseraph.so` directly, copy it beside the binary, and give the binary an
//! `$ORIGIN` rpath so it is found there.
//!
//! FFI directory resolution:
//!   1. `SERAPH_LIB_DIR` env var — explicit override pointing at a local engine
//!      build (a directory with `seraph.dll` + `seraph.dll.lib`, or
//!      `libseraph.so`). Use this for offline builds or to test an unreleased
//!      engine.
//!   2. The GitHub release of Tree-D-Interactive-LLC/seraph the bindings come
//!      from: the tag the `seraph` dependency in Cargo.toml pins, else the
//!      version Cargo.lock resolved it to (`0.4.1` is release `v0.4.1`), else
//!      the latest release. Download its `*-ffi-windows-x86_64-gpu.zip` or
//!      `*-ffi-linux-x86_64-gpu.tar.gz` asset
//!      (candle GPU + CPU fallback, self-contained — no external runtime) and
//!      cache it under OUT_DIR keyed by release tag. This is the source of
//!      truth; there is no vendored copy.
//!
//! The engine DLL exposes a C ABI and the `seraph` crate provides the matching
//! bindings, so both come from the same release: `cargo update -p seraph` (or
//! bumping a pinned tag) moves the bindings and the DLL together, and reruns
//! this script. Taking the latest release regardless of the lock would pair the
//! locked bindings with a newer DLL as soon as a release is published.

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

const REPO: &str = "Tree-D-Interactive-LLC/seraph";

/// What the engine release ships for the target OS: the self-contained
/// candle-GPU FFI asset (CPU fallback), the runtime library, and the file the
/// linker reads.
struct Platform {
    asset_suffix: &'static str,
    runtime_lib: &'static str,
    link_lib: &'static str,
}

fn platform() -> Platform {
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => Platform {
            asset_suffix: "-ffi-linux-x86_64-gpu.tar.gz",
            runtime_lib: "libseraph.so",
            link_lib: "libseraph.so",
        },
        _ => Platform {
            asset_suffix: "-ffi-windows-x86_64-gpu.zip",
            runtime_lib: "seraph.dll",
            link_lib: "seraph.dll.lib",
        },
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let platform = platform();

    let ffi_dir = resolve_ffi_dir(&out_dir, &platform);
    let runtime_lib = ffi_dir.join(platform.runtime_lib);
    let link_lib = ffi_dir.join(platform.link_lib);

    if !runtime_lib.exists() || !link_lib.exists() {
        panic!(
            "SERAPH FFI not found under {} (expected {} + {}). \
             Set SERAPH_LIB_DIR to a local engine build, or ensure the GitHub release \
             is reachable.",
            ffi_dir.display(),
            platform.runtime_lib,
            platform.link_lib
        );
    }

    if platform.link_lib == platform.runtime_lib {
        // Linux links the shared object itself and finds it beside the binary at run time.
        println!("cargo:rustc-link-search=native={}", ffi_dir.display());
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
    } else {
        // Copy the import lib to the name MSVC's linker expects: `seraph.lib`.
        let staged_lib = out_dir.join("seraph.lib");
        fs::copy(&link_lib, &staged_lib).expect("copy seraph.dll.lib -> seraph.lib");
        println!("cargo:rustc-link-search=native={}", out_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=seraph");

    // Place the runtime library next to the output binary. OUT_DIR looks like
    // `target/<profile>/build/<pkg-hash>/out`; the profile dir is three parents up.
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        let dest = profile_dir.join(platform.runtime_lib);
        // Best-effort copy; ignore "file in use" if the server is running.
        let _ = fs::copy(&runtime_lib, &dest);
        // Also copy into the deps dir, where test/example binaries run from.
        let deps = profile_dir.join("deps").join(platform.runtime_lib);
        let _ = fs::copy(&runtime_lib, &deps);
    }

    println!("cargo:rerun-if-env-changed=SERAPH_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
}

/// Resolve the directory holding the engine library for this platform.
fn resolve_ffi_dir(out_dir: &PathBuf, platform: &Platform) -> PathBuf {
    // 1. Explicit override (local engine build / offline).
    if let Ok(dir) = env::var("SERAPH_LIB_DIR") {
        return PathBuf::from(dir);
    }

    // 2. The GitHub release the bindings come from (the source of truth).
    let tag = pinned_tag().or_else(locked_tag);
    match fetch_release_ffi(out_dir, tag.as_deref(), platform) {
        Ok(dir) => dir,
        Err(e) => panic!(
            "could not obtain the SERAPH engine library from GitHub release {} ({e}). \
             Set SERAPH_LIB_DIR to a local engine build (a directory containing {}), or \
             restore network access to GitHub.",
            tag.as_deref().unwrap_or("latest"),
            platform.runtime_lib
        ),
    }
}

/// The release tag the `seraph` dependency in Cargo.toml pins, if any.
fn pinned_tag() -> Option<String> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?).join("Cargo.toml");
    let text = fs::read_to_string(manifest).ok()?;
    let line = text.lines().find(|l| {
        let l = l.trim_start();
        (l.starts_with("seraph ") || l.starts_with("seraph=")) && l.contains("tag")
    })?;
    let after = &line[line.find("tag")? + 3..];
    let q1 = after.find('"')? + 1;
    let q2 = after[q1..].find('"')? + q1;
    Some(after[q1..q2].to_string())
}

/// The release tag of the `seraph` version Cargo.lock resolved, if there is a
/// lock: `version = "0.4.1"` under `name = "seraph"` is release `v0.4.1`.
fn locked_tag() -> Option<String> {
    let lock = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?).join("Cargo.lock");
    let text = fs::read_to_string(lock).ok()?;
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"seraph\"" {
            let version = lines.next()?.trim().strip_prefix("version = \"")?.strip_suffix('"')?;
            return Some(format!("v{version}"));
        }
    }
    None
}

/// Resolve + download + extract a release's FFI archive (the latest when `tag`
/// is `None`). Cached by tag.
fn fetch_release_ffi(out_dir: &PathBuf, tag: Option<&str>, platform: &Platform) -> Result<PathBuf, String> {
    let (tag, url) = release_asset_url(tag, platform.asset_suffix)?;

    let cache = out_dir.join(format!("seraph-ffi-{tag}"));
    let lib = cache.join(platform.runtime_lib);
    if lib.exists() {
        return Ok(cache); // already downloaded for this release
    }

    fs::create_dir_all(&cache).map_err(|e| format!("create cache dir: {e}"))?;
    let archive = out_dir.join(format!("seraph-ffi-{tag}{}", platform.asset_suffix));

    // Download (curl is bundled on Windows 10+).
    run("curl", &["-fsSL", "-o", &archive.to_string_lossy(), &url])
        .map_err(|e| format!("curl download {url}: {e}"))?;

    // Extract (Windows `tar` is bsdtar and reads zips; GNU tar reads the gzip).
    run("tar", &["-xf", &archive.to_string_lossy(), "-C", &cache.to_string_lossy()])
        .map_err(|e| format!("tar extract: {e}"))?;
    let _ = fs::remove_file(&archive);

    if lib.exists() {
        Ok(cache)
    } else {
        Err(format!("{} not present after extraction", platform.runtime_lib))
    }
}

/// Query the GitHub API for a release (the latest when `tag` is `None`) and
/// return (tag, asset_url) for the FFI build whose name ends with `suffix`.
fn release_asset_url(tag: Option<&str>, suffix: &str) -> Result<(String, String), String> {
    let api = match tag {
        Some(t) => format!("https://api.github.com/repos/{REPO}/releases/tags/{t}"),
        None => format!("https://api.github.com/repos/{REPO}/releases/latest"),
    };
    let out = Command::new("curl")
        .args(["-fsSL", "-H", "User-Agent: seraph-memory-build", "-H", "Accept: application/vnd.github+json", &api])
        .output()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("GitHub API request failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let json = String::from_utf8_lossy(&out.stdout);

    // Scan every `browser_download_url` value and pick the one ending in the FFI
    // asset suffix. (The asset's API `url` and `name` fields also contain the
    // suffix/zip, so we must match the download URL field specifically.)
    let mut pos = 0;
    let mut url = None;
    while let Some((val, next)) = quoted_value_after(&json, "\"browser_download_url\"", pos) {
        if val.ends_with(suffix) {
            url = Some(val);
            break;
        }
        pos = next;
    }
    let url = url.ok_or_else(|| format!("no asset matching *{suffix} in release {}", tag.unwrap_or("latest")))?;

    let tag = quoted_value_after(&json, "\"tag_name\"", 0)
        .map(|(v, _)| v)
        .unwrap_or_else(|| "latest".to_string());

    Ok((tag, url))
}

/// Find `key` in `json` starting at `from`, then return the next double-quoted
/// string value (after the following `:`) and the index just past it. A tiny,
/// dependency-free reader for the few flat string fields we need.
fn quoted_value_after(json: &str, key: &str, from: usize) -> Option<(String, usize)> {
    let k = json.get(from..)?.find(key)? + from;
    let after = k + key.len();
    let colon = json.get(after..)?.find(':')? + after;
    let q1 = json.get(colon..)?.find('"')? + colon + 1;
    let q2 = json.get(q1..)?.find('"')? + q1;
    Some((json[q1..q2].to_string(), q2))
}

fn run(cmd: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .map_err(|e| format!("spawn {cmd}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{cmd} exited with {status}"))
    }
}
