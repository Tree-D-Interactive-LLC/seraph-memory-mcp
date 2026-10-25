//! Build script: obtain the SERAPH engine DLL, link its import library, and
//! place the runtime DLL next to the produced binary so it is found at load
//! time.
//!
//! The MSVC linker expects the bare name `seraph.lib`, but the release ships
//! `seraph.dll.lib`, so we copy it under that name into OUT_DIR and link from
//! there. We then copy `seraph.dll` into the target profile dir (e.g.
//! `target/release/`) so the .exe can load it as a sibling.
//!
//! FFI directory resolution:
//!   1. `SERAPH_LIB_DIR` env var — explicit override pointing at a local engine
//!      build (a directory with `seraph.dll` + `seraph.dll.lib`). Use this for
//!      offline builds or to test an unreleased engine.
//!   2. The GitHub release of Tree-D-Interactive-LLC/seraph whose tag the
//!      `seraph` dependency in Cargo.toml pins (the latest release when it pins
//!      none): download its `*-ffi-windows-x86_64-gpu.zip` asset (candle GPU +
//!      CPU fallback, self-contained — no external runtime) and cache it under
//!      OUT_DIR keyed by release tag. This is the source of truth; there is no
//!      vendored copy.
//!
//! The engine DLL exposes a C ABI and the `seraph` crate provides the matching
//! bindings, so both come from the same release: bumping the tag in Cargo.toml
//! moves the bindings and the DLL together, and reruns this script.

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

const REPO: &str = "Tree-D-Interactive-LLC/seraph";
/// Asset suffix for the self-contained candle-GPU Windows FFI build.
const ASSET_SUFFIX: &str = "-ffi-windows-x86_64-gpu.zip";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let ffi_dir = resolve_ffi_dir(&out_dir);
    let dll = ffi_dir.join("seraph.dll");
    let import_lib = ffi_dir.join("seraph.dll.lib");

    if !dll.exists() || !import_lib.exists() {
        panic!(
            "SERAPH FFI not found under {} (expected seraph.dll + seraph.dll.lib). \
             Set SERAPH_LIB_DIR to a local engine build, or ensure the GitHub release \
             is reachable.",
            ffi_dir.display()
        );
    }

    // Copy the import lib to the name MSVC's linker expects: `seraph.lib`.
    let staged_lib = out_dir.join("seraph.lib");
    fs::copy(&import_lib, &staged_lib).expect("copy seraph.dll.lib -> seraph.lib");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=seraph");

    // Place seraph.dll next to the output binary. OUT_DIR looks like
    // `target/<profile>/build/<pkg-hash>/out`; the profile dir is three parents up.
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        let dest = profile_dir.join("seraph.dll");
        // Best-effort copy; ignore "file in use" if the server is running.
        let _ = fs::copy(&dll, &dest);
        // Also copy into the deps dir, where test/example binaries run from.
        let deps = profile_dir.join("deps").join("seraph.dll");
        let _ = fs::copy(&dll, &deps);
    }

    println!("cargo:rerun-if-env-changed=SERAPH_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
}

/// Resolve the directory holding `seraph.dll` + `seraph.dll.lib`.
fn resolve_ffi_dir(out_dir: &PathBuf) -> PathBuf {
    // 1. Explicit override (local engine build / offline).
    if let Ok(dir) = env::var("SERAPH_LIB_DIR") {
        return PathBuf::from(dir);
    }

    // 2. The GitHub release the bindings are pinned to (the source of truth).
    let tag = pinned_tag();
    match fetch_release_ffi(out_dir, tag.as_deref()) {
        Ok(dir) => dir,
        Err(e) => panic!(
            "could not obtain the SERAPH engine DLL from GitHub release {} ({e}). \
             Set SERAPH_LIB_DIR to a local engine build (a directory containing seraph.dll \
             + seraph.dll.lib), or restore network access to GitHub.",
            tag.as_deref().unwrap_or("latest")
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

/// Resolve + download + extract a release's FFI zip (the latest when `tag` is
/// `None`). Cached by tag.
fn fetch_release_ffi(out_dir: &PathBuf, tag: Option<&str>) -> Result<PathBuf, String> {
    let (tag, url) = release_asset_url(tag)?;

    let cache = out_dir.join(format!("seraph-ffi-{tag}"));
    let dll = cache.join("seraph.dll");
    if dll.exists() {
        return Ok(cache); // already downloaded for this release
    }

    fs::create_dir_all(&cache).map_err(|e| format!("create cache dir: {e}"))?;
    let zip = out_dir.join(format!("seraph-ffi-{tag}.zip"));

    // Download (curl is bundled on Windows 10+).
    run("curl", &["-fsSL", "-o", &zip.to_string_lossy(), &url])
        .map_err(|e| format!("curl download {url}: {e}"))?;

    // Extract (Windows `tar` is bsdtar and reads zips).
    run("tar", &["-xf", &zip.to_string_lossy(), "-C", &cache.to_string_lossy()])
        .map_err(|e| format!("tar extract: {e}"))?;
    let _ = fs::remove_file(&zip);

    if dll.exists() {
        Ok(cache)
    } else {
        Err("seraph.dll not present after extraction".into())
    }
}

/// Query the GitHub API for a release (the latest when `tag` is `None`) and
/// return (tag, asset_url) for the FFI build whose name ends with `ASSET_SUFFIX`.
fn release_asset_url(tag: Option<&str>) -> Result<(String, String), String> {
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
        if val.ends_with(ASSET_SUFFIX) {
            url = Some(val);
            break;
        }
        pos = next;
    }
    let url = url.ok_or_else(|| format!("no asset matching *{ASSET_SUFFIX} in release {}", tag.unwrap_or("latest")))?;

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
