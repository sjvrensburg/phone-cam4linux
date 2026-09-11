//! Fetches the upstream `scrcpy-server.jar` for a pinned scrcpy release and embeds it
//! in the crate via `include_bytes!`. The server jar runs entirely on the Android side
//! (Camera2 capture + H.264 encode) and is never executed on the host, so reusing the
//! real, upstream-built artifact avoids reimplementing that logic in Rust.
//!
//! The download is verified against a pinned SHA-256 before being written to `OUT_DIR`,
//! so a compromised or unexpected upstream artifact fails the build instead of silently
//! being embedded.

use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

/// Pinned scrcpy release. Bump this (and SERVER_SHA256) together when updating.
const SCRCPY_VERSION: &str = "3.1";
/// SHA-256 of `scrcpy-server-v3.1` from
/// https://github.com/Genymobile/scrcpy/releases/download/v3.1/scrcpy-server-v3.1
const SERVER_SHA256: &str = "958f0944a62f23b1f33a16e9eb14844c1a04b882ca175a738c16d23cb22b86c0";

fn server_url() -> String {
    format!(
        "https://github.com/Genymobile/scrcpy/releases/download/v{v}/scrcpy-server-v{v}",
        v = SCRCPY_VERSION
    )
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=SCRCPY_SERVER_VERSION={SCRCPY_VERSION}");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let dest = Path::new(&out_dir).join("scrcpy-server.jar");

    // Allow a pre-fetched jar (e.g. vendored, or fetched by CI ahead of time) to skip
    // the network round-trip: PHONE_CAM4LINUX_SERVER_JAR=/path/to/scrcpy-server.jar
    if let Ok(local) = std::env::var("PHONE_CAM4LINUX_SERVER_JAR") {
        let bytes = std::fs::read(&local)
            .unwrap_or_else(|e| panic!("failed to read PHONE_CAM4LINUX_SERVER_JAR={local}: {e}"));
        verify_and_write(&bytes, &dest);
        return;
    }

    if dest.exists() {
        // Already downloaded by a previous build in this OUT_DIR; verify + reuse.
        let bytes = std::fs::read(&dest).expect("read cached server jar");
        if verify(&bytes) {
            return;
        }
    }

    let url = server_url();
    println!("cargo:warning=downloading scrcpy-server v{SCRCPY_VERSION} from {url}");

    let bytes = fetch(&url).unwrap_or_else(|e| {
        panic!(
            "failed to download scrcpy-server.jar from {url}: {e}\n\
             Set PHONE_CAM4LINUX_SERVER_JAR=/path/to/scrcpy-server.jar to use a local copy instead."
        )
    });

    verify_and_write(&bytes, &dest);
}

fn fetch(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let agent = ureq::AgentBuilder::new().build();
    let resp = agent.get(url).call()?;
    let mut bytes = Vec::new();
    resp.into_reader().read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn verify(bytes: &[u8]) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hex::encode(hasher.finalize());
    digest == SERVER_SHA256
}

fn verify_and_write(bytes: &[u8], dest: &Path) {
    if !verify(bytes) {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hex::encode(hasher.finalize());
        panic!(
            "scrcpy-server.jar hash mismatch: expected {SERVER_SHA256}, got {digest}.\n\
             Update SERVER_SHA256 in build.rs if you intentionally bumped SCRCPY_VERSION."
        );
    }
    let mut f = std::fs::File::create(dest).expect("create OUT_DIR/scrcpy-server.jar");
    f.write_all(bytes).expect("write server jar");
}
