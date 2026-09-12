//! Minimal ADB transport, implemented by shelling out to the system `adb` binary.
//!
//! A pure-Rust USB ADB client (e.g. the `adb_client` crate) was considered to avoid the
//! system-`adb` dependency, but its public API only exposes `forward`/`reverse` tunnel
//! setup through a connection to a *running adb server* (`adb start-server`, i.e. the
//! same daemon the `adb` binary talks to) rather than direct USB stream opening to an
//! arbitrary `localabstract:` service. Reimplementing that raw USB protocol ourselves
//! is a much larger undertaking than this MVP justifies, so we shell out to `adb`
//! instead -- it is a single, common dependency (`android-tools`/`platform-tools`) and
//! this is how effectively every scrcpy-like tool works.

use crate::error::{Error, Result};
use std::io::Write;
use std::process::{Child, Command, Stdio};

const DEVICE_SERVER_PATH: &str = "/data/local/tmp/scrcpy-server.jar";

/// A USB-attached Android device, addressed by ADB serial (or the sole attached device).
#[derive(Debug, Clone)]
pub struct AdbDevice {
    pub serial: Option<String>,
}

impl AdbDevice {
    /// Picks the sole attached USB device, or fails if zero or more than one are present.
    pub fn autodetect() -> Result<Self> {
        let out = adb(&[], &["devices"])?;
        let serials: Vec<&str> = out
            .lines()
            .skip(1)
            .filter_map(|l| l.split_whitespace().next())
            .filter(|s| !s.is_empty())
            .collect();
        match serials.len() {
            0 => Err(Error::NoDevice),
            1 => Ok(Self {
                serial: Some(serials[0].to_string()),
            }),
            _ => Err(Error::AdbCommand(format!(
                "multiple devices attached ({serials:?}); pass a serial explicitly"
            ))),
        }
    }

    pub fn with_serial(serial: impl Into<String>) -> Self {
        Self {
            serial: Some(serial.into()),
        }
    }

    fn args<'a>(&'a self, rest: &'a [&'a str]) -> Vec<&'a str> {
        let mut v = Vec::with_capacity(rest.len() + 2);
        if let Some(serial) = &self.serial {
            v.push("-s");
            v.push(serial.as_str());
        }
        v.extend_from_slice(rest);
        v
    }

    /// Pushes the embedded scrcpy-server jar to the device's tmp dir.
    pub fn push_server_jar(&self, jar: &[u8]) -> Result<()> {
        let tmp = std::env::temp_dir().join("phone-cam4linux-scrcpy-server.jar");
        std::fs::write(&tmp, jar)?;
        let tmp_str = tmp.to_string_lossy().to_string();
        adb(
            &self.args(&[]),
            &["push", &tmp_str, DEVICE_SERVER_PATH],
        )?;
        Ok(())
    }

    /// Sets up `adb forward tcp:<local_port> localabstract:scrcpy_<scid>` and returns the
    /// local TCP port scrcpy's server will be reachable on. Port 0 asks adb to pick a free one.
    pub fn forward(&self, scid_hex8: &str) -> Result<u16> {
        let remote = format!("localabstract:scrcpy_{scid_hex8}");
        let out = adb(&self.args(&[]), &["forward", "tcp:0", &remote])?;
        out.trim()
            .parse::<u16>()
            .map_err(|_| Error::AdbCommand(format!("unexpected `adb forward` output: {out:?}")))
    }

    pub fn remove_forward(&self, port: u16) {
        let local = format!("tcp:{port}");
        let _ = adb(&self.args(&[]), &["forward", "--remove", &local]);
    }

    /// Starts the scrcpy server on-device as a background shell process and returns the
    /// child handle for the local `adb shell` process driving it (killing it stops the
    /// remote server too, since it's the foreground process of that shell).
    ///
    /// `server_args` are the `key=value` options passed to the server (must include
    /// `scid=<hex8>` matching the socket name used in [`Self::forward`]).
    pub fn start_server(&self, server_args: &[String]) -> Result<Child> {
        let cmd = server_command(server_args);
        let mut argv = self.args(&[]);
        argv.push("shell");
        argv.push(&cmd);
        spawn_adb(&argv)
    }

    /// Runs the scrcpy server to completion (for one-shot modes such as
    /// `list_camera_sizes=true`) and returns everything it printed.
    pub fn run_server_once(&self, server_args: &[String]) -> Result<String> {
        let cmd = server_command(server_args);
        adb(&self.args(&[]), &["shell", &cmd])
    }
}

fn server_command(server_args: &[String]) -> String {
    format!(
        "CLASSPATH={DEVICE_SERVER_PATH} app_process / com.genymobile.scrcpy.Server {} {}",
        env!("SCRCPY_SERVER_VERSION"),
        server_args.join(" ")
    )
}

fn adb_binary() -> Result<&'static str> {
    static CHECKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ok = *CHECKED.get_or_init(|| {
        Command::new("adb")
            .arg("version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    });
    if ok {
        Ok("adb")
    } else {
        Err(Error::AdbNotFound)
    }
}

fn adb(pre_args: &[&str], args: &[&str]) -> Result<String> {
    let bin = adb_binary()?;
    let mut full = Vec::with_capacity(pre_args.len() + args.len());
    full.extend_from_slice(pre_args);
    full.extend_from_slice(args);
    log::debug!("adb {}", full.join(" "));
    let output = Command::new(bin).args(&full).output()?;
    if !output.status.success() {
        return Err(Error::AdbCommand(format!(
            "`adb {}` failed: {}",
            full.join(" "),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn spawn_adb(args: &[&str]) -> Result<Child> {
    let bin = adb_binary()?;
    log::debug!("adb {} (background)", args.join(" "));
    Ok(Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?)
}

/// Generates an 8 hex-digit session id, used both as the `scid=` server argument and the
/// `scrcpy_<scid>` local socket name, matching scrcpy's own convention (letting multiple
/// concurrent sessions on one device coexist).
pub fn random_scid_hex8() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    // scrcpy parses scid with Integer.parseInt(value, 16) into a *signed* 32-bit int,
    // so it must fit in 31 bits -- mask off the top bit or values with the high hex
    // digit >= 8 overflow and the server aborts with NumberFormatException.
    let mixed = (nanos ^ (pid << 32)) as u32 & 0x7fff_ffff;
    format!("{mixed:08x}")
}

/// Best-effort: ensure the writer side of a spawned child's stdin is flushed/closed so
/// `adb shell` doesn't hang waiting for input.
pub fn close_stdin(child: &mut Child) {
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.flush();
    }
}
