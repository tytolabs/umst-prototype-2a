// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! rapl_measure — Minimal CLI energy sampler for Python interop.
//!
//! Wraps `PowermetricsSampler::monitor()` (the same trapezoidal-integration path
//! used by `hardware_heat_experiment`) so Python callers get the canonical Rust
//! implementation instead of a reimplemented powermetrics parser.
//!
//! Called by `rapl_energy.py` via subprocess when the compiled binary is present.
//! Falls back to Linux sysfs RAPL when powermetrics is unavailable.
//!
//! Usage:
//!   sudo cargo run --release --bin rapl_measure -- --duration-ms 5000
//!   sudo ./target/release/rapl_measure --duration-ms 3000 --interval-ms 200
//!
//! Stdout (one JSON line, ready for json.loads in Python):
//!   {"cpu_uj":12345.6,"gpu_uj":2345.6,"ane_uj":123.4,"total_uj":14814.6,
//!    "duration_ms":5000.0,"source":"powermetrics","available":true}
//!
//! Exit codes: 0 = success (available=true or available=false), 1 = arg error

use clap::Parser;
use std::time::{Duration, Instant};
use umst_core::hardware::rapl::PowermetricsSampler;

#[derive(Parser, Debug)]
#[command(name = "rapl_measure")]
#[command(about = "Measure local CPU/GPU/ANE energy via Apple PMU or Linux RAPL")]
struct Args {
    /// Duration to measure in milliseconds
    #[arg(long, default_value = "2000")]
    duration_ms: u64,

    /// Polling interval in milliseconds (lower = smoother, higher CPU overhead)
    #[arg(long, default_value = "200")]
    interval_ms: u64,
}

fn main() {
    let args = Args::parse();

    // ── macOS path: PowermetricsSampler::monitor() (trapezoidal integration) ──
    // A probe call tells us if powermetrics is accessible (requires sudo on macOS).
    if PowermetricsSampler::sample().is_some() {
        let t0 = Instant::now();
        let monitor = PowermetricsSampler::monitor(args.interval_ms);
        std::thread::sleep(Duration::from_millis(args.duration_ms));
        let delta = monitor.stop();

        // Use wall-clock duration as the authoritative time span
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            r#"{{"cpu_uj":{:.1},"gpu_uj":{:.1},"ane_uj":{:.1},"total_uj":{:.1},"duration_ms":{:.1},"source":"powermetrics","available":true}}"#,
            delta.cpu_uj, delta.gpu_uj, delta.ane_uj, delta.total_uj, wall_ms
        );
        return;
    }

    // ── Linux sysfs RAPL fallback ─────────────────────────────────────────────
    #[cfg(target_os = "linux")]
    {
        let rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj";
        let read_uj = |p: &str| -> Option<u64> {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|s| s.trim().parse().ok())
        };

        if let Some(before) = read_uj(rapl_path) {
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(args.duration_ms));
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

            if let Some(after) = read_uj(rapl_path) {
                let cpu_uj = after.saturating_sub(before) as f64;
                println!(
                    r#"{{"cpu_uj":{:.1},"gpu_uj":0.0,"ane_uj":0.0,"total_uj":{:.1},"duration_ms":{:.1},"source":"rapl_sysfs","available":true}}"#,
                    cpu_uj, cpu_uj, wall_ms
                );
                return;
            }
        }
    }

    // ── Fallback: hardware measurement not available ──────────────────────────
    // Python caller should use ENERGY_UJ_PER_TOKEN proxy instead.
    println!(
        r#"{{"cpu_uj":0.0,"gpu_uj":0.0,"ane_uj":0.0,"total_uj":0.0,"duration_ms":{:.1},"source":"none","available":false}}"#,
        args.duration_ms as f64
    );
}
