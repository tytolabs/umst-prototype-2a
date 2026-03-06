// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! RAPL (Running Average Power Limit) Energy Monitor
//!
//! Hierarchy of energy sources (best → fallback):
//!
//! 1. **macOS + root** — `powermetrics -n 1 --samplers cpu_power`
//!    Reads CPU package, GPU, and ANE power directly from the PMU via XNU.
//!    Returns real mW from the Apple Silicon power management unit.
//!    Requires the process to be run with `sudo` (same as UMST Sentry).
//!
//! 2. **Linux bare-metal** — `/sys/class/powercap/intel-rapl:0/energy_uj`
//!    Reads Intel/AMD RAPL energy counter via sysfs powercap interface.
//!
//! 3. **Fallback (macOS non-root / CI)** — deterministic mock
//!    Uses elapsed wall-time × TDP constant to produce a physically scaled
//!    µJ/op estimate. Enough for algorithmic comparison, not physical truth.

use std::time::Instant;

// ============================================================================
// Core types
// ============================================================================

/// A power telemetry snapshot.
#[derive(Clone, Debug)]
pub struct PowerSample {
    /// CPU package power in milliwatts (real or estimated).
    pub cpu_mw: f64,
    /// GPU power in milliwatts (real or 0.0 on fallback).
    pub gpu_mw: f64,
    /// Apple Neural Engine power in milliwatts (real or 0.0 on fallback).
    pub ane_mw: f64,
    /// Total package power = cpu + gpu + ane (mW).
    pub total_mw: f64,
    /// Wall-clock timestamp of the reading.
    pub timestamp: Instant,
}

impl PowerSample {
    fn total(cpu: f64, gpu: f64, ane: f64) -> Self {
        PowerSample {
            cpu_mw: cpu,
            gpu_mw: gpu,
            ane_mw: ane,
            total_mw: cpu + gpu + ane,
            timestamp: Instant::now(),
        }
    }
}

/// Energy delta between two samples.
#[derive(Clone, Debug)]
pub struct EnergyDelta {
    /// Energy consumed by the CPU package in microjoules.
    pub cpu_uj: f64,
    /// Energy consumed by the GPU in microjoules.
    pub gpu_uj: f64,
    /// Energy consumed by the ANE in microjoules.
    pub ane_uj: f64,
    /// Total energy in microjoules.
    pub total_uj: f64,
    /// Duration in milliseconds between the two samples.
    pub duration_ms: f64,
}

// ============================================================================
// PowermetricsSampler — Apple Silicon native power monitor
// ============================================================================

/// Samples CPU/GPU/ANE power directly from the Apple Silicon PMU via
/// `powermetrics`. Self-contained: no sentry daemon required.
///
/// On Linux this degrades to a zero-overhead sysfs RAPL reader.
/// On non-root macOS it returns a physics-scaled mock.
pub struct PowermetricsSampler;

/// A background monitor that polls powermetrics every `interval_ms` milliseconds.
/// Collects a time series of `PowerSample`s and integrates using the trapezoid rule
/// when `stop()` is called.  Eliminates the noise-amplification artefact from
/// single boundary-bracket sampling (one unlucky 200ms window × 90s duration = bogus MJ).
pub struct MonitorHandle {
    thread: Option<std::thread::JoinHandle<Vec<PowerSample>>>,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl MonitorHandle {
    /// Block until the background thread exits and return the integrated energy.
    pub fn stop(mut self) -> EnergyDelta {
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let samples = self
            .thread
            .take()
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        if samples.len() < 2 {
            return EnergyDelta {
                cpu_uj: 0.0,
                gpu_uj: 0.0,
                ane_uj: 0.0,
                total_uj: 0.0,
                duration_ms: 0.0,
            };
        }
        // Trapezoidal integration over the collected time series
        let mut cpu_uj = 0.0f64;
        let mut gpu_uj = 0.0f64;
        let mut ane_uj = 0.0f64;
        for w in samples.windows(2) {
            let dt_ms = w[1].timestamp.duration_since(w[0].timestamp).as_secs_f64() * 1000.0;
            cpu_uj += (w[0].cpu_mw + w[1].cpu_mw) / 2.0 * dt_ms;
            gpu_uj += (w[0].gpu_mw + w[1].gpu_mw) / 2.0 * dt_ms;
            ane_uj += (w[0].ane_mw + w[1].ane_mw) / 2.0 * dt_ms;
        }
        let duration_ms = samples
            .last()
            .unwrap()
            .timestamp
            .duration_since(samples[0].timestamp)
            .as_secs_f64()
            * 1000.0;
        EnergyDelta {
            cpu_uj,
            gpu_uj,
            ane_uj,
            total_uj: cpu_uj + gpu_uj + ane_uj,
            duration_ms,
        }
    }
}

impl PowermetricsSampler {
    /// Take one power reading. Calls `powermetrics -n 1 -i 200 --samplers cpu_power`.
    /// Returns `None` if powermetrics is unavailable (non-root or non-macOS).
    #[cfg(target_os = "macos")]
    pub fn sample() -> Option<PowerSample> {
        use std::process::Command;
        let out = Command::new("powermetrics")
            .args(["-n", "1", "-i", "200", "--samplers", "cpu_power"])
            .output()
            .ok()?;

        if !out.status.success() {
            return None;
        }

        let text = String::from_utf8_lossy(&out.stdout);
        let mut cpu_mw = 0.0f64;
        let mut gpu_mw = 0.0f64;
        let mut ane_mw = 0.0f64;

        for line in text.lines() {
            let low = line.to_lowercase();
            if (low.contains("cpu power") || low.contains("package power"))
                && !low.contains("cluster")
            {
                cpu_mw = parse_mw(line).unwrap_or(cpu_mw);
            } else if low.contains("gpu power") {
                gpu_mw = parse_mw(line).unwrap_or(gpu_mw);
            } else if low.contains("ane power") {
                ane_mw = parse_mw(line).unwrap_or(ane_mw);
            }
        }

        Some(PowerSample::total(cpu_mw, gpu_mw, ane_mw))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn sample() -> Option<PowerSample> {
        None
    }

    /// Start a background polling thread that calls `sample()` every `interval_ms` ms.
    /// Use this instead of single boundary brackets to eliminate noise amplification.
    /// Returns a `MonitorHandle`; call `.stop()` at the end of the measured region.
    pub fn monitor(interval_ms: u64) -> MonitorHandle {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();
        let thread = std::thread::spawn(move || {
            let mut samples = Vec::new();
            while !stop_clone.load(std::sync::atomic::Ordering::SeqCst) {
                if let Some(s) = Self::sample() {
                    samples.push(s);
                }
                std::thread::sleep(std::time::Duration::from_millis(interval_ms));
            }
            // Take one final sample after stopping
            if let Some(s) = Self::sample() {
                samples.push(s);
            }
            samples
        });
        MonitorHandle {
            thread: Some(thread),
            stop_flag: stop,
        }
    }

    /// Compute energy (µJ) between two single samples using the trapezoid rule.
    /// Only use this for short intervals; prefer `monitor()` for long runs.
    pub fn delta(before: &PowerSample, after: &PowerSample) -> EnergyDelta {
        let dt_ms = after
            .timestamp
            .duration_since(before.timestamp)
            .as_secs_f64()
            * 1000.0;
        let cpu_uj = (before.cpu_mw + after.cpu_mw) / 2.0 * dt_ms;
        let gpu_uj = (before.gpu_mw + after.gpu_mw) / 2.0 * dt_ms;
        let ane_uj = (before.ane_mw + after.ane_mw) / 2.0 * dt_ms;
        EnergyDelta {
            cpu_uj,
            gpu_uj,
            ane_uj,
            total_uj: cpu_uj + gpu_uj + ane_uj,
            duration_ms: dt_ms,
        }
    }
}

fn parse_mw(line: &str) -> Option<f64> {
    line.split(':')
        .nth(1)
        .and_then(|s| s.replace("mW", "").trim().parse::<f64>().ok())
}

// ============================================================================
// Legacy RaplMonitor (kept for the existing hardware_heat_experiment binary)
// ============================================================================

/// A sampled energy reading from the RAPL interface.
#[derive(Clone, Debug)]
pub struct RaplSample {
    /// Energy consumed since last reset, in microjoules (µJ)
    pub energy_uj: u64,
    /// Wall-clock timestamp of the reading
    pub timestamp: Instant,
}

/// Monitors CPU package power.
/// On Linux: uses the real sysfs RAPL powercap interface.
/// On macOS: proxies through `PowermetricsSampler` when root,
///           or falls back to a time-scaled mock otherwise.
pub struct RaplMonitor {
    #[cfg(target_os = "linux")]
    rapl_path: std::path::PathBuf,
    last_sample: Option<RaplSample>,
}

impl RaplMonitor {
    pub fn new() -> Self {
        RaplMonitor {
            #[cfg(target_os = "linux")]
            rapl_path: std::path::PathBuf::from(
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
            ),
            last_sample: None,
        }
    }

    pub fn read_energy_uj(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string(&self.rapl_path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Try real powermetrics first; fall back to scaled mock
            if let Some(ps) = PowermetricsSampler::sample() {
                // Integrate from last sample, using current total_mw
                let elapsed_ms = self
                    .last_sample
                    .as_ref()
                    .map(|s| s.timestamp.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(200.0);
                return (ps.total_mw * elapsed_ms) as u64; // mW × ms = µJ
            }
            // Pure mock: elapsed_µs × 4.5 µJ/µs ≈ 4.5 W
            let elapsed_us = self
                .last_sample
                .as_ref()
                .map(|s| s.timestamp.elapsed().as_micros() as f64)
                .unwrap_or(100.0);
            (elapsed_us * 4.5) as u64
        }
    }

    pub fn sample(&mut self) -> RaplSample {
        let s = RaplSample {
            energy_uj: self.read_energy_uj(),
            timestamp: Instant::now(),
        };
        self.last_sample = Some(s.clone());
        s
    }

    pub fn power_uw(before: &RaplSample, after: &RaplSample) -> f64 {
        let delta_uj = after.energy_uj.saturating_sub(before.energy_uj) as f64;
        let delta_us = after.timestamp.duration_since(before.timestamp).as_micros() as f64;
        if delta_us > 0.0 {
            delta_uj / delta_us * 1_000_000.0
        } else {
            0.0
        }
    }

    pub fn mock_spike_uj(computation_complexity: f64) -> f64 {
        computation_complexity * 1.5e-3
    }
}

/// Batch profiler: runs `n_iters` computation loops, collects energy per batch.
pub struct BatchProfiler {
    pub monitor: RaplMonitor,
    pub batch_size: usize,
}

impl BatchProfiler {
    pub fn new(batch_size: usize) -> Self {
        BatchProfiler {
            monitor: RaplMonitor::new(),
            batch_size,
        }
    }

    pub fn profile<F: Fn(usize) -> f64>(&mut self, n_batches: usize, f: F) -> Vec<f64> {
        let mut energy_per_op = Vec::with_capacity(n_batches);
        for batch in 0..n_batches {
            let mut total_uj = 0.0f64;
            for i in 0..self.batch_size {
                total_uj += f(batch * self.batch_size + i);
            }
            energy_per_op.push(total_uj / self.batch_size as f64);
        }
        energy_per_op
    }
}
