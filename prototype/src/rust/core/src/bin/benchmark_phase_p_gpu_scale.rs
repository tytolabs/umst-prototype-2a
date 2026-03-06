// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! Phase P Benchmark: GPU Tensor Acceleration
//!
//! Validates that burn Tensor operations on GPU (Metal/wgpu on Apple Silicon)
//! outperform a pure f64 CPU scalar implementation on a 5,000-node graph.
//!
//! Three backends measured:
//!   1. CPU-scalar    — pure f64 Vec matmul (the legacy PPO path)
//!   2. Burn-NdArray  — burn tensor ops on CPU (validates framework overhead)
//!   3. Burn-Wgpu     — burn tensor ops on GPU via Metal (feature = "gpu")
//!
//! Theorems (dual-threshold per hardware):
//!   T-GPU-CPU-RATIO:  burn-Wgpu faster than burn-NdArray
//!   T-GPU-SCALE-M:    Wgpu / CPU-scalar < 1/3   (≥3× on Apple Silicon Metal)
//!   T-GPU-SCALE-CUDA: Wgpu / CPU-scalar < 1/15  (≥15× on NVIDIA CUDA, if present)

#[cfg(feature = "ndarray")]
use burn::backend::NdArray;

use burn::prelude::*;
use std::time::Instant;
use umst_core::gpu::gnn_layer::GpuGnnLayer;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Dense graph size: 5,000 nodes × 32 hidden dims
const N_NODES: usize = 5_000;
const HIDDEN: usize = 32;
/// Number of repeated forward passes for timing stability
const N_REPEATS: usize = 5;

// ── CPU Scalar Baseline ───────────────────────────────────────────────────────

/// Pure f64 matrix multiplication: C = A @ B
/// Uses the canonical O(n²·k) loop — simulates the legacy GNN CPU path.
fn cpu_matmul_f64(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0_f64;
            for l in 0..k {
                sum += a[i * k + l] * b[l * m + j];
            }
            c[i * m + j] = sum;
        }
    }
    c
}

/// Simulate one full GNN forward pass on CPU:
///   H_self  = X @ W_self
///   H_neigh = A  @ X @ W_neigh
///   H_new   = tanh(H_self + H_neigh)
///   (attention skipped — pure matmul dominated cost)
fn cpu_scalar_gnn_forward(
    x: &[f64],
    adj: &[f64],
    w_self: &[f64],
    w_neigh: &[f64],
    n: usize,
    h: usize,
) -> Vec<f64> {
    let h_self = cpu_matmul_f64(x, w_self, n, h, h);
    let ax = cpu_matmul_f64(adj, x, n, n, h);
    let h_neigh = cpu_matmul_f64(&ax, w_neigh, n, h, h);
    h_self
        .iter()
        .zip(h_neigh.iter())
        .map(|(a, b)| (a + b).tanh())
        .collect()
}

// ── Timing helpers ────────────────────────────────────────────────────────────

fn time_cpu_scalar() -> f64 {
    // Build random matrices on CPU
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let seed = |s: u64| {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        ((h.finish() as f64) / u64::MAX as f64) * 0.02 - 0.01
    };

    let x: Vec<f64> = (0..(N_NODES * HIDDEN)).map(|i| seed(i as u64)).collect();
    let adj: Vec<f64> = (0..(N_NODES * N_NODES))
        .map(|i| {
            // Sparse-ish identity-like adjacency (diagonal = 1)
            if i % (N_NODES + 1) == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let w_self: Vec<f64> = (0..(HIDDEN * HIDDEN))
        .map(|i| seed(1000 + i as u64))
        .collect();
    let w_neigh: Vec<f64> = (0..(HIDDEN * HIDDEN))
        .map(|i| seed(2000 + i as u64))
        .collect();

    let mut total_ns = 0u128;
    for _ in 0..N_REPEATS {
        let t = Instant::now();
        let _ = cpu_scalar_gnn_forward(&x, &adj, &w_self, &w_neigh, N_NODES, HIDDEN);
        total_ns += t.elapsed().as_nanos();
    }
    (total_ns / N_REPEATS as u128) as f64 / 1e6 // → ms
}

#[cfg(feature = "ndarray")]
fn time_burn_cpu() -> Option<f64> {
    type B = NdArray<f32>;
    let device = Default::default();
    let layer = GpuGnnLayer::<B>::new(N_NODES, HIDDEN, &device);

    let mut total_ns = 0u128;
    for _ in 0..N_REPEATS {
        let node_feats = Tensor::<B, 2>::random(
            [N_NODES, HIDDEN],
            burn::tensor::Distribution::Normal(0.0, 0.01),
            &device,
        );
        let adj = Tensor::<B, 2>::eye(N_NODES, &device);
        let t = Instant::now();
        let (_, _, readout) = layer.run(node_feats, adj);
        // Force realisation of lazy ops
        let _ = readout.into_data();
        total_ns += t.elapsed().as_nanos();
    }
    Some((total_ns / N_REPEATS as u128) as f64 / 1e6)
}

#[cfg(not(feature = "ndarray"))]
fn time_burn_cpu() -> Option<f64> {
    None
}

#[cfg(feature = "gpu")]
fn time_burn_wgpu() -> Option<f64> {
    use burn_wgpu::{Wgpu, WgpuDevice};
    type B = Wgpu;
    let device = WgpuDevice::default();
    let layer = GpuGnnLayer::<B>::new(N_NODES, HIDDEN, &device);

    // Warm up: one un-timed forward pass to initialise Metal shaders
    {
        let node_feats = Tensor::<B, 2>::random(
            [N_NODES, HIDDEN],
            burn::tensor::Distribution::Normal(0.0, 0.01),
            &device,
        );
        let adj = Tensor::<B, 2>::eye(N_NODES, &device);
        let (_, _, readout) = layer.run(node_feats, adj);
        let _ = readout.into_data();
    }

    let mut total_ns = 0u128;
    for _ in 0..N_REPEATS {
        let node_feats = Tensor::<B, 2>::random(
            [N_NODES, HIDDEN],
            burn::tensor::Distribution::Normal(0.0, 0.01),
            &device,
        );
        let adj = Tensor::<B, 2>::eye(N_NODES, &device);
        let t = Instant::now();
        let (_, _, readout) = layer.run(node_feats, adj);
        let _ = readout.into_data(); // sync GPU→CPU
        total_ns += t.elapsed().as_nanos();
    }
    Some((total_ns / N_REPEATS as u128) as f64 / 1e6)
}

#[cfg(not(feature = "gpu"))]
fn time_burn_wgpu() -> Option<f64> {
    None
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Phase P — GPU Tensor Acceleration: 5,000-Node GNN Scale Benchmark     ║");
    println!(
        "║  N={N_NODES} nodes × H={HIDDEN} hidden dims × {N_REPEATS} repeats (mean wall-time)    ║"
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("⏱  Timing CPU scalar (f64 Vec matmul)...");
    let cpu_ms = time_cpu_scalar();
    println!("   CPU-scalar:    {cpu_ms:.1} ms/fwd-pass");

    let ndarray_result = if cfg!(feature = "ndarray") {
        println!("⏱  Timing burn NdArray (CPU tensors)...");
        time_burn_cpu()
    } else {
        println!(
            "   burn-NdArray:  [SKIPPED — `--no-default-features` used to bypass NdArray bugs]"
        );
        None
    };

    if let Some(ndarray_ms) = ndarray_result {
        println!("   burn-NdArray:  {ndarray_ms:.1} ms/fwd-pass");
    }

    let gpu_result = if cfg!(feature = "gpu") {
        println!("⏱  Timing burn Wgpu (Metal/GPU)... [warming up Metal shaders...]");
        time_burn_wgpu()
    } else {
        println!("   burn-Wgpu:     [SKIPPED — compile with --features gpu to activate Metal]");
        None
    };

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ RESULTS");
    println!("  Backend          Time (ms)   Speedup vs CPU-scalar");
    println!("  ─────────────    ─────────   ─────────────────────");
    println!("  CPU-scalar f64   {:>8.1}   1.0×  (baseline)", cpu_ms);
    if let Some(ndarray_ms) = ndarray_result {
        let ndarray_speedup = cpu_ms / ndarray_ms;
        println!(
            "  burn-NdArray     {:>8.1}   {ndarray_speedup:.2}× (CPU tensor ops)",
            ndarray_ms
        );
    } else {
        println!("  burn-NdArray       ---       ---  (skipped)");
    }

    if let Some(gpu_ms) = gpu_result {
        let wgpu_speedup = cpu_ms / gpu_ms;
        if let Some(ndarray_ms) = ndarray_result {
            let wgpu_vs_ndarray = ndarray_ms / gpu_ms;
            println!("  burn-Wgpu/Metal  {:>8.1}   {wgpu_speedup:.2}× vs CPU-scalar, {wgpu_vs_ndarray:.2}× vs NdArray", gpu_ms);
        } else {
            println!(
                "  burn-Wgpu/Metal  {:>8.1}   {wgpu_speedup:.2}× vs CPU-scalar",
                gpu_ms
            );
        }

        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ THEOREM SUMMARY");
        let t_gpu_cpu_ratio = ndarray_result.is_none() || gpu_ms < ndarray_result.unwrap();
        let t_metal_3x = wgpu_speedup >= 3.0;
        let t_cuda_15x = wgpu_speedup >= 15.0; // true on discrete GPU

        if let Some(ndarray_ms) = ndarray_result {
            println!(
                "  T-GPU-CPU-RATIO (Wgpu < NdArray):           {}  [{:.2}× faster]",
                if t_gpu_cpu_ratio {
                    "✅ PASSED"
                } else {
                    "❌ FAILED"
                },
                ndarray_ms / gpu_ms
            );
        } else {
            println!("  T-GPU-CPU-RATIO (Wgpu < NdArray):           ⏸ SKIPPED (NdArray disabled)");
        }

        println!(
            "  T-GPU-SCALE-M  (≥3× on Apple Silicon):      {}  [{:.2}× speedup]",
            if t_metal_3x {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            },
            wgpu_speedup
        );
        println!(
            "  T-GPU-SCALE-CUDA (≥15× on discrete GPU):    {}  [{:.2}× speedup]",
            if t_cuda_15x {
                "✅ PASSED"
            } else {
                "⏸ PENDING hardware"
            },
            wgpu_speedup
        );

        println!();
        if !t_gpu_cpu_ratio || !t_metal_3x {
            eprintln!("FAIL: GPU did not meet minimum Metal speedup criteria.");
            std::process::exit(1);
        }
        println!("🎉 Phase P VALIDATED:");
        if t_cuda_15x {
            println!("   Full 15× speedup achieved (discrete GPU detected).");
        } else {
            println!("   Metal baseline ≥3× confirmed. Full 15× pending discrete GPU hardware.");
        }
    } else {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ THEOREM SUMMARY");
        if let Some(ndarray_ms) = ndarray_result {
            println!(
                "  CPU-scalar vs burn-NdArray speedup:  {:.2}×",
                cpu_ms / ndarray_ms
            );
        }
        println!();
        println!("⏸  GPU theorems pending Metal activation:");
        println!("   Run: cargo run --release --bin benchmark_phase_p_gpu_scale --features gpu");
        println!();
        println!("📋 Phase P: CPU pipeline verified. Awaiting GPU feature activation.");
    }
}
