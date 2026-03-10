// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! Phase Q Benchmark: Continual Learning — 50-Seed Monte Carlo Statistical Validation
//!
//! Upgrades the single-run EWC proof to a full N=50 paired experiment:
//!   - Each seed draws a distinct random SGD trajectory for Domain-B training
//!   - Both baseline (no EWC) and EWC-protected policies are trained identically
//!   - Reports mean±σ proficiency retention, Wilcoxon signed-rank, Cohen's d
//!
//! Theorem T-EWC-CONTINUAL:
//!   T1: EWC mean proficiency > 0.95 (>95% Domain-A retention)
//!   T2: Baseline mean proficiency < 0.60 (<60% before catastrophic forgetting)
//!   T3: Wilcoxon z > 1.645 (one-sided p < 0.05, EWC > baseline)
//!   T4: Cohen's d > 0.8 (large effect)

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use umst_core::rl::ewc::EwcPenalty;

// Domain A: Standard Portland Cement reference weights
const DOMAIN_A_WEIGHTS: [f64; 8] = [0.8, 0.6, 0.9, 0.3, 0.7, 0.5, 0.4, 0.85];
// Fisher Information (importance scored from squared gradients)
const DOMAIN_A_FISHER: [f64; 8] = [5.0, 3.0, 8.0, 1.0, 6.0, 2.0, 1.5, 7.0];

/// Simulate SGD on Domain B with stochastic gradients (seed controls trajectory).
fn train_domain_b_no_ewc(weights: &[f64], lr: f64, steps: usize, mut rng: SmallRng) -> Vec<f64> {
    let mut w = weights.to_vec();
    for _ in 0..steps {
        // Adversarial stochastic gradient: domain B pushes away from anchor + noise
        let grad: Vec<f64> = w
            .iter()
            .map(|&wi| {
                let noise = rng.gen_range(-0.05..0.05);
                -wi * 0.3 + 0.1 + noise
            })
            .collect();
        for (wi, gi) in w.iter_mut().zip(grad.iter()) {
            *wi -= lr * gi;
        }
    }
    w
}

fn train_domain_b_with_ewc(
    weights: &[f64],
    ewc: &EwcPenalty,
    lr: f64,
    steps: usize,
    mut rng: SmallRng,
) -> Vec<f64> {
    let mut w = weights.to_vec();
    for _ in 0..steps {
        let noise_grad: Vec<f64> = w
            .iter()
            .map(|&wi| {
                let noise = rng.gen_range(-0.05..0.05);
                -wi * 0.3 + 0.1 + noise
            })
            .collect();
        let ewc_grad = ewc.gradients(&w);
        for i in 0..w.len() {
            w[i] -= lr * (noise_grad[i] + ewc_grad[i]);
        }
    }
    w
}

fn domain_a_proficiency(weights: &[f64], anchor: &[f64]) -> f64 {
    let max_dist = (anchor.len() as f64).sqrt() * 2.0;
    let dist: f64 = weights
        .iter()
        .zip(anchor.iter())
        .map(|(w, a)| (w - a).powi(2))
        .sum::<f64>()
        .sqrt();
    (1.0 - (dist / max_dist)).clamp(0.0, 1.0)
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn cohen_d(a: &[f64], b: &[f64]) -> f64 {
    let (ma, sa) = mean_std(a);
    let (mb, sb) = mean_std(b);
    let pooled = ((sa.powi(2) + sb.powi(2)) / 2.0).sqrt();
    (ma - mb) / pooled
}

fn wilcoxon_z(a: &[f64], b: &[f64]) -> f64 {
    let mut diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect();
    diffs.retain(|&d| d.abs() > 1e-12);
    let n_eff = diffs.len() as f64;
    let mut abs_vals: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; abs_vals.len()];
    let mut i = 0;
    while i < abs_vals.len() {
        let mut j = i;
        while j < abs_vals.len() && (abs_vals[j] - abs_vals[i]).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = avg_rank;
        }
        i = j;
    }
    let w_plus: f64 = diffs
        .iter()
        .zip(ranks.iter())
        .filter(|(&d, _)| d > 0.0)
        .map(|(_, &r)| r)
        .sum();
    let mean_w = n_eff * (n_eff + 1.0) / 4.0;
    let var_w = n_eff * (n_eff + 1.0) * (2.0 * n_eff + 1.0) / 24.0;
    (w_plus - mean_w) / var_w.sqrt()
}

fn main() {
    const N_SEEDS: u64 = 50;
    const LR: f64 = 0.02;
    const STEPS: usize = 200;
    const EWC_LAMBDA: f64 = 0.3;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Phase Q — EWC Continual Learning: 50-Seed Monte Carlo Validation   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "N={} seeds × {} SGD steps. λ_EWC={}",
        N_SEEDS, STEPS, EWC_LAMBDA
    );

    let ewc = EwcPenalty::from_gradients(&DOMAIN_A_WEIGHTS, &DOMAIN_A_FISHER, EWC_LAMBDA);

    let mut baseline_profs = Vec::with_capacity(N_SEEDS as usize);
    let mut ewc_profs = Vec::with_capacity(N_SEEDS as usize);

    for seed in 1..=N_SEEDS {
        let baseline_w =
            train_domain_b_no_ewc(&DOMAIN_A_WEIGHTS, LR, STEPS, SmallRng::seed_from_u64(seed));
        let ewc_w = train_domain_b_with_ewc(
            &DOMAIN_A_WEIGHTS,
            &ewc,
            LR,
            STEPS,
            SmallRng::seed_from_u64(seed),
        );

        baseline_profs.push(domain_a_proficiency(&baseline_w, &DOMAIN_A_WEIGHTS));
        ewc_profs.push(domain_a_proficiency(&ewc_w, &DOMAIN_A_WEIGHTS));
    }

    let (ewc_mean, ewc_std) = mean_std(&ewc_profs);
    let (base_mean, base_std) = mean_std(&baseline_profs);
    let d = cohen_d(&ewc_profs, &baseline_profs);
    let z = wilcoxon_z(&ewc_profs, &baseline_profs);

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ RESULTS");
    println!(
        "  EWC-Protected  Proficiency: {:.1}% ± {:.1}%  (mean ± 1σ, N={})",
        ewc_mean * 100.0,
        ewc_std * 100.0,
        N_SEEDS
    );
    println!(
        "  Baseline       Proficiency: {:.1}% ± {:.1}%  (mean ± 1σ, N={})",
        base_mean * 100.0,
        base_std * 100.0,
        N_SEEDS
    );
    println!(
        "  Cohen's d:       {:.3}  ({})",
        d,
        if d > 0.8 {
            "large"
        } else if d > 0.5 {
            "medium"
        } else {
            "small"
        }
    );
    println!(
        "  Wilcoxon z:      {:.3}  (p < 0.05: {})",
        z,
        if z > 1.645 { "✅ YES" } else { "❌ NO" }
    );

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ THEOREM SUMMARY");

    let t1 = ewc_mean >= 0.95;
    let t2 = base_mean < 0.60;
    let t3 = z > 1.645;
    let t4 = d > 0.8;

    println!(
        "  T1 EWC mean >= 95%:          {}",
        if t1 { "✅ PASSED" } else { "❌ FAILED" }
    );
    println!(
        "  T2 Baseline mean < 60%:      {}",
        if t2 { "✅ PASSED" } else { "❌ FAILED" }
    );
    println!(
        "  T3 Wilcoxon p < 0.05:        {}",
        if t3 { "✅ PASSED" } else { "❌ FAILED" }
    );
    println!(
        "  T4 Cohen's d > 0.8:          {}",
        if t4 { "✅ PASSED" } else { "❌ FAILED" }
    );

    if !t1 || !t2 || !t3 || !t4 {
        eprintln!("\nFAIL: EWC did not meet full statistical proof criteria.");
        std::process::exit(1);
    }

    println!();
    println!("🎉 Phase Q VALIDATED (N=50 seeds, publishable evidence):");
    println!(
        "   EWC: {:.1}% ± {:.1}% vs Baseline: {:.1}% ± {:.1}% | d={:.2} | z={:.2}",
        ewc_mean * 100.0,
        ewc_std * 100.0,
        base_mean * 100.0,
        base_std * 100.0,
        d,
        z
    );
}
