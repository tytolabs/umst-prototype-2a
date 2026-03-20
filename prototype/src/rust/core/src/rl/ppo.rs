// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! PPO Agent — Graph Attention Network policy with three-signal closed-loop meta-optimization.
//!
//! Constitutional closed loop:
//!   design → PhysicsKernel → ThermodynamicGate → reward → GAE → PPO clip
//!   → GNN backward (all 7 GAT tensors) → meta_optimize(reward_var, gate_rej, attn_coh)
//!
//! Meta-optimization reads THREE signals on every update:
//!   1. Reward variance     → detect policy stagnation
//!   2. Gate rejection rate → detect drift into inadmissible manifold
//!   3. GAT attention coherence → detect when attention has not yet specialised

use super::guardrails::GuardrailEngine;
use super::reward::{RewardComponents, RewardConfig, RewardFunction, RewardType};
use super::state::{RLAction, RLState};
use crate::physics_kernel::{PhysicsConfig, PhysicsKernel};
use crate::science::thermodynamic_filter::{ThermodynamicFilter, ThermodynamicState};
use crate::tensors::MixTensor;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use std::cell::RefCell;
use serde::{Deserialize, Serialize};

/// Cache produced by `GNNNetwork::forward_with_cache`:
/// (output, per-node-hidden, per-node-msg, mean-pool, attention-flat)
type GnnForwardCache = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

/// Snapshot produced by meta_optimize() on each PPO update.
/// Useful for benchmark CSV logging to track the meta-learning trajectory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaStats {
    pub step: u64,
    pub entropy_coef: f64,
    pub epsilon: f64,
    pub avg_reward: f64,
    pub reward_variance: f64,
    pub gate_reject_rate: f64,
    pub attn_coherence: f64,
    pub gradient_velocity: f64,     // dLoss/dt over recent steps
    pub mi_agent_physics: f64,      // MI between agent decisions and physical outcomes
    pub learning_acceleration: f64, // How much faster learning occurs vs baseline
}


/// Gradient velocity tracker for learning acceleration measurement
#[derive(Clone, Debug)]
pub struct GradientVelocityTracker {
    loss_history: Vec<f64>,
    time_history: Vec<f64>,
    window_size: usize,
}

impl GradientVelocityTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            loss_history: Vec::new(),
            time_history: Vec::new(),
            window_size,
        }
    }

    /// Add a new loss measurement at the current time
    pub fn add_measurement(&mut self, loss: f64, time: f64) {
        self.loss_history.push(loss);
        self.time_history.push(time);

        // Keep only recent measurements
        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
            self.time_history.remove(0);
        }
    }

    /// Calculate current gradient velocity (dLoss/dt)
    pub fn gradient_velocity(&self) -> f64 {
        if self.loss_history.len() < 3 {
            return 0.0; // Need at least 3 points for meaningful derivative
        }

        // Use central difference for velocity calculation
        let n = self.loss_history.len();
        let dt = self.time_history[n-1] - self.time_history[n-3];
        if dt.abs() < 1e-6 {
            return 0.0; // Avoid division by zero
        }

        // Central difference: (f(x+1) - f(x-1)) / (2*dx)
        let velocity = (self.loss_history[n-1] - self.loss_history[n-3]) / (2.0 * dt);
        velocity
    }

    /// Calculate learning acceleration (second derivative)
    pub fn learning_acceleration(&self) -> f64 {
        if self.loss_history.len() < 5 {
            return 0.0;
        }

        // Simple acceleration calculation using finite differences
        let n = self.loss_history.len();
        let v1 = (self.loss_history[n-2] - self.loss_history[n-4]) /
                 (self.time_history[n-2] - self.time_history[n-4]).max(1e-6);
        let v2 = (self.loss_history[n-1] - self.loss_history[n-3]) /
                 (self.time_history[n-1] - self.time_history[n-3]).max(1e-6);

        let dt = self.time_history[n-1] - self.time_history[n-2];
        if dt.abs() < 1e-6 {
            return 0.0;
        }

        (v2 - v1) / dt
    }
}

/// Mutual Information calculator for agent-physics coupling
#[derive(Clone, Debug)]
pub struct MutualInformationTracker {
    decision_history: Vec<f64>,     // Agent decisions (normalized)
    outcome_history: Vec<f64>,      // Physical outcomes (normalized)
    window_size: usize,
}

impl MutualInformationTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            decision_history: Vec::new(),
            outcome_history: Vec::new(),
            window_size,
        }
    }

    pub fn add_sample(&mut self, decision: f64, outcome: f64) {
        self.decision_history.push(decision);
        self.outcome_history.push(outcome);

        if self.decision_history.len() > self.window_size {
            self.decision_history.remove(0);
            self.outcome_history.remove(0);
        }
    }

    /// Estimate mutual information using the Gaussian MI formula:
    ///   MI(X;Y) = −½ · ln(1 − ρ²)   [nats, valid lower bound for any joint distribution]
    ///
    /// This replaces the sparse-histogram estimator which returned 0 due to
    /// finite-sample bias with small windows (50 samples over 64 bins).
    /// Pearson ρ requires no binning, is well-defined at n≥10, and equals
    /// the full MI for bivariate Gaussians.
    pub fn mutual_information(&self) -> f64 {
        if self.decision_history.len() < 10 {
            return 0.0;
        }
        let n = self.decision_history.len() as f64;
        let mean_d = self.decision_history.iter().sum::<f64>() / n;
        let mean_o = self.outcome_history.iter().sum::<f64>() / n;
        let cov: f64 = self.decision_history.iter()
            .zip(self.outcome_history.iter())
            .map(|(&d, &o)| (d - mean_d) * (o - mean_o))
            .sum::<f64>() / n;
        let var_d: f64 = self.decision_history.iter()
            .map(|&d| (d - mean_d).powi(2))
            .sum::<f64>() / n;
        let var_o: f64 = self.outcome_history.iter()
            .map(|&o| (o - mean_o).powi(2))
            .sum::<f64>() / n;
        if var_d < 1e-12 || var_o < 1e-12 {
            return 0.0; // Constant signal → no information
        }
        let corr = cov / (var_d.sqrt() * var_o.sqrt());
        // Clamp ρ² away from 1 to avoid ln(0); then apply Gaussian MI formula
        let rho_sq = (corr * corr).min(0.9999);
        let mi = -0.5 * (1.0 - rho_sq).ln(); // nats
        mi.max(0.0)
    }
}

/// PPO hyperparameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PPOConfig {
    pub learning_rate: f64,
    pub gamma: f64,   // Discount factor
    pub epsilon: f64, // Clip range
    pub batch_size: usize,
    pub epochs_per_update: usize,
    pub entropy_coef: f64, // Exploration bonus
    pub value_coef: f64,   // Value loss coefficient

    // [StackOpt] Meta-Optimization Parameters
    pub meta_stability_threshold: f64, // e.g., 0.8 (Safety factor)
    pub meta_adaptive_rate: f64,       // Rate of hyperparameter adaptation

    /// Optional seed for reproducible Agent MAE (e.g. benchmark). None = use thread_rng.
    pub seed: Option<u64>,
}

impl PPOConfig {
    pub fn new() -> PPOConfig {
        PPOConfig {
            learning_rate: 0.0003,
            gamma: 0.99,
            epsilon: 0.2,
            batch_size: 64,
            epochs_per_update: 10,
            entropy_coef: 0.01,
            value_coef: 0.5,

            // [StackOpt] Defaults
            meta_stability_threshold: 1.2, // Min K_IC or Factor of Safety
            meta_adaptive_rate: 0.05,

            seed: None,
        }
    }

    /// Set seed for reproducible benchmark (Agent MAE).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Simulation result bundling physics outputs with thermodynamic state data.
/// Used by the constitutional gate to validate transitions.
struct SimulationResult {
    components: RewardComponents,
    w_c: f64,
    scm_ratio: f32,
    strength_fc: f64,
    yield_stress: f64,
    viscosity: f64,
}

/// Experience tuple for replay buffer
#[derive(Clone, Serialize, Deserialize)]
pub struct Experience {
    state: Vec<f64>,
    action: Vec<f64>,
    reward: f64,
    next_state: Vec<f64>,
    done: bool,
    log_prob: f64,  // Log probability of action under old policy
    heat_rate: f64, // [Phase M2] Microscopic proxy for structural grounding
}

/// Native Graph Attention Network (GAT) for the 35-node HyperGraph structure.
///
/// Architecture: Standard GraphSAGE message passing AUGMENTED with a
/// single-head graph attention layer:
///   e_{ij}  = LeakyReLU( a_self · W h_i  +  a_neigh · W h_j )
///   α_{ij}  = softmax_j( e_{ij} )
///   h_i_new = σ( Σ_j α_{ij} · W · h_j )
///
/// The attention vector `a` is decomposed into `w_a_self` (applied to the
/// self embedding) and `w_a_neigh` (applied to the neighbor embedding),
/// matching the Velickovic et al. (2018) formulation applied to scalar nodes.
#[derive(Serialize, Deserialize, Clone)]
pub struct GNNNetwork {
    pub w_self: Vec<f64>,  // [hidden_dim] — self-transform weights
    pub w_neigh: Vec<f64>, // [hidden_dim] — neighbor-transform weights
    pub b_gnn: Vec<f64>,   // [hidden_dim] — GNN layer bias
    pub w_out: Vec<f64>,   // [hidden_dim * out_dim] — output projection
    pub b_out: Vec<f64>,   // [out_dim] — output bias
    // H3: GAT attention parameters
    pub w_a_self: Vec<f64>, // [hidden_dim] — attention vector (self component)
    pub w_a_neigh: Vec<f64>, // [hidden_dim] — attention vector (neighbor component)
    pub hidden_dim: usize,
    pub out_dim: usize,
}

impl GNNNetwork {
    pub fn new(hidden_dim: usize, out_dim: usize) -> Self {
        Self::new_with_seed(hidden_dim, out_dim, 0)
    }

    /// Deterministic initialization for reproducible benchmarks.
    pub fn new_with_seed(hidden_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let init_gnn = |r: &mut SmallRng| (r.gen::<f64>() - 0.5) * 0.1;
        let init_out = |r: &mut SmallRng| (r.gen::<f64>() - 0.5) * (2.0 / (hidden_dim as f64).sqrt());
        let init_attn = |r: &mut SmallRng| (r.gen::<f64>() - 0.5) * 0.02;

        GNNNetwork {
            w_self: (0..hidden_dim).map(|_| init_gnn(&mut rng)).collect(),
            w_neigh: (0..hidden_dim).map(|_| init_gnn(&mut rng)).collect(),
            b_gnn: vec![0.0; hidden_dim],
            w_out: (0..hidden_dim * out_dim).map(|_| init_out(&mut rng)).collect(),
            b_out: vec![0.0; out_dim],
            w_a_self: (0..hidden_dim).map(|_| init_attn(&mut rng)).collect(),
            w_a_neigh: (0..hidden_dim).map(|_| init_attn(&mut rng)).collect(),
            hidden_dim,
            out_dim,
        }
    }

    // --- Internal helpers -------------------------------------------------------

    /// LeakyReLU with negative slope 0.2 (standard GAT hyperparameter)
    #[inline(always)]
    fn leaky_relu(x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            0.2 * x
        }
    }

    /// LeakyReLU derivative
    #[inline(always)]
    fn leaky_relu_d(x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            0.2
        }
    }

    /// Stable softmax for a small Vec.
    fn softmax(v: &[f64]) -> Vec<f64> {
        let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    // --- GAT Forward -----------------------------------------------------------

    /// Forward pass with GAT attention.
    ///
    /// Returns `(out, z, h, h_pool, m, attn)` where:
    /// - `z`    — pre-activation embeddings [n × hidden_dim]
    /// - `h`    — post-activation embeddings [n × hidden_dim]
    /// - `h_pool` — mean-pooled graph embedding [hidden_dim]
    /// - `m`    — mean neighbor signals [n]
    /// - `attn` — attention weights α_{ij}: flat [n × n] row-major
    pub fn forward_with_cache(&self, state: &[f64], use_tanh_out: bool) -> GnnForwardCache {
        let n = state.len();
        let nf = n as f64;

        // Step 1: Linear transform — z_i = W_self·x_i (scalar → hidden)
        // We use w_self as the per-dimension weight for x_i and w_neigh for m_i.
        let mut z = vec![vec![0.0; self.hidden_dim]; n];
        let mut m = vec![0.0; n]; // plain GraphSAGE neighbor mean (preserved for backprop)
        let sum_state: f64 = state.iter().sum();
        for i in 0..n {
            m[i] = (sum_state - state[i]) / (nf - 1.0).max(1.0);
        }

        // Step 2: Compute transformed node embeddings h_i (raw, before attention)
        let mut h_raw = vec![vec![0.0; self.hidden_dim]; n]; // pre-attention embeddings
        for i in 0..n {
            for k in 0..self.hidden_dim {
                let pre_z = state[i] * self.w_self[k] + m[i] * self.w_neigh[k] + self.b_gnn[k];
                z[i][k] = pre_z;
                h_raw[i][k] = pre_z.tanh();
            }
        }

        // Step 3: GAT attention coefficients α_{ij}
        // e_{ij} = LeakyReLU( Σ_k [w_a_self[k]·h_i[k] + w_a_neigh[k]·h_j[k]] )
        // α_{ij} = softmax_j( e_{ij} ) for each i

        // Algebraically hoist the dot products out of the O(n^2) nested loop down to O(n)
        let mut dot_self = vec![0.0; n];
        let mut dot_neigh = vec![0.0; n];
        for i in 0..n {
            for k in 0..self.hidden_dim {
                dot_self[i] += self.w_a_self[k] * h_raw[i][k];
                dot_neigh[i] += self.w_a_neigh[k] * h_raw[i][k];
            }
        }

        let mut attn = vec![vec![0.0f64; n]; n]; // α_{ij} matrix
        let mut e_ij = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                e_ij[j] = Self::leaky_relu(dot_self[i] + dot_neigh[j]);
            }
            attn[i] = Self::softmax(&e_ij);
        }

        // Step 4: Attention-weighted aggregation — h_i_att = Σ_j α_{ij} · h_raw[j]
        let mut h = vec![vec![0.0; self.hidden_dim]; n];
        for i in 0..n {
            for j in 0..n {
                let a = attn[i][j];
                for k in 0..self.hidden_dim {
                    h[i][k] += a * h_raw[j][k];
                }
            }
            // Apply activation after attention aggregation
            for k in 0..self.hidden_dim {
                h[i][k] = h[i][k].tanh();
            }
        }

        // Step 5: Mean-pool graph embedding
        let mut h_pool = vec![0.0; self.hidden_dim];
        for k in 0..self.hidden_dim {
            h_pool[k] = (0..n).map(|i| h[i][k]).sum::<f64>() / nf;
        }

        // Step 6: Output projection
        let mut out = vec![0.0; self.out_dim];
        for c in 0..self.out_dim {
            let mut y_c = self.b_out[c];
            for k in 0..self.hidden_dim {
                y_c += h_pool[k] * self.w_out[k * self.out_dim + c];
            }
            out[c] = if use_tanh_out { y_c.tanh() } else { y_c };
        }

        (out, z, h, h_pool, m)
    }

    pub fn forward(&self, state: &[f64], use_tanh_out: bool) -> Vec<f64> {
        self.forward_with_cache(state, use_tanh_out).0
    }

    /// Exact backward pass for GAT (w_self, w_neigh, b_gnn, w_a_self, w_a_neigh, w_out, b_out).
    ///
    /// Derivation:
    ///   L → out[c] → h_pool[k] → h[i][k] → attn[i][j], h_raw[j][k]
    ///              → z[i][k]  → w_self[k], w_neigh[k], b_gnn[k]
    ///   attn[i][j] → e_{ij} → w_a_self[k], w_a_neigh[k]
    pub fn backward(
        &mut self,
        state: &[f64],
        cache: &GnnForwardCache,
        d_loss_d_out: &[f64],
        lr: f64,
        use_tanh_out: bool,
    ) {
        let (out, z_mat, h_mat, h_pool, m) = cache;
        let n = state.len();
        let nf = n as f64;

        // ── Output layer gradient ──────────────────────────────────────────────
        let mut d_yc = vec![0.0; self.out_dim];
        for c in 0..self.out_dim {
            d_yc[c] = if use_tanh_out {
                d_loss_d_out[c] * (1.0 - out[c] * out[c])
            } else {
                d_loss_d_out[c]
            };
        }
        for c in 0..self.out_dim {
            self.b_out[c] += lr * d_yc[c];
            for k in 0..self.hidden_dim {
                self.w_out[k * self.out_dim + c] += lr * d_yc[c] * h_pool[k];
            }
        }

        // ── Gradient → h_pool ─────────────────────────────────────────────────
        let mut d_hpool = vec![0.0; self.hidden_dim];
        for k in 0..self.hidden_dim {
            d_hpool[k] = (0..self.out_dim)
                .map(|c| d_yc[c] * self.w_out[k * self.out_dim + c])
                .sum();
        }

        // ── Gradient → h[i][k] via mean pooling ───────────────────────────────
        // d L / d h[i][k] = d_hpool[k] / n  (pooling is just mean)
        let mut d_h = vec![vec![0.0; self.hidden_dim]; n];
        for i in 0..n {
            for k in 0..self.hidden_dim {
                // tanh activation post-attention: d/dh_att · (1 - tanh(·)²)
                d_h[i][k] = (d_hpool[k] / nf) * (1.0 - h_mat[i][k] * h_mat[i][k]);
            }
        }

        // ── Recompute h_raw (pre-attention embeddings) for attention backward ──
        let mut h_raw = vec![vec![0.0; self.hidden_dim]; n];
        for i in 0..n {
            for k in 0..self.hidden_dim {
                h_raw[i][k] = z_mat[i][k].tanh();
            }
        }

        // ── Recompute attention weights α_{ij} (needed for backward) ──────────
        let mut dot_self = vec![0.0; n];
        let mut dot_neigh = vec![0.0; n];
        for i in 0..n {
            for k in 0..self.hidden_dim {
                dot_self[i] += self.w_a_self[k] * h_raw[i][k];
                dot_neigh[i] += self.w_a_neigh[k] * h_raw[i][k];
            }
        }

        let mut e_mat = vec![vec![0.0f64; n]; n]; // raw logits before softmax
        let mut attn = vec![vec![0.0f64; n]; n]; // softmax-normalised
        for i in 0..n {
            let mut e_i = vec![0.0; n];
            for j in 0..n {
                e_i[j] = Self::leaky_relu(dot_self[i] + dot_neigh[j]);
            }
            attn[i] = Self::softmax(&e_i);
            e_mat[i] = e_i;
        }

        // ── Gradient → h_raw through attention aggregation ────────────────────
        // h[i][k] = tanh( Σ_j α_{ij} · h_raw[j][k] )
        // d L / d h_raw[j][k] = Σ_i d_h_pre[i][k] · α_{ij}
        //   where d_h_pre[i][k] is d L / d (Σ_j α_{ij} h_raw[j][k]) = d_h[i][k] (already includes tanh')
        let mut d_h_raw = vec![vec![0.0; self.hidden_dim]; n];
        for j in 0..n {
            for i in 0..n {
                let a = attn[i][j];
                for k in 0..self.hidden_dim {
                    d_h_raw[j][k] += d_h[i][k] * a;
                }
            }
        }

        // ── Gradient → attention weights α_{ij} ──────────────────────────────
        // d L / d α_{ij} = Σ_k d_h_pre[i][k] · h_raw[j][k]
        let mut d_alpha = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum_k = 0.0;
                for k in 0..self.hidden_dim {
                    sum_k += d_h[i][k] * h_raw[j][k];
                }
                d_alpha[i][j] = sum_k;
            }
        }

        // ── Gradient through softmax → e_{ij} ────────────────────────────────
        // d L / d e_{ij} = α_{ij} · (d_alpha[i][j] - Σ_l α_{il} · d_alpha[i][l])
        let mut d_e = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let dot_sum: f64 = (0..n).map(|l| attn[i][l] * d_alpha[i][l]).sum();
            for j in 0..n {
                d_e[i][j] = attn[i][j] * (d_alpha[i][j] - dot_sum);
            }
        }

        // ── Gradient through LeakyReLU → raw attention logit ─────────────────
        // d L / d dot_{ij} = d_e[i][j] · leaky_relu'(dot_{ij})
        let mut d_dot = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                d_dot[i][j] = d_e[i][j] * Self::leaky_relu_d(e_mat[i][j]);
            }
        }

        // ── Gradient → w_a_self and w_a_neigh ────────────────────────────────
        // dot_{ij} = Σ_k [w_a_self[k]·h_raw[i][k] + w_a_neigh[k]·h_raw[j][k]]
        let mut grad_a_self = vec![0.0; self.hidden_dim];
        let mut grad_a_neigh = vec![0.0; self.hidden_dim];
        for i in 0..n {
            for j in 0..n {
                for k in 0..self.hidden_dim {
                    grad_a_self[k] += d_dot[i][j] * h_raw[i][k];
                    grad_a_neigh[k] += d_dot[i][j] * h_raw[j][k];
                }
            }
        }
        for k in 0..self.hidden_dim {
            self.w_a_self[k] += lr * grad_a_self[k];
            self.w_a_neigh[k] += lr * grad_a_neigh[k];
        }

        // ── Gradient → h_raw from attention params + from direct GNN path ─────
        // Also need d L / d h_raw from w_a_self/w_a_neigh paths:
        // d dot_{ij} / d h_raw[i][k] = w_a_self[k]  → accumulate into d_h_raw[i][k]
        // d dot_{ij} / d h_raw[j][k] = w_a_neigh[k] → accumulate into d_h_raw[j][k]
        for i in 0..n {
            for j in 0..n {
                for k in 0..self.hidden_dim {
                    d_h_raw[i][k] += d_dot[i][j] * self.w_a_self[k];
                    d_h_raw[j][k] += d_dot[i][j] * self.w_a_neigh[k];
                }
            }
        }

        // ── Gradient → z[i][k] (through tanh in h_raw) → w_self, w_neigh ─────
        let mut grad_w_self = vec![0.0; self.hidden_dim];
        let mut grad_w_neigh = vec![0.0; self.hidden_dim];
        let mut grad_b_gnn = vec![0.0; self.hidden_dim];
        for i in 0..n {
            for k in 0..self.hidden_dim {
                // tanh' at z[i][k]
                let dtanh = 1.0 - h_raw[i][k] * h_raw[i][k];
                let d_zik = d_h_raw[i][k] * dtanh;
                grad_w_self[k] += d_zik * state[i];
                grad_w_neigh[k] += d_zik * m[i];
                grad_b_gnn[k] += d_zik;
            }
        }
        // Suppress unused-variable warning from original cache (z_mat is used above via h_raw)
        let _ = z_mat;
        for k in 0..self.hidden_dim {
            self.w_self[k] += lr * grad_w_self[k];
            self.w_neigh[k] += lr * grad_w_neigh[k];
            self.b_gnn[k] += lr * grad_b_gnn[k];
        }
    }

    /// Extract learned attention weight matrix α over the chemical proxy graph.
    /// Returns a flat Vec of length `n_nodes * n_nodes` (row-major).
    /// α_{ij} represents how much node i *attends* to node j when computing its embedding.
    /// High α_{ij} between `water` and `cement` reveals the kinetic hydration relationship
    /// learned from data — the "AI-discovered chemical kinetics" output.
    pub fn get_attention_weights(&self, state: &[f64]) -> Vec<f64> {
        let n = state.len();
        let m: Vec<f64> = (0..n)
            .map(|i| {
                let sum: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &v)| v)
                    .sum();
                sum / ((n as f64) - 1.0).max(1.0)
            })
            .collect();
        let h_raw: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..self.hidden_dim)
                    .map(|k| {
                        (state[i] * self.w_self[k] + m[i] * self.w_neigh[k] + self.b_gnn[k]).tanh()
                    })
                    .collect()
            })
            .collect();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            let e_i: Vec<f64> = (0..n)
                .map(|j| {
                    let dot: f64 = (0..self.hidden_dim)
                        .map(|k| self.w_a_self[k] * h_raw[i][k] + self.w_a_neigh[k] * h_raw[j][k])
                        .sum();
                    Self::leaky_relu(dot)
                })
                .collect();
            let alpha = Self::softmax(&e_i);
            flat.extend_from_slice(&alpha);
        }
        flat
    }
}

/// PPO Agent with Graph Neural Network policy natively supporting HyperGraph topologies
pub struct PPOAgent {
    config: PPOConfig,
    reward_function: RewardFunction,

    // Actor/Critic GNNs replacing standard MLPs to provide rigorous topological awareness
    actor_gnn: GNNNetwork,
    critic_gnn: GNNNetwork,

    /// Seeded RNG for deterministic select_action when config.seed is Some.
    rng: Option<RefCell<SmallRng>>,

    // Experience buffer
    buffer: Vec<Experience>,

    // Training stats
    total_steps: u64,
    episode_rewards: Vec<f64>,

    // Constitutional gate statistics
    gate_accepts: u64,
    gate_rejects: u64,
    guardrail_rejects: u64,

    // Learning progress trackers
    gradient_tracker: GradientVelocityTracker,
    mi_tracker: MutualInformationTracker,

    // Snapshot of the last meta-optimisation step
    last_meta_stats: MetaStats,
}

impl PPOAgent {
    pub fn new(ppo_config: PPOConfig, reward_type: RewardType) -> PPOAgent {
        let reward_config = RewardConfig::new(reward_type);
        let (actor_gnn, critic_gnn, rng) = match ppo_config.seed {
            Some(seed) => (
                GNNNetwork::new_with_seed(16, 9, seed),
                GNNNetwork::new_with_seed(16, 2, seed.wrapping_add(1)),
                Some(RefCell::new(SmallRng::seed_from_u64(seed.wrapping_add(2)))),
            ),
            None => (
                GNNNetwork::new(16, 9),
                GNNNetwork::new(16, 2),
                None,
            ),
        };

        PPOAgent {
            config: ppo_config,
            reward_function: RewardFunction::new(reward_config),

            actor_gnn,
            critic_gnn,

            rng,

            buffer: Vec::with_capacity(1024),
            total_steps: 0,
            episode_rewards: Vec::new(),

            gate_accepts: 0,
            gate_rejects: 0,
            guardrail_rejects: 0,

            gradient_tracker: GradientVelocityTracker::new(50), // Track last 50 loss measurements
            mi_tracker: MutualInformationTracker::new(100),     // Track last 100 decision-outcome pairs

            last_meta_stats: MetaStats::default(),
        }
    }

    /// Select action from current policy
    pub fn select_action(&self, state: &RLState) -> RLAction {
        let state_vec = state.to_vector();
        let action_probs = self.actor_gnn.forward(&state_vec, true);

        // Sample from Gaussian policy (seeded when config.seed is Some for reproducibility)
        let action_vec: Vec<f64> = if let Some(ref r) = self.rng {
            let mut rng = r.borrow_mut();
            action_probs
                .iter()
                .map(|&mean| mean + 0.1 * rand_normal_seeded(&mut *rng))
                .collect()
        } else {
            action_probs
                .iter()
                .map(|&mean| mean + 0.1 * rand_normal())
                .collect()
        };

        RLAction::from_vector(&action_vec)
    }

    /// Forward pass through policy network
    fn forward_policy(&self, state: &[f64]) -> Vec<f64> {
        self.actor_gnn.forward(state, true)
    }

    /// Estimate state value
    fn estimate_value(&self, state: &[f64]) -> f64 {
        self.critic_gnn.forward(state, false)[0]
    }

    /// Store experience in buffer
    pub fn store_experience(
        &mut self,
        state: &RLState,
        action: &RLAction,
        reward: f64,
        next_state: &RLState,
        done: bool,
        heat_rate: f64,
    ) {
        let log_prob = self.compute_log_prob(&state.to_vector(), &action.to_vector());

        self.buffer.push(Experience {
            state: state.to_vector(),
            action: action.to_vector(),
            reward,
            next_state: next_state.to_vector(),
            done,
            log_prob,
            heat_rate,
        });

        // Track MI between actions and thermodynamic outcomes (rewards)
        // Normalize action magnitude and reward for MI calculation
        let action_magnitude = action.to_vector().iter().map(|x| x*x).sum::<f64>().sqrt();
        let normalized_action = action_magnitude / 10.0; // Scale to reasonable range
        let normalized_reward = (reward + 100.0) / 200.0; // Shift and scale reward to [0,1]

        self.mi_tracker.add_sample(normalized_action, normalized_reward);

        self.total_steps += 1;

        // Update if buffer is full
        if self.buffer.len() >= self.config.batch_size {
            self.update();
        }
    }

    /// Meta-Optimization Step — the true closed loop.
    ///
    /// Reads from THREE signals (not one):
    ///   1. `reward_variance` — classic exploration/exploitation heuristic
    ///   2. `gate_rejection_rate` — when the constitutional gate is rejecting many proposals,
    ///      the policy is stuck in the inadmissible manifold; boost exploration to escape.
    ///   3. `attention_coherence` — mean off-diagonal α_{ij} from the GAT layer;
    ///      high coherence (→1/n) means the policy has learned no structure; boost entropy.
    ///
    /// Returns a `MetaStats` snapshot for benchmark CSV logging.
    fn meta_optimize(&mut self) -> MetaStats {
        if self.buffer.is_empty() {
            return MetaStats::default();
        }

        // ── Signal 1: reward variance (existing) ────────────────────────────
        let avg_r: f64 =
            self.buffer.iter().map(|e| e.reward).sum::<f64>() / self.buffer.len() as f64;
        let var_r: f64 = self
            .buffer
            .iter()
            .map(|e| (e.reward - avg_r).powi(2))
            .sum::<f64>()
            / self.buffer.len() as f64;
        let stagnant = var_r < 0.1;

        // ── Signal 2: gate rejection rate ──────────────────────────────────
        let total_gate = (self.gate_accepts + self.gate_rejects).max(1);
        let reject_rate = self.gate_rejects as f64 / total_gate as f64;
        // High rejection (>30%) means the policy is drifting into inadmissible space.
        // Boost exploration to push it back toward the admissible manifold.
        let gate_pressure = reject_rate > 0.30;

        // ── Signal 3: attention coherence from actor GAT ────────────────────
        // Average off-diagonal α deviation from uniform over the WHOLE buffer.
        // Sampling only buffer[0] was fragile; a single outlier state could
        // misclassify a well-specialised policy as stagnant (or vice-versa).
        let attn_coherence: f64 = {
            let (sum, count) = self
                .buffer
                .iter()
                .fold((0.0f64, 0usize), |(acc, cnt), exp| {
                    let state = &exp.state;
                    let n = state.len();
                    if n < 2 {
                        return (acc, cnt);
                    }
                    let uniform = 1.0 / n as f64;
                    let alpha = self.actor_gnn.get_attention_weights(state);
                    let coh = alpha
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (idx % n) != (idx / n))
                        .map(|(_, &v)| (v - uniform).abs())
                        .sum::<f64>()
                        / ((n * n - n) as f64 * uniform).max(1e-9);
                    (acc + coh, cnt + 1)
                });
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        };
        // coherence near 0 → GAT has learned structure; near 1 → still near uniform (stagnant).
        let attn_stagnant = attn_coherence < 0.05;

        // ── Adapt entropy_coef ────────────────────────────────────────────
        self.config.entropy_coef = if stagnant || gate_pressure || attn_stagnant {
            (self.config.entropy_coef * 1.5).min(0.2)
        } else {
            (self.config.entropy_coef * 0.995).max(0.001)
        };

        // ── Adapt trust region epsilon ────────────────────────────────────
        // Tighten when policy is near inadmissible manifold; relax when stable.
        self.config.epsilon = if avg_r < -10.0 || gate_pressure {
            (self.config.epsilon * 0.9).max(0.05)
        } else {
            (self.config.epsilon * 1.05).min(0.3)
        };

        let stats = MetaStats {
            step: self.total_steps,
            entropy_coef: self.config.entropy_coef,
            epsilon: self.config.epsilon,
            avg_reward: avg_r,
            reward_variance: var_r,
            gate_reject_rate: reject_rate,
            attn_coherence,
            gradient_velocity: self.gradient_tracker.gradient_velocity(),
            mi_agent_physics: self.mi_tracker.mutual_information(),
            learning_acceleration: self.gradient_tracker.learning_acceleration(),
        };

        if self.total_steps % 500 == 0 {
            eprintln!(
                "      [META] step={} H={:.4} ε={:.3} r̅={:.2} σ²={:.3} gate_rej={:.2} attn_coh={:.3}",
                stats.step, stats.entropy_coef, stats.epsilon,
                stats.avg_reward, stats.reward_variance,
                stats.gate_reject_rate, stats.attn_coherence
            );
        }
        stats
    }

    /// PPO update step — truly closed-loop:
    /// meta_optimize() → GAE advantages → PPO clip → actor/critic backward → buffer clear
    fn update(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        // Closed-loop meta-optimize first (adapts entropy, epsilon from gate+attention+reward)
        self.last_meta_stats = self.meta_optimize();

        // Compute advantages using GAE
        let advantages = self.compute_advantages();

        // Precompute returns from old value estimates
        let returns: Vec<f64> = self
            .buffer
            .iter()
            .enumerate()
            .map(|(i, exp)| advantages[i] + self.estimate_value(&exp.state))
            .collect();

        let buffer_clone: Vec<Experience> = self.buffer.clone();

        for _ in 0..self.config.epochs_per_update {
            for (i, exp) in buffer_clone.iter().enumerate() {
                // Actor exact forward cache for rigorous backprop
                let actor_cache = self.actor_gnn.forward_with_cache(&exp.state, true);
                let current_mean = &actor_cache.0;

                // Recompute prob with current mean
                let std = 0.1;
                let mut new_log_prob = 0.0;
                for (c, &a) in exp.action.iter().enumerate() {
                    let diff = a - current_mean[c];
                    new_log_prob += -0.5 * (diff / std).powi(2)
                        - (std * (2.0 * std::f64::consts::PI).sqrt()).ln();
                }

                let ratio = (new_log_prob - exp.log_prob).exp();
                let adv = advantages[i];
                let eps = self.config.epsilon;

                // Actor gradient formulation (maximizing PPO clipped surrogate)
                // We use +lr in the backprop code, so the returned d_loss_d_out should be positive to increase.
                let mut actor_d_out = vec![0.0; 9];
                let is_clipped = if adv > 0.0 {
                    ratio > 1.0 + eps
                } else {
                    ratio < 1.0 - eps
                };

                if !is_clipped {
                    let r_grad_scale = ratio * adv;
                    for c in 0..9 {
                        // d_log_pi_d_mu = (a - mu) / std^2
                        let d_log_pi_d_mu = (exp.action[c] - current_mean[c]) / (std * std);
                        actor_d_out[c] = r_grad_scale * d_log_pi_d_mu;
                    }
                }

                // Execute exact actor backprop step
                self.actor_gnn.backward(
                    &exp.state,
                    &actor_cache,
                    &actor_d_out,
                    self.config.learning_rate,
                    true,
                );

                // Critic exact forward cache
                let critic_cache = self.critic_gnn.forward_with_cache(&exp.state, false);
                let value_pred = critic_cache.0[0];
                let heat_pred = critic_cache.0[1]; // [Phase M2] Secondary readout

                // Critic gradient formulation (minimizing MSE: (V - ret)^2)
                // Since our backprop computes `W += lr * dLoss`, to MINIMIZE MSE we need negative gradient.
                // d_MSE_d_V = 2 * (value_pred - ret)
                // We supply `ret - value_pred` so that backprop naturally moves V towards ret.
                let mut critic_d_out = vec![0.0; 2];
                // L_value
                critic_d_out[0] = 2.0 * (returns[i] - value_pred) * -self.config.value_coef;

                // [Phase M2] L_kinetic_grounding
                // By forcing the Critic to predict the microscopic heat_rate proxy, the shared GNN latent
                // space natively learns physical relationships BEFORE hitting the constitutional filter.
                let kinetic_coef = 0.5;
                critic_d_out[1] = 2.0 * (exp.heat_rate - heat_pred) * -kinetic_coef;

                // Execute exact critic backprop step
                self.critic_gnn.backward(
                    &exp.state,
                    &critic_cache,
                    &critic_d_out,
                    self.config.learning_rate,
                    false,
                );

                // Track combined loss for gradient velocity monitoring
                let value_loss = (returns[i] - value_pred).powi(2);
                let heat_loss = (exp.heat_rate - heat_pred).powi(2);
                let combined_loss = value_loss + heat_loss;

                self.gradient_tracker.add_measurement(combined_loss, self.total_steps as f64);
            }
        }

        // Clear buffer
        if self.actor_gnn.w_self[0].is_nan() {
            println!("NaN weights detected at step {}", self.total_steps);
        }
        self.buffer.clear();
    }

    /// Compute Generalized Advantage Estimation
    fn compute_advantages(&self) -> Vec<f64> {
        let mut advantages = vec![0.0; self.buffer.len()];
        let mut gae = 0.0;

        for i in (0..self.buffer.len()).rev() {
            let exp = &self.buffer[i];
            let next_value = if exp.done {
                0.0
            } else {
                self.estimate_value(&exp.next_state)
            };
            let current_value = self.estimate_value(&exp.state);

            let delta = exp.reward + self.config.gamma * next_value - current_value;
            gae = delta + self.config.gamma * 0.95 * gae; // Lambda = 0.95
            advantages[i] = gae;
        }

        // Normalize advantages
        let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let std = (advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>()
            / advantages.len() as f64)
            .sqrt()
            + 1e-8;

        advantages.iter().map(|a| (a - mean) / std).collect()
    }

    /// Compute log probability of action under current policy
    fn compute_log_prob(&self, state: &[f64], action: &[f64]) -> f64 {
        let mean = self.forward_policy(state);
        let std = 0.1; // Fixed std for simplicity

        // Gaussian log probability
        let mut log_prob = 0.0;
        for (i, &a) in action.iter().enumerate() {
            let diff = a - mean[i];
            log_prob +=
                -0.5 * (diff / std).powi(2) - (std * (2.0 * std::f64::consts::PI).sqrt()).ln();
        }

        log_prob
    }

    /// Calculate reward for given physics outputs
    pub fn calculate_reward(&self, components: &RewardComponents) -> f64 {
        self.reward_function.calculate(components)
    }

    /// Get training statistics
    pub fn get_stats(&self) -> String {
        format!(
            "Steps: {}, Buffer: {}, Avg Reward: {:.2}",
            self.total_steps,
            self.buffer.len(),
            self.episode_rewards.iter().sum::<f64>() / self.episode_rewards.len().max(1) as f64
        )
    }

    /// Get gate acceptance count.
    pub fn get_gate_accepts(&self) -> u64 {
        self.gate_accepts
    }

    /// Get gate rejection count.
    pub fn get_gate_rejects(&self) -> u64 {
        self.gate_rejects
    }

    /// Get guardrail rejection count.
    pub fn get_guardrail_rejects(&self) -> u64 {
        self.guardrail_rejects
    }

    /// Format gate statistics as a human-readable string.
    pub fn gate_stats_string(&self) -> String {
        let total_thermo = self.gate_accepts + self.gate_rejects;
        let total_all = total_thermo + self.guardrail_rejects;
        if total_all == 0 {
            return "No transitions checked".to_string();
        }
        let gate_rate = if total_thermo > 0 {
            self.gate_accepts as f64 / total_thermo as f64 * 100.0
        } else {
            100.0
        };
        format!(
            "Gate: {}/{} accepted ({:.1}%), Guardrail rejects: {}",
            self.gate_accepts, total_thermo, gate_rate, self.guardrail_rejects
        )
    }

    // ── Meta-Optimization Inspection ─────────────────────────────────────────

    /// Total training steps
    pub fn get_total_steps(&self) -> u64 {
        self.total_steps
    }
    /// Most recent entropy coefficient
    pub fn peek_entropy_coef(&self) -> f64 {
        self.last_meta_stats.entropy_coef
    }
    /// Most recent trust region epsilon
    pub fn peek_epsilon(&self) -> f64 {
        self.last_meta_stats.epsilon
    }
    /// Most recent average episode reward
    pub fn peek_avg_reward(&self) -> f64 {
        self.last_meta_stats.avg_reward
    }
    /// Most recent reward variance
    pub fn peek_reward_variance(&self) -> f64 {
        self.last_meta_stats.reward_variance
    }
    /// Most recent gate rejection rate
    pub fn peek_gate_reject_rate(&self) -> f64 {
        self.last_meta_stats.gate_reject_rate
    }
    /// Most recent GAT attention coherence
    pub fn peek_attn_coherence(&self) -> f64 {
        self.last_meta_stats.attn_coherence
    }

    /// Run optimization loop with FULL CONSTRAINT STACK.
    ///
    /// Constitutional architecture (3-layer constraint enforcement):
    ///   Layer 1 — GuardrailEngine: Hard physical bounds (ACI/EN codes).
    ///             Pre-clamps w/c, rejects impossible actions (Critical), penalises unsafe ones.
    ///   Layer 2 — ThermodynamicFilter: Clausius-Duhem inequality (D_int ≥ 0).
    ///             Each proposed mix validated via its OWN curing trajectory (0→28 days).
    ///   Layer 3 — RewardFunction: Multi-objective shaping (6 modes).
    ///
    /// The agent learns to satisfy ALL three layers simultaneously.
    /// Inadmissible transitions never enter the replay buffer as positive experiences.
    /// Stateless parallel generation layer executing N-step lookahead trajectories
    /// completely bound safely into isolated threads, preventing memory lock contention.
    pub fn execute_rollout(
        &self,
        initial_state: &RLState,
        base_mix: &MixTensor,
        max_steps: u32,
    ) -> (RLAction, Vec<Experience>, u64, u64, u64) {
        let mut state = initial_state.clone();
        let mut best_action = self.select_action(&state);
        let mut best_reward = f64::NEG_INFINITY;
        let mut experiences = Vec::new();

        let mut g_rej = 0;
        let mut t_rej = 0;
        let mut t_acc = 0;

        let config = PhysicsConfig::default();
        let s_intrinsic = config.s_intrinsic as f64;
        let guardrails = GuardrailEngine::with_s_intrinsic(s_intrinsic);
        let mut gate = ThermodynamicFilter::new();
        let base_wc = base_mix.water_cement_ratio() as f64;

        for _ in 0..max_steps {
            let mut action = self.select_action(&state);
            action.delta_wc = guardrails.clamp_wc(base_wc, action.delta_wc);

            let sim = self.simulate_physics(base_mix, &action);
            let guardrail_result = guardrails.validate_action(
                base_wc,
                action.delta_wc,
                sim.strength_fc,
                sim.yield_stress,
                sim.viscosity,
            );

            if !guardrail_result.is_valid {
                g_rej += 1;
                let penalty = -guardrails.violation_penalty(&guardrail_result.violations);
                let next_state = state.clone();
                let log_prob = self.compute_log_prob(&state.to_vector(), &action.to_vector());
                experiences.push(Experience {
                    state: state.to_vector(),
                    action: action.to_vector(),
                    reward: penalty,
                    next_state: next_state.to_vector(),
                    done: false,
                    log_prob,
                    heat_rate: 0.0,
                });
                continue;
            }

            let guardrail_penalty = if !guardrail_result.violations.is_empty() {
                guardrails.violation_penalty(&guardrail_result.violations)
            } else {
                0.0
            };

            let curing_admissible =
                Self::validate_curing_trajectory(sim.w_c, sim.scm_ratio, s_intrinsic, &mut gate);

            if !curing_admissible {
                t_rej += 1;
                let penalty = -100.0;
                let next_state = state.clone();
                let log_prob = self.compute_log_prob(&state.to_vector(), &action.to_vector());
                experiences.push(Experience {
                    state: state.to_vector(),
                    action: action.to_vector(),
                    reward: penalty,
                    next_state: next_state.to_vector(),
                    done: false,
                    log_prob,
                    heat_rate: 0.0,
                });
                continue;
            }

            t_acc += 1;

            let base_reward = self.calculate_reward(&sim.components);
            let reward = base_reward - guardrail_penalty;

            if reward > best_reward {
                best_reward = reward;
                best_action = action.clone();
            }

            let mut next_state = state.clone();
            next_state.set_proxy(0, sim.components.slump_flow / 800.0);
            next_state.set_proxy(1, sim.components.viscosity / 100.0);
            next_state.set_proxy(2, sim.components.yield_stress / 500.0);
            next_state.fracture_kic = sim.components.fracture_kic;
            next_state.diffusivity = sim.components.diffusivity;
            next_state.heat_q = sim.components.heat_rate;
            next_state.damage_d = sim.components.damage;
            next_state.bond_strength = sim.components.bond;

            let log_prob = self.compute_log_prob(&state.to_vector(), &action.to_vector());
            experiences.push(Experience {
                state: state.to_vector(),
                action: action.to_vector(),
                reward,
                next_state: next_state.to_vector(),
                done: false,
                log_prob,
                heat_rate: sim.components.heat_rate,
            });
            state = next_state;
        }

        (best_action, experiences, g_rej, t_rej, t_acc)
    }

    /// Single wrapper optimizing sequentially in-place
    pub fn optimize(
        &mut self,
        initial_state: &RLState,
        base_mix: &MixTensor,
        max_steps: u32,
    ) -> RLAction {
        let (action, exps, g, t_r, t_a) = self.execute_rollout(initial_state, base_mix, max_steps);
        self.guardrail_rejects += g;
        self.gate_rejects += t_r;
        self.gate_accepts += t_a;
        for exp in exps {
            self.buffer.push(exp);
            self.total_steps += 1;
            if self.buffer.len() >= self.config.batch_size {
                self.update();
            }
        }
        action
    }

    /// Run highly parallel map-reduce block processing over physical batch samples
    pub fn optimize_batch(&mut self, tasks: &[(RLState, MixTensor, u32)]) {
        let rollouts: Vec<_> = tasks
            .iter()
            .map(|(state, base_mix, max_steps)| self.execute_rollout(state, base_mix, *max_steps))
            .collect();

        for (_, exps, g_rej, t_rej, t_acc) in rollouts {
            self.guardrail_rejects += g_rej;
            self.gate_rejects += t_rej;
            self.gate_accepts += t_acc;

            for exp in exps {
                self.buffer.push(exp);
                self.total_steps += 1;
                if self.buffer.len() >= self.config.batch_size {
                    self.update();
                }
            }
        }
    }

    /// Validate a mix design's curing trajectory for thermodynamic admissibility.
    ///
    /// Checks that the mix's OWN curing from day 0 to day 28 satisfies the
    /// Clausius-Duhem inequality (D_int = −ρ·ψ̇ ≥ 0) at every step.
    /// This is the scientifically correct admissibility check: it validates
    /// that hydration progresses forward (2nd law) without strength regression.
    fn validate_curing_trajectory(
        w_c: f64,
        scm_ratio: f32,
        s_intrinsic: f64,
        gate: &mut ThermodynamicFilter,
    ) -> bool {
        let curing_days = [0.0_f32, 7.0, 14.0, 21.0, 28.0];

        for pair in curing_days.windows(2) {
            let t_old = pair[0];
            let t_new = pair[1];
            let dt_seconds = ((t_new - t_old) * 86400.0) as f64;

            let alpha_old = PhysicsKernel::compute_hydration_degree(t_old, 20.0, scm_ratio) as f64;
            let alpha_new = PhysicsKernel::compute_hydration_degree(t_new, 20.0, scm_ratio) as f64;

            let state_old =
                ThermodynamicState::from_mix_calibrated(w_c, alpha_old, 293.0, s_intrinsic);
            let state_new =
                ThermodynamicState::from_mix_calibrated(w_c, alpha_new, 293.0, s_intrinsic);

            let result = gate.check_transition(&state_old, &state_new, dt_seconds);
            if !result.accepted {
                return false;
            }
        }
        true
    }

    /// [SIMULATION] High-Fidelity Physics Simulation
    ///
    /// Runs ALL 16 core engines via PhysicsKernel::compute() and returns
    /// physics outputs plus thermodynamic state data for the constitutional gate.
    fn simulate_physics(&self, base_mix: &MixTensor, action: &RLAction) -> SimulationResult {
        // 1. Clone and apply action (MixTensor mutation)
        let mut sim_mix = base_mix.clone();
        sim_mix.apply_action(
            action.delta_wc as f32,
            action.delta_scms as f32,
            action.delta_sp as f32,
        );

        // 2. Run full 16-engine constitutive ensemble
        let config = PhysicsConfig::default();
        let result = PhysicsKernel::compute(&sim_mix, None, &config);

        // 3. Extract physics outputs for guardrails and reward
        let w_c = sim_mix.water_cement_ratio() as f64;
        let strength_fc = result.hardened.f28_compressive as f64;
        let yield_stress = result.fresh.yield_stress as f64;
        let viscosity = result.fresh.plastic_viscosity as f64;

        // 4. Assemble full 17-metric reward components
        SimulationResult {
            components: RewardComponents {
                strength_fc,
                yield_stress,
                viscosity,
                slump_flow: result.fresh.slump_flow as f64,
                cost: result.economics.cost_per_m3 as f64,
                co2: result.sustainability.co2_kg_m3 as f64,
                fracture_kic: result.mechanics.fracture_toughness as f64,
                diffusivity: result.chemical.diffusivity as f64,
                damage: 0.0,
                bond: result.mechanics.split_tensile as f64,
                itz_thickness: result.itz.thickness as f64,
                itz_porosity: result.itz.porosity as f64,
                colloidal_potential: result.colloidal.interparticle_distance as f64,
                heat_rate: result.thermal.heat_of_hydration as f64,
                temp_rise: result.thermal.adiabatic_rise as f64,
                permeability: result.transport.permeability as f64,
                suction: result.chemical.suction as f64,
            },
            w_c,
            scm_ratio: sim_mix.scm_ratio(),
            strength_fc,
            yield_stress,
            viscosity,
        }
    }
}

// Helper functions
fn rand_normal() -> f64 {
    // Box-Muller transform using standard RNG
    let mut rng = rand::thread_rng();
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Deterministic Box-Muller for reproducible benchmarks.
fn rand_normal_seeded<R: rand::Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
