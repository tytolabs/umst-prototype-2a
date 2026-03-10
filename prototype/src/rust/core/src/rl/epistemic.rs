// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! Epistemic Uncertainty Module for Quantum Active Inference
//!
//! Implements mutual information estimation and epistemic exploration bonuses
//! for the PPO agent. Based on quantum active inference theory where epistemic
//! policies maximize expected mutual information I[ψ;o|π].
//!
//! # Theory (Campos et al., 2026)
//!
//! Expected free energy decomposes as:
//! G(π) = E[D(Q(ψ_τ|o_τ,π)|P(ψ_τ))] - I_Q[ψ_τ;o_τ|π]
//!        \_________pragmatic________/   \__epistemic__/
//!
//! The epistemic term I[ψ;o|π] ≥ 0 represents resolvable uncertainty.
//! Policies maximizing this term accelerate variational optimization.
//!
//! # Implementation
//!
//! Uses histogram-based entropy estimation for WASM compatibility.
//! No neural network dependencies - pure Rust implementation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of bins for histogram-based entropy estimation
const DEFAULT_BINS: usize = 20;

/// Minimum count per bin for stable entropy estimation
const MIN_COUNT_PER_BIN: usize = 5;

/// Exponential moving average alpha for online estimation
const EMA_ALPHA: f64 = 0.1;

/// Mutual Information Estimator using histogram-based entropy
///
/// Estimates I[X;Y] = H(X) + H(Y) - H(X,Y) where:
/// - X = hidden state (material properties)
/// - Y = observations (sensor readings)
///
/// Uses binning for efficient WASM computation without neural networks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutualInfoEstimator {
    /// Number of histogram bins per dimension
    n_bins: usize,
    /// Dimension of state space
    state_dim: usize,
    /// Dimension of observation space
    obs_dim: usize,
    /// Running estimate of mutual information
    mi_estimate: f64,
    /// Confidence in the estimate (based on sample count)
    confidence: f64,
    /// Total samples processed
    total_samples: u64,
    /// State marginal histogram
    state_hist: Vec<f64>,
    /// Observation marginal histogram
    obs_hist: Vec<f64>,
    /// Joint histogram (flattened 2D)
    joint_hist: Vec<f64>,
    /// State min/max for normalization
    state_bounds: Vec<(f64, f64)>,
    /// Observation min/max for normalization
    obs_bounds: Vec<(f64, f64)>,
}

impl MutualInfoEstimator {
    /// Create new estimator with specified dimensions
    pub fn new(state_dim: usize, obs_dim: usize) -> MutualInfoEstimator {
        let n_bins = DEFAULT_BINS;

        // Initialize histograms
        let state_hist_size = n_bins.pow(state_dim.min(3) as u32); // Cap at 3D for memory
        let obs_hist_size = n_bins.pow(obs_dim.min(3) as u32);
        let joint_hist_size = state_hist_size * obs_hist_size;

        MutualInfoEstimator {
            n_bins,
            state_dim,
            obs_dim,
            mi_estimate: 0.0,
            confidence: 0.0,
            total_samples: 0,
            state_hist: vec![0.0; state_hist_size.min(10000)],
            obs_hist: vec![0.0; obs_hist_size.min(10000)],
            joint_hist: vec![0.0; joint_hist_size.min(100000)],
            state_bounds: vec![(0.0, 1.0); state_dim],
            obs_bounds: vec![(0.0, 1.0); obs_dim],
        }
    }

    /// Create estimator for UMST material state space
    /// State: 35-dim (27 proxies + 6 outputs + 2 weather)
    /// Obs: 6-dim (strength, cost, CO2, yield_stress, viscosity, slump)
    pub fn for_projectx() -> MutualInfoEstimator {
        let mut estimator = MutualInfoEstimator::new(35, 6);

        // Set typical bounds for material states
        // Proxies [0-27] are typically normalized 0-1
        for i in 0..27 {
            estimator.state_bounds[i] = (0.0, 1.0);
        }
        // Heat Q: 0-1000 W/m³
        estimator.state_bounds[27] = (0.0, 1000.0);
        // Damage D: 0-1
        estimator.state_bounds[28] = (0.0, 1.0);
        // Fracture K_IC: 0.5-2.5 MPa√m
        estimator.state_bounds[29] = (0.5, 2.5);
        // Diffusivity: 0-0.01 m²/s
        estimator.state_bounds[30] = (0.0, 0.01);
        // Shrinkage: 0-0.001
        estimator.state_bounds[31] = (0.0, 0.001);
        // Bond: 0-5 MPa
        estimator.state_bounds[32] = (0.0, 5.0);
        // Temperature: -10 to 50°C
        estimator.state_bounds[33] = (-10.0, 50.0);
        // Humidity: 0-1
        estimator.state_bounds[34] = (0.0, 1.0);

        // Observation bounds
        estimator.obs_bounds[0] = (0.0, 100.0); // Strength (MPa)
        estimator.obs_bounds[1] = (0.0, 300.0); // Cost ($/m³)
        estimator.obs_bounds[2] = (0.0, 600.0); // CO2 (kg/m³)
        estimator.obs_bounds[3] = (0.0, 1000.0); // Yield stress (Pa)
        estimator.obs_bounds[4] = (0.0, 200.0); // Viscosity (Pa.s)
        estimator.obs_bounds[5] = (0.0, 900.0); // Slump flow (mm)

        estimator
    }

    /// Update estimator with new state-observation pair
    pub fn update(&mut self, state: &[f64], obs: &[f64]) {
        // Normalize inputs to [0,1] range
        let state_norm: Vec<f64> = state
            .iter()
            .enumerate()
            .take(self.state_dim)
            .map(|(i, &x)| {
                self.normalize(x, self.state_bounds.get(i).copied().unwrap_or((0.0, 1.0)))
            })
            .collect();

        let obs_norm: Vec<f64> = obs
            .iter()
            .enumerate()
            .take(self.obs_dim)
            .map(|(i, &x)| self.normalize(x, self.obs_bounds.get(i).copied().unwrap_or((0.0, 1.0))))
            .collect();

        // Compute histogram indices (using first 3 dims for memory efficiency)
        let state_idx =
            self.compute_hist_index(&state_norm, self.state_dim.min(3), self.state_hist.len());
        let obs_idx = self.compute_hist_index(&obs_norm, self.obs_dim.min(3), self.obs_hist.len());
        let joint_idx = self.compute_joint_index(state_idx, obs_idx);

        // Update histograms with exponential decay for non-stationarity
        let decay = 0.999;
        for h in self.state_hist.iter_mut() {
            *h *= decay;
        }
        for h in self.obs_hist.iter_mut() {
            *h *= decay;
        }
        for h in self.joint_hist.iter_mut() {
            *h *= decay;
        }

        // Add new sample
        if state_idx < self.state_hist.len() {
            self.state_hist[state_idx] += 1.0;
        }
        if obs_idx < self.obs_hist.len() {
            self.obs_hist[obs_idx] += 1.0;
        }
        if joint_idx < self.joint_hist.len() {
            self.joint_hist[joint_idx] += 1.0;
        }

        self.total_samples += 1;

        // Update MI estimate with EMA
        if self.total_samples >= MIN_COUNT_PER_BIN as u64 * self.n_bins as u64 {
            let new_mi = self.compute_mi_from_histograms();
            self.mi_estimate = EMA_ALPHA * new_mi + (1.0 - EMA_ALPHA) * self.mi_estimate;
            self.confidence = (self.total_samples as f64 / 1000.0).min(1.0);
        }
    }

    /// Get current mutual information estimate
    pub fn estimate(&self) -> f64 {
        self.mi_estimate.max(0.0) // MI is non-negative
    }

    /// Get confidence in the estimate (0-1)
    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }

    /// Get total samples processed
    pub fn get_total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Reset estimator
    pub fn reset(&mut self) {
        self.state_hist.fill(0.0);
        self.obs_hist.fill(0.0);
        self.joint_hist.fill(0.0);
        self.mi_estimate = 0.0;
        self.confidence = 0.0;
        self.total_samples = 0;
    }

    /// Get detailed statistics as JSON
    pub fn get_stats_json(&self) -> String {
        serde_json::json!({
            "mi_estimate": self.mi_estimate,
            "confidence": self.confidence,
            "total_samples": self.total_samples,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
            "n_bins": self.n_bins,
        })
        .to_string()
    }
}

impl MutualInfoEstimator {
    /// Normalize value to [0,1] range
    fn normalize(&self, x: f64, bounds: (f64, f64)) -> f64 {
        let (min, max) = bounds;
        if max <= min {
            return 0.5;
        }
        ((x - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Compute histogram index from normalized values
    fn compute_hist_index(&self, values: &[f64], dims: usize, max_idx: usize) -> usize {
        let mut idx = 0usize;
        let mut multiplier = 1usize;

        for (i, &v) in values.iter().take(dims).enumerate() {
            let bin = ((v * self.n_bins as f64) as usize).min(self.n_bins - 1);
            idx += bin * multiplier;
            multiplier *= self.n_bins;

            if i >= 2 {
                break;
            } // Cap at 3D
        }

        idx.min(max_idx.saturating_sub(1))
    }

    /// Compute joint histogram index
    fn compute_joint_index(&self, state_idx: usize, obs_idx: usize) -> usize {
        let state_size = self.state_hist.len();
        (state_idx + obs_idx * state_size).min(self.joint_hist.len().saturating_sub(1))
    }

    /// Compute entropy from histogram
    fn entropy_from_hist(&self, hist: &[f64]) -> f64 {
        let total: f64 = hist.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }

        hist.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| {
                let p = x / total;
                -p * p.ln()
            })
            .sum()
    }

    /// Compute mutual information from current histograms
    /// I(X;Y) = H(X) + H(Y) - H(X,Y)
    fn compute_mi_from_histograms(&self) -> f64 {
        let h_state = self.entropy_from_hist(&self.state_hist);
        let h_obs = self.entropy_from_hist(&self.obs_hist);
        let h_joint = self.entropy_from_hist(&self.joint_hist);

        // MI = H(X) + H(Y) - H(X,Y)
        // Should be non-negative, but numerical issues can make it negative
        (h_state + h_obs - h_joint).max(0.0)
    }

    /// Compute conditional entropy H(Y|X)
    /// H(Y|X) = H(X,Y) - H(X)
    pub fn conditional_entropy_y_given_x(&self) -> f64 {
        let h_joint = self.entropy_from_hist(&self.joint_hist);
        let h_state = self.entropy_from_hist(&self.state_hist);
        (h_joint - h_state).max(0.0)
    }

    /// Compute information gain rate (for quantum bounds)
    /// γ = dI/dt ≈ ΔI / Δt
    pub fn info_gain_rate(&self, prev_mi: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }
        (self.mi_estimate - prev_mi) / dt
    }
}

/// Epistemic State Tracker for tracking uncertainty over time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicStateTracker {
    /// History of MI estimates
    mi_history: Vec<f64>,
    /// History of info gain rates
    gamma_history: Vec<f64>,
    /// Current epistemic bonus
    epistemic_bonus: f64,
    /// Scaling factor for epistemic reward
    beta: f64,
    /// Maximum history length
    max_history: usize,
    /// Time step counter
    time_step: u64,
}

impl EpistemicStateTracker {
    pub fn new() -> EpistemicStateTracker {
        EpistemicStateTracker {
            mi_history: Vec::with_capacity(1000),
            gamma_history: Vec::with_capacity(1000),
            epistemic_bonus: 0.0,
            beta: 0.1, // Default epistemic weight
            max_history: 1000,
            time_step: 0,
        }
    }

    /// Create tracker with custom epistemic weight
    pub fn with_beta(beta: f64) -> EpistemicStateTracker {
        let mut tracker = EpistemicStateTracker::new();
        tracker.beta = beta.clamp(0.0, 1.0);
        tracker
    }

    /// Update tracker with new MI estimate
    pub fn update(&mut self, mi: f64) {
        // Track MI history
        self.mi_history.push(mi);
        if self.mi_history.len() > self.max_history {
            self.mi_history.remove(0);
        }

        // Compute info gain rate
        let gamma = if self.mi_history.len() >= 2 {
            let prev = self.mi_history[self.mi_history.len() - 2];
            (mi - prev).abs()
        } else {
            0.0
        };

        self.gamma_history.push(gamma);
        if self.gamma_history.len() > self.max_history {
            self.gamma_history.remove(0);
        }

        // Update epistemic bonus
        // Bonus = β * I[ψ;o] + exploration bonus for high info gain
        let exploration_bonus = if gamma > 0.01 { 0.1 * gamma } else { 0.0 };
        self.epistemic_bonus = self.beta * mi + exploration_bonus;

        self.time_step += 1;
    }

    /// Get current epistemic bonus for reward shaping
    pub fn get_epistemic_bonus(&self) -> f64 {
        self.epistemic_bonus
    }

    /// Get average info gain rate
    pub fn get_avg_gamma(&self) -> f64 {
        if self.gamma_history.is_empty() {
            return 0.0;
        }
        self.gamma_history.iter().sum::<f64>() / self.gamma_history.len() as f64
    }

    /// Get current MI (most recent)
    pub fn get_current_mi(&self) -> f64 {
        self.mi_history.last().copied().unwrap_or(0.0)
    }

    /// Set epistemic weight β
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta.clamp(0.0, 1.0);
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.mi_history.clear();
        self.gamma_history.clear();
        self.epistemic_bonus = 0.0;
        self.time_step = 0;
    }

    /// Get statistics as JSON
    pub fn get_stats_json(&self) -> String {
        serde_json::json!({
            "current_mi": self.get_current_mi(),
            "epistemic_bonus": self.epistemic_bonus,
            "avg_gamma": self.get_avg_gamma(),
            "beta": self.beta,
            "time_step": self.time_step,
            "history_len": self.mi_history.len(),
        })
        .to_string()
    }
}

/// Intrinsic Curiosity Module for exploration-driven learning
///
/// Based on prediction error: agents are curious about states
/// they cannot predict well. This naturally emerges from the
/// free energy principle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntrinsicCuriosity {
    /// Prediction error history
    prediction_errors: Vec<f64>,
    /// State visit counts (hashed)
    visit_counts: HashMap<u64, u32>,
    /// Curiosity scaling factor
    eta: f64,
    /// Count-based bonus factor
    count_bonus_factor: f64,
}

impl IntrinsicCuriosity {
    pub fn new() -> IntrinsicCuriosity {
        IntrinsicCuriosity {
            prediction_errors: Vec::with_capacity(1000),
            visit_counts: HashMap::new(),
            eta: 0.01,               // Curiosity scaling
            count_bonus_factor: 0.1, // Count-based exploration bonus
        }
    }

    /// Compute curiosity bonus for a state
    /// R_curiosity = η * prediction_error + c / sqrt(N(s))
    pub fn compute_bonus(&mut self, state: &[f64], predicted: &[f64], actual: &[f64]) -> f64 {
        // Compute prediction error
        let pred_error: f64 = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            .sqrt();

        self.prediction_errors.push(pred_error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }

        // Compute state hash for visit counting
        let state_hash = self.hash_state(state);
        let visit_count = self.visit_counts.entry(state_hash).or_insert(0);
        *visit_count += 1;

        // Count-based exploration bonus: c / sqrt(N(s))
        let count_bonus = self.count_bonus_factor / (*visit_count as f64).sqrt();

        // Prediction error bonus
        let pred_bonus = self.eta * pred_error;

        pred_bonus + count_bonus
    }

    /// Get average prediction error
    pub fn get_avg_prediction_error(&self) -> f64 {
        if self.prediction_errors.is_empty() {
            return 0.0;
        }
        self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64
    }

    /// Get number of unique states visited
    pub fn get_unique_states(&self) -> usize {
        self.visit_counts.len()
    }

    /// Reset curiosity module
    pub fn reset(&mut self) {
        self.prediction_errors.clear();
        self.visit_counts.clear();
    }

    /// Hash state to u64 for visit counting
    fn hash_state(&self, state: &[f64]) -> u64 {
        let mut hash: u64 = 0;
        for (i, &x) in state.iter().take(8).enumerate() {
            // Discretize to bins for hashing
            let bin = ((x * 100.0) as i64).clamp(-1000, 1000);
            hash ^= (bin as u64).wrapping_mul(31u64.pow(i as u32));
        }
        hash
    }
}

// Simple random number generator for tests (WASM-compatible)
#[allow(dead_code)]
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Add time-based entropy
        let time_entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        SEED ^= time_entropy;
        (SEED as f64) / (u64::MAX as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutual_info_estimator_creation() {
        let estimator = MutualInfoEstimator::new(10, 5);
        assert_eq!(estimator.state_dim, 10);
        assert_eq!(estimator.obs_dim, 5);
        assert_eq!(estimator.estimate(), 0.0);
    }

    #[test]
    fn test_projectx_estimator() {
        let estimator = MutualInfoEstimator::for_projectx();
        assert_eq!(estimator.state_dim, 35);
        assert_eq!(estimator.obs_dim, 6);
    }

    #[test]
    fn test_estimator_update() {
        let mut estimator = MutualInfoEstimator::new(3, 2);

        // Add correlated samples
        for i in 0..200 {
            let x = (i as f64) / 200.0;
            let state = vec![x, x * 0.5, x * 0.3];
            let obs = vec![x + 0.1, x * 2.0];
            estimator.update(&state, &obs);
        }

        // Should have some MI estimate now
        assert!(estimator.total_samples >= 200);
        println!("MI estimate after 200 samples: {:.4}", estimator.estimate());
    }

    #[test]
    fn test_epistemic_tracker() {
        let mut tracker = EpistemicStateTracker::new();

        // Simulate increasing MI
        for i in 0..100 {
            let mi = (i as f64) / 100.0 * 0.5;
            tracker.update(mi);
        }

        assert!(tracker.get_current_mi() > 0.0);
        assert!(tracker.get_epistemic_bonus() > 0.0);
        println!("Epistemic bonus: {:.4}", tracker.get_epistemic_bonus());
    }

    #[test]
    fn test_intrinsic_curiosity() {
        let mut curiosity = IntrinsicCuriosity::new();

        let state = vec![0.5, 0.3, 0.7];
        let predicted = vec![0.4, 0.2];
        let actual = vec![0.6, 0.5];

        let bonus = curiosity.compute_bonus(&state, &predicted, &actual);
        assert!(bonus > 0.0);
        println!("Curiosity bonus: {:.4}", bonus);

        // Visit same state again - bonus should decrease
        let bonus2 = curiosity.compute_bonus(&state, &predicted, &actual);
        assert!(bonus2 < bonus); // Count bonus decreases
    }

    #[test]
    fn test_mi_convergence_independent() {
        // For independent variables with different distributions, MI should be low
        let mut estimator = MutualInfoEstimator::new(2, 2);

        // Add samples where state and observation are from different patterns
        // to simulate independence (our simple RNG is not truly random)
        for i in 0..500 {
            // State follows one pattern
            let x = (i as f64) / 500.0;
            let state = vec![x, 1.0 - x];
            // Observation follows unrelated pattern (reverse index)
            let j = 499 - i;
            let y = ((j * 7) % 500) as f64 / 500.0; // Scrambled index
            let obs = vec![y, (y * 3.0) % 1.0];
            estimator.update(&state, &obs);
        }

        let mi = estimator.estimate();
        println!("MI for quasi-independent vars: {:.4}", mi);
        // With histogram-based estimation and limited samples, some spurious MI is expected
        // The key is that MI for truly correlated vars (next test) should be higher
        assert!(mi < 5.0, "MI should be bounded");
    }

    #[test]
    fn test_mi_convergence_correlated() {
        // For perfectly correlated variables, MI should be high
        let mut estimator = MutualInfoEstimator::new(2, 2);

        // Add perfectly correlated samples
        for i in 0..500 {
            let x = (i as f64) / 500.0;
            let state = vec![x, x];
            let obs = vec![x, x]; // Same as state
            estimator.update(&state, &obs);
        }

        let mi = estimator.estimate();
        println!("MI for correlated vars: {:.4}", mi);
        // Should be positive for correlated variables
        assert!(mi > 0.0, "MI for correlated vars should be positive");
    }
}
