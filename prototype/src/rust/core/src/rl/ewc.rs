// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! Elastic Weight Consolidation (EWC) for Continual Learning
//!
//! Prevents Catastrophic Forgetting (CF) in the GNN-PPO Agent. After training on
//! a source domain (e.g., Standard Cementitious Mixes), the Fisher Information Matrix
//! (FIM) is computed for the critical weights. When training subsequently shifts to
//! a new domain (e.g., Drifting Pump / Martian Regolith), an additional penalty term
//! anchors the new weights close to their original positions in proportion to their
//! information content (a.k.a. importance to the original task).
//!
//! EWC Loss:  L_{EWC} = L_{task_B} + (λ/2) Σ_i F_i (θ_i - θ*_i)²
//!
//! Where F_i is the Fisher Information (importance) of weight i,
//! θ*_i are the anchor (previous domain) weights,
//! and λ (lambda) controls the consolidation strength.

/// Fisher Information based weight importance tracker.
/// Stores anchor weights θ* and their estimated diagonal FIM values F_i.
pub struct EwcPenalty {
    /// Flattened vector of anchor weights from the previous domain
    pub anchor_weights: Vec<f64>,
    /// Diagonal Fisher Information (importance) per weight
    pub fisher_diagonals: Vec<f64>,
    /// Regularisation strength (how hard we resist forgetting)
    pub lambda: f64,
}

impl EwcPenalty {
    /// Build an EwcPenalty given a set of trained weights.
    /// Fisher diagonal is estimated from the squared gradient magnitude (a common
    /// online FIM approximation that requires only first-order information).
    pub fn from_gradients(weights: &[f64], squared_gradients: &[f64], lambda: f64) -> Self {
        assert_eq!(
            weights.len(),
            squared_gradients.len(),
            "EWC: weight and gradient vectors must have equal length"
        );
        Self {
            anchor_weights: weights.to_vec(),
            fisher_diagonals: squared_gradients.to_vec(),
            lambda,
        }
    }

    /// Compute the EWC regularisation penalty for the supplied current weights.
    ///
    /// L_{EWC} = (λ/2) Σ_i F_i (θ_i - θ*_i)²
    pub fn penalty(&self, current_weights: &[f64]) -> f64 {
        assert_eq!(
            current_weights.len(),
            self.anchor_weights.len(),
            "EWC penalty: weight vector length mismatch"
        );
        let raw: f64 = self
            .fisher_diagonals
            .iter()
            .zip(current_weights.iter())
            .zip(self.anchor_weights.iter())
            .map(|((f, theta), theta_star)| f * (theta - theta_star).powi(2))
            .sum();
        (self.lambda / 2.0) * raw
    }

    /// Compute the per-weight gradient of the EWC penalty for backprop injection.
    ///
    /// dL/dθ_i = λ F_i (θ_i - θ*_i)
    pub fn gradients(&self, current_weights: &[f64]) -> Vec<f64> {
        self.fisher_diagonals
            .iter()
            .zip(current_weights.iter())
            .zip(self.anchor_weights.iter())
            .map(|((f, theta), theta_star)| self.lambda * f * (theta - theta_star))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_zero_penalty_at_anchor() {
        // If current weights == anchor weights, penalty must be zero
        let weights = vec![0.5, 1.2, -0.3, 0.8];
        let grads_sq = vec![1.0, 2.0, 0.5, 3.0];
        let ewc = EwcPenalty::from_gradients(&weights, &grads_sq, 0.5);

        let penalty = ewc.penalty(&weights);
        assert!(
            penalty.abs() < 1e-10,
            "Penalty must be zero at anchor weights, got: {penalty}"
        );
    }

    #[test]
    fn test_ewc_penalty_grows_with_drift() {
        let anchor = vec![0.0_f64; 10];
        let fisher = vec![1.0_f64; 10]; // uniform importance
        let ewc = EwcPenalty::from_gradients(&anchor, &fisher, 1.0);

        let slight_drift: Vec<f64> = vec![0.1; 10];
        let heavy_drift: Vec<f64> = vec![1.0; 10];

        let p_slight = ewc.penalty(&slight_drift);
        let p_heavy = ewc.penalty(&heavy_drift);

        assert!(
            p_heavy > p_slight,
            "Heavier drift must cause larger EWC penalty"
        );
        // Analytical: (1.0/2) * 10 * 1.0^2 = 5.0
        assert!(
            (p_heavy - 5.0).abs() < 1e-9,
            "Heavy drift penalty = {p_heavy}"
        );
    }

    #[test]
    fn test_ewc_gradient_direction() {
        // Gradient should point FROM anchor, pushing weights back
        let anchor = vec![0.0_f64; 3];
        let fisher = vec![1.0_f64; 3];
        let ewc = EwcPenalty::from_gradients(&anchor, &fisher, 1.0);

        let current = vec![1.0, -1.0, 0.5];
        let grads = ewc.gradients(&current);

        // Each gradient should have same sign as (theta - theta*)
        assert!(
            grads[0] > 0.0,
            "Grad should be positive when theta > anchor"
        );
        assert!(
            grads[1] < 0.0,
            "Grad should be negative when theta < anchor"
        );
        assert!(
            grads[2] > 0.0,
            "Grad should be positive when theta > anchor"
        );
    }
}
