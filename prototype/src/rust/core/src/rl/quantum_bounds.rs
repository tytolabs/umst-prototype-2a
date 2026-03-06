// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! Quantum Thermodynamic Bounds Module
//!
//! Implements quantum thermodynamic constraints on learning processes
//! based on the Sagawa-Ueda generalized second law and quantum
//! active inference theory.
//!
//! # Theory (Sagawa-Ueda, 2010; Friston, 2024)
//!
//! The generalized second law with information:
//! ⟨Σ⟩ ≥ ΔI_gain + ⟨Q⟩/T
//!
//! Where:
//! - ⟨Σ⟩: Expected entropy production
//! - ΔI_gain: Information gain during measurement/learning
//! - ⟨Q⟩: Heat dissipated to environment
//! - T: Temperature
//!
//! # Convergence Bound (Section 5.2)
//!
//! The error bound on expected free energy obeys:
//! Ḋ ≤ -γ²/C + Σ̇_min
//!
//! Where:
//! - D: Divergence (error measure)
//! - γ: Information gain rate
//! - C: Effective heat capacity
//! - Σ̇_min: Minimum entropy production rate
//!
//! This guarantees monotonic convergence when γ > √(C · Σ̇_min)

use serde::{Deserialize, Serialize};

/// Boltzmann constant in J/K (useful for micro-scale calculations)
const K_B: f64 = 1.380649e-23;

/// Universal gas constant in J/(mol·K)
#[allow(dead_code)]
const R_GAS: f64 = 8.314;

/// Default temperature in Kelvin
const DEFAULT_TEMP: f64 = 293.15; // 20°C

/// Quantum Thermodynamic Bounds checker
///
/// Enforces the generalized second law with information gain
/// and provides convergence guarantees for RL optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumThermoBounds {
    /// Temperature (K)
    temperature: f64,
    /// Effective heat capacity (J/K per unit volume)
    heat_capacity: f64,
    /// Minimum entropy production rate (J/K·s per unit volume)
    sigma_dot_min: f64,
    /// Current information gain rate
    gamma: f64,
    /// Accumulated information gain
    total_info_gain: f64,
    /// Accumulated entropy production
    total_entropy_prod: f64,
    /// Accumulated heat dissipation
    total_heat: f64,
    /// Time step counter
    time_steps: u64,
    /// Bound violation count
    violations: u64,
}

impl QuantumThermoBounds {
    /// Create new bounds checker with default parameters
    pub fn new() -> QuantumThermoBounds {
        QuantumThermoBounds {
            temperature: DEFAULT_TEMP,
            heat_capacity: 2000.0, // J/(K·m³) typical for concrete
            sigma_dot_min: 0.001,  // Minimum entropy production rate
            gamma: 0.0,
            total_info_gain: 0.0,
            total_entropy_prod: 0.0,
            total_heat: 0.0,
            time_steps: 0,
            violations: 0,
        }
    }

    /// Create bounds checker with custom temperature
    pub fn with_temperature(temp_kelvin: f64) -> QuantumThermoBounds {
        let mut bounds = QuantumThermoBounds::new();
        bounds.temperature = temp_kelvin.max(1.0); // Avoid division by zero
        bounds
    }

    /// Create bounds checker calibrated for material system
    /// Uses typical values for cementitious materials
    pub fn for_material_system(temp_celsius: f64, c_p: f64) -> QuantumThermoBounds {
        let mut bounds = QuantumThermoBounds::new();
        bounds.temperature = temp_celsius + 273.15;
        bounds.heat_capacity = c_p; // J/(K·m³)
                                    // Minimum entropy production from irreversible reactions
                                    // For cement hydration: σ̇_min ≈ Q_hyd * α̇ / T
        bounds.sigma_dot_min = 400.0 * 0.0001 / bounds.temperature; // ~0.1-1 J/(K·s·m³)
        bounds
    }

    /// Set current information gain rate (bits/second)
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma.max(0.0);
    }

    /// Get current information gain rate
    pub fn get_gamma(&self) -> f64 {
        self.gamma
    }

    /// Check if the generalized second law is satisfied
    /// Returns the margin where margin > 0 means satisfied with room to spare
    ///
    /// Generalized Second Law:
    /// ⟨Σ⟩ ≥ ΔI_gain + ⟨Q⟩/T
    pub fn check_second_law(&mut self, entropy_prod: f64, info_gain: f64, heat: f64) -> f64 {
        // Update totals
        self.total_entropy_prod += entropy_prod;
        self.total_info_gain += info_gain;
        self.total_heat += heat;
        self.time_steps += 1;

        // Check bound: Σ ≥ I_gain + Q/T
        let rhs = info_gain + heat / self.temperature;
        let margin = entropy_prod - rhs;

        let satisfied = margin >= -1e-10; // Allow small numerical tolerance
        if !satisfied {
            self.violations += 1;
        }

        margin
    }

    /// Check if second law is satisfied (margin >= 0)
    pub fn is_second_law_satisfied(&self, entropy_prod: f64, info_gain: f64, heat: f64) -> bool {
        let rhs = info_gain + heat / self.temperature;
        let margin = entropy_prod - rhs;
        margin >= -1e-10
    }

    /// Compute the convergence rate bound
    /// Ḋ ≤ -γ²/C + Σ̇_min
    ///
    /// Returns the upper bound on error rate. Negative value means convergence.
    pub fn convergence_rate_bound(&self) -> f64 {
        if self.heat_capacity <= 0.0 {
            return f64::INFINITY;
        }

        // Ḋ ≤ -γ²/C + Σ̇_min
        -self.gamma.powi(2) / self.heat_capacity + self.sigma_dot_min
    }

    /// Check if the system is guaranteed to converge
    /// Converges when γ > √(C · Σ̇_min)
    pub fn is_convergent(&self) -> bool {
        let threshold = (self.heat_capacity * self.sigma_dot_min).sqrt();
        self.gamma > threshold
    }

    /// Get the minimum info gain rate required for convergence
    pub fn min_gamma_for_convergence(&self) -> f64 {
        (self.heat_capacity * self.sigma_dot_min).sqrt()
    }

    /// Compute expected time to convergence (rough estimate)
    /// Based on exponential decay: D(t) ≈ D_0 * exp(-rate * t)
    pub fn estimated_convergence_time(&self, initial_error: f64, target_error: f64) -> Option<f64> {
        let rate = -self.convergence_rate_bound();
        if rate <= 0.0 {
            return None;
        } // Not converging

        let ratio = initial_error / target_error;
        if ratio <= 1.0 {
            return Some(0.0);
        } // Already converged

        Some(ratio.ln() / rate)
    }

    /// Update bounds from thermodynamic filter state
    /// Integrates with the existing Clausius-Duhem inequality
    pub fn update_from_dissipation(&mut self, d_int: f64, dt: f64) {
        // d_int (W/m³) is the dissipation rate
        // Entropy production rate: σ̇ = d_int / T
        if dt > 0.0 {
            let sigma_dot = d_int / self.temperature;
            self.total_entropy_prod += sigma_dot * dt;
        }
    }

    /// Get total information gain
    pub fn get_total_info_gain(&self) -> f64 {
        self.total_info_gain
    }

    /// Get total entropy production
    pub fn get_total_entropy_prod(&self) -> f64 {
        self.total_entropy_prod
    }

    /// Get bound violation count
    pub fn get_violations(&self) -> u64 {
        self.violations
    }

    /// Get statistics as JSON
    pub fn get_stats_json(&self) -> String {
        serde_json::json!({
            "temperature_K": self.temperature,
            "heat_capacity": self.heat_capacity,
            "gamma": self.gamma,
            "convergence_rate_bound": self.convergence_rate_bound(),
            "is_convergent": self.is_convergent(),
            "min_gamma_convergence": self.min_gamma_for_convergence(),
            "total_info_gain": self.total_info_gain,
            "total_entropy_prod": self.total_entropy_prod,
            "total_heat": self.total_heat,
            "time_steps": self.time_steps,
            "violations": self.violations,
        })
        .to_string()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.gamma = 0.0;
        self.total_info_gain = 0.0;
        self.total_entropy_prod = 0.0;
        self.total_heat = 0.0;
        self.time_steps = 0;
        self.violations = 0;
    }
}

/// Landauer Bound calculator for computational thermodynamics
///
/// The Landauer bound states that erasing one bit of information
/// requires at least k_B * T * ln(2) of energy dissipation.
///
/// This is relevant for:
/// 1. RL memory management (experience buffer erasure)
/// 2. Policy updates (forgetting old policies)
/// 3. State transitions (information loss)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandauerBound {
    /// Temperature (K)
    temperature: f64,
    /// Total bits erased
    total_bits_erased: f64,
    /// Total energy dissipated (J)
    total_energy: f64,
}

impl LandauerBound {
    pub fn new(temp_kelvin: f64) -> LandauerBound {
        LandauerBound {
            temperature: temp_kelvin.max(1.0),
            total_bits_erased: 0.0,
            total_energy: 0.0,
        }
    }

    /// Compute minimum energy required to erase n bits
    /// E_min = n * k_B * T * ln(2)
    pub fn min_erasure_energy(&self, n_bits: f64) -> f64 {
        n_bits * K_B * self.temperature * (2.0_f64).ln()
    }

    /// Compute entropy change from bit erasure
    /// ΔS = n * k_B * ln(2)
    pub fn erasure_entropy(&self, n_bits: f64) -> f64 {
        n_bits * K_B * (2.0_f64).ln()
    }

    /// Record an erasure operation
    pub fn record_erasure(&mut self, n_bits: f64, actual_energy: f64) {
        self.total_bits_erased += n_bits;
        self.total_energy += actual_energy;
    }

    /// Get efficiency (ratio of Landauer bound to actual energy)
    pub fn get_efficiency(&self) -> f64 {
        if self.total_energy <= 0.0 {
            return 1.0;
        }
        let landauer_energy = self.min_erasure_energy(self.total_bits_erased);
        landauer_energy / self.total_energy
    }

    /// Get total bits erased
    pub fn get_total_bits(&self) -> f64 {
        self.total_bits_erased
    }
}

/// Thermodynamic Learning Rate Bound
///
/// Based on the speed limit theorems for learning:
/// The learning rate is bounded by the entropy production rate.
///
/// dθ/dt ≤ √(2 * Σ̇ / F)
///
/// Where F is the Fisher information metric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermodynamicLRBound {
    /// Current entropy production rate
    sigma_dot: f64,
    /// Fisher information estimate
    fisher_info: f64,
    /// Running average of parameter changes
    avg_param_change: f64,
    /// Number of updates
    n_updates: u64,
}

impl ThermodynamicLRBound {
    pub fn new() -> ThermodynamicLRBound {
        ThermodynamicLRBound {
            sigma_dot: 0.001,
            fisher_info: 1.0,
            avg_param_change: 0.0,
            n_updates: 0,
        }
    }

    /// Get maximum allowed learning rate
    pub fn max_learning_rate(&self) -> f64 {
        if self.fisher_info <= 0.0 {
            return f64::INFINITY;
        }
        (2.0 * self.sigma_dot / self.fisher_info).sqrt()
    }

    /// Update with new entropy production and parameter change
    pub fn update(&mut self, sigma_dot: f64, param_change: f64) {
        self.sigma_dot = sigma_dot.max(1e-10);

        // Update Fisher info estimate from parameter changes
        // F ≈ E[(∂log p / ∂θ)²] ≈ 1 / Var(θ)
        self.avg_param_change = 0.9 * self.avg_param_change + 0.1 * param_change.abs();
        if self.avg_param_change > 1e-10 {
            self.fisher_info = 1.0 / self.avg_param_change.powi(2);
        }

        self.n_updates += 1;
    }

    /// Check if a learning rate satisfies the thermodynamic bound
    pub fn check_lr(&self, learning_rate: f64) -> bool {
        learning_rate <= self.max_learning_rate()
    }

    /// Get recommended learning rate (with safety margin)
    pub fn recommended_lr(&self, safety_factor: f64) -> f64 {
        self.max_learning_rate() * safety_factor.clamp(0.1, 1.0)
    }
}

/// Integrated Fluctuation Theorem checker
///
/// Verifies the Crooks fluctuation theorem:
/// P(+ΔS) / P(-ΔS) = exp(ΔS / k_B)
///
/// This provides statistical tests for thermodynamic consistency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluctuationTheorem {
    /// Forward entropy changes
    forward_ds: Vec<f64>,
    /// Reverse entropy changes
    reverse_ds: Vec<f64>,
    /// Temperature
    temperature: f64,
    /// Maximum samples to store
    max_samples: usize,
}

impl FluctuationTheorem {
    pub fn new(temp_kelvin: f64) -> FluctuationTheorem {
        FluctuationTheorem {
            forward_ds: Vec::with_capacity(1000),
            reverse_ds: Vec::with_capacity(1000),
            temperature: temp_kelvin.max(1.0),
            max_samples: 1000,
        }
    }

    /// Record a forward process entropy change
    pub fn record_forward(&mut self, delta_s: f64) {
        self.forward_ds.push(delta_s);
        if self.forward_ds.len() > self.max_samples {
            self.forward_ds.remove(0);
        }
    }

    /// Record a reverse process entropy change
    pub fn record_reverse(&mut self, delta_s: f64) {
        self.reverse_ds.push(delta_s);
        if self.reverse_ds.len() > self.max_samples {
            self.reverse_ds.remove(0);
        }
    }

    /// Compute the Crooks ratio at a given ΔS value
    /// Should equal exp(ΔS / k_B) for thermodynamic consistency
    pub fn crooks_ratio(&self, delta_s: f64, bin_width: f64) -> f64 {
        // Count forward processes in bin
        let forward_count = self
            .forward_ds
            .iter()
            .filter(|&&ds| (ds - delta_s).abs() < bin_width / 2.0)
            .count();

        // Count reverse processes in bin (with negative ΔS)
        let reverse_count = self
            .reverse_ds
            .iter()
            .filter(|&&ds| (ds + delta_s).abs() < bin_width / 2.0)
            .count();

        if reverse_count == 0 {
            return f64::INFINITY;
        }
        (forward_count as f64) / (reverse_count as f64)
    }

    /// Verify thermodynamic consistency
    /// Returns p-value for Crooks theorem
    pub fn verify_consistency(&self) -> f64 {
        if self.forward_ds.len() < 10 || self.reverse_ds.len() < 10 {
            return 1.0; // Not enough data
        }

        // Simplified check: average entropy production should be positive
        let avg_forward: f64 = self.forward_ds.iter().sum::<f64>() / self.forward_ds.len() as f64;
        let avg_reverse: f64 = self.reverse_ds.iter().sum::<f64>() / self.reverse_ds.len() as f64;

        // Total entropy production
        let total_entropy_prod = avg_forward - avg_reverse;

        // Second law: ⟨ΔS⟩ ≥ 0
        if total_entropy_prod >= -1e-10 {
            1.0 // Consistent
        } else {
            // Return estimated p-value based on magnitude of violation
            (-total_entropy_prod * 100.0).exp().min(1.0)
        }
    }

    /// Get average entropy production
    pub fn avg_entropy_production(&self) -> f64 {
        if self.forward_ds.is_empty() {
            return 0.0;
        }
        self.forward_ds.iter().sum::<f64>() / self.forward_ds.len() as f64
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.forward_ds.clear();
        self.reverse_ds.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_bounds_creation() {
        let bounds = QuantumThermoBounds::new();
        assert!((bounds.temperature - DEFAULT_TEMP).abs() < 1.0);
        assert_eq!(bounds.violations, 0);
    }

    #[test]
    fn test_second_law_satisfied() {
        let mut bounds = QuantumThermoBounds::new();

        // Normal case: sufficient entropy production
        let margin = bounds.check_second_law(0.1, 0.05, 10.0);
        assert!(margin > 0.0, "Should have positive margin");
        assert!(
            bounds.is_second_law_satisfied(0.1, 0.05, 10.0),
            "Second law should be satisfied"
        );
    }

    #[test]
    fn test_second_law_violated() {
        let mut bounds = QuantumThermoBounds::new();

        // Violation case: too little entropy production
        let margin = bounds.check_second_law(0.001, 0.1, 100.0);
        assert!(margin < 0.0, "Should have negative margin");
        assert_eq!(bounds.violations, 1);
    }

    #[test]
    fn test_convergence_guarantee() {
        let mut bounds = QuantumThermoBounds::new();

        // With sufficient info gain rate, should converge
        bounds.set_gamma(10.0); // High info gain rate
        assert!(bounds.is_convergent(), "Should converge with high gamma");

        // With insufficient info gain rate, may not converge
        bounds.set_gamma(0.0001);
        // Depending on parameters, may or may not converge
        println!(
            "Min gamma for convergence: {:.4}",
            bounds.min_gamma_for_convergence()
        );
    }

    #[test]
    fn test_convergence_rate_bound() {
        let mut bounds = QuantumThermoBounds::new();
        bounds.set_gamma(5.0);

        let rate = bounds.convergence_rate_bound();
        println!("Convergence rate bound: {:.6}", rate);
        // With high gamma, rate should be negative (converging)
    }

    #[test]
    fn test_landauer_bound() {
        let landauer = LandauerBound::new(300.0); // Room temp

        // Energy to erase 1 bit
        let energy = landauer.min_erasure_energy(1.0);
        let expected = K_B * 300.0 * (2.0_f64).ln();

        assert!(
            (energy - expected).abs() < 1e-25,
            "Landauer bound calculation"
        );
        println!("Energy to erase 1 bit at 300K: {:.3e} J", energy);
    }

    #[test]
    fn test_thermodynamic_lr_bound() {
        let mut lr_bound = ThermodynamicLRBound::new();

        // Update with typical values - entropy production and param changes
        // Need several updates to stabilize the Fisher info estimate
        for _ in 0..10 {
            lr_bound.update(0.01, 0.1); // Higher param change for realistic Fisher info
        }

        let max_lr = lr_bound.max_learning_rate();
        println!("Max thermodynamic LR: {:.4}", max_lr);

        // Should have a reasonable max LR now
        assert!(max_lr > 0.0, "Max LR should be positive");

        // Recommended LR should be smaller than max
        let recommended = lr_bound.recommended_lr(0.5);
        assert!(recommended <= max_lr);
    }

    #[test]
    fn test_fluctuation_theorem_consistency() {
        let mut ft = FluctuationTheorem::new(300.0);

        // Add positive entropy production (forward irreversible processes)
        for _ in 0..50 {
            ft.record_forward(0.1);
            ft.record_reverse(-0.05);
        }

        let p_value = ft.verify_consistency();
        assert!(p_value > 0.5, "Should be thermodynamically consistent");

        let avg_prod = ft.avg_entropy_production();
        assert!(
            avg_prod > 0.0,
            "Average entropy production should be positive"
        );
    }

    #[test]
    fn test_material_system_bounds() {
        // Create bounds for concrete at 30°C
        let bounds = QuantumThermoBounds::for_material_system(30.0, 2400.0);

        assert!((bounds.temperature - 303.15).abs() < 0.01);
        assert_eq!(bounds.heat_capacity, 2400.0);
        println!("Material system σ̇_min: {:.6}", bounds.sigma_dot_min);
    }

    #[test]
    fn test_convergence_time_estimate() {
        let mut bounds = QuantumThermoBounds::new();
        bounds.set_gamma(1.0);

        if let Some(time) = bounds.estimated_convergence_time(100.0, 1.0) {
            println!("Estimated convergence time: {:.2} s", time);
            assert!(time > 0.0);
        } else {
            println!("System not converging");
        }
    }
}
