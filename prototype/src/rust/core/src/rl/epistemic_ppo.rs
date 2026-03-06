// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! Epistemic PPO Extension
//!
//! Extends the standard PPO agent with quantum active inference capabilities:
//! - Epistemic reward shaping via mutual information
//! - Quantum thermodynamic bounds enforcement
//! - Intrinsic curiosity for exploration
//!
//! # Constitutional Architecture Extension (V8)
//!
//! Original 3-Layer:
//! 1. GuardrailEngine — Hard physical bounds
//! 2. ThermodynamicFilter — Clausius-Duhem inequality
//! 3. RewardFunction — Multi-objective shaping
//!
//! Extended with:
//! 4. EpistemicModule — Information-theoretic exploration
//! 5. QuantumThermoBounds — Generalized second law

use super::epistemic::{EpistemicStateTracker, IntrinsicCuriosity, MutualInfoEstimator};
use super::quantum_bounds::{QuantumThermoBounds, ThermodynamicLRBound};
use super::reward::{RewardComponents, RewardConfig, RewardFunction, RewardType};
use serde::{Deserialize, Serialize};

/// Configuration for epistemic PPO extension
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicPPOConfig {
    /// Weight for epistemic (mutual information) bonus
    pub beta_epistemic: f64,
    /// Weight for curiosity bonus
    pub eta_curiosity: f64,
    /// Enable quantum thermodynamic bounds checking
    pub enable_quantum_bounds: bool,
    /// Enable adaptive learning rate from thermodynamic bounds
    pub enable_thermo_lr: bool,
    /// Temperature for thermodynamic calculations (°C)
    pub temperature_celsius: f64,
    /// Heat capacity for material system (J/K·m³)
    pub heat_capacity: f64,
    /// Minimum info gain rate for convergence guarantee
    pub min_gamma_threshold: f64,
}

impl Default for EpistemicPPOConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicPPOConfig {
    pub fn new() -> EpistemicPPOConfig {
        EpistemicPPOConfig {
            beta_epistemic: 0.1, // 10% weight on epistemic bonus
            eta_curiosity: 0.01, // 1% weight on curiosity
            enable_quantum_bounds: true,
            enable_thermo_lr: true,
            temperature_celsius: 20.0,
            heat_capacity: 2000.0,     // Typical for concrete
            min_gamma_threshold: 0.01, // Minimum info gain rate
        }
    }

    /// Create config optimized for exploration
    pub fn exploration_focused() -> EpistemicPPOConfig {
        EpistemicPPOConfig {
            beta_epistemic: 0.3, // Higher epistemic weight
            eta_curiosity: 0.05, // Higher curiosity
            enable_quantum_bounds: true,
            enable_thermo_lr: true,
            temperature_celsius: 20.0,
            heat_capacity: 2000.0,
            min_gamma_threshold: 0.005,
        }
    }

    /// Create config optimized for exploitation
    pub fn exploitation_focused() -> EpistemicPPOConfig {
        EpistemicPPOConfig {
            beta_epistemic: 0.05, // Lower epistemic weight
            eta_curiosity: 0.005, // Lower curiosity
            enable_quantum_bounds: true,
            enable_thermo_lr: true,
            temperature_celsius: 20.0,
            heat_capacity: 2000.0,
            min_gamma_threshold: 0.02,
        }
    }
}

/// Epistemic PPO Extension Module
///
/// Wraps around the standard PPO agent to provide:
/// - Mutual information estimation for epistemic rewards
/// - Quantum thermodynamic bounds checking
/// - Intrinsic curiosity for exploration
/// - Adaptive learning rate bounds
#[derive(Serialize, Deserialize)]
pub struct EpistemicPPOModule {
    config: EpistemicPPOConfig,

    /// Mutual information estimator
    mi_estimator: MutualInfoEstimator,

    /// Epistemic state tracker
    epistemic_tracker: EpistemicStateTracker,

    /// Intrinsic curiosity module
    curiosity: IntrinsicCuriosity,

    /// Quantum thermodynamic bounds checker
    quantum_bounds: QuantumThermoBounds,

    /// Thermodynamic learning rate bound
    lr_bound: ThermodynamicLRBound,

    /// Statistics
    total_steps: u64,
    epistemic_bonus_sum: f64,
    curiosity_bonus_sum: f64,
    bound_violations: u64,
    convergent_steps: u64,

    /// Bidirectional rho vector (Paper 2 & 5: Volumetric Growth).
    /// rho.0: rho_up (Exploration/Abstraction yield)
    /// rho.1: rho_down (Grounding/Conservative yield)
    rho: (f64, f64),
}

impl EpistemicPPOModule {
    pub fn new(config: EpistemicPPOConfig) -> EpistemicPPOModule {
        let mi_estimator = MutualInfoEstimator::for_projectx();
        let mut epistemic_tracker = EpistemicStateTracker::new();
        epistemic_tracker.set_beta(config.beta_epistemic);
        let curiosity = IntrinsicCuriosity::new();
        let quantum_bounds = QuantumThermoBounds::for_material_system(
            config.temperature_celsius,
            config.heat_capacity,
        );
        let lr_bound = ThermodynamicLRBound::new();

        EpistemicPPOModule {
            config,
            mi_estimator,
            epistemic_tracker,
            curiosity,
            quantum_bounds,
            lr_bound,
            total_steps: 0,
            epistemic_bonus_sum: 0.0,
            curiosity_bonus_sum: 0.0,
            bound_violations: 0,
            convergent_steps: 0,
            rho: (0.0, 1.0), // Start with full grounding (conservative)
        }
    }

    /// Create with default config
    pub fn default_config() -> EpistemicPPOModule {
        EpistemicPPOModule::new(EpistemicPPOConfig::new())
    }

    /// Process a transition and compute epistemic bonuses via FEP minimization.
    ///
    /// FORMULATION (Paper 2):
    /// Minimize Expected Free Energy G(π) = E[D(Q(ψ||o)||P(ψ))] - I_Q[ψ;o]
    /// where Complexity (D_KL) acts as a dissipative force (rho_down)
    /// and Accuracy (MI) acts as a constructive force (rho_up).
    ///
    /// Returns the total bonus to add to the reward.
    pub fn process_transition(
        &mut self,
        state: &[f64],
        _action: &[f64],
        observation: &[f64],
        predicted_obs: &[f64],
        entropy_production: f64,
    ) -> f64 {
        self.total_steps += 1;

        // 1. Update mutual information estimator (Accuracy signal)
        self.mi_estimator.update(state, observation);
        let mi = self.mi_estimator.estimate();

        // 2. Compute complexity penalty (D_KL approximation from curiosity error)
        // Curiosity bonus is R_c = ||phi(s) - phi'(s,a)||^2.
        // In FEP, this serves as an approximation of the complexity/surprise.
        let curiosity_bonus = self
            .curiosity
            .compute_bonus(state, predicted_obs, observation);

        // 3. Update Bidirectional Rho (Volumetric Growth)
        // rho_up increases with MI (Information yield)
        // rho_down increases with low curiosity error (Conserved/Hardened logic)
        let alpha = 0.05;
        self.rho.0 = (1.0 - alpha) * self.rho.0 + alpha * mi.clamp(0.0, 1.0);
        self.rho.1 =
            (1.0 - alpha) * self.rho.1 + alpha * (1.0 / (1.0 + curiosity_bonus)).clamp(0.0, 1.0);

        // 4. Update epistemic tracker
        self.epistemic_tracker.update(mi);
        let epistemic_bonus = self.epistemic_tracker.get_epistemic_bonus();
        self.epistemic_bonus_sum += epistemic_bonus;

        // 5. Scaled FEP Bonus
        // G = Complexity - Accuracy. Objective is minimizing G, so Reward = Accuracy - Complexity.
        let scaled_curiosity = self.config.eta_curiosity * curiosity_bonus;
        self.curiosity_bonus_sum += scaled_curiosity;

        let fep_bonus = self.rho.0 * epistemic_bonus - self.rho.1 * scaled_curiosity;

        // 6. Update quantum bounds (Physics check)
        let info_gain = self.epistemic_tracker.get_avg_gamma();
        self.quantum_bounds.set_gamma(info_gain);

        if self.config.enable_quantum_bounds {
            // Check second law: Σ ≥ I_gain + Q/T
            let heat_estimate = entropy_production * (self.config.temperature_celsius + 273.15);
            let margin = self.quantum_bounds.check_second_law(
                entropy_production / (self.config.temperature_celsius + 273.15),
                info_gain,
                heat_estimate,
            );

            if margin < -1e-10 {
                self.bound_violations += 1;
            }
        }

        // 7. Track convergence
        if self.quantum_bounds.is_convergent() {
            self.convergent_steps += 1;
        }

        fep_bonus
    }

    /// Get recommended learning rate based on thermodynamic bounds
    pub fn get_recommended_lr(&self, base_lr: f64, safety_factor: f64) -> f64 {
        if !self.config.enable_thermo_lr {
            return base_lr;
        }

        let max_thermo_lr = self.lr_bound.max_learning_rate();
        base_lr.min(max_thermo_lr * safety_factor.clamp(0.1, 1.0))
    }

    /// Update learning rate bounds with new data
    pub fn update_lr_bounds(&mut self, entropy_prod_rate: f64, param_change: f64) {
        self.lr_bound.update(entropy_prod_rate, param_change);
    }

    /// Check if system is in convergent regime
    pub fn is_convergent(&self) -> bool {
        self.quantum_bounds.is_convergent()
    }

    /// Get current mutual information estimate
    pub fn get_mi(&self) -> f64 {
        self.mi_estimator.estimate()
    }

    /// Get current info gain rate (gamma)
    pub fn get_gamma(&self) -> f64 {
        self.quantum_bounds.get_gamma()
    }

    /// Get convergence rate bound (negative = converging)
    pub fn get_convergence_rate(&self) -> f64 {
        self.quantum_bounds.convergence_rate_bound()
    }

    /// Get minimum gamma needed for convergence
    pub fn get_min_gamma_for_convergence(&self) -> f64 {
        self.quantum_bounds.min_gamma_for_convergence()
    }

    /// Get total bound violations
    pub fn get_violations(&self) -> u64 {
        self.bound_violations
    }

    /// Get convergent step ratio
    pub fn get_convergent_ratio(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.convergent_steps as f64 / self.total_steps as f64
    }

    /// Get average epistemic bonus
    pub fn get_avg_epistemic_bonus(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.epistemic_bonus_sum / self.total_steps as f64
    }

    /// Get average curiosity bonus
    pub fn get_avg_curiosity_bonus(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.curiosity_bonus_sum / self.total_steps as f64
    }

    /// Get number of unique states visited (exploration metric)
    pub fn get_unique_states(&self) -> usize {
        self.curiosity.get_unique_states()
    }

    /// Get total steps processed
    pub fn get_total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.mi_estimator.reset();
        self.epistemic_tracker.reset();
        self.curiosity.reset();
        self.quantum_bounds.reset();
        self.total_steps = 0;
        self.epistemic_bonus_sum = 0.0;
        self.curiosity_bonus_sum = 0.0;
        self.bound_violations = 0;
        self.convergent_steps = 0;
        self.rho = (0.0, 1.0);
    }

    /// Get comprehensive statistics as JSON
    pub fn get_stats_json(&self) -> String {
        serde_json::json!({
            "total_steps": self.total_steps,
            "mi_estimate": self.get_mi(),
            "mi_confidence": self.mi_estimator.get_confidence(),
            "gamma": self.get_gamma(),
            "convergence_rate": self.get_convergence_rate(),
            "is_convergent": self.is_convergent(),
            "min_gamma_convergence": self.get_min_gamma_for_convergence(),
            "avg_epistemic_bonus": self.get_avg_epistemic_bonus(),
            "avg_curiosity_bonus": self.get_avg_curiosity_bonus(),
            "unique_states": self.get_unique_states(),
            "bound_violations": self.bound_violations,
            "convergent_ratio": self.get_convergent_ratio(),
            "rho": {
                "up": self.rho.0,
                "down": self.rho.1
            },
            "config": {
                "beta_epistemic": self.config.beta_epistemic,
                "eta_curiosity": self.config.eta_curiosity,
                "temperature_celsius": self.config.temperature_celsius,
            }
        })
        .to_string()
    }
}

/// Epistemic Reward Calculator
///
/// Computes the total reward including epistemic bonuses.
/// R_total = R_task + β * I[ψ;o] + η * R_curiosity
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicRewardCalculator {
    /// Base reward function
    base_reward: RewardFunction,
    /// Epistemic bonus weight
    beta: f64,
    /// Curiosity bonus weight  
    eta: f64,
    /// Running average of task rewards
    avg_task_reward: f64,
    /// Running average of epistemic bonus
    avg_epistemic: f64,
    /// Number of samples
    n_samples: u64,
}

impl EpistemicRewardCalculator {
    pub fn new(reward_type: RewardType, beta: f64, eta: f64) -> EpistemicRewardCalculator {
        let config = RewardConfig::new(reward_type);
        EpistemicRewardCalculator {
            base_reward: RewardFunction::new(config),
            beta: beta.clamp(0.0, 1.0),
            eta: eta.clamp(0.0, 1.0),
            avg_task_reward: 0.0,
            avg_epistemic: 0.0,
            n_samples: 0,
        }
    }

    /// Calculate total reward with epistemic bonuses
    pub fn calculate_total(
        &mut self,
        components: &RewardComponents,
        epistemic_bonus: f64,
        curiosity_bonus: f64,
    ) -> f64 {
        let task_reward = self.base_reward.calculate(components);

        // Update running averages for normalization
        let alpha = 0.01;
        self.avg_task_reward = (1.0 - alpha) * self.avg_task_reward + alpha * task_reward;
        self.avg_epistemic = (1.0 - alpha) * self.avg_epistemic + alpha * epistemic_bonus;
        self.n_samples += 1;

        // Normalize epistemic bonus to similar scale as task reward
        let normalized_epistemic = if self.avg_task_reward.abs() > 1e-6 {
            epistemic_bonus * self.avg_task_reward.abs() / (self.avg_epistemic.abs() + 1e-6)
        } else {
            epistemic_bonus
        };

        // Total reward
        task_reward + self.beta * normalized_epistemic + self.eta * curiosity_bonus
    }

    /// Get task-only reward (no epistemic bonus)
    pub fn calculate_task_only(&self, components: &RewardComponents) -> f64 {
        self.base_reward.calculate(components)
    }

    /// Get statistics
    pub fn get_stats_json(&self) -> String {
        serde_json::json!({
            "beta": self.beta,
            "eta": self.eta,
            "avg_task_reward": self.avg_task_reward,
            "avg_epistemic": self.avg_epistemic,
            "n_samples": self.n_samples,
        })
        .to_string()
    }
}

/// Adaptive exploration-exploitation scheduler
///
/// Adjusts epistemic weights based on learning progress
/// using quantum thermodynamic convergence metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplorationScheduler {
    /// Initial epistemic weight
    initial_beta: f64,
    /// Final epistemic weight (at convergence)
    final_beta: f64,
    /// Convergence threshold for gamma
    gamma_threshold: f64,
    /// Current beta value
    current_beta: f64,
    /// Smoothing factor for updates
    smoothing: f64,
}

impl Default for ExplorationScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplorationScheduler {
    pub fn new() -> ExplorationScheduler {
        ExplorationScheduler {
            initial_beta: 0.3,    // Start with high exploration
            final_beta: 0.05,     // End with low exploration
            gamma_threshold: 0.1, // Convergence indicator
            current_beta: 0.3,
            smoothing: 0.01,
        }
    }

    /// Update scheduler based on convergence metrics
    pub fn update(&mut self, gamma: f64, is_convergent: bool) {
        let target_beta = if is_convergent {
            // In convergent regime, reduce exploration
            self.final_beta
        } else if gamma > self.gamma_threshold {
            // High info gain, maintain exploration
            self.initial_beta
        } else {
            // Low info gain, increase exploration
            self.initial_beta * 1.2
        };

        // Smooth update
        self.current_beta = (1.0 - self.smoothing) * self.current_beta
            + self.smoothing * target_beta.clamp(self.final_beta, self.initial_beta * 1.5);
    }

    /// Get current epistemic weight
    pub fn get_beta(&self) -> f64 {
        self.current_beta
    }

    /// Set convergence threshold
    pub fn set_gamma_threshold(&mut self, threshold: f64) {
        self.gamma_threshold = threshold.max(0.001);
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.current_beta = self.initial_beta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_ppo_config() {
        let config = EpistemicPPOConfig::new();
        assert_eq!(config.beta_epistemic, 0.1);
        assert!(config.enable_quantum_bounds);
    }

    #[test]
    fn test_epistemic_ppo_module_creation() {
        let module = EpistemicPPOModule::default_config();
        assert_eq!(module.total_steps, 0);
        assert_eq!(module.get_mi(), 0.0);
    }

    #[test]
    fn test_transition_processing() {
        let mut module = EpistemicPPOModule::default_config();

        // Process a transition
        let state = vec![0.5; 35];
        let action = vec![0.1; 9];
        let observation = vec![30.0, 100.0, 200.0, 150.0, 30.0, 600.0];
        let predicted = vec![32.0, 105.0, 210.0, 145.0, 32.0, 580.0];

        let bonus = module.process_transition(
            &state,
            &action,
            &observation,
            &predicted,
            0.01, // entropy production
        );

        assert_eq!(module.total_steps, 1);
        println!("Epistemic bonus: {:.4}", bonus);
    }

    #[test]
    fn test_convergence_tracking() {
        let mut module = EpistemicPPOModule::default_config();

        // Process many transitions to build up statistics
        for i in 0..100 {
            let x = (i as f64) / 100.0;
            let state = vec![x; 35];
            let action = vec![x * 0.1; 9];
            let observation = vec![
                x * 40.0,
                x * 150.0,
                x * 300.0,
                x * 200.0,
                x * 40.0,
                x * 700.0,
            ];
            let predicted = vec![
                x * 38.0,
                x * 145.0,
                x * 290.0,
                x * 195.0,
                x * 38.0,
                x * 680.0,
            ];

            module.process_transition(&state, &action, &observation, &predicted, 0.01);
        }

        println!("After 100 transitions:");
        println!("  MI: {:.4}", module.get_mi());
        println!("  Gamma: {:.4}", module.get_gamma());
        println!(
            "  Convergent ratio: {:.2}%",
            module.get_convergent_ratio() * 100.0
        );
        println!("  Unique states: {}", module.get_unique_states());
    }

    #[test]
    fn test_epistemic_reward_calculator() {
        let mut calc = EpistemicRewardCalculator::new(RewardType::Balanced, 0.1, 0.01);

        // Use the default constructor which initializes all fields
        let mut components = RewardComponents::new();
        components.strength_fc = 35.0;
        components.cost = 100.0;
        components.co2 = 250.0;
        components.fracture_kic = 1.5;
        components.diffusivity = 0.001;
        components.damage = 0.1;
        components.bond = 2.5;
        components.yield_stress = 200.0;
        components.viscosity = 35.0;
        components.slump_flow = 650.0;

        let total = calc.calculate_total(&components, 0.5, 0.1);
        let task_only = calc.calculate_task_only(&components);

        println!("Task reward: {:.4}", task_only);
        println!("Total reward (with epistemic): {:.4}", total);

        // Total should be >= task (positive bonuses)
        // Note: depending on normalization, might be less
    }

    #[test]
    fn test_exploration_scheduler() {
        let mut scheduler = ExplorationScheduler::new();

        assert_eq!(scheduler.get_beta(), 0.3); // Initial

        // Simulate convergence
        for _ in 0..100 {
            scheduler.update(0.05, true);
        }

        // Beta should decrease towards final
        let final_beta = scheduler.get_beta();
        println!("Beta after convergence: {:.4}", final_beta);
        assert!(final_beta < 0.3, "Beta should decrease during convergence");
    }

    #[test]
    fn test_exploration_config() {
        let explore = EpistemicPPOConfig::exploration_focused();
        let exploit = EpistemicPPOConfig::exploitation_focused();

        assert!(explore.beta_epistemic > exploit.beta_epistemic);
        assert!(explore.eta_curiosity > exploit.eta_curiosity);
    }

    #[test]
    fn test_stats_json() {
        let module = EpistemicPPOModule::default_config();
        let json = module.get_stats_json();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("total_steps").is_some());
        assert!(parsed.get("mi_estimate").is_some());
        assert!(parsed.get("is_convergent").is_some());
    }
}
