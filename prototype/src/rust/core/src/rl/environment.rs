// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! Epistemic POMDP Environment
//!
//! Manages the partially observable state and handles the transition between
//! epistemic sensing actions (which reveal proxies) and physical actuation.

use super::state::{ActionType, RLState};
use crate::constitution::{
    execute_constitutional_functor, AxiologicalFloor, ConstitutionalViolation, PhysicalSubstrate,
    ThermodynamicallyAdmissible,
};
use crate::physics_kernel::{IndustrialResult, PhysicsConfig, PhysicsKernel};
use crate::tensors::MixTensor;

/// The result of an agent taking an action in the POMDP
pub struct StepResult {
    pub next_state: RLState,
    pub reward: f64,
    pub done: bool,
    pub info: Option<String>,
}

pub trait EpistemicEnvironment {
    /// Resets the environment to an initial state
    fn reset(&mut self) -> RLState;

    /// Steps the environment forward by applying ActionType
    /// Returns a Result enforcing the Axiological Veto (Layer 4.5)
    fn step(&mut self, action: ActionType) -> Result<StepResult, ConstitutionalViolation>;
}

pub struct ConcretePOMDP {
    pub current_state: RLState,
    pub true_hidden_mix: MixTensor,
    pub step_count: u32,
    pub max_steps: u32,
    pub proxy_cost: f64, // The "Epistemic Toll"
}

impl ConcretePOMDP {
    pub fn new(base_mix: MixTensor, max_steps: u32, proxy_cost: f64) -> Self {
        Self {
            current_state: RLState::new(),
            true_hidden_mix: base_mix,
            step_count: 0,
            max_steps,
            proxy_cost,
        }
    }

    /// Privileged function to run the physics engine and reveal a specific proxy
    fn measure_proxy(&self, index: usize) -> f64 {
        let config = PhysicsConfig::default();
        let result = PhysicsKernel::compute(&self.true_hidden_mix, None, &config);

        // Normalize physical properties using their theoretical/practical maximum bounds
        // to maintain state values inside [0, 1] for the PPO agent, replacing arbitrary scales.
        let max_slump_flow_bound = 1200.0; // Theoretical full collapse for self-compacting
        let max_viscosity_bound = 300.0; // High-viscosity 3DCP limit
        let max_yield_stress_bound = 5000.0; // Typical extruder maximum

        match index {
            0 => (result.fresh.slump_flow as f64 / max_slump_flow_bound).min(1.0),
            1 => (result.fresh.plastic_viscosity as f64 / max_viscosity_bound).min(1.0),
            2 => (result.fresh.yield_stress as f64 / max_yield_stress_bound).min(1.0),
            // ... 24 other proxies ...
            _ => 0.0,
        }
    }
}

/// Wrapper to hold the proposed transition state for validation
pub struct ProposedTransition {
    pub proposed_mix: MixTensor,
    pub result: IndustrialResult,
}

impl ThermodynamicallyAdmissible for ProposedTransition {
    fn check_clausius_duhem(&self) -> Result<(), ConstitutionalViolation> {
        // Layer 0: Clausius-Duhem Strict Bound
        // Entropy production MUST be positive (∆S ≥ 0).
        if self.result.thermal.heat_of_hydration < 0.0 {
            return Err(ConstitutionalViolation::ThermodynamicAdmissibility {
                detail: "Negative heat of hydration predicted; impossible thermodynamics.".into(),
            });
        }
        Ok(())
    }
}

impl PhysicalSubstrate for ProposedTransition {
    fn check_substrate_envelope(&self) -> Result<(), ConstitutionalViolation> {
        // Layer 2: Physical Hardware Limits
        // Replaced arbitrary 25kPa limit with actual Hagen-Poiseuille torque threshold
        let r_nozzle = 0.0015_f32; // m
        let l_nozzle = 0.05_f32; // m
        let q_target = 0.0001_f32; // m^3/s high flow rate

        let r_4 = r_nozzle * r_nozzle * r_nozzle * r_nozzle;
        let delta_p = (8.0_f32 * self.result.fresh.plastic_viscosity * l_nozzle * q_target)
            / (std::f32::consts::PI * r_4);

        let pitch_area = 0.0005_f32; // m^2 effective sweep area
        let gear_ratio = 10.0_f32;
        let t_motor = (delta_p * pitch_area) / gear_ratio;

        let safe_torque_nm = 22.5_f32; // Hardware Nema limit

        if t_motor > safe_torque_nm {
            return Err(ConstitutionalViolation::PhysicalSubstrate {
                detail: format!(
                    "Hagen-Poiseuille torque {:.1} N·m exceeds hardware limit ({:.1} N·m).",
                    t_motor, safe_torque_nm
                ),
            });
        }
        Ok(())
    }
}

impl AxiologicalFloor for ProposedTransition {
    fn check_axiological_veto(&self) -> Result<(), ConstitutionalViolation> {
        // Layer 4.5: The Veto
        // Layer 4.5: The Veto
        // Replaced arbitrary 10.0 MPa with ACI 318 / EN 206 structural code absolute minimum
        let structural_code_floor_mpa = 17.5; // EN 206 C16/20 cylinder strength
        if self.result.hardened.f28_compressive < structural_code_floor_mpa {
            return Err(ConstitutionalViolation::AxiologicalFloor {
                detail: format!(
                    "Code compliance failure: {} MPa < {} MPa EN 206 floor.",
                    self.result.hardened.f28_compressive, structural_code_floor_mpa
                ),
            });
        }
        Ok(())
    }
}

impl EpistemicEnvironment for ConcretePOMDP {
    fn reset(&mut self) -> RLState {
        self.current_state = RLState::new();
        self.step_count = 0;
        self.current_state.clone()
    }

    fn step(&mut self, action: ActionType) -> Result<StepResult, ConstitutionalViolation> {
        self.step_count += 1;
        let done = self.step_count >= self.max_steps;

        match action {
            ActionType::Sense(proxy_idx) => {
                // If proxy already measured, penalize heavily for waste
                if self.current_state.has_proxy(proxy_idx) {
                    return Ok(StepResult {
                        next_state: self.current_state.clone(),
                        reward: -10.0,
                        done,
                        info: Some("Proxy already measured".to_string()),
                    });
                }

                // Epistemic Action: Pay the thermodynamic toll to reveal state
                let value = self.measure_proxy(proxy_idx);
                self.current_state.set_proxy(proxy_idx, value);

                Ok(StepResult {
                    next_state: self.current_state.clone(),
                    reward: -self.proxy_cost, // Apply the Epistemic Toll
                    done,
                    info: Some(format!("Measured proxy {}", proxy_idx)),
                })
            }
            ActionType::Actuate(phys_action) => {
                // 1. Propose the action on a clone of the hidden state
                let mut proposed_mix = self.true_hidden_mix.clone();
                proposed_mix.apply_action(
                    phys_action.delta_wc as f32,
                    phys_action.delta_scms as f32,
                    phys_action.delta_sp as f32,
                );

                // 2. Compute physics for the proposed state
                let config = PhysicsConfig::default();
                let result = PhysicsKernel::compute(&proposed_mix, None, &config);

                // 3. Build the transition proof object
                let transition = ProposedTransition {
                    proposed_mix,
                    result,
                };

                // 4. Force the transition through the 9-Layer Constitutional Functor
                // If it fails ANY layer, it immediately returns a ConstitutionalViolation (Err)
                // short-circuiting the RL agent loop.
                let admissible_transition = execute_constitutional_functor(transition)?;

                // 5. If admissible, apply to true state
                self.true_hidden_mix = admissible_transition.payload.proposed_mix;
                let final_result = admissible_transition.payload.result;

                Ok(StepResult {
                    next_state: self.current_state.clone(),
                    reward: final_result.hardened.f28_compressive as f64, // Placeholder task reward
                    done,
                    info: Some("Actuation Algebraically Verified and Successful".to_string()),
                })
            }
        }
    }
}
