// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! ECS Physical Components
//! These are strictly typed components that attach to physical Entities
//! to represent their state across various mathematical categories.

use crate::science::domain::MaterialComponent;
use serde::{Deserialize, Serialize};

/// The base composition component of an entity (e.g., a bucket of concrete).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixtureComposition {
    pub materials: Vec<MaterialComponent>,
}

/// A component representing the thermodynamic boundaries of the entity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermodynamicState {
    pub temperature_c: f32,
    pub entropy_production_rate: f64,
    pub chemical_potential: f64,
}

/// A component capturing the rheological (flow) state of a material.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RheologyProfile {
    pub yield_stress_pa: f32,
    pub plastic_viscosity_pas: f32,
    pub slump_flow_mm: f32,
}

/// A component tracking the chemical hydration or kinetics over time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HydrationKinetics {
    pub degree_of_hydration: f32, // 0.0 - 1.0
    pub age_hours: f32,
    pub peak_temperature_c: f32,
}

/// A component enforcing the axiological (safety/toxicity) bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToxicityProfile {
    pub co2_footprint_kg_m3: f32,
    pub leachable_heavy_metals_ppm: f32,
    pub is_flammable: bool,
}

/// A component that tracks the epistemic state and proxy measurements
/// for a given entity during the RL POMDP loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicStateComponent {
    pub measured_proxies: Vec<String>,
    pub proxy_values: std::collections::HashMap<String, f64>,
    pub convergence_score: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Exp 2: 3D Concrete Printer Simulation (Axiological Veto)
// ─────────────────────────────────────────────────────────────────────────────

/// Motor state for the extrusion drive mechanism.
/// Tracks real torque vs the ISO-certified max safe operating torque.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotorState {
    /// Current applied torque in Newton-Meters
    pub torque_nm: f64,
    /// Maximum torque the hardware can safely sustain (ISO 10218 limit)
    pub max_safe_torque_nm: f64,
    /// Counter of how many sequential steps torque has been at max
    pub sustained_overload_steps: u32,
}

impl MotorState {
    pub fn new(max_safe: f64) -> Self {
        Self {
            torque_nm: 0.0,
            max_safe_torque_nm: max_safe,
            sustained_overload_steps: 0,
        }
    }
    pub fn is_overloaded(&self) -> bool {
        self.torque_nm > self.max_safe_torque_nm
    }
}

/// Flow state for extrudate material moving through the nozzle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowState {
    /// Target flow rate in mL/s
    pub target_flow_rate_ml_s: f64,
    /// Actual achieved flow rate based on motor + clogging
    pub actual_flow_rate_ml_s: f64,
    /// 0.0 = no clog, 1.0 = completely blocked
    pub clogging_factor: f64,
}

impl FlowState {
    pub fn new(target: f64) -> Self {
        Self {
            target_flow_rate_ml_s: target,
            actual_flow_rate_ml_s: target,
            clogging_factor: 0.0,
        }
    }
}

/// Top-level Robotic Extruder entity aggregate — groups motor + flow for system queries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoboticExtruder {
    pub name: String,
    pub is_constrained: bool, // true = DUMSTO constitutional veto active
}
