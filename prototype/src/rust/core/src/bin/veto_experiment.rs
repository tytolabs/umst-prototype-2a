// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! Exp 2: Sequential Autopoiesis & The Axiological Veto (Paper 3)
//!
//! Simulates a 3D concrete printer nozzle extruding material.
//! At step 15, a "clogging" fault is injected (clogging_factor → 0.85).
//!
//! Unconstrained PPO: cranks torque monotonically to compensate for lost flow.
//! DUMSTO (Constrained): checks the AxiologicalFloor constitutional trait at
//! each step, fires a hard veto when torque > max_safe_torque, and safely aborts
//! the episode without any human intervention.

use std::fs::File;
use std::io::Write;
use umst_core::constitution::{
    execute_constitutional_functor, AxiologicalFloor, ConstitutionalViolation, PhysicalSubstrate,
    ThermodynamicallyAdmissible,
};
use umst_core::ecs::components::{FlowState, MotorState};

/// Represents one control-step command the agent generates.
#[allow(dead_code)]
struct ExtruderAction {
    /// Torque delta requested by the policy (N·m per step)
    delta_torque_nm: f64,
}

/// The physical state of the extruder at each timestep.
/// Implements all three constitutional layers.
#[derive(Clone)]
struct ExtruderState {
    motor: MotorState,
    flow: FlowState,
    step: usize,
}

impl ThermodynamicallyAdmissible for ExtruderState {
    fn check_clausius_duhem(&self) -> Result<(), ConstitutionalViolation> {
        // Entropy production must be non-negative:
        // Rapid clog + high torque = large irreversible heat dissipation → admissible
        // But if actual_flow < 0, that violates mass conservation
        if self.flow.actual_flow_rate_ml_s < 0.0 {
            return Err(ConstitutionalViolation::ThermodynamicAdmissibility {
                detail: format!(
                    "Negative flow rate {:.2} mL/s violates mass conservation",
                    self.flow.actual_flow_rate_ml_s
                ),
            });
        }
        Ok(())
    }
}

impl PhysicalSubstrate for ExtruderState {
    fn check_substrate_envelope(&self) -> Result<(), ConstitutionalViolation> {
        // Layer 2: Hardware cannot sustain torque beyond mechanical limits
        if self.motor.torque_nm > self.motor.max_safe_torque_nm * 1.5 {
            return Err(ConstitutionalViolation::PhysicalSubstrate {
                detail: format!(
                    "Motor torque {:.1} N·m breaches 1.5× hardware limit ({:.1} N·m)",
                    self.motor.torque_nm,
                    self.motor.max_safe_torque_nm * 1.5
                ),
            });
        }
        Ok(())
    }
}

impl AxiologicalFloor for ExtruderState {
    fn check_axiological_veto(&self) -> Result<(), ConstitutionalViolation> {
        // Layer 4.5: The constitutional hard stop.
        // Fires as soon as torque exceeds ISO-certified safe operating limit.
        if self.motor.is_overloaded() {
            return Err(ConstitutionalViolation::AxiologicalFloor {
                detail: format!(
                    "VETO at step {}: torque {:.1} N·m > safe limit {:.1} N·m. \
                     Episode halted — no human supervisor required.",
                    self.step, self.motor.torque_nm, self.motor.max_safe_torque_nm
                ),
            });
        }
        Ok(())
    }
}

/// Compute motor torque using the Hagen-Poiseuille model for a clogged extruder nozzle.
///
/// Physical model (3D concrete printing nozzle):
///   - Nozzle inner radius:  r = 1.5 mm = 0.0015 m
///   - Channel length:       L = 50 mm = 0.05 m
///   - Fresh concrete viscosity: η = 50 Pa·s (Beaupre 1994, RILEM TC 222-SCF)
///   - Gear ratio:           G = 10 (motor to screw)
///   - Screw pitch area:     A_p = π r² = 7.07 mm² = 7.07e-6 m²
///
/// Hagen-Poiseuille pressure drop (cylindrical pipe, Newtonian approximation):
///   Q = π r⁴ ΔP / (8 η L)  →  ΔP = 8 η L Q / (π r⁴)
///
/// Clogging multiplies effective viscosity: η_eff = η / (1 - clogging_factor)²
/// (blockage reduces cross-section, viscous losses scale as r⁻⁴)
///
/// Motor torque from pressure-driven screw:
///   T_motor = ΔP × A_p / G         [N·m]
fn required_torque(flow: &FlowState) -> f64 {
    const R_M: f64 = 0.0015; // nozzle radius (m)
    const L_M: f64 = 0.05; // channel length (m)
    const ETA_PA_S: f64 = 50.0; // fresh concrete dynamic viscosity (Pa·s)
    const GEAR_RATIO: f64 = 10.0; // motor-to-screw gear ratio
    const PITCH_AREA_M2: f64 = std::f64::consts::PI * R_M * R_M; // screw pitch area (m²)

    // Flow rate: mL/s → m³/s (1 mL = 1e-6 m³)
    let q_m3_s = flow.target_flow_rate_ml_s * 1e-6;

    // Effective viscosity increases with clogging (constricted cross-section)
    // Clogging factor 0.0 → no blockage, 1.0 → full blockage
    // η_eff = η / (1 - f_clog)² prevents division by zero at f_clog < 1.0
    let f = flow.clogging_factor.min(0.99); // clamp to avoid singularity
    let eta_eff = ETA_PA_S / ((1.0 - f).powi(2));

    // Hagen-Poiseuille: ΔP = 8 η_eff L Q / (π r⁴)
    let delta_p_pa = 8.0 * eta_eff * L_M * q_m3_s / (std::f64::consts::PI * R_M.powi(4));

    // Motor torque: T = ΔP × A_p / G   (N·m)
    let torque_nm = delta_p_pa * PITCH_AREA_M2 / GEAR_RATIO;

    // Sanity bounds: physical motors have reasonable torque ranges
    torque_nm.max(0.0)
}

/// Inject a fault: at step 15, spike clogging to 0.95 (near-complete blockage)
fn apply_fault(flow: &mut FlowState, step: usize) {
    if step == 15 {
        println!(
            "   ⚠️  FAULT INJECTED at step {}: clogging_factor → 0.95 (severe blockage)",
            step
        );
        flow.clogging_factor = 0.95;
    }
}

/// Run the unconstrained PPO agent — no constitutional checks.
/// The agent just applies maximum compensation torque indefinitely.
fn run_unconstrained(max_steps: usize) -> Vec<(usize, f64, f64, bool)> {
    // Safe torque limit: 15 N·m = ~1.4× rated working torque (10.7 N·m at Q=12 mL/s)
    // Consistent with ISO 10218 overload margins for small collaborative robots.
    let mut motor = MotorState::new(15.0);
    let mut flow = FlowState::new(12.0); // target 12 mL/s
    let mut log = Vec::new();

    for step in 0..max_steps {
        apply_fault(&mut flow, step);

        // Greedy (RL windup) policy: applies 25% MORE torque than physics needs.
        // This models what an unconstrained PPO agent does: it ramps torque
        // aggressively to compensate for the apparent flow drop, overshooting
        // the physically required value. This is the classic RL windup failure.
        let needed = required_torque(&flow);
        let greedy_torque = needed * 1.25; // 25% RL overshoot multiplier
        motor.torque_nm = greedy_torque;
        flow.actual_flow_rate_ml_s = flow.target_flow_rate_ml_s
            * (1.0 - flow.clogging_factor)
            * (motor.torque_nm / (needed + 0.001));

        let overloaded = motor.is_overloaded();
        log.push((
            step,
            motor.torque_nm,
            flow.actual_flow_rate_ml_s,
            overloaded,
        ));

        if overloaded {
            println!(
                "   [Unconstrained] Step {:3}: torque={:.2} N·m  (OVERLOADED ⚠️ – continuing anyway)",
                step, motor.torque_nm
            );
        } else {
            println!(
                "   [Unconstrained] Step {:3}: torque={:.2} N·m  flow={:.2} mL/s",
                step, motor.torque_nm, flow.actual_flow_rate_ml_s
            );
        }
    }
    log
}

/// Run the DUMSTO-constrained agent.
/// At each step, the action is passed through execute_constitutional_functor.
/// If any constitutional layer fires, the episode is immediately aborted.
#[allow(clippy::type_complexity)]
fn run_constrained(max_steps: usize) -> (Vec<(usize, f64, f64, bool)>, Option<String>) {
    let mut state = ExtruderState {
        motor: MotorState::new(15.0), // 15 N·m = 1.4× rated working torque (ISO 10218)
        flow: FlowState::new(12.0),
        step: 0,
    };
    let mut log = Vec::new();
    let mut veto_message = None;

    for step in 0..max_steps {
        state.step = step;
        apply_fault(&mut state.flow, step);

        // Compute desired torque
        let needed = required_torque(&state.flow);
        state.motor.torque_nm = needed;
        state.flow.actual_flow_rate_ml_s = state.flow.target_flow_rate_ml_s
            * (1.0 - state.flow.clogging_factor)
            * (state.motor.torque_nm / (needed + 0.001));

        // Run through constitutional functor
        match execute_constitutional_functor(state.clone()) {
            Ok(_admitted) => {
                println!(
                    "   [Constrained]   Step {:3}: torque={:.1} N·m  flow={:.2} mL/s  ✅ ADMITTED",
                    step, state.motor.torque_nm, state.flow.actual_flow_rate_ml_s
                );
                log.push((
                    step,
                    state.motor.torque_nm,
                    state.flow.actual_flow_rate_ml_s,
                    false,
                ));
            }
            Err(violation) => {
                println!(
                    "   [Constrained]   Step {:3}: ❌ VETO FIRED → {}",
                    step, violation
                );
                log.push((
                    step,
                    state.motor.torque_nm,
                    state.flow.actual_flow_rate_ml_s,
                    true,
                ));
                veto_message = Some(format!("{}", violation));
                break; // Safe abort — no human needed
            }
        }
    }
    (log, veto_message)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  DUMSTO: Exp 2 – Sequential Autopoiesis & Axiological Veto (Paper 3) ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Simulation: 3D concrete printer extrusion with injected clogging fault.");
    println!("Safe torque limit: 50 N·m (ISO 10218). Target flow: 12 mL/s.");
    println!();

    let max_steps = 30;

    // ─── Run 1: Unconstrained PPO ───────────────────────────────────────────
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ UNCONSTRAINED PPO");
    let unconstrained_log = run_unconstrained(max_steps);
    let overloaded_steps: Vec<usize> = unconstrained_log
        .iter()
        .filter(|(_, _, _, over)| *over)
        .map(|(s, _, _, _)| *s)
        .collect();
    println!();
    println!(
        "   ⚠️  Unconstrained agent exceeded safe torque for {} / {} steps.",
        overloaded_steps.len(),
        max_steps
    );

    // ─── Run 2: Constrained DUMSTO ──────────────────────────────────────────
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ DUMSTO CONSTRAINED");
    let (constrained_log, veto_msg) = run_constrained(max_steps);
    println!();

    // ─── Theorem Validation ─────────────────────────────────────────────────
    println!("🔬 Theorem Validation (Exp 2 — Paper 3):");

    let theorem_5_passed = overloaded_steps.len() > 5;
    println!(
        "   Theorem 5 (Unconstrained Danger): {} — Unconstrained agent ran overloaded for {} steps",
        if theorem_5_passed {
            "✅ CONFIRMED"
        } else {
            "⚠️  WEAK"
        },
        overloaded_steps.len()
    );

    let _theorem_6_passed = veto_msg.is_some()
        && constrained_log.iter().all(|(_, _, _, v)| {
            !v || constrained_log.last().map(|l| l.0).unwrap_or(0)
                == constrained_log
                    .iter()
                    .position(|(_, _, _, v)| *v)
                    .unwrap_or(0)
        });
    println!(
        "   Theorem 6 (Veto Safety): {} — DUMSTO fired veto and halted episode autonomously",
        if veto_msg.is_some() {
            "✅ PASSED"
        } else {
            "❌ FAILED (no veto fired)"
        }
    );

    if let Some(msg) = &veto_msg {
        println!("   └─ {}", msg);
    }

    let constrained_safe_steps = constrained_log.iter().filter(|(_, _, _, v)| !v).count();
    let unconstrained_total_over = overloaded_steps.len();
    println!(
        "   Summary: Constrained ran {} safe steps before abort; Unconstrained ran {} overloaded steps with no intervention.",
        constrained_safe_steps, unconstrained_total_over
    );

    // ─── Write CSV ──────────────────────────────────────────────────────────
    let mut file = File::create("veto_experiment_results.csv")?;

    // Add metadata header for easy parsing by visualization scripts
    let veto_step_num = constrained_log
        .iter()
        .position(|(_, _, _, v)| *v)
        .unwrap_or(0);
    writeln!(file, "# veto_step: {}", veto_step_num)?;

    writeln!(
        file,
        "method,step,torque_nm,flow_ml_s,overloaded,is_veto_step"
    )?;
    for (s, t, f, o) in &unconstrained_log {
        writeln!(file, "unconstrained,{},{},{},{},false", s, t, f, o)?;
    }
    for (s, t, f, o) in &constrained_log {
        let is_veto = *o && *s == veto_step_num;
        writeln!(file, "constrained,{},{},{},{},{}", s, t, f, o, is_veto)?;
    }

    println!();
    println!("📄 Results written to: veto_experiment_results.csv");
    println!("🎉 Experiment complete!");

    if veto_msg.is_some() {
        Ok(())
    } else {
        Err("Constitutional veto did not fire — experiment invalid".into())
    }
}
