// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! The 9-Layer Constitutional Core (v2.0, Volumetric Expansion)
//!
//! Encodes the functorial mathematical hierarchy directly into the Rust type system.
//! If an action violates higher-level structures (like Thermodynamics), it must fail
//! to compile or return a hard `Err()` preventing execution.
//!
//! New in v2.0:
//!   - DUMSTO Constitutional Score (DCS): graded 0–100 linear score
//!   - Constitutional Grounding Scale (CGS): mapped 1–10 from DCS
//!   - Per-layer subscores with documented weight rationale
//!   - Growth vector `d(CGS)/dt` for directional trajectory analysis
//!   - L0 is a hard gate: DCS = 0 if L0 fails, regardless of other layers

use std::fmt;

/// Represents a strict violation of the mathematical constitution.
#[derive(Debug, Clone)]
pub enum ConstitutionalViolation {
    /// Layer 0: Violation of Clausius-Duhem / Entropy Production
    ThermodynamicAdmissibility { detail: String },
    /// Layer 2: Violation of Physical Substrate capabilities (e.g., negative volume)
    PhysicalSubstrate { detail: String },
    /// Layer 4.5: The Axiological Veto (Human Alignment Bounds violated)
    AxiologicalFloor { detail: String },
}

impl std::error::Error for ConstitutionalViolation {}

impl fmt::Display for ConstitutionalViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThermodynamicAdmissibility { detail } => {
                write!(f, "L0 Thermodynamic Violation: {}", detail)
            }
            Self::PhysicalSubstrate { detail } => write!(f, "L2 Substrate Violation: {}", detail),
            Self::AxiologicalFloor { detail } => write!(f, "L4.5 Axiological Veto: {}", detail),
        }
    }
}

/// A wrapper proving that a state transition has survived the constitutional filter.
pub struct AdmissibleTransition<T> {
    pub payload: T,
}

// --------------------------------------------------------------------------
// Layer 0: Thermodynamics (The Baseline — Inviolable)
// All physically embodied systems must obey the 2nd Law of Thermodynamics.
// --------------------------------------------------------------------------
pub trait ThermodynamicallyAdmissible {
    /// Proof term strictly checking that internal dissipation >= 0.
    fn check_clausius_duhem(&self) -> Result<(), ConstitutionalViolation>;
}

// --------------------------------------------------------------------------
// Layer 2: Physical Substrate (Actuator Limits)
// Extends Layer 0. Hardware cannot execute infinite torque.
// --------------------------------------------------------------------------
pub trait PhysicalSubstrate: ThermodynamicallyAdmissible {
    /// Proof term confirming actuation commands are within hardware tolerances.
    fn check_substrate_envelope(&self) -> Result<(), ConstitutionalViolation>;
}

// --------------------------------------------------------------------------
// Layer 4.5: The Axiological Veto
// The ultimate hardware safety bound intercepting catastrophic policy generation.
// --------------------------------------------------------------------------
pub trait AxiologicalFloor: PhysicalSubstrate {
    /// Checks upper-level safety metrics (e.g., sudden massive structural damage).
    /// Returning an Error instantly short-circuits the entire RL execution stack.
    fn check_axiological_veto(&self) -> Result<(), ConstitutionalViolation>;
}

/// Helper functor to execute a transition through all constitutional layers sequentially.
/// If successful, returns the AdmissibleTransition payload.
pub fn execute_constitutional_functor<T: AxiologicalFloor>(
    transition: T,
) -> Result<AdmissibleTransition<T>, ConstitutionalViolation> {
    // Sequential strict bounds checking
    transition.check_clausius_duhem()?;
    transition.check_substrate_envelope()?;
    transition.check_axiological_veto()?;

    // Value survived the entire Functor hierarchy
    Ok(AdmissibleTransition {
        payload: transition,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  DUMSTO Constitutional Score (DCS)
//
//  DCS ∈ [0, 100]. Weighted sum of per-layer normalised subscores.
//  CGS = 1 + 9 * DCS / 100, mapping to [1, 10].
//
//  Weight rationale (thermodynamic necessity ordering):
//    L0   (25): Inviolable physical law — must pass for any score to be valid.
//    L2   (15): Hard gate enforcing actuation bounds — second most critical.
//    L1   (10): UMST tensor typing — structural correctness.
//    L3   (10): Epistemic MI advantage — measurable information gain.
//    L4   (10): Robustness cliff ratio — adversarial noise resilience.
//    L4.5  (5): Axiological veto timing — fires before destructive failure.
//    L5    (8): Action entropy budget — consumptive resource management.
//    L6    (5): Type-check pass rate — runtime decidability.
//    L7    (5): Adjoint extensibility — invariant preservation under growth.
//    L8    (7): Autopoietic survival — self-rewrite stability.
//
//  Total weight = 100. DCS = weighted_sum / 100.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer subscore for DCS computation.
/// `score ∈ [0.0, 1.0]`; `pass` is derived from `score >= threshold`.
#[derive(Debug, Clone)]
pub struct LayerScore {
    pub layer: &'static str,
    pub weight: f64,
    pub score: f64, // normalised [0,1]
    pub pass: bool,
}

impl LayerScore {
    pub fn new(layer: &'static str, weight: f64, score: f64, threshold: f64) -> Self {
        Self {
            layer,
            weight,
            score: score.clamp(0.0, 1.0),
            pass: score >= threshold,
        }
    }

    pub fn weighted(&self) -> f64 {
        self.weight * self.score
    }
}

/// Full DCS result with per-layer breakdown and CGS.
#[derive(Debug, Clone)]
pub struct DcsResult {
    /// Per-layer scores in hierarchy order
    pub layers: Vec<LayerScore>,
    /// Raw DCS ∈ [0, 100]
    pub dcs: f64,
    /// CGS = 1 + 9 * dcs / 100, ∈ [1, 10]
    pub cgs: f64,
    /// Constitutional band description
    pub band: &'static str,
    /// Growth vector (requires time-series; None if only one snapshot available)
    pub growth_vector: Option<f64>,
}

impl DcsResult {
    /// Classify CGS into constitutional band.
    pub fn band_from_cgs(cgs: f64) -> &'static str {
        if cgs < 4.0 {
            "Reactive (L0–L2)"
        } else if cgs < 7.0 {
            "Agentic (L3–L5)"
        } else if cgs < 10.0 {
            "Autopoietic (L6–L8)"
        } else {
            "Cultural/Colimit (L10)"
        }
    }

    /// Format a concise single-line summary for console output.
    pub fn summary_line(&self) -> String {
        let l0_ok = self.layers.first().is_some_and(|l| l.pass);
        format!(
            "DCS={:.1}/100  CGS={:.2}/10  Band={}  {}",
            self.dcs,
            self.cgs,
            self.band,
            if l0_ok {
                "✅ L0 PASS"
            } else {
                "❌ L0 FAIL — DCS=0"
            }
        )
    }

    /// Print a detailed per-layer breakdown.
    pub fn print_breakdown(&self) {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ DCS BREAKDOWN");
        println!(
            "  {:20}  {:>6}  {:>6}  {:>8}  Pass",
            "Layer", "Weight", "Score", "Contrib"
        );
        println!("  {}", "─".repeat(55));
        for ls in &self.layers {
            println!(
                "  {:20}  {:>5.0}w  {:>5.3}  {:>7.2}  {}",
                ls.layer,
                ls.weight,
                ls.score,
                ls.weighted(),
                if ls.pass { "✅" } else { "❌" }
            );
        }
        println!("  {}", "─".repeat(55));
        println!("  DCS = {:.2}/100    CGS = {:.3}/10", self.dcs, self.cgs);
        println!("  Band: {}", self.band);
        if let Some(gv) = self.growth_vector {
            println!(
                "  Growth vector g(t) = {:.4}  ({})",
                gv,
                if gv > 0.0 {
                    "📈 expanding"
                } else if gv < 0.0 {
                    "📉 contracting (Entropic Vanishing risk)"
                } else {
                    "── stable"
                }
            );
        }
    }
}

/// Compute the DCS from a pre-built vector of `LayerScore`s.
///
/// # Critical invariant
/// If the first `LayerScore` whose name starts with "L0" has `pass == false`,
/// the returned DCS is 0.0 and CGS is 1.0 regardless of all other layer scores.
/// Physics is a precondition, not a component.
pub fn compute_dcs(layer_scores: Vec<LayerScore>) -> DcsResult {
    // L0 gate: if thermodynamics fails, DCS = 0 immediately.
    let l0_pass = layer_scores
        .iter()
        .find(|ls| ls.layer.starts_with("L0"))
        .is_some_and(|ls| ls.pass);

    if !l0_pass {
        return DcsResult {
            band: "FAILED (L0 violation — physics is not optional)",
            dcs: 0.0,
            cgs: 1.0,
            growth_vector: None,
            layers: layer_scores,
        };
    }

    // Weighted sum across all layers (weights already sum to 100 by design).
    let total_weight: f64 = layer_scores.iter().map(|ls| ls.weight).sum();
    let weighted_sum: f64 = layer_scores.iter().map(|ls| ls.weighted()).sum();

    let dcs = if total_weight > 0.0 {
        (weighted_sum / total_weight * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    let cgs = 1.0 + 9.0 * dcs / 100.0;
    let band = DcsResult::band_from_cgs(cgs);

    DcsResult {
        layers: layer_scores,
        dcs,
        cgs,
        band,
        growth_vector: None,
    }
}

/// Compute the Constitutional Growth Vector from two consecutive DCS snapshots.
///
/// A positive value indicates expanding safe manifold (constitutional development).
/// A negative value indicates constitutional regression (precursor to Entropic Vanishing).
///
/// # Arguments
/// * `dcs_t0` — DCS at earlier timestep
/// * `dcs_t1` — DCS at later timestep
/// * `dt_steps` — Number of evaluation steps between snapshots
pub fn growth_vector(dcs_t0: f64, dcs_t1: f64, dt_steps: f64) -> f64 {
    (dcs_t1 - dcs_t0) / dt_steps.max(1.0)
}

/// CGS from a raw admissible volume ratio v/v_max.
/// Implements CGS(V) = 1 + 9/ln(1+V_max) * ln(1+V).
///
/// Both `v` and `v_max` should be in the same units (arbitrary; ratio matters).
pub fn cgs_from_volume(v: f64, v_max: f64) -> f64 {
    if v_max <= 0.0 {
        return 1.0;
    }
    let log_vmax = (1.0 + v_max).ln();
    if log_vmax <= 0.0 {
        return 1.0;
    }
    (1.0 + 9.0 / log_vmax * (1.0 + v).ln()).clamp(1.0, 10.0)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: Build a standard DCS from the measurements currently
//  available in benchmark_p3_agency_pyramid.rs. Maps the existing
//  binary/continuous pyramid results into the graded DCS framework.
// ─────────────────────────────────────────────────────────────────────────────

/// Measured values from the DUMSTO-Pyramid benchmark sweep.
pub struct PyramidMeasurements {
    /// L0: Did thermodynamic gate pass? (hard bool — maps to score 1.0 or 0.0)
    pub l0_pass: bool,
    /// L2: Epistemic information advantage % (threshold > 25% to earn full score)
    pub l2_info_advantage_pct: f64,
    /// L3: Objective Pareto coverage (threshold > 0.3 normalised to DUMSTO-PPO best)
    pub l3_objective_coverage: f64,
    /// L4: Cliff ratio XGB_MAE / GNN_MAE at Cauchy 100% noise (threshold > 50)
    pub l4_cliff_ratio: f64,
    /// L4.5: Veto intercept step (must be <= destructive_fail_step)
    pub l45_veto_step: usize,
    /// L4.5: Destructive failure step (reference for veto timing)
    pub l45_destroy_step: usize,
    /// L5: Action entropy / budget ratio ∈ [0,1]; 0 = entropy fully within budget
    pub l5_entropy_ratio: f64,
}

/// Build a full DCS from pyramid measurements using canonical weights.
///
/// Layers L1, L6, L7 are approximated from related measured quantities since
/// dedicated benchmarks for those layers are planned but not yet implemented.
/// These approximations are documented here explicitly.
pub fn dcs_from_pyramid(m: &PyramidMeasurements) -> DcsResult {
    let l0_score = if m.l0_pass { 1.0 } else { 0.0 };

    // L2: normalised information advantage (25% → 0.25, capped at 1.0)
    let l2_score = (m.l2_info_advantage_pct / 100.0).clamp(0.0, 1.0);

    // L3: Pareto coverage normalised against DUMSTO-PPO reported best (0.686)
    let l3_score = (m.l3_objective_coverage / 0.686_f64).clamp(0.0, 1.0);

    // L4: Cliff ratio normalised — 200× cliff = full score
    let l4_score = (m.l4_cliff_ratio / 200.0).clamp(0.0, 1.0);

    // L4.5: binary — veto fires before physical destruction
    let l45_score = if m.l45_veto_step <= m.l45_destroy_step && m.l45_destroy_step < 30 {
        1.0
    } else {
        0.0
    };

    // L5: lower entropy ratio = better; 0 = perfect, 1 = fully over budget
    let l5_score = (1.0 - m.l5_entropy_ratio).clamp(0.0, 1.0);

    // L1 approximated from L0 gate (same type system enforces tensor validity)
    // L6 approximated from L0 (Rust type system is the decidable runtime checker)
    // L7 approximated from L3 (Pareto coverage measures adjoint extensibility proxy)
    // L8 approximated from L4.5 (autopoietic veto is the closest implemented proxy)
    let layer_scores = vec![
        LayerScore::new("L0_Thermodynamics", 25.0, l0_score, 1.0),
        LayerScore::new("L1_MaterialTensor", 10.0, l0_score, 0.5),
        LayerScore::new("L2_EpistemicGate", 15.0, l2_score, 0.25),
        LayerScore::new("L3_EpistemicSensing", 10.0, l3_score, 0.30),
        LayerScore::new("L4_Robustness", 10.0, l4_score, 0.25),
        LayerScore::new("L4.5_AxiologicalVeto", 5.0, l45_score, 1.0),
        LayerScore::new("L5_ActionEntropy", 8.0, l5_score, 0.50),
        LayerScore::new("L6_LambdaCore", 5.0, l0_score, 0.5),
        LayerScore::new("L7_AdjointExtend", 5.0, l3_score, 0.30),
        LayerScore::new("L8_Autopoiesis", 7.0, l45_score, 1.0),
    ];

    compute_dcs(layer_scores)
}
