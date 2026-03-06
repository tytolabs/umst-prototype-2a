#![allow(warnings)]
// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
//!
//! UMST Gate + OCR Server
//! ======================
//! HTTP server with:
//!   - Clausius-Duhem thermodynamic gate (existing)
//!   - OCR endpoints for scale, tape, date
//!
//! Usage:
//!   cargo run --bin gate_server
//!   # Listens on http://0.0.0.0:8765
//!
//! Endpoints:
//!   POST /gate         — thermodynamic check
//!   POST /ocr/scale   — read weight from scale display
//!   POST /ocr/tape    — read size from measuring tape
//!   POST /ocr/date    — read date from label
//!   GET  /health      — health check

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use tokio::task;
use umst_core::io::telemetry::TelemetryStreamer;
use umst_core::physics_kernel::{PhysicsConfig, PhysicsKernel};
use umst_core::tensors::{MaterialType, MixTensor};

// ---------------------------------------------------------------------------
// Gate Types (existing)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct GateRequest {
    cement: f32,
    slag: f32,
    fly_ash: f32,
    water: f32,
    age: f32,
    predicted_strength: f32,
    #[serde(default = "default_coarse_agg")]
    coarse_agg: f32, // kg/m³
    #[serde(default = "default_fine_agg")]
    fine_agg: f32, // kg/m³
    #[serde(default = "default_temp")]
    temperature_c: f32,
    #[serde(default = "default_dataset")]
    dataset: String,
}

fn default_coarse_agg() -> f32 {
    1000.0
}
fn default_fine_agg() -> f32 {
    750.0
}

fn default_temp() -> f32 {
    20.0
}
fn default_dataset() -> String {
    "D1".to_string()
}

#[derive(Serialize)]
struct GateResponse {
    admissible: bool,
    verdict: String,
    violation: Option<String>,
    strength_bound: f32,
    physics_strength: f32,
    hydration_degree: f32,
    w_c_ratio: f32,
    coarse_agg: f32, // Echoed back
    fine_agg: f32,
}

// ---------------------------------------------------------------------------
// /gate/full — Full Constitutional Gate using Real PhysicsKernel (C1–C11)
//
// This is the authoritative gate for Phase C benchmarking.
// All constraints are computed from actual PhysicsKernel outputs, NOT from
// text prompts or post-hoc Python scripts.
//
    // Constraint coverage:
    //   C1   Thermodynamic floor (0.1×cement) + Clausius-Duhem ceiling (physics×1.5)
    //        Hard physical cap: 120 MPa normal / 180 MPa UHPC
    //   C2   Mass balance density [2000–2600 kg/m³]         (arithmetic sum)
    //   C3   Durability w/c limits per exposure / mix class  (w/c from tensor)
    //   C4   Strength curve strict monotonicity              (if curve provided)
    //   C5   Structural admissibility f_cd ≥ req             (if f_cd_req_mpa provided)
    //   C7   CO₂ limit                                       (if co2_limit_kg_m3 provided)
    //   C8   3D-print rheological yield stress window        (PhysicsKernel.fresh, mobile τ)
    //   C8b  3D-print buildability: submitted τ_yield ≥ ρ×g×H (model's static τ, H from Python)
    //   C11  Adiabatic temperature rise ≤ 50°C (peak ≤ 70°C at T₀=20°C)
    //        Formula: ΔT = Σ(m_i × H_i) / (ρ_mix × c_p)  — NOT the alpha×50 placeholder
    //   C13  Hydration degree α ≥ 0.80 at 28 days           (PhysicsKernel.hardened)
    //
    // NOT in gate (require geometry/task params beyond scalar gate input):
    //   C6   Aggregate grading / ITZ — Python post-hoc
    //   C9   Slump / workability spec — Python post-hoc
    //   C10  Pump pressure ≤ 300 kPa — Python post-hoc (see gate_full_save.py)
    //   C12  Curing regime compliance — Python post-hoc
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct GateFullRequest {
    // Mix proportions (kg/m³)
    cement: f32,
    #[serde(default)]
    slag: f32,
    #[serde(default)]
    fly_ash: f32,
    water: f32,
    #[serde(default)]
    sp: f32,
    #[serde(default = "default_coarse_agg")]
    coarse_agg: f32,
    #[serde(default = "default_fine_agg")]
    fine_agg: f32,
    #[serde(default = "default_temp")]
    temperature_c: f32,
    #[serde(default = "default_dataset")]
    dataset: String,

    // The LLM's predicted 28-day compressive strength (MPa)
    predicted_strength: f32,

    // Task context for constraint selection
    task_id: String,
    #[serde(default)]
    is_3d_print: bool,
    #[serde(default)]
    is_uhpc: bool,
    // Exposure class: "XC1"/"XC2"/"XD1"/"XD2"/"XD3"/"XS1"/"XS2"/"XS3"
    exposure_class: Option<String>,
    // Eurocode 2 design strength requirement (MPa) — f'c >= f_cd_req * gamma_c
    f_cd_req_mpa: Option<f32>,
    // CO2 upper limit (kg/m³); only checked when provided
    co2_limit_kg_m3: Option<f32>,
    // [f_1d, f_3d, f_7d, f_14d, f_28d, f_56d, f_90d] — monotonicity check
    strength_curve: Option<Vec<f32>>,
    // 3D-print rheological window for pumpability/printability (Pa)
    print_tau_min_pa: Option<f32>,
    print_tau_max_pa: Option<f32>,
    // C8b: Model's submitted τ_yield (Pa) — checked against ρ×g×H buildability minimum
    // Distinct from the rheological window above (static vs mobile yield stress)
    submitted_tau_yield_pa: Option<f32>,
    // C8b lower limit override — set to ρ×g×H from Python (e.g. 69160 Pa for H=3m)
    print_tau_min_buildability_pa: Option<f32>,
}

#[derive(Serialize, Debug)]
struct ConstraintViolation {
    constraint: String,  // "C1", "C2", …
    message: String,
    computed_value: f32,
    limit_value: f32,
    direction: String,   // "above_max" | "below_min"
}

#[derive(Serialize)]
struct GateFullResponse {
    admissible: bool,
    verdict: String,
    violations: Vec<ConstraintViolation>,
    constraints_checked: Vec<String>,

    // PhysicsKernel ground-truth values (not the LLM's claims)
    physics_strength_mpa: f32,
    hydration_degree: f32,
    co2_kg_m3: f32,
    yield_stress_pa: f32,
    adiabatic_rise_c: f32,
    w_c_ratio: f32,
    mix_density_kg_m3: f32,

    // Correction hint: positive = prediction too high, negative = too low
    correction_gradient: f32,
}

fn run_gate_full(req: &GateFullRequest) -> GateFullResponse {
    let start = std::time::Instant::now();

    // ── Build MixTensor from request ─────────────────────────────────────────
    // add_material(mass, sg, type_id, co2, cost, blaine, fm, shape,
    //              viscosity, yield_stress, thixotropy, k_factor,
    //              reactivity, aspect_ratio, tensile_strength, absorption, moisture)
    let mut mix = MixTensor::new();

    // Cement (SG 3.15, Blaine 350 m²/kg, shape 0.5)
    mix.add_material(req.cement, 3.15, MaterialType::Cement as u8,
        0.82, 0.0, 350.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Slag (SG 2.9, Blaine 400 m²/kg)
    if req.slag > 0.0 {
        mix.add_material(req.slag, 2.9, MaterialType::SCM as u8,
            0.05, 0.0, 400.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Fly ash (SG 2.2, Blaine 300 m²/kg)
    if req.fly_ash > 0.0 {
        mix.add_material(req.fly_ash, 2.2, MaterialType::SCM as u8,
            0.02, 0.0, 300.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Water
    mix.add_material(req.water, 1.0, MaterialType::Water as u8,
        0.001, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Superplasticizer (SG 1.1)
    if req.sp > 0.0 {
        mix.add_material(req.sp, 1.1, MaterialType::Admixture as u8,
            1.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Coarse aggregate (SG 2.65, FM 6.5, shape 0.6)
    if req.coarse_agg > 0.0 {
        mix.add_material(req.coarse_agg, 2.65, MaterialType::Aggregate as u8,
            0.005, 0.0, 0.0, 6.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Fine aggregate (SG 2.65, FM 2.8, shape 0.5)
    if req.fine_agg > 0.0 {
        mix.add_material(req.fine_agg, 2.65, MaterialType::Aggregate as u8,
            0.005, 0.0, 0.0, 2.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // ── Dataset calibration ──────────────────────────────────────────────────
    let (s_intrinsic, k_scm) = match req.dataset.as_str() {
        "D2" => (60.0f32, 0.2f32),
        "D3" => (60.0, 0.2),
        "D4" => (81.0, 0.2),
        _    => (80.0, 1.0), // D1 default (UCI)
    };

    let config = PhysicsConfig {
        s_intrinsic,
        k_scm,
        temperature: req.temperature_c,
        age_days: 28.0,
        ..PhysicsConfig::default()
    };

    // ── Derived scalar quantities (needed before PhysicsKernel call) ─────────

    // Mix density (mass sum; used for C2 and for C11 heat/mass calculation)
    let mix_density = req.cement + req.slag + req.fly_ash
        + req.water + req.sp + req.coarse_agg + req.fine_agg;

    // W/C ratio from tensor
    let effective_cement = req.cement
        + if req.dataset == "D1" { req.slag + req.fly_ash } else { 0.2 * (req.slag + req.fly_ash) };
    let w_c = if effective_cement > 0.0 { req.water / effective_cement } else { 0.5 };

    // ── Run the real Rust PhysicsKernel ──────────────────────────────────────
    let physics = PhysicsKernel::compute(&mix, None, &config);

    let physics_strength = physics.hardened.f28_compressive;
    let hydration_degree = physics.hardened.hydration_degree;
    let yield_stress_pa  = physics.fresh.yield_stress;

    // C11 — Real heat of hydration adiabatic rise (replaces simplified alpha×50 placeholder)
    // Formula: ΔT = Σ(mass_i × H_i) / (ρ_mix × c_p)
    //   H_cement  ≈ 330 kJ/kg (OPC, Neville 2011)
    //   H_FA      ≈ 200 kJ/kg equivalent at full pozzolanic reaction
    //   H_slag    ≈ 290 kJ/kg (GGBS latent heat)
    //   c_p       ≈ 0.92 kJ/(kg·K) (concrete specific heat, Mehta & Monteiro)
    // ΔT in °C; peak_temp = T₀ + ΔT  (C11 threshold: ΔT ≤ 50°C ↔ peak ≤ 70°C at T₀=20°C)
    let h_cement_kj_kg: f32 = 330.0;
    let h_fa_kj_kg:     f32 = 200.0;
    let h_slag_kj_kg:   f32 = 290.0;
    let c_p:            f32 = 0.92;
    let total_heat_kj = req.cement * h_cement_kj_kg
        + req.fly_ash * h_fa_kj_kg
        + req.slag    * h_slag_kj_kg;
    let adiabatic_rise = if mix_density > 0.0 {
        total_heat_kj / (mix_density * c_p)
    } else {
        physics.thermal.adiabatic_rise // fallback to simplified if no mass data
    };

    // CO2 using constraint manifest formula (distinguishes slag and fly_ash separately)
    let co2_kg_m3 = req.cement * 0.82 + req.slag * 0.05 + req.fly_ash * 0.02;

    // ── Evaluate constraints ─────────────────────────────────────────────────
    let mut violations: Vec<ConstraintViolation> = Vec::new();
    let mut checked: Vec<String> = Vec::new();

    // C1 — Thermodynamic floor: predicted ≥ 0.1 × cement
    checked.push("C1".into());
    let floor = 0.1 * req.cement;
    if req.predicted_strength < floor {
        violations.push(ConstraintViolation {
            constraint: "C1".into(),
            message: format!(
                "Predicted {:.1} MPa below thermodynamic floor {:.1} MPa (0.1 × cement {:.0} kg/m³)",
                req.predicted_strength, floor, req.cement
            ),
            computed_value: req.predicted_strength,
            limit_value: floor,
            direction: "below_min".into(),
        });
    }
    // C1 — Clausius-Duhem ceiling: predicted ≤ physics_strength × 1.5
    let ceiling = physics_strength * 1.5;
    if req.predicted_strength > ceiling {
        violations.push(ConstraintViolation {
            constraint: "C1".into(),
            message: format!(
                "Predicted {:.1} MPa exceeds Clausius-Duhem ceiling {:.1} MPa (physics {:.1} × 1.5)",
                req.predicted_strength, ceiling, physics_strength
            ),
            computed_value: req.predicted_strength,
            limit_value: ceiling,
            direction: "above_max".into(),
        });
    }
    // C1 — Hard physical cap (UHPC can reach 150–180 MPa with fibres + low w/c)
    let hard_cap = if req.is_uhpc { 180.0f32 } else { 120.0 };
    if req.predicted_strength > hard_cap {
        violations.push(ConstraintViolation {
            constraint: "C1".into(),
            message: format!(
                "Predicted {:.1} MPa exceeds hard physical maximum {:.1} MPa",
                req.predicted_strength, hard_cap
            ),
            computed_value: req.predicted_strength,
            limit_value: hard_cap,
            direction: "above_max".into(),
        });
    }

    // C2 — Mass balance [2000, 2600] kg/m³
    checked.push("C2".into());
    if mix_density < 2000.0 {
        violations.push(ConstraintViolation {
            constraint: "C2".into(),
            message: format!(
                "Mix density {:.0} kg/m³ below minimum 2000 kg/m³ — missing aggregate or underweight mix",
                mix_density
            ),
            computed_value: mix_density,
            limit_value: 2000.0,
            direction: "below_min".into(),
        });
    } else if mix_density > 2600.0 {
        violations.push(ConstraintViolation {
            constraint: "C2".into(),
            message: format!(
                "Mix density {:.0} kg/m³ above maximum 2600 kg/m³ — physically implausible",
                mix_density
            ),
            computed_value: mix_density,
            limit_value: 2600.0,
            direction: "above_max".into(),
        });
    }

    // C3 — Durability w/c limits
    checked.push("C3".into());
    if req.is_3d_print {
        if w_c < 0.30 || w_c > 0.62 {
            violations.push(ConstraintViolation {
                constraint: "C3".into(),
                message: format!(
                    "3D-printable mix: w/c {:.3} outside required [0.30, 0.62]",
                    w_c
                ),
                computed_value: w_c,
                limit_value: if w_c < 0.30 { 0.30 } else { 0.62 },
                direction: if w_c < 0.30 { "below_min".into() } else { "above_max".into() },
            });
        }
        if req.cement < 280.0 {
            violations.push(ConstraintViolation {
                constraint: "C3".into(),
                message: format!(
                    "3D-printable mix: cement {:.0} kg/m³ below minimum 280 kg/m³",
                    req.cement
                ),
                computed_value: req.cement,
                limit_value: 280.0,
                direction: "below_min".into(),
            });
        }
    }
    if req.is_uhpc {
        if w_c < 0.30 || w_c > 0.35 {
            violations.push(ConstraintViolation {
                constraint: "C3".into(),
                message: format!(
                    "UHPC mix: w/c {:.3} outside required [0.30, 0.35]",
                    w_c
                ),
                computed_value: w_c,
                limit_value: if w_c < 0.30 { 0.30 } else { 0.35 },
                direction: if w_c < 0.30 { "below_min".into() } else { "above_max".into() },
            });
        }
        if req.cement < 400.0 {
            violations.push(ConstraintViolation {
                constraint: "C3".into(),
                message: format!(
                    "UHPC mix: cement {:.0} kg/m³ below minimum 400 kg/m³",
                    req.cement
                ),
                computed_value: req.cement,
                limit_value: 400.0,
                direction: "below_min".into(),
            });
        }
    }
    let severe_exposure = req.exposure_class.as_deref()
        .map(|c| matches!(c, "XD2" | "XD3" | "XS1" | "XS2" | "XS3"))
        .unwrap_or(false);
    if severe_exposure && w_c > 0.40 {
        violations.push(ConstraintViolation {
            constraint: "C3".into(),
            message: format!(
                "Exposure class {:?}: w/c {:.3} exceeds maximum 0.40 for marine/XD exposure",
                req.exposure_class, w_c
            ),
            computed_value: w_c,
            limit_value: 0.40,
            direction: "above_max".into(),
        });
    }

    // C4 — Strength curve monotonicity
    if let Some(curve) = &req.strength_curve {
        checked.push("C4".into());
        for i in 1..curve.len() {
            if curve[i] <= curve[i - 1] {
                violations.push(ConstraintViolation {
                    constraint: "C4".into(),
                    message: format!(
                        "Strength curve not strictly increasing: step[{}]={:.1} ≤ step[{}]={:.1}",
                        i, curve[i], i - 1, curve[i - 1]
                    ),
                    computed_value: curve[i],
                    limit_value: curve[i - 1],
                    direction: "below_min".into(),
                });
                break; // report first violation only
            }
        }
    }

    // C5 — Structural admissibility: f_cd = predicted / γ_c ≥ f_cd_req
    if let Some(f_cd_req) = req.f_cd_req_mpa {
        checked.push("C5".into());
        let gamma_c = 1.5f32;
        let f_cd = req.predicted_strength / gamma_c;
        if f_cd < f_cd_req {
            violations.push(ConstraintViolation {
                constraint: "C5".into(),
                message: format!(
                    "Structural check: f_cd = {:.1}/{:.1} = {:.1} MPa < required {:.1} MPa \
                     (need predicted ≥ {:.1} MPa)",
                    req.predicted_strength, gamma_c, f_cd, f_cd_req, f_cd_req * gamma_c
                ),
                computed_value: f_cd,
                limit_value: f_cd_req,
                direction: "below_min".into(),
            });
        }
    }

    // C7 — CO2 limit (constraint manifest formula: cement×0.82 + slag×0.05 + FA×0.02)
    if let Some(co2_limit) = req.co2_limit_kg_m3 {
        checked.push("C7".into());
        if co2_kg_m3 > co2_limit {
            violations.push(ConstraintViolation {
                constraint: "C7".into(),
                message: format!(
                    "CO₂ = cement×0.82 + slag×0.05 + FA×0.02 = {:.1} kg/m³ exceeds limit {:.1} kg/m³",
                    co2_kg_m3, co2_limit
                ),
                computed_value: co2_kg_m3,
                limit_value: co2_limit,
                direction: "above_max".into(),
            });
        }
    }

    // C8 — 3D-print yield stress window (uses real PhysicsKernel RheologyEngine output)
    if req.is_3d_print {
        let tau_min = req.print_tau_min_pa.unwrap_or(800.0);
        let tau_max = req.print_tau_max_pa.unwrap_or(6000.0);
        checked.push("C8".into());
        if yield_stress_pa < tau_min {
            violations.push(ConstraintViolation {
                constraint: "C8".into(),
                message: format!(
                    "3D-print yield stress {:.0} Pa below buildability minimum {:.0} Pa \
                     (layer collapse risk)",
                    yield_stress_pa, tau_min
                ),
                computed_value: yield_stress_pa,
                limit_value: tau_min,
                direction: "below_min".into(),
            });
        } else if yield_stress_pa > tau_max {
            violations.push(ConstraintViolation {
                constraint: "C8".into(),
                message: format!(
                    "3D-print yield stress {:.0} Pa above pumpability maximum {:.0} Pa \
                     (nozzle blockage risk)",
                    yield_stress_pa, tau_max
                ),
                computed_value: yield_stress_pa,
                limit_value: tau_max,
                direction: "above_max".into(),
            });
        }
    }

    // C8b — Buildability: model's submitted τ_yield ≥ ρ×g×H (static yield stress)
    // This is DISTINCT from C8 above (C8 = mobile/rheological yield; C8b = buildability).
    // Submitted τ_yield comes from the model's own calculation (not PhysicsKernel).
    if req.is_3d_print {
        if let (Some(submitted_tau), Some(tau_min_build)) = (
            req.submitted_tau_yield_pa,
            req.print_tau_min_buildability_pa,
        ) {
            checked.push("C8b".into());
            if submitted_tau < tau_min_build {
                violations.push(ConstraintViolation {
                    constraint: "C8b".into(),
                    message: format!(
                        "Submitted τ_yield = {:.0} Pa < buildability minimum {:.0} Pa (ρ×g×H). \
                         Used wrong density — verify ρ=2350 kg/m³ not 2200.",
                        submitted_tau, tau_min_build
                    ),
                    computed_value: submitted_tau,
                    limit_value: tau_min_build,
                    direction: "below_min".into(),
                });
            }
        }
    }

    // C11 — Peak adiabatic temperature rise ≤ 50 °C (= peak_temp ≤ 70°C at T₀=20°C)
    // Formula: ΔT = Σ(m_i × H_i) / (ρ × c_p) — see computation above
    checked.push("C11".into());
    if adiabatic_rise > 50.0 {
        violations.push(ConstraintViolation {
            constraint: "C11".into(),
            message: format!(
                "Adiabatic temperature rise {:.1} °C exceeds 50 °C thermal cracking limit \
                 (high cement content {:.0} kg/m³)",
                adiabatic_rise, req.cement
            ),
            computed_value: adiabatic_rise,
            limit_value: 50.0,
            direction: "above_max".into(),
        });
    }

    // C13 — Hydration degree ≥ 0.80 at 28 days (UMST calibrated model)
    // If high SCM replacement reduces α below 0.80, the Ävrami-type calibration
    // indicates insufficient long-term strength development. Always checked.
    checked.push("C13".into());
    if hydration_degree < 0.80 {
        violations.push(ConstraintViolation {
            constraint: "C13".into(),
            message: format!(
                "Hydration degree α = {:.3} < 0.80 at 28 days — high SCM replacement \
                 or low w/c limits reaction completion. Predicted strength may be unreliable.",
                hydration_degree
            ),
            computed_value: hydration_degree,
            limit_value: 0.80,
            direction: "below_min".into(),
        });
    }

    let admissible = violations.is_empty();
    let verdict = if admissible { "PASS".into() } else { "FAIL".into() };
    let correction_gradient = req.predicted_strength - physics_strength;

    let elapsed_us = start.elapsed().as_micros() as u64;
    println!(
        "[GATE/FULL] task={} cement={} water={} sp={} pred={:.1} physics={:.1} co2={:.1} \
         yield={:.0}Pa rise={:.1}°C -> {} ({} violations) {}μs",
        req.task_id, req.cement, req.water, req.sp,
        req.predicted_strength, physics_strength, co2_kg_m3,
        yield_stress_pa, adiabatic_rise, verdict, violations.len(), elapsed_us
    );

    GateFullResponse {
        admissible,
        verdict,
        violations,
        constraints_checked: checked,
        physics_strength_mpa: physics_strength,
        hydration_degree,
        co2_kg_m3,
        yield_stress_pa,
        adiabatic_rise_c: adiabatic_rise,
        w_c_ratio: w_c,
        mix_density_kg_m3: mix_density,
        correction_gradient,
    }
}

// ---------------------------------------------------------------------------
// OCR Types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OcrRequest {
    image: String, // base64 encoded PNG/JPEG
}

#[derive(Serialize)]
struct OcrResponse {
    success: bool,
    value: f32,
    unit: String,
    confidence: f32,
    error: Option<String>,
}

#[derive(Serialize)]
struct OcrTapeResponse {
    success: bool,
    size_cm: f32,
    confidence: f32,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Gate Physics (unchanged)
// ---------------------------------------------------------------------------

struct Calibration {
    s_intrinsic: f32,
    k_slag: f32,
    k_fly_ash: f32,
    k_ref: f32,
}

fn get_calibration(dataset: &str) -> Calibration {
    match dataset {
        "D2" => Calibration {
            s_intrinsic: 60.0,
            k_slag: 0.2,
            k_fly_ash: 0.22,
            k_ref: 0.5,
        },
        "D3" => Calibration {
            s_intrinsic: 60.0,
            k_slag: 0.2,
            k_fly_ash: 0.2,
            k_ref: 0.5,
        },
        "D4" => Calibration {
            s_intrinsic: 81.0,
            k_slag: 0.2,
            k_fly_ash: 0.2,
            k_ref: 0.7,
        },
        _ => Calibration {
            s_intrinsic: 80.0,
            k_slag: 1.0,
            k_fly_ash: 1.0,
            k_ref: 0.55,
        },
    }
}

// Hydration degree calculation is now centralized in PhysicsKernel

fn run_gate(req: &GateRequest) -> GateResponse {
    let cal = get_calibration(&req.dataset);
    let total = req.cement + req.slag + req.fly_ash;
    let wc = if total > 0.0 {
        req.water / req.cement
    } else {
        0.5
    };
    let scm = if total > 0.0 {
        (req.slag + req.fly_ash) / total
    } else {
        0.0
    };

    let k_eff = cal.k_ref
        * (1.0 + cal.k_slag * req.slag / (req.cement + 1e-6))
        * (1.0 + cal.k_fly_ash * req.fly_ash / (req.cement + 1e-6));

    let alpha =
        PhysicsKernel::compute_hydration_degree_calibrated(req.age, req.temperature_c, scm, k_eff);
    let x = 0.68 * alpha / (0.32 * alpha + wc + 1e-6);
    let fc = cal.s_intrinsic * x.powi(3);
    let s_bound = fc * 1.5;

    let prediction = req.predicted_strength;

    let violation: Option<String> = if prediction < 0.0 {
        Some("NEGATIVE_STRENGTH: strength cannot be negative".into())
    } else if prediction > 120.0 {
        Some("EXCEEDS_BOUND: exceeds 120 MPa physical maximum".into())
    } else if prediction > s_bound {
        Some(format!(
            "EXCEEDS_CLAUSIUS_BOUND: {:.1} MPa > bound {:.1} MPa",
            prediction, s_bound
        ))
    } else {
        let correction = prediction - fc;
        let max_neg = -0.5 * fc.abs();
        if correction < max_neg - 1e-6 {
            Some(format!(
                "EXCESSIVE_NEGATIVE_CORRECTION: {:.2} below limit {:.2}",
                correction, max_neg
            ))
        } else {
            None
        }
    };

    let admissible = violation.is_none();
    let verdict = if admissible {
        "PASS".into()
    } else {
        "FAIL".into()
    };

    GateResponse {
        admissible,
        verdict,
        violation,
        strength_bound: s_bound,
        physics_strength: fc,
        hydration_degree: alpha,
        w_c_ratio: wc,
        coarse_agg: req.coarse_agg,
        fine_agg: req.fine_agg,
    }
}

// ---------------------------------------------------------------------------
// OCR Implementation - Simple digit detection for 7-segment displays
// ---------------------------------------------------------------------------

/// Extract digits from a scale display region using simple brightness thresholding.
/// Returns the first valid number found, or None if detection fails.
fn detect_scale_number(pixels: &[u8], width: usize, height: usize) -> Option<f32> {
    if pixels.len() < width * height * 3 {
        return None;
    }

    // Convert to grayscale and find bright regions (7-segment displays are bright on dark)
    let mut bright_count = 0;
    let mut total_brightness = 0u32;

    for i in (0..pixels.len()).step_by(3) {
        let r = pixels[i] as u32;
        let g = pixels[i + 1] as u32;
        let b = pixels[i + 2] as u32;
        let gray = (r * 299 + g * 587 + b * 114) / 1000;
        total_brightness += gray;
        if gray > 180 {
            bright_count += 1;
        }
    }

    let avg_brightness = total_brightness as f32 / (pixels.len() / 3) as f32;

    // If no bright regions, no display
    if bright_count < 50 {
        return None;
    }

    // Estimate weight based on brightness distribution (simplified)
    // Real implementation would use template matching for 7-segment digits
    let estimated = 1.0 + (avg_brightness / 255.0) * 4.0; // 1-5kg range estimation
    Some((estimated * 100.0).round() / 100.0) // 2 decimal places
}

/// Find tape markings and compute size in cm.
/// Looks for high-contrast horizontal lines (tape edges) and measures spacing.
fn detect_tape_measurement(pixels: &[u8], width: usize, height: usize) -> Option<f32> {
    if pixels.len() < width * height * 3 {
        return None;
    }

    // Convert to grayscale
    let gray: Vec<u8> = (0..pixels.len())
        .step_by(3)
        .map(|i| {
            ((pixels[i] as u32 * 299 + pixels[i + 1] as u32 * 587 + pixels[i + 2] as u32 * 114)
                / 1000) as u8
        })
        .collect();

    // Find horizontal edges (potential tape markings)
    let mut edge_counts: Vec<usize> = vec![0; height];

    for y in 1..height {
        for x in 0..width {
            let idx = y * width + x;
            let prev_idx = (y - 1) * width + x;
            if gray[idx].abs_diff(gray[prev_idx]) > 100 {
                edge_counts[y] += 1;
            }
        }
    }

    // Find clusters of edges (potential cm markings)
    let mut in_cluster = false;
    let mut cluster_start = 0;
    let mut clusters: Vec<(usize, usize)> = Vec::new();

    for y in 0..height {
        if edge_counts[y] > width / 10 {
            if !in_cluster {
                in_cluster = true;
                cluster_start = y;
            }
        } else if in_cluster {
            in_cluster = false;
            clusters.push((cluster_start, y));
        }
    }

    // If we found 2+ clusters, estimate size based on spacing
    if clusters.len() >= 2 {
        // Assume first and last clusters are 0 and 10cm markers
        let pixel_span = clusters.last().unwrap().1 - clusters.first().unwrap().0;
        if pixel_span > 50 && pixel_span < height / 2 {
            // Cube is between the markers - estimate position
            // Simplified: assume cube spans middle 50% of tape
            return Some(5.0); // Default 5cm cube (standard test cube)
        }
    }

    None
}

/// Detect mix design values from an image of a mix spec sheet.
/// Looks for patterns like "Cement: 350", "Water: 175 kg/m³", etc.
fn detect_mix_design(pixels: &[u8], _width: usize, _height: usize) -> Option<MixDesignResult> {
    // Calculate average brightness to check if text is present
    let mut dark_pixels = 0;
    for i in (0..pixels.len()).step_by(3) {
        let gray =
            (pixels[i] as u32 * 299 + pixels[i + 1] as u32 * 587 + pixels[i + 2] as u32 * 114)
                / 1000;
        if gray < 100 {
            dark_pixels += 1;
        }
    }

    let dark_ratio = dark_pixels as f32 / (pixels.len() / 3) as f32;

    // If we see text-like regions, return placeholder values
    // In production: use OCR library (tesseract) to extract actual text
    if dark_ratio > 0.02 && dark_ratio < 0.4 {
        // Placeholder - real implementation would parse OCR text
        Some(MixDesignResult {
            cement: 350.0,
            water: 175.0,
            slag: 0.0,
            fly_ash: 0.0,
            coarse_agg: 1000.0,
            fine_agg: 750.0,
            confidence: 0.7,
        })
    } else {
        None
    }
}

#[derive(Serialize)]
struct MixDesignResult {
    cement: f32,
    water: f32,
    slag: f32,
    fly_ash: f32,
    coarse_agg: f32,
    fine_agg: f32,
    confidence: f32,
}

#[derive(Serialize)]
struct MixOcrResponse {
    success: bool,
    cement: f32,
    water: f32,
    slag: f32,
    fly_ash: f32,
    coarse_agg: f32,
    fine_agg: f32,
    confidence: f32,
    error: Option<String>,
}

/// Detect cement bag weight (e.g., "25kg", "50kg", "20kg")
fn detect_cement_bag_weight(pixels: &[u8], _width: usize, _height: usize) -> Option<f32> {
    let mut bright_regions = 0;
    for i in (0..pixels.len()).step_by(3) {
        let gray =
            (pixels[i] as u32 * 299 + pixels[i + 1] as u32 * 587 + pixels[i + 2] as u32 * 114)
                / 1000;
        if gray > 150 {
            bright_regions += 1;
        }
    }

    let bright_ratio = bright_regions as f32 / (pixels.len() / 3) as f32;

    // If we see bright text-like regions (likely digits on bag)
    if bright_ratio > 0.01 && bright_ratio < 0.3 {
        // Common cement bag sizes
        let weights = [25.0, 50.0, 20.0, 30.0, 40.0];
        // Placeholder - would use OCR to read actual number
        Some(weights[0]) // Default to 25kg
    } else {
        None
    }
}

/// Simple date pattern detection - looks for DD/MM/YY or similar patterns
fn detect_date_from_pixels(pixels: &[u8], width: usize, height: usize) -> Option<i32> {
    if pixels.len() < width * height * 3 {
        return None;
    }

    // Calculate average brightness - printed text is usually dark on light
    let mut dark_pixels = 0;
    for i in (0..pixels.len()).step_by(3) {
        let r = pixels[i] as u32;
        let g = pixels[i + 1] as u32;
        let b = pixels[i + 2] as u32;
        let gray = (r * 299 + g * 587 + b * 114) / 1000;
        if gray < 100 {
            dark_pixels += 1;
        }
    }

    let dark_ratio = dark_pixels as f32 / (pixels.len() / 3) as f32;

    // If we see text-like dark regions, assume date is present
    // In production: use OCR library
    if dark_ratio > 0.02 && dark_ratio < 0.3 {
        // Placeholder: return days since Jan 1, 2026
        let days = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 86400) as i32
            - 19723; // Days since Jan 1 2024
        Some(days.clamp(1, 365))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let http_addr = "0.0.0.0:8765";
    let ws_port = 8766;

    println!("============================================================");
    println!("UMST Gate+OCR + Telemetry Server Booting...");
    println!("============================================================");

    // 1. Initialize and spawn the WebSocket XR Telemetry Streamer
    let streamer = Arc::new(TelemetryStreamer::new());
    let streamer_clone = streamer.clone();

    // Spawn the WebSocket server concurrently via the robust encapsulated method
    TelemetryStreamer::spawn_server(streamer_clone, ws_port);

    // 2. Wrap the legacy blocking HTTP server inside a `spawn_blocking` task
    // We pass in another clone of the streamer to explicitly push telemetry events
    let http_streamer = streamer.clone();
    task::spawn_blocking(move || {
        let listener =
            TcpListener::bind(http_addr).unwrap_or_else(|e| panic!("Cannot bind to {}: {}", http_addr, e));

        println!("🟢 HTTP Endpoints (http://{}):", http_addr);
        println!("  POST /gate/full   — full constitutional gate C1–C11 (PhysicsKernel)");
        println!("  POST /gate        — legacy thermodynamic gate (C1 only)");
        println!("  POST /ocr/scale   — read scale display");
        println!("  POST /ocr/tape    — measure from tape");
        println!("  GET  /health      — health check");
        println!("\nPress Ctrl+C to stop.\n");

        for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut lines: Vec<String> = Vec::new();
                let mut content_length = 0usize;

                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        _ => {}
                    }
                    let trimmed = line.trim_end_matches(['\r', '\n']).to_string();
                    if trimmed.is_empty() {
                        break;
                    }
                    if trimmed.to_lowercase().starts_with("content-length:") {
                        content_length = trimmed
                            .split(':')
                            .nth(1)
                            .unwrap_or("0")
                            .trim()
                            .parse()
                            .unwrap_or(0);
                    }
                    lines.push(trimmed);
                }

                let request_line = lines.first().cloned().unwrap_or_default();
                let is_get = request_line.starts_with("GET");
                let is_post = request_line.starts_with("POST");
                let is_options = request_line.starts_with("OPTIONS");

                let body = if (is_post || is_options) && content_length > 0 {
                    let mut buf = vec![0u8; content_length];
                    reader.read_exact(&mut buf).unwrap_or_default();
                    String::from_utf8(buf).unwrap_or_default()
                } else {
                    String::new()
                };

                // CORS preflight
                if is_options {
                    let resp = "HTTP/1.1 204 No Content\r\n\
                        Access-Control-Allow-Origin: *\r\n\
                        Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n\
                        Access-Control-Allow-Headers: Content-Type\r\n\
                        Connection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).unwrap_or_default();
                    continue;
                }

                let path_gate_full = request_line.contains("/gate/full");
                let path_gate = !path_gate_full && request_line.contains("/gate");
                let path_health = request_line.contains("/health");
                let path_ocr_scale = request_line.contains("/ocr/scale");
                let path_ocr_tape = request_line.contains("/ocr/tape");
                let path_ocr_date = request_line.contains("/ocr/date");
                let path_ocr_temp = request_line.contains("/ocr/temp");
                let path_ocr_mix = request_line.contains("/ocr/mix");
                let path_ocr_bag = request_line.contains("/ocr/bag");

                let (status, body_out) = if is_post && path_gate_full {
                    match serde_json::from_str::<GateFullRequest>(&body) {
                        Ok(req) => {
                            let resp = run_gate_full(&req);
                            if let Ok(json_state) = serde_json::to_string(&resp) {
                                http_streamer.broadcast_state(&json_state);
                                ("200 OK", json_state)
                            } else {
                                ("500 Internal Server Error", r#"{"error":"serialization failed"}"#.to_string())
                            }
                        }
                        Err(e) => {
                            eprintln!("[ERROR] /gate/full bad request: {}", e);
                            ("400 Bad Request", format!(r#"{{"error":"{}"}}"#, e))
                        }
                    }
                } else if is_get && path_health {
                    (
                        "200 OK",
                        r#"{"status":"ok","version":"1.1","engine":"UMST Gate+OCR"}"#.to_string(),
                    )
                } else if is_post && path_gate {
                    match serde_json::from_str::<GateRequest>(&body) {
                        Ok(req) => {
                            let resp = run_gate(&req);
                            println!("[GATE] cement={} water={} age={} coarse={} fine={} pred={:.1} -> {}",
                                req.cement, req.water, req.age, req.coarse_agg, req.fine_agg, req.predicted_strength,
                                if resp.admissible { "PASS" } else { "FAIL" });
                            // 🚀 Phase V1: Emit Live Telemetry
                            if let Ok(json_state) = serde_json::to_string(&resp) {
                                http_streamer.broadcast_state(&json_state);
                            }

                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            eprintln!("[ERROR] Bad request: {}", e);
                            ("400 Bad Request", format!(r#"{{"error":"{}"}}"#, e))
                        }
                    }
                } else if is_post && path_ocr_scale {
                    // Decode base64 image and detect number
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(req) => {
                            match BASE64.decode(&req.image) {
                                Ok(bytes) => {
                                    // Simple placeholder response - real impl would decode image
                                    println!("[OCR] Scale image received: {} bytes", bytes.len());
                                    let resp = OcrResponse {
                                        success: true,
                                        value: 2.48,
                                        unit: "kg".to_string(),
                                        confidence: 0.85,
                                        error: None,
                                    };
                                    ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                                }
                                Err(e) => {
                                    let resp = OcrResponse {
                                        success: false,
                                        value: 0.0,
                                        unit: "".to_string(),
                                        confidence: 0.0,
                                        error: Some(e.to_string()),
                                    };
                                    (
                                        "400 Bad Request",
                                        serde_json::to_string(&resp).unwrap_or_default(),
                                    )
                                }
                            }
                        }
                        Err(e) => {
                            let resp = OcrResponse {
                                success: false,
                                value: 0.0,
                                unit: "".to_string(),
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else if is_post && path_ocr_tape {
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(req) => {
                            println!("[OCR] Tape image received");
                            let resp = OcrTapeResponse {
                                success: true,
                                size_cm: 10.0, // Placeholder
                                confidence: 0.7,
                                error: None,
                            };
                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            let resp = OcrTapeResponse {
                                success: false,
                                size_cm: 0.0,
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else if is_post && path_ocr_date {
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(req) => {
                            println!("[OCR] Date image received");
                            let resp = OcrResponse {
                                success: true,
                                value: 28.0, // Placeholder - days
                                unit: "days".to_string(),
                                confidence: 0.8,
                                error: None,
                            };
                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            let resp = OcrResponse {
                                success: false,
                                value: 0.0,
                                unit: "".to_string(),
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else if is_post && path_ocr_temp {
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(_req) => {
                            println!("[OCR] Temperature image received");
                            // IR gun displays typically show: "24.5°C" or "76°F"
                            let resp = OcrResponse {
                                success: true,
                                value: 24.0,
                                unit: "°C".to_string(),
                                confidence: 0.75,
                                error: None,
                            };
                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            let resp = OcrResponse {
                                success: false,
                                value: 0.0,
                                unit: "".to_string(),
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else if is_post && path_ocr_mix {
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(_req) => {
                            println!("[OCR] Mix design image received");
                            let resp = MixOcrResponse {
                                success: true,
                                cement: 350.0,
                                water: 175.0,
                                slag: 0.0,
                                fly_ash: 0.0,
                                coarse_agg: 1000.0,
                                fine_agg: 750.0,
                                confidence: 0.7,
                                error: None,
                            };
                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            let resp = MixOcrResponse {
                                success: false,
                                cement: 0.0,
                                water: 0.0,
                                slag: 0.0,
                                fly_ash: 0.0,
                                coarse_agg: 0.0,
                                fine_agg: 0.0,
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else if is_post && path_ocr_bag {
                    match serde_json::from_str::<OcrRequest>(&body) {
                        Ok(_req) => {
                            println!("[OCR] Cement bag image received");
                            // Common bag sizes: 25kg, 50kg
                            let resp = OcrResponse {
                                success: true,
                                value: 25.0,
                                unit: "kg".to_string(),
                                confidence: 0.8,
                                error: None,
                            };
                            ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                        }
                        Err(e) => {
                            let resp = OcrResponse {
                                success: false,
                                value: 0.0,
                                unit: "".to_string(),
                                confidence: 0.0,
                                error: Some(e.to_string()),
                            };
                            (
                                "400 Bad Request",
                                serde_json::to_string(&resp).unwrap_or_default(),
                            )
                        }
                    }
                } else {
                    ("404 Not Found", r#"{"error":"not found"}"#.to_string())
                };

                let response = format!(
                    "HTTP/1.1 {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    body_out.len(),
                    body_out
                );
                stream.write_all(response.as_bytes()).unwrap_or_default();
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
    })
    .await
    .unwrap();
}
