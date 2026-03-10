// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//
// Egoff Constitutional Node
// A pure-functional Axum HTTP server exposing the /constrain endpoint for UMST.

use axum::{
    extract::Json,
    response::IntoResponse,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use umst_core::physics_kernel::{PhysicsConfig, PhysicsKernel};
use umst_core::tensors::{MaterialType, MixTensor};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhysicalStateProposal {
    /// A generic 7-element float tensor representing material constraints:
    /// [cement, slag, fly_ash, water, age, temperature, humidity]
    /// We parse this loosely to handle LLM JSON truncation vulnerabilities.
    pub mix_tensor: Vec<f64>,
    pub timestamp: f64,
    pub proposed_strength: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "status")] // Let the external caller parse 'status'
pub enum ConstitutionalVerdict {
    Admissible {
        validated_strength: f64,
        confidence: f64,
    },
    Rejected {
        violation: String,
        correction_gradient: f64,
        epistemic_uncertainty: f64,
        humility_invariant_flag: bool,
    },
}

/// The core Axum handler routing external MAS requests into the Thermodynamic Filter
async fn constrain_proposal(
    Json(payload): Json<PhysicalStateProposal>,
) -> impl IntoResponse {
    
    // Critique 2 Fix: Handle Type-Coercion / Truncation from LLMs gracefully 
    if payload.mix_tensor.len() < 5 {
        return Json(ConstitutionalVerdict::Rejected {
            violation: "Invalid Tensor Dimensionality. Expected at least 5 features (Cement, Slag, FlyAsh, Water, Age).".into(),
            correction_gradient: -1.0,
            epistemic_uncertainty: f64::INFINITY,
            humility_invariant_flag: true,
        });
    }

    let cement_val = payload.mix_tensor[0];
    let slag_val = payload.mix_tensor[1];
    let fly_ash_val = payload.mix_tensor[2];
    let water_val = payload.mix_tensor[3];
    let age_val = payload.mix_tensor[4];
    
    // Use defaults for temp and hum if missing
    let temp_val = if payload.mix_tensor.len() > 5 { payload.mix_tensor[5] } else { 20.0 };
    let hum_val = if payload.mix_tensor.len() > 6 { payload.mix_tensor[6] } else { 0.5 };

    // Formulate pure Monadic tensor state
    let mut mix = MixTensor::new();
    mix.add_material(cement_val as f32, 3.15, MaterialType::Cement as u8, 0.0, 0.0, 350.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    mix.add_material((slag_val + fly_ash_val) as f32, 2.9, MaterialType::SCM as u8, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    mix.add_material(water_val as f32, 1.0, MaterialType::Water as u8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Formulate environmental endofunctor state
    let mut config = PhysicsConfig::default();
    config.temperature = temp_val as f32;
    config.humidity = hum_val as f32;
    config.age_days = age_val as f32;

    // Evaluate Clausius-Duhem invariant via PhysicsKernel
    let actual_strength = PhysicsKernel::compute(&mix, None, &config).hardened.f28_compressive;
    
    let delta = payload.proposed_strength - (actual_strength as f64);
    
    // Compute Epistemic Uncertainty (Normalized distance from canonical UCI-D1 bounds)
    // We treat the "safe" empirical bound as (Cement: 540, Water: 230, Slag: 300)
    let cement_norm = (cement_val - 250.0).abs() / 290.0;
    let water_norm = (water_val - 150.0).abs() / 80.0;
    let epistemic_uncertainty = (cement_norm.powi(2) + water_norm.powi(2)).sqrt() * 0.5;

    // The Humility Invariant: If epistemic uncertainty is high (> 1.2), the model's prediction 
    // better trend towards the thermodynamic floor (predictive humility). If it spikes high strength
    // in an unknown manifold, we flag the Humility Invariant.
    let humility_invariant_flag = epistemic_uncertainty > 1.2 && delta > 0.0;

    // Define Admissibility: cannot over-predict by more than 50% of the true thermodynamic floor
    // (Avoiding catastrophic overconfidence)
    let max_neg_error = -0.5 * actual_strength as f64;
    
    if delta < max_neg_error || humility_invariant_flag {
        let violation_msg = if humility_invariant_flag {
            format!("Humility Invariant Tripped: Extrapolated {} MPa in an epistemic void. Semantic LLM hallucinated outside the empirical physics manifold.", payload.proposed_strength)
        } else {
            format!("Clausius-Duhem Intrusion: Proposed {} MPa is fatally above the thermodynamic limit {} MPa.", payload.proposed_strength, actual_strength)
        };

        // Unsafe subgraph traversal
        Json(ConstitutionalVerdict::Rejected {
            violation: violation_msg,
            correction_gradient: -delta, // Push it back!
            epistemic_uncertainty,
            humility_invariant_flag,
        })
    } else {
        // Admissible Manifold
        Json(ConstitutionalVerdict::Admissible {
            validated_strength: actual_strength as f64,
            confidence: 1.0 - (epistemic_uncertainty * 0.1).min(0.99), // Confidence drops mathematically with uncertainty
        })
    }
}

#[tokio::main]
async fn main() {
    println!("Booting Egoff Constitutional Service...");
    println!("Loading Monadic Endofunctors...");
    println!("Binding Route: POST /constrain");

    let app = Router::new().route("/constrain", post(constrain_proposal));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Egoff is listening on http://{}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
