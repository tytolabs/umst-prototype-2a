// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//
// Egoff CLI Tool (Path B)
// Connects Cursor's interactive IDE layer to the native Rust Constitutional Gate.
// Replaces the NodeJS version for maximum Rust purity.

use serde::{Deserialize, Serialize};
use std::env;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhysicalStateProposal {
    pub mix_tensor: Vec<f64>,
    pub timestamp: f64,
    pub proposed_strength: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "status")]
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: egoff_cli <mix_tensor_json_array> <proposed_strength>");
        eprintln!("Example: egoff_cli \"[350, 50, 0, 150, 28]\" 45.0");
        std::process::exit(1);
    }

    let mix_tensor_json = &args[1];
    let proposed_strength_str = &args[2];

    let mix_tensor: Vec<f64> = match serde_json::from_str(mix_tensor_json) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to parse mix_tensor: {}", e);
            std::process::exit(1);
        }
    };

    let proposed_strength: f64 = match proposed_strength_str.parse() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to parse proposed_strength: {}", e);
            std::process::exit(1);
        }
    };

    let proposal = PhysicalStateProposal {
        mix_tensor,
        timestamp: 0.0,
        proposed_strength,
    };

    let client = reqwest::Client::new();
    let res = client.post("http://127.0.0.1:3000/constrain")
        .json(&proposal)
        .send()
        .await?;

    if !res.status().is_success() {
        eprintln!("Egoff Server Error: {}", res.status());
        std::process::exit(1);
    }

    let verdict: ConstitutionalVerdict = res.json().await?;

    match verdict {
        ConstitutionalVerdict::Admissible { validated_strength, confidence: _ } => {
            println!("[EGOFF ADMISSIBLE]: The proposed state fits within physical bounds. Thermodynamic Floor: {} MPa", validated_strength);
        },
        ConstitutionalVerdict::Rejected { violation, correction_gradient, epistemic_uncertainty: _, humility_invariant_flag: _ } => {
            println!("[EGOFF REJECTED]: {}\nCorrection Gradient Provided: {}", violation, correction_gradient);
        }
    }

    Ok(())
}
