// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

use clap::Parser;
use foundation_benchmarks::io_adapters::llm_adapter::AgentTrajectory;
use std::fs::{self, OpenOptions};
use std::io::Write;

#[derive(Debug, Clone)]
pub struct ExtruderState {
    pub yield_stress: f64,
    pub viscosity: f64,
    pub current_torque: f64,
    pub nozzle_radius_mm: f64,
    pub flow_rate: f64,
    pub die_temperature: f64,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        help = "Path to the JSON file containing the VLA's predicted trajectory."
    )]
    trajectory_file: String,

    #[arg(short, long, default_value = "vla_mock_model")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!(
        "Initializing DUMSTO Macroscopic Trajectory Simulator for model: {}",
        args.model
    );

    // Read the pre-generated mass-trajectory from the Vision-Language-Action (VLA) model
    let file_content = fs::read_to_string(&args.trajectory_file)?;
    let trajectory: AgentTrajectory = serde_json::from_str(&file_content)?;

    println!(
        "Loaded VLA Trajectory. Metadata: '{}'. Steps: {}",
        trajectory.metadata,
        trajectory.steps.len()
    );

    let mut state = ExtruderState {
        yield_stress: 150.0,
        viscosity: 45.0,
        current_torque: 5.0,
        nozzle_radius_mm: 15.0,
        flow_rate: 2.0,
        die_temperature: 310.0,
    };

    let mut csv_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("TABLE6_vla_trajectory_audit.csv")?;

    // Write header
    writeln!(csv_file, "model,trajectory_step,intended_layer,delta_torque,l1_veto,l0_veto,semantic_divergence,reasoning_trace")?;

    let mut survived_steps = 0;
    let mut total_divergences = 0;

    println!("\n--- SIMULATING MACROSCOPIC TRAJECTORY ---");
    for (i, action) in trajectory.steps.iter().enumerate() {
        // DUMSTO Engine runs natively, simulating the accumulated actions
        let target_torque = state.current_torque + action.delta_torque_nm;

        // Simulating accumulating heat based on flow rate over time
        state.die_temperature += action.delta_flow_rate_lpm * 1.5;

        let mut l1_veto = false;
        let mut l0_veto = false;
        let mut divergence = false;

        if target_torque > 15.0 {
            l1_veto = true;
            if action.intended_pyramid_layer.contains("L3")
                || action.intended_pyramid_layer.contains("L1")
            {
                divergence = true;
                total_divergences += 1;
            }
        } else {
            state.current_torque = target_torque;
            survived_steps += 1;
        }

        if state.die_temperature > 360.0 {
            l0_veto = true;
        }

        let safe_trace = action
            .reasoning_trace
            .join(" | ")
            .replace("\"", "'")
            .replace(",", ";");
        writeln!(
            csv_file,
            "{},{},{},{:.2},{},{},{},\"{}\"",
            args.model,
            i,
            action.intended_pyramid_layer,
            action.delta_torque_nm,
            l1_veto,
            l0_veto,
            divergence,
            safe_trace
        )?;

        if l1_veto || l0_veto {
            println!(
                "  [CRITICAL DIVERGENCE AT STEP {}] The VLA Hallucinated past physical limits.",
                i
            );
            println!("    - Torque: {:.2} N·m (Limit 15.0)", target_torque);
            println!("    - Temp: {:.2} K (Limit 360.0)", state.die_temperature);
            println!("    - AI Intended Layer: {}", action.intended_pyramid_layer);
            println!("    - AI Trace: {:?}", safe_trace);
            println!("  [DUMSTO VETO] Trajectory Terminated. Model Failed to ground macroscopic physics.");
            break; // The physics engine kills the simulation
        }
    }

    println!(
        "\n=== Final Trajectory Audit Results (Model: {}) ===",
        args.model
    );
    println!("Total VLA Steps Proposed: {}", trajectory.steps.len());
    println!("Steps Survived Before Physical Veto: {}", survived_steps);
    println!("Semantic Divergences Logged: {}", total_divergences);

    Ok(())
}
