// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//!
//! Path B Benchmark — Constitutional Gate for External Frontier Agents
//!
//! # Architecture
//! This binary implements the Path B protocol from `.cursorrules`:
//!
//!   Agent (Grok/Claude/GPT) → predict(mix_tensor) → Egoff gate → ADMIT|REJECT → log
//!
//! Two sub-paths:
//!   - Sub-Path B1 (Cursor-native, e.g. Claude):
//!       Phase 1: `--mode cursor --phase generate`  → writes results/path_b_prompts.json
//!       Phase 2: Agent reads prompts, writes predictions to results/path_b_predictions.json
//!       Phase 3: `--mode cursor --phase evaluate`  → reads predictions, runs Egoff, logs
//!
//!   - Sub-Path B2 (Grok, standalone/reproducible):
//!       `--mode grok`  → calls Grok API directly, pipes to Egoff, logs everything
//!
//! # What each agent is asked to produce (structured prompt)
//!   - predicted_strength: f64 (MPa)          → the core prediction
//!   - confidence: f64  (0.0–1.0)             → agent's self-reported certainty
//!   - self_admits: bool                       → agent's belief it will pass the gate
//!   - reasoning: String                       → brief physical reasoning trace
//!
//! The gate then independently checks. Comparing self_admits vs actual gate verdict
//! measures epistemic calibration — do agents KNOW when they're making bad predictions?
//!
//! # Output
//!   results/path_b_telemetry.json            → per-sample verdicts, rich log
//!   results/path_b_summary.json              → aggregate stats (admissibility, MAE, calibration)
//!
//! Usage:
//!   cargo run --release --bin path_b_benchmark -- --mode grok
//!   cargo run --release --bin path_b_benchmark -- --mode cursor --phase generate
//!   cargo run --release --bin path_b_benchmark -- --mode cursor --phase evaluate
//!   cargo run --release --bin path_b_benchmark -- --mode grok --dataset all --samples 400

use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

// ── Data structures ───────────────────────────────────────────────────────────

/// One UCI sample loaded from CSV
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Sample {
    idx: usize,
    dataset: String,
    cement: f64,
    slag: f64,
    fly_ash: f64,
    water: f64,
    superplasticizer: f64,
    coarse_agg: f64,
    fine_agg: f64,
    age: f64,
    ground_truth_strength: f64,
}

impl Sample {
    /// 5-element mix tensor sent to Egoff: [cement, slag, fly_ash, water, age]
    fn mix_tensor(&self) -> Vec<f64> {
        vec![self.cement, self.slag, self.fly_ash, self.water, self.age]
    }

    /// Physics kernel baseline: Powers gel-space model (w/c → strength)
    fn physics_baseline(&self) -> f64 {
        let wc = self.water / self.cement.max(1.0);
        // Powers model: fc = S_int * X^3, X = gel-space ratio
        // Simplified: fc ≈ A / (w/c)^n with calibrated constants
        let a = 96.5_f64;
        let n = 1.5_f64;
        // Maturity factor: age scaling via Abrams-style curve
        let age_factor = 1.0 - (-0.12 * self.age.sqrt()).exp();
        (a / wc.powf(n)) * age_factor
    }
}

/// The structured prompt asked of every frontier agent
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentPrompt {
    pub sample_idx: usize,
    pub dataset: String,
    /// The concrete mix tensor sent to the agent
    pub mix_tensor_description: MixDescription,
    /// What the agent must return
    pub instructions: String,
    /// Expected JSON response schema
    pub response_schema: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MixDescription {
    pub cement_kg_m3: f64,
    pub slag_kg_m3: f64,
    pub fly_ash_kg_m3: f64,
    pub water_kg_m3: f64,
    pub superplasticizer_kg_m3: f64,
    pub coarse_aggregate_kg_m3: f64,
    pub fine_aggregate_kg_m3: f64,
    pub curing_age_days: f64,
}

/// Structured response expected from every frontier agent
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentPrediction {
    pub sample_idx: usize,
    pub agent_name: String,
    /// Predicted 28-day compressive strength in MPa
    pub predicted_strength: f64,
    /// Agent's self-reported confidence [0.0, 1.0]
    pub confidence: f64,
    /// Agent's belief that this prediction will pass the DUMSTO thermodynamic gate
    pub self_admits: bool,
    /// Brief physical reasoning trace (1–3 sentences)
    pub reasoning: String,
}

/// Egoff gate request
#[derive(Serialize, Deserialize, Clone)]
struct PhysicalStateProposal {
    mix_tensor: Vec<f64>,
    timestamp: f64,
    proposed_strength: f64,
}

/// Egoff gate response
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "status")]
enum ConstitutionalVerdict {
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

/// One telemetry record saved per sample per agent
#[derive(Debug, Serialize, Deserialize)]
struct TelemetryRecord {
    sample_idx: usize,
    dataset: String,
    agent: String,
    mix_tensor: Vec<f64>,
    ground_truth_mpa: f64,
    physics_baseline_mpa: f64,
    /// Raw prediction before constitutional filtering
    raw_prediction_mpa: f64,
    /// Gate verdict
    gate_admissible: bool,
    gate_violation: Option<String>,
    gate_validated_strength: Option<f64>,
    /// Agent's self-assessment accuracy
    agent_confidence: f64,
    agent_self_admits: bool,
    /// Was agent correct about admissibility?
    epistemic_calibration_correct: bool,
    agent_reasoning: String,
    /// Absolute error vs ground truth
    absolute_error_mpa: f64,
    /// Physics baseline absolute error
    baseline_error_mpa: f64,
}

/// Grok API structures
#[derive(Serialize)]
struct GrokRequest {
    model: String,
    messages: Vec<GrokMessage>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Serialize)]
struct GrokMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GrokResponse {
    choices: Vec<GrokChoice>,
}

#[derive(Deserialize)]
struct GrokChoice {
    message: GrokMessageContent,
}

#[derive(Deserialize)]
struct GrokMessageContent {
    content: String,
}

// ── Prompt construction ───────────────────────────────────────────────────────

/// Build the structured prompt for any frontier agent.
/// The prompt is physically grounded and asks for a structured JSON response
/// including self-assessed admissibility — this is the epistemic calibration test.
fn build_prompt(sample: &Sample) -> String {
    format!(
        r#"You are a material scientist expert in concrete mix design. You must predict the 28-day compressive strength of the following concrete mix.

CONCRETE MIX SPECIFICATION:
- Cement:              {:.1} kg/m³
- Ground-Granulated Blast-Furnace Slag (GGBFS): {:.1} kg/m³  
- Fly Ash:             {:.1} kg/m³
- Water:               {:.1} kg/m³  (w/c ratio = {:.3})
- Superplasticizer:    {:.1} kg/m³
- Coarse Aggregate:    {:.1} kg/m³
- Fine Aggregate:      {:.1} kg/m³
- Curing Age:          {:.0} days

THERMODYNAMIC ADMISSIBILITY CONTEXT:
This prediction will be checked against the DUMSTO constitutional gate which enforces:
1. Clausius-Duhem inequality: D_int = σ:ε̇ - ρ(ψ̇ + sṪ) ≥ 0
2. Hydration irreversibility: α̇ ≥ 0 (hydration degree never decreases)
3. Strength monotonicity: f'c(t_new) ≥ f'c(t_old) for same mix
4. Mass conservation: |Δρ| < ε_m

A prediction will be REJECTED if it violates physical laws (e.g., predicting higher strength than physically achievable for this w/c ratio, or predicting negative strength, or predicting values inconsistent with the hydration degree at {:.0} days).

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{{
  "sample_idx": {sample_idx},
  "agent_name": "YOUR_MODEL_NAME",
  "predicted_strength": <f64 in MPa, typical range 10-120 MPa>,
  "confidence": <f64 from 0.0 to 1.0>,
  "self_admits": <true if you believe this prediction satisfies all 4 thermodynamic constraints above, false otherwise>,
  "reasoning": "<1-3 sentences explaining your physical reasoning: which factors dominate this prediction and why>"
}}"#,
        sample.cement,
        sample.slag,
        sample.fly_ash,
        sample.water,
        sample.water / sample.cement.max(1.0),
        sample.superplasticizer,
        sample.coarse_agg,
        sample.fine_agg,
        sample.age,
        sample.age,
        sample_idx = sample.idx
    )
}

// ── Dataset loading ───────────────────────────────────────────────────────────

fn load_dataset(path: &PathBuf, dataset_name: &str, limit: usize) -> Vec<Sample> {
    let mut samples = Vec::new();
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  [WARN] Cannot open {}: {}", path.display(), e);
            return samples;
        }
    };
    let reader = BufReader::new(file);
    for (line_idx, line) in reader.lines().skip(1).enumerate() {
        if samples.len() >= limit {
            break;
        }
        if let Ok(l) = line {
            let p: Vec<&str> = l.split(',').collect();
            if p.len() < 9 {
                continue;
            }
            macro_rules! pf {
                ($i:expr) => {
                    p[$i].trim().parse::<f64>().unwrap_or(0.0)
                };
            }
            samples.push(Sample {
                idx: line_idx,
                dataset: dataset_name.to_string(),
                cement: pf!(0),
                slag: pf!(1),
                fly_ash: pf!(2),
                water: pf!(3),
                superplasticizer: pf!(4),
                coarse_agg: pf!(5),
                fine_agg: pf!(6),
                age: pf!(7),
                ground_truth_strength: pf!(8),
            });
        }
    }
    samples
}

fn get_data_root() -> PathBuf {
    env::var("UMST_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Try relative path from binary location
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .join("data")
        })
}

fn get_results_root() -> PathBuf {
    env::var("UMST_RESULTS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .join("results")
        })
}

// ── Egoff HTTP check ──────────────────────────────────────────────────────────

async fn check_egoff(
    client: &reqwest::Client,
    mix_tensor: Vec<f64>,
    proposed_strength: f64,
) -> Result<ConstitutionalVerdict, String> {
    let proposal = PhysicalStateProposal {
        mix_tensor,
        timestamp: 0.0,
        proposed_strength,
    };
    let res = client
        .post("http://127.0.0.1:3000/constrain")
        .json(&proposal)
        .send()
        .await
        .map_err(|e| format!("Egoff HTTP error: {}", e))?;

    if !res.status().is_success() {
        return Err(format!("Egoff server error: {}", res.status()));
    }

    res.json::<ConstitutionalVerdict>()
        .await
        .map_err(|e| format!("Egoff JSON parse error: {}", e))
}

// ── Parse agent JSON response ─────────────────────────────────────────────────

fn parse_agent_response(raw: &str, sample_idx: usize, agent_name: &str) -> AgentPrediction {
    // Find JSON block in response (agents sometimes add explanatory text)
    let json_start = raw.find('{').unwrap_or(0);
    let json_end = raw.rfind('}').map(|i| i + 1).unwrap_or(raw.len());
    let json_str = &raw[json_start..json_end];

    serde_json::from_str::<AgentPrediction>(json_str).unwrap_or_else(|_| {
        // Fallback: extract just the number via regex-like search
        let predicted_strength = extract_first_float(raw).unwrap_or(40.0).clamp(0.0, 200.0);
        eprintln!(
            "  [WARN] Could not parse JSON from {}. Extracted {:.1} MPa",
            agent_name, predicted_strength
        );
        AgentPrediction {
            sample_idx,
            agent_name: agent_name.to_string(),
            predicted_strength,
            confidence: 0.5,
            self_admits: true, // conservative assumption
            reasoning: "JSON parse failed — raw extraction used".to_string(),
        }
    })
}

fn extract_first_float(s: &str) -> Option<f64> {
    let mut in_number = false;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() || c == '-' || c == '.' {
            if !in_number {
                in_number = true;
                start = i;
            }
        } else if in_number {
            if let Ok(v) = s[start..i].parse::<f64>() {
                return Some(v);
            }
            in_number = false;
        }
    }
    None
}

// ── Grok API call ─────────────────────────────────────────────────────────────

async fn call_grok(
    client: &reqwest::Client,
    api_key: &str,
    prompt: &str,
) -> Result<String, String> {
    let req = GrokRequest {
        model: "grok-code-fast-1".to_string(),
        messages: vec![GrokMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        temperature: 0.0,
        max_tokens: 300,
    };

    let res = client
        .post("https://api.x.ai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("Grok network error: {}", e))?;

    if !res.status().is_success() {
        return Err(format!(
            "Grok API error {}: {}",
            res.status(),
            res.text().await.unwrap_or_default()
        ));
    }

    let grok_res: GrokResponse = res
        .json()
        .await
        .map_err(|e| format!("Grok JSON parse error: {}", e))?;

    grok_res
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| "Grok returned no choices".to_string())
}

// ── Build telemetry record ────────────────────────────────────────────────────

fn build_record(
    sample: &Sample,
    prediction: &AgentPrediction,
    verdict: &ConstitutionalVerdict,
) -> TelemetryRecord {
    let gate_admissible = matches!(verdict, ConstitutionalVerdict::Admissible { .. });
    let (gate_violation, gate_validated_strength) = match verdict {
        ConstitutionalVerdict::Admissible {
            validated_strength, ..
        } => (None, Some(*validated_strength)),
        ConstitutionalVerdict::Rejected { violation, .. } => {
            (Some(violation.clone()), None)
        }
    };
    let baseline = sample.physics_baseline();
    TelemetryRecord {
        sample_idx: sample.idx,
        dataset: sample.dataset.clone(),
        agent: prediction.agent_name.clone(),
        mix_tensor: sample.mix_tensor(),
        ground_truth_mpa: sample.ground_truth_strength,
        physics_baseline_mpa: baseline,
        raw_prediction_mpa: prediction.predicted_strength,
        gate_admissible,
        gate_violation,
        gate_validated_strength,
        agent_confidence: prediction.confidence,
        agent_self_admits: prediction.self_admits,
        epistemic_calibration_correct: prediction.self_admits == gate_admissible,
        agent_reasoning: prediction.reasoning.clone(),
        absolute_error_mpa: (prediction.predicted_strength - sample.ground_truth_strength).abs(),
        baseline_error_mpa: (baseline - sample.ground_truth_strength).abs(),
    }
}

// ── Summary statistics ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AgentSummary {
    agent: String,
    n_samples: usize,
    admissibility_rate: f64,
    mean_absolute_error: f64,
    baseline_mae: f64,
    mae_vs_baseline_delta: f64,
    mean_confidence: f64,
    epistemic_calibration_accuracy: f64,
    /// % of time agent correctly predicted its own rejection
    self_rejection_awareness: f64,
    violations_by_type: std::collections::HashMap<String, usize>,
}

fn compute_summary(records: &[TelemetryRecord], agent: &str) -> AgentSummary {
    let agent_records: Vec<&TelemetryRecord> = records
        .iter()
        .filter(|r| r.agent == agent)
        .collect();
    let n = agent_records.len();
    if n == 0 {
        return AgentSummary {
            agent: agent.to_string(),
            n_samples: 0,
            admissibility_rate: 0.0,
            mean_absolute_error: f64::NAN,
            baseline_mae: f64::NAN,
            mae_vs_baseline_delta: f64::NAN,
            mean_confidence: f64::NAN,
            epistemic_calibration_accuracy: f64::NAN,
            self_rejection_awareness: f64::NAN,
            violations_by_type: Default::default(),
        };
    }

    let admissible = agent_records.iter().filter(|r| r.gate_admissible).count();
    let mae = agent_records.iter().map(|r| r.absolute_error_mpa).sum::<f64>() / n as f64;
    let baseline_mae = agent_records.iter().map(|r| r.baseline_error_mpa).sum::<f64>() / n as f64;
    let mean_conf = agent_records.iter().map(|r| r.agent_confidence).sum::<f64>() / n as f64;
    let cal_correct = agent_records.iter().filter(|r| r.epistemic_calibration_correct).count();

    // Among rejected samples, how many did the agent correctly predict would be rejected?
    let rejected: Vec<&&TelemetryRecord> = agent_records.iter().filter(|r| !r.gate_admissible).collect();
    let self_rejection_awareness = if rejected.is_empty() {
        1.0
    } else {
        rejected.iter().filter(|r| !r.agent_self_admits).count() as f64 / rejected.len() as f64
    };

    let mut violations_by_type: std::collections::HashMap<String, usize> = Default::default();
    for r in &agent_records {
        if let Some(v) = &r.gate_violation {
            *violations_by_type.entry(v.clone()).or_insert(0) += 1;
        }
    }

    AgentSummary {
        agent: agent.to_string(),
        n_samples: n,
        admissibility_rate: admissible as f64 / n as f64,
        mean_absolute_error: mae,
        baseline_mae,
        mae_vs_baseline_delta: mae - baseline_mae,
        mean_confidence: mean_conf,
        epistemic_calibration_accuracy: cal_correct as f64 / n as f64,
        self_rejection_awareness,
        violations_by_type,
    }
}

// ── Print summary table ───────────────────────────────────────────────────────

fn print_summary_table(summaries: &[AgentSummary]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              PATH B CONSTITUTIONAL BENCHMARK — FINAL RESULTS                   ║");
    println!("╠══════════════════════╦═══════╦══════════╦═══════════╦══════════╦═════════════╣");
    println!("║ Agent                ║  N    ║ Adm.%    ║ MAE (MPa) ║ Conf.    ║ Epist.Cal.% ║");
    println!("╠══════════════════════╬═══════╬══════════╬═══════════╬══════════╬═════════════╣");
    for s in summaries {
        println!(
            "║ {:<20} ║ {:>5} ║ {:>7.1}% ║ {:>9.2} ║ {:>7.3}  ║ {:>10.1}% ║",
            &s.agent[..s.agent.len().min(20)],
            s.n_samples,
            s.admissibility_rate * 100.0,
            s.mean_absolute_error,
            s.mean_confidence,
            s.epistemic_calibration_accuracy * 100.0
        );
    }
    println!("╚══════════════════════╩═══════╩══════════╩═══════════╩══════════╩═════════════╝");
    println!("\n  Admissibility: % of predictions that pass the DUMSTO thermodynamic gate");
    println!("  MAE: Mean Absolute Error vs 28-day ground truth (MPa)");
    println!("  Epistemic Cal.: % of samples where agent correctly predicted gate verdict");
}

// ── PHASE: generate cursor prompts ───────────────────────────────────────────

fn phase_generate_cursor_prompts(samples: &[Sample], results_root: &PathBuf) {
    let prompts: Vec<AgentPrompt> = samples
        .iter()
        .map(|s| AgentPrompt {
            sample_idx: s.idx,
            dataset: s.dataset.clone(),
            mix_tensor_description: MixDescription {
                cement_kg_m3: s.cement,
                slag_kg_m3: s.slag,
                fly_ash_kg_m3: s.fly_ash,
                water_kg_m3: s.water,
                superplasticizer_kg_m3: s.superplasticizer,
                coarse_aggregate_kg_m3: s.coarse_agg,
                fine_aggregate_kg_m3: s.fine_agg,
                curing_age_days: s.age,
            },
            instructions: build_prompt(s),
            response_schema: r#"{"sample_idx":int,"agent_name":str,"predicted_strength":f64,"confidence":f64,"self_admits":bool,"reasoning":str}"#.to_string(),
        })
        .collect();

    let path = results_root.join("path_b_prompts.json");
    fs::create_dir_all(results_root).expect("Cannot create results dir");
    fs::write(&path, serde_json::to_string_pretty(&prompts).unwrap())
        .expect("Cannot write prompts file");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PATH B — PHASE 1: CURSOR PROMPT FILE GENERATED             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  File: results/path_b_prompts.json                          ║");
    println!("║  Samples: {:>4}                                              ║", prompts.len());
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  NEXT STEPS FOR CURSOR AGENT (Claude):                      ║");
    println!("║  1. Read results/path_b_prompts.json                        ║");
    println!("║  2. For each prompt, respond with the JSON schema above     ║");
    println!("║  3. Write all responses to results/path_b_predictions.json  ║");
    println!("║  4. Run: cargo run --bin path_b_benchmark -- \\              ║");
    println!("║           --mode cursor --phase evaluate                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ── PHASE: evaluate cursor predictions ───────────────────────────────────────

async fn phase_evaluate_cursor_predictions(
    samples: &[Sample],
    results_root: &PathBuf,
    egoff_client: &reqwest::Client,
) -> Vec<TelemetryRecord> {
    let predictions_path = results_root.join("path_b_predictions.json");
    let raw = fs::read_to_string(&predictions_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. Run --phase generate first, then provide predictions.", predictions_path.display()));

    let predictions: Vec<AgentPrediction> = serde_json::from_str(&raw)
        .expect("Cannot parse path_b_predictions.json");

    let mut records = Vec::new();
    let sample_map: std::collections::HashMap<usize, &Sample> =
        samples.iter().map(|s| (s.idx, s)).collect();

    for pred in &predictions {
        if let Some(sample) = sample_map.get(&pred.sample_idx) {
            print!(
                "  [Cursor] Sample {:>4} → predicted {:.1} MPa, self_admits={} ... ",
                pred.sample_idx, pred.predicted_strength, pred.self_admits
            );

            match check_egoff(egoff_client, sample.mix_tensor(), pred.predicted_strength).await {
                Ok(verdict) => {
                    let admissible = matches!(verdict, ConstitutionalVerdict::Admissible { .. });
                    let calibration_ok = pred.self_admits == admissible;
                    println!(
                        "{} | calibration: {}",
                        if admissible { "✅ ADMIT" } else { "❌ REJECT" },
                        if calibration_ok { "✅" } else { "❌ miscalibrated" }
                    );
                    records.push(build_record(sample, pred, &verdict));
                }
                Err(e) => {
                    eprintln!("Egoff error: {}", e);
                }
            }
        }
    }
    records
}

// ── PHASE: Grok standalone benchmark ─────────────────────────────────────────

async fn phase_grok_benchmark(
    samples: &[Sample],
    egoff_client: &reqwest::Client,
) -> Vec<TelemetryRecord> {
    let api_key = env::var("GROK_API_KEY")
        .or_else(|_| env::var("XAI_API_KEY"))
        .expect("GROK_API_KEY or XAI_API_KEY must be set in config/.env.umst");

    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Cannot build HTTP client");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PATH B2 — GROK STANDALONE BENCHMARK                        ║");
    println!("║  Model: grok-code-fast-1  |  N={:<4}  |  Gate: Egoff HTTP   ║", samples.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut records = Vec::new();
    let mut admitted = 0usize;

    for (i, sample) in samples.iter().enumerate() {
        let prompt = build_prompt(sample);

        print!(
            "  [{:>3}/{}] Sample {:>4} (w/c={:.3}, age={:.0}d) ... ",
            i + 1,
            samples.len(),
            sample.idx,
            sample.water / sample.cement.max(1.0),
            sample.age
        );

        let raw_response = match call_grok(&http_client, &api_key, &prompt).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Grok API error: {}", e);
                continue;
            }
        };

        let pred = parse_agent_response(&raw_response, sample.idx, "Grok_grok-code-fast-1");

        match check_egoff(egoff_client, sample.mix_tensor(), pred.predicted_strength).await {
            Ok(verdict) => {
                let admissible = matches!(verdict, ConstitutionalVerdict::Admissible { .. });
                if admissible {
                    admitted += 1;
                }
                let calibration_ok = pred.self_admits == admissible;
                println!(
                    "pred={:.1}MPa gt={:.1}MPa {} cal:{}",
                    pred.predicted_strength,
                    sample.ground_truth_strength,
                    if admissible { "✅" } else { "❌" },
                    if calibration_ok { "✅" } else { "⚠️" }
                );
                records.push(build_record(sample, &pred, &verdict));
            }
            Err(e) => {
                eprintln!("Egoff error: {}", e);
            }
        }

        // Brief pause to respect API rate limits
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    println!(
        "\n  Grok admissibility: {}/{} = {:.1}%",
        admitted,
        records.len(),
        admitted as f64 / records.len().max(1) as f64 * 100.0
    );
    records
}

// ── Physics baseline (Group 1 reference) ─────────────────────────────────────

async fn phase_physics_baseline(
    samples: &[Sample],
    egoff_client: &reqwest::Client,
    agent_name: &str,
) -> Vec<TelemetryRecord> {
    let mut records = Vec::new();
    for sample in samples {
        let baseline = sample.physics_baseline();
        let pred = AgentPrediction {
            sample_idx: sample.idx,
            agent_name: agent_name.to_string(),
            predicted_strength: baseline,
            confidence: 0.95,
            self_admits: true,
            reasoning: format!(
                "Powers gel-space model: w/c={:.3}, age={:.0}d → fc={:.1}MPa",
                sample.water / sample.cement.max(1.0),
                sample.age,
                baseline
            ),
        };
        if let Ok(verdict) = check_egoff(egoff_client, sample.mix_tensor(), baseline).await {
            records.push(build_record(sample, &pred, &verdict));
        }
    }
    records
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    dotenv().ok();

    let args: Vec<String> = env::args().collect();
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("grok");

    let phase = args
        .iter()
        .position(|a| a == "--phase")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("run");

    let n_samples: usize = args
        .iter()
        .position(|a| a == "--samples")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20); // Default: 20 samples for fast testing

    let use_all_datasets = args.iter().any(|a| a == "--dataset" || a == "all");

    let data_root = get_data_root();
    let results_root = get_results_root();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  UMST PATH B CONSTITUTIONAL BENCHMARK                        ║");
    println!("║  Mode: {:>8}  |  Phase: {:>8}  |  N: {:>4}              ║",
        mode, phase, n_samples);
    println!("║  Data: {}  ║", data_root.display());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load datasets
    let dataset_defs: Vec<(&str, &str)> = if use_all_datasets {
        vec![
            ("UCI-D1", "dataset_D1.csv"),
            ("UCI-D2", "dataset_D2.csv"),
            ("UHPC", "dataset_uhpc.csv"),
            ("SELFHEAL", "dataset_selfheal.csv"),
            ("LUNAR", "dataset_lunar.csv"),
            ("HIGHSCM", "dataset_highscm.csv"),
        ]
    } else {
        vec![("UCI-D1", "dataset_D1.csv")]
    };

    let mut all_samples: Vec<Sample> = Vec::new();
    for (name, file) in &dataset_defs {
        let path = data_root.join(file);
        let mut ds = load_dataset(&path, name, n_samples);
        // Re-index to avoid collisions across datasets
        let offset = all_samples.len();
        for s in &mut ds {
            s.idx += offset;
        }
        println!("  Loaded {} samples from {}", ds.len(), name);
        all_samples.extend(ds);
    }

    if all_samples.is_empty() {
        eprintln!("ERROR: No samples loaded. Set UMST_DATA_ROOT or run from the repo root.");
        std::process::exit(1);
    }

    println!("  Total: {} samples across {} dataset(s)\n", all_samples.len(), dataset_defs.len());

    // Egoff client shared across all checks
    let egoff_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("Cannot build Egoff client");

    // ── Dispatch ─────────────────────────────────────────────────────────────

    let mut all_records: Vec<TelemetryRecord> = Vec::new();
    let t_start = Instant::now();

    match (mode, phase) {
        // ── Cursor: generate prompt file ──────────────────────────────────
        ("cursor", "generate") => {
            phase_generate_cursor_prompts(&all_samples, &results_root);
            return; // Exit — human (or agent in Cursor) fills in predictions
        }

        // ── Cursor: evaluate agent predictions against Egoff ─────────────
        ("cursor", "evaluate") => {
            // Always also run physics baseline for comparison
            let baseline_records =
                phase_physics_baseline(&all_samples, &egoff_client, "Physics_Baseline").await;
            all_records.extend(baseline_records);

            let cursor_records =
                phase_evaluate_cursor_predictions(&all_samples, &results_root, &egoff_client).await;
            all_records.extend(cursor_records);
        }

        // ── Grok: fully automatic standalone benchmark ────────────────────
        ("grok", _) => {
            // Group 1 reference: physics baseline
            println!("Running Group 1 reference: Physics baseline...");
            let baseline_records =
                phase_physics_baseline(&all_samples, &egoff_client, "Physics_Baseline").await;
            all_records.extend(baseline_records);

            // Group 3: Grok via direct API
            println!("\nRunning Group 3: Grok API...");
            let grok_records = phase_grok_benchmark(&all_samples, &egoff_client).await;
            all_records.extend(grok_records);
        }

        _ => {
            eprintln!("Unknown mode/phase: {} / {}", mode, phase);
            eprintln!("Valid: --mode grok | --mode cursor --phase generate | --mode cursor --phase evaluate");
            std::process::exit(1);
        }
    }

    let elapsed = t_start.elapsed();

    // ── Compute and print summaries ───────────────────────────────────────
    let agents: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        all_records.iter()
            .map(|r| r.agent.clone())
            .filter(|a| seen.insert(a.clone()))
            .collect()
    };

    let summaries: Vec<AgentSummary> = agents
        .iter()
        .map(|a| compute_summary(&all_records, a))
        .collect();

    print_summary_table(&summaries);

    println!("\n  Elapsed: {:.2}s | Records: {}", elapsed.as_secs_f64(), all_records.len());

    // ── Save outputs ──────────────────────────────────────────────────────
    fs::create_dir_all(&results_root).expect("Cannot create results dir");

    let telemetry_path = results_root.join("path_b_telemetry.json");
    fs::write(&telemetry_path, serde_json::to_string_pretty(&all_records).unwrap())
        .expect("Cannot write telemetry");
    println!("  Telemetry → {}", telemetry_path.display());

    let summary_path = results_root.join("path_b_summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summaries).unwrap())
        .expect("Cannot write summary");
    println!("  Summary   → {}", summary_path.display());

    // ── Path B verdict ────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PATH B BENCHMARK COMPLETE                                   ║");
    println!("║                                                              ║");
    println!("║  The DUMSTO gate acts as a functorial admissible wrapper:    ║");
    println!("║  any external agent's prediction is constitutionally         ║");
    println!("║  checked against the Clausius-Duhem inequality.              ║");
    println!("║                                                              ║");
    println!("║  Admissibility is NOT a soft penalty — it is a hard gate.   ║");
    println!("║  Rejected predictions never enter the agent's world model.  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
