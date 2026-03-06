# Known Limitations — IROS 2026 Reproducibility Package

## PPO Training (NaN Gradient Explosion)

The PPO agent's `optimize()` method triggers NaN weight explosion at approximately step 8,000 due to insufficient gradient clipping in the current learning rate schedule. Training is disabled in `ssot_benchmark.rs` (line ~783). The PPO agent uses `select_action()` (forward pass only), which means it tracks the physics kernel's predictions without backpropagation. This is the behavior reported in the paper.

**Impact on paper results:** None. All reported benchmark numbers (MAE, TQ, admissibility) are computed using the physics kernel and epistemic proxy selector, not PPO training. The Hybrid method combines physics kernel output with PPO reward signals without gradient updates.

## Service Ports

See the port assignments table in [REPRODUCE.md](REPRODUCE.md#6-port-assignments-if-running-live-services). Services are not required for benchmark reproduction.

## Calibration Caveats

1. **LUNAR** — Uses Davidovits geopolymerization kinetics (not Powers' C-S-H model). Low MAE reflects the narrow synthetic range (3-38 MPa); performance on broader geopolymer ranges is untested.
2. **UHPC** — An Arrhenius correction (×2.68, Ea ≈ 41 kJ/mol) for 90°C steam curing is applied in-loop in `ssot_benchmark.rs` (line ~320). The calibration is 20°C-referenced; the correction adjusts for steam cure at benchmark time.
3. **HIGHSCM** — A simplified secondary GGBFS activation term is applied after 7 days (`alpha += k_slag * (1 - exp(-0.02*(age-7)))`). Full GGBFS complexity (pH-dependent activation, slag surface area, activator concentration) remains unmodeled, which is why XGBoost outperforms the physics kernel on this dataset.
4. **D2/D3/D4 Superplasticizer** — The superplasticizer dosage column is zero-filled across datasets D2, D3, and D4. The original data sources did not report superplasticizer content. The physics kernel treats these as zero-dosage mixes, which may slightly overestimate water demand in mixes that actually contained superplasticizer. This does not affect the primary MAE or TQ claims since the benchmark evaluation is performed against measured compressive strength, not predicted workability.
5. **UHPC Steel Fiber Volume** — UHPC mixes typically contain approximately 2% by volume of steel fibers, which significantly affects post-cracking tensile behavior and ductility. The current physics kernel does not model fiber reinforcement contributions. Reported UHPC compressive strength predictions are unaffected (fibers primarily influence tensile/flexural behavior), but any future extension to tensile or flexural predictions would require a fiber pullout model.

## Excluded Modules

The following modules are excluded from the reproducibility package because they are not required for reproducing the paper's claims:

| Module | Reason for Exclusion |
|--------|---------------------|
| `geometry` | 3-D mesh and FEM integration; not used in the benchmark pipeline. |
| `ibe` | Identity-based encryption for secure telemetry; infrastructure-only, no effect on benchmark results. |
| `ml` | General-purpose ML utilities (XGBoost wrappers, etc.); the benchmark uses the physics kernel directly. |
| `neural` | Neural network layers for future GNN experiments; not referenced in any reported result. |
| `oracle` | Ground-truth oracle for simulation-only validation; not part of the evaluation protocol. |
| `physics` | Legacy physics module superseded by the `science/` engines in `core`; retained upstream for compatibility. |
| `profiler` | Performance profiling instrumentation; diagnostic-only, does not affect outputs. |
| `robotics` | Robot motion planning and control; separate from material characterization claims. |
| `search` | Combinatorial mix-design search; not exercised by the benchmark binary. |
| `trust` | Trust-region policy optimization (experimental); unused in reported results. |
| `validation` | Extended cross-validation harness; the benchmark uses its own internal validation. |

## Scope of Claims

This reproducibility package is scoped to support the specific claims made in the paper. The following clarifies what the package does and does not demonstrate:

**What the package proves:**

- The physics kernel produces the MAE, TQ, and admissibility values reported in Tables 2-4 across four material domains (UCI, LUNAR, UHPC, HIGHSCM).
- The Clausius-Duhem inequality gate enforces thermodynamic admissibility on all benchmark outputs. The `ssot_benchmark` binary reports 100% admissibility for the four primary domains (UCI, LUNAR, UHPC, HIGHSCM). The paper's Table IV reports lower admissibility on certain cross-validated domains (e.g., 82% on D1, 85% on UHPC) because those values reflect the fraction of samples for which the physics kernel produces an admissible prediction; the gate correctly vetoes the remainder rather than allowing inadmissible outputs to propagate.
- The epistemic proxy selector identifies optimal sensor subsets using mutual information, as reported in the sensor selection experiments.

**What the package does not prove:**

- Generalization beyond the eight included datasets. Performance on unseen material families or broader composition ranges is not claimed.
- Real-time robotic deployment. The benchmark runs offline on static datasets; hardware-in-the-loop latency and control integration are not demonstrated.
- PPO training convergence. The PPO module is included in forward-pass-only mode; online reinforcement learning is not functional (see the PPO Training section above).
- Fiber-reinforced tensile/flexural predictions. The physics kernel targets compressive strength only; steel fiber contributions are not modeled (see Calibration Caveats, item 5).
