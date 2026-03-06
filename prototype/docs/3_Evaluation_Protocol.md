# Evaluation Protocol

> IROS 2026 Reproducibility Package
> Paper: "Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization"

## 1. Overview

This document specifies the evaluation methodology used in the paper, including
metrics, comparison protocol, noise robustness testing, and expected results.
All evaluation procedures are deterministic and fully reproducible from the
provided binaries and datasets.

## 2. Core Prediction Metrics

Four standard regression metrics are reported for all strength prediction tasks:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | (1/n) * sum(\|y_i - y_hat_i\|) | Mean absolute prediction error (MPa) |
| RMSE | sqrt((1/n) * sum((y_i - y_hat_i)^2)) | Root mean squared error (MPa), penalizes large deviations |
| R² | 1 - SS_res / SS_tot | Coefficient of determination, fraction of variance explained |
| MAPE | (100/n) * sum(\|y_i - y_hat_i\| / \|y_i\|) | Mean absolute percentage error (%) |

All metrics are computed on held-out test splits. Lower MAE/RMSE/MAPE and
higher R² indicate better predictive performance.

## 3. Thermodynamic Safety Metrics

These metrics assess physical consistency of predictions:

### 3.1 Admissibility Rate

Percentage of predictions passing the Clausius-Duhem inequality check:

    Admissibility = (# predictions with D_int >= 0) / (# total predictions) * 100%

The architecture enforces 100% admissibility by design through the thermodynamic
gate (see Theorem 2 in the paper). Baseline models without physics constraints
typically achieve lower admissibility rates.

### 3.2 Physical Feasibility

Binary check for physically meaningful predictions:
- Strength must be non-negative
- Strength must not exceed theoretical maximum for the mix design
- Monotonicity with respect to age (strength should not decrease under
  standard curing conditions)

## 4. Epistemic Quality Metrics

### 4.1 TQ Score (Thermodynamic Quality)

The TQ metric quantifies epistemic value of proxy measurements:

    TQ = f(I(proxy; target), thermodynamic_consistency, uncertainty)

where I(proxy; target) is the mutual information between the proxy variable
and the prediction target. Higher TQ indicates a more informative and
physically consistent proxy. See Theorem 1 in the paper.

### 4.2 Cohen's d Effect Size

Measures the practical significance of TQ-guided proxy selection versus
random or heuristic selection:

    d = (mean_TQ_guided - mean_baseline) / pooled_std

Interpretation thresholds:
- |d| < 0.2: negligible effect
- 0.2 <= |d| < 0.5: small effect
- 0.5 <= |d| < 0.8: medium effect
- |d| >= 0.8: large effect

## 5. Computational Efficiency Metrics

| Metric | Description |
|--------|-------------|
| Inference latency | Wall-clock time per prediction (milliseconds) |
| Model size | Binary size on disk (MB) |
| Energy bound | Landauer limit verification (Theorem 3) |

Latency is measured as the median over 1000 predictions after a 100-prediction
warmup phase. All timing measurements use monotonic clocks.

## 6. Fair Comparison Protocol: Plateau Standard

To ensure fair comparison across methods, we adopt the following protocol:

### 6.1 Training to Convergence

All models (ours and baselines) are trained until performance plateaus:
- Convergence criterion: < 0.1% improvement in validation loss over 10
  consecutive epochs
- Maximum epoch cap to prevent runaway training
- Early stopping with patience matching the convergence window

### 6.2 Identical Data Splits

- **Split ratio**: 70% train / 15% validation / 15% test
- **Stratification**: By dataset source and strength quartile
- **Deterministic seed**: Fixed seed = 42 for all random operations
- **Reproducibility**: Identical splits guaranteed across runs

### 6.3 Deterministic Execution

- All random number generators seeded deterministically
- No non-deterministic GPU operations (CPU-only for reproducibility)
- Rust's deterministic compilation ensures binary reproducibility

## 7. Noise Robustness Protocol

Predictions are evaluated under three Gaussian noise levels applied to input
features to test robustness:

| Level | Noise Std (sigma) | Description |
|-------|-------------------|-------------|
| Clean | 0.0 | No noise (baseline) |
| Low | 0.01 * feature_range | 1% of feature range |
| Medium | 0.05 * feature_range | 5% of feature range |
| High | 0.10 * feature_range | 10% of feature range |

Noise is added independently to each input feature. Results are averaged
over 5 noise realizations per level. The physics kernel's constitutive
constraints provide inherent regularization against noisy inputs.

## 8. Expected Results Summary

### 8.1 Strength Prediction (Table 2 in paper)

| Dataset | MAE (MPa) | RMSE (MPa) | R² | Admissibility |
|---------|-----------|------------|-----|---------------|
| D1 UCI | ~4-6 | ~5-8 | >0.85 | 100% |
| D2 NDT | ~3-5 | ~4-7 | >0.80 | 100% |
| D3 Sun | ~3-5 | ~4-7 | >0.80 | 100% |
| D4 RH | ~3-6 | ~4-8 | >0.80 | 100% |
| D5 UHPC | ~5-10 | ~7-12 | >0.85 | 100% |
| D6 LUNAR | ~2-4 | ~3-5 | >0.80 | 100% |
| D7 SELFHEAL | ~3-5 | ~4-7 | >0.80 | 100% |
| D8 HIGHSCM | ~3-6 | ~4-8 | >0.80 | 100% |

### 8.2 Epistemic Sensing (Table 3 in paper)

- TQ-guided proxy selection consistently outperforms random selection
- Cohen's d >= 0.8 (large effect) across all tested configurations
- Mutual information heatmaps reveal physically meaningful proxy rankings

### 8.3 Thermodynamic Gate (Table 4 in paper)

- 100% admissibility rate for physics-constrained predictions
- Baseline ML models show < 100% admissibility without gate enforcement
- Veto rate correlates with physical implausibility of input conditions

## 9. Running the Evaluation

To reproduce all paper results:

```bash
# Build all binaries
cd src/rust && cargo build --release

# Run main benchmark (Tables 2-4)
./target/release/ssot_benchmark

# Run epistemic experiments (Theorem 1)
./target/release/epistemic_experiment

# Run thermodynamic gate experiments (Theorem 2)
./target/release/veto_experiment

# Run energy bound verification (Theorem 3)
./target/release/hardware_heat_experiment
```

Results are written to stdout and to CSV files in the `results/` directory.
See `docs/4_Binaries.md` for complete binary documentation.
