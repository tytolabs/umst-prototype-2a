# Rust Binary Reference

> UMST Prototype 2a
> Paper: "Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization"

## 1. Overview

The Rust codebase compiles to 22 binaries: 5 essential for reproducing paper results
and 17 optional utilities for exploration, visualization, and extended analysis.

## 2. Build Instructions

### Prerequisites

- Rust toolchain (edition 2021 or later)
- Cargo package manager (included with Rust)

### Building

```bash
cd src/rust
cargo build --release
```

All binaries are output to:

```
src/rust/target/release/
```

Debug builds (without `--release`) are placed in `src/rust/target/debug/` but
are significantly slower and not recommended for benchmarking.

### Build Verification

```bash
cd src/rust
cargo test
```

All tests must pass before running experiments.

## 3. Essential Binaries (5)

These binaries reproduce the core results reported in the paper.

### 3.1 `ssot_benchmark`

**Purpose**: Main benchmark runner. Reproduces Tables 2-4 from the paper.

**What it does**:
- Loads all 8 datasets (18,146 samples)
- Runs all 15 science engines
- Applies 3 noise levels (clean, low, medium, high)
- Computes MAE, RMSE, R², MAPE for each configuration
- Enforces thermodynamic gate and reports admissibility rates
- Writes results to stdout and CSV

**Usage**:
```bash
./target/release/ssot_benchmark
```

**Expected runtime**: ~5-15 minutes depending on hardware.

---

### 3.2 `epistemic_experiment`

**Purpose**: Epistemic sensing validation. Supports Theorem 1 from the paper.

**What it does**:
- Computes mutual information between all proxy-target pairs
- Generates MI heatmaps
- Calculates TQ (Thermodynamic Quality) scores
- Computes Cohen's d effect size for TQ-guided vs. random proxy selection
- Outputs statistical significance tests

**Usage**:
```bash
./target/release/epistemic_experiment
```

**Expected runtime**: ~2-5 minutes.

---

### 3.3 `veto_experiment`

**Purpose**: Thermodynamic gate rejection demonstration. Supports Theorem 2.

**What it does**:
- Generates physically implausible input scenarios
- Passes them through the thermodynamic gate (Clausius-Duhem check)
- Demonstrates veto behavior on inadmissible predictions
- Reports rejection rates and dissipation values
- Compares gated vs. ungated prediction quality

**Usage**:
```bash
./target/release/veto_experiment
```

**Expected runtime**: ~1-3 minutes.

---

### 3.4 `hardware_heat_experiment`

**Purpose**: Landauer energy bound verification. Supports Theorem 3.

**What it does**:
- Measures computational energy per bit of information processed
- Verifies that the architecture's information-theoretic operations
  respect the Landauer limit (kT ln 2 per bit erasure)
- Reports energy efficiency metrics

**Energy source hierarchy (strict → fallback)**:
- **macOS + root**: Real Apple Silicon PMU via `powermetrics` — run with `sudo`
- **Linux**: Real sysfs RAPL energy counters (Intel/AMD)
- **Fallback** (macOS non-root, CI): FLOP-count proxy (4.5 µJ/µs). Valid for
  algorithm comparison only; not physical proof. The binary prints which mode
  is active at startup.

**Usage**:
```bash
# Strict no-fallback mode (real hardware power) on macOS:
sudo ./target/release/hardware_heat_experiment

# Fallback mode (non-root):
./target/release/hardware_heat_experiment
```

**Expected runtime**: ~1-2 minutes.

---

### 3.5 `egoff_cli`

**Purpose**: Constitutional gate CLI interface for interactive exploration.

**What it does**:
- Provides a command-line interface to the thermodynamic gate
- Accepts individual material compositions and returns admissibility verdict
- Useful for exploring gate behavior on custom inputs
- Supports batch mode for processing CSV files

**Usage**:
```bash
# Interactive mode
./target/release/egoff_cli

# Batch mode
./target/release/egoff_cli --input data/custom_mixes.csv --output results/gate_results.csv
```

## 4. Optional Binaries (17)

These utilities provide additional analysis, visualization, and exploration
capabilities beyond the core paper results.

| Binary | Description |
|--------|-------------|
| `hydration_demo` | Avrami-Parrott hydration curve visualization |
| `strength_predictor` | Standalone strength prediction from mix design |
| `rheology_analyzer` | Herschel-Bulkley rheological parameter fitting |
| `maturity_calculator` | Arrhenius maturity index computation |
| `porosity_estimator` | Powers' model porosity estimation |
| `transport_simulator` | Chloride/moisture diffusion simulation |
| `fracture_analyzer` | Fracture mechanics parameter computation |
| `shrinkage_model` | Drying shrinkage prediction |
| `creep_model` | Viscoelastic creep prediction |
| `thermal_solver` | Heat of hydration temperature evolution |
| `carbonation_model` | CO2 carbonation depth estimation |
| `durability_assessor` | Service life prediction tool |
| `selfheal_simulator` | Self-healing kinetics simulation |
| `geopolymer_engine` | Geopolymer activation model |
| `tensor_inspector` | Material-state tensor visualization |
| `mi_calculator` | Standalone mutual information computation |
| `dataset_validator` | CSV dataset schema validation and statistics |

### Usage Pattern

All optional binaries follow the same invocation pattern:

```bash
./target/release/<binary_name> [--help] [OPTIONS]
```

Use `--help` with any binary to see its full argument list.

## 5. Port Assignments

The following port assignments are used when binaries are run as live services
(e.g., for real-time prediction or dashboard integration):

| Port | Service | Binary |
|------|---------|--------|
| 8080 | Main benchmark API | `ssot_benchmark` |
| 8081 | Epistemic sensing API | `epistemic_experiment` |
| 8082 | Thermodynamic gate API | `egoff_cli` |
| 8083 | Strength prediction | `strength_predictor` |
| 8084 | Hydration model | `hydration_demo` |
| 8085 | Rheology analysis | `rheology_analyzer` |
| 8086 | Maturity computation | `maturity_calculator` |
| 8087 | Transport simulation | `transport_simulator` |
| 8088 | Tensor inspection | `tensor_inspector` |
| 8089 | Dataset validation | `dataset_validator` |

**Note**: Service mode is optional. All binaries default to CLI batch mode
unless the `--serve` flag is provided.

## 6. Output Artifacts

Essential binaries produce the following output files:

| Binary | Output | Location |
|--------|--------|----------|
| `ssot_benchmark` | Benchmark results CSV | `results/benchmark_results.csv` |
| `epistemic_experiment` | MI heatmap data, TQ scores | `results/epistemic/` |
| `veto_experiment` | Veto statistics, dissipation log | `results/veto/` |
| `hardware_heat_experiment` | Energy measurements | `results/energy/` |
| `egoff_cli` | Gate verdicts (batch mode) | User-specified via `--output` |

## 7. Troubleshooting

**Build fails with missing dependencies**:
```bash
cargo update
cargo build --release
```

**Binary not found after build**:
Ensure you are running from the `src/rust/` directory and the build completed
without errors. Check `target/release/` for the compiled binaries.

**Slow performance**:
Ensure you are using `--release` builds. Debug builds are 10-50x slower.

**Dataset not found errors**:
Binaries expect datasets in `../../data/` relative to the binary location,
or in the `data/` directory at the repository root. Set the `DATA_DIR`
environment variable to override:
```bash
DATA_DIR=/path/to/data ./target/release/ssot_benchmark
```
