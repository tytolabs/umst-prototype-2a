# Reproducing Paper Results — UMST Prototype 2a

This document describes how to reproduce the benchmark numbers reported in the paper.

Scope note: reproduced values are protocol-specific. Treat them as
experiment-scoped evidence, not universal guarantees.

## Prerequisites

- Rust 1.75+ (`rustup update stable`)
- Python 3.10+
- ~4 GB disk for build artifacts

## 1. Build the Rust core

```bash
cd prototype/src/rust
cargo build --release
```

Expected: `Finished release profile` with warnings only (no errors).

## 2. Run the SSOT benchmark

**Important:** The benchmark uses relative paths and must be run from `prototype/src/rust/core`:

```bash
cd prototype/src/rust/core
cargo run --release --bin ssot_benchmark
```

Alternatively, from the package root:
```bash
cd prototype/src/rust/core && cargo run --release --bin ssot_benchmark
```

This produces the MAE, TQ, and admissibility numbers used in the paper-aligned
tables for this package revision.
Results are written to `prototype/results/canonical/tables/`.
All randomness is seeded (PPO agent, RandomForest, noise injection) for reproducible TABLE3 outputs.

### 2a. hardware_heat_experiment (Theorem 3 — Landauer energy)

**Hardware-valid runs** (integrated energy, not the synthetic proxy):

- **macOS (Apple Silicon):** `powermetrics` must succeed — typically **`sudo`**.
- **Linux:** package RAPL counter readable under sysfs (e.g. `intel-rapl:0/energy_uj`); phase totals use **Δenergy** across each measured region.

**Strict no-fallback (CI / audit):** if the environment variable `UMST_HARDWARE_STRICT` is set to `1`, `true`, `yes`, or `on`, the binary **exits immediately** when neither powermetrics nor Linux RAPL is available (no silent proxy run).

```bash
# macOS — strict (fails without sudo / PMU). Use env VAR=value for portable assignment.
sudo env UMST_HARDWARE_STRICT=1 ./prototype/src/rust/target/release/hardware_heat_experiment

# Linux — strict (requires RAPL sysfs)
env UMST_HARDWARE_STRICT=1 ./prototype/src/rust/target/release/hardware_heat_experiment
```

Without `sudo` on macOS, the binary falls back to a FLOP-count proxy unless strict mode is enabled. The startup message indicates the path (`✅ Apple Silicon PMU`, `✅ Linux — RAPL`, or the proxy warning).

**Output path:** `thermal_proof.csv` is written to the **current working directory**; run from a known directory (e.g. `cd prototype/src/rust/core` or package root) if you need a stable artifact location.

Validation note: Theorem 8 plausibility is evaluated in **µJ/op** (not total µJ).
If you rerun after updating `hardware_heat_experiment.rs`, confirm the output line
reads `ΔE/op=... µJ/op vs predicted ... µJ/op`.

## 3. Python dependencies (optional — for figure generation only)

```bash
pip install -r requirements.txt
```

Then run figure scripts in `prototype/scripts/`.

## 4. Expected results

| Domain  | Physics Kernel MAE | Hybrid MAE | TQ    | Admissibility (reported protocol) |
|---------|--------------------|------------|-------|---------------|
| UCI     | 4.21 MPa           | 3.87 MPa   | 0.686 | 100%          |
| LUNAR   | 1.83 MPa           | 1.76 MPa   | 0.701 | 100%          |
| UHPC    | 5.44 MPa           | 4.92 MPa   | 0.673 | 100%          |
| HIGHSCM | 6.12 MPa           | 5.61 MPa   | 0.648 | 100%          |

See `KNOWN_LIMITATIONS.md` for PPO training status and calibration caveats.
For full caveats, confidence framing, and conflict-resolution notes, see
`prototype/results/MASTER_RESULTS.md`.

## 5. Verification

To verify that your results match the expected values:

1. **MAE columns** should match within rounding tolerance (less than 0.05 MPa difference from the table above).
2. **TQ (Thermodynamic Quality)** values should match to three decimal places.
3. **Admissibility** should reproduce the reported benchmark values for this
   protocol (100% in the table above). Divergence indicates build/data/config
   differences and should be investigated before claim reuse.
4. **Reproducibility:** TABLE3 outputs are deterministic. Run the benchmark twice
   from `prototype/src/rust/core`; the JSON and CSV outputs should be identical.

The benchmark prints a `PASS`/`FAIL` summary line at the end of execution. All four domains should report `PASS`. If any domain reports `FAIL`, check that (a) the datasets in `prototype/data/` have not been modified, and (b) you are using Rust 1.75 or later.

## 6. Port assignments (if running live services)

| Service              | Port | Binary        |
|----------------------|------|---------------|
| DUMSTO Gate + OCR    | 8765 | `gate_server` |
| Egoff Agent HTTP     | 3000 | `egoff_cli`   |
| PPO Server           | 8080 | `ppo_server`  |
| WebSocket Telemetry  | 8766 | `gate_server` (secondary) |

Services are optional for benchmark reproduction; `ssot_benchmark` runs standalone.

## 7. Reproducibility (TABLE3 drift fix)

TABLE3 outputs were previously non-deterministic due to:
- **PPO Agent:** `select_action` used `rand::thread_rng()` for exploration noise; GNN weights used unseeded `thread_rng`.
- **RandomForest:** smartcore's `Default::default()` used `seed: 0`; explicit `with_seed(42)` ensures stability.
- **CWD dependency:** Relative paths `../../../data/` require running from `prototype/src/rust/core`.

Fixes applied:
- `PPOConfig::with_seed(42)` seeds GNN init and action sampling.
- `RandomForestRegressorParameters::default().with_seed(42)` in `fit_rf`.
- Noise injection already used `SmallRng::seed_from_u64(12345)`.

Run twice from `prototype/src/rust/core`; TABLE3 JSON and CSV should match exactly.

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `cargo build` fails with syntax errors | Ensure Rust >= 1.75. Run `rustup update stable` to upgrade. |
| Linker errors on Linux | Install build essentials: `sudo apt install build-essential pkg-config libssl-dev`. |
| Linker errors on macOS | Install Xcode command-line tools: `xcode-select --install`. |
| Python `ModuleNotFoundError` | Run `pip install -r requirements.txt` from the package root. |
| Benchmark prints unexpected MAE values | Verify dataset integrity: the CSV files in `prototype/data/` must not be modified. Re-extract from the original archive if needed. |
| `Permission denied` running the binary | Make it executable: `chmod +x prototype/src/rust/target/release/ssot_benchmark`. |
| `hardware_heat_experiment` shows fallback mode | Run with `sudo` on macOS for real PMU power; non-root uses FLOP proxy. |

For package overview and directory structure, see [README.md](README.md).
