# Reproducing Paper Results — IROS 2026 Reproducibility Package

This document describes how to reproduce the benchmark numbers reported in the IROS 2026 paper.

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

```bash
./prototype/src/rust/target/release/ssot_benchmark
```

This produces the MAE, TQ, and admissibility numbers reported in Tables 2-4.
Results are written to `prototype/results/`.

## 3. Python dependencies (optional — for figure generation only)

```bash
pip install -r requirements.txt
```

Then run figure scripts in `prototype/scripts/`.

## 4. Expected results

| Domain  | Physics Kernel MAE | Hybrid MAE | TQ    | Admissibility |
|---------|--------------------|------------|-------|---------------|
| UCI     | 4.21 MPa           | 3.87 MPa   | 0.686 | 100%          |
| LUNAR   | 1.83 MPa           | 1.76 MPa   | 0.701 | 100%          |
| UHPC    | 5.44 MPa           | 4.92 MPa   | 0.673 | 100%          |
| HIGHSCM | 6.12 MPa           | 5.61 MPa   | 0.648 | 100%          |

See `KNOWN_LIMITATIONS.md` for PPO training status and calibration caveats.

## 5. Verification

To verify that your results match the expected values:

1. **MAE columns** should match within rounding tolerance (less than 0.05 MPa difference from the table above).
2. **TQ (Thermodynamic Quality)** values should match to three decimal places.
3. **Admissibility** must be exactly 100% for all domains. Any non-100% value indicates a build or data issue.

The benchmark prints a `PASS`/`FAIL` summary line at the end of execution. All four domains should report `PASS`. If any domain reports `FAIL`, check that (a) the datasets in `prototype/data/` have not been modified, and (b) you are using Rust 1.75 or later.

## 6. Port assignments (if running live services)

| Service              | Port | Binary        |
|----------------------|------|---------------|
| DUMSTO Gate + OCR    | 8765 | `gate_server` |
| Egoff Agent HTTP     | 3000 | `egoff_cli`   |
| PPO Server           | 8080 | `ppo_server`  |
| WebSocket Telemetry  | 8766 | `gate_server` (secondary) |

Services are optional for benchmark reproduction; `ssot_benchmark` runs standalone.

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| `cargo build` fails with syntax errors | Ensure Rust >= 1.75. Run `rustup update stable` to upgrade. |
| Linker errors on Linux | Install build essentials: `sudo apt install build-essential pkg-config libssl-dev`. |
| Linker errors on macOS | Install Xcode command-line tools: `xcode-select --install`. |
| Python `ModuleNotFoundError` | Run `pip install -r requirements.txt` from the package root. |
| Benchmark prints unexpected MAE values | Verify dataset integrity: the CSV files in `prototype/data/` must not be modified. Re-extract from the original archive if needed. |
| `Permission denied` running the binary | Make it executable: `chmod +x prototype/src/rust/target/release/ssot_benchmark`. |

For package overview and directory structure, see [README.md](README.md).
