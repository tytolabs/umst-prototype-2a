# UMST Prototype 2a: Epistemic Sensing Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18940933.svg)](https://doi.org/10.5281/zenodo.18940933)

**Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization**

Santhosh Shyamsundar and Santosh Prabhu Shenbagamoorthy — Studio Tyto, Chennai, India

Preprint, March 2026.

---

## Overview

This package provides the complete code, data, and instructions to reproduce all
results reported in the paper. The system introduces a unified material-state
tensor representation coupled with an epistemic sensing architecture that
achieves physics-constrained material characterization across diverse
concrete and cementitious material families.

**Key contributions:**

- A Rust physics kernel comprising 15 thermodynamic and rheological science
  engines (27,542 LOC). In evaluated benchmark protocols, gate-checked outputs
  satisfy 100% admissibility for the reported tasks.
- An epistemic sensing module using mutual information for proxy/sensor
  selection. In the reported benchmark, this yields about a 60% reduction in
  required measurements versus random ordering.
- Reported Phase-T tracking and benchmark metrics include about 88% timing error
  reduction, a 71-fold margin relative to the specified threshold, and large
  effect sizes on the benchmark suite. See `prototype/results/MASTER_RESULTS.md`
  for experiment-scoped values, caveats, and known limitations.

## Epistemic Taxonomy for Claims

Use this package with the following claim labels:

| Label | Meaning |
|---|---|
| **Established** | Reproduced by executable benchmark artifacts in this package |
| **Extension** | New application with bounded evidence and explicit caveats |
| **Speculative** | Motivated hypothesis not resolved by package evidence alone |
| **Falsifiable** | Has explicit pass/fail criteria and can be contradicted by reruns |

Important: avoid promoting experiment-scoped outcomes to universal claims.

## Directory Structure

```
umst-prototype-2a/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── MANIFEST.md                # Complete file manifest with audit tags
├── REPRODUCE.md               # Step-by-step reproduction instructions
├── KNOWN_LIMITATIONS.md       # Known limitations and scope
├── requirements.txt           # Python dependencies
├── prototype/
│   ├── data/                  # 8 CSV datasets (18,146 samples)
│   ├── docs/                  # Architecture, datasets, evaluation, binaries
│   ├── results/               # Canonical result tables
│   ├── scripts/               # Python analysis and plotting scripts
│   └── src/rust/              # Rust physics kernel (27,542 LOC)
│       └── core/src/tensors/kleisli.rs  # KleisliArrow admissibility monad
└── ros2_bridge/               # ROS2 bridge (Python nodes → REST/WS → Rust gate)
```

## Quick Start

### Prerequisites

| Requirement  | Version   | Notes                    |
|-------------|-----------|--------------------------|
| Rust        | >= 1.75   | With cargo               |
| Python      | >= 3.10   | For analysis scripts     |
| Disk space  | ~4 GB     | Build artifacts + data   |

### Three-Step Reproduction

**Step 1 — Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Step 2 — Build the Rust physics kernel:**

```bash
cd prototype/src/rust
cargo build --release
```

**Step 3 — Run the primary benchmark:**

`ssot_benchmark` resolves datasets and writes `prototype/results/canonical/tables/` using paths relative to **`prototype/src/rust/core`**. Run it via cargo from that directory (do not invoke the binary from the repo root unless you `cd` there first):

```bash
cd prototype/src/rust/core && cargo run --release --bin ssot_benchmark
```

Equivalent after a release build:

```bash
cd prototype/src/rust/core && ../../target/release/ssot_benchmark
```

For the full reproduction workflow (all experiments, tables, and figures), see
[REPRODUCE.md](REPRODUCE.md).

## Essential Binaries

After building, six binaries are available in `prototype/src/rust/target/release/`:

| Binary                    | Purpose                                      |
|--------------------------|----------------------------------------------|
| `ssot_benchmark`         | Primary material-state tensor benchmark      |
| `gate_server`            | REST/WebSocket gate server (port 8765/8766)  |
| `epistemic_experiment`   | Epistemic sensing / sensor selection          |
| `veto_experiment`        | Thermodynamic admissibility veto gate         |
| `hardware_heat_experiment` | Hardware-in-the-loop heat validation        |
| `egoff_cli`              | EGoFF composition and analysis CLI            |

### ROS2 Bridge

The `ros2_bridge/` directory provides a ROS2 Python package that bridges
the gate server to ROS2 topics. See `ros2_bridge/README.md` for setup.

## Key Results

Results reproduced by running `ssot_benchmark` across four material domains:

| Domain  | Physics Kernel MAE | Hybrid MAE | TQ    | Admissibility |
|---------|--------------------|------------|-------|---------------|
| UCI     | 4.21 MPa           | 3.87 MPa   | 0.686 | 100%          |
| LUNAR   | 1.83 MPa           | 1.76 MPa   | 0.701 | 100%          |
| UHPC    | 5.44 MPa           | 4.92 MPa   | 0.673 | 100%          |
| HIGHSCM | 6.12 MPa           | 5.61 MPa   | 0.648 | 100%          |

For the reported benchmark domains, gate-checked predictions satisfy 100%
admissibility under the configured constraints. See
`prototype/results/MASTER_RESULTS.md` and `prototype/results/` for canonical
tables, caveats, and experiment-specific scope. Use [REPRODUCE.md](REPRODUCE.md)
for reproduction steps.

## Architecture

The system follows a layered functional programming architecture:

```
Pure Functions ──► Functors ──► Composition ──► Boundary (I/O)
```

Three core subsystems:

1. **Physics Kernel** — 15 science engines implementing thermodynamic and
   rheological constitutive models in Rust. Each engine is a pure function
   mapping material state tensors to predicted properties.

2. **Thermodynamic Gate** — Clausius-Duhem inequality enforcement layer that
   vetoes predictions that violate configured constitutional checks; reported
   admissibility is **protocol- and method-specific** (see `MASTER_RESULTS.md` and TABLE3 exports).

3. **Epistemic Sensing** — Mutual-information-based sensor selection that
   identifies the most informative measurements, reducing required sensors
   by 60% while maintaining prediction accuracy.

The physics kernel and ML components interact through a hybrid architecture
where physics-constrained predictions are composed with data-driven
corrections at the functor level.

## Companion Documents

- [REPRODUCE.md](REPRODUCE.md) — Complete step-by-step reproduction instructions
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — Known limitations and scope boundaries
- [MANIFEST.md](MANIFEST.md) — File manifest with descriptions and checksums

## Citation

```
@article{shyamsundar2026umst2a,
  title     = {Towards Unified Material-State Tensors: Epistemic Sensing
               Architecture for Physics-Constrained Material Characterization},
  author    = {Shyamsundar, Santhosh and Shenbagamoorthy, Santosh Prabhu},
  year      = {2026},
  note      = {Preprint}
}
```

## License

Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto.

This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.
