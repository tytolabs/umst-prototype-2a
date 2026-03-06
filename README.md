# UMST Prototype 2a: Epistemic Sensing Architecture

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
  engines (27,542 LOC) with 100% thermodynamic admissibility enforced via the
  Clausius-Duhem inequality gate.
- An epistemic sensing module using mutual information for optimal sensor
  selection, achieving a 60% reduction in required measurements compared to
  random selection.
- Demonstrated 88% timing error reduction, 71-fold safety margin improvement,
  and 3.5x Cohen's d effect size across 18,146 samples spanning 8 material
  families.

## Directory Structure

```
umst-prototype-2a/
├── README.md                  # This file
├── LICENSE                    # CC-BY 4.0 License
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

```bash
./prototype/src/rust/target/release/ssot_benchmark
```

For the full reproduction workflow (all experiments, tables, and figures), see
[REPRODUCE.md](REPRODUCE.md).

## Essential Binaries

After building, five binaries are available in `prototype/src/rust/target/release/`:

| Binary                    | Purpose                                      |
|--------------------------|----------------------------------------------|
| `ssot_benchmark`         | Primary material-state tensor benchmark      |
| `epistemic_experiment`   | Epistemic sensing / sensor selection          |
| `veto_experiment`        | Thermodynamic admissibility veto gate         |
| `hardware_heat_experiment` | Hardware-in-the-loop heat validation        |
| `egoff_cli`              | EGoFF composition and analysis CLI            |

## Key Results

Results reproduced by running `ssot_benchmark` across four material domains:

| Domain  | Physics Kernel MAE | Hybrid MAE | TQ    | Admissibility |
|---------|--------------------|------------|-------|---------------|
| UCI     | 4.21 MPa           | 3.87 MPa   | 0.686 | 100%          |
| LUNAR   | 1.83 MPa           | 1.76 MPa   | 0.701 | 100%          |
| UHPC    | 5.44 MPa           | 4.92 MPa   | 0.673 | 100%          |
| HIGHSCM | 6.12 MPa           | 5.61 MPa   | 0.648 | 100%          |

All predictions satisfy 100% thermodynamic admissibility via the
Clausius-Duhem inequality gate. See `prototype/results/` for full canonical
tables and [REPRODUCE.md](REPRODUCE.md) for reproduction steps.

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
   vetoes any prediction violating the second law of thermodynamics, ensuring
   100% admissibility across all outputs.

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

This work is licensed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0). See [LICENSE](LICENSE) for details.
