# MANIFEST — UMST Prototype 2a

**Paper:** "Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization"
**Generated:** 2026-03-05
**Build verified:** cargo build --release (PASS), ssot_benchmark (PASS — 18,146 samples, 8 datasets, 3 noise levels)
**Total:** 116 files, ~5.8 MB (excluding build artifacts)

## Audit Legend

| Tag | Meaning |
|-----|---------|
| [MATH] | Equations verified against paper |
| [TRUTH] | Numbers match published claims |
| [ATTR] | Attribution and licensing verified |
| [FP] | Functional programming purity verified |
| [I/O] | Data paths and contracts verified |
| [QUALITY] | Code quality checked |

---

## Datasets (8 files)

| File | Size | Rows | Audit | Status |
|------|------|------|-------|--------|
| `prototype/data/dataset_D1.csv` | 71K | 1,030 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_D2.csv` | 320K | 4,891 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_D3.csv` | 184K | 2,780 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_D4.csv` | 475K | 7,445 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_uhpc.csv` | 67K | 500 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_lunar.csv` | 61K | 500 | [TRUTH][FORMAT][I/O] | PASS — range 3-38 MPa matches KNOWN_LIMITATIONS |
| `prototype/data/dataset_selfheal.csv` | 76K | 500 | [TRUTH][FORMAT][I/O] | PASS |
| `prototype/data/dataset_highscm.csv` | 75K | 500 | [TRUTH][FORMAT][I/O] | PASS |

**Total: 18,146 samples (matches paper claim)**
**Column format:** cement,slag,fly_ash,water,superplasticizer,coarse_agg,fine_agg,age,strength,source,temperature,humidity

## Canonical Results (3 files)

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/results/canonical/tables/TABLE3_robustness_cliff.csv` | 1.5K | [TRUTH][MATH] | PASS — D1 Physics MAE, Hybrid MAE |
| `prototype/results/canonical/tables/TABLE3_robustness_cliff.json` | 11K | [TRUTH] | PASS |
| `prototype/results/canonical/tables/meta_trajectory.csv` | 2.5K | [TRUTH] | PASS |

## Rust Source — Pure Layer (28 files)

FP annotation: `pure` = no side effects, deterministic, referentially transparent

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/src/rust/core/src/lib.rs` | 1.4K | [ATTR][FP] | PASS — module declarations |
| `prototype/src/rust/core/src/physics_kernel.rs` | 28K | [MATH][TRUTH][FP][ATTR] | PASS — Avrami, Powers, 15 science engines |
| `prototype/src/rust/core/src/formulas.rs` | 9.0K | [MATH][FP][ATTR] | PASS — strength formulas |
| `prototype/src/rust/core/src/safety.rs` | 3.5K | [FP][ATTR] | PASS — safety constraints |
| `prototype/src/rust/core/src/tests_physics.rs` | 8.5K | [MATH][TRUTH] | PASS — verification tests |
| `prototype/src/rust/core/src/constitution/mod.rs` | 15K | [MATH][FP][ATTR] | PASS — thermodynamic gate, admissibility |
| `prototype/src/rust/core/src/math/mod.rs` | 144B | [FP] | PASS |
| `prototype/src/rust/core/src/math/ekf.rs` | 6.6K | [MATH][FP] | PASS — Extended Kalman Filter |
| `prototype/src/rust/core/src/math/kalman.rs` | 2.8K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/math/ode_solver.rs` | 3.3K | [MATH][FP] | PASS — ODE integration |
| `prototype/src/rust/core/src/math/ols.rs` | 10K | [MATH][FP] | PASS — Ordinary Least Squares |
| `prototype/src/rust/core/src/science/mod.rs` | 455B | [FP] | PASS |
| `prototype/src/rust/core/src/science/rheology.rs` | 48K | [MATH][FP] | PASS — Bingham, Herschel-Bulkley |
| `prototype/src/rust/core/src/science/strength.rs` | 6.7K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/science/thermodynamic_filter.rs` | 14K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/science/maturity.rs` | 2.3K | [MATH][FP] | PASS — Arrhenius maturity |
| `prototype/src/rust/core/src/science/porosity.rs` | 1.8K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/science/chemo_water.rs` | 3.2K | [MATH][FP] | PASS — chemically bound water |
| `prototype/src/rust/core/src/science/colloidal.rs` | 2.8K | [MATH][FP] | PASS — colloidal interactions |
| `prototype/src/rust/core/src/science/color.rs` | 2.1K | [FP] | PASS — color-based sensing |
| `prototype/src/rust/core/src/science/cost.rs` | 1.9K | [FP] | PASS — cost estimation |
| `prototype/src/rust/core/src/science/domain.rs` | 2.4K | [FP] | PASS — domain constraints |
| `prototype/src/rust/core/src/science/fracture.rs` | 3.0K | [MATH][FP] | PASS — fracture mechanics |
| `prototype/src/rust/core/src/science/itz.rs` | 2.5K | [MATH][FP] | PASS — interfacial transition zone |
| `prototype/src/rust/core/src/science/transport.rs` | 2.7K | [MATH][FP] | PASS — transport phenomena |
| `prototype/src/rust/core/src/science/sustainability.rs` | 2.2K | [FP] | PASS — sustainability metrics |
| `prototype/src/rust/core/src/science/materials.rs` | 3.1K | [FP] | PASS — material properties |
| `prototype/src/rust/core/src/science/thermo.rs` | 2.6K | [MATH][FP] | PASS — thermodynamic models |

## Rust Source — Functor Layer (4 files)

FP annotation: `functor` = structure-preserving map between categories

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/src/rust/core/src/sensing/mod.rs` | 127B | [FP] | PASS |
| `prototype/src/rust/core/src/sensing/proxies.rs` | 1.6K | [FP][ATTR] | PASS |
| `prototype/src/rust/core/src/epistemic_proxy_selector.rs` | 22K | [MATH][FP][ATTR] | PASS — MI computation, proxy selection |
| `prototype/src/rust/core/src/tensors/functor.rs` | 3.8K | [FP] | PASS — functorial tensor ops |

## Rust Source — Composition Layer (18 files)

FP annotation: `composition` = combines pure components via functorial composition

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/src/rust/core/src/rl/mod.rs` | 1.8K | [FP] | PASS |
| `prototype/src/rust/core/src/rl/ppo.rs` | 52K | [MATH][FP] | PASS — PPO agent (forward-pass only, no training) |
| `prototype/src/rust/core/src/rl/epistemic.rs` | 21K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/rl/epistemic_ppo.rs` | 21K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/rl/reward.rs` | 15K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/optimization/mod.rs` | 121B | [FP] | PASS |
| `prototype/src/rust/core/src/optimization/topology.rs` | 6.9K | [MATH][FP] | PASS |
| `prototype/src/rust/core/src/optimization/monte_carlo.rs` | 3.5K | [FP] | PASS |
| `prototype/src/rust/core/src/rl/concrete_provider.rs` | 4.1K | [I/O][FP] | PASS — concrete data provider |
| `prototype/src/rust/core/src/rl/environment.rs` | 5.3K | [FP] | PASS — RL environment |
| `prototype/src/rust/core/src/rl/ewc.rs` | 3.8K | [MATH][FP] | PASS — Elastic Weight Consolidation |
| `prototype/src/rust/core/src/rl/explainability.rs` | 4.2K | [FP] | PASS — policy explainability |
| `prototype/src/rust/core/src/rl/federated.rs` | 3.5K | [FP] | PASS — federated learning |
| `prototype/src/rust/core/src/rl/guardrails.rs` | 2.9K | [FP] | PASS — safety guardrails |
| `prototype/src/rust/core/src/rl/liquid_ppo.rs` | 6.1K | [MATH][FP] | PASS — Liquid PPO variant |
| `prototype/src/rust/core/src/rl/quantum_bounds.rs` | 3.4K | [MATH][FP] | PASS — quantum-inspired bounds |
| `prototype/src/rust/core/src/rl/state.rs` | 2.7K | [FP] | PASS — state representation |
| `prototype/src/rust/core/src/rl/traits.rs` | 1.8K | [FP] | PASS — trait definitions |

## Rust Source — Boundary Layer (16 files)

FP annotation: `boundary` = I/O at system edge (only place side effects are permitted)

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/src/rust/core/src/data_provider.rs` | 12K | [I/O][ATTR] | PASS — CSV loading, relative paths |
| `prototype/src/rust/core/src/io/mod.rs` | 102B | [I/O] | PASS |
| `prototype/src/rust/core/src/io/telemetry.rs` | 2.7K | [I/O] | PASS |
| `prototype/src/rust/core/src/ecs/mod.rs` | 396B | [FP] | PASS |
| `prototype/src/rust/core/src/ecs/components.rs` | 4.0K | [FP] | PASS |
| `prototype/src/rust/core/src/ecs/systems.rs` | 2.3K | [FP] | PASS |
| `prototype/src/rust/core/src/ecs/world.rs` | 2.0K | [FP] | PASS |
| `prototype/src/rust/core/src/hardware/mod.rs` | 96B | [I/O] | PASS |
| `prototype/src/rust/core/src/hardware/rapl.rs` | 12K | [I/O] | PASS — RAPL energy measurement |
| `prototype/src/rust/core/src/gpu/mod.rs` | 285B | [I/O] | PASS |
| `prototype/src/rust/core/src/gpu/gnn_layer.rs` | 7.7K | [MATH][I/O] | PASS |
| `prototype/src/rust/core/src/tensors/mod.rs` | 425B | [FP] | PASS |
| `prototype/src/rust/core/src/tensors/geometry.rs` | 3.6K | [MATH][FP] | PASS — geometric tensor ops |
| `prototype/src/rust/core/src/tensors/hyper_graph_tensor.rs` | 5.2K | [MATH][FP] | PASS — hyper-graph tensor |
| `prototype/src/rust/core/src/tensors/mix.rs` | 2.8K | [FP] | PASS — mix design tensors |
| `prototype/src/rust/core/src/tensors/sparse.rs` | 3.1K | [MATH][FP] | PASS — sparse tensor ops |

## Rust Source — Entry Points (22 binaries)

| File | Size | Essential? | Audit | Status |
|------|------|-----------|-------|--------|
| `ssot_benchmark.rs` | 68K | **ESSENTIAL** | [MATH][TRUTH][I/O] | PASS — Tables 2-4, 15 engines, Arrhenius ×2.68 |
| `epistemic_experiment.rs` | 31K | **ESSENTIAL** | [MATH][TRUTH] | PASS — TQ=0.686, d=2.791 |
| `veto_experiment.rs` | 14K | **ESSENTIAL** | [MATH] | PASS — Theorem 2 |
| `hardware_heat_experiment.rs` | 14K | **ESSENTIAL** | [MATH] | PASS — Theorem 3 |
| `egoff_cli.rs` | 2.7K | **ESSENTIAL** | [I/O] | PASS — Constitution gate CLI |
| `constitution_rejection_rate.rs` | 4.2K | USEFUL | [MATH] | PASS |
| `thermodynamic_gate.rs` | 12K | USEFUL | [MATH] | PASS |
| `physics_compute.rs` | 8.1K | USEFUL | [MATH] | PASS |
| `pareto_design_benchmark.rs` | 14K | OPTIONAL | [MATH] | PASS |
| `gate_server.rs` | 50K | OPTIONAL | [I/O] | PASS — infrastructure |
| `ppo_server.rs` | 18K | OPTIONAL | [I/O] | PASS — infrastructure |
| (+ 11 more benchmarks) | | OPTIONAL | [MATH] | PASS |

## Build Files (5 files)

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/src/rust/Cargo.toml` | 160B | [ATTR][I/O] | PASS — workspace, core-only member |
| `prototype/src/rust/Cargo.lock` | — | [I/O] | PASS — pinned dependency versions for reproducibility |
| `prototype/src/rust/core/Cargo.toml` | 2.9K | [ATTR][I/O] | PASS — named authors, license CC-BY-4.0 |
| `prototype/src/rust/core/clippy.toml` | 271B | [QUALITY] | PASS |
| `prototype/src/rust/core/static/telemetry_viewer.html` | 9.4K | [I/O] | PASS — compile-time include |

## Python Scripts (2 files)

| File | Size | Audit | Status |
|------|------|-------|--------|
| `prototype/scripts/gen_paper2_datasets.py` | 8.2K | [I/O][QUALITY][ATTR] | PASS |
| `prototype/scripts/gen_paper2_visuals.py` | 13K | [I/O][QUALITY][ATTR] | PASS |

## Documentation (10 files)

| File | Size | Audit | Status |
|------|------|-------|--------|
| `README.md` | 5.9K | [ATTR][FORMAT] | PASS — package overview, quickstart |
| `MANIFEST.md` | 12.6K | [ATTR][FORMAT] | PASS — this file; complete audit manifest |
| `LICENSE` | 1.4K | [ATTR] | PASS — CC-BY 4.0, named authors |
| `REPRODUCE.md` | 1.7K | [TRUTH][FORMAT] | PASS — build + run instructions |
| `KNOWN_LIMITATIONS.md` | 2.0K | [TRUTH] | PASS — PPO NaN, calibration caveats match code |
| `requirements.txt` | 321B | [QUALITY] | PASS — numpy, matplotlib, pandas, litellm |
| `prototype/docs/1_Architecture.md` | 8.2K | [ATTR][FORMAT] | PASS — FP architecture, science engines, data flow |
| `prototype/docs/2_Datasets.md` | 6.2K | [ATTR][FORMAT] | PASS — 8 datasets, schema, provenance |
| `prototype/docs/3_Evaluation_Protocol.md` | 6.3K | [ATTR][FORMAT] | PASS — metrics, fair comparison protocol |
| `prototype/docs/4_Binaries.md` | 7.0K | [ATTR][FORMAT] | PASS — 22 binaries, build instructions, ports |

---

## Verification Results

| Check | Result |
|-------|--------|
| `cargo build --release` | PASS (warnings only, no errors) |
| `ssot_benchmark` runs from package | PASS (18,146 samples, 3 noise levels) |
| Data paths (`../../../data/`) | PASS (resolves correctly) |
| SPDX and attribution compliance | PASS (CC-BY-4.0 headers on all source files) |
| Total files | 116 |
| Package size | ~5.8 MB |

## FP Architecture Summary

```
pure (math, science, constitution, formulas, physics_kernel)
  ↓ deterministic, no side effects
functor (sensing, epistemic_proxy_selector, tensors/functor)
  ↓ structure-preserving maps
composition (rl, optimization)
  ↓ combines pure + functor layers
boundary (io, data_provider, hardware, gpu)
  ↓ side effects confined here
entry (bin/) → benchmarks and experiments
```
