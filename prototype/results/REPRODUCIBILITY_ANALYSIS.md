# TABLE3 Reproducibility Analysis — Root Cause & Fix

## Summary

**Verdict: PASS** (after fixes)

Two consecutive runs from `prototype/src/rust/core` now produce identical TABLE3 JSON and CSV outputs.

## Root Cause

Reproducibility drift was caused by **unseeded randomness** in the PPO Agent path:

| Component | Source | Evidence |
|-----------|--------|----------|
| **PPO Agent `select_action`** | `rand::thread_rng()` in `rand_normal()` | `ppo.rs:1346` — Box-Muller uses thread-local RNG |
| **GNNNetwork init** | `rand::thread_rng()` in `GNNNetwork::new()` | `ppo.rs:259-263` — Xavier init uses thread_rng |
| **RandomForest** | `Default::default()` → `seed: 0` | smartcore RF; seed 0 may vary across platforms |
| **CWD dependency** | Relative paths `../../../data/` | Paths resolve from CWD; wrong CWD → empty data → panic |

**Admissibility** values (100% for Physics/Hybrid/Agent, 36–38% for UHPC XGBoost) were **already deterministic** — they depend only on `check_admissibility()` and the thermodynamic filter, which are pure functions of record + calibration. The drift was in **Agent_MAE** (GNN-PPO column) and minor MAE differences from RF non-determinism.

## Fixes Applied

1. **PPOConfig::with_seed(42)** — Seeds GNN init and action sampling when `config.seed` is `Some`.
2. **GNNNetwork::new_with_seed()** — Deterministic Xavier init using `SmallRng::seed_from_u64(seed)`.
3. **PPOAgent** — Stores `Option<RefCell<SmallRng>>`; when seeded, `select_action` uses `rand_normal_seeded(&mut rng)`.
4. **RandomForest** — `RandomForestRegressorParameters::default().with_seed(42)` in `fit_rf()`.
5. **REPRODUCE.md** — Correct run instructions (must run from `prototype/src/rust/core`).
6. **Path error message** — Panic now shows CWD and expected command when data load fails.

## Files Modified

- `prototype/src/rust/core/src/rl/ppo.rs` — PPOConfig seed, GNNNetwork::new_with_seed, PPOAgent rng, rand_normal_seeded
- `prototype/src/rust/core/src/bin/ssot_benchmark.rs` — PPOConfig::with_seed(42), RF with_seed(42), improved panic
- `REPRODUCE.md` — Run instructions, reproducibility section

## Validation

```bash
cd prototype/src/rust/core
cargo run --release --bin ssot_benchmark  # run 1
# save TABLE3_robustness_cliff.json/csv
cargo run --release --bin ssot_benchmark  # run 2
diff run1.json run2.json  # IDENTICAL
diff run1.csv run2.csv    # IDENTICAL
```
