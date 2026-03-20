# Reproducibility audit — UMST Prototype 2a

**Date:** 2026-03-19  
**Scope:** Determinism, hardware-run validity, generated/stale artifacts, doc ↔ result consistency.

Evidence in this file references commands run on the maintainer machine (macOS, darwin 25.x) unless noted.

---

## Checklist (PASS / FAIL)

| Item | Status | Evidence / path |
|------|--------|-----------------|
| `ssot_benchmark` CWD contract documented | **PASS** | `README.md`, `REPRODUCE.md`, `prototype/src/rust/core/src/bin/ssot_benchmark.rs` (`../../../data/`, `../../../results/...`). |
| `README` quickstart matches CWD requirement | **PASS** | Fixed: `README.md` Step 3 uses `cd prototype/src/rust/core && …`. |
| TABLE3 deterministic (two consecutive runs) | **PASS** | `diff` of `prototype/results/canonical/tables/TABLE3_robustness_cliff.csv` after two `cargo run --release --bin ssot_benchmark` from `prototype/src/rust/core` — identical. |
| `MASTER_RESULTS.md` Group 1 tables vs TABLE3 JSON | **PASS** | §1.1–1.3 and §1.4 bullets updated to match `prototype/results/canonical/tables/TABLE3_robustness_cliff.json` / CSV (2026-03-19). |
| 18,146 vs 22,344 sample claims reconciled in manifest | **PASS** | `MANIFEST.md` explicit reconciliation note. |
| Strict hardware mode (no silent proxy) | **PASS** | `UMST_HARDWARE_STRICT`: `hardware_heat_experiment` exits `1` when no PMU/RAPL; use `env UMST_HARDWARE_STRICT=1` for reliable env passing. |
| Linux RAPL integrated totals (not “Linux ✅” + proxy) | **PASS** (code) | `hardware_heat_experiment.rs`: RAPL Δenergy per phase when sysfs readable; **FAIL** on machines without RAPL (expected). |
| `thermal_proof.csv` location stable | **FAIL** | Written to **process CWD** only (`hardware_heat_experiment.rs`); not pinned to `prototype/results/`. Mitigation: document in `REPRODUCE.md`. |
| `prototype/src/rust/target/` in VCS | **PASS** | `.gitignore` lists `prototype/src/rust/target/`; treat as local-only build output. |
| `fair_comparison_2026-02-25.json` / stale SSOT | **FAIL** (known) | `MASTER_RESULTS.md` §1.5: file still NaN — do not use; **human** should delete or quarantine under `archive/` if desired. |
| Phase T “paper-only” Neural ODE raw JSON | **FAIL** (known) | `MASTER_RESULTS.md` §T.4: v6 adjoint / some Landauer rows lack independent raw JSON. |
| LLM / multi-agent benchmarks re-runnable without API keys | **FAIL** (expected) | Agent JSON under `prototype/results/agents/` is evidence; full rerun needs external models and credentials. |
| `cargo build --release` | **PASS** | `cargo build --release -p umst-core` completed with warnings only (2026-03-19). |

---

## Commands already executed (concrete)

```text
# Build
cd prototype/src/rust && cargo build --release -p umst-core
# → Finished release profile (warnings only)

# TABLE3 / determinism
cd prototype/src/rust/core && cargo run --release --bin ssot_benchmark
# (twice; ~55s each)
# → TABLE3 CSV/JSON unchanged between runs (byte-identical to prior committed CSV in this workspace)

# Strict hardware guard (no sudo, macOS)
env UMST_HARDWARE_STRICT=1 prototype/src/rust/target/release/hardware_heat_experiment
# → exit code 1, stderr explains powermetrics/RAPL requirement
```

---

## Suggested next commands (human / other hosts)

1. **macOS hardware-valid thermal run:**  
   `cd <repo> && sudo env UMST_HARDWARE_STRICT=1 ./prototype/src/rust/target/release/hardware_heat_experiment`  
   (Requires interactive sudo.)

2. **Linux strict thermal run:**  
   `env UMST_HARDWARE_STRICT=1 ./prototype/src/rust/target/release/hardware_heat_experiment`  
   (Requires readable RAPL `energy_uj`; may need root depending on sysfs permissions.)

3. **Regenerate TABLE3 after code/data changes:**  
   `cd prototype/src/rust/core && cargo run --release --bin ssot_benchmark`  
   Then diff `prototype/results/canonical/tables/TABLE3_robustness_cliff.*` and update `MASTER_RESULTS.md` Group 1 if values shift.

4. **Optional cleanup:** Move or remove stale `prototype/results/ssot/fair_comparison_2026-02-25.json` after confirming nothing imports it (grep first).

---

## Files touched in this audit pass

- `README.md` — CWD-correct benchmark invocation.
- `MANIFEST.md` — sample-count reconciliation (18,146 vs 22,344).
- `REPRODUCE.md` — strict hardware env, `env` prefix, `thermal_proof.csv` CWD note.
- `KNOWN_LIMITATIONS.md` — strict mode + Linux RAPL behavior.
- `prototype/results/MASTER_RESULTS.md` — Group 1 tables aligned to TABLE3.
- `prototype/src/rust/core/src/bin/hardware_heat_experiment.rs` — `UMST_HARDWARE_STRICT`, Linux RAPL Δenergy, accurate banners.
- `REPRODUCIBILITY_AUDIT.md` — this checklist.
