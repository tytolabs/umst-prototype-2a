# Architecture Overview

> IROS 2026 Reproducibility Package
> Paper: "Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization"

## 1. System Overview

This prototype implements a physics-constrained material characterization architecture
built around three core abstractions: (i) a Unified Material-State Tensor, (ii) a
Physics Kernel comprising 15 science engines, and (iii) an Epistemic Sensing Functor
for information-theoretic proxy selection. A thermodynamic gate enforces physical
admissibility on all predictions.

The implementation is written in Rust (27,542 LOC) with Python utilities for dataset
generation and visualization.

## 2. Unified Material-State Tensor

The central data representation is a sparse hypergraph embedded in R^64. Each material
sample is encoded as a tensor that captures:

- **Compositional features**: cement, slag, fly ash, water, superplasticizer,
  coarse aggregate, fine aggregate
- **Process parameters**: age, temperature, humidity, curing conditions
- **Derived physics fields**: hydration degree, porosity, gel-space ratio,
  maturity index, rheological parameters
- **Epistemic metadata**: measurement uncertainty, proxy quality scores

The sparse representation enables efficient computation while preserving the
multi-physics coupling structure inherent in cementitious materials.

## 3. Physics Kernel

The kernel comprises 15 science engines, each implementing a constitutive law or
physics model:

| # | Engine | Constitutive Law | Domain |
|---|--------|-----------------|--------|
| 1 | Hydration | Avrami-Parrott kinetics | Cement hydration degree |
| 2 | Strength | Powers' Gel-Space Ratio | Compressive strength |
| 3 | Rheology | Herschel-Bulkley model | Yield stress / viscosity |
| 4 | Maturity | Arrhenius equivalent age | Time-temperature history |
| 5 | Porosity | Powers' porosity model | Pore structure evolution |
| 6 | Transport | Diffusion-based transport | Chloride / moisture ingress |
| 7 | Fracture | Linear elastic fracture mechanics | Crack propagation |
| 8 | Thermodynamic Filter | Clausius-Duhem inequality | Admissibility screening |
| 9 | Shrinkage | Drying shrinkage model | Volume change |
| 10 | Creep | Viscoelastic creep | Time-dependent deformation |
| 11 | Thermal | Heat of hydration | Temperature evolution |
| 12 | Carbonation | CO2 diffusion model | Carbonation depth |
| 13 | Durability | Service life prediction | Long-term performance |
| 14 | Self-Healing | Autogenous healing kinetics | Crack closure |
| 15 | Geopolymer | Alkali-activation model | Alternative binder systems |

### 3.1 Key Constitutive Laws

**Avrami-Parrott Hydration:**

    alpha(t) = alpha_u * (1 - exp(-k * t^n))

where alpha_u is ultimate hydration degree, k is rate constant, and n is the
Avrami exponent governing nucleation-and-growth kinetics.

**Powers' Gel-Space Ratio:**

    X = c_gel * alpha / (c_gel * alpha + w/c)
    sigma_c = sigma_0 * X^m

where X is the gel-space ratio, alpha is hydration degree, w/c is water-cement
ratio, and m is an empirical exponent (typically ~2.6-3.0).

**Herschel-Bulkley Rheology:**

    tau = tau_0 + K * (d_gamma/dt)^n

where tau_0 is yield stress, K is consistency index, and n is flow behavior index.

**Arrhenius Maturity:**

    M(t) = integral_0^t exp(E_a/R * (1/T_ref - 1/T(s))) ds

where E_a is apparent activation energy, R is the gas constant, and T_ref is the
reference temperature.

## 4. Thermodynamic Gate

The thermodynamic gate enforces the Clausius-Duhem inequality on all predictions,
guaranteeing physical admissibility:

    D_int = sigma : epsilon_dot - rho * (psi_dot + s * T_dot) >= 0

where D_int is internal dissipation, sigma is stress, epsilon_dot is strain rate,
psi is Helmholtz free energy, s is entropy, and T is temperature.

**Enforcement mechanism**: Every prediction produced by the physics kernel is passed
through the gate. Predictions violating D_int >= 0 are rejected (vetoed) and flagged.
The architecture achieves 100% thermodynamic admissibility on all tested datasets.

This corresponds to **Theorem 2** in the paper.

## 5. Epistemic Sensing Functor

The epistemic sensing layer computes mutual information between observable proxy
variables and target quantities to enable optimal sensor/proxy selection:

    I(X; Y) = H(X) - H(X|Y)

The **TQ metric** (Thermodynamic Quality score) combines:
- Mutual information between proxy and target
- Physics-consistency score from the thermodynamic gate
- Measurement uncertainty propagation

This corresponds to **Theorem 1** in the paper.

The functor architecture ensures that epistemic computations compose cleanly with
physics kernel outputs through categorical functor laws.

## 6. Functional Programming Architecture

The codebase follows a strict layered functional programming architecture:

```
Layer 0: Pure Functions    — Stateless computations, constitutive laws
Layer 1: Functors          — Lifting pure functions over data structures
Layer 2: Composition       — Engine pipelines, kernel orchestration
Layer 3: Boundary          — I/O, serialization, file system access
Layer 4: Entry             — CLI binaries, main functions
```

**Design invariants:**
- Inner layers never depend on outer layers
- Side effects are confined to Layers 3-4
- All physics computations occur in Layers 0-1
- Composition (Layer 2) orchestrates without introducing state

## 7. Source Code Map

```
src/rust/
├── src/
│   ├── science/           # Layer 0-1: 15 science engines
│   │   ├── hydration.rs
│   │   ├── strength.rs
│   │   ├── rheology.rs
│   │   ├── maturity.rs
│   │   ├── porosity.rs
│   │   ├── transport.rs
│   │   ├── fracture.rs
│   │   ├── thermodynamic.rs
│   │   ├── shrinkage.rs
│   │   ├── creep.rs
│   │   ├── thermal.rs
│   │   ├── carbonation.rs
│   │   ├── durability.rs
│   │   ├── self_healing.rs
│   │   └── geopolymer.rs
│   ├── tensor/            # Unified Material-State Tensor
│   ├── epistemic/         # Epistemic sensing functor, MI, TQ
│   ├── gate/              # Thermodynamic gate (Clausius-Duhem)
│   ├── kernel/            # Layer 2: Composition, orchestration
│   ├── boundary/          # Layer 3: I/O, serialization
│   └── bin/               # Layer 4: 22 CLI binaries
├── Cargo.toml
└── target/release/        # Compiled binaries (after build)
```

## 8. Data Flow

```
Input CSV ──► Parse & Validate ──► Material-State Tensor (R^64)
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │   Physics Kernel     │
                              │   (15 engines)       │
                              └─────────┬───────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │ Thermodynamic Gate   │
                              │ D_int >= 0 ?         │
                              └─────────┬───────────┘
                                   ┌────┴────┐
                                   │         │
                                 PASS      VETO
                                   │         │
                                   ▼         ▼
                              Prediction   Rejected
                                   │
                                   ▼
                              ┌─────────────────────┐
                              │ Epistemic Sensing    │
                              │ (MI, TQ, proxy sel.) │
                              └─────────────────────┘
```
