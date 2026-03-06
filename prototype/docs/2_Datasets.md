# Dataset Documentation

> IROS 2026 Reproducibility Package
> Paper: "Towards Unified Material-State Tensors: Epistemic Sensing Architecture for Physics-Constrained Material Characterization"

## 1. Overview

This package includes 8 CSV datasets totaling 18,146 samples spanning compressive
strength prediction, non-destructive testing, environmental exposure, and specialty
concrete formulations. All datasets are normalized to a common column schema for
unified processing by the physics kernel.

## 2. Unified Column Schema

All datasets conform to the following schema. Missing or inapplicable columns are
filled with domain-appropriate defaults (typically 0.0).

| Column | Unit | Description |
|--------|------|-------------|
| `cement` | kg/m³ | Portland cement content |
| `slag` | kg/m³ | Ground granulated blast-furnace slag |
| `fly_ash` | kg/m³ | Fly ash content |
| `water` | kg/m³ | Water content |
| `superplasticizer` | kg/m³ | Superplasticizer dosage |
| `coarse_agg` | kg/m³ | Coarse aggregate content |
| `fine_agg` | kg/m³ | Fine aggregate content |
| `age` | days | Specimen age at testing |
| `strength` | MPa | Compressive strength (target variable) |
| `source` | string | Dataset identifier tag |
| `temperature` | °C | Ambient or curing temperature |
| `humidity` | % RH | Relative humidity during curing |

## 3. Dataset Descriptions

### D1: UCI Concrete Compressive Strength

- **Samples**: 1,030
- **Source tag**: `uci`
- **Description**: Canonical benchmark dataset for concrete compressive strength
  prediction. Contains 8 input variables (cement, slag, fly ash, water,
  superplasticizer, coarse aggregate, fine aggregate, age) and compressive
  strength as the target.
- **Strength range**: ~2-82 MPa
- **License**: UCI Machine Learning Repository (public domain for research)
- **Notes**: Widely used baseline in civil engineering ML literature. All columns
  populated.

### D2: Zenodo NDT

- **Samples**: 4,891
- **Source tag**: `zenodo_ndt`
- **Description**: Non-destructive testing dataset containing ultrasonic pulse
  velocity, rebound hammer, and core strength measurements.
- **License**: CC-BY 4.0
- **Notes**: The `superplasticizer` column is 0.0 for 100% of samples (not
  recorded in original data). Temperature and humidity fields derived from
  test conditions.

### D3: Zenodo Sun

- **Samples**: 2,780
- **Source tag**: `zenodo_sun`
- **Description**: Solar reflectance and thermal mass dataset capturing
  environmental exposure effects on concrete properties.
- **License**: CC-BY 4.0
- **Notes**: The `superplasticizer` column is 0.0 for 100% of samples.
  Temperature data reflects solar exposure conditions.

### D4: Zenodo RH

- **Samples**: 7,445
- **Source tag**: `zenodo_rh`
- **Description**: Relative humidity and curing conditions dataset. Largest
  single dataset in the package, providing extensive coverage of
  moisture-dependent strength development.
- **License**: CC-BY 4.0
- **Notes**: The `superplasticizer` column is 0.0 for 100% of samples.
  Rich humidity data enables transport and shrinkage engine validation.

### D5: UHPC (Ultra-High Performance Concrete)

- **Samples**: 500
- **Source tag**: `uhpc`
- **Description**: Synthetically generated dataset representing ultra-high
  performance concrete formulations with strength values typically exceeding
  120 MPa. Generated using physics-informed sampling of realistic mix
  proportions and constitutive models.
- **Strength range**: ~100-200 MPa
- **Notes**: Tests physics kernel extrapolation to high-strength domain.

### D6: LUNAR (Geopolymer)

- **Samples**: 500
- **Source tag**: `lunar`
- **Description**: Geopolymer concrete dataset representing alkali-activated
  binder systems. Synthetically generated with strength range 3-38 MPa
  to test the geopolymer science engine.
- **Strength range**: 3-38 MPa
- **Notes**: Zero Portland cement content. Exercises the geopolymer
  engine pathway.

### D7: SELFHEAL (Self-Healing Concrete)

- **Samples**: 500
- **Source tag**: `selfheal`
- **Description**: Self-healing concrete formulations incorporating
  crystalline admixtures or bacterial agents. Synthetically generated
  to validate the self-healing science engine.
- **Notes**: Includes healing-agent dosage encoded in auxiliary columns.

### D8: HIGHSCM (High Supplementary Cementitious Materials)

- **Samples**: 500
- **Source tag**: `highscm`
- **Description**: High SCM replacement concrete (>50% cement replacement
  with slag, fly ash, or silica fume). Synthetically generated to test
  kernel behavior at extreme SCM ratios.
- **Notes**: Low cement content with correspondingly high slag/fly_ash
  values. Tests hydration model under high-replacement conditions.

## 4. Dataset Summary Table

| ID | Name | Samples | Source | Synthetic | SP Data |
|----|------|---------|--------|-----------|---------|
| D1 | UCI Concrete | 1,030 | Public repository | No | Yes |
| D2 | Zenodo NDT | 4,891 | CC-BY 4.0 | No | No (all 0) |
| D3 | Zenodo Sun | 2,780 | CC-BY 4.0 | No | No (all 0) |
| D4 | Zenodo RH | 7,445 | CC-BY 4.0 | No | No (all 0) |
| D5 | UHPC | 500 | Generated | Yes | Yes |
| D6 | LUNAR | 500 | Generated | Yes | Yes |
| D7 | SELFHEAL | 500 | Generated | Yes | Yes |
| D8 | HIGHSCM | 500 | Generated | Yes | Yes |
| | **Total** | **18,146** | | | |

## 5. Data Integrity Notes

- All CSV files use UTF-8 encoding with comma delimiters
- No missing values in primary columns (cement through strength)
- Temperature and humidity columns may contain default values (20°C, 60% RH)
  where original data did not record environmental conditions
- The `source` column enables per-dataset filtering in all analysis scripts
- Synthetic datasets (D5-D8) were generated using physics-informed sampling
  to cover domain regions underrepresented in public datasets

## 6. Reproducing Dataset Generation

Synthetic datasets can be regenerated using the Python scripts in `src/python/`:

```bash
cd src/python
python generate_datasets.py --output ../../data/
```

This produces deterministic output given the fixed random seeds specified in the
generation scripts. Public datasets (D1-D4) must be obtained from their original
sources and placed in the `data/` directory.
