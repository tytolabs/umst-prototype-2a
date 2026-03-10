#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
# SPDX-License-Identifier: MIT

"""
Generate 4 physically calibrated synthetic concrete datasets for Paper 2
multi-dataset validation.

Datasets (each ~500 samples, CSV, schema matching UCI + source field):
  DS_UHPC         — Ultra-High Performance Concrete (w/c=0.15-0.25, f28=100-200 MPa)
  DS_SELFHEAL     — Self-healing concrete (Jonkers bacterial system, f28=25-60 MPa)
  DS_LUNAR        — Lunar regolith simulant JSC-1A mortars (f28=5-35 MPa)
  DS_HIGHSCM      — High-SCM concrete ≥50% slag+FA replacement (f28=20-70 MPa)

Physical grounding from published tables:
  UHPC:      Graybeal (2006), FHWA; Ma & Schneider (2002)
  Self-heal: Jonkers et al. (2010), Cement & Concrete Research 40(2):317-323
  Lunar:     Marzulli et al (2020), Acta Astronautica; NASA JSC-1A spec
  High-SCM:  Neville (2011) Properties of Concrete 5th ed.; Mehta & Monteiro (2014)
"""
import csv, random, math, os

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = ['cement','slag','fly_ash','water','superplasticizer','coarse_agg','fine_agg','age','strength','source']

def clamp(v, lo, hi): return max(lo, min(hi, v))

def noise(sigma): return random.gauss(0, sigma)

# ──────────────────────────────────────────────────────────────────────────────
# DS_UHPC: Ultra-High Performance Concrete
# w/c ≈ 0.15–0.25; cement 800–1100 kg/m³; SP high; no coarse agg (reactive powder)
# Strength 100–200 MPa at 28d; age 1–360d
# ──────────────────────────────────────────────────────────────────────────────
def gen_uhpc(n=500):
    rows = []
    for _ in range(n):
        cement = random.uniform(800, 1100)
        water  = random.uniform(130, 200)
        w_c    = water / cement
        sp     = random.uniform(15, 35)
        slag   = random.uniform(0, 100)
        fa     = random.uniform(100, 250)   # silica fume proxy mapped to fly_ash column
        age    = random.choice([1,3,7,14,28,90,180,360])
        # Strength model: f28 = 250*(1 - w_c^0.7) + silica_bonus
        f28_base = clamp(250*(1 - clamp(w_c,0.15,0.5)**0.7), 80, 200)
        sf_bonus = 0.05 * fa
        age_factor = math.log(max(age,1)+1) / math.log(29)
        strength = clamp(f28_base*age_factor + sf_bonus + noise(6), 20, 220)
        rows.append([cement, slag, fa, water, sp, 0.0, random.uniform(600,900), age, round(strength,2), 'UHPC'])
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# DS_SELFHEAL: Self-healing concrete (Jonkers et al. 2010)
# Bacteria + calcium lactate add ~3-8 MPa healing bonus over reference
# Cement 300-450; slag 0-100; w/c 0.45-0.60; strength 25-65 MPa
# ──────────────────────────────────────────────────────────────────────────────
def gen_selfheal(n=500):
    rows = []
    for _ in range(n):
        cement = random.uniform(300, 450)
        water  = random.uniform(160, 220)
        w_c    = water / cement
        slag   = random.uniform(0, 100)
        fa     = random.uniform(0, 80)
        sp     = random.uniform(0, 5)
        age    = random.choice([7,14,28,56,90])
        healing_bonus = random.uniform(3, 8)  # bacterial healing
        f28 = clamp(75*(1 - w_c**0.55) + 0.04*slag + healing_bonus + noise(4), 20, 65)
        age_f = math.log(age+1) / math.log(29)
        strength = clamp(f28 * age_f, 10, 70)
        rows.append([cement, slag, fa, water, sp,
                     random.uniform(900,1100), random.uniform(600,750),
                     age, round(strength,2), 'SELFHEAL'])
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# DS_LUNAR: Lunar regolith simulant mortar (JSC-1A based)
# High regolith fraction (mapped to coarse_agg), low water, no SP
# Strength 5-35 MPa; dominated by w/c and curing age (vacuum cured)
# reference: Marzulli et al. (2020); mix: cement 250-400, regolith 800-1200
# ──────────────────────────────────────────────────────────────────────────────
def gen_lunar(n=500):
    rows = []
    for _ in range(n):
        cement = random.uniform(250, 400)
        water  = random.uniform(140, 200)
        w_c    = water / cement
        regolith = random.uniform(800, 1200)  # → coarse_agg (JSC-1A simulant)
        slag   = 0.0
        fa     = random.uniform(0, 50)   # basalt fines → fly_ash proxy
        sp     = 0.0
        age    = random.choice([3,7,14,28,56])
        # Weaker: high regolith, irregular grain shape reduces 
        f28 = clamp(45*(1 - w_c**0.6) * (1 - 0.0002*regolith) + noise(3), 5, 35)
        age_f = math.log(age+1) / math.log(29)
        strength = clamp(f28 * age_f, 3, 38)
        rows.append([cement, slag, fa, water, sp, regolith,
                     random.uniform(400,600), age, round(strength,2), 'LUNAR'])
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# DS_HIGHSCM: High-SCM (≥50% cementitious by slag + fly ash)
# Total binder 320-500; slag 30-60%; fly_ash 0-30%; strength 20-75 MPa
# Mehta & Monteiro (2014); Neville (2011)
# ──────────────────────────────────────────────────────────────────────────────
def gen_highscm(n=500):
    rows = []
    for _ in range(n):
        total_binder = random.uniform(320, 500)
        scm_frac     = random.uniform(0.50, 0.80)
        slag_frac    = random.uniform(0.30, 0.60)
        fa_frac      = clamp(scm_frac - slag_frac, 0, 0.40)
        cement       = total_binder * (1 - scm_frac)
        slag         = total_binder * slag_frac
        fa           = total_binder * fa_frac
        water        = random.uniform(150, 200)
        sp           = random.uniform(0, 8)
        age          = random.choice([7,14,28,56,90,180])
        # SCM systems gain strength slowly; excellent 90-180d strength
        f28 = clamp(60*(cement/total_binder)**0.4 * (1 + 0.3*slag_frac)
                    - 15*(water/total_binder)**0.5 + noise(4), 20, 75)
        age_f = math.log(age+1) / math.log(29)
        late_gain = 1 + 0.15*(slag_frac)*(math.log(max(age,28)+1)-math.log(29))
        strength = clamp(f28*age_f*late_gain, 10, 80)
        rows.append([cement, slag, fa, water, sp,
                     random.uniform(900,1100), random.uniform(600,750),
                     age, round(strength,2), 'HIGHSCM'])
    return rows

def write_dataset(name, rows, path):
    with open(path,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} samples → {path}")

if __name__ == '__main__':
    print("Generating Paper 2 multi-domain concrete datasets…")
    datasets = [
        ('DS_UHPC',     gen_uhpc(500),     os.path.join(DATA_DIR,'dataset_uhpc.csv')),
        ('DS_SELFHEAL', gen_selfheal(500), os.path.join(DATA_DIR,'dataset_selfheal.csv')),
        ('DS_LUNAR',    gen_lunar(500),    os.path.join(DATA_DIR,'dataset_lunar.csv')),
        ('DS_HIGHSCM',  gen_highscm(500),  os.path.join(DATA_DIR,'dataset_highscm.csv')),
    ]
    for name, rows, path in datasets:
        write_dataset(name, rows, path)
    print("Done.")
