# ESSD mean harvest ages

NFI harvest ages → cascade-fill onto disturbance trends → figures (Fig. 6, SI12, SI13).

Formerly `essd_rotation` ("rotation period"). Old column names (`rotation_period*`), output keys (`rotation_table`, `trends_with_rotation*`) and file names are still read, so older configs and tables keep working. The bookkeeping run tag letter `o` is also unchanged, to keep existing run names.

## Terms

| Name | Meaning |
|------|---------|
| **ESPAR / Ess** | NFI species code (2-digit) |
| **TFV** | Forest-type code |
| **SER** | Small ecological region |
| **GRECO** | Large ecological region (letter from SER) |
| **mean_harvest_age** | Mean corrected harvest age (years) |
| **mean_harvest_age_err** | SE × finite-sample correction |
| **harvest_age_sd** | Standard deviation of tree harvest ages (sample actually used; donor sample if gap-filled) |
| **n_trees** | Number of harvested trees in that sample |
| **gap_filled** | Rate cascade from trends (`0` SER · `1` GRECO · `2` national) — **preserved** |
| **harvest_age_gap_filled** | `0` matched TFV×SER · `1` TFV×GRECO · `2` national TFV · `3` alias-at-SER |
| **agent** | Mean harvest age only on `harvest`; `natural` → NaN |

## Pipeline

```text
NFI PLACETTE + ARBRE
        │  data/espar_to_tfv.csv · data/espar_corr_age.csv
        ▼
00_build_harvest_ages.py  →  age_coupe_*_NFI.csv
        │  + disturbance trends (essd_disturbances)
        ▼
01_attach_harvest_age.py     →  mean_harvest_age.csv
                             disturbance_trends_with_harvest_age.csv
        ▼
04_merge_combined_table.py →  …_and_size_frequency.csv
                             …_size_frequency_and_severity.csv
        ▼
02_make_figures.py        →  Fig. 6, SI12, SI13
# or: 03_run_all.py
```

```bash
cd essd_harvest_age
python3 -m pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit paths

python3 00_build_harvest_ages.py
python3 01_attach_harvest_age.py        # re-run after updating disturbance trends
python3 04_merge_combined_table.py   # join MLE + severity (Data_clean)
python3 02_make_figures.py
# or: python3 03_run_all.py
```

Python ≥ 3.10; run from this directory.

## Config

See `config.example.yaml`.

| Key | Role |
|-----|------|
| `nfi.placette` / `nfi.arbre` | IFN CSVs |
| `trends_csv` | Disturbance trends to attach onto |
| `ser_shapefile` / `figure_dir` | Figures |
| `min_count` | Use NFI stratum only if `count > min_count` (default **5**) |
| `age_offset` | `age = AGE13 + corr_age + age_offset` (default **3**) |
| `include_mixtures` | Also assign trees to TFV 80 / 90 / 100 |
| `tfv_aliases` | Sparse TFV ← denser source (table below) |

## Species → TFV

| File | Role |
|------|------|
| `data/espar_to_tfv.csv` | Ess → pure + mixture TFVs |
| `data/melange_label_to_tfv.csv` | Mixture labels → 80 / 90 / 100 / 200 |
| `data/espar_corr_age.csv` | Age corrections |

**TFV 200** = NFI pool **80∪90** (not an alias). Other types may alias *onto* 200.

## Method

**00 ages.** Harvested trees between NFI visits (`coupe==1` from VEGET/VEGET5), with `AGE13`, `CIBLE != 0`, `AGE13 < max_age13`. Age = `AGE13 + corr_age + age_offset`. Aggregate mean/median/std/IQR/count at TFV, TFV×GRECO, TFV×SER (and optional SER forest-pooled diagnostic CSV).

**01 attach.**

1. Apply **TFV aliases** on age tables.
2. Cascade (`count > min_count`): **TFV×SER → TFV×GRECO → national TFV** (stop).
3. Alias targets filled at SER → `harvest_age_gap_filled = 3` (matched species at SER stay `0`).
4. Natural agent → NaN harvest age / `harvest_age_gap_filled`.
5. Trends `gap_filled` (rate provenance) is **not** overwritten.

`mean_harvest_age = mean`; SE = `(std/√n)×finite-sample correction`.

| Target ← source | Note |
|-----------------|------|
| 401 ← 100 | open deciduous |
| 402 ← 200 | open conifers |
| 111 ← 100 | mixed deciduous in island |
| 222 ← 200 | mixed conifers in island |
| 310 ← 100 | mainly deciduous |
| 320 ← 200 | mainly conifers |
| 403 ← 401 | open mixed (after 401) |
| 58 ← 200 | mountain/stone pines (sparse NFI) |

**21 / 28** categories have direct NFI ages (`min_count: 5`); **8** aliases cover the rest.

### Coverage (current France run)

**1,874** harvest TFV×SER cells (86 SERs; all with `total_area > 0`); **0** missing after aliases + cascade.

| `harvest_age_gap_filled` | n | % area |
|---------------------|--:|--:|
| 0 matched TFV×SER | 449 | 67.3% |
| 1 GRECO | 653 | 12.9% |
| 2 national | 297 | 2.9% |
| 3 alias-at-SER | 475 | 17.0% |
| **1–3 any fill** | **1,425** | **32.7%** |
| 1–2 geographic only | 950 | 15.7% |

Of geographic fills (1–2): **106** unmatched TFVs (2.9% area), **844** matched TFVs missing at SER (12.8% area). NFI: **450** TFV×SER strata with `count > 5`.

**02 figures.** Harvest only; area-weighted by TFV×SER `total_area`. SI13 ordered by increasing TFV forest area. Manuscript CIs use `1.96×SE`.

| File | Paper |
|------|--------|
| `Figure_6_mean_harvest_age.png` | Fig. 6 |
| `Figure_SI12_relative_uncertainty_mean_harvest_age.png` | Fig. SI12 |
| `Figure_SI13_mean_harvest_age_by_species_category.png` | Fig. SI13 |

## Outputs

| File | Content |
|------|---------|
| `age_coupe_*_NFI.csv` | Stratified harvest ages |
| `mean_harvest_age.csv` | Compact mean harvest age + `harvest_age_gap_filled` |
| `disturbance_trends_with_harvest_age.csv` | Trends + mean harvest age; keeps rate `gap_filled` |
| `disturbance_trends_with_harvest_age_and_size_frequency.csv` | + MLE (`mle_gap_filled`) |
| `disturbance_trends_with_harvest_age_size_frequency_and_severity.csv` | + severity (`severity_gap_filled`) |

η² variance partitioning (incl. mean harvest age): `essd_disturbances/07_compute_variance.py`.

## Layout

```text
00–02_*.py  03_run_all.py
config.example.yaml  requirements.txt
data/   # ESPAR CSVs
lib/
```
