# ESSD age maps

National forest age from AGB via Chapman–Richards inversion, GRECO soft blending, and paper figures.

## Data

Published age GeoTIFFs (`forest_age_q50/q25/q75.tif`) are in the companion Zenodo data deposit (**DOI pending**). See the [repository root README](../README.md).

## Terms

| Name | Meaning |
|------|---------|
| **GRECO** | Ecological region (letters A–K) |
| **TFV** | Forest-type code (BD Forêt) |
| **CR** | Chapman–Richards growth curve |
| **weights** | Distance/tanh soft membership per GRECO (`0…1000`) |
| **tile** | Age map using one GRECO’s CR params, only where that GRECO’s weight > 0 |
| **weighted age** | `sum(age_i · w_i) / sum(w_i)` |
| **fixed** | age==0 → nodata where AGB==0 and disturbance year ∉ {2019,2020,2021} |

## Pipeline

```text
AGB + TFV + GRECO shp + growth_curves CSV + agent stack
        │
        ├─► 00_build_greco_weights.py           →  greco_{A..K}_weights.tif
        ├─► 00b_materialize_disturbance_year.py →  latest_disturbance_year_le2021_30m.tif
        │                                         (years ≤ 2021; auto-run by 03 if missing)
        └─► 01_build_age_tiles.py               →  Age_tiles/forest_age_greco_{id}_q{q}.tif
                 │
                 ▼
           02_blend_weighted_age.py             →  forest_age_blend_q{q}.tif
                 │
                 ▼
           03_fix_nodata.py                   →  forest_age_q{q}.tif
                 │   (keep age==0 if disturbance year ∈ {2019,2020,2021})
                 ▼
           03b_build_summaries.py               →  forest_age_q{q}_hist.csv + _5km.tif
                                                    + _cv_5km.tif (spatial CV)
                 ▼
           04_make_figures.py                   →  Fig. 3, SI3–SI6 PNGs
# or: 05_run_all.py  (00→03; optional --with-figures → 04)
#     03b is separate — run before 04 when hists / 5 km / CV need refresh
```

### Local run

All GRECO tiles run **one after another**; **01** only fills pixels where that GRECO’s weight > 0.

```bash
cd essd_age_maps
python3 -m pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit paths (a local config.yaml may already exist)

# Full compute pipeline (reuses existing GRECO weights by default):
./run_local.sh
# same as:
python3 05_run_all.py --skip-weights

# Include figures:
WITH_FIGURES=1 ./run_local.sh
# or: python3 05_run_all.py --skip-weights --with-figures

# Smoke (one GRECO, few blocks):
python3 05_run_all.py --skip-weights --greco A --max-blocks 20

# Step by step:
python3 00_build_greco_weights.py
python3 00b_materialize_disturbance_year.py   # optional; 03 auto-builds if missing
python3 01_build_age_tiles.py                 # all GRECOs sequentially
python3 02_blend_weighted_age.py
python3 03_fix_nodata.py
python3 03b_build_summaries.py                # hist + 5 km median + spatial CV
python3 04_make_figures.py
```

Run scripts from this directory (they import `lib/`). Python ≥ 3.10.

## Config

See `config.example.yaml`. Same roots as the other `essd_*` packages:

| Key | Typical value |
|-----|----------------|
| `features_root` | `…/Resultat_France/Features_30m` |
| `output_dir` | `…/Resultat_France/Data_clean` |
| `figure_dir` | `…/Resultat_France/Figures/ESSD` |

- Rasters (`agb`, `tfv`) are relative to `features_root`; `disturbance_year` to `output_dir`.
- Weights / age tiles / weighted / fixed → `output_dir` (`GRECO/`, `Age_tiles/`, …).
- Growth curves default: `output_dir/growth_curves_agb_greco.csv` (from `essd_growth_curves`).
- Product flags (`quantile`, `not_corrected`, …) set output filenames.
- `max_distance_km` soft edges (default `30`).
- **Figures:** paths under `figures:` (required for `04`).

## Method

**00 weights:** signed distance from GRECO polygon edges → `tanh` soft weights scaled to integers `[0, 1000]` (`nodata=-9999`).

**01 tiles:** requires `greco_{id}_weights.tif`. Block-wise invert only where **weight > 0** (soft edge), using that GRECO’s CR params (`a_q`, `b_q`, `c_q`):  
`age = −ln(1 − (AGB/a)^(1/c)) / b`. TFV `0→100`, `400→401`; mask `≥500`; fallback TFV 100/200. Elsewhere nodata.

**02 blend:** tiled weighted mean of all GRECO tiles × weights.

**00b disturbance year:** from the agent multilayer, latest year with **year ≤ 2021** → `latest_disturbance_year_le2021_30m.tif` (also: `essd_disturbances/00_materialize_latest_year.py --max-year 2021`).

**03 fix:** force age 0 → nodata where AGB is 0 and latest disturbance year is **not** in `disturbance_keep_years` (default `{2019, 2020, 2021}`); preserves those recent clearcuts. Auto-builds the le2021 year map if missing.

**03b summaries:** from each fixed age map → `forest_age_q{q}_hist.csv`, median `*_5km.tif`, and spatial CV `*_cv_5km.tif` (σ/μ of 30 m ages > 0 inside each 5 km cell).

**04 figures:** self-contained plots adapted from `Figures/Fig_age_distribution.py` (does not call that script). Map panels mask 5 km cells with BD Forêt forest fraction ≤ 10% (`figures.tree_cover`); age maps keep age==0.

| PNG | Paper |
|-----|-------|
| `Figure_3_forest_age_map.png` | Fig. 3 (a KDE, b 5 km map, c composition×region bubbles) |
| `Figure_SI3_age_uncertainty_cv_gami_diff.png` | SI3 (rel. unc, CV, Ours−GAMI @ 5 km) |
| `Figure_SI4_age_kde_by_species.png` | SI4 |
| `Figure_SI5_age_kde_by_greco.png` | SI5 |
| `Figure_SI6_age_bubbles_species_gami_diff.png` | SI6 |

SI3 needs `figures.age_cv_5km` (from 03b) and `figures.ours_minus_gami_5km`. NFI CSVs are optional (`nfi_aires_age`, `nfi_surf_esspre`, `nfi_surf_greco`) — omit → Fig. 3a without NFI; SI4/SI5 skipped. SI4/SI5 GAMI KDEs need `gami_hist_by_tfv` / `gami_hist_by_greco` (optional `_lo`/`_hi`). SI6 panel (b) needs `figures.ours_minus_gami_tfv_csv` (CSV stats or diff-hist NPZ).

## Layout

```text
essd_age_maps/
  00_… 05_run_all.py
  run_local.sh
  config.example.yaml
  lib/weights|invert|blend|postprocess|figures/
```

Helpers live under `lib/` (not a pip package).

## Performance

| Knob | Effect |
|------|--------|
| `--greco` | Run a single GRECO tile (smoke / debug) |
| `block_size` | 01 invert window size (default 2048) |
| `02` blend | Keeps all age + weight GeoTIFFs open across spatial tiles |

**01** only computes where `greco_*_weights > 0` — run **00** first (or point `outputs.weights_dir` at existing weights).
