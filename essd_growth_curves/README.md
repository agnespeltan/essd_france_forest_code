# ESSD growth curves

Build GRECO × TFV AGB–age statistics from 30 m rasters, then fit Chapman–Richards curves.

## Data

Published curves (`growth_curves_agb_greco.csv`) are in the companion Zenodo data deposit (**DOI pending**). See the [repository root README](../README.md).

## Terms


| Name               | Meaning                                                             |
| ------------------ | ------------------------------------------------------------------- |
| **EFDA**           | Stand age from latest disturbance year before `reference_year`      |
| **GAMI**           | Mean age map for older stands                                       |
| **EFDA-free**      | No EFDA disturbance before `reference_year` (GAMI mask)             |
| **GRECO**          | Ecological region (`GRECO` → stats column `grid`)                   |
| **TFV**            | Forest-type code                                                    |
| **reference_year** | AGB epoch (default 2020); age = `reference_year −` disturbance year |


## Pipeline

```text
agent multilayer
      │
      ▼
00_materialize_year.py  →  latest_disturbance_year_lt2020_30m.tif
      ├─► 01_build_efda.py  →  growth_stats_efda_greco.parquet
      └─► 02_build_gami.py  →  growth_stats_gami_greco.parquet
                │
                ▼
        03_fit_curves.py  →  growth_curves_agb_greco.csv
                │
        ┌───────┴────────┐
        ▼                ▼
05_make_figures.py   06_compute_variance.py
  Fig. 2 / SI1 / SI2   η² CSV (+ text summary)
# or: 04_run_all.py  (00–03 only)
```

```bash
cd essd_growth_curves
python3 -m pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit paths

python3 00_materialize_year.py       # once (--force to rebuild)
python3 01_build_efda.py --workers 4
python3 02_build_gami.py --workers 4
python3 03_fit_curves.py --workers 4
python3 05_make_figures.py           # Fig. 2 / SI1 / SI2
python3 06_compute_variance.py       # η² equal-weight TFV×GRECO; gap-fills excluded

# or: python3 04_run_all.py --workers 4
```

Run scripts from this directory (they import `lib/`). Python ≥ 3.10.  
If stats parquets are already available, only `03_fit_curves.py` is needed.

## Config

Paths and knobs live in `config.yaml` (see `config.example.yaml`).


| Key                                        | Role                                                            |
| ------------------------------------------ | --------------------------------------------------------------- |
| `features_root`                            | 30 m raster root                                                |
| `greco_shapefile`                          | GRECO polygons                                                  |
| `output_dir`                               | Stats parquets, year cache, curve CSVs                          |
| `figure_dir`                               | PNG output for `05`                                             |
| `figures.area_parquet`                     | TFV×GRECO pixel counts (gap-fill in `03`, weights in `05`/`06`) |
| `rasters.agent_multilayer`                 | Multi-band disturbance agent (→ year map)                       |
| `rasters.gami_age_mean`                    | GAMI age                                                        |
| `rasters.agb`                              | AGB target                                                      |
| `reference_year`                           | AGB epoch / year-map cutoff (default 2020)                      |
| `age_baseline_age` / `age_baseline_factor` | EFDA young-stand offset (default 2 / 0.85)                      |
| `r2_min`                                   | Keep local fits if \min(R^2_{25,50,75}) > this (default 0.2)    |
| `min_age_gami`                             | Keep GAMI ages above this (default 35)                          |
| `workers`                                  | Parallel jobs (default 4)                                       |
| `excluded_tfv`                             | Skip these TFVs (default `0`, `400`, `forest`, `504`, `506`)    |




## Inputs

All rasters must already be on a common **30 m** grid **aligned with AGB** (same CRS/extent/pixels). This code does not warp.

Required: TFV, agent multilayer, GAMI age, AGB, GRECO shapefile.

## Method

**Year map (**`00`**):** reduce agent bands to one DEFLATE `uint16` GeoTIFF — latest year `< reference_year` (newest→oldest per tile). Used by EFDA and GAMI.

**EFDA (**`01`**):** window year map + AGB by GRECO × TFV; percentiles 5…95.

**GAMI (**`02`**):** GAMI age > 0 and year-map nodata (EFDA-free), same strata.

**Fit (**`03`**):**

1. Merge EFDA (`age ≥ age_baseline_age` after subtracting `f × AGB(a₀)`) with GAMI (`age > min_age_gami`).
2. Per TFV × GRECO, fit quantiles 25/50/75:

   \mathrm{AGB}_\tau(\mathrm{age}) = a(\tau)(1 - e^{-b\mathrm{age}})^{c}
   
   shared (b,c), monotone a(\tau), unweighted; trim decreasing tails after age ≈ 90.
3. Keep fits with \min(R^2_{25},R^2_{50},R^2_{75}) > 0.2. Gap-fill TFV×GRECO with area > 0: same GRECO group (max-area donor), else climate map J/K→F and B/F→G (`gap_filled=1`, `fill_source`, `fill_level`).



## Outputs

`growth_stats_*_greco.parquet` — one row per GRECO × TFV × age bin:


| Column        | Meaning                                  |
| ------------- | ---------------------------------------- |
| `grid`        | GRECO id                                 |
| `TFV`         | Forest type                              |
| `year`        | EFDA: disturbance year · GAMI: stand age |
| `biomasseM_*` | AGB (`_50th`, `_count`, …)               |


`growth_curves_agb_greco.csv` — one row per TFV × GRECO:


| Column                       | Meaning                            |
| ---------------------------- | ---------------------------------- |
| `a_q`, `b_q`, `c_q`          | CR params for quantile q           |
| `R2_q`, `RMSE_q`, …          | Diagnostics                        |
| `ymax_q`, `t50_q`, …         | Derived metrics                    |
| `gap_filled`                 | `0` fitted · `1` gap-filled        |
| `fill_source` / `fill_level` | Donor GRECO · `group` or `climate` |


**05 figures:** plots via `lib/figures/fig_growth.py`. PNGs go to `figure_dir`:


| File                                                 | Paper    |
| ---------------------------------------------------- | -------- |
| `Figure_2_agb_growth_species_regions_metrics.png`    | Fig. 2   |
| `Figure_SI1_growth_metrics_by_species_class.png`     | Fig. SI1 |
| `Figure_SI2_growth_metrics_by_ecological_region.png` | Fig. SI2 |


**06 variance:** `growth_curves_eta2_variance.csv` in `output_dir` — equal-weight one-way η² (species TFV vs GRECO) for AGBmax / R₃₀ / t₅₀; gap-filled excluded; 95% CI by MC perturbation within each 25–75th percentile range (n=1000) + label-permutation p (n=1000). Use `--include-gap-filled` to keep fills.

Extra inputs under `figures:` (no hardcoded paths):


| Key                          | Required? | Role                                                                                    |
| ---------------------------- | --------- | --------------------------------------------------------------------------------------- |
| `area_parquet`               | **yes**   | Gap-fill (`03`) + figure weights (`05`); 30 m counts → Mha (η² in `06` is equal-weight) |
| `nfi_*` / `robinson_parquet` | optional  | Fig. 2 / SI2 overlays; omit → RS-only                                                   |


Area weights: `× 0.09 / 10⁶`. Scripts under `scripts/` (`*_vs_nfi.py`) need local NFI access and are not part of the public figure path.

## Layout

```text
00–06_*.py           # pipeline, figures, η²
04_run_all.py        # orchestrator (00–03)
config.example.yaml
requirements.txt
lib/                 # helpers (not a pip package)
scripts/             # optional NFI sweeps (not required to publish)
```

