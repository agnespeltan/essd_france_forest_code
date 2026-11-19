# ESSD growth curves

Build GRECO × TFV AGB–age statistics from 30 m rasters, then fit Chapman–Richards curves.

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
   The joint fit (`least_squares`) is run from 19 starting points: the warm start from a single fit of the median series, plus 18 starts with a(τ) kept away from its upper bound, crossed with a grid of (b, c) (`_extra_starts` in `lib/fit/chapman_richards.py`). The lowest-cost solution is kept. With a single start, the solver could stall where the warm start put a(τ) on its bound; the result then depended on the scipy version (5 strata differed between scipy 1.8 and 1.15). With multi-start the fits agree across versions to about 1e-6.
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
| `t50_q`, `t90_q`, `R30_q`, …  | Derived metrics (R30 = AGB(30)/30) |
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


Area weights: `× 0.09 / 10⁶`. The sensitivity tests (`08`, `09`) need the NFI tables (`nfi_greco_csv`, `nfi_species_csv`) as well as `area_parquet`.

## Layout

```text
00–06_*.py           # pipeline, figures, η²
04_run_all.py        # orchestrator (00–03)
07_bootstrap_curves.py           # bootstrap SE of the curves
08_sensitivity_tests.py          # refits with changed settings, scored vs NFI
09_make_sensitivity_figures.py   # SI sensitivity figures
config.example.yaml
requirements.txt
lib/                 # helpers (not a pip package); lib/sensitivity/ for 08–09
```


## Sensitivity tests (08, 09)

The young-stand baseline factor `age_baseline_factor` (f = 0.85, with a0 = `age_baseline_age` = 2) is an empirical calibration parameter, tuned against the NFI reference curves. The tail truncation (`tail_trim_age` 90 yr, `tail_trim_delta` 0.05, `tail_trim_patience` 1), the quality threshold `r2_min` (0.2) and the lower bound on the rate parameter `cr_b_min` (0.01 yr⁻¹) are fit settings. `08` refits the curves with one of them changed at a time (all other settings from the config) and scores each run against the NFI; `09` draws the SI figures. The runs are listed in the `sensitivity` block of the config:

```yaml
sensitivity:
  out_dir: sensitivity            # under output_dir
  reference_factor: 0.85
  baseline_grid: {ages: [2], factors: [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]}
  settings:
    trim_age_r2: {tail_trim_age: [none, 70, 90, 110, 130], r2_min: [0, 0.1, 0.2, 0.3, 0.5]}
    trim_delta:  {tail_trim_delta: [0.01, 0.02, 0.05, 0.10, 0.20, 0.50], tail_trim_patience: [1, 2, 3]}
    b_min:       {cr_b_min: [0.01, 0.005, 0.002, 0.001, 0.0005]}
```

```bash
python3 08_sensitivity_tests.py                  # all sweeps; --only b_min for one, --force to refit
python3 08_sensitivity_tests.py --score-only     # re-score existing curves without refitting
python3 08_sensitivity_tests.py --trim-diagnostic
python3 09_make_sensitivity_figures.py
```

Outputs, one folder per sweep in `<output_dir>/sensitivity/` (`baseline_factor`, `trim_age_r2`, `trim_delta`, `b_min`):
- one `growth_curves_agb_*.csv` per run;
- `*_vs_nfi_summary.csv`: AGB-trajectory RMSE / bias vs NFI per run, at GRECO and species level;
- `baseline_factor_sensitivity_table.csv`: one row per run, with national area-weighted RS and NFI values of AGBmax, t50 and R30, Δ% vs the published setting, and RMSE / bias vs NFI;
- `metric_by_factor_long.csv`: the same metrics per GRECO and per species.

In `<output_dir>/sensitivity/`: `fit_settings_sensitivity_table.csv` (every run of every sweep) and, with `--trim-diagnostic`, `tail_trim_diagnostic.csv` (points removed by the default trim, per TFV × GRECO × quantile). Figures in `figure_dir`: `Figure_SI_baseline_factor_sensitivity.png` and `Figure_SI_fit_settings_sensitivity.png`.

## Bootstrap uncertainty (07)

```bash
python3 07_bootstrap_curves.py --config config.yaml --n-boot 200 --merge
python3 06_compute_variance.py            # η² CI from bootstrap SE (--mc envelope = former method)
```

For each fitted TFV × GRECO, the age points are resampled, keeping q25, q50 and q75 of one age together. Each sample is refit with `fit_quantiles` (same trim and baseline). Outputs in `output_dir`:
- `growth_curves_agb_greco_bootstrap.csv`: every draw.
- `growth_curves_agb_greco_uncertainty.csv`: per stratum, the fit SE / 95 % CI, the between-stand spread and the total.

`--merge` adds these columns to `growth_curves_agb_greco.csv`. A backup is written first (`*.pre_bootstrap.csv`). See `lib/fit/bootstrap.py` for definitions.

**Two-way partition (06).** `growth_curves_eta2_twoway.csv` splits the variance of AGBmax, R30 and t50 into four parts: unique to species (TFV), unique to region (GRECO), shared, and residual. It uses additive OLS commonality analysis on the same 241 cells, with equal weights, adjusted-R² variants, type II F-tests and the same MC draws. See `lib/variance/twoway.py`. Every MC CI (one-way and two-way) takes the spread of the η² draws and centres it on the point estimate (`recentred_ci` in `lib/variance/twoway.py`). Each draw adds noise to values that already carry their error, which lowers η², so the raw percentiles of the draws would sit below the point.
