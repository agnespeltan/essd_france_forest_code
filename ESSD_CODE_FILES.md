# ESSD code: files and what they do

Inventory of the five code folders that make the ESSD France forest products.
`config.yaml` (local paths), `__pycache__/` and `.claude/` are not listed and are not published.

Run order across folders:

```
essd_growth_curves  →  essd_age_maps
essd_disturbances (00–04)  →  essd_harvest_age  →  essd_disturbances (05, 07)
all of the above  →  essd_package/build_deposit.py  →  Zenodo zips
```

Every folder has the same frame: numbered scripts (`00_…`, `01_…`) are the steps you run, `lib/` holds the code they call, `config.example.yaml` lists the paths to set (copy to `config.yaml`), `requirements.txt` the Python packages, `README.md` the documentation.

---

## 1. `essd_growth_curves/`: AGB–age growth curves (Fig. 2, SI1, SI2)

Chapman–Richards curves of AGB against age per species category (TFV) × large ecological region (GRECO), for the 25th, 50th and 75th AGB percentiles.

### Pipeline scripts

| File | What it does |
|---|---|
| `00_materialize_year.py` | Turns the multi-band disturbance-agent stack into one map of the latest disturbance year (before the reference year). |
| `01_build_efda.py` | AGB statistics by age for young stands, age = years since the EFDA disturbance. |
| `02_build_gami.py` | AGB statistics by age from the GAMI age map, on pixels without an EFDA disturbance. |
| `03_fit_curves.py` | Fits the Chapman–Richards curves on EFDA + GAMI, gap-fills poor fits, writes `growth_curves_agb_greco.csv` (a, b, c, errors, fit scores, t50, t90, **R30**). |
| `04_run_all.py` | Runs 00 → 03 in order. |
| `05_make_figures.py` | Fig. 2, SI1, SI2. |
| `06_compute_variance.py` | η² of AGBmax, R30, t50 between species and regions (one-way and two-way). |
| `07_bootstrap_curves.py` | Bootstrap uncertainty of the curve parameters (SE used in 06). |
| `08_sensitivity_tests.py` | Sensitivity tests: refits the curves with one setting changed at a time (baseline a0 × f grid, tail-trim age / tolerance / patience, R² threshold, bound on b), scores each run against the NFI (AGB trajectories; R30, AGBmax, t50). Sweeps are listed in the config `sensitivity` block. |
| `09_make_sensitivity_figures.py` | SI figures of the sensitivity tests (baseline factor f; all fit settings) and the combined `fit_settings_sensitivity_table.csv`. |

### `lib/`

| File | What it does |
|---|---|
| `__init__.py` | Marks `lib` as the shared helpers of the scripts. |
| `config.py` | Reads and checks `config.yaml`. |
| `paths.py` | Turns the config entries into absolute paths. |
| `constants.py` | Scientific constants of the pipeline. |
| `raster_stats/agent_stack.py` | Agent stack → single-band latest disturbance year (cached). |
| `raster_stats/common.py` | Polygon windows and percentile helpers shared by EFDA and GAMI. |
| `raster_stats/efda.py` | EFDA AGB-by-age statistics per GRECO. |
| `raster_stats/gami.py` | GAMI AGB-by-age statistics per GRECO (EFDA-free pixels). |
| `fit/preprocess.py` | Merges EFDA + GAMI, applies the age filters, the young-stand baseline and the tail truncation. |
| `fit/chapman_richards.py` | The 3-parameter fit (shared b, c across percentiles; ordered a) and derived metrics (t50, t90, R30). |
| `fit/fill_missing.py` | Quality filter (R² threshold) and gap-filling of rejected or missing TFV × GRECO curves. |
| `fit/pipeline.py` | preprocess → fit → gap-fill → CSV. |
| `fit/bootstrap.py` | Bootstrap resampling of the fits. |
| `variance/eta2.py` | One-way η² with Monte Carlo CIs. |
| `variance/twoway.py` | Two-way partition: unique species, unique region, shared, residual. |
| `sensitivity/__init__.py` | Reads the `sensitivity` config block, sweep folders, NFI input paths. |
| `sensitivity/nfi.py` | Loads the NFI AGB tables (per GRECO, per species) and TFV areas; RMSE of the RS curves against the NFI AGB trajectories. |
| `sensitivity/sweep.py` | Refits per run: a0 × f baseline grid, or one setting at a time; tail-trim diagnostic. |
| `sensitivity/metrics.py` | R30, AGBmax, t50 from RS and NFI; national values, Δ% vs the published setting, RMSE / bias per run. |
| `sensitivity/figures.py` | Draws the two SI sensitivity figures. |
| `figures/data.py` | Loads curves, NFI series, area weights and the GRECO shapefile for the figures. |
| `figures/cr.py` | Chapman–Richards formulas and derived metrics used in the figures. |
| `figures/constants.py` | Labels, colours, `RS_CARBON = 0.475`. |
| `figures/fig_growth.py` | Draws Fig. 2, SI1, SI2. |

---

## 2. `essd_age_maps/`: 30 m forest age map (Fig. 3, SI3–SI6)

Inverts the growth curves on the 2020 AGB map; one map per curve percentile (central = 50th, upper = 25th → older, lower = 75th → younger).

### Pipeline scripts

| File | What it does |
|---|---|
| `00_build_greco_weights.py` | Weight rasters per GRECO that fade smoothly (tanh) across region borders. |
| `00b_materialize_disturbance_year.py` | Latest disturbance year, kept only in the 2019–2021 window. |
| `01_build_age_tiles.py` | Inverts AGB → age with each GRECO's curves where that GRECO's weight > 0. |
| `02_blend_weighted_age.py` | Blends the GRECO tiles with the weights into one national map (`forest_age_blend_<label>.tif`). |
| `03_finalize_age_map.py` | Sets age 0 to nodata where AGB is 0, unless the pixel was disturbed in 2019–2021 → `forest_age_<label>.tif`; then builds the figure inputs `forest_age_<label>_hist.csv`, `_5km.tif`, `_cv_5km.tif`. `--all-quantiles` runs central, upper and lower. |
| `04_make_figures.py` | Fig. 3, SI3–SI6. |
| `05_run_all.py` | Runs the whole pipeline. |
| `run_local.sh` | Shell wrapper to run it on a local machine. |

### `lib/`

| File | What it does |
|---|---|
| `config.py` | Reads and checks `config.yaml`. |
| `constants.py` | Nodata values and other shared constants. |
| `naming.py` | Output file names; `AGE_LABEL` maps 50/25/75 → central/upper/lower. |
| `weights/distance_masks.py` | Builds the tanh distance weights. |
| `invert/chapman_richards.py` | Analytic inversion of the Chapman–Richards curve (AGB → age). |
| `invert/polygons.py` | Raster windows around each GRECO. |
| `invert/tiles.py` | Writes the per-GRECO age tiles. |
| `blend/weighted.py` | Weighted blend into the national map. |
| `raster/disturbance_year.py` | Latest disturbance year capped to the keep window. |
| `postprocess/fix_nodata.py` | The age 0 → nodata rule. |
| `postprocess/summaries.py` | Histogram CSV, 5 km median map, 5 km spatial CV map. |
| `figures/data.py` | Loads figure inputs listed in `config.figures`. |
| `figures/constants.py` | Colours, labels, species tables for the figures. |
| `figures/fig_age.py` | Draws Fig. 3, SI3–SI6. |
| `__init__.py` files | Package markers. |

---

## 3. `essd_disturbances/`: disturbance rate, patch size, severity (Fig. 4–5, SI7–SI11)

Per species category (TFV) × ecological region (SER) × agent (harvest / natural), 1999–2024, with SER → GRECO → France gap-filling.

### Pipeline scripts

| File | What it does |
|---|---|
| `00_materialize_latest_year.py` | Latest disturbance year + agent maps from the multi-band agent stack. |
| `01_build_patches.py` | Yearly disturbed area and patch statistics per TFV × SER × agent. |
| `01b_build_forest_area.py` | Forest area per TFV × SER from the TFV map (denominator of the rate). |
| `02_fit_trends.py` | Mann–Kendall + Theil–Sen trends of rate and patch size, **2020 levels, their errors and slope types** → `disturbance_trends.csv`. |
| `03_build_agb_stats.py` | AGB percentiles before (2021–2024) and after (2016–2019) disturbance. |
| `04_compute_severity.py` | Relative and absolute AGB loss (severity) with uncertainties → `disturbance_severity.csv`. |
| `05_make_figures.py` | Joins trends + severity; Fig. 4–5, SI7–SI11; national totals `disturbance_national_uncertainty.csv` (MtC). |
| `06_run_all.py` | Runs 00 → 04 in order. |
| `07_compute_variance.py` | η² (one-way and two-way species vs region) for the disturbance metrics and mean harvest age; the only η² output. |

### `lib/`

| File | What it does |
|---|---|
| `config.py` | Reads and checks `config.yaml`. |
| `constants.py` | Shared constants. |
| `levels.py` | **One rule for the 2020 levels**: fitted line if the trend is significant and physical, otherwise the mean (rate) or median (patch size); slope types significant / zero / null / none. Used by 02, the figures and 07. |
| `raster/agent_stack.py` | Agent stack → latest year + latest agent (cached). |
| `raster/patches.py` | Yearly patch statistics. |
| `raster/forest_area.py` | Forest TFV × SER area census. |
| `raster/agb_stats.py` | AGB percentiles per disturbance-year window. |
| `raster/geometry.py` | Raster windows around each SER polygon. |
| `raster/jobs.py` | Runs SER polygons in parallel. |
| `trends/mann_kendall.py` | Mann–Kendall test, Theil–Sen slope on calendar years, SEs. |
| `trends/pipeline.py` | Trend fitting with the SER → GRECO → France cascade; calls `levels.add_2020_levels`. |
| `severity/agb_loss.py` | Severity definition and uncertainties. |
| `severity/pipeline.py` | Severity with the SER → GRECO → France cascade. |
| `variance/eta2.py` | One-way η², Monte Carlo CIs, permutation p-values. |
| `variance/twoway.py` | Two-way partition species vs region (agent as covariate). |
| `variance/pipeline.py` | Builds the metric table (gap-filled rows masked) and runs η². |
| `figures/prepare_table.py` | Joins trends + severity into the figure table. |
| `figures/fig_disturbances.py` | Draws Fig. 4–5, SI7–SI11 and the national totals. |
| `__init__.py` files | Package markers. |

---

## 4. `essd_harvest_age/`: mean harvest age (Fig. 6, SI12, SI13)

Formerly `essd_rotation`. NFI ages of harvested trees, attached to the disturbance strata.
Harvest age = NFI age (AGE13) + species age correction (`corr_age`, 3–9 yr) + 3 yr.

### Pipeline scripts

| File | What it does |
|---|---|
| `00_build_harvest_ages.py` | Aggregates NFI harvested-tree ages into stratified tables (mean, median, SD, IQR, number of trees). |
| `01_attach_harvest_age.py` | Attaches mean harvest age to the disturbance strata with the SER → GRECO → France cascade. |
| `02_make_figures.py` | Fig. 6, SI12, SI13. |
| `03_run_all.py` | Runs 00 → 01 → 04 → 02. |
| `04_merge_combined_table.py` | Merges trends + harvest age with patch-size fits and severity into the `Data_clean` tables (incl. `mean_harvest_age.csv`). |

### `data/` (lookup tables shipped with the code)

| File | What it does |
|---|---|
| `espar_corr_age.csv` | Age correction per NFI species code (ESPAR): `ess, latin, french, corr_age`. |
| `espar_to_tfv.csv` | NFI species code → TFV species category. |
| `melange_label_to_tfv.csv` | Mixed-stand labels → TFV. |

### `lib/`

| File | What it does |
|---|---|
| `config.py` | Reads and checks `config.yaml`. |
| `constants.py` | Shared constants. |
| `species.py` | Loads the ESPAR → TFV and age-correction lookups. |
| `nfi/harvest_ages.py` | Builds the stratified NFI harvest-age tables. |
| `harvest_age/se.py` | Standard error of the mean harvest age. |
| `harvest_age/attach.py` | Cascade attach onto the trend rows (carries `mean_harvest_age`, `_err`, `harvest_age_sd`, `n_trees`). |
| `harvest_age/merge.py` | The merge used by 04. |
| `figures/fig_harvest_age.py` | Draws Fig. 6, SI12, SI13. |
| `__init__.py` files | Package markers. |

---

## 5. `essd_package/`: Zenodo deposit

| File | What it does |
|---|---|
| `build_deposit.py` | Adds the species and region codes to every published CSV (`TFV_num`, IGN `TFV` code, `TFV_name`, `GRECO_num`, `GRECO`, `SER_num`, `SER`), orders the columns, checks keys and codes, writes the four product zips and `tfv_codes.csv` / `ser_codes.csv`; with `--age-map` rebuilds `age_map.zip` and checks the tifs against the published CRCs. Computes no values. |
