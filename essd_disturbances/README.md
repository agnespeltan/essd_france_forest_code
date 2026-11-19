# ESSD disturbances

Disturbance **rates**, **patch sizes**, and **AGB-loss severity** for France at species category (TFV) × small ecological region (SER) × agent. Trends use Mann–Kendall / Theil–Sen on annual patch stats; severity uses a space-for-time AGB quantile contrast. Figures (Fig. 4–5, SI7–SI11) are written by `05_make_figures.py`.

## Terms

| Name | Meaning |
|------|---------|
| **SER** | Sylvo-écorégion (`ser` / `grid`) |
| **GRECO** | Ecological region (`ser // 100` → letter A–K) |
| **TFV** | Forest-type / species-category code |
| **agent** | `natural` (stack codes 1+2) or `harvest` (3) |
| **disturbance_rate** | Disturbed area / forest area (annual fraction) |
| **gap_filled** | `0` SER · `1` GRECO · `2` national |
| **total_area** | SER×TFV forest mask area (Mha); **kept** when copying GRECO/national fits onto leftover SERs |

## Pipeline

```text
agent_multilayer + TFV + ser_shapefile + rasters.agb
        │
        ▼
00_materialize_latest_year.py  →  latest_disturbance_year_le2024_30m.tif
                                    latest_disturbance_agent_le2024_30m.tif
        │
        ├─► 01_build_patches.py     →  disturbance_patches.parquet
        │   01b_build_forest_area.py →  forest_tfv_ser_area.parquet
        │                                    │
        │                                    ▼
        │                              02_fit_trends.py  →  disturbance_trends.csv
        │
        └─► 03_build_agb_stats.py   →  disturbance_agb_stats.parquet
                                             │
                                             ▼
                                       04_compute_severity.py  →  disturbance_severity.csv
                                             │
                                             ▼
                                       05_make_figures.py  →  PNGs (+ joined figure CSV)
# or: 06_run_all.py  (00–04 only)
# then: 07_compute_variance.py
```

```bash
cd essd_disturbances
python3 -m pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit paths

python3 00_materialize_latest_year.py   # once (--force to rebuild year+agent)
python3 01_build_patches.py --workers 4
python3 01b_build_forest_area.py   # TFV mask × SER census (or built on demand by 02)
python3 02_fit_trends.py
python3 03_build_agb_stats.py --workers 4
python3 04_compute_severity.py
python3 05_make_figures.py          # joins trends+severity; writes PNGs
python3 07_compute_variance.py      # one-way η² (SER / TFV / agent)
# or: python3 06_run_all.py --workers 4
# smoke: --max-polygons 2
# CSV only: python3 05_make_figures.py --prepare-only
```

Run scripts from this directory (they import `lib/`). Python ≥ 3.10.

## Config

See `config.example.yaml`.

- **SER polygons:** `ser_shapefile` only (`France_ser.shp`); `codeser` like `A11` → numeric id `111`. No SER raster.
- **Agent stack:** `rasters.agent_multilayer`, `agent_first_year` (band 1), `agent_last_year`, `agent_nodata` (255).
- **Latest maps:** written by `00` (or pin with `rasters.disturbance_year` / `rasters.disturbance_agent`). Required by `03`.
- **Trend years:** `year_start`–`year_end` inclusive (defaults `1999`–`2024`). Figures report **2020** levels from the fitted line (or median when the MK trend is not kept).
- **Severity windows** are inclusive: `before_range` (default `2021-2024`) = pre-disturbance cohort; `after_range` (default `2016-2019`) = post-disturbance cohort.
- **SI11 Kowalski:** optional `kowalski_loss_csv` + `kowalski_grid_shp` (Planet 10 km Europe grid, clipped to France).
- Default TFV exclusions: `0`, `400`, `forest`, `504`, `506`.

## Method

**00 latest year + agent:** one block-wise pass over the agent stack → two co-registered DEFLATE GeoTIFFs (newest→oldest per tile; years `agent_first_year`…`agent_last_year` inclusive): year (`uint16`) and agent at that year (`uint8`). Same approach as `essd_growth_curves`.

**01 patches:** per SER × year × TFV × agent — pixel count, patch count, mean/std patch size (8-connect). Fire (2) collapsed into natural (1). Still reads the multilayer stack (needs every year).

**01b forest area:** census of every SER×TFV with pixels on the TFV mask (independent of disturbance). Used as the trends universe so never-disturbed forest gets structural-zero rows.

**02 trends:** Quiet years (no patches) are filled as `disturbance_rate=0` / `patch_count=0` over `year_start`–`year_end` (patch sizes stay NaN). When the Mann–Kendall trend is not kept, the 2020 level is: **rate = mean** over the 26 zero-filled years, SE `σ/√N × c(N)` (`mean_std_*`; same finite-sample `c(N)` as severity); **patch size = median** over years with patches (quiet years excluded), SE `1.253 σ/√N × c(N)` (`median_std_*`). Mann–Kendall + Theil–Sen are kept only when `n_years_positive ≥ min_years_ser` (default 10); thinner series get `trend_* = no trend` and a null slope. The Theil–Sen slope and Kendall–Theil intercept are computed on calendar years (`sen_slope_xy` in `lib/trends/mann_kendall.py`), not on the position in the series, so the years missing from patch-size series (no patches) are handled correctly; the Mann–Kendall test itself only uses the order of the values. Cascade **SER → GRECO → national** with `gap_filled` 0/1/2 uses **all** SERs at the donor level. In practice, every SER key with at least one disturbed year is fitted at SER level, so trends all have `gap_filled = 0`. Gap-fill copies trend/mean statistics but **restores SER×TFV `total_area` from the mask census**. Mask forest with **no** observed disturbance (either agent) gets **structural zeros** (`mean=0`, SE=0, `n_years_positive=0`). Independent of severity.

**03 AGB stats:** for each configured year window, sample `rasters.agb` on pixels whose **latest** disturbance year (from `00`) falls in that window, using the co-registered latest-agent map; write quantiles + `count` and `year_range`. For the **before** window only, pixels with `AGB == 0` are excluded.

**04 severity:** conservative loss (Mg ha⁻¹ biomass)  
`agb_lost = max(0, q50_before − ½(q05_after + q50_after))`  
(+ SEs). Relative loss = `100 × agb_lost / q50_before` when `q50_before > 0`. SER usable only if both windows exist; GRECO/national fills use **all** SERs at that level. Same `gap_filled` codes. After the AGB-stats universe is filled, any TFV×SER×agent that has a **disturbance rate** (trends) but still no severity is gap-filled the same way (GRECO → national) for both harvest and natural.

**05 figures:** join trends + severity; convert absolute severity to **MgC ha⁻¹** with carbon fraction **0.475**. National / SER / forest-type / TFV aggregates are **area-weighted** with TFV-block SEs; relative trends are `100 × mean(slope) / mean(2020 level)`. SI TFV order uses disturbance `total_area`. SI11 compares rate-weighted harvest+natural severity to Kowalski SER means (`ours − Kowalski`).

## Outputs

| File | Content |
|------|---------|
| `latest_disturbance_year_le2024_30m.tif` | Latest disturbance year map |
| `latest_disturbance_agent_le2024_30m.tif` | Agent at that latest year (1/2/3) |
| `disturbance_patches.parquet` | Annual patch stats (input to trends) |
| `forest_tfv_ser_area.parquet` | TFV mask × SER census (`n_pixels`, `total_area`) |
| `disturbance_agb_stats.parquet` | Before/after AGB quantiles (input to severity) |
| `disturbance_trends.csv` | Mann–Kendall + means + `gap_filled` + `total_area`, plus the published 2020 levels (`disturbance_rate_2020`, `mean_patch_size_2020` in ha, their `_2020_error`) and `slope_type_*`, all from `lib/levels.py` |
| `disturbance_severity.csv` | `agb_lost` / `relative_agb_lost` (+ errors) + `gap_filled` |
| `disturbance_trends_for_figures.csv` | Joined trends+severity for figures (no mean harvest age) |
| `disturbance_national_uncertainty.csv` | National 2020 values (rate, rate trend, patch size, patch trend, severity, AGB loss) with SE and 95 % CI (TFV-block errors), written by `05` |
| `disturbance_eta2_variance.csv` | One-way η² (SER / TFV / agent) + MC CI + permutation p |

**05 PNGs** (`figure_dir`, default `Resultat_France/Figures/ESSD`):

| File | Paper |
|------|--------|
| `Figure_4_disturbance_rate_and_patch_size.png` | Fig. 4 — rates, rate trends, patch size, patch trends |
| `Figure_5_disturbance_severity_and_agb_lost.png` | Fig. 5 — absolute/relative severity, rate×severity, total AGB loss |
| `Figure_SI7_relative_uncertainty_disturbance_rate.png` | Fig. SI7 |
| `Figure_SI8_disturbance_rate_by_species_category.png` | Fig. SI8 |
| `Figure_SI9_relative_uncertainty_severity_and_agb_lost.png` | Fig. SI9 |
| `Figure_SI10_disturbance_severity_by_species_category.png` | Fig. SI10 |
| `Figure_SI11_kowalski_severity_planet_10km.png` | Fig. SI11 — Kowalski SER abs/rel + ours−Kowalski diffs |

## Layout

```text
00–05_*.py              # pipeline + figures
06_run_all.py           # orchestrator (00–04)
07_compute_variance.py  # one-way and two-way η² (the only η² output)
config.example.yaml
requirements.txt
lib/                    # helpers (not a pip package)
  raster/               # agent stack, patches, AGB stats
  trends/               # Mann–Kendall + gap-fill cascade
  severity/             # conservative AGB loss + cascade
  figures/              # prepare_table + fig_disturbances (2020 levels from variance/pipeline.py)
  variance/             # η²
```

**05 figures** need `figure_dir`, `ser_shapefile`, `france_outline_shapefile`, `kowalski_loss_csv`, `kowalski_grid_shp` and `year_start` in the config; nothing is hard-coded. Totals are in MtC.

**07 variance:** equal-weight one-way η² for rate / patch / severity (/ mean harvest age if `--harvest-age` or `disturbance_trends_with_harvest_age.csv` exists). Factors: species (TFV), region (SER), agent. Gap-filled cells excluded. MC 95% CI from row SEs (n=1000); permutation p (n=1000). Mean harvest age is harvest-only and skipped for the agent factor. It also writes `disturbance_eta2_twoway.csv`, which splits each metric's variance into four parts: unique to species (TFV), unique to region (SER), shared, and residual. Agent is a covariate, and the parts are given as % of the variance left after agent. The method is additive OLS commonality analysis, with adjusted-R² variants and type II F-tests; see `lib/variance/twoway.py`. Every MC CI (one-way and two-way) takes the spread of the η² draws and centres it on the point estimate (`recentred_ci` in `lib/variance/twoway.py`). Each draw adds noise to values that already carry their error, which lowers η², so the raw percentiles of the draws would sit below the point.
