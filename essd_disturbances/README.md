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

**02 trends:** Mann–Kendall + Theil–Sen on annual rates/patch metrics from `01` when enough years (`min_years_mk`); thinner series keep **median-only** (`trend_* = no trend`). Cascade **SER → GRECO → national** with `gap_filled` 0/1/2 uses **all** SERs at the donor level. Gap-fill copies trend/median statistics but **restores SER×TFV `total_area`**. Independent of severity.

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
| `disturbance_agb_stats.parquet` | Before/after AGB quantiles (input to severity) |
| `disturbance_trends.csv` | Mann–Kendall + medians + `gap_filled` + `total_area` |
| `disturbance_severity.csv` | `agb_lost` / `relative_agb_lost` (+ errors) + `gap_filled` |
| `disturbance_trends_for_figures.csv` | Joined trends+severity for figures (no rotation) |
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
07_compute_variance.py  # one-way η²
config.example.yaml
requirements.txt
lib/                    # helpers (not a pip package)
  raster/               # agent stack, patches, AGB stats
  trends/               # Mann–Kendall + gap-fill cascade
  severity/             # conservative AGB loss + cascade
  figures/              # prepare_table + fig_disturbances (self-contained)
  variance/             # η²
```

**07 variance:** equal-weight one-way η² for rate / patch / severity (/ rotation if `--rotation` or `disturbance_trends_with_rotation.csv` exists). Factors: species (TFV), region (SER), agent. Gap-filled cells excluded. MC 95% CI from row SEs (n=1000); permutation p (n=1000). Rotation is harvest-only and skipped for the agent factor.
