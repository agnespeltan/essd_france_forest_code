# ESSD France forest code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23503540.svg)](https://doi.org/10.5281/zenodo.23503540)

Python pipelines for species- and region-specific **growth curves**, **forest age**, **disturbance** metrics, and **harvest rotation periods** for metropolitan French forests (30 m, reference year 2020). Companion code to Pellissier-Tanon et al., *Earth System Science Data* (submitted).

**Code DOI:** [10.5281/zenodo.23503540](https://doi.org/10.5281/zenodo.23503540) · **GitHub:** https://github.com/agnespeltan/essd_france_forest_code

## Packages

| Package | Role |
|---------|------|
| [`essd_growth_curves`](essd_growth_curves/) | EFDA/GAMI stats → Chapman–Richards AGB–age curves (GRECO × TFV) |
| [`essd_age_maps`](essd_age_maps/) | Invert CR curves → 30 m age map (median / p25 / p75) |
| [`essd_disturbances`](essd_disturbances/) | Rates, patch sizes, AGB-loss severity (TFV × SER × agent) |
| [`essd_rotation`](essd_rotation/) | NFI harvest ages → rotation periods on disturbance strata |

Each package has `config.example.yaml`, `requirements.txt`, and its own README. Copy the example config to `config.yaml` and set local paths (inputs are not redistributed).

## Data (Zenodo)

Published products (CSVs + age GeoTIFFs) are on Zenodo (separate deposit from this code):

- **DOI:** [10.5281/zenodo.23499335](https://doi.org/10.5281/zenodo.23499335)
- Archives: `growth_curves.zip`, `age_map.zip`, `disturbance_rate_patches.zip`, `disturbance_severity.zip`, `rotation_periods.zip`, plus `tfv_codes.csv` / `ser_codes.csv`

**Data license:** CC BY 4.0

## Paper

Pellissier-Tanon, A., Xu, Y., Ritter, F., Besic, N., Ciais, P.  
*Species- and region-specific growth curves, forest age, and disturbance trends for French forests at 30 m resolution (2020).*  
Earth System Science Data (submitted).

## Authors

| Name | ORCID |
|------|-------|
| Agnès Pellissier-Tanon | [0000-0003-0840-0933](https://orcid.org/0000-0003-0840-0933) |
| Yidi Xu | [0000-0003-1528-4256](https://orcid.org/0000-0003-1528-4256) |
| François Ritter | [0000-0001-6123-2145](https://orcid.org/0000-0001-6123-2145) |
| Nikola Besic | [0000-0002-2597-4042](https://orcid.org/0000-0002-2597-4042) |
| Philippe Ciais | [0000-0001-8560-4943](https://orcid.org/0000-0001-8560-4943) |

**Contact:** Agnès Pellissier-Tanon — agnes.pellissier-tanon@lsce.ipsl.fr  
Laboratoire des Sciences du Climat et de l’Environnement (LSCE), France

## License

This repository is released under the [MIT License](LICENSE).  
Redistributed lookup tables under `essd_rotation/data/` are small species-mapping helpers for the pipeline (not NFI microdata).
