"""Disturbance combined figures (self-contained; used by 05_make_figures.py).

Writes Fig. 4–5, SI7–SI10, and SI11 (Kowalski severity maps; no exploitability / rotation).
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib as mpl
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds

from matplotlib.colors import TwoSlopeNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
import re
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory

# Interactive mode is set in run_figures(interactive=...); default is off / Agg.

# Solid bar-plot colors
NATURAL_SOLID_COLOR = 'firebrick'
HARVEST_SOLID_COLOR = 'teal'

# Configuration and data loading
year_start = 1999

# Severity: conservative AGB loss (Mg/ha biomass in CSV → MgC ha$_\mathrm{D}^{-1}$ below)
AGBD_LOSS_COLUMN = 'agbd_lost_conservative'  # Options: 'agbd_lost_conservative' or 'agbd_lost'
RELATIVE_AGBD_LOSS_COLUMN = f'relative_{AGBD_LOSS_COLUMN}'
# Local AGB-loss intensity: disturbance rate × absolute severity (MgC ha$_\mathrm{F}^{-1}$)
RATE_X_SEVERITY_COLUMN = 'rate_x_severity'
RATE_X_SEVERITY_ERROR_COLUMN = f'{RATE_X_SEVERITY_COLUMN}_error'
_REL_AGBD_STD = 'relative_agbd_lost'
_REL_AGBD_CONS = 'relative_agbd_lost_conservative'
_REL_AGBD_COLUMNS = (_REL_AGBD_STD, _REL_AGBD_CONS, RELATIVE_AGBD_LOSS_COLUMN)
ROTATION_PERIOD_COLUMN = 'rotation_period'
ROTATION_PERIOD_ERROR_COLUMN = 'rotation_period_err'
TREE_COVER_LOW_THRESHOLD = 0.10
_TREE_COVER_PATH = '/home/apelliss/Documents/Resultat_France/Features_30m/tree_cover_map.tif'
_EXPLOITABILITY_PATH = (
    '/home/apelliss/Documents/Resultat_France/Features_30m/Geomorphometry_30m/exploitabilite.tif'
)
_FOREST_TFV_MASK_PATH = (
    '/home/apelliss/Documents/Resultat_France/Features_30m/Foret_30m/BD_Foret_TFVnum_30m_valideNR.tif'
)
_PATCH_PIXEL_AREA_HA = (30.0 * 30.0) / 10000.0  # 30 m pixel = 0.09 ha
_5KM_TARGET_RES_M = 5000.0
_EXPLOITABLE_FRAC_FOREST_5KM_TIF = (
    '/home/apelliss/Documents/Resultat_France/Data/exploitability_fraction_forest_5km_tc10.tif'
)
_FOREST_PIXEL_COUNT_5KM_TIF = (
    '/home/apelliss/Documents/Resultat_France/Data/forest_pixel_count_5km_tc10.tif'
)
_DOT_WHISKER_OFFSET = 0.2
_DOT_MARKERSIZE = 11
_DOT_MARKER_ALPHA = 1.0
_DOT_BOX_ASPECT_GROUP = 1
_DOT_BOX_ASPECT_TFV = 0.6
_LEGEND_MARKERSIZE = _DOT_MARKERSIZE + 3
_AXIS_TICK_LABELSIZE = 14
_LEGEND_FONTSIZE = 14
_MAP_CBAR_TICK_FONTSIZE = 14
# Inset colorbar geometry in map-axes fractions (MT panel i reference).
_MAP_CBAR_X = 0.9
_MAP_CBAR_Y = 0.5
_MAP_CBAR_WIDTH = 0.03
_MAP_CBAR_HEIGHT = 0.4
_CLIP_ANNOT_FONTSIZE = 11
_XTICK_LABEL_PAD = 40
_ROTATION_MID_TITLE_X_FRAC = 0.82  # between exploitability and rotation maps, biased right
_HARVEST_MARKER = 'o'
_NATURAL_MARKER = 's'

# AGB map uncertainty (Mg/ha biomass). Set False to use quantile SE only (strip if baked into CSV).
INCLUDE_AGB_MAP_ERROR = False
AGB_MAP_SIGMA_MG_HA = 60.0
AGBD_LOSS_MAP_SIGMA_MG_HA = np.sqrt(2.0) * AGB_MAP_SIGMA_MG_HA
CARBON_FRACTION = 0.475
# Kowalski / Planet AGBD loss on 10 km ETRS89-LAEA grid (Europe → clipped to France)
_KOWALSKI_LOSS_CSV = (
    '/home/apelliss/Documents/Autres_donnes/Severity_Katja/planet_agbd_loss_10km.csv'
)
_KOWALSKI_GRID_SHP = (
    '/home/apelliss/Documents/Autres_donnes/Severity_Katja/grid_10km_3035.shp'
)
_FRANCE_OUTLINE_SHP = '/home/apelliss/Documents/IGN_donnees/France_1km.shp'
# Normal approximation: 95% CI = estimate ± z * SE
NATIONAL_CI_Z_95 = 1.96
# Relative-uncertainty map masking (unused): was grey when area < 0.2% and rel. unc. > 200%
_REL_UNC_AREA_MASK_PCT = 0.2
_REL_UNC_MASK_PCT = 200.0
_REL_UNC_VMAP_CAP = 300.0

# Single-quantity SE, including rate × severity and carbon flux: TFV-block of the row-level
# errors. Carbon flux is a sum; the others are area-weighted means.
# Relative trends are formed at aggregation: 100 × mean(slope) / mean(2020 level).
TFV_COL = 'TFV'

# Forest type mapping dictionary
dict_group_en = {
    0.: 'New cut',
    1.0: 'Deciduous oak', 6.0: 'Evergreen oak', 9.0: 'Beech', 10.0: 'Chestnut', 14.0: 'Locust',
    19.0: 'Poplar',
    49.0: 'Other deciduous', 100.0: 'Mixed deciduous', 111.0: 'Mixed deciduous\nin island',
    51.0: 'Maritime pine', 52.0: 'Scotch pine', 53.0: 'Austrian pine', 57.0: 'Aleppo pine',
    58.0: 'Mountain\n& stone pines',
    81.0: 'Other pines', 80.0: 'Mixed pines',
    61.0: 'Fir & spruce', 63.0: 'Larch', 64.0: 'Douglas',
    91.0: 'Other non pine', 90.0: 'Mixed non pine',
    200.0: 'Mixed conifers', 222.0: 'Mixed conifers\nin island',
    310.0: 'Mainly deciduous', 320.0: 'Mainly conifers', 400.0: 'New cut',
    401.0: 'Open deciduous', 402.0: 'Open conifers', 403.0: 'Open mixed',
    'Forest': 'Forest'
}

# TFV to group mapping: Pure deciduous, Pure conifers, Mixed deciduous, Mixed conifers
# Fixed order for group bar charts - always show all 4 categories
TFV_GROUP_ORDER = ['Pure deciduous', 'Pure conifers', 'Mixed deciduous', 'Mixed conifers']

# Same palette as Fig_age_distribution.py (dominant category / growth curves)
TFV_GROUP_COLORS = {
    'Pure deciduous': 'forestgreen',
    'Pure conifers': 'mediumvioletred',
    'Mixed deciduous': '#1565C0',
    'Mixed conifers': '#E65100',
}
TFV_BROAD_COLORS = {
    'Deciduous': '#228833',
    'Conifers': 'mediumvioletred',
}

# SI TFV figures: horizontal (TFV on y, values on x), Fig_age_distribution S11 style
_SI_FIG_HEIGHT = 9.0
_SI_PANEL_WIDTH = 2.9
_SI_DOT_MARKERSIZE = 5
_SI_ERROR_CAPSIZE = 0
_SI_ERROR_LW = 0.9
_SI_YLABEL_FONTSIZE = 11
_SI_TICK_LABELSIZE = 11
_SI_XTICK_LABELSIZE = 10
_SI_PANEL_LABEL_FONTSIZE = 14
_SI_LEGEND_FONTSIZE = 11
_SI_CLIP_ANNOT_FONTSIZE = 8
_SI_XTICK_LABEL_PAD = 36
_SI_YTICK_LABELSIZE = 9
_SI_LEGEND_MARKERSIZE = 7
_SI_LEGEND_ELEMENTS = [
    Line2D([0], [0], marker=_HARVEST_MARKER, linestyle='None', markerfacecolor='gray', markeredgecolor='gray',
           markersize=_SI_LEGEND_MARKERSIZE, alpha=_DOT_MARKER_ALPHA, label='Harvest'),
    Line2D([0], [0], marker=_NATURAL_MARKER, linestyle='None', markerfacecolor='gray', markeredgecolor='gray',
           markersize=_SI_LEGEND_MARKERSIZE, alpha=_DOT_MARKER_ALPHA, label='Natural'),
]


def _category_color(item, group_col='TFV'):
    """Forest-category color from TFV_group name or TFV code (Fig_age_distribution palette)."""
    if group_col == 'TFV_group':
        return TFV_GROUP_COLORS.get(item, 'gray')
    tfv_group = dict_group_tfvs.get(float(item))
    return TFV_GROUP_COLORS.get(tfv_group, 'gray')


def _y_floor_for_column(column):
    """Physical floor for nonnegative disturbance-area metrics (%)."""
    if not column:
        return None
    if (
        'd_area_2020' in column
        or column in _REL_AGBD_COLUMNS
        or column == AGBD_LOSS_COLUMN
        or column == RATE_X_SEVERITY_COLUMN
        or column == 'lost_agb'
        or column == 'mean_patch_size_2020'
    ):
        return 0.0
    return None


def _y_cap_for_column(column):
    """Physical upper cap for bounded metrics."""
    if column in _REL_AGBD_COLUMNS:
        return 100.0
    return None


def _clip_relative_agbd_pct(df, columns=None):
    """Clamp relative AGB loss (%) to [0, 100]. Standard errors are clipped at 0 only."""
    cols = columns if columns is not None else _REL_AGBD_COLUMNS
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(lower=0.0, upper=100.0)
        err = f'{col}_error'
        if err in df.columns:
            df[err] = pd.to_numeric(df[err], errors='coerce').clip(lower=0.0)
    return df


dict_group_tfvs = {
    0.0: 'Other', 1.0: 'Pure deciduous', 6.0: 'Pure deciduous', 9.0: 'Pure deciduous',
    10.0: 'Pure deciduous', 14.0: 'Pure deciduous', 19.0: 'Pure deciduous',
    49.0: 'Pure deciduous', 100.0: 'Mixed deciduous', 111.0: 'Mixed deciduous',
    51.0: 'Pure conifers', 52.0: 'Pure conifers', 53.0: 'Pure conifers', 57.0: 'Pure conifers',
    58.0: 'Pure conifers', 81.0: 'Pure conifers', 80.0: 'Mixed conifers',
    61.0: 'Pure conifers', 63.0: 'Pure conifers', 64.0: 'Pure conifers',
    91.0: 'Pure conifers', 90.0: 'Mixed conifers',
    200.0: 'Mixed conifers', 222.0: 'Mixed conifers',
    310.0: 'Mixed deciduous', 320.0: 'Mixed conifers', 400.0: 'Other',
    401.0: 'Mixed deciduous', 402.0: 'Mixed conifers', 403.0: 'Mixed deciduous',
}



def load_kowalski_france_severity(
    loss_csv: str | Path = _KOWALSKI_LOSS_CSV,
    grid_shp: str | Path = _KOWALSKI_GRID_SHP,
    france_shp: str | Path = _FRANCE_OUTLINE_SHP,
    target_crs: str | None = 'EPSG:2154',
) -> gpd.GeoDataFrame:
    """Join Planet/Kowalski 10 km severity to the LAEA grid and clip to France.

    Absolute severity is converted Mg/ha biomass → MgC ha⁻¹ (× ``CARBON_FRACTION``).
    Relative severity is stored as percent (CSV fraction × 100).
    """
    loss_csv = Path(loss_csv).expanduser()
    grid_shp = Path(grid_shp).expanduser()
    france_shp = Path(france_shp).expanduser()
    if not loss_csv.is_file():
        raise FileNotFoundError(f'Kowalski loss CSV not found: {loss_csv}')
    if not grid_shp.is_file():
        raise FileNotFoundError(f'Kowalski grid shapefile not found: {grid_shp}')
    if not france_shp.is_file():
        raise FileNotFoundError(f'France outline shapefile not found: {france_shp}')

    csv = pd.read_csv(loss_csv)
    need = {'gridid', 'planet_mean_sev_abs', 'planet_mean_sev_rel'}
    missing = need - set(csv.columns)
    if missing:
        raise KeyError(f'Kowalski CSV missing columns: {sorted(missing)}')

    grid = gpd.read_file(grid_shp)
    if 'gridid' not in grid.columns:
        raise KeyError(f'Kowalski grid missing gridid; columns={list(grid.columns)}')

    merged = grid.merge(csv, on='gridid', how='inner')
    france = gpd.read_file(france_shp)
    if france.crs is None:
        raise ValueError(f'France outline has no CRS: {france_shp}')
    if merged.crs is None:
        raise ValueError(f'Kowalski grid has no CRS: {grid_shp}')

    france_in_grid = france.to_crs(merged.crs)
    fr_geom = france_in_grid.union_all() if hasattr(france_in_grid, 'union_all') else france_in_grid.unary_union
    out = merged.loc[merged.intersects(fr_geom)].copy()
    out['planet_mean_sev_abs_mgc'] = (
        pd.to_numeric(out['planet_mean_sev_abs'], errors='coerce') * CARBON_FRACTION
    ).clip(lower=0.0)
    out['planet_mean_sev_rel_pct'] = (
        pd.to_numeric(out['planet_mean_sev_rel'], errors='coerce') * 100.0
    ).clip(lower=0.0, upper=100.0)

    if target_crs is not None:
        out = out.to_crs(target_crs)
    return out


def _ensure_ser_num(ser_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add numeric ``ser_num`` from ``codeser`` when missing (A11 → 111)."""
    out = ser_gdf.copy()
    if 'ser_num' in out.columns:
        return out
    if 'codeser' not in out.columns:
        raise KeyError(f'SER shapefile needs ser_num or codeser; columns={list(out.columns)}')
    out['ser_num'] = out['codeser'].map(
        lambda c: (ord(str(c).strip()[0].upper()) - 64) * 100 + int(str(c).strip()[1:])
    )
    return out


def aggregate_kowalski_to_ser(
    kow: gpd.GeoDataFrame,
    ser_gdf: gpd.GeoDataFrame,
    *,
    value_cols: tuple[str, ...] = ('planet_mean_sev_abs_mgc', 'planet_mean_sev_rel_pct'),
) -> gpd.GeoDataFrame:
    """Area-weighted mean of Kowalski 10 km cells inside each SER.

    Dissolves multipart SERs before overlay so every polygon part is included
    (``drop_duplicates`` would keep only one part and leave large SERs empty).
    """
    ser = _ensure_ser_num(ser_gdf)
    if kow.crs is None or ser.crs is None:
        raise ValueError('Kowalski and SER GeoDataFrames must have a CRS')
    ser = ser.to_crs(kow.crs)
    keep = ['geometry', *value_cols]
    cells = kow[keep].dropna(subset=list(value_cols), how='all')
    if cells.empty:
        out = ser[['ser_num', 'geometry']].copy()
        for c in value_cols:
            out[c] = np.nan
        return out

    ser_diss = ser[['ser_num', 'geometry']].dissolve(by='ser_num').reset_index()
    pieces = gpd.overlay(
        cells, ser_diss,
        how='intersection', keep_geom_type=False,
    )
    if pieces.empty:
        out = ser[['ser_num', 'geometry']].copy()
        for c in value_cols:
            out[c] = np.nan
        return out

    pieces = pieces.copy()
    pieces['_w'] = pieces.geometry.area
    rows = []
    for sid, g in pieces.groupby('ser_num'):
        w = g['_w'].to_numpy(dtype=float)
        row = {'ser_num': sid}
        for col in value_cols:
            v = pd.to_numeric(g[col], errors='coerce').to_numpy(dtype=float)
            ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
            row[col] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        rows.append(row)
    stats = pd.DataFrame(rows)
    out = ser.merge(stats, on='ser_num', how='left')
    return out


def rate_weighted_severity_by_ser(
    harvest_agg: pd.DataFrame,
    natural_agg: pd.DataFrame,
    *,
    sev_col: str,
    rate_col: str = 'd_area_2020_perc',
    sev_err_col: str | None = None,
    ser_col: str = 'ser',
    out_col: str = 'rate_weighted_severity',
    out_se_col: str | None = None,
) -> pd.DataFrame:
    """Rate-weighted mean of harvest + natural severity per SER.

    ``(r_h · s_h + r_n · s_n) / (r_h + r_n)`` using Fig. 4 rates and Fig. 5 severities.
    If ``sev_err_col`` is set, also returns SE treating rates as fixed weights and
    agents as independent: ``√[(r_h/W)² σ_h² + (r_n/W)² σ_n²]``.
    """
    need = {ser_col, rate_col, sev_col}
    if sev_err_col:
        need.add(sev_err_col)
    for name, df in (('harvest', harvest_agg), ('natural', natural_agg)):
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f'{name} agg missing columns: {sorted(missing)}')

    cols = [ser_col, rate_col, sev_col] + ([sev_err_col] if sev_err_col else [])
    h = harvest_agg[cols].copy()
    n = natural_agg[cols].copy()
    h[ser_col] = pd.to_numeric(h[ser_col], errors='coerce')
    n[ser_col] = pd.to_numeric(n[ser_col], errors='coerce')
    m = h.merge(n, on=ser_col, how='outer', suffixes=('_h', '_n'))
    rh = pd.to_numeric(m[f'{rate_col}_h'], errors='coerce').clip(lower=0.0).fillna(0.0)
    rn = pd.to_numeric(m[f'{rate_col}_n'], errors='coerce').clip(lower=0.0).fillna(0.0)
    sh = pd.to_numeric(m[f'{sev_col}_h'], errors='coerce')
    sn = pd.to_numeric(m[f'{sev_col}_n'], errors='coerce')
    # If one agent has no severity, weight only the other when its rate > 0.
    wh = rh.where(sh.notna(), 0.0)
    wn = rn.where(sn.notna(), 0.0)
    w = wh + wn
    num = wh * sh.fillna(0.0) + wn * sn.fillna(0.0)
    mean = np.where(w > 0, num / w, np.nan)
    out = pd.DataFrame({ser_col: m[ser_col], out_col: mean})
    if sev_err_col:
        se_h = pd.to_numeric(m[f'{sev_err_col}_h'], errors='coerce').clip(lower=0.0).fillna(0.0)
        se_n = pd.to_numeric(m[f'{sev_err_col}_n'], errors='coerce').clip(lower=0.0).fillna(0.0)
        se_h = se_h.where(sh.notna(), 0.0)
        se_n = se_n.where(sn.notna(), 0.0)
        se = np.where(
            w > 0,
            np.sqrt((wh / w) ** 2 * se_h ** 2 + (wn / w) ** 2 * se_n ** 2),
            np.nan,
        )
        out[out_se_col or f'{out_col}_se'] = se
    return out.dropna(subset=[ser_col])


def write_si11_kowalski_figure(
    figure_dir: str | Path,
    *,
    loss_csv: str | Path = _KOWALSKI_LOSS_CSV,
    grid_shp: str | Path = _KOWALSKI_GRID_SHP,
    france_shp: str | Path = _FRANCE_OUTLINE_SHP,
    ser_shapefile: str | Path = '/home/apelliss/Documents/IGN_donnees/France_ser.shp',
    harvest_agg: pd.DataFrame | None = None,
    natural_agg: pd.DataFrame | None = None,
    abs_sev_col: str = AGBD_LOSS_COLUMN,
    rel_sev_col: str = RELATIVE_AGBD_LOSS_COLUMN,
    rate_col: str = 'd_area_2020_perc',
    save_name: str = 'Figure_SI11_kowalski_severity_planet_10km.png',
) -> Path:
    """Write SI11: Kowalski SER severity + ours−Kowalski difference maps."""
    figure_dir = Path(figure_dir).expanduser()
    figure_dir.mkdir(parents=True, exist_ok=True)
    france_outline = gpd.read_file(Path(france_shp).expanduser())
    target_crs = str(france_outline.crs) if france_outline.crs is not None else 'EPSG:2154'
    kow = load_kowalski_france_severity(
        loss_csv=loss_csv,
        grid_shp=grid_shp,
        france_shp=france_shp,
        target_crs=target_crs,
    )
    if france_outline.crs is not None and str(france_outline.crs) != target_crs:
        france_outline = france_outline.to_crs(target_crs)

    ser_raw = gpd.read_file(Path(ser_shapefile).expanduser())
    if ser_raw.crs is not None and str(ser_raw.crs) != target_crs:
        ser_raw = ser_raw.to_crs(target_crs)
    ser_kow = aggregate_kowalski_to_ser(kow, ser_raw)

    print(
        f'SI11 Kowalski France cells: {len(kow)} '
        f'(abs non-null={int(kow.planet_mean_sev_abs_mgc.notna().sum())}, '
        f'rel non-null={int(kow.planet_mean_sev_rel_pct.notna().sum())}); '
        f'SER with abs={int(ser_kow.planet_mean_sev_abs_mgc.notna().sum())}/'
        f'{ser_kow["ser_num"].nunique()}'
    )

    abs_labels = lambda vmin, vmax: [
        f'{vmin:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
        f'{vmax:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
    ]
    rel_labels = lambda vmin, vmax: [f'{vmin:.0f}%', f'{vmax:.0f}%']

    have_ours = harvest_agg is not None and natural_agg is not None
    ser_ours = None
    if have_ours:
        abs_err_col = f'{abs_sev_col}_error'
        rel_err_col = f'{rel_sev_col}_error'
        abs_w = rate_weighted_severity_by_ser(
            harvest_agg, natural_agg, sev_col=abs_sev_col, rate_col=rate_col,
            sev_err_col=abs_err_col if abs_err_col in harvest_agg.columns else None,
            out_col='ours_abs_rate_weighted', out_se_col='ours_abs_rate_weighted_se',
        )
        rel_w = rate_weighted_severity_by_ser(
            harvest_agg, natural_agg, sev_col=rel_sev_col, rate_col=rate_col,
            sev_err_col=rel_err_col if rel_err_col in harvest_agg.columns else None,
            out_col='ours_rel_rate_weighted', out_se_col='ours_rel_rate_weighted_se',
        )
        ser_ours = (
            _ensure_ser_num(ser_raw)
            .merge(abs_w, left_on='ser_num', right_on='ser', how='left')
            .drop(columns=['ser'], errors='ignore')
            .merge(rel_w, left_on='ser_num', right_on='ser', how='left')
            .drop(columns=['ser'], errors='ignore')
        )
        # Join Kowalski SER values for ours − Kowalski difference maps.
        kow_stats = (
            ser_kow.drop_duplicates('ser_num')[
                ['ser_num', 'planet_mean_sev_abs_mgc', 'planet_mean_sev_rel_pct']
            ]
        )
        ser_ours = ser_ours.merge(kow_stats, on='ser_num', how='left')
        ser_ours['diff_abs_ours_minus_kow'] = (
            ser_ours['ours_abs_rate_weighted'] - ser_ours['planet_mean_sev_abs_mgc']
        )
        ser_ours['diff_rel_ours_minus_kow'] = (
            ser_ours['ours_rel_rate_weighted'] - ser_ours['planet_mean_sev_rel_pct']
        )

        print(
            f'SI11 ours rate-weighted SER: '
            f'abs={int(ser_ours.ours_abs_rate_weighted.notna().sum())}, '
            f'rel={int(ser_ours.ours_rel_rate_weighted.notna().sum())}; '
            f'diff abs non-null={int(ser_ours.diff_abs_ours_minus_kow.notna().sum())}'
        )

    abs_diff_labels = lambda vmin, vmax: [
        f'{vmin:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
        '0',
        f'{vmax:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
    ]
    rel_diff_labels = lambda vmin, vmax: [f'{vmin:.0f}%', '0%', f'{vmax:.0f}%']

    if have_ours:
        abs_diff = ser_ours['diff_abs_ours_minus_kow'].dropna()
        rel_diff = ser_ours['diff_rel_ours_minus_kow'].dropna()
        abs_dmax = float(np.nanmax(np.abs(abs_diff))) if len(abs_diff) else 1.0
        rel_dmax = float(np.nanmax(np.abs(rel_diff))) if len(rel_diff) else 1.0
        kow_abs = ser_kow['planet_mean_sev_abs_mgc'].dropna()
        kow_rel = ser_kow['planet_mean_sev_rel_pct'].dropna()
        abs_vmin = float(kow_abs.min()) if len(kow_abs) else 0.0
        abs_vmax = float(kow_abs.max()) if len(kow_abs) else 1.0
        rel_vmin = float(kow_rel.min()) if len(kow_rel) else 0.0
        rel_vmax = float(kow_rel.max()) if len(kow_rel) else 1.0
        panels = [
            (ser_kow, 'planet_mean_sev_abs_mgc',
             'Kowalski absolute severity (SER)', abs_labels, abs_vmin, abs_vmax,
             'hot_r', 'continuous'),
            (ser_kow, 'planet_mean_sev_rel_pct',
             'Kowalski relative severity (SER)', rel_labels, rel_vmin, rel_vmax,
             'hot_r', 'continuous'),
            (ser_ours, 'diff_abs_ours_minus_kow',
             'Ours − Kowalski absolute severity', abs_diff_labels, -abs_dmax, abs_dmax,
             'RdBu_r', 'diverge'),
            (ser_ours, 'diff_rel_ours_minus_kow',
             'Ours − Kowalski relative severity', rel_diff_labels, -rel_dmax, rel_dmax,
             'RdBu_r', 'diverge'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 10.0))
        letters = ('a', 'b', 'c', 'd')
    else:
        panels = [
            (ser_kow, 'planet_mean_sev_abs_mgc', 'Kowalski absolute severity (SER)',
             abs_labels, None, None, 'hot_r', 'continuous'),
            (ser_kow, 'planet_mean_sev_rel_pct', 'Kowalski relative severity (SER)',
             rel_labels, None, None, 'hot_r', 'continuous'),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
        letters = ('a', 'b')

    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (gdf, col, title, label_fn, vmin_fix, vmax_fix, cmap_name, kind), letter in zip(
        axes_flat, panels, letters
    ):
        valid = gdf[col].dropna()
        if valid.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        vmin = float(vmin_fix) if vmin_fix is not None else float(valid.min())
        vmax = float(vmax_fix) if vmax_fix is not None else float(valid.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmax = vmin + 1.0 if np.isfinite(vmin) else 1.0
            vmin = vmin if np.isfinite(vmin) else 0.0
        cmap_obj = mpl.colormaps[cmap_name].copy()
        cmap_obj.set_bad('none')
        if kind == 'diverge' and vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            ticks = [vmin, 0.0, vmax]
            tick_labels = label_fn(vmin, vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            ticks = [vmin, vmax]
            tick_labels = label_fn(vmin, vmax)
            if len(tick_labels) > 2:
                tick_labels = [tick_labels[0], tick_labels[-1]]
        gdf.plot(
            column=col, cmap=cmap_obj, ax=ax, legend=False, norm=norm,
            edgecolor='none', linewidth=0, missing_kwds={'color': 'lightgray'},
        )
        france_outline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=11)
        ax.set_axis_off()
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='top', ha='left', clip_on=False, zorder=20)
        ax._map_cbar_spec = {
            'mappable': ax.collections[0],
            'ticks': ticks,
            'labels': tick_labels,
            'title': title,
        }

    plt.tight_layout(pad=0.2)
    # Extra column gap so left-column colorbars do not sit against the right maps.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, hspace=0.20, wspace=0.18)
    fig.canvas.draw()

    for ax in axes_flat:
        spec = getattr(ax, '_map_cbar_spec', None)
        if spec is None:
            continue
        pos = ax.get_position()
        fig.text(
            0.5 * (pos.x0 + pos.x1), pos.y1 + 0.008, spec['title'],
            fontsize=15, ha='center', va='bottom', clip_on=False,
        )
        fig_x = pos.x0 + _MAP_CBAR_X * pos.width
        fig_y = pos.y0 + _MAP_CBAR_Y * pos.height - _MAP_CBAR_HEIGHT * pos.height / 2
        cbar_ax = fig.add_axes([fig_x, fig_y, _MAP_CBAR_WIDTH * pos.width, _MAP_CBAR_HEIGHT * pos.height])
        cbar = plt.colorbar(spec['mappable'], cax=cbar_ax, orientation='vertical')
        cbar.set_ticks(spec['ticks'])
        cbar.set_ticklabels(spec['labels'], fontsize=_MAP_CBAR_TICK_FONTSIZE)
        cbar.ax.tick_params(labelsize=_MAP_CBAR_TICK_FONTSIZE)

    save_path = figure_dir / save_name
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.30)
    plt.close(fig)
    print(f'Wrote {save_path}')
    return save_path


def run_figures(
    trend_csv: str | Path,
    figure_dir: str | Path,
    *,
    ser_shapefile: str | Path = '/home/apelliss/Documents/IGN_donnees/France_ser.shp',
    greco_shapefile: str | Path = '/home/apelliss/Documents/IGN_donnees/France_greco.shp',
    kowalski_loss_csv: str | Path | None = None,
    kowalski_grid_shp: str | Path | None = None,
    interactive: bool = False,
) -> Path:
    """Load joined trends+severity CSV and write Fig. 4–5, SI7–SI11 PNGs."""
    from pathlib import Path as _Path
    trend_csv = _Path(trend_csv).expanduser()
    figure_dir = _Path(figure_dir).expanduser()
    figure_dir.mkdir(parents=True, exist_ok=True)
    ser_shapefile = _Path(ser_shapefile).expanduser()
    greco_shapefile = _Path(greco_shapefile).expanduser()
    _kowalski_csv = _Path(kowalski_loss_csv).expanduser() if kowalski_loss_csv else Path(_KOWALSKI_LOSS_CSV)
    _kowalski_grid = _Path(kowalski_grid_shp).expanduser() if kowalski_grid_shp else Path(_KOWALSKI_GRID_SHP)
    if not interactive:
        plt.ioff()
    else:
        plt.ion()

    # Load disturbance trend data
    trend_results = pd.read_csv(trend_csv)

    # Data processing
    trend_ok = trend_results.copy().drop_duplicates()
    # Stand types only: drop non-forest (TFV <= 0) and recently cut without species (TFV 400);
    # keep other open/new-cut classes below 500 (401–403).
    _tfv_num = pd.to_numeric(trend_ok['TFV'], errors='coerce')
    _tfv_keep = _tfv_num.notna() & (_tfv_num > 0) & (_tfv_num < 500) & (_tfv_num != 400)
    trend_ok = trend_ok.loc[_tfv_keep].copy()
    trend_ok['TFV'] = _tfv_num.loc[_tfv_keep].astype(int)

    # Convert AGB from Mg/ha to MgC ha$_\mathrm{D}^{-1}$ using factor 0.475
    # Determine which column to use based on configuration and availability
    agbd_loss_col = AGBD_LOSS_COLUMN # 'agbd_lost'
    agbd_loss_error_col = f'{AGBD_LOSS_COLUMN}_error'
    relative_agbd_loss_error_col = f'{RELATIVE_AGBD_LOSS_COLUMN}_error'

    if agbd_loss_col not in trend_ok.columns:
        raise KeyError(f"Column '{agbd_loss_col}' not found in the CSV file. Available columns: {list(trend_ok.columns)}")
    if agbd_loss_error_col not in trend_ok.columns:
        raise KeyError(f"Column '{agbd_loss_error_col}' not found in the CSV file. Available columns: {list(trend_ok.columns)}")

    # Always use AGBD_LOSS_COLUMN as the working column name for consistency
    trend_ok[AGBD_LOSS_COLUMN] = (
        pd.to_numeric(trend_ok[agbd_loss_col], errors='coerce') * CARBON_FRACTION
    ).clip(lower=0.0)
    _csv_loss_err = trend_ok[agbd_loss_error_col].astype(float)
    # Rebuilt CSV often has map sigma baked in (median SE ~ sqrt(2)*60); detect then strip or keep.
    _pos_err = _csv_loss_err[_csv_loss_err > 1]
    _map_sigma_in_csv = len(_pos_err) > 0 and _pos_err.quantile(0.5) > 15.0
    if INCLUDE_AGB_MAP_ERROR:
        if _map_sigma_in_csv:
            _abs_err_mg = _csv_loss_err
        else:
            _abs_err_mg = np.sqrt(_csv_loss_err ** 2 + AGBD_LOSS_MAP_SIGMA_MG_HA ** 2)
    else:
        if _map_sigma_in_csv:
            _abs_err_mg = np.sqrt(np.maximum(_csv_loss_err ** 2 - AGBD_LOSS_MAP_SIGMA_MG_HA ** 2, 0.0))
            # Relative SE in CSV tracks ~100*abs/q50; scale after stripping map sigma.
            _scale = np.where(_csv_loss_err > 0, _abs_err_mg / _csv_loss_err, 0.0)
            if relative_agbd_loss_error_col in trend_ok.columns:
                trend_ok[relative_agbd_loss_error_col] = (
                    trend_ok[relative_agbd_loss_error_col].astype(float) * _scale
                ).clip(lower=0)
        else:
            _abs_err_mg = _csv_loss_err
    trend_ok[f'{AGBD_LOSS_COLUMN}_error'] = np.maximum(_abs_err_mg * CARBON_FRACTION, 0.0)

    # Relative AGB loss is a fraction of stand AGB: clamp to [0, 100]%.
    _clip_relative_agbd_pct(trend_ok)

    def _as_float(series):
        """Numeric column as a float array, with non-numeric values set to 0."""
        return np.nan_to_num(pd.to_numeric(series, errors='coerce').to_numpy(dtype=float), nan=0.0)


    def _fitted_or_fallback(trend, intercept, slope, intercept_se, slope_se, median, median_se,
                            invalid_below, fallback_floor):
        """2020 level from intercept + (2020-1999)×slope, or the median when that line is not kept.

        The point slope is 0 when the line is not kept. The slope SE stays the regression SE.
        Covariance is (2020-1999)×σ_slope² only on rows that keep the fitted line.
        """
        dt = float(2020 - year_start)
        keep = trend.astype(str).to_numpy() != 'no trend'
        a = _as_float(intercept)
        b = _as_float(slope)
        sa = _as_float(intercept_se)
        sb = np.maximum(_as_float(slope_se), 0.0)
        med = _as_float(median)
        sm = np.maximum(_as_float(median_se), 0.0)
        fitted = a + b * dt
        fitted_se = np.sqrt(sa ** 2 + (sb * dt) ** 2)
        keep = keep & np.isfinite(fitted) & (fitted >= invalid_below)
        fallback = np.where(np.isfinite(med), med, fallback_floor)
        fallback = np.maximum(fallback, fallback_floor)
        level = np.where(keep, fitted, fallback)
        level_se = np.maximum(np.where(keep, fitted_se, sm), 0.0)
        slope_hat = np.where(keep, b, 0.0)
        cov = np.where(keep, dt * sb ** 2, 0.0)
        return level, level_se, slope_hat, sb, cov, keep


    def add_row_disturbance_metrics(df):
        """Value and standard error of every disturbance quantity, one row at a time."""
        level, level_se, slope, slope_se, cov, _ = _fitted_or_fallback(
            df['trend_disturbance_rate'],
            df['intercept_disturbance_rate'], df['slope_disturbance_rate'],
            df['intercept_std_disturbance_rate'], df['slope_std_disturbance_rate'],
            df['median_disturbance_rate'], df['median_std_disturbance_rate'],
            invalid_below=0.0, fallback_floor=0.0,
        )
        df['disturbance_rate_2020'] = level
        df['disturbance_rate_2020_error'] = level_se
        df['d_area_2020_perc'] = level * 100.0
        df['d_area_2020_error_perc'] = level_se * 100.0
        df['d_area_2020'] = level * df['total_area'].fillna(0.0).to_numpy(dtype=float) * 1e6
        df['d_area_2020_error'] = level_se * df['total_area'].fillna(0.0).to_numpy(dtype=float) * 1e6
        df['slope_pp'] = slope * 100.0
        df['slope_pp_err'] = slope_se * 100.0
        df['d_area_level_slope_cov'] = cov * 100.0 ** 2

        level_px, level_px_se, slope_px, slope_px_se, cov_px, _ = _fitted_or_fallback(
            df['trend_mean_patch_size'],
            df['intercept_mean_patch_size'], df['slope_mean_patch_size'],
            df['intercept_std_mean_patch_size'], df['slope_std_mean_patch_size'],
            df['median_mean_patch_size'], df['median_std_mean_patch_size'],
            invalid_below=1.0, fallback_floor=1.0,
        )
        ha = _PATCH_PIXEL_AREA_HA
        df['mean_patch_size_2020'] = level_px * ha
        df['mean_patch_size_2020_error'] = level_px_se * ha
        df['slope_patch_ha'] = slope_px * ha
        df['slope_patch_ha_err'] = slope_px_se * ha
        df['patch_level_slope_cov'] = cov_px * ha ** 2

        severity = df[AGBD_LOSS_COLUMN].fillna(0.0).to_numpy(dtype=float)
        severity_se = df[f'{AGBD_LOSS_COLUMN}_error'].fillna(0.0).to_numpy(dtype=float)
        rate = df['disturbance_rate_2020'].to_numpy(dtype=float)
        rate_se = df['disturbance_rate_2020_error'].to_numpy(dtype=float)
        area = df['total_area'].fillna(0.0).to_numpy(dtype=float)
        sig_area = severity * rate_se * area
        sig_loss = severity_se * rate * area
        df['lost_agb'] = np.maximum(severity * rate * area, 0.0)
        df['lost_agb_error'] = np.sqrt(sig_area ** 2 + sig_loss ** 2)
        df[RATE_X_SEVERITY_COLUMN] = np.maximum(severity * rate, 0.0)
        df[RATE_X_SEVERITY_ERROR_COLUMN] = np.sqrt((severity * rate_se) ** 2 + (rate * severity_se) ** 2)
        return df


    # Row-level quantities and their errors. Aggregation only combines these columns.
    trend_ok = add_row_disturbance_metrics(trend_ok)


    trend_ok = trend_ok.drop_duplicates()

    # Replace 'D' by 'P' in FT column only for poplars (TFV == 19)
    if 'FT' in trend_ok.columns:
        poplar_mask = trend_ok['TFV'] == 19
        trend_ok.loc[poplar_mask, 'FT'] = trend_ok.loc[poplar_mask, 'FT'].astype(str).str.replace('D', 'P', regex=False)

    # %%
    # Load France shapefile for mapping
    shapefile_path = str(ser_shapefile)
    france_ser = gpd.read_file(shapefile_path)
    if 'ser_num' not in france_ser.columns:
        # IGN shapefile uses codeser (e.g. 'A11' -> 111)
        france_ser['ser_num'] = france_ser['codeser'].map(
            lambda c: (ord(str(c).strip()[0].upper()) - 64) * 100 + int(str(c).strip()[1:])
        )

    # Load GRECO shapefile for overlay
    greco_gdf = gpd.read_file(str(greco_shapefile))
    france_gdf = gpd.read_file('/home/apelliss/Documents/IGN_donnees/France_1km.shp')


    def _group_df_for_national_se(df, tfv_col=TFV_COL):
        """Restore TFV when groupby(..., include_groups=False) dropped it."""
        work = df.copy()
        key = getattr(df, 'name', None)
        if tfv_col not in work.columns and key is not None and not isinstance(key, tuple):
            work[tfv_col] = key
        missing = [c for c in (tfv_col,) if c not in work.columns]
        if missing:
            raise KeyError(
                f'{missing} required for TFV-block SE; use include_groups=True in groupby '
                f'or pass a slice that still contains {tfv_col!r}.'
            )
        return work


    def var_sum_oneway_block(sig, groups, row_weights=None):
        """Variance of a sum with one shared shock inside each group: (Σ w_i σ_i)²."""
        sig = np.asarray(sig, dtype=float)
        if row_weights is None:
            w = np.ones_like(sig)
        else:
            w = np.asarray(row_weights, dtype=float)
            if w.shape != sig.shape:
                raise ValueError('row_weights must match sig length')
        v = 0.0
        for _, idx in pd.Series(sig).groupby(groups).groups.items():
            v += float((sig[idx] * w[idx]).sum() ** 2)
        return v


    def stderr_area_weighted_mean_tfv_block(df_subset, sigma_col=None, weight_col='total_area',
                                            tfv_col=TFV_COL):
        """SE of an area-weighted mean when row errors share one shock within each TFV."""
        if sigma_col is None:
            sigma_col = f'{AGBD_LOSS_COLUMN}_error'
        work = _group_df_for_national_se(df_subset, tfv_col=tfv_col)
        w = work[weight_col].fillna(0).to_numpy(dtype=float)
        sig = np.maximum(work[sigma_col].fillna(0).to_numpy(dtype=float), 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return np.nan
        v_sum = var_sum_oneway_block(sig, work[tfv_col].to_numpy(), row_weights=w)
        return float(np.sqrt(v_sum) / w_sum)


    def _mean_tfv(df, value_col, sigma_col, weight_col='total_area', tfv_col=TFV_COL):
        """Area-weighted mean and its TFV-block SE, from row-level values and errors."""
        work = _group_df_for_national_se(df, tfv_col=tfv_col)
        w = work[weight_col].fillna(0).to_numpy(dtype=float)
        x = work[value_col].fillna(0).to_numpy(dtype=float)
        if float(np.sum(w)) <= 0:
            return np.nan, np.nan
        mean = float(np.average(x, weights=w))
        se = stderr_area_weighted_mean_tfv_block(
            work, sigma_col=sigma_col, weight_col=weight_col, tfv_col=tfv_col,
        )
        return mean, se


    def _cov_area_weighted_means(df, cov_col, weight_col='total_area', tfv_col=TFV_COL):
        """Cov of two area-weighted means from within-row covariances.

        cov_col is (2020-1999)×σ_slope² on rows that keep the fitted line, else 0.
        Inside a TFV the slope shocks are one shared shock, so the covariance of the
        weighted sums is (Σ w_i √cov_i)².
        """
        work = _group_df_for_national_se(df, tfv_col=tfv_col).reset_index(drop=True)
        w = work[weight_col].fillna(0).to_numpy(dtype=float)
        cov = np.maximum(work[cov_col].fillna(0).to_numpy(dtype=float), 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return 0.0
        sqrt_cov = np.sqrt(cov)
        v = 0.0
        for _, idx in pd.Series(sqrt_cov).groupby(work[tfv_col].to_numpy()).groups.items():
            ii = np.asarray(list(idx), dtype=int)
            v += float((w[ii] * sqrt_cov[ii]).sum() ** 2)
        return v / w_sum ** 2


    # Relative trend at every aggregated scale: 100 × mean(slope) / mean(2020 level).
    _RATIO_OF_MEANS_SPECS = {
        'd_area_evolution_perc': (
            'slope_pp', 'slope_pp_err', 'd_area_2020_perc', 'd_area_2020_error_perc',
            'd_area_level_slope_cov',
        ),
        'mean_patch_size_evolution_perc': (
            'slope_patch_ha', 'slope_patch_ha_err',
            'mean_patch_size_2020', 'mean_patch_size_2020_error',
            'patch_level_slope_cov',
        ),
    }


    def ratio_of_means_stats(df, slope_col, slope_se_col, level_col, level_se_col, cov_col,
                             tfv_col=TFV_COL):
        """100 × mean(slope) / mean(2020 level) and the delta-method SE, including covariance."""
        s, se_s = _mean_tfv(df, slope_col, slope_se_col, tfv_col=tfv_col)
        level, se_level = _mean_tfv(df, level_col, level_se_col, tfv_col=tfv_col)
        if not np.isfinite(level) or level <= 0 or not np.isfinite(s):
            return 0.0, 0.0
        se_s = 0.0 if not np.isfinite(se_s) else float(se_s)
        se_level = 0.0 if not np.isfinite(se_level) else float(se_level)
        cov = _cov_area_weighted_means(df, cov_col, tfv_col=tfv_col)
        cap = se_s * se_level
        cov = float(np.clip(cov, -cap, cap))
        ratio = 100.0 * s / level
        var = (
            (100.0 / level) ** 2 * se_s ** 2
            + (100.0 * s / level ** 2) ** 2 * se_level ** 2
            - 2.0 * (100.0 / level) * (100.0 * s / level ** 2) * cov
        )
        return float(ratio), float(np.sqrt(max(var, 0.0)))


    def se_lost_agb_national(df, tfv_col=TFV_COL):
        """SE of the sum of lost biomass (TgC): TFV-block of the row-level flux errors."""
        work = _group_df_for_national_se(df, tfv_col=tfv_col)
        sig = np.maximum(work['lost_agb_error'].fillna(0).to_numpy(dtype=float), 0.0)
        return float(np.sqrt(var_sum_oneway_block(sig, work[tfv_col].to_numpy())))


    def _sigma_col_for_value(value_col):
        """Default per-row sigma column for area-weighted means."""
        known = {
            'd_area_2020_perc': 'd_area_2020_error_perc',
            'd_area_evolution_perc': 'd_area_evolution_error_perc',
            'mean_patch_size_2020': 'mean_patch_size_2020_error',
            'mean_patch_size_evolution_perc': 'mean_patch_size_evolution_error_perc',
            'slope_pp': 'slope_pp_err',
            'slope_patch_ha': 'slope_patch_ha_err',
            AGBD_LOSS_COLUMN: agbd_loss_error_col,
            RELATIVE_AGBD_LOSS_COLUMN: relative_agbd_loss_error_col,
            RATE_X_SEVERITY_COLUMN: RATE_X_SEVERITY_ERROR_COLUMN,
            ROTATION_PERIOD_COLUMN: ROTATION_PERIOD_ERROR_COLUMN,
        }
        if value_col in known:
            return known[value_col]
        return f'{value_col}_error'


    def _rotation_positive_subset(df):
        """Rows with a positive rotation period, its error, and forest area."""
        cols = [ROTATION_PERIOD_COLUMN, ROTATION_PERIOD_ERROR_COLUMN, 'total_area', TFV_COL]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return df.iloc[0:0].copy()
        work = df[cols].copy()
        return work[
            np.isfinite(work[ROTATION_PERIOD_COLUMN]) & (work[ROTATION_PERIOD_COLUMN] > 0)
            & np.isfinite(work[ROTATION_PERIOD_ERROR_COLUMN]) & (work[ROTATION_PERIOD_ERROR_COLUMN] >= 0)
            & (work['total_area'] > 0)
        ]


    def _rotation_mean_subset(df):
        """Rows with a positive rotation period and forest area."""
        cols = [ROTATION_PERIOD_COLUMN, 'total_area']
        if any(c not in df.columns for c in cols):
            return df.iloc[0:0].copy()
        work = df[cols].copy()
        return work[
            np.isfinite(work[ROTATION_PERIOD_COLUMN]) & (work[ROTATION_PERIOD_COLUMN] > 0)
            & (work['total_area'] > 0)
        ]


    def national_area_weighted_mean_stderr(df, value_col, sigma_col=None, tfv_col=TFV_COL,
                                           weight_col='total_area'):
        """Area-weighted mean and TFV-block SE."""
        if sigma_col is None:
            sigma_col = _sigma_col_for_value(value_col)
        return _mean_tfv(df, value_col, sigma_col, weight_col=weight_col, tfv_col=tfv_col)


    def national_agbd_loss_mean_stderr(df, tfv_col=TFV_COL):
        """Area-weighted mean AGBD loss (MgC ha$_\mathrm{D}^{-1}$) and its TFV-block SE."""
        return national_area_weighted_mean_stderr(df, AGBD_LOSS_COLUMN, tfv_col=tfv_col)


    def national_relative_agbd_loss_mean_stderr(df, tfv_col=TFV_COL):
        """Area-weighted mean relative AGBD loss (%) and its TFV-block SE."""
        return national_area_weighted_mean_stderr(df, RELATIVE_AGBD_LOSS_COLUMN, tfv_col=tfv_col)


    def national_mean_patch_size_mean_stderr(df, tfv_col=TFV_COL):
        """Area-weighted mean patch size (ha) and its TFV-block SE."""
        return national_area_weighted_mean_stderr(
            df, 'mean_patch_size_2020', 'mean_patch_size_2020_error', tfv_col=tfv_col,
        )


    def national_mean_patch_size_evolution_mean_stderr(df, tfv_col=TFV_COL):
        """100 × mean(patch slope) / mean(2020 patch size), with the TFV-block delta-method SE."""
        return ratio_of_means_stats(
            df, *_RATIO_OF_MEANS_SPECS['mean_patch_size_evolution_perc'], tfv_col=tfv_col,
        )


    def _clip_national_ci_bounds(table, z=NATIONAL_CI_Z_95):
        """Clip 95% CIs to physically valid ranges (normal SE can exceed [0, value])."""
        bounds = {
            'Disturbance rate (natural)': (0.0, 100.0),
            'Disturbance rate (harvest)': (0.0, 100.0),
            'Mean patch size (natural)': (0.0, None),
            'Mean patch size (harvest)': (0.0, None),
            'AGB loss (natural)': (0.0, None),
            'AGB loss (harvest)': (0.0, None),
            'Relative AGB loss (natural)': (0.0, 100.0),
            'Relative AGB loss (harvest)': (0.0, 100.0),
            'Harvest total AGB loss': (0.0, None),
            'Natural total AGB loss': (0.0, None),
            'Total AGB loss': (0.0, None),
        }
        lo = table['value'] - z * table['se']
        hi = table['value'] + z * table['se']
        for comp, (lo_b, hi_b) in bounds.items():
            mask = table['component'] == comp
            if lo_b is not None:
                lo.loc[mask] = lo.loc[mask].clip(lower=lo_b)
            if hi_b is not None:
                hi.loc[mask] = hi.loc[mask].clip(upper=hi_b)
        return lo, hi


    def build_national_uncertainty_table(natural_df, harvest_df):
        """National harvest/natural budget from row-level values and the aggregation rules above."""
        nat_rate, nat_rate_se = national_area_weighted_mean_stderr(
            natural_df, 'd_area_2020_perc', 'd_area_2020_error_perc',
        )
        harv_rate, harv_rate_se = national_area_weighted_mean_stderr(
            harvest_df, 'd_area_2020_perc', 'd_area_2020_error_perc',
        )
        nat_patch, nat_patch_se = national_mean_patch_size_mean_stderr(natural_df)
        harv_patch, harv_patch_se = national_mean_patch_size_mean_stderr(harvest_df)
        nat_ptev, nat_ptev_se = national_mean_patch_size_evolution_mean_stderr(natural_df)
        harv_ptev, harv_ptev_se = national_mean_patch_size_evolution_mean_stderr(harvest_df)
        nat_agbd, nat_agbd_se = national_agbd_loss_mean_stderr(natural_df)
        harv_agbd, harv_agbd_se = national_agbd_loss_mean_stderr(harvest_df)
        nat_rel_agbd, nat_rel_agbd_se = national_relative_agbd_loss_mean_stderr(natural_df)
        harv_rel_agbd, harv_rel_agbd_se = national_relative_agbd_loss_mean_stderr(harvest_df)
        nat_flux = float(natural_df['lost_agb'].fillna(0).sum())
        se_nat = se_lost_agb_national(natural_df)
        cc = float(harvest_df['lost_agb'].fillna(0).sum())
        se_cc = se_lost_agb_national(harvest_df)
        total_flux = nat_flux + cc
        total_se = float(np.sqrt(se_cc ** 2 + se_nat ** 2))
        table = pd.DataFrame([
            {'component': 'Disturbance rate (natural)', 'value': nat_rate, 'se': nat_rate_se, 'unit': '%'},
            {'component': 'Disturbance rate (harvest)', 'value': harv_rate, 'se': harv_rate_se, 'unit': '%'},
            {'component': 'Mean patch size (natural)', 'value': nat_patch, 'se': nat_patch_se, 'unit': 'ha'},
            {'component': 'Mean patch size (harvest)', 'value': harv_patch, 'se': harv_patch_se, 'unit': 'ha'},
            {'component': 'Patch size trend (natural)', 'value': nat_ptev, 'se': nat_ptev_se, 'unit': '%/yr'},
            {'component': 'Patch size trend (harvest)', 'value': harv_ptev, 'se': harv_ptev_se, 'unit': '%/yr'},
            {'component': 'AGB loss (natural)', 'value': nat_agbd, 'se': nat_agbd_se, 'unit': r'MgC ha$_\mathrm{D}^{-1}$'},
            {'component': 'AGB loss (harvest)', 'value': harv_agbd, 'se': harv_agbd_se, 'unit': r'MgC ha$_\mathrm{D}^{-1}$'},
            {'component': 'Relative AGB loss (natural)', 'value': nat_rel_agbd, 'se': nat_rel_agbd_se, 'unit': '%'},
            {'component': 'Relative AGB loss (harvest)', 'value': harv_rel_agbd, 'se': harv_rel_agbd_se, 'unit': '%'},
            {'component': 'Harvest total AGB loss', 'value': cc, 'se': se_cc, 'unit': 'TgC'},
            {'component': 'Natural total AGB loss', 'value': nat_flux, 'se': se_nat, 'unit': 'TgC'},
            {'component': 'Total AGB loss', 'value': total_flux, 'se': total_se, 'unit': 'TgC'},
        ])
        table['pct_error'] = np.where(
            table['value'].abs() > 0,
            100.0 * table['se'] / table['value'].abs(),
            np.nan,
        )
        lo, hi = _clip_national_ci_bounds(table)
        table['ci_95_low'] = lo
        table['ci_95_high'] = hi
        table['ci_95'] = np.maximum(table['value'] - lo, hi - table['value'])
        return table


    def print_national_uncertainty_table(table):
        """Print the national uncertainty budget."""
        print('\nNational uncertainty (TFV-block of the row-level errors)')
        print(f'ci_95: half-width of the 95% CI (± value; z = {NATIONAL_CI_Z_95:.2f}, clipped to physical bounds)')
        out = table[['component', 'value', 'ci_95', 'unit', 'pct_error']].copy()
        print(out.to_string(index=False, formatters={
            'value': lambda x: f'{x:10.3f}',
            'ci_95': lambda x: f'{x:10.3f}' if pd.notna(x) else '        n/a',
            'pct_error': lambda x: f'{x:8.1f}%' if pd.notna(x) else '     n/a',
        }))


    def aggregate_regional_data(df, agent_type):
        """Aggregate disturbance rows to SER.

        Means use the TFV-block SE of the row errors. Relative trends are
        100 × mean(slope) / mean(2020 level). Lost biomass is the sum of the
        row fluxes, with the same TFV-block applied to the row flux errors.
        """
        if agent_type == 'natural':
            filtered_df = df.loc[df.agent == 'natural']
        elif agent_type == 'harvest':
            filtered_df = df.loc[df.agent == 'harvest']
        else:
            raise ValueError(f"agent_type must be 'natural' or 'harvest', got {agent_type!r}")

        fill_values = {
            'd_area_2020_perc': 0, 'd_area_2020_error_perc': 0,
            'slope_pp': 0, 'slope_pp_err': 0, 'd_area_level_slope_cov': 0,
            'mean_patch_size_2020': 0, 'mean_patch_size_2020_error': 0,
            'slope_patch_ha': 0, 'slope_patch_ha_err': 0, 'patch_level_slope_cov': 0,
            'disturbance_rate_2020_error': 0,
            agbd_loss_error_col: 0,
        }
        if RELATIVE_AGBD_LOSS_COLUMN in filtered_df.columns:
            fill_values[RELATIVE_AGBD_LOSS_COLUMN] = 0
        if relative_agbd_loss_error_col in filtered_df.columns:
            fill_values[relative_agbd_loss_error_col] = 0
        if RATE_X_SEVERITY_COLUMN in filtered_df.columns:
            fill_values[RATE_X_SEVERITY_COLUMN] = 0
        if RATE_X_SEVERITY_ERROR_COLUMN in filtered_df.columns:
            fill_values[RATE_X_SEVERITY_ERROR_COLUMN] = 0
        filtered_df = filtered_df.fillna(fill_values)
        if 'lost_agb' in filtered_df.columns:
            filtered_df = filtered_df.fillna({'lost_agb': 0, 'lost_agb_error': 0})

        def _mean_of(value_col, sigma_col):
            def _fn(x):
                if x['total_area'].sum() <= 0:
                    return 0.0
                mean, _ = _mean_tfv(x, value_col, sigma_col)
                return 0.0 if not np.isfinite(mean) else mean
            return _fn

        def _se_of(value_col, sigma_col):
            def _fn(x):
                if x['total_area'].sum() <= 0:
                    return 0.0
                _, se = _mean_tfv(x, value_col, sigma_col)
                return 0.0 if not np.isfinite(se) else se
            return _fn

        def _ratio_mean(value_col):
            spec = _RATIO_OF_MEANS_SPECS[value_col]
            def _fn(x):
                if x['total_area'].sum() <= 0:
                    return 0.0
                ratio, _ = ratio_of_means_stats(x, *spec)
                return 0.0 if not np.isfinite(ratio) else float(ratio)
            return _fn

        def _ratio_se(value_col):
            spec = _RATIO_OF_MEANS_SPECS[value_col]
            def _fn(x):
                if x['total_area'].sum() <= 0:
                    return 0.0
                _, se = ratio_of_means_stats(x, *spec)
                return 0.0 if not np.isfinite(se) else float(se)
            return _fn

        agg_dict = {
            'd_area_2020_perc': _mean_of('d_area_2020_perc', 'd_area_2020_error_perc'),
            'd_area_2020_error_perc': _se_of('d_area_2020_perc', 'd_area_2020_error_perc'),
            'd_area_evolution_perc': _ratio_mean('d_area_evolution_perc'),
            'd_area_evolution_error_perc': _ratio_se('d_area_evolution_perc'),
            'mean_patch_size_2020': _mean_of('mean_patch_size_2020', 'mean_patch_size_2020_error'),
            'mean_patch_size_2020_error': _se_of('mean_patch_size_2020', 'mean_patch_size_2020_error'),
            'mean_patch_size_evolution_perc': _ratio_mean('mean_patch_size_evolution_perc'),
            'mean_patch_size_evolution_error_perc': _ratio_se('mean_patch_size_evolution_perc'),
            AGBD_LOSS_COLUMN: _mean_of(AGBD_LOSS_COLUMN, agbd_loss_error_col),
            agbd_loss_error_col: _se_of(AGBD_LOSS_COLUMN, agbd_loss_error_col),
        }
        if RELATIVE_AGBD_LOSS_COLUMN in filtered_df.columns:
            agg_dict[RELATIVE_AGBD_LOSS_COLUMN] = _mean_of(
                RELATIVE_AGBD_LOSS_COLUMN, relative_agbd_loss_error_col,
            )
        if relative_agbd_loss_error_col in filtered_df.columns:
            agg_dict[relative_agbd_loss_error_col] = _se_of(
                RELATIVE_AGBD_LOSS_COLUMN, relative_agbd_loss_error_col,
            )
        if RATE_X_SEVERITY_COLUMN in filtered_df.columns:
            agg_dict[RATE_X_SEVERITY_COLUMN] = _mean_of(
                RATE_X_SEVERITY_COLUMN, RATE_X_SEVERITY_ERROR_COLUMN,
            )
        if RATE_X_SEVERITY_ERROR_COLUMN in filtered_df.columns:
            agg_dict[RATE_X_SEVERITY_ERROR_COLUMN] = _se_of(
                RATE_X_SEVERITY_COLUMN, RATE_X_SEVERITY_ERROR_COLUMN,
            )
        if ROTATION_PERIOD_COLUMN in filtered_df.columns:
            def _rot_mean(x):
                work = _rotation_mean_subset(x)
                if work.empty or work['total_area'].sum() <= 0:
                    return np.nan
                return float(np.average(work[ROTATION_PERIOD_COLUMN], weights=work['total_area']))
            agg_dict[ROTATION_PERIOD_COLUMN] = _rot_mean
        if ROTATION_PERIOD_ERROR_COLUMN in filtered_df.columns:
            def _rot_se(x):
                work = _rotation_positive_subset(x)
                if work.empty or work['total_area'].sum() <= 0:
                    return 0.0
                se = stderr_area_weighted_mean_tfv_block(work, ROTATION_PERIOD_ERROR_COLUMN)
                return 0.0 if not np.isfinite(se) else float(se)
            agg_dict[ROTATION_PERIOD_ERROR_COLUMN] = _rot_se
        if 'lost_agb' in filtered_df.columns:
            agg_dict['lost_agb'] = lambda x: float(np.sum(x['lost_agb']))
            agg_dict['lost_agb_error'] = lambda x: se_lost_agb_national(x)

        agg_df = (filtered_df.groupby('ser', group_keys=False)
                  .apply(lambda x: pd.Series({k: v(x) for k, v in agg_dict.items()}), include_groups=False)
                  .reset_index())
        for col in ('d_area_2020_perc', 'd_area_2020_error_perc', AGBD_LOSS_COLUMN,
                    agbd_loss_error_col, RATE_X_SEVERITY_COLUMN, RATE_X_SEVERITY_ERROR_COLUMN):
            if col in agg_df.columns:
                agg_df[col] = agg_df[col].clip(lower=0.0)
        _clip_relative_agbd_pct(agg_df)
        return agg_df


    # Aggregate data by region
    forest_trend_natural_agg = aggregate_regional_data(trend_ok, 'natural')
    forest_trend_harvest_agg = aggregate_regional_data(trend_ok, 'harvest')

    # ===============================
    # HELPER FUNCTIONS
    # ===============================

    def _forest_exploitable_masks(exploit, forest, tree_cover=None,
                                  exp_nodata=None, forest_nodata=None, tc_nodata=None):
        """Boolean masks for TFV-valid forest and exploitable pixels at 30 m."""
        valid_exp = np.isfinite(exploit)
        if exp_nodata is not None:
            valid_exp &= ~np.isclose(exploit, exp_nodata)
        is_forest = np.isfinite(forest) & (forest > 0)
        if forest_nodata is not None:
            is_forest &= ~np.isclose(forest, forest_nodata)
        if tree_cover is not None:
            valid_tc = np.isfinite(tree_cover) & (tree_cover > TREE_COVER_LOW_THRESHOLD)
            if tc_nodata is not None:
                valid_tc &= ~np.isclose(tree_cover, tc_nodata)
            is_forest &= valid_tc
        exploitable = is_forest & valid_exp & (exploit == 1)
        return is_forest, exploitable


    def compute_exploitability_fraction_forest_5km():
        """5 km share of exploitable forest pixels (TFV valid, exploitabilite == 1)."""
        with (
            rasterio.open(_EXPLOITABILITY_PATH) as exp_src,
            rasterio.open(_FOREST_TFV_MASK_PATH) as forest_src,
        ):
            crs = exp_src.crs
            scale_factor = max(1, int(round(_5KM_TARGET_RES_M / abs(exp_src.transform.a))))
            out_h = (exp_src.height + scale_factor - 1) // scale_factor
            out_w = (exp_src.width + scale_factor - 1) // scale_factor
            forest_acc = np.zeros((out_h, out_w), dtype=np.float64)
            exploitable_forest_acc = np.zeros((out_h, out_w), dtype=np.float64)
            out_transform = exp_src.transform * Affine.scale(scale_factor)
            exp_nodata = exp_src.nodata
            forest_nodata = forest_src.nodata

            for _, window in exp_src.block_windows(1):
                exploit = exp_src.read(1, window=window).astype(np.float32, copy=False)
                forest = forest_src.read(1, window=window).astype(np.float32, copy=False)
                is_forest, exploitable_forest = _forest_exploitable_masks(
                    exploit, forest,
                    exp_nodata=exp_nodata, forest_nodata=forest_nodata,
                )

                row_off, col_off = window.row_off, window.col_off
                h, w = window.height, window.width
                row_idx = (np.arange(h, dtype=np.int64)[:, None] + row_off) // scale_factor
                col_idx = (np.arange(w, dtype=np.int64)[None, :] + col_off) // scale_factor
                flat_idx = (row_idx * out_w + col_idx).ravel()
                np.add.at(forest_acc.ravel(), flat_idx, is_forest.ravel().astype(np.float64))
                np.add.at(exploitable_forest_acc.ravel(), flat_idx, exploitable_forest.ravel().astype(np.float64))

            frac = np.full((out_h, out_w), np.nan, dtype=np.float32)
            has_forest = forest_acc > 0
            frac[has_forest] = (exploitable_forest_acc[has_forest] / forest_acc[has_forest]).astype(np.float32)
            raster_bounds = array_bounds(out_h, out_w, out_transform)

        return frac, raster_bounds, forest_acc.astype(np.float32), out_transform, crs


    def _load_tree_cover_5km_for_frac(frac_shape):
        """5 km tree cover aligned to the exploitability fraction grid (display mask only)."""
        with rasterio.open(_TREE_COVER_PATH) as tc_src:
            tree_cover_5km = tc_src.read(
                1, out_shape=frac_shape, resampling=Resampling.average,
            ).astype(np.float32)
            if tc_src.nodata is not None:
                tree_cover_5km[np.isclose(tree_cover_5km, tc_src.nodata)] = np.nan
        return tree_cover_5km


    def _save_exploitability_fraction_forest_5km_cache(frac, transform, crs, forest_px_5km):
        os.makedirs(os.path.dirname(_EXPLOITABLE_FRAC_FOREST_5KM_TIF), exist_ok=True)
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1,
            'height': frac.shape[0],
            'width': frac.shape[1],
            'transform': transform,
            'crs': crs,
            'nodata': -9999.0,
            'compress': 'deflate',
        }
        frac_out = np.where(np.isfinite(frac), frac, profile['nodata']).astype(np.float32)
        with rasterio.open(_EXPLOITABLE_FRAC_FOREST_5KM_TIF, 'w', **profile) as dst:
            dst.write(frac_out, 1)
        forest_out = np.where(np.isfinite(forest_px_5km), forest_px_5km, profile['nodata']).astype(np.float32)
        with rasterio.open(_FOREST_PIXEL_COUNT_5KM_TIF, 'w', **profile) as dst:
            dst.write(forest_out, 1)


    def _load_exploitability_fraction_forest_5km_cache():
        with rasterio.open(_EXPLOITABLE_FRAC_FOREST_5KM_TIF) as src:
            frac = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999.0
            frac[np.isclose(frac, nodata)] = np.nan
            raster_bounds = array_bounds(src.height, src.width, src.transform)
        with rasterio.open(_FOREST_PIXEL_COUNT_5KM_TIF) as src:
            forest_px_5km = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999.0
            forest_px_5km[np.isclose(forest_px_5km, nodata)] = np.nan
        tree_cover_5km = _load_tree_cover_5km_for_frac(frac.shape)
        return frac, raster_bounds, forest_px_5km, tree_cover_5km


    def load_or_compute_exploitability_fraction_forest_5km():
        """Load cached 5 km forest exploitability fraction, or compute and save once."""
        cache_files = (
            _EXPLOITABLE_FRAC_FOREST_5KM_TIF,
            _FOREST_PIXEL_COUNT_5KM_TIF,
        )
        if all(os.path.isfile(path) for path in cache_files):
            print(f'Using cached 5 km forest exploitability fraction: {_EXPLOITABLE_FRAC_FOREST_5KM_TIF}')
            return _load_exploitability_fraction_forest_5km_cache()
        print('Computing 5 km forest exploitability fraction (cached for next runs)...')
        frac, raster_bounds, forest_px_5km, transform, crs = (
            compute_exploitability_fraction_forest_5km()
        )
        _save_exploitability_fraction_forest_5km_cache(
            frac, transform, crs, forest_px_5km,
        )
        tree_cover_5km = _load_tree_cover_5km_for_frac(frac.shape)
        return frac, raster_bounds, forest_px_5km, tree_cover_5km


    def plot_exploitability_fraction_forest_5km(ax, frac, raster_bounds, forest_px_5km, tree_cover_5km,
                                                cbar_tick_fontsize=_MAP_CBAR_TICK_FONTSIZE):
        """5 km forest exploitability fraction; tree cover <= 10% and non-forest cells masked transparent."""
        left, bottom, right, top = raster_bounds
        extent = [left, right, bottom, top]
        low_tc = (np.isnan(tree_cover_5km)) | (tree_cover_5km <= TREE_COVER_LOW_THRESHOLD)
        no_forest = (np.isnan(forest_px_5km)) | (forest_px_5km <= 0)

        france_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=1)
        ax.set_facecolor('none')

        valid_frac = np.isfinite(frac) & ~no_forest & ~low_tc
        if not np.any(valid_frac):
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center')
            return

        vmin, vmax = 0.0, 1.0
        cmap_obj = mpl.colormaps['viridis_r'].copy()
        cmap_obj.set_bad((0, 0, 0, 0))
        display = np.ma.masked_where(~valid_frac, frac)
        im = ax.imshow(
            display, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=extent,
            interpolation='none', aspect='equal', zorder=2,
        )
        france_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=4)

        ax._map_cbar_spec = {
            'mappable': im,
            'ticks': [0, 1],
            'labels': ['0%', '100%'],
            'cbar_tick_fontsize': cbar_tick_fontsize,
        }
        ax.set_axis_off()


    def _exploitability_region_masks(nomser):
        """Boolean masks for manuscript exploitability regions (NomSER patterns)."""
        return {
            'National': np.ones(len(nomser), dtype=bool),
            'Alps': nomser.str.contains('Alpes', case=False, na=False).to_numpy(),
            'Pyrénées': nomser.str.contains('Pyr', case=False, na=False).to_numpy(),
            'Corsica': nomser.str.contains('Corse', case=False, na=False).to_numpy(),
        }


    def summarize_exploitability_fraction_forest_5km(frac, forest_px_5km, raster_bounds, ser_gdf=None):
        """National/regional forest exploitability (%), weighted by 30 m forest pixel count per 5 km cell."""
        if ser_gdf is None:
            ser_gdf = france_ser
        left, bottom, right, top = raster_bounds
        h, w = frac.shape
        xs = np.linspace(left, right, w, endpoint=False) + (right - left) / (2 * w)
        ys = np.linspace(top, bottom, h, endpoint=False) - (top - bottom) / (2 * h)
        xx, yy = np.meshgrid(xs, ys)
        forest_mask = np.isfinite(frac) & np.isfinite(forest_px_5km) & (forest_px_5km > 0)
        if not np.any(forest_mask):
            return pd.DataFrame()

        ptg = gpd.GeoDataFrame(
            {
                'exp_frac': frac[forest_mask].ravel(),
                'forest_px': forest_px_5km[forest_mask].ravel(),
            },
            geometry=gpd.points_from_xy(xx[forest_mask], yy[forest_mask]),
            crs=ser_gdf.crs,
        )
        joined = gpd.sjoin_nearest(
            ptg, ser_gdf[['ser_num', 'NomSER', 'geometry']], how='left',
        )
        rows = []
        for region, mask in _exploitability_region_masks(joined['NomSER']).items():
            sub = joined.loc[mask]
            if sub.empty or sub['forest_px'].sum() <= 0:
                continue
            pct = 100.0 * float(np.average(sub['exp_frac'], weights=sub['forest_px']))
            rows.append({'region': region, 'pct_exploitable': pct, 'n_cells': len(sub)})
        return pd.DataFrame(rows)


    def print_exploitability_manuscript_summary(summary_df):
        """Print forest exploitability percentages for manuscript text."""
        if summary_df is None or summary_df.empty:
            return
        lookup = summary_df.set_index('region')['pct_exploitable']
        print('\nForest exploitability (5 km, TFV forest, forest-pixel weighted):')
        for region in ('National', 'Alps', 'Pyrénées', 'Corsica'):
            if region in lookup.index:
                print(f'  {region}: {lookup[region]:.0f}%')


    def _fmt_ci95(mean, se, mean_nd=1, half_nd=None):
        """Format mean ± 95% CI half-width for manuscript text."""
        half = NATIONAL_CI_Z_95 * float(se)
        if half_nd is None:
            half_nd = 2 if half < 1.0 else 0
        return f'{float(mean):.{mean_nd}f}±{half:.{half_nd}f}'


    def print_manuscript_patch_rotation_summary(harvest_df, natural_df):
        """Print patch size and rotation period with 95% CI half-widths for manuscript text.

        Figures plot mean ± SE; manuscript quotes use ±1.96×SE (95% CI).
        """
        harv_patch, harv_patch_se = national_mean_patch_size_mean_stderr(harvest_df)
        nat_patch, nat_patch_se = national_mean_patch_size_mean_stderr(natural_df)
        harv_ptev, harv_ptev_se = national_mean_patch_size_evolution_mean_stderr(harvest_df)
        nat_ptev, nat_ptev_se = national_mean_patch_size_evolution_mean_stderr(natural_df)
        print('\nManuscript patch size (ha$_\\mathrm{D}$, 95% CI):')
        print(f'  Harvest national: {_fmt_ci95(harv_patch, harv_patch_se)}')
        print(f'  Natural national: {_fmt_ci95(nat_patch, nat_patch_se)}')
        print('\nManuscript patch size trend relative to 2020 (%/yr, 95% CI):')
        print(f'  Harvest national: {_fmt_ci95(harv_ptev, harv_ptev_se)}')
        print(f'  Natural national: {_fmt_ci95(nat_ptev, nat_ptev_se)}')

        mp = harvest_df[(harvest_df['TFV'] == 51) & (harvest_df['ser'] == 621)]
        if not mp.empty:
            mp_mean = float(np.average(mp['mean_patch_size_2020'], weights=mp['total_area']))
            print(f'  Maritime pine, Landes (TFV 51, SER 621): {mp_mean:.2f} ha$_\\mathrm{{D}}$ (area-weighted mean)')

        rot = harvest_df.dropna(subset=[ROTATION_PERIOD_COLUMN])
        rot = rot[rot[ROTATION_PERIOD_COLUMN] > 0]
        if rot.empty:
            return
        nat_rot, nat_rot_se = national_area_weighted_mean_stderr(
            rot, ROTATION_PERIOD_COLUMN, ROTATION_PERIOD_ERROR_COLUMN,
        )
        print('\nManuscript rotation period (years, 95% CI = 1.96×SE; figures use ±SE):')
        print(f'  National: {_fmt_ci95(nat_rot, nat_rot_se, mean_nd=0)}')

        # SER extremes used in the manuscript (harvest SER means).
        if 'ser' in rot.columns:
            ser_stats = []
            for ser, g in rot.groupby('ser'):
                st = _rotation_weighted_stats(g)
                if np.isfinite(st['mean']) and st['mean'] > 0:
                    ser_stats.append((ser, float(st['mean']), float(st['std'])))
            if ser_stats:
                ser_df = pd.DataFrame(ser_stats, columns=['ser', 'mean', 'se'])
                # Attach NomSER when available from france_ser.
                if 'france_ser' in globals() and 'NomSER' in france_ser.columns:
                    names = france_ser[['ser_num', 'NomSER']].drop_duplicates('ser_num')
                    ser_df = ser_df.merge(names, left_on='ser', right_on='ser_num', how='left')
                else:
                    ser_df['NomSER'] = ser_df['ser'].astype(str)
                top = ser_df.nlargest(3, 'mean')
                bot = ser_df.nsmallest(3, 'mean')
                print('  Longest SERs:')
                for _, r in top.iterrows():
                    print(f"    {r['NomSER']}: {_fmt_ci95(r['mean'], r['se'], mean_nd=0)}")
                print('  Shortest SERs:')
                for _, r in bot.iterrows():
                    print(f"    {r['NomSER']}: {_fmt_ci95(r['mean'], r['se'], mean_nd=0)}")

        rot = rot.copy()
        rot['TFV_group'] = rot['TFV'].map(dict_group_tfvs)
        print('  TFV groups:')
        for g in TFV_GROUP_ORDER:
            grp = rot.loc[rot['TFV_group'] == g]
            if grp.empty:
                continue
            st = _rotation_weighted_stats(grp)
            print(f"    {g}: {_fmt_ci95(st['mean'], st['std'], mean_nd=0)}")

        print('  Species:')
        for tfv, label in (
            (19, 'Poplar'), (51, 'Maritime pine'), (63, 'Larch'), (9, 'Beech'),
        ):
            grp = rot.loc[rot['TFV'] == tfv]
            if grp.empty:
                continue
            st = _rotation_weighted_stats(grp)
            print(f"    {label}: {_fmt_ci95(st['mean'], st['std'], mean_nd=0)}")

    # %%

    def _shared_map_vlim(*series_sources):
        """Shared (vmin, vmax) = (min of mins, max of maxes) across GeoDataFrame columns.

        Each argument is ``(gdf, column)``. Returns ``(None, None)`` if no valid data.
        """
        mins, maxs = [], []
        for gdf, column in series_sources:
            if gdf is None or gdf.empty or column not in gdf.columns:
                continue
            valid = gdf[column].dropna()
            if valid.empty:
                continue
            mins.append(float(valid.min()))
            maxs.append(float(valid.max()))
        if not mins:
            return None, None
        return min(mins), max(maxs)


    def _shared_map_vlim_second_max(*series_sources):
        """Shared (vmin, vmax) with vmax = second-largest finite value across sources.

        Falls back to the maximum when fewer than two distinct values exist.
        Each argument is ``(gdf, column)``. Returns ``(None, None)`` if no valid data.
        """
        vals = []
        for gdf, column in series_sources:
            if gdf is None or gdf.empty or column not in gdf.columns:
                continue
            valid = pd.to_numeric(gdf[column], errors='coerce').dropna()
            if valid.empty:
                continue
            vals.append(valid.to_numpy(dtype=float))
        if not vals:
            return None, None
        arr = np.concatenate(vals)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None
        vmin = float(arr.min())
        uniq = np.unique(arr)
        vmax = float(uniq[-2]) if uniq.size >= 2 else float(uniq[-1])
        return vmin, vmax


    def plot_map(gdf, column, cmap, ax, title, vmin=None, vmax=None, cbar_tick_fontsize=_MAP_CBAR_TICK_FONTSIZE,
                 cbar_center_tick=True, add_cbar=False):
        """Plot a choropleth map with consistent styling.

        Colorbar is deferred by default (``add_cbar=False``) so callers can finalize
        layout first, then attach with ``_attach_deferred_map_colorbars`` (MT panel i).
        """

        # Check if GeoDataFrame is empty or column doesn't exist
        if gdf.empty or column not in gdf.columns:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center')
            return

        # Filter out rows where column is NaN for min/max calculation
        valid_data = gdf[column].dropna()
        if valid_data.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center')
            return

        # Check if this is an evolution plot or area plot (row 1)
        is_evolution = (
            'evolution' in column.lower()
            or 'evolution' in title.lower()
            or 'trend relative' in title.lower()
        )
        is_area = 'd_area_2020' in column
        is_error = 'error' in title.lower() or 'uncertainty' in title.lower()

        if vmin is not None and vmax is not None:
            actual_vmin, actual_vmax = vmin, vmax
        else:
            data_min, data_max = valid_data.min(), valid_data.max()
            actual_vmin, actual_vmax = data_min, data_max

        # Handle 'spectral' -> 'Spectral' conversion
        if cmap.lower() == 'spectral':
            cmap = 'Spectral_r'

        # Get colormap
        cmap_obj = mpl.colormaps[cmap].copy()
        cmap_obj.set_bad('none')  # Make no-data values transparent

        # For area plots (row 1), use TwoSlopeNorm (but not for error plots)
        # Natural area map (subplot a) uses vcenter=0.2, harvest uses 0.3
        vcenter = None
        if is_area and not is_error:
            if 'natural' in title.lower():
                vcenter = 0.2
            else:
                vcenter = 0.3
            # Only use TwoSlopeNorm if vcenter is between vmin and vmax
            if actual_vmin < vcenter < actual_vmax:
                norm = TwoSlopeNorm(vmin=actual_vmin, vcenter=vcenter, vmax=actual_vmax)
            else:
                # Fall back to normal normalization if vcenter is not valid
                norm = Normalize(vmin=actual_vmin, vmax=actual_vmax)
                vcenter = None
        # For evolution plots, always use TwoSlopeNorm centered at 0 (but not for error plots)
        # Extend vmin below 0 if needed for proper centering
        elif is_evolution and not is_error:
            if actual_vmin > 0:
                # Set vmin to a negative value to allow centering at 0
                norm_vmin = -actual_vmax  # Symmetric range around 0
            else:
                norm_vmin = actual_vmin
            norm = TwoSlopeNorm(vmin=norm_vmin, vcenter=0, vmax=actual_vmax)
        else:
            norm = Normalize(vmin=actual_vmin, vmax=actual_vmax)

        # Plot without automatic legend, then create custom colorbar (no SER edge lines).
        gdf.plot(
            column=column, cmap=cmap_obj, ax=ax, legend=False, norm=norm,
            edgecolor='none', linewidth=0, missing_kwds={'color': 'lightgray'},
        )

        # Check if any collections were created before creating colorbar
        if len(ax.collections) == 0:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center')
            return

        # Set custom ticks
        if is_area and not is_error and vcenter is not None and actual_vmin < vcenter < actual_vmax:
            # For area plots (row 1): include vcenter as center (unless center tick suppressed)
            if cbar_center_tick:
                ticks = [actual_vmin, vcenter, actual_vmax]
                labels = [f'{actual_vmin:.2f}%', f'{vcenter:.1f}%', f'{actual_vmax:.2f}%']
            else:
                ticks = [actual_vmin, actual_vmax]
                labels = [f'{actual_vmin:.2f}%', f'{actual_vmax:.2f}%']
        elif is_evolution and not is_error:
            # For evolution plots: always include 0 in the middle
            ticks = [actual_vmin, 0, actual_vmax]
            labels = [f'{actual_vmin:.0f}%/yr', '0%/yr', f'{actual_vmax:.0f}%/yr']
        elif is_evolution and is_error:
            # For evolution error plots: use %/yr unit
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.0f}%/yr', f'{actual_vmax:.0f}%/yr']
        elif 'rel_unc' in column.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.0f}%', f'{actual_vmax:.0f}%']
        elif 'relative' in column.lower() or 'relative' in title.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.0f}%', f'{actual_vmax:.0f}%']
        elif 'rotation' in column.lower() or 'rotation' in title.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.0f} yr', f'{actual_vmax:.0f} yr']
        elif 'lost_agb' in column.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.2f} TgC', f'{actual_vmax:.2f} TgC']
        elif (
            column == RATE_X_SEVERITY_COLUMN
            or 'rate × severity' in title.lower()
            or 'rate x severity' in title.lower()
            or 'agb-loss intensity' in title.lower()
            or 'agb loss intensity' in title.lower()
        ):
            ticks = [actual_vmin, actual_vmax]
            labels = [
                f'{actual_vmin:.2f} MgC\u00a0ha$_\\mathrm{{F}}^{{-1}}$',
                f'{actual_vmax:.2f} MgC\u00a0ha$_\\mathrm{{F}}^{{-1}}$',
            ]
        elif 'mean_patch_size_2020' in column.lower() or 'patch size' in title.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.2f} ha$_\\mathrm{{D}}$', f'{actual_vmax:.2f} ha$_\\mathrm{{D}}$']
        elif 'error_perc' in column.lower() or 'area' in title.lower() or 'disturbed' in title.lower():
            ticks = [actual_vmin, actual_vmax]
            labels = [f'{actual_vmin:.1f}%', f'{actual_vmax:.1f}%']
        else:
            ticks = [actual_vmin, actual_vmax]
            labels = [
                f'{actual_vmin:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
                f'{actual_vmax:.0f} MgC\u00a0ha$_\\mathrm{{D}}^{{-1}}$',
            ]

        if add_cbar:
            _attach_map_colorbar(ax, ax.collections[0], ticks, labels, cbar_tick_fontsize=cbar_tick_fontsize)
        else:
            ax._map_cbar_spec = {
                'mappable': ax.collections[0],
                'ticks': ticks,
                'labels': labels,
                'cbar_tick_fontsize': cbar_tick_fontsize,
            }

        ax.set_axis_off()


    def _attach_map_colorbar(ax, mappable, ticks, labels, cbar_tick_fontsize=_MAP_CBAR_TICK_FONTSIZE,
                             cbar_x=_MAP_CBAR_X, cbar_y=_MAP_CBAR_Y,
                             cbar_width=_MAP_CBAR_WIDTH, cbar_height=_MAP_CBAR_HEIGHT):
        """Inset vertical colorbar using the current axes position in figure coordinates."""
        fig = ax.get_figure()
        pos = ax.get_position()
        fig_x = pos.x0 + cbar_x * pos.width
        fig_y = pos.y0 + cbar_y * pos.height - cbar_height * pos.height / 2
        fig_width = cbar_width * pos.width
        fig_height = cbar_height * pos.height
        cbar_ax = fig.add_axes([fig_x, fig_y, fig_width, fig_height])
        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels, fontsize=cbar_tick_fontsize)
        cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        return cbar


    def _attach_deferred_map_colorbars(axes):
        """Add colorbars stored by plot_map / exploitability after layout is finalized."""
        fig = None
        for ax in axes:
            spec = getattr(ax, '_map_cbar_spec', None)
            if spec is None:
                continue
            mappable = spec.get('mappable')
            if mappable is None:
                if not ax.collections:
                    continue
                mappable = ax.collections[0]
            fig = ax.get_figure()
            _attach_map_colorbar(
                ax, mappable, spec['ticks'], spec['labels'],
                cbar_tick_fontsize=spec.get('cbar_tick_fontsize', _MAP_CBAR_TICK_FONTSIZE),
                cbar_x=spec.get('cbar_x', _MAP_CBAR_X),
                cbar_y=spec.get('cbar_y', _MAP_CBAR_Y),
                cbar_width=spec.get('cbar_width', _MAP_CBAR_WIDTH),
                cbar_height=spec.get('cbar_height', _MAP_CBAR_HEIGHT),
            )
            del ax._map_cbar_spec
        if fig is not None:
            fig.canvas.draw_idle()


    def _build_sorted_tfvs_by_total_area(natural_data, harvest_data):
        """TFV order for SI charts: ascending SER×TFV ``total_area`` (disturbance mask).

        Area is taken once per TFV×SER (not summed across agents). With the gap-fill
        fix, ``total_area`` is always the SER forest mask — same source as map weights.
        """
        parts = []
        for df in (natural_data, harvest_data):
            if df is None or df.empty or 'total_area' not in df.columns:
                continue
            cols = [c for c in ('TFV', 'ser', 'total_area') if c in df.columns]
            if 'TFV' not in cols or 'total_area' not in cols:
                continue
            parts.append(df[cols].drop_duplicates(subset=[c for c in ('TFV', 'ser') if c in cols]))
        if not parts:
            return []
        area = pd.concat(parts, ignore_index=True)
        if 'ser' in area.columns:
            area = area.drop_duplicates(subset=['TFV', 'ser'])
        else:
            area = area.drop_duplicates(subset=['TFV'])
        by_tfv = (
            area.groupby('TFV', as_index=True)['total_area']
            .sum()
            .sort_values(ascending=True)
        )
        ordered = []
        for tfv in by_tfv.index:
            try:
                tfv_f = float(tfv)
            except (TypeError, ValueError):
                continue
            if tfv_f in dict_group_en:
                ordered.append(int(tfv_f) if tfv_f == int(tfv_f) else tfv_f)
        print(f'SI TFV order from disturbance total_area ({len(ordered)} TFVs)')
        return ordered


    def _relative_uncertainty_pct(values, errors):
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(values) > 1e-12, errors / np.abs(values) * 100.0, np.nan)


    def _append_relative_uncertainty_columns(
        gdf, pairs, area_col='d_area_2020_perc',
        area_thr=_REL_UNC_AREA_MASK_PCT, rel_thr=_REL_UNC_MASK_PCT,
    ):
        """Add relative-uncertainty (%) columns."""
        out = gdf.copy()
        for value_col, error_col, out_col in pairs:
            if value_col not in out.columns or error_col not in out.columns:
                continue
            out[out_col] = _relative_uncertainty_pct(
                out[value_col].to_numpy(), out[error_col].to_numpy(),
            )
        return out


    def _is_populated_map_ax(ax):
        return ax is not None and bool(getattr(ax, '_map_cbar_spec', None) or ax.collections)


    def _is_labelable_panel_ax(ax):
        """True for maps with data (often axis-off) or live chart axes; false for spacers."""
        if ax is None:
            return False
        if _is_populated_map_ax(ax):
            return True
        return bool(ax.axison)


    def _finalize_relative_uncertainty_maps_figure(fig, axes_grid, row_titles, save_path,
                                                   col_headers=('', '')):
        """Layout, titles, and inset colorbars for SI7/SI8 uncertainty map grids.

        ``row_titles`` entries are either a shared string or ``(title_h, title_n)``
        matching the corresponding main-figure map titles.
        """
        map_axes = []
        for ax_h, ax_n in axes_grid:
            if _is_populated_map_ax(ax_h):
                map_axes.append(ax_h)
            if _is_populated_map_ax(ax_n):
                map_axes.append(ax_n)

        # Narrow figure + near-zero wspace: equal-aspect France maps leave side padding
        # inside wide cells, which reads as a gap between the two columns.
        plt.tight_layout(pad=0.2)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, hspace=0.18, wspace=0.0)
        fig.canvas.draw()

        ax_h0, ax_n0 = axes_grid[0]
        header_y_pad = 0.042 if any(col_headers) else 0.012
        for ax, header in ((ax_h0, col_headers[0]), (ax_n0, col_headers[1])):
            if _is_populated_map_ax(ax) and header:
                pos = ax.get_position()
                fig.text((pos.x0 + pos.x1) / 2, pos.y1 + header_y_pad, header,
                         ha='center', va='bottom', fontsize=16, clip_on=False)

        for row_i, (ax_h, ax_n) in enumerate(axes_grid):
            if row_i >= len(row_titles):
                continue
            titles = row_titles[row_i]
            if isinstance(titles, (tuple, list)):
                title_h = titles[0] if len(titles) > 0 else None
                title_n = titles[1] if len(titles) > 1 else None
            else:
                title_h = title_n = titles
            if _is_populated_map_ax(ax_h) and title_h:
                _place_fig_map_title(fig, ax_h, title_h)
            if _is_populated_map_ax(ax_n) and title_n:
                _place_fig_map_title(fig, ax_n, title_n)

        _attach_deferred_map_colorbars(map_axes)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.30)
        fig.show()
        plt.show(block=False)


    def plot_relative_uncertainty_map(gdf, column, ax, panel_label=None, vmin=None, vmax=None):
        """Choropleth of relative uncertainty (%); missing SERs plot as grey."""
        if gdf.empty or column not in gdf.columns:
            ax.set_axis_off()
            return
        plot_map(gdf, column, 'hot_r', ax, 'Relative uncertainty (%)',
                 vmin=vmin, vmax=vmax, cbar_center_tick=False, add_cbar=False)
        _overlay_map_borders(ax)
        if panel_label:
            _place_subplot_label(ax, panel_label, fontsize=14, style='map')


    def _apply_fixed_ylim_ticks(ax, y_min, y_max, tick_positions, tick_labels=None):
        ax.set_ylim(y_min, y_max)
        ax._ylim_fixed = (float(y_min), float(y_max))
        ax._independent_ylim = True
        if tick_labels is None:
            tick_labels = [f'{t:g}' for t in tick_positions]
        _set_custom_yticks(ax, tick_positions, tick_labels)
        ax._y_tick_label_strings = list(tick_labels)
        _force_plain_y_ticks(ax)


    def _value_axis_is_x(ax):
        """True when SI TFV panels plot values on x (categories on y)."""
        return bool(getattr(ax, '_si_horizontal', False))


    def _set_value_axis_lim(ax, lo, hi):
        lo, hi = float(lo), float(hi)
        if _value_axis_is_x(ax):
            ax.set_xlim(lo, hi)
            ax._xlim_fixed = (lo, hi)
            ax._ylim_fixed = (lo, hi)  # lim helpers / annotate still read _ylim_fixed
        else:
            ax.set_ylim(lo, hi)
            ax._ylim_fixed = (lo, hi)


    def _set_value_axis_locator(ax, n_ticks=5, min_n_ticks=None):
        if min_n_ticks is None:
            min_n_ticks = n_ticks
        locator = MaxNLocator(nbins=n_ticks, min_n_ticks=min_n_ticks)
        if _value_axis_is_x(ax):
            ax.xaxis.set_major_locator(locator)
            _force_plain_x_ticks(ax)
        else:
            ax.yaxis.set_major_locator(locator)
            _force_plain_y_ticks(ax)


    def _apply_ylim_from_error_specs(ax, y_min=0.0, pad_frac=0.08, n_ticks=5):
        """Set value-axis lim from plotted error-bar tops (and optional floor), with padded headroom."""
        specs = getattr(ax, '_clipped_error_specs', None) or []
        if specs:
            y_top = max(s['top'] for s in specs if np.isfinite(s['top']))
            y_top = y_top + pad_frac * max(y_top - y_min, 1e-9)
        else:
            y_top = y_min + 1.0
        _set_value_axis_lim(ax, y_min, y_top)
        ax._independent_ylim = True
        _set_value_axis_locator(ax, n_ticks=n_ticks)


    def _apply_ylim_from_error_specs_both(ax, pad_frac=0.08, n_ticks=5):
        """Set value-axis lim to include all error-bar tops and bottoms, with padding."""
        specs = getattr(ax, '_clipped_error_specs', None) or []
        if specs:
            y_lo = min(s['bottom'] for s in specs if np.isfinite(s['bottom']))
            y_hi = max(s['top'] for s in specs if np.isfinite(s['top']))
            span = y_hi - y_lo
            pad = pad_frac * span if span > 0 else 0.01
            y_min, y_max = y_lo - pad, y_hi + pad
            _set_value_axis_lim(ax, y_min, y_max)
            ax._independent_ylim = True
        _set_value_axis_locator(ax, n_ticks=n_ticks)


    def _apply_nice_yticks(ax, nbins=4):
        """Snap value-axis ticks to round 1–2–5 steps within the current lim."""
        locator = MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10], min_n_ticks=3)
        if _value_axis_is_x(ax):
            ax.xaxis.set_major_locator(locator)
            _force_plain_x_ticks(ax)
        else:
            ax.yaxis.set_major_locator(locator)
            _force_plain_y_ticks(ax)


    def _lock_adaptive_dot_ylim(ax, pad_frac=0.05, y_floor=None):
        """Adaptive value-axis lim from whisker min/max with a small buffer.

        ``y_floor`` only expands the data range (e.g. include 0); omit it to fit content.
        """
        specs = getattr(ax, '_clipped_error_specs', None) or []
        if specs:
            y_lo = min(s['bottom'] for s in specs if np.isfinite(s['bottom']))
            y_hi = max(s['top'] for s in specs if np.isfinite(s['top']))
        else:
            return
        if y_floor is not None:
            y_lo = min(y_lo, y_floor)
        span = y_hi - y_lo
        pad = pad_frac * span if span > 0 else max(abs(y_hi), abs(y_lo), 1.0) * pad_frac
        y_min, y_max = y_lo - pad, y_hi + pad
        _set_value_axis_lim(ax, y_min, y_max)
        ax._independent_ylim = True
        ax._include_zero_in_ylim = False
        _set_value_axis_locator(ax, n_ticks=5, min_n_ticks=3)


    def _lock_restrictive_dot_ylim(ax, pad_frac=0.05, include_zero=True, n_ticks=5,
                                   pad_below=True, pad_frac_below=None, pad_frac_above=None):
        """Tighter value-axis lim from means only (+ buffer); clipped whiskers get annotated later.

        ``pad_frac_below`` / ``pad_frac_above`` override ``pad_frac`` per side.
        If ``pad_below`` is False, the lower limit is not padded.
        """
        limits = _axis_dot_mean_limits(ax)
        if limits is None:
            return
        y_lo, y_hi = float(limits[0]), float(limits[1])
        if include_zero:
            y_lo = min(y_lo, 0.0)
            y_hi = max(y_hi, 0.0)
        span = y_hi - y_lo
        base = span if span > 0 else max(abs(y_hi), abs(y_lo), 1.0)
        frac_lo = pad_frac if pad_frac_below is None else pad_frac_below
        frac_hi = pad_frac if pad_frac_above is None else pad_frac_above
        pad_lo = frac_lo * base
        pad_hi = frac_hi * base
        y_min = y_lo - pad_lo if pad_below else y_lo
        y_max = y_hi + pad_hi
        _set_value_axis_lim(ax, y_min, y_max)
        ax._independent_ylim = True
        ax._annotate_clipped_errors = True
        _set_value_axis_locator(ax, n_ticks=n_ticks)


    def _tfv_xtick_labels(sorted_tfvs):
        return [dict_group_en.get(float(tfv), str(tfv)) for tfv in sorted_tfvs]


    def _format_panel_title(title, max_chars=42):
        """Two-line title, preferring a split before 'relative to' or a trailing unit."""
        if not title or '\n' in title:
            return title
        lower = title.lower()
        marker = ' relative to '
        if marker in lower:
            idx = lower.index(marker)
            return title[:idx] + '\n' + title[idx + 1:].lstrip()
        return _wrap_row_title(title, max_chars=max_chars)


    def _protect_parenthetical_units(title):
        """Replace spaces inside trailing ``(...)`` so wraps cannot split the unit."""
        start = title.rfind(' (')
        if start < 0 or not title.endswith(')'):
            return title, None
        unit = title[start + 1:]  # includes leading '('
        if ' ' not in unit:
            return title, None
        token = unit.replace(' ', '\u00a0')
        return title[:start + 1] + token, token


    def _wrap_row_title(title, max_chars=32):
        """Split long row titles onto two lines; keep parenthetical units on one line."""
        if not title or '\n' in title or len(title) <= max_chars:
            return title
        # Prefer: name on line 1, full unit "(...)" on line 2.
        unit_start = title.rfind(' (')
        if unit_start > 0 and title.endswith(')') and unit_start <= max_chars + 8:
            head = title[:unit_start].rstrip()
            unit = title[unit_start + 1:]  # "(MgC ha_...)"
            if head and ' ' not in unit.replace('\u00a0', ''):
                # unit may still contain normal spaces — keep it intact on line 2
                return head + '\n' + title[unit_start + 1:]
            if head:
                return head + '\n' + title[unit_start + 1:].replace(' ', '\u00a0')

        protected, _ = _protect_parenthetical_units(title)
        words = protected.split(' ')
        if len(words) < 2:
            return title
        target = len(protected) / 2
        line1, length = [], 0
        split_at = len(words)
        for i, word in enumerate(words):
            add = len(word) + (1 if line1 else 0)
            if line1 and length + add > target + 4:
                split_at = i
                break
            line1.append(word)
            length += add
        else:
            split_at = len(words)
        if split_at <= 0 or split_at >= len(words):
            return title
        return ' '.join(words[:split_at]) + '\n' + ' '.join(words[split_at:])


    def _format_si_ylabel(title, max_chars=28):
        """Two-line SI axis title when the label is long enough to risk overlap."""
        return _wrap_row_title(title, max_chars=max_chars)


    def _format_si_xlabel(title):
        """SI value-axis xlabel: keep parenthetical units on one line (never split)."""
        if not title:
            return title
        unit_start = title.rfind(' (')
        if unit_start > 0 and title.endswith(')'):
            head = title[:unit_start].rstrip()
            unit = title[unit_start + 1:].replace(' ', '\u00a0')
            # Short enough → single line; else name above, intact unit below.
            if len(head) + 1 + len(unit) <= 40:
                return head + ' ' + unit
            if len(head) > 28:
                head = _wrap_row_title(head, max_chars=28)
            return head + '\n' + unit
        protected, _ = _protect_parenthetical_units(title)
        return protected


    def _si_ylabel_bboxes(fig, axes, renderer):
        bboxes = []
        for ax in axes:
            if not ax.yaxis.get_label().get_text().strip():
                bboxes.append(None)
                continue
            bb = ax.yaxis.get_label().get_window_extent(renderer).transformed(fig.transFigure.inverted())
            bboxes.append(bb)
        return bboxes


    def _si_ylabels_overlap(bboxes):
        for i in range(len(bboxes) - 1):
            if bboxes[i] is not None and bboxes[i + 1] is not None and bboxes[i].overlaps(bboxes[i + 1]):
                return True
        return False


    def _ensure_si_ylabels_no_overlap(fig, axes, wrap_chars=(36, 28, 22)):
        """Wrap stacked SI y-axis titles onto two lines when they overlap."""
        active = [ax for ax in axes if ax.yaxis.get_label().get_text().strip()]
        if not active:
            return
        for max_chars in wrap_chars:
            for ax in active:
                raw = getattr(ax, '_si_ylabel_raw', None)
                if raw is None:
                    raw = ax.yaxis.get_label().get_text().replace('\n', ' ').strip()
                    ax._si_ylabel_raw = raw
                ax.yaxis.get_label().set_text(_format_si_ylabel(raw, max_chars=max_chars))
            fig.align_ylabels(axes)
            fig.canvas.draw()
            if not _si_ylabels_overlap(_si_ylabel_bboxes(fig, axes, fig.canvas.get_renderer())):
                return


    def _place_fig_map_title(fig, ax, title, y_pad=0.012):
        """Map title above the axes in figure coordinates."""
        if not title:
            return
        pos = ax.get_position()
        fig.text(
            0.5 * (pos.x0 + pos.x1), pos.y1 + y_pad, _format_panel_title(title),
            fontsize=16, ha='center', va='bottom', clip_on=False,
        )


    def _si_tfv_category_ylim(n_tfvs):
        """Category-axis lim with first TFV at the bottom (SI1: ascending area)."""
        n = int(n_tfvs)
        return (-0.6, n - 0.4)


    def _apply_si_tfv_yticklabels(axes, sorted_tfvs):
        """Set shared TFV y-tick labels on the leftmost panel only (sharey-safe)."""
        axes = np.atleast_1d(np.asarray(axes)).ravel()
        if sorted_tfvs is None or len(axes) == 0:
            return
        y_positions = np.arange(len(sorted_tfvs))
        labels = _tfv_xtick_labels(sorted_tfvs)
        ylim = _si_tfv_category_ylim(len(sorted_tfvs))
        for i, ax in enumerate(axes):
            ax.set_yticks(y_positions)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_yticklabels(labels, fontsize=_SI_YTICK_LABELSIZE)
                ax.tick_params(axis='y', which='major', labelleft=True,
                               labelsize=_SI_YTICK_LABELSIZE, length=3)
            else:
                # Do not call set_yticklabels([]) — with sharey it clears ax0 labels.
                ax.tick_params(axis='y', which='major', labelleft=False, length=0)


    def _style_si_tfv_axis(ax):
        """Shared SI TFV axis styling (Fig_age_distribution simplified look)."""
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)


    def _finalize_si_tfv_vertical_figure(fig, axes, save_path, sorted_tfvs=None):
        """Panel labels, shared TFV y-labels, and legend for horizontal SI TFV panels."""
        axes = np.atleast_1d(np.asarray(axes)).ravel()
        label_axes = [ax for ax in axes if _is_labelable_panel_ax(ax)]
        panel_labels = _subplot_labels_sequential(len(label_axes))
        for ax, panel_label in zip(label_axes, panel_labels):
            _style_si_tfv_axis(ax)
            _place_subplot_label(ax, panel_label, fontsize=_SI_PANEL_LABEL_FONTSIZE, style='si')
        for ax in axes:
            if ax not in label_axes:
                _style_si_tfv_axis(ax)
        _apply_si_tfv_yticklabels(axes, sorted_tfvs)
        plt.tight_layout()
        fig.subplots_adjust(top=0.90, bottom=0.14, left=0.28, right=0.98, wspace=0.18)
        # Re-apply after layout: tight_layout / sharey can drop y-tick labels.
        _apply_si_tfv_yticklabels(axes, sorted_tfvs)
        bbox_top = max(ax.get_position().y1 for ax in axes)
        fig.legend(
            handles=_SI_LEGEND_ELEMENTS, loc='lower center',
            bbox_to_anchor=(0.5, bbox_top + 0.02), bbox_transform=fig.transFigure,
            ncol=2, fontsize=_SI_LEGEND_FONTSIZE, frameon=False,
        )
        for ax in axes:
            if getattr(ax, '_annotate_clipped_errors', False):
                ax._clipped_label_fontsize = _SI_CLIP_ANNOT_FONTSIZE
                _annotate_clipped_bar_errors(ax)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.20)
        fig.show()
        plt.show(block=False)


    def calculate_weighted_stats(group, column, sigma_col=None):
        """Area-weighted mean and pooled SE only; requires an explicit sigma column."""
        if column in _RATIO_OF_MEANS_SPECS:
            return calculate_weighted_stats_national(group, column)
        sigma_col = sigma_col or _sigma_col_for_value(column)
        if sigma_col is None or sigma_col not in group.columns:
            raise KeyError(f"Missing sigma column '{sigma_col}' for '{column}'")
        return calculate_weighted_stats_national(group, column, sigma_col)


    def _area_weighted_mean_se(group, value_col, sigma_col):
        """Area-weighted mean and TFV-block SE of the row-level errors."""
        return _mean_tfv(group, value_col, sigma_col)


    def calculate_weighted_stats_national(group, value_col, sigma_col=None):
        """Area-weighted mean and TFV-block SE of the row-level values.

        Relative trends are 100 × mean(slope) / mean(2020 level), formed on this group.
        """
        if value_col in _RATIO_OF_MEANS_SPECS:
            ratio, se = ratio_of_means_stats(group, *_RATIO_OF_MEANS_SPECS[value_col])
            return pd.Series({'mean': ratio, 'std': se})
        if sigma_col is None:
            sigma_col = _sigma_col_for_value(value_col)
        mean, se = _area_weighted_mean_se(group, value_col, sigma_col)
        return pd.Series({'mean': mean, 'std': se})


    def calculate_sum_stats(group, column, error_column=None, flux_kind=None):
        """Sum of lost biomass (TgC) and the TFV-block SE of the row flux errors."""
        sum_val = np.sum(group[column])
        if column == 'lost_agb':
            err_val = se_lost_agb_national(group)
        else:
            err_val = np.nan
        return pd.Series({'mean': sum_val, 'std': err_val})

    def _set_xticklabels_centered(ax, labels, fontsize=_AXIS_TICK_LABELSIZE, rotation=45, pad=_XTICK_LABEL_PAD):
        """X labels centered on each tick; extra pad pushes them below the axis."""
        if not labels:
            ax.set_xticklabels([])
            return
        ax.set_xticklabels(labels, fontsize=fontsize)
        plt.setp(
            ax.get_xticklabels(),
            rotation=rotation,
            ha='center',
            va='top',
            ma='center',
            rotation_mode='anchor',
        )
        ax.tick_params(axis='x', pad=pad, labelsize=fontsize)


    def _set_xticklabels_si(ax, labels, fontsize=_SI_XTICK_LABELSIZE, rotation=45, pad=_SI_XTICK_LABEL_PAD):
        """SI TFV x labels: 45° rotation, centred on each tick (Fig_age_distribution tick scale)."""
        if not labels:
            ax.set_xticklabels([])
            return
        ax.set_xticklabels(labels, fontsize=fontsize)
        plt.setp(
            ax.get_xticklabels(),
            rotation=rotation,
            ha='center',
            va='top',
            rotation_mode='anchor',
        )
        ax.tick_params(axis='x', pad=pad, labelsize=fontsize)


    def _force_plain_y_ticks(ax):
        """Disable y-axis shorthand/offset notation (e.g., k, 1eX)."""
        formatter = ax.yaxis.get_major_formatter()
        if hasattr(formatter, 'set_scientific'):
            formatter.set_scientific(False)
        if hasattr(formatter, 'set_useOffset'):
            formatter.set_useOffset(False)
        try:
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        except AttributeError:
            # Not a ScalarFormatter (e.g. FormatStrFormatter): nothing to do.
            pass
        ax.yaxis.offsetText.set_visible(False)


    def _force_plain_x_ticks(ax):
        """Disable x-axis shorthand/offset notation (e.g., k, 1eX)."""
        formatter = ax.xaxis.get_major_formatter()
        if hasattr(formatter, 'set_scientific'):
            formatter.set_scientific(False)
        if hasattr(formatter, 'set_useOffset'):
            formatter.set_useOffset(False)
        try:
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        except AttributeError:
            pass
        ax.xaxis.offsetText.set_visible(False)


    def _matplotlib_yerr(err_lo, err_hi, y_floor):
        """Shape (2, n) asymmetric or (n,) symmetric yerr for errorbar."""
        err_lo = np.atleast_1d(np.asarray(err_lo, dtype=float))
        err_hi = np.atleast_1d(np.asarray(err_hi, dtype=float))
        if y_floor is not None:
            return np.vstack([err_lo, err_hi])
        return err_hi


    def _plot_dot_whisker(ax, x, mean, se, facecolor, alpha=_DOT_MARKER_ALPHA, filled=True, markersize=_DOT_MARKERSIZE,
                          marker='o', y_floor=None, y_cap=None, capsize=3.5, elinewidth=1.2,
                          horizontal=False):
        """Area-weighted mean as a dot with error whiskers (vertical or horizontal)."""
        mean = np.atleast_1d(np.asarray(mean, dtype=float))
        err_lo, err_hi, _, _ = _yerr_for_nonneg_mean(mean, se, y_floor=y_floor, y_cap=y_cap)
        color = to_rgba(facecolor, alpha)
        yerr = _matplotlib_yerr(err_lo, err_hi, y_floor)
        cat = np.atleast_1d(np.asarray(x, dtype=float))
        if horizontal:
            # Values on x, categories on y; l/r clipping tracked as bottom/top for reuse.
            if filled:
                ax.errorbar(
                    mean, cat, xerr=yerr, fmt=marker, color=color, markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.6, ecolor=color, capsize=capsize, linewidth=elinewidth, markersize=markersize, zorder=4,
                )
            else:
                ax.errorbar(
                    mean, cat, xerr=yerr, fmt=marker, color=color, markerfacecolor='none', markeredgecolor=color,
                    markeredgewidth=1.1, ecolor=color, capsize=capsize, linewidth=elinewidth, markersize=markersize - 0.5, zorder=4,
                )
        elif filled:
            ax.errorbar(
                cat, mean, yerr=yerr, fmt=marker, color=color, markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=0.6, ecolor=color, capsize=capsize, linewidth=elinewidth, markersize=markersize, zorder=4,
            )
        else:
            ax.errorbar(
                cat, mean, yerr=yerr, fmt=marker, color=color, markerfacecolor='none', markeredgecolor=color,
                markeredgewidth=1.1, ecolor=color, capsize=capsize, linewidth=elinewidth, markersize=markersize - 0.5, zorder=4,
            )
        facecolors = np.broadcast_to(np.asarray(facecolor, dtype=object), mean.shape)
        alphas = np.broadcast_to(np.asarray(alpha, dtype=float), mean.shape)
        point_colors = [to_rgba(fc, a) for fc, a in zip(facecolors.flat, alphas.flat)]
        _append_clipped_error_specs(
            ax, cat, mean, err_lo, err_hi, colors=point_colors,
            marker=marker, markersize=markersize, horizontal=horizontal,
        )


    def _rotation_weighted_stats(group):
        """Area-weighted mean ± TFV-block SE for rotation period."""
        mean_work = _rotation_mean_subset(group)
        if mean_work.empty or mean_work['total_area'].sum() <= 0:
            return pd.Series({'mean': np.nan, 'std': 0.0})
        mean = float(np.average(mean_work[ROTATION_PERIOD_COLUMN], weights=mean_work['total_area']))
        se_work = _rotation_positive_subset(group)
        se = 0.0
        if not se_work.empty and se_work['total_area'].sum() > 0:
            se = stderr_area_weighted_mean_tfv_block(se_work, ROTATION_PERIOD_ERROR_COLUMN)
        if not np.isfinite(se):
            se = 0.0
        return pd.Series({'mean': mean, 'std': float(se)})


    def plot_bar_chart(ax, data, column, title, color, sorted_tfvs, xticks=None, xlabel=None, add_vline=False, show_ylabels=True):
        """Dot/whisker chart (area-weighted mean ± pooled SE)."""
        grouped = data.groupby('TFV', group_keys=False).apply(
            lambda x: calculate_weighted_stats(x, column), include_groups=False
        ).reset_index()
        if sorted_tfvs is not None:
            grouped = grouped.set_index('TFV').reindex(sorted_tfvs).reset_index()
        else:
            grouped = grouped.sort_values('mean', ascending=False)
        x_positions = np.arange(len(grouped))
        y_floor = _y_floor_for_column(column)
        _plot_dot_whisker(
            ax, x_positions, grouped['mean'], grouped['std'].fillna(0), color,
            alpha=_DOT_MARKER_ALPHA, y_floor=y_floor,
        )
        ax.set_ylabel(title, fontsize=16)
        ax.set_xticks(x_positions)
        if show_ylabels:
            _set_xticklabels_centered(ax, [dict_group_en[float(tfv)] for tfv in grouped['TFV']])
        else:
            ax.set_xticklabels([])
        if xticks:
            ax.set_yticks(xticks[0], labels=xticks[1])
            ax._y_tick_label_strings = list(xticks[1])
        if add_vline:
            ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.tick_params(labelsize=_AXIS_TICK_LABELSIZE)
        return grouped


    def _resolve_sigma_col(column, sigma_col=None):
        if sigma_col is not None:
            return sigma_col
        resolved = _sigma_col_for_value(column)
        return resolved if resolved else None


    def plot_combined_bar_chart(ax, natural_data, harvest_data, column, title,
                                sorted_tfvs=None, xticks=None, xlabel=None, add_vline=False, show_ylabels=True,
                                use_sum=False, error_column=None, flux_kind=None, show_yaxis=True,
                                use_national_stderr=True, sigma_col=None, si_style=False):
        """Harvest vs natural dot/whisker plots (area-weighted mean ± SE)."""
        sigma_col = _resolve_sigma_col(column, sigma_col)
        if use_sum:
            stat_fn = lambda x: calculate_sum_stats(x, column, error_column, flux_kind=flux_kind)
        elif use_national_stderr and sigma_col:
            stat_fn = lambda x: calculate_weighted_stats_national(x, column, sigma_col)
        else:
            stat_fn = lambda x: calculate_weighted_stats(x, column, sigma_col)
        natural_grouped = natural_data.groupby('TFV', group_keys=False).apply(
            stat_fn, include_groups=False
        ).reset_index()
        harvest_grouped = harvest_data.groupby('TFV', group_keys=False).apply(
            stat_fn, include_groups=False
        ).reset_index()

        # Merge on TFV to ensure same forest types
        combined = natural_grouped.merge(harvest_grouped, on='TFV', suffixes=('_nat', '_har'))

        # Sort using provided order if available
        if sorted_tfvs is not None:
            combined = combined.set_index('TFV').reindex(sorted_tfvs).reset_index()
        else:
            # Sort by natural values (decreasing)
            combined = combined.sort_values('mean_nat', ascending=False)

        x_positions = np.arange(len(combined))
        cat_colors = [_category_color(tfv, 'TFV') for tfv in combined['TFV']]
        std_har = combined['std_har'].fillna(0) if 'std_har' in combined.columns else 0.0
        std_nat = combined['std_nat'].fillna(0) if 'std_nat' in combined.columns else 0.0
        y_floor = _y_floor_for_column(column)
        y_cap = _y_cap_for_column(column)
        horizontal = bool(si_style)
        if horizontal:
            ax._si_horizontal = True
        dot_kw = dict(
            alpha=_DOT_MARKER_ALPHA, y_floor=y_floor, y_cap=y_cap,
            markersize=_SI_DOT_MARKERSIZE if si_style else _DOT_MARKERSIZE,
            capsize=_SI_ERROR_CAPSIZE if si_style else 3.5,
            elinewidth=_SI_ERROR_LW if si_style else 1.2,
            horizontal=horizontal,
        )
        for i, c in enumerate(cat_colors):
            _plot_dot_whisker(
                ax, x_positions[i] - _DOT_WHISKER_OFFSET, combined['mean_har'].iloc[i],
                std_har.iloc[i] if hasattr(std_har, 'iloc') else std_har[i], c,
                marker=_HARVEST_MARKER, **dot_kw,
            )
            _plot_dot_whisker(
                ax, x_positions[i] + _DOT_WHISKER_OFFSET, combined['mean_nat'].iloc[i],
                std_nat.iloc[i] if hasattr(std_nat, 'iloc') else std_nat[i], c,
                marker=_NATURAL_MARKER, **dot_kw,
            )
        if not si_style:
            ax.set_box_aspect(_DOT_BOX_ASPECT_TFV)
        label_fs = _SI_YLABEL_FONTSIZE if si_style else 16
        if horizontal:
            # Values on x; TFV category labels applied in finalize (shared y).
            ax.set_xlabel(_format_si_xlabel(title), fontsize=label_fs)
            ax.set_ylabel('')
            ax.set_yticks(x_positions)
            ax.set_ylim(*_si_tfv_category_ylim(len(combined)))
            ax.tick_params(axis='y', which='major', labelleft=False, length=0)
            if show_yaxis and xticks:
                _set_custom_xticks(ax, xticks[0], xticks[1])
                ax._y_tick_label_strings = list(xticks[1])
            if add_vline:
                ax.axvline(0, color='black', linestyle=':', linewidth=1)
            _style_si_tfv_axis(ax)
            ax.tick_params(axis='x', labelsize=_SI_TICK_LABELSIZE)
            return combined

        if show_yaxis:
            ax.set_ylabel(title, fontsize=label_fs)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        ax.set_xticks(x_positions)
        if show_ylabels:
            xlabels = [dict_group_en[float(tfv)] for tfv in combined['TFV']]
            _set_xticklabels_centered(ax, xlabels)
        else:
            ax.set_xticklabels([])

        if show_yaxis and xticks:
            _set_custom_yticks(ax, xticks[0], xticks[1])
            ax._y_tick_label_strings = list(xticks[1])
        if add_vline:
            ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.tick_params(labelsize=_AXIS_TICK_LABELSIZE)
        # Don't add legend here - will be added once above subfigure c
        return combined

    def plot_combined_bar_chart_by_group(ax, natural_data, harvest_data, column, title,
                                        sorted_groups=None, xticks=None, xlabel=None, add_vline=False, show_ylabels=True,
                                        use_sum=False, error_column=None, flux_kind=None, show_yaxis=True,
                                        use_national_stderr=True, sigma_col=None):
        """Harvest vs natural dot/whisker by TFV_group (area-weighted mean ± SE)."""
        nat = natural_data.dropna(subset=['TFV_group'])
        har = harvest_data.dropna(subset=['TFV_group'])
        if nat.empty or har.empty:
            ax.set_axis_off()
            return None

        sigma_col = _resolve_sigma_col(column, sigma_col)
        if use_sum:
            stat_fn = lambda x: calculate_sum_stats(x, column, error_column, flux_kind=flux_kind)
        elif use_national_stderr and sigma_col:
            stat_fn = lambda x: calculate_weighted_stats_national(x, column, sigma_col)
        else:
            stat_fn = lambda x: calculate_weighted_stats(x, column, sigma_col)
        natural_grouped = nat.groupby('TFV_group', group_keys=False).apply(
            stat_fn, include_groups=False
        ).reset_index()
        harvest_grouped = har.groupby('TFV_group', group_keys=False).apply(
            stat_fn, include_groups=False
        ).reset_index()
        combined = natural_grouped.merge(harvest_grouped, on='TFV_group', how='outer', suffixes=('_nat', '_har'))
        combined = combined.fillna(0)

        if sorted_groups is not None:
            # Show all groups in order; add missing ones with 0
            combined = combined.set_index('TFV_group').reindex(sorted_groups).reset_index()
            combined = combined.fillna(0)
        else:
            combined = combined.sort_values('mean_nat', ascending=False)

        x_positions = np.arange(len(combined))
        cat_colors = [_category_color(g, 'TFV_group') for g in combined['TFV_group']]
        std_har = combined['std_har'].fillna(0) if 'std_har' in combined.columns else 0.0
        std_nat = combined['std_nat'].fillna(0) if 'std_nat' in combined.columns else 0.0
        y_floor = _y_floor_for_column(column)
        y_cap = _y_cap_for_column(column)
        for i, c in enumerate(cat_colors):
            _plot_dot_whisker(
                ax, x_positions[i] - _DOT_WHISKER_OFFSET, combined['mean_har'].iloc[i], std_har.iloc[i], c,
                alpha=_DOT_MARKER_ALPHA, marker=_HARVEST_MARKER, y_floor=y_floor, y_cap=y_cap,
            )
            _plot_dot_whisker(
                ax, x_positions[i] + _DOT_WHISKER_OFFSET, combined['mean_nat'].iloc[i], std_nat.iloc[i], c,
                alpha=_DOT_MARKER_ALPHA, marker=_NATURAL_MARKER, y_floor=y_floor, y_cap=y_cap,
            )
        if show_yaxis:
            ax.set_ylabel(_format_panel_title(title), fontsize=16)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        ax.set_xticks(x_positions)
        if show_ylabels:
            _set_xticklabels_centered(ax, combined['TFV_group'].tolist())
        else:
            ax.set_xticklabels([])
        if show_yaxis and xticks:
            _set_custom_yticks(ax, xticks[0], xticks[1])
            ax._y_tick_label_strings = list(xticks[1])
        if add_vline:
            ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.tick_params(labelsize=_AXIS_TICK_LABELSIZE)
        return combined

    def plot_rotation_period_boxplot(ax, data, group_col, sorted_items, title, label_fn=None,
                                     box_aspect=None, show_ylabel=True, si_style=False):
        """Rotation period: area-weighted mean with error whiskers (± SE, not 95% CI)."""
        required_cols = [group_col, ROTATION_PERIOD_COLUMN, 'total_area']
        if TFV_COL in data.columns and TFV_COL not in required_cols:
            required_cols.append(TFV_COL)
        if ROTATION_PERIOD_ERROR_COLUMN in data.columns:
            required_cols = required_cols + [ROTATION_PERIOD_ERROR_COLUMN]
        if any(col not in data.columns for col in required_cols):
            ax.set_axis_off()
            return None

        work = data[required_cols].copy()
        work = work[
            work[group_col].notna()
            & np.isfinite(work[ROTATION_PERIOD_COLUMN])
            & (work[ROTATION_PERIOD_COLUMN] > 0)
        ]
        if work.empty:
            ax.set_axis_off()
            return None

        if sorted_items is None:
            order = work.groupby(group_col)[ROTATION_PERIOD_COLUMN].median().sort_values(ascending=False).index.tolist()
        else:
            order = [item for item in sorted_items if item in set(work[group_col])]
        if not order:
            ax.set_axis_off()
            return None

        positions = np.arange(len(order))
        if label_fn is None:
            labels = [str(item) for item in order]
        else:
            labels = [label_fn(item) for item in order]
        means, stds = [], []
        cat_colors = [_category_color(item, group_col) for item in order]
        horizontal = bool(si_style)
        if horizontal:
            ax._si_horizontal = True
        dot_kw = dict(
            alpha=_DOT_MARKER_ALPHA,
            markersize=_SI_DOT_MARKERSIZE if si_style else _DOT_MARKERSIZE,
            capsize=_SI_ERROR_CAPSIZE if si_style else 3.5,
            elinewidth=_SI_ERROR_LW if si_style else 1.2,
            horizontal=horizontal,
        )
        for pos, item, color in zip(positions, order, cat_colors):
            st = _rotation_weighted_stats(work.loc[work[group_col] == item])
            means.append(st['mean'])
            stds.append(st['std'])  # SE (figures); manuscript text uses 1.96×SE
            _plot_dot_whisker(ax, pos, st['mean'], st['std'], color, **dot_kw)

        if not si_style:
            if box_aspect is None and group_col != 'TFV_group':
                box_aspect = _DOT_BOX_ASPECT_TFV
            if box_aspect is not None:
                ax.set_box_aspect(box_aspect)
        label_fs = _SI_YLABEL_FONTSIZE if si_style else 16
        rotation_ylabel = 'Rotation period (years)'
        if horizontal:
            ax.set_xlabel(_format_si_xlabel(rotation_ylabel), fontsize=label_fs)
            ax.set_ylabel('')
            ax.set_yticks(positions)
            ax.set_ylim(*_si_tfv_category_ylim(len(order)))
            ax.tick_params(axis='y', which='major', labelleft=False, length=0)
            _style_si_tfv_axis(ax)
            ax.tick_params(axis='x', labelsize=_SI_TICK_LABELSIZE)
            ax.set_title(title, fontsize=16)
            return pd.DataFrame({group_col: order, 'mean': means, 'std': stds})

        if show_ylabel:
            ax.set_ylabel(rotation_ylabel, fontsize=label_fs)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=True)
        ax.set_xticks(positions)
        _set_xticklabels_centered(ax, labels)
        ax.tick_params(labelsize=_AXIS_TICK_LABELSIZE)
        ax.set_title(title, fontsize=16)
        return pd.DataFrame({group_col: order, 'mean': means, 'std': stds})

    _COMBINED_FIGURE_DIR = str(figure_dir)
    _BAR_LEGEND_ELEMENTS = [
        Line2D([0], [0], marker=_HARVEST_MARKER, linestyle='None', markerfacecolor='gray', markeredgecolor='black',
               markersize=_LEGEND_MARKERSIZE, alpha=_DOT_MARKER_ALPHA, label='Harvest'),
        Line2D([0], [0], marker=_NATURAL_MARKER, linestyle='None', markerfacecolor='gray', markeredgecolor='black',
               markersize=_LEGEND_MARKERSIZE, alpha=_DOT_MARKER_ALPHA, label='Natural'),
    ]


    def _make_disturbance_combined_figure(n_rows, spacer_width=0.28):
        """4-column grid: harvest map | spacer | TFV-group dots | natural map.

        The empty spacer column leaves room for colorbars + dot y-labels; the
        natural map sits closer on the right (no spacer there).
        ``spacer_width`` is the relative width of the gap between maps and dots.
        """
        fig, axes = plt.subplots(
            n_rows, 4, figsize=(17, 5.0 * n_rows),
            gridspec_kw={
                'width_ratios': [1.0, spacer_width, 1.25, 1.0],
                'wspace': 0.12,
                'hspace': 0.28,
            },
        )
        if n_rows == 1:
            axes = np.asarray([axes])
        for ax_sp in axes[:, 1]:
            ax_sp.set_axis_off()
        return fig, axes


    def _unravel_combined_axes(axes):
        """Per row: (harvest map, natural map, TFV-group dots)."""
        return [(axes[i, 0], axes[i, 3], axes[i, 2]) for i in range(axes.shape[0])]


    def _make_si_tfv_vertical_figure(n_rows):
        """1 × n horizontal SI panels (TFV on shared y, values on x)."""
        n_panels = int(n_rows)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(_SI_PANEL_WIDTH * n_panels + 2.2, _SI_FIG_HEIGHT),
            sharey=True,
            gridspec_kw={'wspace': 0.12},
        )
        return fig, axes


    def _overlay_map_borders(ax):
        """Black France outer edge only."""
        france_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=11)


    def _plot_ser_map(gdf, column, cmap, ax, title, vmin=None, vmax=None, **kwargs):
        """Choropleth + France outline; turn axis off if the column is missing."""
        if column not in gdf.columns:
            ax.set_axis_off()
            return
        plot_map(gdf, column, cmap, ax, title, vmin=vmin, vmax=vmax, **kwargs)
        if _is_populated_map_ax(ax):
            _overlay_map_borders(ax)


    def _apply_dot_ylim(ax, mode, **kwargs):
        if mode == 'adaptive':
            _lock_adaptive_dot_ylim(ax, **kwargs)
        elif mode == 'restrictive':
            _lock_restrictive_dot_ylim(ax, **kwargs)
        elif mode == 'error_specs':
            _apply_ylim_from_error_specs(ax, **kwargs)
        elif mode == 'error_specs_both':
            _apply_ylim_from_error_specs_both(ax, **kwargs)
        elif mode is not None:
            raise ValueError(f'Unknown ylim mode: {mode!r}')


    def _patch_size_split_vlims(merged_h, merged_n, column='mean_patch_size_2020'):
        """Harvest/natural patch-size color scales meet at a shared split (cap natural at 3 ha)."""
        vmin_h, vmax_h = _shared_map_vlim((merged_h, column))
        vmin_n, vmax_n = _shared_map_vlim((merged_n, column))
        if None in (vmin_h, vmax_h, vmin_n, vmax_n):
            vmin_h, vmax_n = _PATCH_PIXEL_AREA_HA, 3.0
            split = 0.5 * (vmin_h + vmax_n)
        else:
            vmax_n = min(vmax_n, 3.0)
            split = 0.5 * (vmax_h + vmin_n)
            if not (vmin_h < split < vmax_n):
                split = 0.5 * (vmin_h + vmax_n)
        return vmin_h, split, split, vmax_n


    def _plot_combined_metric_row(
        ax_h, ax_n, ax_dots, *,
        column, cmap, title_h, title_n, ylabel,
        natural_df, harvest_df, groups,
        merged_h, merged_n,
        vmin=None, vmax=None, vmin_h=None, vmax_h=None, vmin_n=None, vmax_n=None,
        cbar_center_tick=True, sigma_col=None, use_national_stderr=True,
        add_vline=False, show_ylabels=False, xticks=None,
        ylim_mode='adaptive', ylim_kw=None, custom_yticks=None,
        require_column=False, top_legend=False,
    ):
        """One combined-figure row: harvest map | TFV-group dots | natural map."""
        if require_column and column not in merged_h.columns and column not in merged_n.columns:
            ax_h.set_axis_off()
            ax_n.set_axis_off()
            ax_dots.set_axis_off()
            return

        split_scales = any(v is not None for v in (vmin_h, vmax_h, vmin_n, vmax_n))
        if split_scales:
            _plot_ser_map(merged_h, column, cmap, ax_h, title_h, vmin=vmin_h, vmax=vmax_h,
                          cbar_center_tick=cbar_center_tick)
            _plot_ser_map(merged_n, column, cmap, ax_n, title_n, vmin=vmin_n, vmax=vmax_n,
                          cbar_center_tick=cbar_center_tick)
        else:
            if vmin is None and vmax is None:
                vmin, vmax = _shared_map_vlim((merged_h, column), (merged_n, column))
            _plot_ser_map(merged_h, column, cmap, ax_h, title_h, vmin=vmin, vmax=vmax,
                          cbar_center_tick=cbar_center_tick)
            _plot_ser_map(merged_n, column, cmap, ax_n, title_n, vmin=vmin, vmax=vmax,
                          cbar_center_tick=cbar_center_tick)

        if require_column and (column not in natural_df.columns or column not in harvest_df.columns):
            ax_dots.set_axis_off()
            return

        plot_combined_bar_chart_by_group(
            ax_dots, natural_df, harvest_df, column, ylabel, groups,
            xticks=xticks, show_ylabels=show_ylabels, add_vline=add_vline,
            use_national_stderr=use_national_stderr, sigma_col=sigma_col,
        )
        _apply_dot_ylim(ax_dots, ylim_mode, **(ylim_kw or {}))
        if custom_yticks is not None:
            _set_custom_yticks(ax_dots, custom_yticks[0], custom_yticks[1])
        if top_legend:
            ax_dots.legend(
                handles=_BAR_LEGEND_ELEMENTS, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=2, fontsize=_LEGEND_FONTSIZE, frameon=False,
            )


    def _plot_si_tfv_panel(
        ax, natural_df, harvest_df, column, ylabel, sorted_tfvs, *,
        sigma_col=None, use_national_stderr=True, add_vline=False,
        ylim_mode='restrictive', ylim_kw=None, custom_yticks=None, require_column=False,
    ):
        """One SI per-TFV panel (harvest vs natural dots; TFV on y, values on x)."""
        if require_column and (column not in natural_df.columns or column not in harvest_df.columns):
            ax.set_axis_off()
            return
        plot_combined_bar_chart(
            ax, natural_df, harvest_df, column, ylabel,
            sorted_tfvs=sorted_tfvs, show_ylabels=False, add_vline=add_vline,
            use_national_stderr=use_national_stderr, sigma_col=sigma_col, si_style=True,
        )
        _apply_dot_ylim(ax, ylim_mode, **(ylim_kw or {}))
        if custom_yticks is not None:
            _set_custom_xticks(ax, custom_yticks[0], custom_yticks[1])
            ax._y_tick_label_strings = list(custom_yticks[1])


    def _yerr_for_nonneg_mean(mean, se, y_floor=None, y_cap=None):
        """Symmetric or clipped error bars with optional lower/upper physical bounds."""
        se = np.maximum(np.asarray(se, dtype=float), 0.0)
        mean = np.asarray(mean, dtype=float)
        err_lo = se.copy()
        err_hi = se.copy()
        if y_floor is not None:
            err_lo = np.minimum(err_lo, np.maximum(mean - y_floor, 0.0))
        if y_cap is not None:
            err_hi = np.minimum(err_hi, np.maximum(y_cap - mean, 0.0))
        return err_lo, err_hi, mean - err_lo, mean + err_hi


    def _append_clipped_error_specs(ax, x_positions, means, err_lo, err_hi, colors=None,
                                    marker='o', markersize=_DOT_MARKERSIZE, horizontal=False):
        """Track error-bar extremes for clipped-whisker annotations after value-axis lim is set."""
        if not hasattr(ax, '_clipped_error_specs'):
            ax._clipped_error_specs = []
        x_positions = np.atleast_1d(np.asarray(x_positions, dtype=float))
        means = np.atleast_1d(np.asarray(means, dtype=float))
        err_lo = np.atleast_1d(np.asarray(err_lo, dtype=float))
        err_hi = np.atleast_1d(np.asarray(err_hi, dtype=float))
        if colors is None:
            colors = [None] * len(x_positions)
        for x, mean, lo, hi, color in zip(x_positions, means, err_lo, err_hi, colors):
            mean = 0.0 if not np.isfinite(mean) else float(mean)
            lo = 0.0 if not np.isfinite(lo) else float(lo)
            hi = 0.0 if not np.isfinite(hi) else float(hi)
            spec = {
                'x': float(x), 'mean': mean, 'top': mean + hi, 'bottom': mean - lo,
                'marker': marker, 'markersize': markersize,
                'horizontal': bool(horizontal),
            }
            if color is not None:
                spec['color'] = color
            ax._clipped_error_specs.append(spec)


    def _max_decimals_in_tick_label_strings(label_strings):
        """Decimal places used in explicit y-tick label strings."""
        max_d = 0
        for raw in label_strings:
            s = str(raw).strip().replace(',', '.').replace('−', '-')
            if not s or s in ('-', '+'):
                continue
            if '.' in s:
                max_d = max(max_d, len(s.split('.', 1)[1]))
        return max_d


    def _decimals_from_major_formatter(ax):
        fmt = ax.yaxis.get_major_formatter()
        if isinstance(fmt, FormatStrFormatter):
            fmtstr = getattr(fmt, 'fmt', '') or ''
            match = re.search(r'%\.(\d+)[fge]', fmtstr)
            if match:
                return int(match.group(1))
        return None


    def _decimals_for_clipped_labels(axes):
        """Match clipped edge labels to y-axis tick precision (shared across a row)."""
        max_d = None
        for ax in axes:
            stored = getattr(ax, '_y_tick_label_strings', None)
            if stored:
                d = _max_decimals_in_tick_label_strings(stored)
                max_d = d if max_d is None else max(max_d, d)
            labels = [t.get_text().strip() for t in ax.get_yticklabels() if t.get_text().strip()]
            if labels:
                d = _max_decimals_in_tick_label_strings(labels)
                max_d = d if max_d is None else max(max_d, d)
            d = _decimals_from_major_formatter(ax)
            if d is not None:
                max_d = d if max_d is None else max(max_d, d)
        return max_d


    def _format_clipped_axis_value(ax, value):
        value = float(value)
        decimals = getattr(ax, '_clipped_label_decimals', None)
        if decimals is not None:
            return f'{value:.{decimals}f}'
        formatter = ax.yaxis.get_major_formatter()
        try:
            return formatter.format_data(value)
        except (AttributeError, TypeError, ValueError):
            pass
        return f'{value:.3g}'


    def _apply_clipped_label_decimals(axes):
        decimals = _decimals_for_clipped_labels(axes)
        if decimals is not None:
            for ax in axes:
                ax._clipped_label_decimals = decimals


    def _patch_vertical_extent(patch):
        """Vertical span of a bar Rectangle or boxplot PathPatch."""
        if hasattr(patch, 'get_height'):
            y0 = float(patch.get_y())
            h = float(patch.get_height())
            return min(y0, y0 + h), max(y0, y0 + h)
        verts = patch.get_path().vertices
        if verts is not None and len(verts):
            y = verts[:, 1]
            return float(np.min(y)), float(np.max(y))
        return None


    def _axis_bar_patch_limits(ax):
        """Y range from bar patches only (exclude error-bar whiskers)."""
        lows, highs = [], []
        for patch in ax.patches:
            ext = _patch_vertical_extent(patch)
            if ext is None:
                continue
            lows.append(ext[0])
            highs.append(ext[1])
        if not highs:
            return None
        return min(lows), max(highs)


    def _line_is_dot_marker(line):
        """True for errorbar mean markers (circle or square), not zero reference lines."""
        marker = line.get_marker()
        if marker in (None, 'None', 'none', ''):
            return False
        ls = line.get_linestyle()
        return ls in (None, '', ' ', 'None', 'none')


    def _axis_dot_whisker_limits(ax):
        """Y range from plotted dots ± whiskers (for shared row ylim)."""
        specs = getattr(ax, '_clipped_error_specs', None) or []
        if specs:
            lows = [float(s['bottom']) for s in specs]
            highs = [float(s['top']) for s in specs]
            return min(lows), max(highs)
        lows, highs = [], []
        for line in ax.lines:
            if not _line_is_dot_marker(line):
                continue
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                lows.append(float(np.min(y)))
                highs.append(float(np.max(y)))
        if not highs:
            limits = _axis_bar_patch_limits(ax)
            if limits is not None:
                return limits
            return None
        return min(lows), max(highs)


    def _axis_dot_mean_limits(ax):
        """Y range from dot means only (ignore whisker extent)."""
        specs = getattr(ax, '_clipped_error_specs', None) or []
        means = [float(s['mean']) for s in specs if np.isfinite(s.get('mean', np.nan))]
        if means:
            return min(means), max(means)
        vals = []
        for line in ax.lines:
            if not _line_is_dot_marker(line):
                continue
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                vals.extend(y.tolist())
        if not vals:
            limits = _axis_bar_patch_limits(ax)
            if limits is not None:
                return limits
            return None
        return min(vals), max(vals)


    def _axis_box_body_limits(ax):
        """Y range from box bodies and mean points (exclude whiskers)."""
        specs = getattr(ax, '_box_whisker_specs', None)
        if specs:
            lows = [min(s['q1'], s['med'], s['q3']) for s in specs]
            highs = [max(s['q1'], s['med'], s['q3']) for s in specs]
            return min(lows), max(highs)
        lows, highs = [], []
        for patch in ax.patches:
            ext = _patch_vertical_extent(patch)
            if ext is None:
                continue
            lows.append(ext[0])
            highs.append(ext[1])
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                lows.append(float(np.min(y)))
                highs.append(float(np.max(y)))
        if not highs:
            return None
        return min(lows), max(highs)


    def _clipped_label_fontsize(ax):
        stored = getattr(ax, '_clipped_label_fontsize', None)
        if stored is not None:
            return stored
        tick_labels = ax.get_yticklabels()
        if tick_labels:
            return tick_labels[0].get_fontsize()
        return _CLIP_ANNOT_FONTSIZE


    def _annotate_clipped_bar_errors(ax, y_top=None, y_bottom=None):
        """Clipped-whisker labels for vertical or horizontal (SI) value axes."""
        specs = getattr(ax, '_clipped_error_specs', None) or []
        if not specs:
            return
        horizontal = bool(getattr(ax, '_si_horizontal', False) or specs[0].get('horizontal', False))
        if y_top is None or y_bottom is None:
            if horizontal:
                y_bottom, y_top = ax.get_xlim()
            else:
                y_bottom, y_top = ax.get_ylim()

        for txt in list(getattr(ax, '_clipped_error_texts', [])):
            try:
                txt.remove()
            except ValueError:
                pass
        ax._clipped_error_texts = []

        # Optional per-point category nudge (data coords).
        nudges = getattr(ax, '_clipped_label_x_nudges', None) or {}
        bottom_ys = getattr(ax, '_clipped_label_bottom_ys', None) or {}
        default_bottom_y = float(getattr(ax, '_clipped_label_bottom_y', 0.04))
        fontsize = _clipped_label_fontsize(ax)

        if horizontal:
            # y = category (data), x = axes fraction for edge labels.
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            for s in specs:
                color = s.get('color', 'black')
                y0 = float(s['x'])
                dy, va = 0.05, 'bottom'
                if y0 < np.round(y0):
                    dy, va = -0.05, 'top'
                for key, nudge in nudges.items():
                    if abs(y0 - float(key)) < 1e-9:
                        dy = float(nudge)
                        va = 'top' if dy < 0 else 'bottom'
                        break
                y = y0 + dy
                if s['top'] > y_top + 1e-12:
                    t = ax.text(
                        0.96, y, f"{s['top']:.0f}", transform=trans,
                        ha='right', va=va, fontsize=fontsize,
                        color=color, clip_on=True, zorder=30,
                    )
                    ax._clipped_error_texts.append(t)
                if s['bottom'] < y_bottom - 1e-12:
                    x_bot = default_bottom_y
                    for key, yv in bottom_ys.items():
                        if abs(y0 - float(key)) < 1e-9:
                            x_bot = float(yv)
                            break
                    ha_bot = 'left' if x_bot >= 0 else 'right'
                    t = ax.text(
                        x_bot, y, f"{s['bottom']:.0f}", transform=trans,
                        ha=ha_bot, va=va, fontsize=fontsize,
                        color=color, clip_on=False, zorder=30,
                    )
                    ax._clipped_error_texts.append(t)
                if s['mean'] > y_top + 1e-12 or s['mean'] < y_bottom - 1e-12:
                    inset = 0.04 * (y_top - y_bottom)
                    x_mark = (y_top - inset) if s['mean'] > y_top else (y_bottom + inset)
                    ax.plot(
                        x_mark, s['x'], linestyle='None',
                        marker=s.get('marker', 'o'),
                        markersize=s.get('markersize', _DOT_MARKERSIZE),
                        color=color, markerfacecolor=color, markeredgecolor=color,
                        markeredgewidth=0.6, zorder=32, clip_on=True,
                    )
            return

        # x in data coords, y in axes fraction → all top/bottom labels share one y.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for s in specs:
            color = s.get('color', 'black')
            x0 = float(s['x'])
            dx, ha = 0.05, 'left'
            # Harvest sits left of the category tick → label on the left, right-aligned.
            if x0 < np.round(x0):
                dx, ha = -0.05, 'right'
            for key, nudge in nudges.items():
                if abs(x0 - float(key)) < 1e-9:
                    dx = float(nudge)
                    ha = 'right' if dx < 0 else 'left'
                    break
            x = x0 + dx
            if s['top'] > y_top + 1e-12:
                t = ax.text(
                    x, 0.96, f"{s['top']:.0f}", transform=trans,
                    ha=ha, va='top', fontsize=fontsize,
                    color=color, clip_on=True, zorder=30,
                )
                ax._clipped_error_texts.append(t)
            if s['bottom'] < y_bottom - 1e-12:
                y_bot = default_bottom_y
                for key, yv in bottom_ys.items():
                    if abs(x0 - float(key)) < 1e-9:
                        y_bot = float(yv)
                        break
                # Below the axis spine → hang under the whisker tip with room to spare.
                va_bot = 'bottom' if y_bot >= 0 else 'top'
                t = ax.text(
                    x, y_bot, f"{s['bottom']:.0f}", transform=trans,
                    ha=ha, va=va_bot, fontsize=fontsize,
                    color=color, clip_on=False, zorder=30,
                )
                ax._clipped_error_texts.append(t)
            # Means outside ylim are clipped by errorbar; redraw at the axis edge.
            if s['mean'] > y_top + 1e-12 or s['mean'] < y_bottom - 1e-12:
                inset = 0.04 * (y_top - y_bottom)
                y_mark = (y_top - inset) if s['mean'] > y_top else (y_bottom + inset)
                ax.plot(
                    s['x'], y_mark, linestyle='None',
                    marker=s.get('marker', 'o'),
                    markersize=s.get('markersize', _DOT_MARKERSIZE),
                    color=color, markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.6, zorder=32, clip_on=True,
                )


    def _annotate_clipped_box_whiskers(ax, y_top, y_bottom):
        specs = getattr(ax, '_box_whisker_specs', None) or []
        for spec in specs:
            x = spec['x']
            if spec['whishi'] > y_top + 1e-12:
                ax.text(
                    x, y_top, f"↑ {_format_clipped_axis_value(ax, spec['whishi'])}",
                    fontsize=8, ha='center', va='bottom', clip_on=False,
                )
            if spec['whislo'] < y_bottom - 1e-12:
                ax.text(
                    x, y_bottom, f"↓ {_format_clipped_axis_value(ax, spec['whislo'])}",
                    fontsize=8, ha='center', va='top', clip_on=False,
                )


    def _nice_tick_ceiling(value):
        """Readable axis top at any scale (rounds up to a nice 1-2-5 style step × power of ten)."""
        if not np.isfinite(value) or value <= 0:
            return 1.0
        exp = 10 ** np.floor(np.log10(value))
        norm = value / exp  # in [1, 10)
        for nice in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
            if norm <= nice * (1 + 1e-9):
                return nice * exp
        return 10 * exp


    def _assign_independent_yticks(ax, n_ticks=5, padding_frac=0.05, label_fmt='{:.1f}'):
        """0-based ylim and explicit y ticks fit to the dot+whisker extent of one panel.

        ``label_fmt`` controls the tick label formatting (use ``'{:g}'`` for small scales).
        """
        limits = _axis_dot_whisker_limits(ax) or _axis_dot_mean_limits(ax)
        if limits is None:
            return
        y_min = 0.0 if _axis_includes_zero_reference(ax) else limits[0]
        ymax = limits[1] * (1.0 + padding_frac)
        top = _nice_tick_ceiling(ymax)
        ticks = np.linspace(y_min, top, n_ticks)
        if y_min <= 0.0 <= top and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.unique(np.append([0.0], ticks[1:])))
        labels = [label_fmt.format(t) for t in ticks]
        ax._ylim_fixed = (float(y_min), float(top))
        ax._independent_ylim = True
        ax._yticks_from_ylim = False
        _set_custom_yticks(ax, ticks, labels)
        ax._y_tick_label_strings = labels
        _force_plain_y_ticks(ax)


    def _yticks_from_ylim(y_min, y_max, n_ticks=5, include_zero=False):
        """Tick positions from synced ylim (error-bar driven), optionally forcing 0."""
        y_min, y_max = float(y_min), float(y_max)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            return [0.0], ['0']
        if include_zero:
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
        ticks = np.linspace(y_min, y_max, n_ticks)
        if include_zero and y_min <= 0.0 <= y_max and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.unique(np.append(ticks, 0.0)))
        labels = [f'{t:.3g}' for t in ticks]
        return list(ticks), labels


    def _set_custom_yticks(ax, tick_positions, tick_labels):
        ticks = list(tick_positions)
        labels = [str(l) for l in tick_labels]
        ax._custom_yticks = (ticks, labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.tick_params(axis='y', labelsize=_AXIS_TICK_LABELSIZE)


    def _set_custom_xticks(ax, tick_positions, tick_labels):
        ticks = list(tick_positions)
        labels = [str(l) for l in tick_labels]
        ax._custom_xticks = (ticks, labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', labelsize=_SI_TICK_LABELSIZE)


    def _restore_custom_yticks(ax):
        stored = getattr(ax, '_custom_yticks', None)
        if stored is None:
            return
        ticks, labels = stored
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.tick_params(axis='y', labelsize=_AXIS_TICK_LABELSIZE)


    def _apply_ytick_visibility(ax):
        if getattr(ax, '_hide_yticks', False):
            ax.set_yticks([])
            ax.tick_params(axis='y', left=False, labelleft=False)


    def _apply_shared_ylim(active, y_min, y_max, padding=0.05):
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        ylim_fixed = next((getattr(ax, '_ylim_fixed', None) for ax in active if getattr(ax, '_ylim_fixed', None)), None)
        if ylim_fixed is not None:
            y_min, y_max = float(ylim_fixed[0]), float(ylim_fixed[1])
            padding = 0.0
        ymax_cap = None
        ymin_cap = None
        for ax in active:
            cap = getattr(ax, '_ylim_ymax', None)
            if cap is not None:
                ymax_cap = cap if ymax_cap is None else min(ymax_cap, cap)
            cap_lo = getattr(ax, '_ylim_ymin', None)
            if cap_lo is not None:
                ymin_cap = cap_lo if ymin_cap is None else max(ymin_cap, cap_lo)
        if ymax_cap is not None:
            y_max = min(y_max, ymax_cap)
        if ymin_cap is not None:
            y_min = max(y_min, ymin_cap)
        if ylim_fixed is None:
            span = y_max - y_min
            pad = span * padding if span > 0 else max(abs(y_max), abs(y_min), 1.0) * padding
            y_min, y_max = y_min - pad, y_max + pad
            if ymax_cap is not None:
                y_max = min(y_max, ymax_cap)
            if ymin_cap is not None:
                y_min = max(y_min, ymin_cap)
        for ax in active:
            ax.set_ylim(y_min, y_max)
            _apply_ytick_visibility(ax)


    def _sync_bar_axes_ylim(axes, padding=0.05, include_zero=False):
        """Shared ylim fit to the full dot+whisker extent (no clipped-value labels)."""
        active = [
            ax for ax in axes
            if ax.axison and ax.get_visible() and _axis_dot_whisker_limits(ax) is not None
        ]
        if not active:
            return
        y_min, y_max = np.inf, -np.inf
        for ax in active:
            limits = _axis_dot_whisker_limits(ax)
            if limits is None:
                continue
            y_min = min(y_min, limits[0])
            y_max = max(y_max, limits[1])
        if include_zero:
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
        _apply_shared_ylim(active, y_min, y_max, padding=padding)
        if len(active) > 1:
            shared_ylim = active[0].get_ylim()
            for ax in active[1:]:
                ax.set_ylim(shared_ylim)
        for ax in active:
            y_bottom, y_top = ax.get_ylim()
            if getattr(ax, '_yticks_from_ylim', False):
                ticks, labels = _yticks_from_ylim(
                    y_bottom, y_top, n_ticks=5, include_zero=getattr(ax, '_include_zero_in_ylim', False),
                )
                _set_custom_yticks(ax, ticks, labels)
            else:
                _restore_custom_yticks(ax)
            _apply_ytick_visibility(ax)


    def _sync_box_axes_ylim(axes, padding=0.05):
        """Shared ylim from box bodies; whiskers may clip with ↑ labels."""
        active = [ax for ax in axes if ax.axison and ax.get_visible() and getattr(ax, '_box_whisker_specs', None)]
        if not active:
            return
        y_min, y_max = np.inf, -np.inf
        for ax in active:
            limits = _axis_box_body_limits(ax)
            if limits is None:
                continue
            y_min = min(y_min, limits[0])
            y_max = max(y_max, limits[1])
        _apply_shared_ylim(active, y_min, y_max, padding=padding)
        _apply_clipped_label_decimals(active)
        for ax in active:
            y_bottom, y_top = ax.get_ylim()
            _annotate_clipped_box_whiskers(ax, y_top, y_bottom)


    def _axis_includes_zero_reference(ax):
        for line in ax.lines:
            if line.get_linestyle() == ':' and len(line.get_ydata()) == 2:
                if np.allclose(line.get_ydata(), 0.0):
                    return True
        limits = _axis_dot_whisker_limits(ax)
        if limits is not None and limits[0] <= 0.0 <= limits[1]:
            return True
        return False


    def _sync_row_chart_ylim(ax_group, padding=0.05):
        """Y limits for the TFV-group dot panel in one row."""
        if getattr(ax_group, '_ylim_fixed', None) is not None:
            # Keep content-fitted limits; do not re-expand to include 0.
            y_min, y_max = ax_group._ylim_fixed
            ax_group.set_ylim(y_min, y_max)
            return
        include_zero = _axis_includes_zero_reference(ax_group)
        if getattr(ax_group, '_independent_ylim', False):
            _sync_bar_axes_ylim([ax_group], padding=padding, include_zero=include_zero)
            return
        _sync_bar_axes_ylim([ax_group], padding=padding, include_zero=include_zero)


    def _subplot_labels_sequential(n_panels):
        """Panel labels a, b, c, … left-to-right, top-to-bottom across rows."""
        return [chr(ord('a') + i) for i in range(n_panels)]


    def _place_subplot_label(ax, label, fontsize=14, style='auto'):
        """Bold panel letter clear of titles, data, and axis frames.

        * Maps (axis off): upper-left *inside* the axes so the letter sits in the
          map margin and does not collide with the centred title above.
        * Charts: just outside the top-left corner of the frame.
        * SI panels (``style='si'``): above the top-left frame (horizontal TFV layout).
        """
        if style == 'si':
            # Horizontal SI panels: letter just above the top-left frame corner.
            x, y, ha, va = 0.0, 1.02, 'left', 'bottom'
        elif style == 'map' or (style == 'auto' and not ax.axison):
            x, y, ha, va = 0.02, 0.97, 'left', 'top'
        else:
            x, y, ha, va = -0.02, 1.06, 'right', 'bottom'
        ax.text(
            x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
            va=va, ha=ha, clip_on=False, zorder=20,
        )


    def _apply_combined_group_xticklabels(row_axes, group_labels):
        """Category labels only on the last row so they do not collide with the row below."""
        wrapped = [str(lab).replace(' ', '\n') for lab in group_labels]
        last = len(row_axes) - 1
        for i, (_, _, ax_g) in enumerate(row_axes):
            if not ax_g.axison:
                continue
            if i < last:
                ax_g.set_xticklabels([])
                ax_g.tick_params(axis='x', labelbottom=False, pad=2)
            else:
                ax_g.tick_params(axis='x', labelbottom=True)
                _set_xticklabels_centered(ax_g, wrapped, rotation=0, pad=6)


    def _finalize_disturbance_combined_figure(
        fig, row_axes, row_title_texts, save_path,
        col0_header='Harvest', col1_header='Natural disturbances',
        row_mid_map_titles=None, map_header_row_indices=(0, 1),
        show_top_legend=True, right_title_rows=None,
        right_title_overrides=None, group_xticklabels=None,
    ):
        """Panel labels, titles, colorbars, save — no manual axes packing."""
        for row_i in map_header_row_indices:
            if row_i >= len(row_axes):
                continue
            ax0, ax1 = row_axes[row_i][0], row_axes[row_i][1]
            ax0.text(0.5, 1.05, col0_header, fontsize=16, transform=ax0.transAxes, va='center', ha='center')
            ax1.text(0.5, 1.05, col1_header, fontsize=16, transform=ax1.transAxes, va='center', ha='center')

        all_axes = [
            ax for row in row_axes for ax in (row[0], row[2], row[1])
            if _is_labelable_panel_ax(ax)
        ]
        for _, _, ax_group in row_axes:
            if ax_group.axison:
                _sync_row_chart_ylim(ax_group)
        _apply_combined_group_xticklabels(row_axes, group_xticklabels or TFV_GROUP_ORDER)

        for row_i, ((ax_map_l, ax_map_r, _), title) in enumerate(zip(row_axes, row_title_texts)):
            if _is_populated_map_ax(ax_map_l):
                _place_fig_map_title(fig, ax_map_l, title)
            if _is_populated_map_ax(ax_map_r) and (right_title_rows is None or row_i in right_title_rows):
                right_title = (right_title_overrides or {}).get(row_i, title)
                _place_fig_map_title(fig, ax_map_r, right_title)

        if row_mid_map_titles is None:
            row_mid_map_titles = [None] * len(row_axes)
        for (ax_map_l, ax_map_r, _), mid_title in zip(row_axes, row_mid_map_titles):
            if not mid_title or not (_is_populated_map_ax(ax_map_l) and _is_populated_map_ax(ax_map_r)):
                continue
            pos_l, pos_r = ax_map_l.get_position(), ax_map_r.get_position()
            x_mid = pos_l.x1 + (pos_r.x0 - pos_l.x1) * _ROTATION_MID_TITLE_X_FRAC
            y_mid = 0.5 * (pos_l.y0 + pos_l.y1)
            fig.text(x_mid, y_mid, _wrap_row_title(mid_title), fontsize=16, rotation=90,
                     transform=fig.transFigure, va='center', ha='center')

        for ax, label in zip(all_axes, _subplot_labels_sequential(len(all_axes))):
            style = 'map' if _is_populated_map_ax(ax) else 'auto'
            _place_subplot_label(ax, label, fontsize=14, style=style)

        if show_top_legend:
            bbox_g = row_axes[0][2].get_position()
            fig.legend(
                handles=_BAR_LEGEND_ELEMENTS, loc='lower center',
                bbox_to_anchor=((bbox_g.x0 + bbox_g.x1) / 2, bbox_g.y1 + 0.01),
                bbox_transform=fig.transFigure, ncol=2, fontsize=_LEGEND_FONTSIZE, frameon=False,
            )

        for _, _, ax_group in row_axes:
            if ax_group.axison and getattr(ax_group, '_annotate_clipped_errors', False):
                _annotate_clipped_bar_errors(ax_group)

        map_axes = [ax for row in row_axes for ax in (row[0], row[1]) if _is_populated_map_ax(ax)]
        _attach_deferred_map_colorbars(map_axes)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.30)
        fig.show()
        plt.show(block=False)
    merged_natural = france_ser.merge(forest_trend_natural_agg, left_on='ser_num', right_on='ser')
    merged_harvest = france_ser.merge(forest_trend_harvest_agg, left_on='ser_num', right_on='ser')

    natural = trend_ok.loc[trend_ok.agent == 'natural'].copy()
    harvest = trend_ok.loc[trend_ok.agent == 'harvest'].copy()

    natural['TFV_group'] = natural['TFV'].map(dict_group_tfvs)
    harvest['TFV_group'] = harvest['TFV'].map(dict_group_tfvs)
    natural_sorted_tfvs_by_area = _build_sorted_tfvs_by_total_area(natural, harvest)
    natural_sorted_tfv_groups = TFV_GROUP_ORDER

    # Shared kwargs for combined map|dots|map rows
    _row_kw = dict(
        natural_df=natural, harvest_df=harvest, groups=natural_sorted_tfv_groups,
        merged_h=merged_harvest, merged_n=merged_natural,
    )

    # %%
    # Rate: area / evolution / patch size / patch-size evolution
    fig_rate, axes_rate = _make_disturbance_combined_figure(4)
    rows_rate = _unravel_combined_axes(axes_rate)
    (ax1, ax2, ax4b), (ax5, ax6, ax8b), (ax21, ax22, ax24b), (ax_ptev_h, ax_ptev_n, ax_ptev_b) = rows_rate

    _plot_combined_metric_row(
        ax1, ax2, ax4b, column='d_area_2020_perc', cmap='hot_r',
        title_h='Harvest rate', title_n='Natural disturbance rate',
        ylabel='Disturbance rate (%)', sigma_col='d_area_2020_error_perc',
        cbar_center_tick=False, ylim_mode='adaptive', ylim_kw=dict(y_floor=0.0),
        custom_yticks=([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.5', '0.75', '1']),
        top_legend=True, **_row_kw,
    )
    _plot_combined_metric_row(
        ax5, ax6, ax8b, column='d_area_evolution_perc', cmap='Spectral',
        title_h='Harvest disturbance rate trend relative to 2020',
        title_n='Natural disturbance rate trend relative to 2020',
        ylabel='Disturbance rate trend relative to 2020 (%/yr)',
        sigma_col='d_area_evolution_error_perc', vmin=-5.0, vmax=5.0,
        add_vline=True, ylim_mode='restrictive', ylim_kw=dict(pad_frac=1), **_row_kw,
    )
    # Mixed conifers (last group) natural: put both clipped whisker labels on the left.
    _mixed_conifers_nat_x = float(TFV_GROUP_ORDER.index('Mixed conifers')) + _DOT_WHISKER_OFFSET
    ax8b._clipped_label_x_nudges = {_mixed_conifers_nat_x: -0.05}
    _vmin_ph, _vmax_ph, _vmin_pn, _vmax_pn = _patch_size_split_vlims(merged_harvest, merged_natural)
    _plot_combined_metric_row(
        ax21, ax22, ax24b, column='mean_patch_size_2020', cmap='viridis',
        title_h='Harvest patch size', title_n='Natural patch size',
        ylabel='Disturbed patch size (ha)', sigma_col='mean_patch_size_2020_error',
        use_national_stderr=True, vmin_h=_vmin_ph, vmax_h=_vmax_ph, vmin_n=_vmin_pn, vmax_n=_vmax_pn,
        ylim_mode='error_specs', ylim_kw=dict(y_min=0.0), **_row_kw,
    )
    _plot_combined_metric_row(
        ax_ptev_h, ax_ptev_n, ax_ptev_b, column='mean_patch_size_evolution_perc', cmap='Spectral',
        title_h='Harvest patch size trend relative to 2020',
        title_n='Natural patch size trend relative to 2020',
        ylabel='Patch size trend relative to 2020 (%/yr)',
        sigma_col='mean_patch_size_evolution_error_perc', vmin=-5.0, vmax=5.0,
        add_vline=True, show_ylabels=True, ylim_mode='error_specs_both', **_row_kw,
    )
    _finalize_disturbance_combined_figure(
        fig_rate, rows_rate,
        [
            'Harvest rate',
            'Harvest disturbance rate trend relative to 2020',
            'Harvest patch size',
            'Harvest patch size trend relative to 2020',
        ],
        f'{_COMBINED_FIGURE_DIR}/Figure_4_disturbance_rate_and_patch_size.png',
        map_header_row_indices=(), show_top_legend=False,
        right_title_overrides={
            0: 'Natural disturbance rate',
            1: 'Natural disturbance rate trend relative to 2020',
            2: 'Natural patch size',
            3: 'Natural patch size trend relative to 2020',
        },
    )

    # %%
    # Severity + C loss: absolute / relative / AGB-loss intensity / total AGB loss
    fig_sev, axes_sev = _make_disturbance_combined_figure(4, spacer_width=0.58)
    rows_sev = _unravel_combined_axes(axes_sev)
    (ax9, ax10, ax12b), (ax13, ax14, ax16b), (ax_rxs_h, ax_rxs_n, ax_rxs_b), (ax17, ax18, ax20b) = rows_sev

    _plot_combined_metric_row(
        ax9, ax10, ax12b, column=AGBD_LOSS_COLUMN, cmap='hot_r',
        title_h='Harvest absolute severity',
        title_n='Natural absolute severity',
        ylabel='Disturbance absolute severity (MgC ha$_\\mathrm{D}^{-1}$)',
        ylim_mode='adaptive', ylim_kw=dict(pad_frac=0.12), **_row_kw,
    )
    _apply_nice_yticks(ax12b, nbins=4)
    _plot_combined_metric_row(
        ax13, ax14, ax16b, column=RELATIVE_AGBD_LOSS_COLUMN, cmap='hot_r',
        title_h='Harvest relative severity',
        title_n='Natural relative severity',
        ylabel='Disturbance relative severity (%)',
        sigma_col=relative_agbd_loss_error_col, require_column=True,
        ylim_mode='restrictive',
        ylim_kw=dict(pad_frac_below=0.25, pad_frac_above=0.35, include_zero=False),
        **_row_kw,
    )
    _apply_nice_yticks(ax16b, nbins=4)
    # Mixed conifers Natural (66): further right + below the axis for clearance from the whisker tip.
    _mixed_conifers_nat_x_sev = float(TFV_GROUP_ORDER.index('Mixed conifers')) + _DOT_WHISKER_OFFSET
    ax16b._clipped_label_x_nudges = {_mixed_conifers_nat_x_sev: 0.12}
    ax16b._clipped_label_bottom_ys = {_mixed_conifers_nat_x_sev: -0.06}

    _vmin_rxs, _vmax_rxs = _shared_map_vlim_second_max(
        (merged_harvest, RATE_X_SEVERITY_COLUMN),
        (merged_natural, RATE_X_SEVERITY_COLUMN),
    )
    _plot_combined_metric_row(
        ax_rxs_h, ax_rxs_n, ax_rxs_b, column=RATE_X_SEVERITY_COLUMN, cmap='viridis',
        title_h='Harvest AGB-loss intensity',
        title_n='Natural AGB-loss intensity',
        ylabel='AGB-loss intensity (MgC ha$_\\mathrm{F}^{-1}$)',
        sigma_col=RATE_X_SEVERITY_ERROR_COLUMN,
        vmin=_vmin_rxs, vmax=_vmax_rxs,
        ylim_mode='adaptive', ylim_kw=dict(y_floor=0.0, pad_frac=0.12), **_row_kw,
    )

    _vmin_agb, _ = _shared_map_vlim(
        (merged_harvest, 'lost_agb'), (merged_natural, 'lost_agb'),
    )
    _vmax_agb = 0.15
    _plot_ser_map(merged_harvest, 'lost_agb', 'viridis', ax17,
                  'Harvest total AGB loss', vmin=_vmin_agb, vmax=_vmax_agb)
    _plot_ser_map(merged_natural, 'lost_agb', 'viridis', ax18,
                  'Natural total AGB loss', vmin=_vmin_agb, vmax=_vmax_agb)
    plot_combined_bar_chart_by_group(
        ax20b, natural, harvest, 'lost_agb', 'Total AGB loss (TgC)',
        sorted_groups=natural_sorted_tfv_groups, show_ylabels=False,
        use_sum=True, error_column='lost_agb_error',
    )
    _lock_adaptive_dot_ylim(ax20b, y_floor=0.0)
    _set_custom_yticks(ax20b, [0, 0.5, 1, 1.5, 2, 2.5], ['0', '0.5', '1', '1.5', '2', '2.5'])

    _finalize_disturbance_combined_figure(
        fig_sev, rows_sev,
        [
            'Harvest absolute severity',
            'Harvest relative severity',
            'Harvest AGB-loss intensity',
            'Harvest total AGB loss',
        ],
        f'{_COMBINED_FIGURE_DIR}/Figure_5_disturbance_severity_and_agb_lost.png',
        map_header_row_indices=(), show_top_legend=True,
        right_title_overrides={
            0: 'Natural absolute severity',
            1: 'Natural relative severity',
            2: 'Natural AGB-loss intensity',
            3: 'Natural total AGB loss',
        },
    )

    # %%
    # Rate by TFV: disturbance rate / trend / patch size / patch-size trend
    fig_tfv_rate, axes_tfv_rate = _make_si_tfv_vertical_figure(4)
    _si_kw = dict(natural_df=natural, harvest_df=harvest, sorted_tfvs=natural_sorted_tfvs_by_area)
    _plot_si_tfv_panel(
        axes_tfv_rate[0], column='d_area_2020_perc', ylabel='Disturbance rate (%)',
        sigma_col='d_area_2020_error_perc', ylim_kw=dict(pad_frac_below=0.05, pad_frac_above=0.15),
        **_si_kw,
    )
    _plot_si_tfv_panel(
        axes_tfv_rate[1], column='d_area_evolution_perc',
        ylabel='Disturbance rate trend relative to 2020 (%/yr)',
        sigma_col='d_area_evolution_error_perc', add_vline=True, ylim_kw=dict(pad_frac=0.5),
        **_si_kw,
    )
    _plot_si_tfv_panel(
        axes_tfv_rate[2], column='mean_patch_size_2020', ylabel='Disturbed patch size (ha)',
        sigma_col='mean_patch_size_2020_error', ylim_kw=dict(pad_frac_below=0.05, pad_frac_above=0.15),
        **_si_kw,
    )
    _plot_si_tfv_panel(
        axes_tfv_rate[3], column='mean_patch_size_evolution_perc',
        ylabel='Patch size trend relative to 2020 (%/yr)',
        sigma_col='mean_patch_size_evolution_error_perc', add_vline=True, ylim_kw=dict(pad_frac=0.5),
        **_si_kw,
    )
    _finalize_si_tfv_vertical_figure(
        fig_tfv_rate, axes_tfv_rate,
        f'{_COMBINED_FIGURE_DIR}/Figure_SI8_disturbance_rate_by_species_category.png',
        sorted_tfvs=natural_sorted_tfvs_by_area,
    )

    # %%
    # Severity + AGB loss by TFV: absolute / relative / intensity / total
    fig_tfv_sev, axes_tfv_sev = _make_si_tfv_vertical_figure(4)
    _plot_si_tfv_panel(
        axes_tfv_sev[0], column=AGBD_LOSS_COLUMN,
        ylabel='Disturbance absolute severity (MgC ha$_\\mathrm{D}^{-1}$)',
        ylim_kw=dict(pad_frac_below=0.05, pad_frac_above=0.15, include_zero=True),
        **_si_kw,
    )
    _plot_si_tfv_panel(
        axes_tfv_sev[1], column=RELATIVE_AGBD_LOSS_COLUMN,
        ylabel='Disturbance relative severity (%)',
        sigma_col=relative_agbd_loss_error_col, require_column=True,
        # Zoom around means (relative losses sit near 100% for most TFVs).
        ylim_kw=dict(pad_frac_below=0.05, pad_frac_above=0.15, include_zero=False),
        **_si_kw,
    )
    _plot_si_tfv_panel(
        axes_tfv_sev[2], column=RATE_X_SEVERITY_COLUMN,
        ylabel='AGB-loss intensity (MgC ha$_\\mathrm{F}^{-1}$)',
        sigma_col=RATE_X_SEVERITY_ERROR_COLUMN,
        ylim_kw=dict(pad_frac_below=0.05, pad_frac_above=0.15, include_zero=True),
        **_si_kw,
    )
    plot_combined_bar_chart(
        axes_tfv_sev[3], natural, harvest, 'lost_agb', 'Total AGB loss (TgC)',
        sorted_tfvs=natural_sorted_tfvs_by_area, show_ylabels=False, show_yaxis=True,
        use_sum=True, error_column='lost_agb_error', si_style=True,
    )
    _lock_adaptive_dot_ylim(axes_tfv_sev[3], y_floor=0.0, pad_frac=0.08)
    _finalize_si_tfv_vertical_figure(
        fig_tfv_sev, axes_tfv_sev,
        f'{_COMBINED_FIGURE_DIR}/Figure_SI10_disturbance_severity_by_species_category.png',
        sorted_tfvs=natural_sorted_tfvs_by_area,
    )

    # %%
    # Relative uncertainty — shared column prep
    _unc_rate_pairs = [
        ('d_area_2020_perc', 'd_area_2020_error_perc', 'rel_unc_d_area_2020_perc'),
        ('d_area_evolution_perc', 'd_area_evolution_error_perc', 'rel_unc_d_area_evolution_perc'),
        ('mean_patch_size_2020', 'mean_patch_size_2020_error', 'rel_unc_mean_patch_size_2020'),
        ('mean_patch_size_evolution_perc', 'mean_patch_size_evolution_error_perc',
         'rel_unc_mean_patch_size_evolution_perc'),
    ]
    _unc_sev_pairs = [
        (AGBD_LOSS_COLUMN, agbd_loss_error_col, 'rel_unc_agbd_loss'),
        (RELATIVE_AGBD_LOSS_COLUMN, relative_agbd_loss_error_col, 'rel_unc_relative_agbd_loss'),
    ]
    _unc_loss_pairs = [
        (RATE_X_SEVERITY_COLUMN, RATE_X_SEVERITY_ERROR_COLUMN, 'rel_unc_rate_x_severity'),
        ('lost_agb', 'lost_agb_error', 'rel_unc_lost_agb'),
    ]
    _unc_all_pairs = _unc_rate_pairs + _unc_sev_pairs + _unc_loss_pairs
    merged_harvest_unc = _append_relative_uncertainty_columns(merged_harvest, _unc_all_pairs)
    merged_natural_unc = _append_relative_uncertainty_columns(merged_natural, _unc_all_pairs)


    def _build_relative_uncertainty_figure(n_rows, figsize, specs, row_titles, save_name,
                                           harvest_gdf, natural_gdf, vmax_overrides=None):
        """``specs``: list of (column, has_natural) or (column, label_h, label_n).

        Panel letters are assigned sequentially over populated panels (a, b, …).
        """
        vmax_overrides = vmax_overrides or {}
        fig, axes = plt.subplots(
            n_rows, 2, figsize=figsize,
            gridspec_kw={'wspace': 0.0, 'hspace': 0.18},
        )
        if n_rows == 1:
            axes = np.asarray([axes])
        axes_grid = []
        label_i = 0
        for row_i, spec in enumerate(specs):
            if len(spec) == 2:
                col, has_natural = spec
                label_n_wanted = bool(has_natural)
            else:
                col, label_h, label_n = spec
                label_n_wanted = label_n is not None
            ax_h, ax_n = axes[row_i]
            vmin_h, vmax_h = _shared_map_vlim((harvest_gdf, col),)
            if col in vmax_overrides:
                vmax_h = vmax_overrides[col]
            if vmax_h is not None:
                vmax_h = min(float(vmax_h), _REL_UNC_VMAP_CAP)
            lab_h = chr(ord('a') + label_i)
            label_i += 1
            plot_relative_uncertainty_map(
                harvest_gdf, col, ax_h, panel_label=lab_h, vmin=vmin_h, vmax=vmax_h,
            )
            if label_n_wanted:
                vmin_n, vmax_n = _shared_map_vlim((natural_gdf, col),)
                if col in vmax_overrides:
                    vmax_n = vmax_overrides[col]
                if vmax_n is not None:
                    vmax_n = min(float(vmax_n), _REL_UNC_VMAP_CAP)
                lab_n = chr(ord('a') + label_i)
                label_i += 1
                plot_relative_uncertainty_map(
                    natural_gdf, col, ax_n, panel_label=lab_n, vmin=vmin_n, vmax=vmax_n,
                )
            else:
                ax_n.set_axis_off()
            axes_grid.append((ax_h, ax_n))
        _finalize_relative_uncertainty_maps_figure(
            fig, axes_grid, row_titles,
            f'{_COMBINED_FIGURE_DIR}/{save_name}',
            col_headers=('', ''),
        )


    # Rate relative uncertainty: titles match Rate combined-figure maps
    # Width ~9.5 keeps equal-aspect France maps packed (14" left a large mid-gap).
    _build_relative_uncertainty_figure(
        4, (9.5, 16),
        [
            ('rel_unc_d_area_2020_perc', True),
            ('rel_unc_d_area_evolution_perc', True),
            ('rel_unc_mean_patch_size_2020', True),
            ('rel_unc_mean_patch_size_evolution_perc', True),
        ],
        [
            ('Harvest rate', 'Natural disturbance rate'),
            (
                'Harvest disturbance rate trend relative to 2020',
                'Natural disturbance rate trend relative to 2020',
            ),
            ('Harvest patch size', 'Natural patch size'),
            (
                'Harvest patch size trend relative to 2020',
                'Natural patch size trend relative to 2020',
            ),
        ],
        'Figure_SI7_relative_uncertainty_disturbance_rate.png',
        merged_harvest_unc, merged_natural_unc,
    )

    # Severity + AGB-loss relative uncertainty: absolute / relative / intensity / total
    _build_relative_uncertainty_figure(
        4, (9.5, 16),
        [
            ('rel_unc_agbd_loss', True),
            ('rel_unc_relative_agbd_loss', True),
            ('rel_unc_rate_x_severity', True),
            ('rel_unc_lost_agb', True),
        ],
        [
            ('Harvest absolute severity', 'Natural absolute severity'),
            ('Harvest relative severity', 'Natural relative severity'),
            ('Harvest AGB-loss intensity', 'Natural AGB-loss intensity'),
            ('Harvest total AGB loss', 'Natural total AGB loss'),
        ],
        'Figure_SI9_relative_uncertainty_severity_and_agb_lost.png',
        merged_harvest_unc, merged_natural_unc,
    )

    # %%
    # SI11 — Kowalski + ours rate-weighted severity (Fig.5 a/c × Fig.4 a/c) vs Kowalski SER
    write_si11_kowalski_figure(
        _COMBINED_FIGURE_DIR,
        loss_csv=_kowalski_csv,
        grid_shp=_kowalski_grid,
        france_shp=_FRANCE_OUTLINE_SHP,
        ser_shapefile=ser_shapefile,
        harvest_agg=forest_trend_harvest_agg,
        natural_agg=forest_trend_natural_agg,
        abs_sev_col=AGBD_LOSS_COLUMN,
        rel_sev_col=RELATIVE_AGBD_LOSS_COLUMN,
        rate_col='d_area_2020_perc',
    )

    # %%
    # National uncertainty table (after figures)
    _nat_national = trend_ok.loc[trend_ok.agent == 'natural']
    _harvest_national = trend_ok.loc[trend_ok.agent == 'harvest']
    national_uncertainty_table = build_national_uncertainty_table(_nat_national, _harvest_national)
    print_national_uncertainty_table(national_uncertainty_table)
    plt.show()

    # %%
    # ========== VARIANCE DECOMPOSITION: SER / TFV / agent (one-way only) ==========
    # Equal weight per SER×TFV×agent cell. η² + MC 95% CI (row-level metric SEs) +
    # permutation p. Region = SER. No stratified analysis.

    _USE_AREA_WEIGHTS_DIST = False  # match Fig_age_distribution
    _ETA_PERM_B_DIST = 999
    _ETA_MC_B_DIST = 1000


    def _wmean_d(vals, weights):
        vals = np.asarray(vals, dtype=float)
        weights = np.asarray(weights, dtype=float)
        ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if not np.any(ok):
            return np.nan
        return float(np.average(vals[ok], weights=weights[ok]))


    def _wvar_d(vals, weights):
        vals = np.asarray(vals, dtype=float)
        weights = np.asarray(weights, dtype=float)
        ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if ok.sum() < 2:
            return np.nan
        v, w = vals[ok], weights[ok]
        mu = np.average(v, weights=w)
        return float(np.average((v - mu) ** 2, weights=w))


    def _wstd_d(vals, weights):
        v = _wvar_d(vals, weights)
        return float(np.sqrt(v)) if np.isfinite(v) else np.nan


    def _wcv_d(vals, weights):
        mu = _wmean_d(vals, weights)
        sd = _wstd_d(vals, weights)
        if not (np.isfinite(mu) and np.isfinite(sd)) or mu == 0:
            return np.nan
        return float(sd / abs(mu))


    def _eta2_arrays_d(y, w, g):
        """Weighted η² from arrays."""
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        g = np.asarray(g)
        ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & pd.notna(g)
        if ok.sum() < 2:
            return np.nan
        y, w, g = y[ok], w[ok], g[ok]
        codes, uniques = pd.factorize(g, sort=False)
        if len(uniques) < 2:
            return np.nan
        mu = float(np.average(y, weights=w))
        sst = float(np.sum(w * (y - mu) ** 2))
        if sst <= 0:
            return np.nan
        nlev = len(uniques)
        sum_w = np.bincount(codes, weights=w, minlength=nlev).astype(float)
        sum_wy = np.bincount(codes, weights=w * y, minlength=nlev).astype(float)
        pos = sum_w > 0
        mug = np.zeros(nlev, dtype=float)
        mug[pos] = sum_wy[pos] / sum_w[pos]
        ssb = float(np.sum(sum_w[pos] * (mug[pos] - mu) ** 2))
        return float(ssb / sst)


    def _eta2_weighted_d(df, ycol, gcol, wcol='area'):
        return _eta2_arrays_d(df[ycol].to_numpy(), df[wcol].to_numpy(), df[gcol].to_numpy())


    def _perm_p_eta2_d(df, ycol, gcol, wcol='area', n_perm=_ETA_PERM_B_DIST, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        y = pd.to_numeric(df[ycol], errors='coerce').to_numpy(dtype=float)
        w = pd.to_numeric(df[wcol], errors='coerce').to_numpy(dtype=float)
        g0 = df[gcol].to_numpy().copy()
        obs = _eta2_arrays_d(y, w, g0)
        if not np.isfinite(obs):
            return np.nan
        n_ge = n_ok = 0
        for _ in range(n_perm):
            eta = _eta2_arrays_d(y, w, rng.permutation(g0))
            if not np.isfinite(eta):
                continue
            n_ok += 1
            if eta >= obs - 1e-15:
                n_ge += 1
        if n_ok == 0:
            return np.nan
        return float((1 + n_ge) / (1 + n_ok))


    def _fmt_p_d(p):
        if not np.isfinite(p):
            return 'p=nan'
        if p < 0.001:
            return 'p<0.001'
        return f'p={p:.3f}'


    def _fmt_eta_ci_d(point, lo, hi, p=None):
        if not np.isfinite(point):
            base = '   nan'
        elif not (np.isfinite(lo) and np.isfinite(hi)):
            base = f'{point:5.1f}'
        else:
            base = f'{point:5.1f} [{lo:.1f}-{hi:.1f}]'
        if p is None:
            return base
        return f'{base} {_fmt_p_d(p)}'


    def _row_ratio_pct(slope, level):
        slope = np.asarray(slope, dtype=float)
        level = np.asarray(level, dtype=float)
        out = np.full(slope.shape, np.nan, dtype=float)
        ok = np.isfinite(slope) & np.isfinite(level) & (level > 0)
        out[ok] = 100.0 * slope[ok] / level[ok]
        return out


    def _row_ratio_pct_se(slope, slope_se, level, level_se, cov=None):
        """Delta-method SE of 100 × slope / level."""
        slope = np.asarray(slope, dtype=float)
        slope_se = np.asarray(slope_se, dtype=float)
        level = np.asarray(level, dtype=float)
        level_se = np.asarray(level_se, dtype=float)
        if cov is None:
            cov = np.zeros_like(slope)
        else:
            cov = np.asarray(cov, dtype=float)
        out = np.full(slope.shape, np.nan, dtype=float)
        ok = (
            np.isfinite(slope) & np.isfinite(slope_se) & np.isfinite(level) & np.isfinite(level_se)
            & (level > 0) & (slope_se >= 0) & (level_se >= 0)
        )
        s, ss, L, sL, c = slope[ok], slope_se[ok], level[ok], level_se[ok], cov[ok]
        cap = ss * sL
        c = np.clip(c, -cap, cap)
        var = (
            (100.0 / L) ** 2 * ss ** 2
            + (100.0 * s / L ** 2) ** 2 * sL ** 2
            - 2.0 * (100.0 / L) * (100.0 * s / L ** 2) * c
        )
        out[ok] = np.sqrt(np.maximum(var, 0.0))
        return out


    # --- Build analysis table (SER × TFV × agent) ---
    _vd = trend_ok.copy()
    if 'aggregation_level' in _vd.columns:
        _vd = _vd.loc[_vd['aggregation_level'].astype(str) == 'ser'].copy()

    _vd['TFV'] = pd.to_numeric(_vd['TFV'], errors='coerce')
    _vd = _vd.loc[_vd['TFV'].notna() & (_vd['TFV'] > 0) & (_vd['TFV'] < 500) & (_vd['TFV'] != 400)].copy()
    _vd['TFV'] = _vd['TFV'].astype(int)
    _vd['ser'] = pd.to_numeric(_vd['ser'], errors='coerce')
    _vd = _vd.loc[_vd['ser'].notna()].copy()
    _vd['ser'] = _vd['ser'].astype(int)
    _vd['agent'] = _vd['agent'].astype(str).str.strip().str.lower()
    _vd = _vd.loc[_vd['agent'].isin(('harvest', 'natural'))].copy()

    _vd['d_area_evolution_perc'] = _row_ratio_pct(_vd['slope_pp'], _vd['d_area_2020_perc'])
    _vd['d_area_evolution_perc_se'] = _row_ratio_pct_se(
        _vd['slope_pp'], _vd['slope_pp_err'],
        _vd['d_area_2020_perc'], _vd['d_area_2020_error_perc'],
        cov=_vd.get('d_area_level_slope_cov'),
    )
    _vd['mean_patch_size_evolution_perc'] = _row_ratio_pct(
        _vd['slope_patch_ha'], _vd['mean_patch_size_2020'],
    )
    _vd['mean_patch_size_evolution_perc_se'] = _row_ratio_pct_se(
        _vd['slope_patch_ha'], _vd['slope_patch_ha_err'],
        _vd['mean_patch_size_2020'], _vd['mean_patch_size_2020_error'],
        cov=_vd.get('patch_level_slope_cov'),
    )

    _vd['tfv_group'] = _vd['TFV'].map(dict_group_tfvs)
    _vd['area'] = pd.to_numeric(_vd['total_area'], errors='coerce').fillna(0.0)
    _vd = _vd.loc[_vd['area'] > 0].copy()
    if not _USE_AREA_WEIGHTS_DIST:
        _vd['area'] = 1.0
    # Include open/new-cut TFVs 401–403 (mapped into Mixed_*); TFV 400 already dropped.
    _vd = _vd.loc[_vd['tfv_group'].isin(TFV_GROUP_ORDER)].copy()
    _vd = _vd.drop_duplicates(subset=['ser', 'TFV', 'agent'], keep='first')

    # Value column → row-level SE column for MC
    _DIST_SE_MAP = {
        'd_area_2020_perc': 'd_area_2020_error_perc',
        'd_area_evolution_perc': 'd_area_evolution_perc_se',
        'mean_patch_size_2020': 'mean_patch_size_2020_error',
        'mean_patch_size_evolution_perc': 'mean_patch_size_evolution_perc_se',
        AGBD_LOSS_COLUMN: f'{AGBD_LOSS_COLUMN}_error',
        RELATIVE_AGBD_LOSS_COLUMN: f'{RELATIVE_AGBD_LOSS_COLUMN}_error',
        RATE_X_SEVERITY_COLUMN: RATE_X_SEVERITY_ERROR_COLUMN,
        ROTATION_PERIOD_COLUMN: ROTATION_PERIOD_ERROR_COLUMN,
    }
    _DIST_NONNEG = {
        'd_area_2020_perc', 'mean_patch_size_2020', AGBD_LOSS_COLUMN,
        RELATIVE_AGBD_LOSS_COLUMN, RATE_X_SEVERITY_COLUMN, ROTATION_PERIOD_COLUMN,
    }

    _DIST_PARAM_SPEC = [
        ('d_area_2020_perc', 'Rate', True),
        ('d_area_evolution_perc', 'Rate trend', True),
        ('mean_patch_size_2020', 'Patch size', True),
        ('mean_patch_size_evolution_perc', 'Patch trend', True),
        (AGBD_LOSS_COLUMN, 'Abs. severity', True),
        (RELATIVE_AGBD_LOSS_COLUMN, 'Rel. severity', True),
        (RATE_X_SEVERITY_COLUMN, 'AGB intensity', True),
        (ROTATION_PERIOD_COLUMN, 'Rotation', False),
    ]
    _DIST_PARAM_SPEC = [(c, n, a) for c, n, a in _DIST_PARAM_SPEC if c in _vd.columns]
    _dist_param_names = [n for _, n, _ in _DIST_PARAM_SPEC]

    print("\n" + "=" * 90)
    print(
        "ONE-WAY η² — SER vs TFV vs agent  "
        + ("(AREA-WEIGHTED)" if _USE_AREA_WEIGHTS_DIST else "(EQUAL WEIGHT PER CELL)")
    )
    print("=" * 90)
    print(
        f"n={len(_vd)} SER×TFV×agent cells | "
        f"harvest={(_vd['agent'] == 'harvest').sum()} | "
        f"natural={(_vd['agent'] == 'natural').sum()} | "
        f"MC B={_ETA_MC_B_DIST} | perm. p B={_ETA_PERM_B_DIST}"
    )
    print("One-way η² do not partition variance additively across factors.")

    # --- One-way point η² ---
    _dist_variance_decomp = []
    for param, pname, agent_ok in _DIST_PARAM_SPEC:
        eta_s = _eta2_weighted_d(_vd, param, 'ser', 'area')
        eta_t = _eta2_weighted_d(_vd, param, 'TFV', 'area')
        eta_a = _eta2_weighted_d(_vd, param, 'agent', 'area') if agent_ok else np.nan
        _dist_variance_decomp.append({
            'parameter': pname,
            'column': param,
            'agent_factor': agent_ok,
            'eta2_ser_pct': 100.0 * eta_s if np.isfinite(eta_s) else np.nan,
            'eta2_tfv_pct': 100.0 * eta_t if np.isfinite(eta_t) else np.nan,
            'eta2_agent_pct': 100.0 * eta_a if np.isfinite(eta_a) else np.nan,
        })
    dist_variance_df = pd.DataFrame(_dist_variance_decomp)
    _avg_s = dist_variance_df['eta2_ser_pct'].mean()
    _avg_t = dist_variance_df['eta2_tfv_pct'].mean()
    _avg_a = dist_variance_df.loc[dist_variance_df['agent_factor'], 'eta2_agent_pct'].mean()

    # --- MC 95% CI from row-level metric SEs ---
    print("\nRunning MC CI for η²...")
    _mc_keys = []
    for _, pname, agent_ok in _DIST_PARAM_SPEC:
        _mc_keys += [f'SER::{pname}', f'TFV::{pname}']
        if agent_ok:
            _mc_keys.append(f'agent::{pname}')
    _mc_store = {k: [] for k in _mc_keys}
    _mc_mean_ser, _mc_mean_tfv, _mc_mean_agent = [], [], []

    _se_cols = {}
    for param, pname, _ in _DIST_PARAM_SPEC:
        se_c = _DIST_SE_MAP.get(param)
        if se_c and se_c in _vd.columns:
            _se_cols[param] = np.maximum(
                pd.to_numeric(_vd[se_c], errors='coerce').fillna(0.0).to_numpy(dtype=float), 0.0
            )
        else:
            _se_cols[param] = np.zeros(len(_vd), dtype=float)
            print(f'  WARNING: no SE for {param!r} — MC uses σ=0')

    _y0 = {
        param: pd.to_numeric(_vd[param], errors='coerce').to_numpy(dtype=float)
        for param, _, _ in _DIST_PARAM_SPEC
    }
    _w_mc = _vd['area'].to_numpy(dtype=float)
    _ser_mc = _vd['ser'].to_numpy()
    _tfv_mc = _vd['TFV'].to_numpy()
    _agent_mc = _vd['agent'].to_numpy()
    _rng_mc = np.random.default_rng(0)

    for _b in range(_ETA_MC_B_DIST):
        _draw_ser, _draw_tfv, _draw_agent = [], [], []
        for param, pname, agent_ok in _DIST_PARAM_SPEC:
            yp = _y0[param] + _rng_mc.normal(0.0, _se_cols[param])
            if param in _DIST_NONNEG:
                yp = np.maximum(yp, 0.0)
            if param == RELATIVE_AGBD_LOSS_COLUMN:
                yp = np.clip(yp, 0.0, 100.0)
            eta_s = _eta2_arrays_d(yp, _w_mc, _ser_mc)
            eta_t = _eta2_arrays_d(yp, _w_mc, _tfv_mc)
            if np.isfinite(eta_s):
                v = 100.0 * eta_s
                _mc_store[f'SER::{pname}'].append(v)
                _draw_ser.append(v)
            if np.isfinite(eta_t):
                v = 100.0 * eta_t
                _mc_store[f'TFV::{pname}'].append(v)
                _draw_tfv.append(v)
            if agent_ok:
                eta_a = _eta2_arrays_d(yp, _w_mc, _agent_mc)
                if np.isfinite(eta_a):
                    v = 100.0 * eta_a
                    _mc_store[f'agent::{pname}'].append(v)
                    _draw_agent.append(v)
        if _draw_ser:
            _mc_mean_ser.append(float(np.mean(_draw_ser)))
        if _draw_tfv:
            _mc_mean_tfv.append(float(np.mean(_draw_tfv)))
        if _draw_agent:
            _mc_mean_agent.append(float(np.mean(_draw_agent)))

    _mc_ci = {
        k: (
            float(np.nanpercentile(v, 2.5)),
            float(np.nanpercentile(v, 97.5)),
        ) if len(v) else (np.nan, np.nan)
        for k, v in _mc_store.items()
    }


    def _mc_mean_ci(vals):
        if not vals:
            return np.nan, np.nan
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    # --- Permutation p ---
    print("Running permutation tests for η²...")
    _rng_perm_d = np.random.default_rng(42)
    _eta_p_d = {}
    for param, pname, agent_ok in _DIST_PARAM_SPEC:
        _eta_p_d[f'SER::{pname}'] = _perm_p_eta2_d(_vd, param, 'ser', rng=_rng_perm_d)
        _eta_p_d[f'TFV::{pname}'] = _perm_p_eta2_d(_vd, param, 'TFV', rng=_rng_perm_d)
        _eta_p_d[f'agent::{pname}'] = (
            _perm_p_eta2_d(_vd, param, 'agent', rng=_rng_perm_d) if agent_ok else np.nan
        )

    # Slim table: η² + MC CI + p only
    _vd_slim = []
    for _, row in dist_variance_df.iterrows():
        p = row['parameter']
        s_lo, s_hi = _mc_ci.get(f'SER::{p}', (np.nan, np.nan))
        t_lo, t_hi = _mc_ci.get(f'TFV::{p}', (np.nan, np.nan))
        a_lo, a_hi = _mc_ci.get(f'agent::{p}', (np.nan, np.nan))
        _vd_slim.append({
            'parameter': p,
            'agent_factor': row['agent_factor'],
            'eta2_ser_pct': row['eta2_ser_pct'],
            'eta2_ser_lo': s_lo,
            'eta2_ser_hi': s_hi,
            'p_ser': _eta_p_d.get(f'SER::{p}', np.nan),
            'eta2_tfv_pct': row['eta2_tfv_pct'],
            'eta2_tfv_lo': t_lo,
            'eta2_tfv_hi': t_hi,
            'p_tfv': _eta_p_d.get(f'TFV::{p}', np.nan),
            'eta2_agent_pct': row['eta2_agent_pct'],
            'eta2_agent_lo': a_lo if row['agent_factor'] else np.nan,
            'eta2_agent_hi': a_hi if row['agent_factor'] else np.nan,
            'p_agent': _eta_p_d.get(f'agent::{p}', np.nan),
        })
    dist_variance_df = pd.DataFrame(_vd_slim)

    print("\nη² (%)  [95% MC]  permutation p (one-way):")
    print(f"{'parameter':<14} {'SER':>28} {'TFV':>28} {'agent':>28}")
    for _, row in dist_variance_df.iterrows():
        a_cell = (
            _fmt_eta_ci_d(row['eta2_agent_pct'], row['eta2_agent_lo'], row['eta2_agent_hi'], row['p_agent'])
            if row['agent_factor'] else 'n/a'
        )
        print(
            f"{row['parameter']:<14} "
            f"{_fmt_eta_ci_d(row['eta2_ser_pct'], row['eta2_ser_lo'], row['eta2_ser_hi'], row['p_ser']):>28} "
            f"{_fmt_eta_ci_d(row['eta2_tfv_pct'], row['eta2_tfv_lo'], row['eta2_tfv_hi'], row['p_tfv']):>28} "
            f"{a_cell:>28}"
        )
    # Mean-row CI = percentiles of (mean η² across params) per MC draw — not mean of CIs
    _s_lo, _s_hi = _mc_mean_ci(_mc_mean_ser)
    _t_lo, _t_hi = _mc_mean_ci(_mc_mean_tfv)
    _a_lo, _a_hi = _mc_mean_ci(_mc_mean_agent)
    print(
        f"{'Mean':<14} "
        f"{_fmt_eta_ci_d(_avg_s, _s_lo, _s_hi):>28} "
        f"{_fmt_eta_ci_d(_avg_t, _t_lo, _t_hi):>28} "
        f"{_fmt_eta_ci_d(_avg_a, _a_lo, _a_hi):>28}"
    )
    _who_bits = [('SER', _avg_s), ('TFV', _avg_t)]
    if np.isfinite(_avg_a):
        _who_bits.append(('agent', _avg_a))
    _who_bits.sort(key=lambda x: -x[1] if np.isfinite(x[1]) else -np.inf)
    print("→ One-way mean η² ranking: " + " > ".join(f"{k} ({v:.1f}%)" for k, v in _who_bits))
    print('p-values: permutation test of factor labels. Rotation: agent η² skipped.')

    dist_variance_paper_summary = {
        'n_cells': int(len(_vd)),
        'area_weighted': bool(_USE_AREA_WEIGHTS_DIST),
        'avg_eta2_ser_pct': float(_avg_s),
        'avg_eta2_ser_lo': float(_s_lo) if np.isfinite(_s_lo) else np.nan,
        'avg_eta2_ser_hi': float(_s_hi) if np.isfinite(_s_hi) else np.nan,
        'avg_eta2_tfv_pct': float(_avg_t),
        'avg_eta2_tfv_lo': float(_t_lo) if np.isfinite(_t_lo) else np.nan,
        'avg_eta2_tfv_hi': float(_t_hi) if np.isfinite(_t_hi) else np.nan,
        'avg_eta2_agent_pct': float(_avg_a) if np.isfinite(_avg_a) else np.nan,
        'avg_eta2_agent_lo': float(_a_lo) if np.isfinite(_a_lo) else np.nan,
        'avg_eta2_agent_hi': float(_a_hi) if np.isfinite(_a_hi) else np.nan,
    }
    print("\ndist_variance_df / dist_variance_paper_summary ready "
          "(η² + 95% MC CI + permutation p; one-way only).")
    # %%

    return figure_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trend_csv", type=Path)
    p.add_argument("figure_dir", type=Path)
    p.add_argument("--ser-shapefile", type=Path, default=None)
    p.add_argument("--greco-shapefile", type=Path, default=None)
    args = p.parse_args()
    kw = {}
    if args.ser_shapefile:
        kw["ser_shapefile"] = args.ser_shapefile
    if args.greco_shapefile:
        kw["greco_shapefile"] = args.greco_shapefile
    run_figures(args.trend_csv, args.figure_dir, **kw)
