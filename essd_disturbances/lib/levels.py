"""2020 disturbance levels: one rule shared by the trends output (02), the figures and 07 (η²).

No imports from other ``lib`` modules, so anything can import it without cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_REF = 2020  # year the levels are reported for
PATCH_PIXEL_AREA_HA = 0.09  # 30 m pixel
LEVEL_TOL = 1e-12  # float noise on the fitted 2020 level


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _kept(trend, fitted: np.ndarray, invalid_below: float) -> np.ndarray:
    """Rows whose fitted 2020 line is used: significant trend and physical 2020 value.

    ``LEVEL_TOL`` absorbs float noise, so a line that reaches the floor exactly (e.g. a
    rate of 0 in 2020) is kept whether it is computed in memory or re-read from CSV.
    """
    sig = trend.astype(str).to_numpy() != "no trend"
    return sig & np.isfinite(fitted) & (fitted >= invalid_below - LEVEL_TOL)


def _fitted_or_fallback(
    trend,
    intercept,
    slope,
    intercept_se,
    slope_se,
    median,
    median_se,
    *,
    year_start: int,
    invalid_below: float,
    fallback_floor: float,
):
    """2020 level from intercept + (YEAR_REF − year_start)·slope, or the fallback level.

    The fitted line is kept when the trend is significant and its 2020 value is
    >= ``invalid_below``; otherwise the level is ``median`` (floored) with SE ``median_se``.
    The point slope is 0 when the line is not kept; the slope SE stays the regression SE
    (0 where there is no fit). Covariance is (YEAR_REF − year_start)·σ_slope² on kept rows.
    Returns (level, level_se, slope, slope_se, cov). Shared by 02 (add_2020_levels),
    the figures and 07.
    """
    dt = float(YEAR_REF - year_start)
    a = _as_float(intercept)
    b = _as_float(slope)
    sa = _as_float(intercept_se)
    sb = np.maximum(np.nan_to_num(_as_float(slope_se), nan=0.0), 0.0)  # no fit → 0
    med = _as_float(median)
    sm = np.maximum(_as_float(median_se), 0.0)
    fitted = a + b * dt
    fitted_se = np.sqrt(sa**2 + (sb * dt) ** 2)
    keep = _kept(trend, fitted, invalid_below)
    fallback = np.where(np.isfinite(med), med, fallback_floor)
    fallback = np.maximum(fallback, fallback_floor)
    level = np.where(keep, fitted, fallback)
    level_se = np.maximum(np.where(keep, fitted_se, sm), 0.0)
    slope_hat = np.where(keep, b, 0.0)
    cov = np.where(keep, dt * sb**2, 0.0)
    return level, level_se, slope_hat, sb, cov


# metric -> (fallback level column, its SE column, floor in the metric's units, unit factor)
LEVEL_METRICS = {
    "disturbance_rate": ("mean_disturbance_rate", "mean_std_disturbance_rate", 0.0, 1.0),
    "mean_patch_size": ("median_mean_patch_size", "median_std_mean_patch_size", 1.0, PATCH_PIXEL_AREA_HA),
}


def add_2020_levels(df: pd.DataFrame, *, year_start: int, min_years: int) -> pd.DataFrame:
    """Add the published 2020 columns to the trends table.

    For each metric in ``LEVEL_METRICS``:
      ``<metric>_2020`` / ``<metric>_2020_error``: level and SE in YEAR_REF
      (rate as a fraction per year, patch size in ha);
      ``slope_type_<metric>``: significant (fitted line used) / zero (≥ min_years
      years with disturbance, fallback level) / null (1 .. min_years−1 years) /
      none (never disturbed).
    """
    out = df.copy()
    npos = pd.to_numeric(out["n_years_positive"], errors="coerce").fillna(0).to_numpy()
    dt = float(YEAR_REF - year_start)
    for metric, (level_col, se_col, floor, unit) in LEVEL_METRICS.items():
        trend = out[f"trend_{metric}"]
        level, level_se, *_ = _fitted_or_fallback(
            trend, out[f"intercept_{metric}"], out[f"slope_{metric}"],
            out[f"intercept_std_{metric}"], out[f"slope_std_{metric}"],
            out[level_col], out[se_col],
            year_start=year_start, invalid_below=floor, fallback_floor=floor,
        )
        fitted = _as_float(out[f"intercept_{metric}"]) + _as_float(out[f"slope_{metric}"]) * dt
        keep = _kept(trend, fitted, floor)
        if (keep & (npos < min_years)).any():
            raise ValueError(f"{metric}: significant trend with fewer than {min_years} years")
        out[f"slope_type_{metric}"] = np.where(
            npos == 0, "none", np.where(npos < min_years, "null", np.where(keep, "significant", "zero"))
        )
        out[f"{metric}_2020"] = level * unit
        out[f"{metric}_2020_error"] = level_se * unit
    return out
