"""Mann–Kendall trend tests and related helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymannkendall import original_test


def finite_sample_correction(n: np.ndarray | pd.Series) -> np.ndarray:
    """Small-n SE factor: 1 if n>=20, sqrt((n-1)/(n-3)) if n>3, else 2."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.sqrt((n - 1) / (n - 3))
    return np.where(n >= 20, 1.0, np.where(n > 3, small, 2.0))


def median_std(std: pd.Series, n_years: pd.Series) -> pd.Series:
    """Asymptotic SE of the median: 1.253 * sigma / sqrt(n) * finite-sample factor."""
    corr = finite_sample_correction(n_years)
    return 1.253 * std / np.sqrt(n_years.to_numpy(dtype=float)) * corr


def mean_std(std: pd.Series, n_years: pd.Series) -> pd.Series:
    """SE of the mean: sigma / sqrt(n) * same finite-sample factor as severity."""
    corr = finite_sample_correction(n_years)
    return std / np.sqrt(n_years.to_numpy(dtype=float)) * corr


def sen_slope_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Theil–Sen slope on actual x (calendar years); Conover intercept median(y) − slope·median(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return float("nan"), float("nan")
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return slope, float(np.median(y) - slope * np.median(x))


def mann_kendall_trend_test(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str],
    min_points: int = 4,
    x0: float | None = None,
) -> pd.DataFrame:
    """Run Mann–Kendall (+ Theil–Sen slope on calendar years, and SEs) per group on ``col``.

    The intercept is the level at year ``x0`` (default: first year in ``df``);
    the pipeline passes ``year_start`` so it always matches the config.

    Non-finite values (e.g. patch size in zero-disturbance years) are dropped
    before the test; ``disturbance_rate`` / ``patch_count`` should already be
    zero-filled over the calendar window.
    """
    results: list[dict] = []
    work = df.copy()
    y0 = float(work["year"].min()) if x0 is None else float(x0)
    work["year_norm"] = work["year"] - y0

    for group_keys, group in work.groupby(group_cols, sort=False):
        group_sorted = group.sort_values("year_norm")
        y = group_sorted[col].to_numpy(dtype=float)
        x = group_sorted["year_norm"].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(x)
        y = y[ok]
        x = x[ok]
        if len(y) < min_points:
            continue

        mk = original_test(y)  # MK test uses the order only: gaps do not matter
        # Theil–Sen on calendar years. pymannkendall's slope/intercept use the
        # position in the series, which is wrong when years are missing (patch
        # size in years without disturbance) or the series starts after y0.
        slope, intercept = sen_slope_xy(x, y)
        trend_line = intercept + slope * x
        residuals = y - trend_line
        n = len(y)
        if n > 2:
            mse = float(np.sum(residuals**2) / (n - 2))
            x_mean = float(np.mean(x))
            ssx = float(np.sum((x - x_mean) ** 2))
            if ssx > 0:
                slope_std = float(np.sqrt(mse / ssx))
                intercept_std = float(np.sqrt(mse * (1.0 / n + x_mean**2 / ssx)))
            else:
                slope_std = np.nan
                intercept_std = np.nan
        else:
            slope_std = np.nan
            intercept_std = np.nan

        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(group_cols, group_keys))
        row.update(
            {
                "trend": mk.trend,
                "pvalue": mk.p,
                "n_years": len(y),
                "slope": slope,
                "intercept": intercept,
                "slope_std": slope_std,
                "intercept_std": intercept_std,
                "z": mk.z,
                "tau": mk.Tau,
            }
        )
        results.append(row)

    return pd.DataFrame(results)


def add_years_to_disturbance_rate_1(
    trend_results: pd.DataFrame,
    n_years: int,
    base_year_start: int,
) -> pd.DataFrame:
    """Years from end of series until Theil–Sen line reaches disturbance_rate=1."""

    def _compute(row: pd.Series) -> float:
        slope, intercept = row["slope"], row["intercept"]
        if not np.isfinite(slope) or slope <= 0:
            return float("inf")
        index_target = (1.0 - intercept) / slope
        year_target = base_year_start + index_target
        return max(0.0, year_target - (base_year_start + n_years - 1))

    out = trend_results.copy()
    out["years_to_disturbance_rate_1"] = out.apply(_compute, axis=1)
    return out


def metric_stats_and_trends(
    df_agg: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    min_points: int,
    x0: float | None = None,
) -> pd.DataFrame:
    """Mann–Kendall + distribution stats for one metric.

    Every group with at least one year gets median/mean/… stats. Mann–Kendall is
    attached only when ``n_years >= min_points``; thinner groups keep medians with
    ``trend_* = "no trend"`` and null slopes (same fallback as a non-significant MK).
    """
    keys = group_cols + ["agent"]
    # skipna=True (default): size metrics ignore NaN zero-years; rate/count use zeros
    stats = (
        df_agg.groupby(keys, sort=False)[metric]
        .agg(
            median="median",
            mean="mean",
            std="std",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            n_obs="count",  # count of non-NA
        )
        .reset_index()
    )
    stats = stats.rename(
        columns={
            "median": f"median_{metric}",
            "mean": f"mean_{metric}",
            "std": f"std_{metric}",
            "q1": f"q1_{metric}",
            "q3": f"q3_{metric}",
            "n_obs": f"n_years_{metric}",
        }
    )
    stats[f"iqr_{metric}"] = stats[f"q3_{metric}"] - stats[f"q1_{metric}"]

    trends = mann_kendall_trend_test(df_agg, metric, keys, min_points=min_points, x0=x0)
    rename = {
        "trend": f"trend_{metric}",
        "pvalue": f"pvalue_{metric}",
        "n_years": f"n_years_mk_{metric}",
        "slope": f"slope_{metric}",
        "intercept": f"intercept_{metric}",
        "slope_std": f"slope_std_{metric}",
        "intercept_std": f"intercept_std_{metric}",
        "z": f"z_{metric}",
        "tau": f"tau_{metric}",
    }
    if trends.empty:
        out = stats.copy()
    else:
        trends = trends.rename(columns=rename)
        out = stats.merge(trends, on=keys, how="left")
        mk_n = f"n_years_mk_{metric}"
        if mk_n in out.columns:
            out[f"n_years_{metric}"] = out[mk_n].fillna(out[f"n_years_{metric}"])
            out = out.drop(columns=[mk_n])

    for c, fill in (
        (f"trend_{metric}", "no trend"),
        (f"slope_{metric}", np.nan),
        (f"intercept_{metric}", np.nan),
        (f"slope_std_{metric}", np.nan),
        (f"intercept_std_{metric}", np.nan),
        (f"pvalue_{metric}", np.nan),
        (f"z_{metric}", np.nan),
        (f"tau_{metric}", np.nan),
    ):
        if c not in out.columns:
            out[c] = fill
        elif c == f"trend_{metric}":
            out[c] = out[c].fillna("no trend")

    out[f"median_std_{metric}"] = median_std(
        out[f"std_{metric}"], out[f"n_years_{metric}"]
    )
    out[f"mean_std_{metric}"] = mean_std(
        out[f"std_{metric}"], out[f"n_years_{metric}"]
    )
    # n<=1: sample std undefined → SE = 0 (point estimate only)
    thin_n = out[f"n_years_{metric}"].fillna(0) <= 1
    out.loc[thin_n, f"median_std_{metric}"] = 0.0
    out.loc[thin_n, f"mean_std_{metric}"] = 0.0
    return out
