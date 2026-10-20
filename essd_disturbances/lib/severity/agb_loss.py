"""Conservative AGB severity (figure definition) with uncertainties."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Var(0.5*(q05+q50)) / (sigma^2/n) under normality
_AFTER_VARIANCE_FACTOR = 1.8128918299389918


def finite_sample_correction(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.sqrt((n - 1) / (n - 3))
    return np.where(n >= 20, 1.0, np.where(n > 3, small, 2.0))


def assign_conservative_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Set agb_lost / relative_agb_lost (+ errors) from before/after quantiles.

    Formula (Fig_disturbances conservative):
        agb_lost = max(0, q50_before - 0.5*(q05_after + q50_after))
    """
    out = df.copy()
    q50_b = out["q50_before"].to_numpy(dtype=float)
    q05_a = out["q05_after"].to_numpy(dtype=float)
    q50_a = out["q50_after"].to_numpy(dtype=float)
    std_b = out["std_before"].to_numpy(dtype=float)
    std_a = out["std_after"].to_numpy(dtype=float)
    n_b = out["n_before"].to_numpy(dtype=float)
    n_a = out["n_after"].to_numpy(dtype=float)

    loss_raw = q50_b - 0.5 * (q05_a + q50_a)

    ok_n = (n_b >= 2) & (n_a >= 2) & np.isfinite(n_b) & np.isfinite(n_a)
    n_b_safe = np.where(ok_n, n_b, np.nan)
    n_a_safe = np.where(ok_n, n_a, np.nan)
    c_b = finite_sample_correction(np.where(ok_n, n_b, 20.0))
    c_a = finite_sample_correction(np.where(ok_n, n_a, 20.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        var_q50_b = (np.pi / 2.0) * std_b**2 / n_b_safe * c_b**2
        var_after = _AFTER_VARIANCE_FACTOR * std_a**2 / n_a_safe * c_a**2
    se_ok = ok_n & np.isfinite(std_b) & np.isfinite(std_a) & np.isfinite(loss_raw)
    se = np.sqrt(np.maximum(var_q50_b + var_after, 0.0))
    se = np.where(se_ok, se, np.nan)

    loss = np.where(np.isfinite(loss_raw) & (loss_raw > 0), loss_raw, 0.0)
    err = np.where(loss > 0, se, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where((loss > 0) & (q50_b > 1e-10), 100.0 * loss_raw / q50_b, 0.0)
        rel = np.clip(rel, 0.0, 100.0)
        ratio = np.divide(loss_raw, q50_b, out=np.zeros_like(loss_raw), where=q50_b > 1e-10)
        inside = se**2 - var_q50_b * (2.0 * ratio - ratio**2)
        rel_err = np.where(
            (loss > 0) & (q50_b > 1e-10) & np.isfinite(se),
            (100.0 / q50_b) * np.sqrt(np.maximum(inside, 0.0)),
            0.0,
        )
    rel_err = np.clip(rel_err, 0.0, 100.0)

    out["agb_lost"] = loss
    out["agb_lost_error"] = np.where(np.isfinite(err), err, 0.0)
    out["relative_agb_lost"] = rel
    out["relative_agb_lost_error"] = rel_err
    return out


QUANTILE_COLS = ("q05", "q25", "q50", "q75", "q95", "mean", "std")


def pair_before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Inner-join before/after tables and rename quantile columns with suffixes."""
    b = before.rename(columns={c: f"{c}_before" for c in QUANTILE_COLS if c in before.columns})
    a = after.rename(columns={c: f"{c}_after" for c in QUANTILE_COLS if c in after.columns})
    if "count" in before.columns:
        b = b.rename(columns={"count": "n_before"})
    if "count" in after.columns:
        a = a.rename(columns={"count": "n_after"})
    merged = b.merge(a, on=keys, how="inner")
    if "n_before" not in merged.columns:
        merged["n_before"] = np.nan
    if "n_after" not in merged.columns:
        merged["n_after"] = np.nan
    return assign_conservative_severity(merged)
