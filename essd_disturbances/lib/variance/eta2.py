"""Equal-weight one-way η² with Monte Carlo CIs and label-permutation p-values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def eta2_arrays(y: np.ndarray, w: np.ndarray, g: np.ndarray) -> float:
    """Weighted η² = SSB / SST (use ``w=1`` for equal weight per cell)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & pd.notna(g)
    if ok.sum() < 2:
        return float("nan")
    y, w, g = y[ok], w[ok], g[ok]
    codes, uniques = pd.factorize(g, sort=False)
    if len(uniques) < 2:
        return float("nan")
    mu = float(np.average(y, weights=w))
    sst = float(np.sum(w * (y - mu) ** 2))
    if sst <= 0:
        return float("nan")
    nlev = len(uniques)
    sum_w = np.bincount(codes, weights=w, minlength=nlev).astype(float)
    sum_wy = np.bincount(codes, weights=w * y, minlength=nlev).astype(float)
    pos = sum_w > 0
    mug = np.zeros(nlev, dtype=float)
    mug[pos] = sum_wy[pos] / sum_w[pos]
    ssb = float(np.sum(sum_w[pos] * (mug[pos] - mu) ** 2))
    return float(ssb / sst)


def eta2_equal(df: pd.DataFrame, ycol: str, gcol: str) -> float:
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    w = np.ones(len(df), dtype=float)
    return eta2_arrays(y, w, df[gcol].to_numpy())


def perm_p_eta2(
    df: pd.DataFrame,
    ycol: str,
    gcol: str,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """One-sided permutation p: shuffle factor labels (greater-or-equal)."""
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    w = np.ones(len(df), dtype=float)
    g0 = df[gcol].to_numpy().copy()
    obs = eta2_arrays(y, w, g0)
    if not np.isfinite(obs):
        return float("nan")
    n_ge = n_ok = 0
    for _ in range(n_perm):
        eta = eta2_arrays(y, w, rng.permutation(g0))
        if not np.isfinite(eta):
            continue
        n_ok += 1
        if eta >= obs - 1e-15:
            n_ge += 1
    if n_ok == 0:
        return float("nan")
    return float((1 + n_ge) / (1 + n_ok))


def mc_eta2_ci_from_se(
    y: np.ndarray,
    se: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    n_mc: int,
    rng: np.random.Generator,
    clip_nonneg: bool = False,
    clip_01_100: bool = False,
) -> dict[str, tuple[float, float]]:
    """95% CI of η² (%) by perturbing ``y`` within row-level SEs (Normal), recentred on the point."""
    from lib.variance.twoway import recentred_ci
    y = np.asarray(y, dtype=float)
    se = np.maximum(np.asarray(se, dtype=float), 0.0)
    w = np.ones(len(y), dtype=float)
    store = {name: [] for name in groups}
    for _ in range(n_mc):
        yp = y + rng.normal(0.0, se)
        if clip_nonneg:
            yp = np.maximum(yp, 0.0)
        if clip_01_100:
            yp = np.clip(yp, 0.0, 100.0)
        for name, g in groups.items():
            eta = eta2_arrays(yp, w, g)
            if np.isfinite(eta):
                store[name].append(100.0 * eta)
    out: dict[str, tuple[float, float]] = {}
    for name, vals in store.items():
        point = 100.0 * eta2_arrays(np.asarray(y, float), np.ones(len(y)), groups[name])
        out[name] = recentred_ci(point, vals, lo_bound=0.0, hi_bound=100.0)
    return out


def summarize_factor_row(
    *,
    parameter: str,
    column: str,
    agent_factor: bool,
    point: dict[str, float],
    ci: dict[str, tuple[float, float]],
    pvals: dict[str, float],
    n_cells: int,
) -> dict[str, Any]:
    """One slim output row for CSV / printing."""
    row: dict[str, Any] = {
        "parameter": parameter,
        "column": column,
        "agent_factor": agent_factor,
        "n_cells": n_cells,
        "eta2_ser_pct": point.get("ser", np.nan),
        "eta2_ser_lo": ci.get("ser", (np.nan, np.nan))[0],
        "eta2_ser_hi": ci.get("ser", (np.nan, np.nan))[1],
        "p_ser": pvals.get("ser", np.nan),
        "eta2_tfv_pct": point.get("tfv", np.nan),
        "eta2_tfv_lo": ci.get("tfv", (np.nan, np.nan))[0],
        "eta2_tfv_hi": ci.get("tfv", (np.nan, np.nan))[1],
        "p_tfv": pvals.get("tfv", np.nan),
        "eta2_agent_pct": point.get("agent", np.nan) if agent_factor else np.nan,
        "eta2_agent_lo": ci.get("agent", (np.nan, np.nan))[0] if agent_factor else np.nan,
        "eta2_agent_hi": ci.get("agent", (np.nan, np.nan))[1] if agent_factor else np.nan,
        "p_agent": pvals.get("agent", np.nan) if agent_factor else np.nan,
    }
    return row
