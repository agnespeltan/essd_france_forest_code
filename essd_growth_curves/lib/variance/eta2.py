"""Equal-weight one-way η² for growth-curve metrics (AGBmax, R30, t50).

Matches manuscript methods: equal weight per TFV×GRECO; gap-filled excluded;
MC 95% CI by perturbing each value within its 25th–75th percentile range;
label-permutation p-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lib.figures.constants import RS_CARBON_CURVES, RS_ENV_HI_Q, RS_ENV_LO_Q, RS_MAIN_Q, TFV_GROUP
from lib.figures.cr import chapman_richards, param_d, t50_from_abcd
from lib.figures.data import load_rs_curves


METRICS = ("agb_max", "r30", "t50")
METRIC_LABELS = {
    "agb_max": "AGBmax",
    "r30": "R30",
    "t50": "t50",
}


def eta2_arrays(y: np.ndarray, w: np.ndarray, g: np.ndarray) -> float:
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


def _perm_p_eta2(
    df: pd.DataFrame,
    ycol: str,
    gcol: str,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
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


def _metric_from_abcd(a, b, c, d, *, carbon: float) -> dict[str, float]:
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d) if np.isfinite(d) else 0.0
    agb = a * carbon
    t50 = float(t50_from_abcd(a, b, c, d))
    r30 = float(chapman_richards(30, a, b, c, d)) * carbon / 30.0
    return {"agb_max": agb, "t50": t50, "r30": r30}


def build_metric_table(
    cfg: dict[str, Any],
    *,
    include_gap_filled: bool = False,
    carbon: float | None = None,
) -> pd.DataFrame:
    """One row per TFV×GRECO with point metrics and 25/75 envelopes for MC."""
    carbon = float(carbon if carbon is not None else RS_CARBON_CURVES)
    ddf = load_rs_curves(cfg, include_gap_filled=include_gap_filled)
    ddf = ddf.loc[ddf["TFV"].map(lambda t: TFV_GROUP.get(float(t)) not in (None, "Other", "Forest"))].copy()

    rows = []
    for _, r in ddf.iterrows():
        mid = _metric_from_abcd(
            r[f"a_{RS_MAIN_Q}"],
            r[f"b_{RS_MAIN_Q}"],
            r[f"c_{RS_MAIN_Q}"],
            param_d(r, RS_MAIN_Q),
            carbon=carbon,
        )
        lo = _metric_from_abcd(
            r[f"a_{RS_ENV_LO_Q}"],
            r[f"b_{RS_ENV_LO_Q}"],
            r[f"c_{RS_ENV_LO_Q}"],
            param_d(r, RS_ENV_LO_Q),
            carbon=carbon,
        )
        hi = _metric_from_abcd(
            r[f"a_{RS_ENV_HI_Q}"],
            r[f"b_{RS_ENV_HI_Q}"],
            r[f"c_{RS_ENV_HI_Q}"],
            param_d(r, RS_ENV_HI_Q),
            carbon=carbon,
        )
        rec = {
            "TFV": int(r["TFV"]),
            "greco": str(r["greco"]),
        }
        for m in METRICS:
            rec[m] = mid[m]
            rec[f"{m}_lo"] = min(lo[m], hi[m])
            rec[f"{m}_hi"] = max(lo[m], hi[m])
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_eta2_table(
    cfg: dict[str, Any],
    *,
    include_gap_filled: bool = False,
    n_mc: int = 1000,
    n_perm: int = 1000,
    seed: int = 0,
    n_boot: int | None = None,  # backwards-compatible alias → n_mc
) -> pd.DataFrame:
    """Equal-weight one-way η² (%) for species (TFV) and region (GRECO)."""
    if n_boot is not None:
        n_mc = n_boot
    df = build_metric_table(cfg, include_gap_filled=include_gap_filled)
    if df.empty:
        raise RuntimeError("No TFV×GRECO growth curves for η² (check gap-fill filter / paths)")

    rng_mc = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 42)
    w = np.ones(len(df), dtype=float)
    g_tfv = df["TFV"].to_numpy()
    g_reg = df["greco"].to_numpy()

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        y = df[metric].to_numpy(dtype=float)
        lo = df[f"{metric}_lo"].to_numpy(dtype=float)
        hi = df[f"{metric}_hi"].to_numpy(dtype=float)
        # If envelope missing, fall back to point (σ=0)
        lo = np.where(np.isfinite(lo), lo, y)
        hi = np.where(np.isfinite(hi), hi, y)
        half = 0.5 * (hi - lo)
        half = np.where(np.isfinite(half) & (half >= 0), half, 0.0)

        sp = 100.0 * eta2_equal(df, metric, "TFV")
        rg = 100.0 * eta2_equal(df, metric, "greco")

        sp_draws, rg_draws = [], []
        for _ in range(n_mc):
            # Uniform within [25th, 75th] metric range (centred on midpoint of envelope)
            u = rng_mc.uniform(-1.0, 1.0, size=len(y))
            yp = y + u * half
            yp = np.maximum(yp, 0.0) if metric != "t50" else yp
            es = eta2_arrays(yp, w, g_tfv)
            er = eta2_arrays(yp, w, g_reg)
            if np.isfinite(es):
                sp_draws.append(100.0 * es)
            if np.isfinite(er):
                rg_draws.append(100.0 * er)

        sp_lo, sp_hi = (
            (float(np.percentile(sp_draws, 2.5)), float(np.percentile(sp_draws, 97.5)))
            if len(sp_draws) >= 10
            else (np.nan, np.nan)
        )
        rg_lo, rg_hi = (
            (float(np.percentile(rg_draws, 2.5)), float(np.percentile(rg_draws, 97.5)))
            if len(rg_draws) >= 10
            else (np.nan, np.nan)
        )

        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "eta2_species_pct": sp,
                "eta2_species_ci_lo": sp_lo,
                "eta2_species_ci_hi": sp_hi,
                "eta2_species_pm": 0.5 * (sp_hi - sp_lo) if np.isfinite(sp_lo) else np.nan,
                "p_species": _perm_p_eta2(df, metric, "TFV", n_perm=n_perm, rng=rng_perm),
                "eta2_region_pct": rg,
                "eta2_region_ci_lo": rg_lo,
                "eta2_region_ci_hi": rg_hi,
                "eta2_region_pm": 0.5 * (rg_hi - rg_lo) if np.isfinite(rg_lo) else np.nan,
                "p_region": _perm_p_eta2(df, metric, "greco", n_perm=n_perm, rng=rng_perm),
                "n_pairs": int(len(df)),
                "include_gap_filled": bool(include_gap_filled),
                "equal_weight": True,
            }
        )
    return pd.DataFrame(rows)


def format_eta2_paragraph(table: pd.DataFrame) -> str:
    """Results-style sentence block with η² ± half 95% CI."""
    if table.empty:
        return ""

    def _pm(row: pd.Series, kind: str) -> str:
        mu = float(row[f"eta2_{kind}_pct"])
        pm = float(row[f"eta2_{kind}_pm"])
        return f"{mu:.0f}±{pm:.0f}%"

    def _p(row: pd.Series, kind: str) -> str:
        p = float(row[f"p_{kind}"])
        if not np.isfinite(p):
            return "n/a"
        if p <= 0.001:
            return "p = 0.001"
        return f"p = {p:.3f}"

    sp_mean = float(table["eta2_species_pct"].mean())
    rg_mean = float(table["eta2_region_pct"].mean())
    sp_pm = 1.96 * float(table["eta2_species_pct"].std(ddof=1)) / np.sqrt(len(table))
    rg_pm = 1.96 * float(table["eta2_region_pct"].std(ddof=1)) / np.sqrt(len(table))

    by = {r["metric"]: r for _, r in table.iterrows()}
    return (
        f"Using one-way η², species identity explains on average "
        f"{sp_mean:.0f}±{sp_pm:.0f}% of variance in growth-curve metrics, "
        f"compared with {rg_mean:.0f}±{rg_pm:.0f}% for ecological region (95% CI). "
        f"Both factors are assessed with permutation tests: "
        f"AGBmax (species {_p(by['agb_max'], 'species')}, region {_p(by['agb_max'], 'region')}); "
        f"R30 (species {_p(by['r30'], 'species')}, region {_p(by['r30'], 'region')}); "
        f"t50 (species {_p(by['t50'], 'species')}, region {_p(by['t50'], 'region')}). "
        f"Relative weights vary by metric: AGBmax "
        f"(species η² = {_pm(by['agb_max'], 'species')} vs. region {_pm(by['agb_max'], 'region')}); "
        f"R30 ({_pm(by['r30'], 'species')} vs. {_pm(by['r30'], 'region')}); "
        f"and t50 ({_pm(by['t50'], 'species')} vs. {_pm(by['t50'], 'region')})."
    )
