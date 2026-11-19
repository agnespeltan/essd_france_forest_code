"""Equal-weight one-way η² for growth-curve metrics (AGBmax, R30, t50).

Matches manuscript methods: equal weight per TFV×GRECO; gap-filled excluded;
MC 95% CI: by default each value is drawn from Normal(value, bootstrap fit SE)
(``07_bootstrap_curves.py`` columns ``AGBmax_se_fit`` / ``R30_se_fit`` / ``t50_se_fit``);
``mc="envelope"`` keeps the former uniform draw within the 25th–75th percentile
curves (between-stand spread; zero width for t50 since b, c are shared).
The CI uses the spread of the MC draws centred on the point estimate
(``recentred_ci``): adding noise to already-noisy values pulls η² down.
Label-permutation p-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lib.figures.constants import RS_CARBON, RS_ENV_HI_Q, RS_ENV_LO_Q, RS_MAIN_Q, TFV_GROUP
from lib.figures.cr import chapman_richards, param_d, t50_from_abcd
from lib.figures.data import load_rs_curves
from lib.variance.twoway import recentred_ci


METRICS = ("agb_max", "r30", "t50")
# metric -> (bootstrap SE column in the curves CSV, in carbon units?)
SE_COLUMNS = {"agb_max": ("AGBmax_se_fit", True), "r30": ("R30_se_fit", True), "t50": ("t50_se_fit", False)}
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
    carbon = float(carbon if carbon is not None else RS_CARBON)
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
            col, in_carbon = SE_COLUMNS[m]
            se = float(r[col]) if col in r.index else np.nan
            rec[f"{m}_se"] = se * carbon if in_carbon else se
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
    mc: str = "bootstrap",
) -> pd.DataFrame:
    """Equal-weight one-way η² (%) for species (TFV) and region (GRECO)."""
    if n_boot is not None:
        n_mc = n_boot
    df = build_metric_table(cfg, include_gap_filled=include_gap_filled)
    if df.empty:
        raise RuntimeError("No TFV×GRECO growth curves for η² (check gap-fill filter / paths)")
    if mc not in ("bootstrap", "envelope"):
        raise ValueError(f"mc must be 'bootstrap' or 'envelope', got {mc!r}")
    if mc == "bootstrap" and not all(np.isfinite(df[f"{m}_se"]).any() for m in METRICS):
        raise RuntimeError("No bootstrap SE columns in the curves CSV: run 07_bootstrap_curves.py --merge "
                           "or use mc='envelope'")

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
        se = df[f"{metric}_se"].to_numpy(dtype=float)
        se = np.where(np.isfinite(se) & (se >= 0), se, 0.0)

        sp = 100.0 * eta2_equal(df, metric, "TFV")
        rg = 100.0 * eta2_equal(df, metric, "greco")

        sp_draws, rg_draws = [], []
        for _ in range(n_mc):
            # Uniform within [25th, 75th] metric range (centred on midpoint of envelope)
            if mc == "bootstrap":
                yp = y + rng_mc.standard_normal(len(y)) * se
            else:
                u = rng_mc.uniform(-1.0, 1.0, size=len(y))
                yp = y + u * half
            yp = np.maximum(yp, 0.0) if metric != "t50" else yp
            es = eta2_arrays(yp, w, g_tfv)
            er = eta2_arrays(yp, w, g_reg)
            if np.isfinite(es):
                sp_draws.append(100.0 * es)
            if np.isfinite(er):
                rg_draws.append(100.0 * er)

        sp_lo, sp_hi = recentred_ci(sp, sp_draws, lo_bound=0.0, hi_bound=100.0)
        rg_lo, rg_hi = recentred_ci(rg, rg_draws, lo_bound=0.0, hi_bound=100.0)

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
                "mc": mc,
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


def compute_eta2_twoway(
    cfg: dict[str, Any],
    *,
    include_gap_filled: bool = False,
    n_mc: int = 1000,
    seed: int = 0,
    mc: str = "bootstrap",
) -> pd.DataFrame:
    """Two-way partition species (TFV) vs region (GRECO): unique / shared / residual (%).

    Same cells, equal weights and MC draws as :func:`compute_eta2_table`
    (bootstrap SE by default); see ``lib/variance/twoway.py``.
    """
    from lib.variance.twoway import TwoWay, partition_with_ci

    df = build_metric_table(cfg, include_gap_filled=include_gap_filled)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        work = df.loc[np.isfinite(df[metric].to_numpy(dtype=float))]
        y = work[metric].to_numpy(dtype=float)
        tw = TwoWay(work["TFV"].to_numpy(), work["greco"].to_numpy())
        if mc == "bootstrap":
            se = np.nan_to_num(work[f"{metric}_se"].to_numpy(dtype=float), nan=0.0)
            draws = [y + rng.standard_normal(len(y)) * se for _ in range(n_mc)]
        else:
            half = 0.5 * (work[f"{metric}_hi"].to_numpy(float) - work[f"{metric}_lo"].to_numpy(float))
            half = np.nan_to_num(np.maximum(half, 0.0), nan=0.0)
            draws = [y + rng.uniform(-1.0, 1.0, len(y)) * half for _ in range(n_mc)]
        res = partition_with_ci(tw, y, draws)
        rows.append({"metric": metric, "metric_label": METRIC_LABELS[metric], "mc": mc, **res})
    return pd.DataFrame(rows)
