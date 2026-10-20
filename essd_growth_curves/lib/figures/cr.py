"""Chapman–Richards helpers and derived growth metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from lib.figures.constants import RS_CARBON_METRICS


def chapman_richards(x, a, b, c, d=0.0):
    """CR form used by RS fits: a*(1-exp(-b*x))**c + d (dry Mg/ha)."""
    return a * (1.0 - np.exp(-b * x)) ** c + d


def cr_nfi(x, a, b, c):
    """NFI / display CR in MgC/ha: a*(1-exp(-b*x))**c."""
    return a * (1.0 - np.exp(-b * x)) ** c


def cr_robinson(x, a, b, k):
    """Cook-Patton / Robinson et al. form (MgC/ha)."""
    return a * (1.0 - b * np.exp(-k * x)) ** (1.0 / (1.0 - 1.0 / 3.0))


def t50_from_abcd(a, b, c, d=0.0, fraction=0.5) -> float:
    a, b, c, d = map(float, (a, b, c, d))
    if not np.all(np.isfinite([a, b, c, d])) or a == 0 or b <= 0 or c <= 0:
        return float("nan")
    target = fraction * a
    scaled = (target - d) / a
    if scaled <= 0:
        return 0.0
    if scaled >= 1:
        return float("nan")
    return float(-np.log(1.0 - scaled ** (1.0 / c)) / b)


def t50_from_robinson(a, b, k, fraction=0.5) -> float:
    a, b, k = map(float, (a, b, k))
    if not np.all(np.isfinite([a, b, k])) or a <= 0 or k <= 0 or b <= 0:
        return float("nan")
    power = 1.0 / (1.0 - 1.0 / 3.0)
    inner = float(fraction) ** (1.0 / power)
    rem = 1.0 - inner
    if rem <= 0 or rem >= b:
        return float("nan")
    return float(-np.log(rem / b) / k)


def param_d(row, q: str) -> float:
    """Return d_q if present, else 0 (new pipeline has no d)."""
    col = f"d_{q}"
    if col in row.index and np.isfinite(row.get(col, np.nan)):
        return float(row[col])
    return 0.0


def rs_metrics(a, b, c, d=0.0, carbon: float = RS_CARBON_METRICS) -> dict | None:
    if not np.all(np.isfinite([a, b, c, d])):
        return None
    return {
        "agb_max": float(a) * carbon,
        "r30": float(chapman_richards(30, a, b, c, d)) * carbon / 30.0,
        "t50": t50_from_abcd(a, b, c, d),
    }


def nfi_metrics(a, b, c) -> dict | None:
    if not np.all(np.isfinite([a, b, c])):
        return None
    return {
        "agb_max": float(a),
        "r30": float(cr_nfi(30, a, b, c)) / 30.0,
        "t50": t50_from_abcd(a, b, c, 0.0),
    }


def robinson_metrics(a, b, k) -> dict | None:
    if not np.all(np.isfinite([a, b, k])):
        return None
    return {
        "agb_max": float(a),
        "r30": float(cr_robinson(30, a, b, k)) / 30.0,
        "t50": t50_from_robinson(a, b, k),
    }


def fit_cr3(ages, values, p0=None):
    """3-param CR fit; returns (a,b,c) or None."""
    ages = np.asarray(ages, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(ages) & np.isfinite(values) & (ages >= 0)
    ages, values = ages[ok], values[ok]
    if len(ages) < 3:
        return None
    vmax = float(np.nanmax(values))
    a_max = max(1.5 * vmax, 1.0)
    if p0 is None:
        p0 = (min(max(vmax * 1.1, 1.0), a_max), 0.05, 1.5)
    else:
        p0 = (float(np.clip(p0[0], 0.0, a_max)), float(p0[1]), float(p0[2]))
    try:
        popt, _ = curve_fit(
            cr_nfi,
            ages,
            values,
            p0=p0,
            bounds=([0, 1e-6, 0.01], [a_max, 1.0, 10]),
            maxfev=10000,
        )
        return popt
    except Exception:
        return None


def weighted_quantiles(values, weights, probs=(0.25, 0.75)):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return [float("nan") for _ in probs]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return [float(np.interp(p, cdf, values)) for p in probs]


def area_weighted_mean_std(vals, weights):
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(vals):
        weights = np.ones(len(vals), dtype=float)
    ok = np.isfinite(vals)
    vals, weights = vals[ok], weights[ok]
    if len(vals) == 0:
        return float("nan"), 0.0
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(vals), dtype=float)
    if len(vals) == 1:
        return float(vals[0]), 0.0
    wsum = float(weights.sum())
    mu = float(np.average(vals, weights=weights))
    w2 = float(np.sum(weights**2))
    denom = wsum - (w2 / wsum)
    if denom <= 0:
        return mu, 0.0
    var = float(np.sum(weights * (vals - mu) ** 2) / denom)
    return mu, float(np.sqrt(max(var, 0.0)))
