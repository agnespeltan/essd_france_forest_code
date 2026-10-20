"""3-parameter Chapman–Richards fitting (shared b,c; monotone a; unweighted)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.stats import linregress
from sklearn.metrics import (
    d2_absolute_error_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from lib.constants import FIT_QUANTILES


def chapman_richards(age: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Return a * (1 - exp(-b * age))**c."""
    return a * (1.0 - np.exp(-b * age)) ** c


def trim_decreasing_tail(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_threshold: float = 90.0,
    delta: float = 0.05,
    patience: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a decreasing tail only for ages above ``x_threshold``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    mask_main = x <= x_threshold
    mask_tail = x > x_threshold
    x_main, y_main = x[mask_main], y[mask_main]
    x_tail, y_tail = x[mask_tail], y[mask_tail]

    y_max = y_tail[0] if len(y_tail) else 0.0
    bad_count = 0
    cutoff = len(x_tail)
    for i in range(1, len(y_tail)):
        if y_tail[i] < y_max * (1.0 - delta):
            bad_count += 1
            if bad_count >= patience:
                cutoff = i - patience + 1
                break
        else:
            y_max = max(y_max, y_tail[i])
            bad_count = 0

    return (
        np.concatenate([x_main, x_tail[:cutoff]]),
        np.concatenate([y_main, y_tail[:cutoff]]),
    )


def _mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _mean_percentage_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) / y_true) * 100.0)


def _systematic_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.all(y_true == y_true[0]):
        return np.nan
    slope, *_ = linregress(y_true, y_pred)
    return float(slope)


def _bounds_for_target(target: str) -> tuple[list[float], list[float]]:
    if target in {"biomasseM", "biomasseP", "heightM", "heightP", "H0", "H1"}:
        return [0.0, 0.1, 0.1], [2.0, 10.0, 10.0]
    if target in {"tree_coverC", "tree_coverP"}:
        return [0.0, 0.1, 0.1], [1.0, 10.0, 10.0]
    return [0.0, 0.1, 0.1], [2.0, 10.0, 10.0]


def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y, y_pred)),
        "D2": float(d2_absolute_error_score(y, y_pred)),
        "MAE": float(mean_absolute_error(y, y_pred)),
        "RMSE": float(root_mean_squared_error(y, y_pred)),
        "MBE": _mean_bias_error(y, y_pred),
        "MPB": _mean_percentage_bias(y, y_pred),
        "slope_bias": _systematic_bias(y, y_pred),
    }


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _monotone_a_from_theta(theta_a: np.ndarray, a_lo: float, a_hi: float) -> np.ndarray:
    """Map unconstrained θ → nondecreasing a in (a_lo, a_hi)."""
    n = len(theta_a)
    a = np.empty(n, dtype=float)
    a[0] = a_lo + (a_hi - a_lo) * float(_sigmoid(theta_a[0]))
    for k in range(1, n):
        a[k] = a[k - 1] + (a_hi - a[k - 1]) * float(_sigmoid(theta_a[k])) * (1.0 - 1e-8)
    return a


def _monotone_theta_from_a(a: np.ndarray, a_lo: float, a_hi: float) -> np.ndarray:
    """Inverse warm-start map for ``_monotone_a_from_theta``."""
    a = np.maximum.accumulate(np.clip(a, a_lo + 1e-6, a_hi - 1e-6))
    theta = np.empty(len(a), dtype=float)
    theta[0] = float(_logit((a[0] - a_lo) / (a_hi - a_lo)))
    for k in range(1, len(a)):
        room = a_hi - a[k - 1]
        frac = (a[k] - a[k - 1]) / room if room > 1e-12 else 0.5
        theta[k] = float(_logit(frac))
    return theta


def fit_quantiles(
    filtered_df: pd.DataFrame,
    target: str,
    quantiles: Iterable[float] = FIT_QUANTILES,
) -> dict[float, dict[str, Any]]:
    """
    Joint CR fit: shared ``(b, c)``, monotone ``a(τ)`` on the original scale,
    equal (unweighted) residuals.

    Scaling uses one shared ``ymax`` across τ so ordered ``a_norm`` stays ordered
    after back-transform. Warm-start uses a single mid-quantile NLS fit.
    """
    taus = [float(t) for t in quantiles]
    ages_all = filtered_df["age"].to_numpy(dtype=float)
    a_lo, b_lo, c_lo = _bounds_for_target(target)[0]
    a_hi, b_hi, c_hi = _bounds_for_target(target)[1]

    series: list[dict[str, Any]] = []
    for tau in taus:
        col = f"{target}_{int(100 * tau)}th"
        y_all = filtered_df[col].to_numpy(dtype=float)
        x, y = trim_decreasing_tail(ages_all, y_all, x_threshold=90.0)
        if len(x) < 3 or not np.isfinite(y).any() or np.nanmax(y) <= 0:
            continue
        series.append(
            {
                "tau": tau,
                "x": x,
                "y": y,
                "ymax": float(np.nanmax(y)),
                "x_scaled": x / 10.0,
            }
        )

    if len(series) < 2:
        return {}

    series.sort(key=lambda s: s["tau"])
    n_tau = len(series)
    y_scale = float(max(s["ymax"] for s in series))
    for s in series:
        s["y_scaled"] = s["y"] / y_scale

    mid = series[n_tau // 2]
    try:
        p0_mid, _ = curve_fit(
            chapman_richards,
            mid["x_scaled"],
            mid["y_scaled"],
            p0=[1.0, 1.0, 3.0],
            bounds=([a_lo, b_lo, c_lo], [a_hi, b_hi, c_hi]),
            maxfev=5000,
        )
        a0_mid, b0, c0 = map(float, p0_mid)
    except Exception:  # noqa: BLE001
        a0_mid, b0, c0 = 1.0, 1.0, 3.0

    a0 = np.array([s["ymax"] / mid["ymax"] * a0_mid for s in series], dtype=float)
    a0 = np.maximum.accumulate(np.clip(a0, a_lo + 1e-6, a_hi - 1e-6))
    theta0 = _monotone_theta_from_a(a0, a_lo, a_hi)
    p0 = np.concatenate([theta0, [b0, c0]])
    lower = np.array([-np.inf] * n_tau + [b_lo, c_lo], dtype=float)
    upper = np.array([np.inf] * n_tau + [b_hi, c_hi], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        a_vec = _monotone_a_from_theta(params[:n_tau], a_lo, a_hi)
        b, c = float(params[-2]), float(params[-1])
        chunks: list[np.ndarray] = []
        for i, s in enumerate(series):
            pred = chapman_richards(s["x_scaled"], float(a_vec[i]), b, c)
            chunks.append(s["y_scaled"] - pred)
        return np.concatenate(chunks)

    sol = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=20000,
        xtol=1e-10,
        ftol=1e-10,
    )
    if not sol.success and sol.cost > residuals(p0).dot(residuals(p0)):
        raise RuntimeError(f"shared-(b,c) fit failed: {sol.message}")

    a_hat = _monotone_a_from_theta(sol.x[:n_tau], a_lo, a_hi)
    b = float(sol.x[-2]) / 10.0
    c = float(sol.x[-1])

    jac = sol.jac
    dof = max(len(sol.fun) - len(sol.x), 1)
    s2 = float(np.sum(sol.fun**2)) / dof
    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
        errs = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        errs = np.full(len(sol.x), np.nan)

    b_err = float(errs[-2]) / 10.0 if np.isfinite(errs[-2]) else np.nan
    c_err = float(errs[-1]) if np.isfinite(errs[-1]) else np.nan

    results: dict[float, dict[str, Any]] = {}
    for i, s in enumerate(series):
        a = float(a_hat[i]) * y_scale
        y_pred = chapman_richards(s["x"], a, b, c)
        results[float(s["tau"])] = {
            "params": (a, b, c),
            "errors": (np.nan, b_err, c_err),  # a is reparameterized
            **_metrics(s["y"], y_pred),
        }
    return results


def results_to_row(
    results: dict[float, dict[str, Any]],
    meta: dict[str, Any],
    quantiles: Iterable[float] = FIT_QUANTILES,
) -> pd.DataFrame:
    """Flatten fitted quantiles + metadata into a one-row DataFrame (no d columns)."""
    row: dict[str, Any] = dict(meta)
    for tau in quantiles:
        suffix = str(int(100 * float(tau)))
        values = results.get(float(tau))
        if values is None:
            for key in (
                f"a_{suffix}", f"b_{suffix}", f"c_{suffix}",
                f"a_error_{suffix}", f"b_error_{suffix}", f"c_error_{suffix}",
                f"R2_{suffix}", f"D2_{suffix}", f"MAE_{suffix}", f"RMSE_{suffix}",
                f"MBE_{suffix}", f"MPB_{suffix}", f"slope_bias_{suffix}",
                f"ymax_{suffix}", f"t50_{suffix}", f"t90_{suffix}", f"r20_{suffix}",
                f"tau_{suffix}", f"egr_{suffix}",
            ):
                row[key] = np.nan
            continue

        a, b, c = values["params"]
        ae, be, ce = values["errors"]
        row[f"a_{suffix}"] = a
        row[f"b_{suffix}"] = b
        row[f"c_{suffix}"] = c
        row[f"a_error_{suffix}"] = ae
        row[f"b_error_{suffix}"] = be
        row[f"c_error_{suffix}"] = ce
        row[f"R2_{suffix}"] = values["R2"]
        row[f"D2_{suffix}"] = values["D2"]
        row[f"MAE_{suffix}"] = values["MAE"]
        row[f"RMSE_{suffix}"] = values["RMSE"]
        row[f"MBE_{suffix}"] = values["MBE"]
        row[f"MPB_{suffix}"] = values["MPB"]
        row[f"slope_bias_{suffix}"] = values["slope_bias"]

        row[f"ymax_{suffix}"] = a
        row[f"t50_{suffix}"] = (-np.log(1.0 - 0.5 ** (1.0 / c)) / b) if a > 0 and b > 0 and c > 0 else np.nan
        row[f"t90_{suffix}"] = (-np.log(1.0 - 0.9 ** (1.0 / c)) / b) if a > 0 and b > 0 and c > 0 else np.nan
        row[f"r20_{suffix}"] = (a * (1.0 - np.exp(-20.0 * b)) ** c) / 20.0 if b > 0 else np.nan
        row[f"tau_{suffix}"] = np.log(c) / b if b > 0 and c > 0 else np.nan
        row[f"egr_{suffix}"] = a * b * c

    row["gap_filled"] = 0
    return pd.DataFrame([row])
