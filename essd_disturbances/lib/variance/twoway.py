"""Two-way variance partition (commonality analysis) for species vs region.

Additive OLS models with one-hot factors (no interaction: one value per cell),
equal weight per cell:

    unique_A  = R²(C+A+B) − R²(C+B)
    unique_B  = R²(C+A+B) − R²(C+A)
    shared    = R²(C+A) + R²(C+B) − R²(C) − R²(C+A+B)
    residual  = 1 − R²(C+A+B)

all divided by (1 − R²(C)), i.e. expressed as % of the variance left after the
optional covariate factor C (e.g. disturbance agent); the four parts sum to 100 %.
Without C this is the usual partition of total variance, and R²(A) equals the
one-way equal-weight η² of A.

Adjusted-R² versions (Ezekiel) correct for the number of factor levels, which
otherwise favours the factor with more levels. Type II F-tests give p-values
for the unique parts.

This file is shared verbatim between essd_growth_curves and essd_disturbances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _dummies(g: np.ndarray) -> np.ndarray:
    codes, uniques = pd.factorize(pd.Series(g).astype(str), sort=True)
    if len(uniques) < 2:
        return np.zeros((len(g), 0))
    return np.eye(len(uniques))[codes][:, 1:]  # drop the first level


class TwoWay:
    """Precomputed projections, so many y vectors (MC draws) are cheap."""

    def __init__(self, g_a, g_b, g_c=None):
        n = len(g_a)
        self.n = n
        blocks = {"A": _dummies(np.asarray(g_a)), "B": _dummies(np.asarray(g_b))}
        blocks["C"] = _dummies(np.asarray(g_c)) if g_c is not None else np.zeros((n, 0))
        self.has_c = blocks["C"].shape[1] > 0
        self.models: dict[str, tuple[np.ndarray, int]] = {}
        for name in ("A", "B", "C", "CA", "CB", "CAB"):
            parts = [np.ones((n, 1))] + [blocks[k] for k in name]
            x = np.hstack(parts)
            q, r = np.linalg.qr(x)
            rank = int(np.sum(np.abs(np.diag(r)) > 1e-10 * max(1.0, np.abs(r).max())))
            if rank < x.shape[1]:  # drop aliased columns (e.g. nested levels)
                u, s, _ = np.linalg.svd(x, full_matrices=False)
                q = u[:, s > 1e-10 * s.max()]
                rank = q.shape[1]
            self.models[name] = (q, rank)

    def _rss(self, y: np.ndarray, name: str) -> float:
        q, _ = self.models[name]
        fit = q @ (q.T @ y)
        return float(np.sum((y - fit) ** 2))

    def partition(self, y: np.ndarray) -> dict[str, float]:
        y = np.asarray(y, dtype=float)
        n = self.n
        sst = float(np.sum((y - y.mean()) ** 2))
        if sst <= 0:
            return {}
        rss = {k: self._rss(y, k) for k in self.models}
        r2 = {k: 1.0 - v / sst for k, v in rss.items()}
        r2["0"] = 0.0
        df = {k: self.models[k][1] for k in self.models}
        df["0"] = 1
        adj = {k: 1.0 - (1.0 - r2[k]) * (n - 1) / max(n - df[k], 1) for k in r2}
        adj["0"] = 0.0

        def parts(r: dict[str, float]) -> dict[str, float]:
            c = r["C"] if self.has_c else 0.0
            ca = r["CA"] if self.has_c else r["A"]
            cb = r["CB"] if self.has_c else r["B"]
            full = r["CAB"]
            denom = 1.0 - c
            return {
                "unique_species": 100.0 * (full - cb) / denom,
                "unique_region": 100.0 * (full - ca) / denom,
                "shared": 100.0 * (ca + cb - c - full) / denom,
                "residual": 100.0 * (1.0 - full) / denom,
            }

        out: dict[str, float] = {
            "oneway_species": 100.0 * r2["A"],
            "oneway_region": 100.0 * r2["B"],
            "covariate": 100.0 * r2["C"] if self.has_c else np.nan,
            "r2_full": 100.0 * r2["CAB"],
        }
        out.update(parts(r2))
        out.update({f"{k}_adj": v for k, v in parts(adj).items()})
        # partial η² and type II F-tests for the unique parts
        df_res = n - df["CAB"]
        for lab, reduced, extra in (("species", "CB", "A"), ("region", "CA", "B")):
            ss = rss[reduced] - rss["CAB"]
            out[f"partial_eta2_{lab}"] = 100.0 * ss / (ss + rss["CAB"]) if ss + rss["CAB"] > 0 else np.nan
            df_num = df["CAB"] - df[reduced]
            if df_num > 0 and df_res > 0 and rss["CAB"] > 0:
                f = (ss / df_num) / (rss["CAB"] / df_res)
                out[f"p_unique_{lab}"] = float(stats.f.sf(f, df_num, df_res))
            else:
                out[f"p_unique_{lab}"] = np.nan
        out["n_cells"] = n
        out["df_species"] = df["CAB"] - df["CB"]
        out["df_region"] = df["CAB"] - df["CA"]
        return out


def recentred_ci(point: float, draws, *, lo_bound: float | None = None,
                 hi_bound: float | None = None) -> tuple[float, float]:
    """95 % CI from MC draws of the inputs, centred on the point estimate.

    Each draw adds noise to values that already carry their measurement error, so
    the draws are shifted (η² is pulled down by the extra noise). Only their
    spread is used: [point − (median − q2.5), point + (q97.5 − median)].
    """
    vals = np.asarray([v for v in draws if np.isfinite(v)], dtype=float)
    if len(vals) < 10 or not np.isfinite(point):
        return float("nan"), float("nan")
    q_lo, med, q_hi = np.percentile(vals, [2.5, 50.0, 97.5])
    lo, hi = point - (med - q_lo), point + (q_hi - med)
    if lo_bound is not None:
        lo = max(lo, lo_bound)
    if hi_bound is not None:
        hi = min(hi, hi_bound)
    return float(lo), float(hi)


CI_KEYS = ("unique_species", "unique_region", "shared", "residual",
           "unique_species_adj", "unique_region_adj", "shared_adj")


def partition_with_ci(tw: TwoWay, y: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    """Point partition of ``y`` plus a recentred 95 % CI of each part over MC draws of y."""
    point = tw.partition(y)
    store: dict[str, list[float]] = {k: [] for k in CI_KEYS}
    for yp in draws:
        d = tw.partition(yp)
        for k in CI_KEYS:
            if k in d and np.isfinite(d[k]):
                store[k].append(d[k])
    for k, vals in store.items():
        point[f"{k}_lo"], point[f"{k}_hi"] = recentred_ci(point.get(k, np.nan), vals)
    point["n_mc"] = len(draws)
    return point
