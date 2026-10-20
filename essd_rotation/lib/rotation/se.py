"""Standard error of the mean for NFI harvest ages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def finite_sample_correction(n: np.ndarray | pd.Series) -> np.ndarray:
    """Same correction as disturbance medians, without the 1.253 median factor.

    n >= 20 → 1;  n > 3 → sqrt((n-1)/(n-3));  else → 2
    """
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.sqrt((n - 1.0) / (n - 3.0))
    return np.where(n >= 20.0, 1.0, np.where(n > 3.0, small, 2.0))


def rotation_period_se(count: np.ndarray | pd.Series, std: np.ndarray | pd.Series) -> np.ndarray:
    """SE of the mean: (std / sqrt(n)) × finite-sample correction."""
    count = np.asarray(count, dtype=float)
    std = np.asarray(std, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(np.where(count > 0, count, np.nan))
    return se * finite_sample_correction(count)


def add_rotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach rotation_period (= mean) and rotation_period_err from count/std."""
    out = df.copy()
    out["rotation_period"] = pd.to_numeric(out["mean"], errors="coerce")
    out["rotation_period_err"] = rotation_period_se(out["count"], out["std"])
    return out
