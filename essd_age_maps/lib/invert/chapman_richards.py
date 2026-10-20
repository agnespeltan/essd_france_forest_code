"""Analytic Chapman–Richards age inversion from AGB."""

from __future__ import annotations

import numpy as np


def chapman_richards_vectorized(agb_values, a, b, c):
    """Vectorized age = −ln(1 − (AGB/a)^(1/c)) / b; clipped asymptote at 0.999·a."""
    with np.errstate(invalid="ignore"):
        age = np.zeros_like(agb_values)
        valid = (agb_values > 0) & (agb_values < a)
        age[~valid] = np.where(
            agb_values[~valid] <= 0,
            0,
            np.round(-np.log(1 - (0.999) ** (1 / c)) / b),
        )
        age[valid] = np.round(
            -np.log(1 - (agb_values[valid] / a) ** (1 / c)) / b
        )
        age = np.nan_to_num(age, nan=-1, posinf=-1, neginf=-1)
        return np.round(age).astype(int)
