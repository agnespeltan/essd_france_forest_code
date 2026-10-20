"""Merge EFDA + GAMI stats and apply age / baseline filters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lib.config import excluded_tfv_set
from lib.paths import stats_path
from lib.raster_stats.common import is_excluded_tfv, normalize_tfv_label


def _subtract_age_baseline(
    df: pd.DataFrame,
    *,
    factor: float = 0.85,
    baseline_age: float = 2,
    taper_years: float | None = None,
) -> pd.DataFrame:
    """
    Subtract a young-stand offset from EFDA quantile columns.

    Per column (each quantile series):

        y'(a) = y(a) − f · y(a0) · w(a)

    with ``w(a) = 1`` if ``taper_years`` is None/≤0 (legacy constant subtract),
    else ``w(a) = max(0, 1 − a / T)`` so the correction fades out by age ``T``.

    ``factor=0`` leaves raw EFDA AGB. Default: age=2, factor=0.85, no taper.
    """
    factor = float(factor)
    baseline_age = float(baseline_age)
    out = df.copy()
    if factor == 0.0:
        return out

    if taper_years is None or float(taper_years) <= 0:
        weight = np.ones(len(out), dtype=float)
    else:
        t = float(taper_years)
        weight = np.maximum(0.0, 1.0 - out["age"].to_numpy(dtype=float) / t)

    for prefix in ("biomasseM",):
        cols = [c for c in out.columns if c.startswith(f"{prefix}_") and c.endswith("th")]
        for col in cols:
            base = (
                out.loc[out["age"] == baseline_age, ["TFV", "grid", col]]
                .rename(columns={col: f"{col}_baseline"})
            )
            out = out.merge(base, on=["TFV", "grid"], how="left")
            out[col] = out[col] - out[f"{col}_baseline"].fillna(0.0) * factor * weight
            out = out.drop(columns=[f"{col}_baseline"])
    return out


def load_merged_stats(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Load EFDA + GAMI growth stats and return a single age–biomass table.

    EFDA: age = reference_year - year.
    Baseline (optional linear taper over ``baseline_taper_years`` T):

        AGB'(a) = AGB(a) − f · AGB(a0) · max(0, 1 − a/T)

    then keep ``age >= baseline_age``. Default: a0=2, f=0.85, no taper (T unset/0).

    GAMI: age from GAMI age column, keep age > min_age_gami.
    Excludes TFV 0 / 400 / forest / 504 / 506 (and any other labels in config.excluded_tfv).
    """
    efda_path = stats_path(cfg, "efda")
    gami_path = stats_path(cfg, "gami")
    ref_year = int(cfg.get("reference_year", 2020))
    min_g = float(cfg.get("min_age_gami", 35))
    excluded = excluded_tfv_set(cfg)

    baseline_age = float(cfg.get("age_baseline_age", 2))
    baseline_factor = float(cfg.get("age_baseline_factor", 0.85))
    taper_years = cfg.get("baseline_taper_years", None)
    if taper_years is not None:
        taper_years = float(taper_years)

    efda = pd.read_parquet(efda_path)
    efda["TFV"] = efda["TFV"].map(normalize_tfv_label)
    efda["age"] = ref_year - efda["year"]
    efda = efda.drop(columns=["year"], errors="ignore")
    efda = efda.drop(columns=["ser"], errors="ignore")
    efda = efda.drop_duplicates().reset_index(drop=True)

    # Baseline must see age==baseline_age rows; then keep [baseline_age, max].
    efda = _subtract_age_baseline(
        efda,
        factor=baseline_factor,
        baseline_age=baseline_age,
        taper_years=taper_years,
    )
    efda = efda.loc[efda["age"] >= baseline_age].reset_index(drop=True)

    gami = pd.read_parquet(gami_path)
    gami["TFV"] = gami["TFV"].map(normalize_tfv_label)
    if "age" not in gami.columns:
        if "year" in gami.columns:
            gami = gami.rename(columns={"year": "age"})
        else:
            raise KeyError(
                f"GAMI stats missing age column (looked for age / year): {gami_path}"
            )
    gami = gami.drop(columns=[c for c in ("year",) if c in gami.columns and c != "age"])
    gami = gami.drop(columns=["ser"], errors="ignore")
    gami = gami.loc[gami["age"] > min_g]
    gami = gami.drop_duplicates().reset_index(drop=True)

    combined = pd.concat([efda, gami], axis=0, ignore_index=True)
    mask = ~combined["TFV"].map(lambda t: is_excluded_tfv(t, excluded))
    combined = combined.loc[mask].reset_index(drop=True)
    return combined


def filter_tfv_greco(
    df: pd.DataFrame,
    target: str,
    tfv: str,
    greco: Any,
    *,
    min_count: int = 10,
) -> pd.DataFrame:
    """Select one TFV × GRECO series for fitting; empty if too sparse."""
    sub = df.loc[(df["TFV"].astype(str) == str(tfv)) & (df["grid"] == greco)].copy()
    target_cols = [c for c in sub.columns if c.startswith(target)]
    if not target_cols:
        return pd.DataFrame()
    sub = sub.dropna(subset=target_cols).sort_values("age")
    count_col = f"{target}_count"
    if count_col not in sub.columns:
        return pd.DataFrame()
    sub = sub.loc[sub[count_col] >= min_count]
    if sub.empty:
        return pd.DataFrame()

    total_age_points = int(sub["age"].nunique())
    efda_age_points = int(sub.loc[sub["age"] < 40, "age"].nunique())
    if total_age_points <= 4 or efda_age_points <= 2:
        return pd.DataFrame()

    out = sub[["age", *target_cols]].copy().reset_index(drop=True)
    out["TFV"] = str(tfv)
    out["greco"] = greco
    out["total_data_points"] = int(sub[count_col].sum())
    out["total_age_points"] = total_age_points
    out["EFDA_data_points"] = int(sub.loc[sub["age"] < 40, count_col].sum())
    out["EFDA_age_points"] = efda_age_points
    out["unique_ID_count_RS"] = float(np_unique_sum(sub.get("unique_ID_count_RS")))
    return out


def np_unique_sum(series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(np.unique(series.dropna()).sum())
