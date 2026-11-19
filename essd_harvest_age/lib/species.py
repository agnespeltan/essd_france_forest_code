"""ESPAR → TFV and age-correction lookups (CSV only; no Excel)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_espar_to_tfv(path: str | Path, *, include_mixtures: bool = True) -> pd.DataFrame:
    """Load long-form ESPAR→TFV map (pure + optional mixture rows)."""
    df = pd.read_csv(path)
    required = {"ess", "tfv", "stand_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["ess"] = pd.to_numeric(df["ess"], errors="coerce").astype("Int64")
    df["tfv"] = pd.to_numeric(df["tfv"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ess", "tfv"]).copy()
    df["ess"] = df["ess"].astype(int)
    df["tfv"] = df["tfv"].astype(int)
    df["stand_type"] = df["stand_type"].astype(str).str.lower()
    if not include_mixtures:
        df = df.loc[df["stand_type"] == "pure"].copy()
    return df


def load_espar_corr_age(path: str | Path) -> pd.DataFrame:
    """Load ESPAR → corr_age table."""
    df = pd.read_csv(path)
    if "ess" not in df.columns or "corr_age" not in df.columns:
        raise KeyError(f"{path} must have columns ess, corr_age")
    out = df[["ess", "corr_age"]].copy()
    out["ess"] = pd.to_numeric(out["ess"], errors="coerce")
    out["corr_age"] = pd.to_numeric(out["corr_age"], errors="coerce")
    out = out.dropna(subset=["ess", "corr_age"]).copy()
    out["ess"] = out["ess"].astype(int)
    return out.drop_duplicates(subset=["ess"], keep="first")


def parse_ess_from_espar(espar: pd.Series) -> pd.Series:
    """NFI ESPAR code → 2-digit Ess integer (NaN if unparseable)."""
    return pd.to_numeric(espar.astype(str).str.extract(r"(\d{2})", expand=False), errors="coerce")


def expand_trees_to_tfv(
    trees: pd.DataFrame,
    espar_map: pd.DataFrame,
    *,
    ess_col: str = "ess",
) -> pd.DataFrame:
    """Duplicate each tree once per TFV assignment (pure and/or mixture)."""
    if ess_col not in trees.columns:
        raise KeyError(f"trees missing column {ess_col!r}")
    keep_map = espar_map[["ess", "tfv", "stand_type"]].drop_duplicates()
    out = trees.merge(keep_map, left_on=ess_col, right_on="ess", how="inner", suffixes=("", "_map"))
    if "ess_map" in out.columns:
        out = out.drop(columns=["ess_map"])
    return out
