"""Build stratified NFI harvest-age tables (mean / median / std / IQR / count)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.config import output_path
from lib.species import (
    expand_trees_to_tfv,
    load_espar_corr_age,
    load_espar_to_tfv,
    parse_ess_from_espar,
)

_PLACETTE_COLS = [
    "CAMPAGNE",
    "IDP",
    "XL",
    "YL",
    "DEP",
    "SER",
    "DIST",
    "SVER",
    "TCAT10",
    "CAM",
    "UTIP",
    "PLISI",
    "TFORM",
    "TM2",
    "SFO",
    "PRELEV5",
    "DCESPAR1",
    "DCESPAR2",
    "VISITE",
]
_ARBRE_COLS = [
    "CAMPAGNE",
    "IDP",
    "A",
    "ESPAR",
    "ORI",
    "CIBLE",
    "V",
    "AGE13",
    "HTOT",
    "W",
    "SIMPLIF",
    "C13",
    "VEGET",
    "VEGET5",
]


def convert_ser(val: object) -> int | float:
    """IGN SER code ``A11`` → numeric ``111``."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    letter = s[0].upper()
    number = s[1:]
    return (ord(letter) - ord("A") + 1) * 100 + int(number)


def _iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


_AGE_AGG = {
    "count": "count",
    "mean": "mean",
    "median": "median",
    "std": "std",
    "iqr": _iqr,
}


def _veget_to_harvest_flag(md: object) -> float:
    """Map NFI VEGET / VEGET5 codes to living (0) / harvested (1)."""
    if pd.isna(md):
        return np.nan
    s = str(md)
    if s in {"6", "7"}:
        return 1.0
    if s == "N":
        return np.nan
    return 0.0


def mark_harvested_trees(df: pd.DataFrame) -> pd.DataFrame:
    """Flag trees harvested between successive NFI visits (``coupe == 1``).

    Uses first-visit rows only, matched to the second visit on ``(IDP, A)``.
    Harvested if status goes living→harvested or stays harvested across visits.
    """
    work = df.copy()
    work["etat1"] = work["VEGET"].map(_veget_to_harvest_flag)
    work["etat2"] = work["VEGET5"].map(_veget_to_harvest_flag)
    work["rank"] = work.groupby(["IDP", "A"])["CAMPAGNE"].rank(method="first")

    first = work.loc[work["rank"] == 1].copy()
    second = work.loc[work["rank"] == 2].copy()
    merged = first[["IDP", "A", "etat1"]].merge(
        second[["IDP", "A", "etat2"]], on=["IDP", "A"], how="inner"
    )
    e1 = merged["etat1"]
    e2 = merged["etat2"]
    merged["coupe"] = np.where(((e1 == 0) & (e2 == 1)) | ((e1 == 1) & (e2 == 1)), 1.0, np.nan)
    merged.loc[e1.isna() | e2.isna(), "coupe"] = np.nan

    out = first.drop(columns=["rank", "etat1", "etat2"], errors="ignore").merge(
        merged[["IDP", "A", "coupe"]], on=["IDP", "A"], how="left"
    )
    return out


def _load_nfi_trees(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load NFI trees and keep first-visit rows harvested between visits.

    Important: ``coupe`` must be computed **before** dropping missing ESPAR.
    Visit-2 harvested records often have ``ESPAR`` / ``VEGET`` empty and only
    ``VEGET5`` filled; filtering ESPAR first breaks the visit pairing.
    """
    nfi = cfg.get("nfi", {})
    placette_path = Path(nfi["placette"]).expanduser()
    arbre_path = Path(nfi["arbre"]).expanduser()
    sep = nfi.get("csv_sep", ";")

    placette = pd.read_csv(placette_path, sep=sep, usecols=lambda c: c in _PLACETTE_COLS)
    arbre = pd.read_csv(arbre_path, sep=sep, usecols=lambda c: c in _ARBRE_COLS)
    trees = arbre.merge(placette, on=["IDP", "CAMPAGNE"], how="inner")

    marked = mark_harvested_trees(trees)
    n_coupe = int((marked["coupe"] == 1).sum())
    cut = marked.loc[marked["coupe"] == 1].copy()
    # Species / age come from the first visit of the harvested tree
    cut = cut.loc[~cut["ESPAR"].isna()].copy()
    cut["ess"] = parse_ess_from_espar(cut["ESPAR"])
    cut = cut.dropna(subset=["ess"]).copy()
    cut["ess"] = cut["ess"].astype(int)
    cut["ser"] = cut["SER"].map(convert_ser)
    cut["GRECO"] = cut["SER"].astype(str).str[0].str.upper()
    print(f"  harvested trees (coupe==1): {n_coupe:,} ; with ESPAR/ess: {len(cut):,}")
    return cut


def _apply_ages(trees: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    species = cfg.get("species", {})
    corr = load_espar_corr_age(species["espar_corr_age"])
    espar_map = load_espar_to_tfv(
        species["espar_to_tfv"],
        include_mixtures=bool(cfg.get("include_mixtures", True)),
    )

    # Already restricted to coupe==1; require cored age on the first visit
    work = trees.loc[
        trees["AGE13"].notna()
        & (trees["CIBLE"] != 0)
        & (pd.to_numeric(trees["AGE13"], errors="coerce") < float(cfg["max_age13"]))
    ].copy()
    print(f"  harvested with AGE13 & CIBLE!=0: {len(work):,}")
    work["AGE13"] = pd.to_numeric(work["AGE13"], errors="coerce")
    work = work.merge(corr, on="ess", how="left")
    work["age"] = work["AGE13"] + work["corr_age"] + float(cfg["age_offset"])
    work = work.dropna(subset=["age", "ser"]).copy()
    work["ser"] = work["ser"].astype(int)

    expanded = expand_trees_to_tfv(work, espar_map, ess_col="ess")
    expanded = expanded.rename(columns={"tfv": "ESSENCE"})
    expanded["ESSENCE"] = expanded["ESSENCE"].astype(int)
    return expanded


def build_harvest_ages(cfg: dict[str, Any]) -> dict[str, Path]:
    """Aggregate corrected *harvested*-tree ages by TFV × (national / GRECO / SER)."""
    out_dir = Path(cfg["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = _load_nfi_trees(cfg)
    aged = _apply_ages(trees, cfg)

    national = {
        "count": int(aged["age"].count()),
        "mean": float(aged["age"].mean()),
        "median": float(aged["age"].median()),
        "std": float(aged["age"].std(ddof=1)) if aged["age"].count() > 1 else np.nan,
        "iqr": _iqr(aged["age"]),
    }
    tables = {
        "ages_tfv": aged.groupby("ESSENCE", sort=True)["age"].agg(**_AGE_AGG).reset_index(),
        "ages_greco_tfv": (
            aged.groupby(["ESSENCE", "GRECO"], sort=True)["age"].agg(**_AGE_AGG).reset_index()
        ),
        "ages_ser_tfv": (
            aged.groupby(["ESSENCE", "ser"], sort=True)["age"].agg(**_AGE_AGG).reset_index()
        ),
        "ages_ser_forest": aged.groupby("ser", sort=True)["age"].agg(**_AGE_AGG).reset_index(),
        "ages_greco": aged.groupby("GRECO", sort=True)["age"].agg(**_AGE_AGG).reset_index(),
        "ages_national": pd.DataFrame([national]),
    }

    written: dict[str, Path] = {}
    for key, df in tables.items():
        path = output_path(cfg, key)
        df.to_csv(path, index=False)
        written[key] = path
        print(f"  {key}: {path} ({len(df)} rows)")
    return written
