"""NFI reference data and AGB-trajectory scores for the sensitivity tests (08).

NFI tables: IGN inventory results, mean BIOM_AR (t dry matter ha⁻¹) per age class
(``figures.nfi_greco_csv`` per GRECO, ``figures.nfi_species_csv`` per species).
Converted to above-ground carbon (MgC ha⁻¹) with the below-ground share removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.fit.chapman_richards import chapman_richards

CARBON = 0.475
NFI_BG_FRAC = 1.0 - 0.221

DICT_AGE = {
    "0 - 10 ans ou non recensable": 5,
    "10 - 20 ans": 15,
    "20 - 30 ans": 25,
    "30 - 40 ans": 35,
    "40 - 50 ans": 45,
    "50 - 60 ans": 55,
    ">= 60 ans": 80,
    "Absence d'arbres mesurés": -10,
}

ESS_TO_TFV = {
    "Chênes pédonculé, rouvre et pubescent": 1,
    "Hêtre": 9,
    "Châtaignier": 10,
    "Pin maritime": 51,
    "Pin sylvestre": 52,
    "Sapin pectiné": 61,
    "Epicéa commun": 61,
    "Douglas": 64,
}


def load_nfi_greco(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=12, skipfooter=1, engine="python", encoding="utf-8")
    df["age"] = df["CAC - CACGK"].map(DICT_AGE)
    df = df.loc[df["age"] > 0].copy()
    df = df.rename(columns={"Code  GRECO": "GRECO"})
    df = df[~df.eq("n.s.").any(axis=1)]
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    for col in (biom,):
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["agbd"] = df[biom] * NFI_BG_FRAC * CARBON
    df["GRECO"] = df["GRECO"].astype(str)
    return df[["GRECO", "age", "agbd"]].dropna()


def load_nfi_species(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=12, skipfooter=1, engine="python", encoding="utf-8")
    df = df.rename(columns={"ESSPRE - ESSD15": "ESS"})
    df = df.loc[df["ESS"].notna()].copy()
    df["age"] = df["CAC - CACGK"].map(DICT_AGE)
    df = df.loc[df["age"] > 0].copy()
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    df = df.replace("n.s.", np.nan).dropna(subset=[biom])
    df[biom] = df[biom].astype(str).str.replace(",", ".", regex=False)
    df[biom] = pd.to_numeric(df[biom], errors="coerce")
    df["agbd"] = df[biom] * NFI_BG_FRAC * CARBON
    df["TFV"] = df["ESS"].map(ESS_TO_TFV)
    df = df.dropna(subset=["TFV"])
    df["TFV"] = df["TFV"].astype(int)
    return df.groupby(["TFV", "age"], as_index=False)["agbd"].mean()


def load_area_weights(path: Path) -> pd.DataFrame:
    area = pd.read_parquet(path)
    area = area.replace({"Forest": "forest"})
    rename = {}
    if "grid" in area.columns:
        rename["grid"] = "greco"
    if "valid_pixel_count" in area.columns:
        rename["valid_pixel_count"] = "area"
    if "a_count" in area.columns:
        rename["a_count"] = "area"
    area = area.rename(columns=rename)
    if "greco" not in area.columns and "GRECO" in area.columns:
        area = area.rename(columns={"GRECO": "greco"})
    area["greco"] = area["greco"].astype(str)
    area["TFV"] = pd.to_numeric(area["TFV"], errors="coerce")
    area = area.dropna(subset=["TFV", "area"])
    area["TFV"] = area["TFV"].astype(int).astype(str)
    area = area.loc[~area["TFV"].isin(["0", "400"])]
    return area[["greco", "TFV", "area"]]


def rs_curve_mgc(row: pd.Series, ages: np.ndarray, q: str | int = "50") -> np.ndarray:
    q = str(int(q))
    a, b, c = float(row[f"a_{q}"]), float(row[f"b_{q}"]), float(row[f"c_{q}"])
    if not np.isfinite([a, b, c]).all() or b <= 0:
        return np.full_like(ages, np.nan, dtype=float)
    return chapman_richards(ages, a, b, c) * CARBON


def score_vs_nfi_greco(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    *,
    q: str | int = "50",
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="inner")
    if cur.empty:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    y_true, y_pred = [], []
    for greco, g_nfi in nfi.groupby("GRECO"):
        sub = cur.loc[cur["greco"] == str(greco)]
        if sub.empty or sub["area"].sum() <= 0:
            continue
        ages = g_nfi["age"].to_numpy(float)
        preds, weights = [], []
        for _, row in sub.iterrows():
            preds.append(rs_curve_mgc(row, ages, q=q))
            weights.append(float(row["area"]))
        mix = np.average(np.vstack(preds), axis=0, weights=np.asarray(weights))
        y_true.append(g_nfi["agbd"].to_numpy(float))
        y_pred.append(mix)

    if not y_true:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    err = yp - yt
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
    }


def score_vs_nfi_species(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    *,
    q: str | int = "50",
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="left")
    cur["area"] = cur["area"].fillna(1.0)

    y_true, y_pred = [], []
    for tfv, g_nfi in nfi.groupby("TFV"):
        sub = cur.loc[cur["TFV"] == str(int(tfv))]
        if sub.empty:
            continue
        ages = g_nfi["age"].to_numpy(float)
        preds, weights = [], []
        for _, row in sub.iterrows():
            preds.append(rs_curve_mgc(row, ages, q=q))
            weights.append(float(row["area"]))
        w = np.asarray(weights)
        if w.sum() <= 0:
            w = np.ones_like(w)
        mix = np.average(np.vstack(preds), axis=0, weights=w)
        y_true.append(g_nfi["agbd"].to_numpy(float))
        y_pred.append(mix)

    if not y_true:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    err = yp - yt
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
    }
