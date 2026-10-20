"""Load RS curves, NFI series, Robinson/CP2, area weights, GRECO shapefile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from lib.figures.constants import (
    GRECO_GROUPS,
    NFI_CARBON,
    NFI_SOIL_FRAC,
    RS_ENV_HI_Q,
    RS_ENV_LO_Q,
    RS_MAIN_Q,
    TFV_GROUP,
)
from lib.figures.cr import (
    fit_cr3,
    nfi_metrics,
    param_d,
    robinson_metrics,
    rs_metrics,
)

# 30 m pixel → Mha
_PIXEL_HA = 0.09
_HA_TO_MHA = 1e-6


def _figures(cfg: dict[str, Any]) -> dict[str, Any]:
    figs = cfg.get("figures")
    if not isinstance(figs, dict):
        raise KeyError("config.figures mapping is required for make_figures")
    return figs


def _require_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve figures.<key> from config; raise if missing or not a file."""
    raw = _figures(cfg).get(key)
    if not raw:
        raise KeyError(f"figures.{key} missing in config")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"figures.{key} not found: {path}")
    return path


def _optional_path(cfg: dict[str, Any], key: str) -> Path | None:
    """Resolve figures.<key> if set and present on disk; else None."""
    figs = cfg.get("figures")
    if not isinstance(figs, dict):
        return None
    raw = figs.get(key)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def load_rs_curves(
    cfg: dict[str, Any],
    *,
    include_gap_filled: bool = True,
) -> pd.DataFrame:
    """
    TFV × GRECO Chapman–Richards fits from the growth-curves pipeline.

    ``include_gap_filled=False`` keeps local fits only (``gap_filled==0``);
    default True keeps climate/group fills for SI coverage.
    """
    out = Path(cfg["output_dir"]).expanduser()
    name = (cfg.get("curves") or {}).get("agb", "growth_curves_agb_greco.csv")
    figs = cfg.get("figures") or {}
    if figs.get("curves_csv"):
        path = Path(figs["curves_csv"]).expanduser()
    else:
        path = out / name
    if not path.is_file():
        raise FileNotFoundError(f"RS curves CSV not found: {path}")
    ddf = pd.read_csv(path)
    ddf = ddf.loc[ddf["TFV"].astype(str) != "forest"].copy()
    if not include_gap_filled and "gap_filled" in ddf.columns:
        ddf = ddf.loc[ddf["gap_filled"].fillna(0).astype(int) == 0].copy()
    ddf["TFV"] = pd.to_numeric(ddf["TFV"], errors="coerce")
    ddf = ddf.dropna(subset=["TFV"])
    ddf["TFV"] = ddf["TFV"].astype(int)
    excl = {str(x) for x in cfg.get("excluded_tfv", [])}
    ddf = ddf.loc[~ddf["TFV"].astype(str).isin(excl)].copy()
    for q in (RS_MAIN_Q, RS_ENV_LO_Q, RS_ENV_HI_Q, "5", "95"):
        col = f"d_{q}"
        if col not in ddf.columns:
            ddf[col] = 0.0
    return ddf.reset_index(drop=True)


def load_area_tfv_greco(cfg: dict[str, Any]) -> pd.DataFrame:
    """GRECO × TFV area in Mha (from 30 m pixel counts × 0.09 / 1e6)."""
    path = _require_path(cfg, "area_parquet")
    area = pd.read_parquet(path).replace({"Forest": "forest"})
    area = area.rename(columns={"grid": "greco", "valid_pixel_count": "area"})
    area["TFV"] = pd.to_numeric(area["TFV"], errors="coerce")
    area = area.dropna(subset=["TFV", "greco"])
    area["TFV"] = area["TFV"].astype(int)
    area["greco"] = area["greco"].astype(str)
    pix = pd.to_numeric(area["area"], errors="coerce").fillna(0.0)
    area["area"] = pix * _PIXEL_HA * _HA_TO_MHA  # Mha
    return area[["greco", "TFV", "area"]]


def load_greco_gdf(cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    path = cfg.get("greco_shapefile")
    if not path:
        raise KeyError("greco_shapefile missing in config")
    gdf = gpd.read_file(path)
    if "GRECO" not in gdf.columns:
        raise KeyError(f"GRECO column missing in {path}")
    return gdf


def load_nfi_species_series(cfg: dict[str, Any]) -> pd.DataFrame:
    """National NFI BIOM_AR by ESS → TFV (MgC/ha), for Fig. 2 panels a–b.

    Returns an empty DataFrame if ``figures.nfi_species_csv`` is unset or missing.
    """
    path = _optional_path(cfg, "nfi_species_csv")
    if path is None:
        return pd.DataFrame(
            columns=["TFV", "age", "agbd", "agbd_errorp", "agbd_errorm", "weight"]
        )

    dict_age = {
        "0 - 10 ans ou non recensable": 5,
        "10 - 20 ans": 15,
        "20 - 30 ans": 25,
        "30 - 40 ans": 35,
        "40 - 50 ans": 45,
        "50 - 60 ans": 55,
        ">= 60 ans": 80,
    }
    dict_tfv = {
        "Chênes pédonculé, rouvre et pubescent": [1, 100],
        "Hêtre": [9, 100],
        "Châtaignier": [10, 100],
        "Charme": [100, 0],
        "Frêne": [100, 0],
        "Autres feuillus": [100, 0],
        "Pin maritime": [51, 80, 0],
        "Pin sylvestre": [52, 80, 0],
        "Sapin pectiné": [61, 90, 0],
        "Epicéa commun": [61, 90, 0],
        "Douglas": [64, 90, 0],
        "Autres résineux": [200, 90, 0],
        "Non défini": [-1],
    }
    net = pd.read_csv(path, skiprows=12, delimiter=";", skipfooter=1, engine="python")
    net = net.rename(columns={"ESSPRE - ESSD15": "ESS"})
    net = net[net["ESS"].notna()].copy()
    net["age"] = net["CAC - CACGK"].map(dict_age)
    net = net[net["age"] > 0].copy()
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    biom_err = "Moyenne hectare BIOM_AR - ½ largeur de l'IC 95 (Tonne de matière sèche/ha)"
    net = net.replace("n.s.", np.nan).dropna(subset=[biom, biom_err])

    def _to_num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(0.0)

    net[biom] = _to_num(net[biom])
    net[biom_err] = _to_num(net[biom_err])
    scale = (1.0 - NFI_SOIL_FRAC) * NFI_CARBON
    net["agbd"] = net[biom] * scale
    net["agbd_errorp"] = (net[biom] + net[biom_err]) * scale
    net["agbd_errorm"] = (net[biom] - net[biom_err]).clip(lower=0) * scale
    net = net[["age", "ESS", "agbd", "agbd_errorp", "agbd_errorm"]]
    net = net.assign(TFV=net["ESS"].map(dict_tfv)).explode("TFV")
    net["TFV"] = pd.to_numeric(net["TFV"], errors="coerce")
    net = net.dropna(subset=["TFV"])
    net["TFV"] = net["TFV"].astype(int)
    net = net.loc[net["TFV"] > 0].copy()
    net["weight"] = 1.0
    return (
        net.groupby(["TFV", "age"], observed=True)
        .agg(
            agbd=("agbd", "mean"),
            agbd_errorp=("agbd_errorp", "mean"),
            agbd_errorm=("agbd_errorm", "mean"),
            weight=("weight", "mean"),
        )
        .reset_index()
    )


def load_nfi_greco_series(cfg: dict[str, Any]) -> pd.DataFrame:
    """NFI BIOM_AR by GRECO (MgC/ha) for regional panels.

    Returns an empty DataFrame if ``figures.nfi_greco_csv`` is unset or missing.
    """
    path = _optional_path(cfg, "nfi_greco_csv")
    if path is None:
        return pd.DataFrame(columns=["GRECO", "age", "agbd", "agbd_error"])

    dict_age = {
        "0 - 10 ans ou non recensable": 5,
        "10 - 20 ans": 15,
        "20 - 30 ans": 25,
        "30 - 40 ans": 35,
        "40 - 50 ans": 45,
        "50 - 60 ans": 55,
        ">= 60 ans": 80,
    }
    df = pd.read_csv(path, sep=";", skiprows=12, skipfooter=1, engine="python", encoding="utf-8")
    df["age"] = df["CAC - CACGK"].map(dict_age)
    df = df.loc[df["age"] > 0].copy()
    df = df.rename(columns={"Code  GRECO": "GRECO"})
    df = df[~df.eq("n.s.").any(axis=1)]
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    biom_err = "Moyenne hectare BIOM_AR - ½ largeur de l'IC 95 (Tonne de matière sèche/ha)"
    for col in (biom, biom_err):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(0)
    scale = (1.0 - NFI_SOIL_FRAC) * NFI_CARBON
    df["agbd"] = df[biom] * scale
    df["agbd_error"] = df[biom_err] * scale
    return df[["GRECO", "age", "agbd", "agbd_error"]]


def _nfi_fit_table(series: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for g, sub in series.groupby(key, observed=True):
        sub = sub.sort_values("age")
        ages = sub["age"].to_numpy(float)
        mean = sub["agbd"].to_numpy(float)
        if "agbd_error" in sub.columns:
            err = sub["agbd_error"].to_numpy(float)
            lo, hi = np.clip(mean - err, 0, None), mean + err
        else:
            lo = sub["agbd_errorm"].to_numpy(float)
            hi = sub["agbd_errorp"].to_numpy(float)
        p_mean = fit_cr3(ages, mean)
        p_lo = fit_cr3(ages, lo, p0=p_mean) if p_mean is not None else fit_cr3(ages, lo)
        p_hi = fit_cr3(ages, hi, p0=p_mean) if p_mean is not None else fit_cr3(ages, hi)
        rec = {
            key: g,
            "a_mean": np.nan,
            "b_mean": np.nan,
            "c_mean": np.nan,
            "a_lo": np.nan,
            "b_lo": np.nan,
            "c_lo": np.nan,
            "a_hi": np.nan,
            "b_hi": np.nan,
            "c_hi": np.nan,
            "fit_ok_mean": p_mean is not None,
            "fit_ok_lo": p_lo is not None,
            "fit_ok_hi": p_hi is not None,
        }
        if p_mean is not None:
            rec["a_mean"], rec["b_mean"], rec["c_mean"] = map(float, p_mean)
        if p_lo is not None:
            rec["a_lo"], rec["b_lo"], rec["c_lo"] = map(float, p_lo)
        if p_hi is not None:
            rec["a_hi"], rec["b_hi"], rec["c_hi"] = map(float, p_hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def load_nfi_fits(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gc_nfi_species, gc_nfi_greco) fitted CR tables from config CSVs.

    Either file may be missing; missing tables are empty DataFrames.
    """
    sp_path = _optional_path(cfg, "nfi_species_curves")
    gre_path = _optional_path(cfg, "nfi_greco_curves")
    sp = pd.read_csv(sp_path) if sp_path is not None else pd.DataFrame()
    gre = pd.read_csv(gre_path) if gre_path is not None else pd.DataFrame()
    return sp, gre


def load_robinson(cfg: dict[str, Any]) -> pd.DataFrame:
    """Robinson/CP2 parquet; empty DataFrame if unset or missing."""
    path = _optional_path(cfg, "robinson_parquet")
    if path is None:
        return pd.DataFrame()
    cp2 = pd.read_parquet(path)
    if "greco" not in cp2.columns:
        if np.issubdtype(cp2["grid"].dtype, np.number):
            cp2["greco"] = cp2["grid"].astype(int).map(lambda x: chr(65 + x))
        else:
            cp2["greco"] = cp2["grid"].astype(str)
    return cp2


def metrics_from_rs(ddf: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for _, row in ddf.iterrows():
        a = row.get(f"a_{RS_MAIN_Q}")
        b = row.get(f"b_{RS_MAIN_Q}")
        c = row.get(f"c_{RS_MAIN_Q}")
        d = param_d(row, RS_MAIN_Q)
        m_c = rs_metrics(a, b, c, d)
        if m_c is None:
            continue
        m_lo = rs_metrics(
            row.get(f"a_{RS_ENV_LO_Q}"),
            row.get(f"b_{RS_ENV_LO_Q}"),
            row.get(f"c_{RS_ENV_LO_Q}"),
            param_d(row, RS_ENV_LO_Q),
        )
        m_hi = rs_metrics(
            row.get(f"a_{RS_ENV_HI_Q}"),
            row.get(f"b_{RS_ENV_HI_Q}"),
            row.get(f"c_{RS_ENV_HI_Q}"),
            param_d(row, RS_ENV_HI_Q),
        )
        tfv = float(row["TFV"])
        greco = row["greco"]
        rec = {
            "source": source,
            "TFV": tfv,
            "greco": greco,
            "tfv_group": TFV_GROUP.get(tfv, "Other"),
            "greco_group": GRECO_GROUPS.get(greco, np.nan),
            **m_c,
            "agb_max_lo": np.nan if m_lo is None else m_lo["agb_max"],
            "agb_max_hi": np.nan if m_hi is None else m_hi["agb_max"],
            "r30_lo": np.nan if m_lo is None else m_lo["r30"],
            "r30_hi": np.nan if m_hi is None else m_hi["r30"],
            "t50_lo": np.nan if m_lo is None else m_lo["t50"],
            "t50_hi": np.nan if m_hi is None else m_hi["t50"],
            "r2": float(row.get("R2_50", row.get("R2_75", np.nan))),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def metrics_from_nfi_species(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if not row.get("fit_ok_mean", True):
            continue
        m = nfi_metrics(row["a_mean"], row["b_mean"], row["c_mean"])
        if m is None:
            continue
        m_lo = nfi_metrics(row.get("a_lo"), row.get("b_lo"), row.get("c_lo"))
        m_hi = nfi_metrics(row.get("a_hi"), row.get("b_hi"), row.get("c_hi"))
        tfv = float(row["TFV"])
        rec = {
            "source": "NFI species",
            "TFV": tfv,
            "greco": np.nan,
            "tfv_group": TFV_GROUP.get(tfv, "Other"),
            "greco_group": "National",
            **m,
            "agb_max_lo": np.nan if m_lo is None else m_lo["agb_max"],
            "agb_max_hi": np.nan if m_hi is None else m_hi["agb_max"],
            "r30_lo": np.nan if m_lo is None else m_lo["r30"],
            "r30_hi": np.nan if m_hi is None else m_hi["r30"],
            "t50_lo": np.nan if m_lo is None else m_lo["t50"],
            "t50_hi": np.nan if m_hi is None else m_hi["t50"],
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def metrics_from_nfi_greco(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if not row.get("fit_ok_mean", True):
            continue
        m = nfi_metrics(row["a_mean"], row["b_mean"], row["c_mean"])
        if m is None:
            continue
        m_lo = nfi_metrics(row.get("a_lo"), row.get("b_lo"), row.get("c_lo"))
        m_hi = nfi_metrics(row.get("a_hi"), row.get("b_hi"), row.get("c_hi"))
        greco = row["greco"]
        rec = {
            "source": "NFI GRECO",
            "TFV": np.nan,
            "greco": greco,
            "tfv_group": "Forest",
            "greco_group": GRECO_GROUPS.get(greco, np.nan),
            **m,
            "agb_max_lo": np.nan if m_lo is None else m_lo["agb_max"],
            "agb_max_hi": np.nan if m_hi is None else m_hi["agb_max"],
            "r30_lo": np.nan if m_lo is None else m_lo["r30"],
            "r30_hi": np.nan if m_hi is None else m_hi["r30"],
            "t50_lo": np.nan if m_lo is None else m_lo["t50"],
            "t50_hi": np.nan if m_hi is None else m_hi["t50"],
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def metrics_from_robinson(cp2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    forest = cp2.loc[cp2["TFV"].astype(str) == "forest"]
    for _, row in forest.iterrows():
        m50 = robinson_metrics(row.get("a_50th"), row.get("b_50th"), row.get("k_50th"))
        if m50 is None:
            continue
        m05 = robinson_metrics(row.get("a_05th"), row.get("b_05th"), row.get("k_05th"))
        m95 = robinson_metrics(row.get("a_95th"), row.get("b_95th"), row.get("k_95th"))
        greco = row["greco"]
        rec = {
            "source": "Robinson",
            "TFV": np.nan,
            "greco": greco,
            "tfv_group": "Forest",
            "greco_group": GRECO_GROUPS.get(greco, np.nan),
            **m50,
            "agb_max_lo": np.nan if m05 is None else m05["agb_max"],
            "agb_max_hi": np.nan if m95 is None else m95["agb_max"],
            "r30_lo": np.nan if m05 is None else m05["r30"],
            "r30_hi": np.nan if m95 is None else m95["r30"],
            "t50_lo": np.nan if m05 is None else m05["t50"],
            "t50_hi": np.nan if m95 is None else m95["t50"],
        }
        rows.append(rec)
    return pd.DataFrame(rows)
