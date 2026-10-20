"""Fit disturbance trends with SER → GRECO → France cascade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.config import excluded_tfv_set, output_path
from lib.constants import (
    AGENT_MAP_NUMERIC,
    GAP_FILLED_GRECO,
    GAP_FILLED_NATIONAL,
    GAP_FILLED_SER,
    GROUPS_BG,
    GROUPS_FT,
    METRICS,
    PIXEL_AREA_MHA,
)
from lib.trends.mann_kendall import add_years_to_disturbance_rate_1, metric_stats_and_trends


def _greco_letter(ser: pd.Series) -> pd.Series:
    g = (ser.astype(int) // 100).clip(lower=0)
    return g.map(lambda x: chr(64 + x) if 1 <= x <= 26 else "Z")


def prepare_patches(df: pd.DataFrame, excluded: frozenset[str]) -> pd.DataFrame:
    """Normalize patches table: agent, GRECO, FT, BG, areas; drop excluded TFVs."""
    out = df.drop_duplicates().copy()
    if "grid" in out.columns and "ser" not in out.columns:
        out = out.rename(columns={"grid": "ser"})
    out["TFV"] = out["TFV"].astype(str)
    out = out.loc[~out["TFV"].str.lower().isin(excluded)].copy()

    if "agent" not in out.columns:
        ac = out["agent_class"]
        if pd.api.types.is_numeric_dtype(ac):
            out["agent"] = ac.astype(int).map(AGENT_MAP_NUMERIC)
        else:
            # Parquet often stores codes as strings "1"/"2"/"3"
            s = ac.astype(str).str.lower().str.strip()
            as_num = pd.to_numeric(s, errors="coerce")
            mapped = as_num.map(AGENT_MAP_NUMERIC)
            out["agent"] = mapped.fillna(
                s.replace({"unplanned": "natural", "harvest": "harvest", "natural": "natural"})
            )

    out["GRECO"] = _greco_letter(out["ser"])
    out["FT"] = out["TFV"].map(GROUPS_FT)
    out["BG"] = out["GRECO"].map(GROUPS_BG)
    out["area"] = out["pixel_count"].astype(float) * PIXEL_AREA_MHA
    out["total_area"] = out["valid_pixel_count"].astype(float) * PIXEL_AREA_MHA
    return out


def _aggregate_annual(df: pd.DataFrame, group_cols: list[str], year_start: int) -> pd.DataFrame:
    keys = group_cols + ["year", "agent"]
    agg = (
        df.groupby(keys, sort=False)
        .agg(
            disturbed_area=("area", "sum"),
            total_area=("total_area", "sum"),
            patch_count=("patch_count", "sum"),
            mean_patch_size=("mean_patch_size", "mean"),
            std_patch_size=("std_patch_size", "mean"),
        )
        .reset_index()
    )
    agg = agg.loc[agg["year"] >= year_start].copy()
    if agg.empty:
        return agg
    agg["disturbance_rate"] = agg["disturbed_area"] / agg["total_area"]
    return agg


def _fit_metrics(
    df_agg: pd.DataFrame,
    group_cols: list[str],
    year_start: int,
    year_end: int,
    min_points: int,
) -> pd.DataFrame:
    if df_agg.empty:
        return pd.DataFrame()

    keys = group_cols + ["agent"]
    pieces: list[pd.DataFrame] = []
    for metric in METRICS:
        piece = metric_stats_and_trends(df_agg, metric, group_cols, min_points=min_points)
        if piece.empty:
            continue
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=keys, how="outer")

    # Prefer disturbance_rate n_years as canonical year_count
    if "n_years_disturbance_rate" in out.columns:
        out["year_count"] = out["n_years_disturbance_rate"]
    else:
        yc = df_agg.groupby(keys)["year"].nunique().reset_index(name="year_count")
        out = out.merge(yc, on=keys, how="left")

    # One forest-area value per group key (do not sum across years/agents).
    total_area = (
        df_agg.groupby(group_cols, sort=False)["total_area"]
        .first()
        .reset_index()
    )
    out = out.merge(total_area, on=group_cols, how="left")

    if "slope_disturbance_rate" in out.columns:
        tmp = out.rename(
            columns={
                "slope_disturbance_rate": "slope",
                "intercept_disturbance_rate": "intercept",
            }
        )
        n_span = int(year_end - year_start + 1)
        tmp = add_years_to_disturbance_rate_1(tmp, n_years=n_span, base_year_start=year_start)
        out["years_to_disturbance_rate_1"] = tmp["years_to_disturbance_rate_1"]

    return out


def _null_thin_trends(df: pd.DataFrame, min_years_ser: int) -> pd.DataFrame:
    """For gap-filled rows with year_count < min_years_ser: keep medians, drop trend slope."""
    if df.empty or "year_count" not in df.columns:
        return df
    out = df.copy()
    thin = out["year_count"].fillna(0) < min_years_ser
    for col in out.columns:
        if col.startswith(("slope", "intercept", "pvalue", "z")):
            out.loc[thin, col] = np.nan
        if col.startswith("trend_") or col == "trend":
            out.loc[thin, col] = "no trend"
    return out


def _expand_to_sers(
    fitted: pd.DataFrame,
    leftover: pd.DataFrame,
    level_cols: list[str],
    gap_filled: int,
    ser_total_area: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Copy level fits onto each leftover SER that belongs to that level key.

    Trend / median statistics are taken from the GRECO or national fit, but
    ``total_area`` is always the SER×TFV forest area (never the donor level's
    summed area).
    """
    if fitted.empty or leftover.empty:
        return pd.DataFrame()

    merge_keys = level_cols + ["agent"]
    meta = leftover.drop_duplicates()
    rows: list[pd.DataFrame] = []
    for _, fit_row in fitted.iterrows():
        mask = pd.Series(True, index=meta.index)
        for c in merge_keys:
            mask &= meta[c] == fit_row[c]
        subset = meta.loc[mask]
        if subset.empty:
            continue
        block = pd.DataFrame([fit_row.to_dict()] * len(subset))
        for c in ("ser", "GRECO", "FT", "BG", "TFV"):
            if c in subset.columns:
                block[c] = subset[c].to_numpy()
        block["gap_filled"] = gap_filled
        rows.append(block)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if ser_total_area is not None and not ser_total_area.empty:
        # Drop donor-level total_area; restore SER×TFV mask area.
        out = out.drop(columns=["total_area"], errors="ignore")
        out = out.merge(ser_total_area, on=["ser", "TFV"], how="left")
    return out


def _ser_keys(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["TFV", "FT", "ser", "GRECO", "BG", "agent"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates()


def _forest_agent_universe(
    df: pd.DataFrame,
    ser_total_area: pd.DataFrame,
) -> pd.DataFrame:
    """Every TFV×SER with forest area > 0 × {natural, harvest}."""
    area = ser_total_area.loc[ser_total_area["total_area"].fillna(0) > 0].copy()
    meta = df.drop_duplicates(["ser", "TFV"])[["ser", "TFV", "FT", "GRECO", "BG"]]
    base = area.merge(meta, on=["ser", "TFV"], how="left")
    # Repair meta if a TFV×SER only appears via area join edge cases
    if base["GRECO"].isna().any():
        base["GRECO"] = base["GRECO"].fillna(_greco_letter(base["ser"]))
    if base["FT"].isna().any():
        base["FT"] = base["FT"].fillna(base["TFV"].map(GROUPS_FT))
    if base["BG"].isna().any():
        base["BG"] = base["BG"].fillna(base["GRECO"].map(GROUPS_BG))
    rows = []
    for agent in ("natural", "harvest"):
        block = base.copy()
        block["agent"] = agent
        rows.append(block)
    return pd.concat(rows, ignore_index=True)[
        ["TFV", "FT", "ser", "GRECO", "BG", "agent", "total_area"]
    ]


def _zero_disturbance_rows(keys: pd.DataFrame) -> pd.DataFrame:
    """Median-at-0 rows for TFV×SER×agent with forest but no observed disturbance."""
    if keys.empty:
        return pd.DataFrame()
    out = keys.copy()
    out["gap_filled"] = GAP_FILLED_SER
    out["year_count"] = 0
    for metric in METRICS:
        out[f"median_{metric}"] = 0.0
        out[f"mean_{metric}"] = 0.0
        out[f"std_{metric}"] = 0.0
        out[f"q1_{metric}"] = 0.0
        out[f"q3_{metric}"] = 0.0
        out[f"iqr_{metric}"] = 0.0
        out[f"median_std_{metric}"] = 0.0
        out[f"n_years_{metric}"] = 0
        out[f"trend_{metric}"] = "no trend"
        out[f"slope_{metric}"] = np.nan
        out[f"intercept_{metric}"] = np.nan
        out[f"slope_std_{metric}"] = np.nan
        out[f"intercept_std_{metric}"] = np.nan
        out[f"pvalue_{metric}"] = np.nan
        out[f"z_{metric}"] = np.nan
        out[f"tau_{metric}"] = np.nan
    out["years_to_disturbance_rate_1"] = np.inf
    return out


def fit_trends(cfg: dict[str, Any]) -> Path:
    """Run SER → GRECO → France cascade and write disturbance_trends.csv."""
    patches_path = Path(cfg["patches_parquet"]).expanduser()
    if not patches_path.is_file():
        raise FileNotFoundError(f"Patches parquet not found: {patches_path}")

    year_start = int(cfg["year_start"])
    year_end = int(cfg["year_end"])
    min_years_ser = int(cfg["min_years_ser"])
    min_years_mk = int(cfg["min_years_mk"])
    excluded = excluded_tfv_set(cfg)

    raw = pd.read_parquet(patches_path)
    df = prepare_patches(raw, excluded)
    df = df.loc[df["year"] >= year_start].copy()

    # SER×TFV forest area from the mask (independent of gap-fill donor level).
    ser_total_area = (
        df.groupby(["ser", "TFV"], sort=False)["total_area"]
        .first()
        .reset_index()
    )

    ser_cols = ["TFV", "FT", "ser", "GRECO", "BG"]
    greco_cols = ["TFV", "FT", "GRECO", "BG"]
    france_cols = ["TFV", "FT"]

    # --- SER: all keys with ≥1 year (MK when enough years; else median-only) ---
    agg_ser = _aggregate_annual(df, ser_cols, year_start)
    yc = agg_ser.groupby(ser_cols + ["agent"])["year"].nunique().reset_index(name="year_count")
    agg_ser = agg_ser.merge(yc, on=ser_cols + ["agent"], how="left")
    fit_ser = _fit_metrics(agg_ser, ser_cols, year_start, year_end, min_years_mk)
    if not fit_ser.empty:
        # Short local series: keep medians, drop unreliable MK slopes (same as before).
        if "year_count" not in fit_ser.columns and "n_years_disturbance_rate" in fit_ser.columns:
            fit_ser = fit_ser.copy()
            fit_ser["year_count"] = fit_ser["n_years_disturbance_rate"]
        fit_ser = _null_thin_trends(fit_ser, min_years_ser)
        fit_ser["gap_filled"] = GAP_FILLED_SER
        done_keys = _ser_keys(fit_ser)
    else:
        done_keys = pd.DataFrame(columns=["TFV", "FT", "ser", "GRECO", "BG", "agent"])

    universe = _ser_keys(df)
    leftover = universe.merge(done_keys, on=list(universe.columns), how="left", indicator=True)
    leftover = leftover.loc[leftover["_merge"] == "left_only"].drop(columns=["_merge"])

    parts = [fit_ser] if not fit_ser.empty else []

    # --- GRECO: fit on ALL SERs at GRECO×TFV×agent, assign to leftovers only ---
    if not leftover.empty:
        agg_g = _aggregate_annual(df, greco_cols, year_start)
        fit_g = _fit_metrics(agg_g, greco_cols, year_start, year_end, min_years_mk)
        fit_g = _null_thin_trends(fit_g, min_years_ser)
        exp_g = _expand_to_sers(
            fit_g, leftover, greco_cols, GAP_FILLED_GRECO, ser_total_area=ser_total_area,
        )
        if not exp_g.empty:
            parts.append(exp_g)
            done_keys = pd.concat([done_keys, _ser_keys(exp_g)], ignore_index=True).drop_duplicates()
            leftover = universe.merge(done_keys, on=list(universe.columns), how="left", indicator=True)
            leftover = leftover.loc[leftover["_merge"] == "left_only"].drop(columns=["_merge"])

    # --- France: fit on ALL SERs nationally, assign to leftovers only ---
    if not leftover.empty:
        agg_f = _aggregate_annual(df, france_cols, year_start)
        fit_f = _fit_metrics(agg_f, france_cols, year_start, year_end, min_years_mk)
        fit_f = _null_thin_trends(fit_f, min_years_ser)
        exp_f = _expand_to_sers(
            fit_f, leftover, france_cols, GAP_FILLED_NATIONAL, ser_total_area=ser_total_area,
        )
        if not exp_f.empty:
            parts.append(exp_f)
            done_keys = pd.concat([done_keys, _ser_keys(exp_f)], ignore_index=True).drop_duplicates()
            leftover = universe.merge(done_keys, on=list(universe.columns), how="left", indicator=True)
            leftover = leftover.loc[leftover["_merge"] == "left_only"].drop(columns=["_merge"])

    # --- Last resort: local median-only for any key still missing ---
    if not leftover.empty:
        df_loc = df.merge(leftover, on=list(universe.columns), how="inner")
        agg_loc = _aggregate_annual(df_loc, ser_cols, year_start)
        fit_loc = _fit_metrics(agg_loc, ser_cols, year_start, year_end, min_years_mk)
        if not fit_loc.empty:
            fit_loc = fit_loc.copy()
            for col in fit_loc.columns:
                if col.startswith(("slope", "intercept", "pvalue", "z")):
                    fit_loc[col] = np.nan
                if col.startswith("trend_") or col == "trend":
                    fit_loc[col] = "no trend"
            fit_loc["gap_filled"] = GAP_FILLED_SER
            parts.append(fit_loc)
            print(f"Trends median-only fallback for {len(fit_loc)} thin TFV×SER×agent keys")

    if not parts:
        raise RuntimeError("No trend results produced; check inputs and exclusions.")

    # Align columns then concat
    all_cols: set[str] = set()
    for p in parts:
        all_cols.update(p.columns)
    ordered = ["TFV", "FT", "ser", "GRECO", "BG", "agent", "gap_filled"]
    ordered = [c for c in ordered if c in all_cols]
    other = sorted(c for c in all_cols if c not in ordered)
    aligned = []
    for p in parts:
        for c in all_cols:
            if c not in p.columns:
                p[c] = np.nan
        aligned.append(p[ordered + other])
    result = pd.concat(aligned, ignore_index=True)
    result["gap_filled"] = result["gap_filled"].astype(int)

    # --- Structural zeros: forest TFV×SER×agent with no disturbance → median 0 ---
    forest_univ = _forest_agent_universe(df, ser_total_area)
    have = result[["TFV", "ser", "agent"]].drop_duplicates()
    missing = forest_univ.merge(
        have, on=["TFV", "ser", "agent"], how="left", indicator=True
    )
    missing = missing.loc[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    if not missing.empty:
        zeros = _zero_disturbance_rows(missing)
        for c in result.columns:
            if c not in zeros.columns:
                zeros[c] = np.nan
        result = pd.concat([result, zeros[result.columns]], ignore_index=True)
        print(
            f"Trends structural zeros (no disturbance, median=0): {len(zeros)} "
            f"TFV×SER×agent keys"
        )

    out = output_path(cfg, "trends")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return out
