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
from lib.levels import add_2020_levels
from lib.raster.forest_area import load_or_build_forest_area
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


def _aggregate_annual(
    df: pd.DataFrame,
    group_cols: list[str],
    year_start: int,
    year_end: int | None = None,
) -> pd.DataFrame:
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
    if year_end is not None:
        agg = agg.loc[agg["year"] <= int(year_end)].copy()
    if agg.empty:
        return agg
    agg["disturbance_rate"] = agg["disturbed_area"] / agg["total_area"]
    if year_end is not None:
        agg = _complete_annual_years(agg, group_cols, year_start, int(year_end))
    return agg


def _complete_annual_years(
    agg: pd.DataFrame,
    group_cols: list[str],
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Fill missing calendar years with rate/count=0 (patch sizes stay NaN).

    Quiet years were previously omitted, which biased medians/MK toward years
    with disturbance only. Completes the grid for every group×agent that appears
    at least once in ``agg``.
    """
    if agg.empty:
        return agg

    stratum = group_cols + ["agent"]
    years = np.arange(int(year_start), int(year_end) + 1, dtype=int)
    keys = agg[stratum].drop_duplicates()
    area = (
        agg.groupby(stratum, sort=False)["total_area"]
        .first()
        .reset_index()
    )
    grid = keys.merge(pd.DataFrame({"year": years}), how="cross")
    grid = grid.merge(area, on=stratum, how="left")

    keep = stratum + [
        "year",
        "disturbed_area",
        "total_area",
        "patch_count",
        "mean_patch_size",
        "std_patch_size",
        "disturbance_rate",
    ]
    observed = agg[[c for c in keep if c in agg.columns]].copy()
    # Prefer observed total_area when present
    out = grid.merge(
        observed.drop(columns=["total_area"], errors="ignore"),
        on=stratum + ["year"],
        how="left",
    )
    out["disturbed_area"] = out["disturbed_area"].fillna(0.0)
    out["patch_count"] = out["patch_count"].fillna(0.0)
    out["disturbance_rate"] = out["disturbance_rate"].fillna(0.0)
    # mean/std patch size: leave NaN for zero years (no patches)
    return out.sort_values(stratum + ["year"], kind="mergesort").reset_index(drop=True)


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
        piece = metric_stats_and_trends(df_agg, metric, group_cols, min_points=min_points, x0=year_start)
        if piece.empty:
            continue
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=keys, how="outer")

    # Calendar years in the (zero-filled) rate series
    if "n_years_disturbance_rate" in out.columns:
        out["year_count"] = out["n_years_disturbance_rate"]
    else:
        yc = df_agg.groupby(keys)["year"].nunique().reset_index(name="year_count")
        out = out.merge(yc, on=keys, how="left")

    # Years with actual disturbance (gate for Mann–Kendall)
    if "disturbance_rate" in df_agg.columns:
        pos = (
            df_agg.assign(_pos=pd.to_numeric(df_agg["disturbance_rate"], errors="coerce") > 0)
            .groupby(keys, sort=False)["_pos"]
            .sum()
            .astype(int)
            .reset_index(name="n_years_positive")
        )
        out = out.merge(pos, on=keys, how="left")
        out["n_years_positive"] = out["n_years_positive"].fillna(0).astype(int)
    else:
        out["n_years_positive"] = 0

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
    """Drop MK slopes when too few positive-disturbance years (keep means/medians)."""
    if df.empty:
        return df
    out = df.copy()
    if "n_years_positive" in out.columns:
        thin = out["n_years_positive"].fillna(0) < min_years_ser
    elif "year_count" in out.columns:
        thin = out["year_count"].fillna(0) < min_years_ser
    else:
        return out
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
    """Every TFV×SER with forest area > 0 × {natural, harvest}.

    ``ser_total_area`` should be the mask census (not disturbance-conditioned
    patches) so never-disturbed forest is included.
    """
    area = ser_total_area.loc[ser_total_area["total_area"].fillna(0) > 0].copy()
    area["TFV"] = area["TFV"].astype(str)
    area["ser"] = pd.to_numeric(area["ser"], errors="coerce").astype(int)
    base = area
    if not all(c in base.columns for c in ("FT", "GRECO", "BG")):
        meta_cols = [c for c in ("ser", "TFV", "FT", "GRECO", "BG") if c in df.columns]
        if meta_cols:
            meta = df[meta_cols].drop_duplicates(["ser", "TFV"]).copy()
            meta["TFV"] = meta["TFV"].astype(str)
            meta["ser"] = pd.to_numeric(meta["ser"], errors="coerce").astype(int)
            base = area.merge(meta, on=["ser", "TFV"], how="left")
    if "GRECO" not in base.columns or base["GRECO"].isna().any():
        base["GRECO"] = base["GRECO"].fillna(_greco_letter(base["ser"])) if "GRECO" in base.columns else _greco_letter(base["ser"])
    if "FT" not in base.columns or base["FT"].isna().any():
        mapped = base["TFV"].map(GROUPS_FT)
        base["FT"] = base["FT"].fillna(mapped) if "FT" in base.columns else mapped
    if "BG" not in base.columns or base["BG"].isna().any():
        mapped = base["GRECO"].map(GROUPS_BG)
        base["BG"] = base["BG"].fillna(mapped) if "BG" in base.columns else mapped
    rows = []
    for agent in ("natural", "harvest"):
        block = base.copy()
        block["agent"] = agent
        rows.append(block)
    return pd.concat(rows, ignore_index=True)[
        ["TFV", "FT", "ser", "GRECO", "BG", "agent", "total_area"]
    ]


def _zero_disturbance_rows(keys: pd.DataFrame) -> pd.DataFrame:
    """Mean/median-at-0 rows for TFV×SER×agent with forest but no observed disturbance."""
    if keys.empty:
        return pd.DataFrame()
    out = keys.copy()
    out["gap_filled"] = GAP_FILLED_SER
    out["year_count"] = 0
    out["n_years_positive"] = 0
    for metric in METRICS:
        out[f"median_{metric}"] = 0.0
        out[f"mean_{metric}"] = 0.0
        out[f"std_{metric}"] = 0.0
        out[f"q1_{metric}"] = 0.0
        out[f"q3_{metric}"] = 0.0
        out[f"iqr_{metric}"] = 0.0
        out[f"median_std_{metric}"] = 0.0
        out[f"mean_std_{metric}"] = 0.0
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
    # MK needs at least this many non-NaN points; rate series is zero-filled so
    # the real gate is n_years_positive via _null_thin_trends (same threshold).
    min_points_mk = min_years_ser
    excluded = excluded_tfv_set(cfg)

    raw = pd.read_parquet(patches_path)
    df = prepare_patches(raw, excluded)
    df = df.loc[df["year"] >= year_start].copy()

    # Mask census (all forest TFV×SER), not disturbance-conditioned patches.
    mask_area = load_or_build_forest_area(cfg)
    ser_total_area = mask_area[["ser", "TFV", "total_area"]].copy()
    ser_total_area["TFV"] = ser_total_area["TFV"].astype(str)
    ser_total_area["ser"] = pd.to_numeric(ser_total_area["ser"], errors="coerce").astype(int)

    ser_cols = ["TFV", "FT", "ser", "GRECO", "BG"]
    greco_cols = ["TFV", "FT", "GRECO", "BG"]
    france_cols = ["TFV", "FT"]

    # --- SER: quiet years as rate=0; MK only kept if n_years_positive >= min_years_ser ---
    agg_ser = _aggregate_annual(df, ser_cols, year_start, year_end)
    yc = agg_ser.groupby(ser_cols + ["agent"])["year"].nunique().reset_index(name="year_count")
    agg_ser = agg_ser.merge(yc, on=ser_cols + ["agent"], how="left")
    fit_ser = _fit_metrics(agg_ser, ser_cols, year_start, year_end, min_points_mk)
    if not fit_ser.empty:
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
        agg_g = _aggregate_annual(df, greco_cols, year_start, year_end)
        fit_g = _fit_metrics(agg_g, greco_cols, year_start, year_end, min_points_mk)
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
        agg_f = _aggregate_annual(df, france_cols, year_start, year_end)
        fit_f = _fit_metrics(agg_f, france_cols, year_start, year_end, min_points_mk)
        fit_f = _null_thin_trends(fit_f, min_years_ser)
        exp_f = _expand_to_sers(
            fit_f, leftover, france_cols, GAP_FILLED_NATIONAL, ser_total_area=ser_total_area,
        )
        if not exp_f.empty:
            parts.append(exp_f)
            done_keys = pd.concat([done_keys, _ser_keys(exp_f)], ignore_index=True).drop_duplicates()
            leftover = universe.merge(done_keys, on=list(universe.columns), how="left", indicator=True)
            leftover = leftover.loc[leftover["_merge"] == "left_only"].drop(columns=["_merge"])

    # --- Last resort: local mean/median-only for any key still missing ---
    if not leftover.empty:
        df_loc = df.merge(leftover, on=list(universe.columns), how="inner")
        agg_loc = _aggregate_annual(df_loc, ser_cols, year_start, year_end)
        fit_loc = _fit_metrics(agg_loc, ser_cols, year_start, year_end, min_points_mk)
        if not fit_loc.empty:
            fit_loc = fit_loc.copy()
            for col in fit_loc.columns:
                if col.startswith(("slope", "intercept", "pvalue", "z")):
                    fit_loc[col] = np.nan
                if col.startswith("trend_") or col == "trend":
                    fit_loc[col] = "no trend"
            fit_loc["gap_filled"] = GAP_FILLED_SER
            parts.append(fit_loc)
            print(f"Trends mean-only fallback for {len(fit_loc)} thin TFV×SER×agent keys")

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

    # --- Structural zeros: mask forest with no disturbance → mean 0 (±0) ---
    forest_univ = _forest_agent_universe(df, mask_area)
    have = result[["TFV", "ser", "agent"]].drop_duplicates().copy()
    have["TFV"] = have["TFV"].astype(str)
    have["ser"] = pd.to_numeric(have["ser"], errors="coerce").astype(int)
    have["agent"] = have["agent"].astype(str)
    forest_univ = forest_univ.copy()
    forest_univ["TFV"] = forest_univ["TFV"].astype(str)
    forest_univ["ser"] = pd.to_numeric(forest_univ["ser"], errors="coerce").astype(int)
    forest_univ["agent"] = forest_univ["agent"].astype(str)
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
            f"Trends structural zeros (no disturbance, mean=0): {len(zeros)} "
            f"TFV×SER×agent keys"
        )

    # Published 2020 levels and slope types (lib/levels.py: same rule as the figures)
    result = add_2020_levels(result, year_start=year_start, min_years=min_years_ser)

    out = output_path(cfg, "trends")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return out
