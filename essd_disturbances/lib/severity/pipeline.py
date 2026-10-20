"""Compute AGB severity with SER → GRECO → France cascade.

After filling the AGB-stats universe, also gap-fill any TFV×SER×agent that has a
disturbance **rate** (trends CSV) but no severity yet — same GRECO → national
donors, both harvest and natural.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.config import excluded_tfv_set, output_path
from lib.constants import (
    AGENT_MAP_NUMERIC,
    AGENT_MAP_STRING,
    GAP_FILLED_GRECO,
    GAP_FILLED_NATIONAL,
    GAP_FILLED_SER,
)
from lib.severity.agb_loss import QUANTILE_COLS, pair_before_after

_KEY_COLS = ["TFV", "ser", "GRECO", "agent"]
_OUT_COLS = [
    "TFV",
    "ser",
    "GRECO",
    "agent",
    "gap_filled",
    "agb_lost",
    "agb_lost_error",
    "relative_agb_lost",
    "relative_agb_lost_error",
    "q50_before",
    "q05_after",
    "q50_after",
    "n_before",
    "n_after",
]


def _greco_letter(ser: pd.Series) -> pd.Series:
    g = (ser.astype(int) // 100).clip(lower=0)
    return g.map(lambda x: chr(64 + x) if 1 <= x <= 26 else "Z")


def _normalize_agb(df: pd.DataFrame, excluded: frozenset[str]) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    if "grid" in out.columns and "ser" not in out.columns:
        out = out.rename(columns={"grid": "ser"})
    # legacy median column
    if "q50" not in out.columns and "median" in out.columns:
        out = out.rename(columns={"median": "q50"})
    out["TFV"] = out["TFV"].astype(str)
    out = out.loc[~out["TFV"].str.lower().isin(excluded)].copy()
    ac = out["agent_class"].astype(str).str.lower()
    out["agent"] = ac.map(lambda x: AGENT_MAP_STRING.get(x, x))
    out["GRECO"] = _greco_letter(out["ser"])
    return out


def _attach_counts(df: pd.DataFrame, counts_path: Path | None) -> pd.DataFrame:
    if counts_path is None or not counts_path.is_file():
        if "count" not in df.columns:
            df = df.copy()
            df["count"] = np.nan
        return df
    counts = pd.read_parquet(counts_path).drop_duplicates()
    if "grid" in counts.columns and "ser" not in counts.columns:
        counts = counts.rename(columns={"grid": "ser"})
    counts["TFV"] = counts["TFV"].astype(str)
    ac = counts["agent_class"].astype(str).str.lower()
    counts["agent"] = ac.map(lambda x: AGENT_MAP_STRING.get(x, x))
    keep = ["TFV", "agent", "ser", "year_range", "count"]
    counts = counts[keep]
    return df.merge(counts, on=["TFV", "agent", "ser", "year_range"], how="left")


def _aggregate_level(
    df: pd.DataFrame,
    year_range: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """Mean of quantiles / std; sum of counts at a coarser level."""
    sub = df.loc[df["year_range"] == year_range].copy()
    if sub.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        data = {c: g[c].mean() for c in QUANTILE_COLS if c in g.columns}
        if "count" in g.columns:
            data["count"] = g["count"].sum(min_count=1)
        return pd.Series(data)

    return sub.groupby(group_cols, sort=False).apply(_agg, include_groups=False).reset_index()


def _severity_at_level(
    df: pd.DataFrame,
    before_range: str,
    after_range: str,
    group_cols: list[str],
) -> pd.DataFrame:
    before = _aggregate_level(df, before_range, group_cols)
    after = _aggregate_level(df, after_range, group_cols)
    if before.empty or after.empty:
        return pd.DataFrame()
    return pair_before_after(before, after, group_cols)


def _ser_universe(df: pd.DataFrame) -> pd.DataFrame:
    return df[_KEY_COLS].drop_duplicates()


def _normalize_agent_series(agent: pd.Series) -> pd.Series:
    """Map numeric / string agent codes to ``natural`` / ``harvest``."""
    out = agent.copy()
    if pd.api.types.is_numeric_dtype(out):
        return out.map(AGENT_MAP_NUMERIC)
    low = out.astype(str).str.lower()
    mapped = low.map(AGENT_MAP_STRING)
    # already publishable labels
    mapped = mapped.fillna(low.where(low.isin({"natural", "harvest"})))
    return mapped


def _load_trends_table(cfg: dict[str, Any], excluded: frozenset[str]) -> pd.DataFrame:
    """Load trends CSV with normalized TFV / agent / GRECO."""
    trends_path = output_path(cfg, "trends")
    if not trends_path.is_file():
        alt = cfg.get("trends_csv") or cfg.get("disturbance_trends_csv")
        trends_path = Path(alt).expanduser() if alt else trends_path
    if not trends_path.is_file():
        raise FileNotFoundError(
            f"Trends CSV required to extend severity onto rate keys: {trends_path}"
        )

    tr = pd.read_csv(trends_path)
    tr["TFV"] = tr["TFV"].astype(str)
    tr = tr.loc[~tr["TFV"].str.lower().isin(excluded)].copy()
    tr["agent"] = _normalize_agent_series(tr["agent"])
    tr = tr.loc[tr["agent"].isin(["natural", "harvest"])].copy()
    if "GRECO" not in tr.columns:
        tr["GRECO"] = _greco_letter(tr["ser"])
    else:
        tr["GRECO"] = tr["GRECO"].astype(str)
    tr["ser"] = pd.to_numeric(tr["ser"], errors="coerce")
    return tr


def _trends_rate_universe(cfg: dict[str, Any], excluded: frozenset[str]) -> pd.DataFrame:
    """TFV×SER×agent keys that have a disturbance rate (trends CSV)."""
    return _load_trends_table(cfg, excluded)[_KEY_COLS].drop_duplicates()


def _structural_zero_rate_keys(tr: pd.DataFrame) -> pd.DataFrame:
    """Trends keys with forest present but no observed disturbance (median rate 0)."""
    yc = pd.to_numeric(tr.get("year_count", 0), errors="coerce").fillna(0)
    med = pd.to_numeric(tr.get("median_disturbance_rate", np.nan), errors="coerce")
    # year_count==0 from structural-zero rows; also treat explicit zero median + 0 years
    mask = (yc <= 0) & (med.fillna(0) <= 0)
    return tr.loc[mask, _KEY_COLS].drop_duplicates()


def _zero_severity_rows(keys: pd.DataFrame) -> pd.DataFrame:
    """Severity = 0 (±0) where there is no disturbance to sample."""
    if keys.empty:
        return pd.DataFrame(columns=_OUT_COLS)
    out = keys.copy()
    out["gap_filled"] = GAP_FILLED_SER
    out["agb_lost"] = 0.0
    out["agb_lost_error"] = 0.0
    out["relative_agb_lost"] = 0.0
    out["relative_agb_lost_error"] = 0.0
    out["q50_before"] = np.nan
    out["q05_after"] = np.nan
    out["q50_after"] = np.nan
    out["n_before"] = 0.0
    out["n_after"] = 0.0
    return out[_OUT_COLS]


def _leftover_keys(universe: pd.DataFrame, done: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()
    if done.empty:
        return universe.copy()
    merged = universe.merge(done, on=_KEY_COLS, how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"])


def _expand_level_to_sers(
    fitted: pd.DataFrame,
    leftover: pd.DataFrame,
    level_cols: list[str],
    gap_filled: int,
) -> pd.DataFrame:
    if fitted.empty or leftover.empty:
        return pd.DataFrame()

    sev_cols = [
        "agb_lost",
        "agb_lost_error",
        "relative_agb_lost",
        "relative_agb_lost_error",
        "q50_before",
        "q05_after",
        "q50_after",
        "n_before",
        "n_after",
    ]
    sev_cols = [c for c in sev_cols if c in fitted.columns]
    keep = level_cols + ["agent"] + sev_cols
    fit = fitted[keep].copy()

    rows: list[pd.DataFrame] = []
    for _, fit_row in fit.iterrows():
        mask = pd.Series(True, index=leftover.index)
        for c in level_cols + ["agent"]:
            mask &= leftover[c] == fit_row[c]
        sers = leftover.loc[mask]
        if sers.empty:
            continue
        block = pd.DataFrame([fit_row.to_dict()] * len(sers))
        block["ser"] = sers["ser"].to_numpy()
        if "GRECO" not in level_cols:
            block["GRECO"] = sers["GRECO"].to_numpy()
        block["gap_filled"] = gap_filled
        rows.append(block)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fill_greco_national(
    df: pd.DataFrame,
    leftover: pd.DataFrame,
    before_range: str,
    after_range: str,
) -> list[pd.DataFrame]:
    """GRECO then national expand onto ``leftover`` keys; return new part frames."""
    parts: list[pd.DataFrame] = []
    if leftover.empty:
        return parts

    done = leftover.iloc[0:0].copy()
    rem = leftover

    fit_g = _severity_at_level(df, before_range, after_range, ["TFV", "GRECO", "agent"])
    exp_g = _expand_level_to_sers(fit_g, rem, ["TFV", "GRECO"], GAP_FILLED_GRECO)
    if not exp_g.empty:
        parts.append(exp_g)
        done = pd.concat([done, exp_g[_KEY_COLS]], ignore_index=True).drop_duplicates()
        rem = _leftover_keys(leftover, done)

    if not rem.empty:
        fit_f = _severity_at_level(df, before_range, after_range, ["TFV", "agent"])
        exp_f = _expand_level_to_sers(fit_f, rem, ["TFV"], GAP_FILLED_NATIONAL)
        if not exp_f.empty:
            parts.append(exp_f)

    return parts


def _align_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    aligned = []
    for p in parts:
        q = p.copy()
        for c in _OUT_COLS:
            if c not in q.columns:
                q[c] = np.nan
        aligned.append(q[_OUT_COLS])
    result = pd.concat(aligned, ignore_index=True)
    result["gap_filled"] = result["gap_filled"].astype(int)
    return result.drop_duplicates(_KEY_COLS, keep="first")


def compute_severity(cfg: dict[str, Any]) -> Path:
    """Run SER → GRECO → France severity cascade; write disturbance_severity.csv.

    1. Fill AGB-stats universe (SER local, then GRECO/national).
    2. Extend onto any trends rate key still missing (GRECO/national only).
    """
    agb_path = Path(cfg["agb_stats_parquet"]).expanduser()
    if not agb_path.is_file():
        raise FileNotFoundError(f"AGB stats parquet not found: {agb_path}")

    before_range = str(cfg["before_range"])
    after_range = str(cfg["after_range"])
    excluded = excluded_tfv_set(cfg)

    counts_raw = cfg.get("agb_counts_parquet")
    counts_path = Path(counts_raw).expanduser() if counts_raw else None

    raw = pd.read_parquet(agb_path)
    df = _normalize_agb(raw, excluded)
    df = _attach_counts(df, counts_path)

    universe = _ser_universe(df)
    parts: list[pd.DataFrame] = []

    # --- SER: both windows must exist (inner join in pair_before_after) ---
    ser_cols = ["TFV", "ser", "GRECO", "agent"]
    fit_ser = _severity_at_level(df, before_range, after_range, ser_cols)
    if not fit_ser.empty:
        fit_ser = fit_ser.copy()
        fit_ser["gap_filled"] = GAP_FILLED_SER
        parts.append(fit_ser)
        done = fit_ser[_KEY_COLS].drop_duplicates()
    else:
        done = pd.DataFrame(columns=_KEY_COLS)

    leftover = _leftover_keys(universe, done)
    parts.extend(_fill_greco_national(df, leftover, before_range, after_range))

    if not parts:
        raise RuntimeError("No severity results produced; check inputs and exclusions.")

    # --- Extend onto trends rate keys still missing severity ---
    result = _align_parts(parts)
    done = result[_KEY_COLS].drop_duplicates()
    done["ser"] = pd.to_numeric(done["ser"], errors="coerce")

    tr = _load_trends_table(cfg, excluded)
    rate_univ = tr[_KEY_COLS].drop_duplicates()
    zero_keys = _structural_zero_rate_keys(tr)

    # Non-zero rates only: borrow GRECO/national AGB (zeros handled last)
    rate_leftover = _leftover_keys(rate_univ, done)
    if not zero_keys.empty:
        rate_leftover = _leftover_keys(rate_leftover, zero_keys)

    n_rate_fill = len(rate_leftover)
    if n_rate_fill:
        extra = _fill_greco_national(df, rate_leftover, before_range, after_range)
        if extra:
            parts.extend(extra)
            result = _align_parts(parts)
        still = _leftover_keys(rate_leftover, result[_KEY_COLS].drop_duplicates())
        print(
            f"Severity extended onto trends rate keys: "
            f"candidates={n_rate_fill}, filled={n_rate_fill - len(still)}, "
            f"unfilled={len(still)}"
        )
    else:
        print("Severity already covers all non-zero trends rate keys.")

    # Structural zeros last: median rate 0 → severity 0 (overwrite any prior fill)
    if not zero_keys.empty:
        done = result[_KEY_COLS].drop_duplicates()
        done["ser"] = pd.to_numeric(done["ser"], errors="coerce")
        zero_need = _leftover_keys(zero_keys, done)
        if not zero_need.empty:
            result = pd.concat([result, _zero_severity_rows(zero_need)], ignore_index=True)
        zk = zero_keys.copy()
        zk["ser"] = pd.to_numeric(zk["ser"], errors="coerce")
        result["ser"] = pd.to_numeric(result["ser"], errors="coerce")
        result = result.merge(zk.assign(_zero=1), on=_KEY_COLS, how="left")
        zmask = result["_zero"].fillna(0) == 1
        n_zero = int(zmask.sum())
        result.loc[zmask, "gap_filled"] = GAP_FILLED_SER
        result.loc[zmask, "agb_lost"] = 0.0
        result.loc[zmask, "agb_lost_error"] = 0.0
        result.loc[zmask, "relative_agb_lost"] = 0.0
        result.loc[zmask, "relative_agb_lost_error"] = 0.0
        result.loc[zmask, "q50_before"] = np.nan
        result.loc[zmask, "q05_after"] = np.nan
        result.loc[zmask, "q50_after"] = np.nan
        result.loc[zmask, "n_before"] = 0.0
        result.loc[zmask, "n_after"] = 0.0
        result = result.drop(columns=["_zero"]).drop_duplicates(_KEY_COLS, keep="last")
        print(
            f"Severity structural zeros (no disturbance): {n_zero} "
            f"(appended={len(zero_need)})"
        )

    out = output_path(cfg, "severity")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return out

