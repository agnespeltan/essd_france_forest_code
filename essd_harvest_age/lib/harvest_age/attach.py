"""Cascade-fill NFI mean harvest age onto disturbance trend rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.config import output_path, tfv_alias_pairs
from lib.constants import (
    GAP_FILLED_ALIAS_SER,
    GAP_FILLED_GRECO,
    GAP_FILLED_NATIONAL,
    GAP_FILLED_SER,
)
from lib.harvest_age.se import add_harvest_age_columns

VALUE_COLS = ("mean_harvest_age", "mean_harvest_age_err", "harvest_age_sd", "n_trees")

_AGENT_MAP = {
    "1": "natural",
    "1.0": "natural",
    "2": "natural",
    "2.0": "natural",
    "3": "harvest",
    "3.0": "harvest",
    "unplanned": "natural",
    "natural": "natural",
    "harvest": "harvest",
}


def _read_age_table(path: Path) -> pd.DataFrame:
    return add_harvest_age_columns(pd.read_csv(path))


def _prepare_tfv_table(df: pd.DataFrame, tfv_col: str = "ESSENCE") -> pd.DataFrame:
    """Numeric TFV rows only (drops Forest / label leftovers)."""
    out = df.copy()
    out["TFV"] = pd.to_numeric(out[tfv_col], errors="coerce")
    out = out.dropna(subset=["TFV"]).copy()
    out["TFV"] = out["TFV"].astype(int)
    return out


def apply_tfv_aliases(
    df: pd.DataFrame,
    aliases: list[tuple[int, int]],
    subset_cols: list[str],
) -> pd.DataFrame:
    """Copy source TFV rows onto missing target TFVs (order matters)."""
    out = df.copy()
    for target, source in aliases:
        src = out.loc[out["TFV"] == source].copy()
        if src.empty:
            continue
        src["TFV"] = target
        if "ESSENCE" in src.columns:
            src["ESSENCE"] = target
        out = pd.concat([out, src], ignore_index=True)
        out = out.drop_duplicates(subset=subset_cols, keep="last")
    return out


def _eligible(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    return df.loc[pd.to_numeric(df["count"], errors="coerce") > min_count].copy()


def _fill_level(
    base: pd.DataFrame,
    donor: pd.DataFrame,
    on: list[str],
    level: int,
) -> pd.DataFrame:
    """Left-fill the harvest-age columns and harvest_age_gap_filled from donor on keys ``on``.

    ``harvest_age_sd`` and ``n_trees`` describe the sample actually used, i.e. the
    donor's sample for gap-filled rows.
    """
    right = donor[on + list(VALUE_COLS)].rename(columns={c: f"_{c}" for c in VALUE_COLS})
    out = base.merge(right, on=on, how="left")
    miss = out["mean_harvest_age"].isna() & out["_mean_harvest_age"].notna()
    for c in VALUE_COLS:
        out.loc[miss, c] = out.loc[miss, f"_{c}"]
    out.loc[miss, "harvest_age_gap_filled"] = level
    return out.drop(columns=[f"_{c}" for c in VALUE_COLS])


def _load_age_inputs(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load age CSVs from output_dir (written by 00) or explicit path overrides."""
    overrides = cfg.get("age_tables", {})
    keys = {
        "ser_tfv": "ages_ser_tfv",
        "greco_tfv": "ages_greco_tfv",
        "tfv": "ages_tfv",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for alias, out_key in keys.items():
        path = (
            Path(overrides[alias]).expanduser()
            if alias in overrides
            else output_path(cfg, out_key)
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Age table not found: {path}\nRun 00_build_harvest_ages.py first."
            )
        loaded[alias] = _read_age_table(path)
    return loaded


def _normalize_agent(series: pd.Series) -> pd.Series:
    a = series.astype(str).str.lower().str.strip()
    return a.map(_AGENT_MAP).fillna(series)


def attach_harvest_age(cfg: dict[str, Any]) -> dict[str, Path]:
    """Attach mean harvest ages to disturbance trends; write harvest-age + joined CSVs.

    Cascade: TFV×SER → TFV×GRECO → national TFV. Alias targets filled at SER
    are coded ``harvest_age_gap_filled=3``. Trends ``gap_filled`` (rate 0/1/2) is
    preserved. Natural-agent rows get NaN harvest age / harvest_age_gap_filled.
    """
    min_count = int(cfg.get("min_count", 5))
    aliases = tfv_alias_pairs(cfg)
    ages = _load_age_inputs(cfg)

    ser_tfv = apply_tfv_aliases(
        _prepare_tfv_table(ages["ser_tfv"]), aliases, ["TFV", "ser"]
    )
    greco_tfv = apply_tfv_aliases(
        _prepare_tfv_table(ages["greco_tfv"]), aliases, ["TFV", "GRECO"]
    )
    tfv_all = apply_tfv_aliases(_prepare_tfv_table(ages["tfv"]), aliases, ["TFV"])

    trends_path = Path(cfg["trends_csv"]).expanduser()
    trends = pd.read_csv(trends_path)
    # Drop legacy Forest aggregate rows if present (not used)
    trends = trends.loc[trends["TFV"].astype(str) != "Forest"].copy()

    if "GRECO" not in trends.columns and "ser" in trends.columns:
        ser_num = pd.to_numeric(trends["ser"], errors="coerce")
        trends["GRECO"] = ser_num.map(
            lambda s: chr(int(s // 100) + ord("A") - 1)
            if pd.notna(s) and int(s) >= 100
            else np.nan
        )

    out = trends.copy()
    out["TFV"] = pd.to_numeric(out["TFV"], errors="coerce")
    out = out.dropna(subset=["TFV"]).copy()
    out["TFV"] = out["TFV"].astype(int)
    out["ser"] = pd.to_numeric(out["ser"], errors="coerce").astype(int)
    if "GRECO" in out.columns:
        out["GRECO"] = out["GRECO"].astype(str).str[0].str.upper()

    # Keep trends gap_filled (rate cascade). Only reset harvest-age columns.
    for col in VALUE_COLS + ("harvest_age_gap_filled",):
        if col in out.columns:
            out = out.drop(columns=[col])
        out[col] = np.nan

    out = _fill_level(out, _eligible(ser_tfv, min_count), ["TFV", "ser"], GAP_FILLED_SER)
    out = _fill_level(out, _eligible(greco_tfv, min_count), ["TFV", "GRECO"], GAP_FILLED_GRECO)
    out = _fill_level(out, _eligible(tfv_all, min_count), ["TFV"], GAP_FILLED_NATIONAL)

    alias_targets = {int(t) for t, _ in aliases}
    alias_ser = out["TFV"].isin(alias_targets) & out["harvest_age_gap_filled"].eq(GAP_FILLED_SER)
    out.loc[alias_ser, "harvest_age_gap_filled"] = GAP_FILLED_ALIAS_SER

    if "agent" in out.columns:
        out["agent"] = _normalize_agent(out["agent"])
        natural = out["agent"].eq("natural")
        out.loc[natural, list(VALUE_COLS)] = np.nan
        out.loc[natural, "harvest_age_gap_filled"] = np.nan

    out = out.drop_duplicates()
    Path(cfg["output_dir"]).expanduser().mkdir(parents=True, exist_ok=True)

    rot_cols = [
        "TFV",
        "ser",
        "GRECO",
        "agent",
        "total_area",
        "mean_harvest_age",
        "mean_harvest_age_err",
        "harvest_age_sd",
        "n_trees",
        "harvest_age_gap_filled",
    ]
    rot_cols = [c for c in rot_cols if c in out.columns]
    rot_path = output_path(cfg, "harvest_age_table")
    out[rot_cols].to_csv(rot_path, index=False)

    trends_out = output_path(cfg, "trends_with_harvest_age")
    out.to_csv(trends_out, index=False)

    print(f"  mean harvest age table: {rot_path} ({len(out)} rows)")
    print(f"  trends+harvest age: {trends_out} ({len(out)} rows)")
    return {"harvest_age_table": rot_path, "trends_with_harvest_age": trends_out}
