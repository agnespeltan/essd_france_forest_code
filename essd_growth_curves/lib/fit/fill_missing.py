"""
Quality filter + gap-fill for TFV × GRECO growth curves.

Rules (applied in order after fitting)
--------------------------------------
0. QUALITY — keep a local fit only if
       min(R2_25, R2_50, R2_75) > r2_min
   Dropped pairs are treated as missing for fill.

1. UNIVERSE — only fill TFV×GRECO with area > 0
   (figures.area_parquet) and not in excluded_tfv.

2. GROUP — same GRECO bioclimatic group
   (Oceanic / Mountainous / Mediterranean).
   Donor = letter with a kept fit and largest area of that TFV.

3. CLIMATE — if group fill fails, use GRECO_CLIMATE_DONOR
   (J/K→F, B/F→G) when that donor has a kept fit.

4. else leave missing (warn).

Filled rows: gap_filled=1, fill_source=<donor>, fill_level=group|climate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lib.constants import GRECO_CLIMATE_DONOR, GRECO_GROUPS
from lib.raster_stats.common import is_excluded_tfv, normalize_tfv_label

_R2_COLS = ("R2_25", "R2_50", "R2_75")


# ---------------------------------------------------------------------------
# Rule 0 — quality filter
# ---------------------------------------------------------------------------

def drop_poor_fits(results: pd.DataFrame, *, r2_min: float = 0.2) -> pd.DataFrame:
    """Rule 0: keep rows with min(R2_25, R2_50, R2_75) > r2_min."""
    if results.empty:
        return results.copy()
    missing = [c for c in _R2_COLS if c not in results.columns]
    if missing:
        raise KeyError(f"Need columns {list(_R2_COLS)}; missing {missing}")
    r2 = results[list(_R2_COLS)].apply(pd.to_numeric, errors="coerce")
    keep = r2.min(axis=1) > float(r2_min)
    n_drop = len(results) - int(keep.sum())
    if n_drop:
        print(f"Rule 0 (quality): dropped {n_drop} with min(R2) <= {r2_min}")
    return results.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Area universe (Rule 1)
# ---------------------------------------------------------------------------

def _load_area_universe(cfg: dict[str, Any], excluded: set[str]) -> pd.DataFrame:
    """Rule 1: TFV×GRECO with area > 0 (columns: TFV, greco, area, greco_group)."""
    figs = cfg.get("figures") or {}
    path = figs.get("area_parquet")
    if not path:
        raise KeyError("figures.area_parquet is required for gap-fill")
    path = Path(str(path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"area_parquet not found: {path}")

    area = pd.read_parquet(path).replace({"Forest": "forest"})
    rename: dict[str, str] = {}
    if "grid" in area.columns:
        rename["grid"] = "greco"
    elif "GRECO" in area.columns:
        rename["GRECO"] = "greco"
    if "valid_pixel_count" in area.columns:
        rename["valid_pixel_count"] = "area"
    elif "a_count" in area.columns:
        rename["a_count"] = "area"
    area = area.rename(columns=rename)
    need = {"greco", "TFV", "area"}
    if not need.issubset(area.columns):
        raise KeyError(f"area_parquet needs {sorted(need)}: {path}")

    area["TFV"] = area["TFV"].map(normalize_tfv_label)
    area["greco"] = area["greco"].astype(str)
    area["area"] = pd.to_numeric(area["area"], errors="coerce").fillna(0.0)
    area = area.loc[
        (area["area"] > 0) & (~area["TFV"].map(lambda t: is_excluded_tfv(t, excluded))),
        ["TFV", "greco", "area"],
    ].drop_duplicates(["TFV", "greco"])
    area["greco_group"] = area["greco"].map(GRECO_GROUPS)
    return area.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Donor choice (Rules 2–3)
# ---------------------------------------------------------------------------

def _best_group_donors(area: pd.DataFrame, kept_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Per (TFV, group): GRECO letter with kept fit and largest area."""
    donors = area.loc[
        area.apply(lambda r: (str(r.TFV), str(r.greco)) in kept_pairs, axis=1)
    ].copy()
    if donors.empty:
        return pd.DataFrame(columns=["TFV", "greco_group", "fill_source"])
    return (
        donors.sort_values("area", ascending=False)
        .groupby(["TFV", "greco_group"], as_index=False)
        .first()[["TFV", "greco_group", "greco"]]
        .rename(columns={"greco": "fill_source"})
    )


def _choose_donor(
    *,
    tf: str,
    greco: str,
    group: Any,
    best_group: pd.DataFrame,
    kept_pairs: set[tuple[str, str]],
    climate_donor: dict[str, str],
) -> tuple[str, str] | None:
    """
    Return (fill_source, fill_level) or None.

    Rule 2 — GROUP: same greco_group, max-area kept fit.
    Rule 3 — CLIMATE: GRECO_CLIMATE_DONOR[greco] if kept.
    """
    # Rule 2
    if pd.notna(group):
        hit = best_group.loc[
            (best_group["TFV"].astype(str) == tf) & (best_group["greco_group"] == group)
        ]
        if not hit.empty:
            cand = str(hit.iloc[0]["fill_source"])
            if cand != greco and (tf, cand) in kept_pairs:
                return cand, "group"

    # Rule 3
    cand = climate_donor.get(greco)
    if cand and cand != greco and (tf, cand) in kept_pairs:
        return cand, "climate"

    return None


def _copy_fill_row(
    source: pd.Series,
    *,
    tf: str,
    greco: str,
    fill_source: str,
    fill_level: str,
) -> pd.Series:
    row = source.copy()
    row["TFV"] = tf
    row["greco"] = greco
    row["gap_filled"] = 1
    row["fill_source"] = fill_source
    row["fill_level"] = fill_level
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_gap_filled_rows(
    results: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    excluded_tfv: Iterable[str] | None = None,
    climate_donor: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply Rules 1–4; see module docstring."""
    if results.empty:
        return results.copy()
    if cfg is None:
        raise ValueError("Gap-fill requires cfg (figures.area_parquet)")

    climate_donor = dict(climate_donor or GRECO_CLIMATE_DONOR)
    excluded = {str(x) for x in (excluded_tfv or ())}
    universe = _load_area_universe(cfg, excluded)

    out = results.copy()
    out["TFV"] = out["TFV"].map(normalize_tfv_label)
    out["greco"] = out["greco"].astype(str)
    out = out.loc[~out["TFV"].map(lambda t: is_excluded_tfv(t, excluded))].reset_index(drop=True)
    out["gap_filled"] = 0
    out["fill_source"] = pd.NA
    out["fill_level"] = pd.NA

    kept_pairs = {(str(t), str(g)) for t, g in zip(out["TFV"], out["greco"])}
    source_by_pair = {(str(r.TFV), str(r.greco)): r for _, r in out.iterrows()}
    best_group = _best_group_donors(universe, kept_pairs)

    to_fill = universe.loc[
        ~universe.apply(lambda r: (str(r.TFV), str(r.greco)) in kept_pairs, axis=1)
    ]

    new_rows: list[pd.Series] = []
    n_group = n_climate = n_miss = 0
    for _, m in to_fill.iterrows():
        tf, greco = str(m.TFV), str(m.greco)
        choice = _choose_donor(
            tf=tf,
            greco=greco,
            group=m.greco_group,
            best_group=best_group,
            kept_pairs=kept_pairs,
            climate_donor=climate_donor,
        )
        if choice is None:
            n_miss += 1
            print(f"Rule 4: no donor for TFV={tf} GRECO={greco} (group={m.greco_group})")
            continue
        fill_source, fill_level = choice
        new_rows.append(
            _copy_fill_row(
                source_by_pair[(tf, fill_source)],
                tf=tf,
                greco=greco,
                fill_source=fill_source,
                fill_level=fill_level,
            )
        )
        if fill_level == "group":
            n_group += 1
        else:
            n_climate += 1

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    print(
        f"Gap-fill: +{len(new_rows)} (group={n_group}, climate={n_climate}); "
        f"still missing={n_miss}"
    )
    out["gap_filled"] = out["gap_filled"].astype(int)
    return out
