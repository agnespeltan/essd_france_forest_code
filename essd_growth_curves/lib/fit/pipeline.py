"""End-to-end fitting: preprocess → fit → gap-fill → CSV."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from lib.config import excluded_tfv_set
from lib.constants import FIT_QUANTILES, TARGET_COLUMN_PREFIX
from lib.fit.chapman_richards import fit_quantiles, results_to_row
from lib.fit.fill_missing import add_gap_filled_rows, drop_poor_fits
from lib.fit.preprocess import filter_tfv_greco, load_merged_stats
from lib.paths import curves_path, output_dir
from lib.raster_stats.common import is_excluded_tfv, normalize_tfv_label


def _fit_one_pair(
    df: pd.DataFrame,
    target: str,
    tfv: str,
    greco: Any,
) -> pd.DataFrame:
    filtered = filter_tfv_greco(df, target, tfv, greco)
    if filtered.empty or int(filtered["total_age_points"].iloc[0]) < 3:
        return pd.DataFrame()
    try:
        fitted = fit_quantiles(filtered, target, FIT_QUANTILES)
    except Exception as exc:  # noqa: BLE001 - keep pool alive on hard fit failures
        print(f"Fit failed for TFV={tfv} GRECO={greco}: {exc}")
        return pd.DataFrame()
    if not fitted:
        return pd.DataFrame()
    meta = {
        "TFV": str(tfv),
        "greco": greco,
        "total_data_points": int(filtered["total_data_points"].iloc[0]),
        "total_age_points": int(filtered["total_age_points"].iloc[0]),
        "EFDA_data_points": int(filtered["EFDA_data_points"].iloc[0]),
        "EFDA_age_points": int(filtered["EFDA_age_points"].iloc[0]),
        "unique_ID_count_RS": float(filtered["unique_ID_count_RS"].iloc[0]),
        "form": "a(tau)*(1-exp(-b*age))**c  [shared b,c; unweighted; monotone a]",
        "target": target,
    }
    return results_to_row(fitted, meta, FIT_QUANTILES)


def _unique_tfv_greco_pairs(df: pd.DataFrame, excluded: set[str]) -> list[tuple[str, Any]]:
    pairs = df[["TFV", "grid"]].drop_duplicates()
    out: list[tuple[str, Any]] = []
    for _, row in pairs.iterrows():
        tfv = normalize_tfv_label(row["TFV"])
        if is_excluded_tfv(tfv, excluded):
            continue
        out.append((tfv, row["grid"]))
    return out


def fit_target(cfg: dict[str, Any], df: pd.DataFrame, target_key: str) -> pd.DataFrame:
    """Fit all TFV×GRECO pairs for one target (e.g. agb → biomasseM)."""
    if target_key not in TARGET_COLUMN_PREFIX:
        raise KeyError(f"Unknown fit target '{target_key}'. Known: {list(TARGET_COLUMN_PREFIX)}")
    target = TARGET_COLUMN_PREFIX[target_key]
    excluded = excluded_tfv_set(cfg)
    pairs = _unique_tfv_greco_pairs(df, excluded)
    workers = int(cfg.get("workers", 4))
    print(
        f"Fitting {target_key} ({target}): {len(pairs)} TFV×GRECO pairs, "
        f"workers={workers} [shared b,c; unweighted; monotone a]"
    )

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for i, (tfv, greco) in enumerate(pairs, start=1):
            row = _fit_one_pair(df, target, tfv, greco)
            if not row.empty:
                frames.append(row)
            if i % 10 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fit_one_pair, df, target, tfv, greco): (tfv, greco)
                for tfv, greco in pairs
            }
            done = 0
            for future in as_completed(futures):
                row = future.result()
                if row is not None and not row.empty:
                    frames.append(row)
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    print(f"  {done}/{len(pairs)}")

    if not frames:
        return pd.DataFrame()

    results = pd.concat(frames, ignore_index=True)
    # Rule 0: min(R2_25,R2_50,R2_75) > r2_min; Rules 1–4: gap-fill (see fill_missing).
    r2_min = float(cfg.get("r2_min", cfg.get("r2_75_min", 0.2)))
    results = drop_poor_fits(results, r2_min=r2_min)
    results = add_gap_filled_rows(results, cfg=cfg, excluded_tfv=excluded)
    return results


def fit_growth_curves(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit Chapman–Richards curves and write configured CSV outputs."""
    df = load_merged_stats(cfg)
    output_dir(cfg).mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}

    for target_key in cfg.get("fit_targets", ["agb"]):
        result = fit_target(cfg, df, target_key)
        out = curves_path(cfg, target_key)
        if result.empty:
            print(f"No fits for {target_key}; writing empty CSV to {out}")
            result = pd.DataFrame(columns=["TFV", "greco", "gap_filled", "form", "target"])
        result.to_csv(out, index=False)
        print(f"Wrote {len(result)} rows → {out}")
        written[target_key] = out
    return written
