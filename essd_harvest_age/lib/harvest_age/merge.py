"""Merge trends+harvest age with size-frequency MLE and severity into Data_clean tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lib.config import output_path

_KEYS = ["ser", "TFV", "agent"]

_MLE_KEEP = [
    "ser",
    "TFV",
    "agent",
    "gap_filled",
    "model_selected",
    "alpha_mle",
    "alpha_se",
    "lambda",
    "lambda_se",
    "uses_lambda",
    "ks",
    "n_fit",
    "n_tail",
    "n_total",
    "xmin_px",
    "x_det_ha",
    # hard-PL bookkeeping export
    "xmax_px",
    "xmax_ha",
    "xmax_rule",
    "hard_xmax_from_trunc",
    "alpha_hard",
    "n_fit_hard",
]

_SEV_KEEP = [
    "ser",
    "TFV",
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


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ser"] = pd.to_numeric(out["ser"], errors="coerce").astype(int)
    out["TFV"] = pd.to_numeric(out["TFV"], errors="coerce")
    out["agent"] = out["agent"].astype(str).str.lower().str.strip()
    return out


def merge_combined_tables(cfg: dict[str, Any]) -> dict[str, Path]:
    """Join mean-harvest-age table with MLE + severity; write two combined CSVs."""
    rot_path = output_path(cfg, "trends_with_harvest_age")
    mle_path = (
        Path(cfg["patch_size_mle_csv"]).expanduser()
        if cfg.get("patch_size_mle_csv")
        else output_path(cfg, "patch_size_mle")
    )
    sev_path = (
        Path(cfg["severity_csv"]).expanduser()
        if cfg.get("severity_csv")
        else output_path(cfg, "severity")
    )
    out_sf = output_path(cfg, "trends_with_harvest_age_size_freq")
    out_full = output_path(cfg, "trends_combined")

    if not rot_path.is_file():
        raise FileNotFoundError(
            f"Trends+harvest age not found: {rot_path}\nRun 01_attach_harvest_age.py first."
        )
    if not mle_path.is_file():
        raise FileNotFoundError(f"MLE CSV not found: {mle_path}")
    if not sev_path.is_file():
        raise FileNotFoundError(f"Severity CSV not found: {sev_path}")

    rot = _norm_keys(pd.read_csv(rot_path))
    mle = _norm_keys(pd.read_csv(mle_path))
    sev = _norm_keys(pd.read_csv(sev_path))

    mle_cols = [c for c in _MLE_KEEP if c in mle.columns]
    mle_m = mle[mle_cols].rename(columns={"gap_filled": "mle_gap_filled"})
    # Drop overlapping non-key cols from prior merges if re-run
    drop_mle = [c for c in mle_m.columns if c not in _KEYS and c in rot.columns]
    rot_clean = rot.drop(columns=drop_mle, errors="ignore")
    merged = rot_clean.merge(mle_m, on=_KEYS, how="left")

    out_dir = Path(cfg["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_sf, index=False)

    sev_cols = [c for c in _SEV_KEEP if c in sev.columns]
    sev_m = sev[sev_cols].rename(columns={"gap_filled": "severity_gap_filled"})
    drop_sev = [c for c in sev_m.columns if c not in _KEYS and c in merged.columns]
    merged2 = merged.drop(columns=drop_sev, errors="ignore").merge(
        sev_m, on=_KEYS, how="left"
    )
    merged2.to_csv(out_full, index=False)

    def _gap_summary(df: pd.DataFrame, col: str) -> str:
        if col not in df.columns:
            return f"{col}: missing"
        vc = df[col].value_counts(dropna=False).to_dict()
        return f"{col}: {vc}"

    print(f"  wrote {out_sf.name} rows={len(merged)} cols={merged.shape[1]}")
    print(f"  wrote {out_full.name} rows={len(merged2)} cols={merged2.shape[1]}")
    for col in ("gap_filled", "harvest_age_gap_filled", "mle_gap_filled", "severity_gap_filled"):
        print(f"  {_gap_summary(merged2, col)}")
    return {
        "trends_with_harvest_age_size_freq": out_sf,
        "trends_combined": out_full,
    }
