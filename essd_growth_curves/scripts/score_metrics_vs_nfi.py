#!/usr/bin/env python3
"""
Score RS-derived metrics (R30, AGBmax, t50) vs NFI using the same
GRECO / species area-weighted aggregations as the AGB trajectory RMSE.

Uses existing curve CSVs from a baseline-factor sweep directory (no refit).

  R30    = AGB(30)/30   [MgC ha⁻¹ yr⁻¹]
  AGBmax = a (= ymax)   [MgC ha⁻¹]
  t50    = age at 50% of asymptote [yr]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.fit.chapman_richards import chapman_richards
from scripts.sweep_baseline_factor_vs_nfi import (  # noqa: E402
    CARBON,
    load_area_weights,
    load_nfi_greco,
    load_nfi_species,
)

Q = "50"


def _interp_agb(ages: np.ndarray, agb: np.ndarray, age_q: float) -> float:
    ages = np.asarray(ages, float)
    agb = np.asarray(agb, float)
    ok = np.isfinite(ages) & np.isfinite(agb)
    ages, agb = ages[ok], agb[ok]
    if len(ages) < 2:
        return float("nan")
    order = np.argsort(ages)
    ages, agb = ages[order], agb[order]
    if age_q < ages[0] or age_q > ages[-1]:
        return float(np.interp(age_q, ages, agb))  # allow mild edge
    return float(np.interp(age_q, ages, agb))


def nfi_metrics(ages: np.ndarray, agb: np.ndarray) -> dict[str, float]:
    """Derived metrics from an NFI age–AGB series (already MgC/ha)."""
    ages = np.asarray(ages, float)
    agb = np.asarray(agb, float)
    ok = np.isfinite(ages) & np.isfinite(agb)
    ages, agb = ages[ok], agb[ok]
    if len(ages) < 2:
        return {"R30": np.nan, "AGBmax": np.nan, "t50": np.nan}

    agbmax = float(np.nanmax(agb))
    agb30 = _interp_agb(ages, agb, 30.0)
    r30 = agb30 / 30.0 if np.isfinite(agb30) else np.nan

    # t50: first age where AGB reaches 50% of observed max (linear interp)
    target = 0.5 * agbmax
    order = np.argsort(ages)
    ages_s, agb_s = ages[order], agb[order]
    t50 = np.nan
    if agb_s[0] >= target:
        t50 = float(ages_s[0])
    else:
        for i in range(1, len(agb_s)):
            if agb_s[i] >= target:
                a0, a1 = ages_s[i - 1], ages_s[i]
                y0, y1 = agb_s[i - 1], agb_s[i]
                if y1 > y0:
                    t50 = float(a0 + (target - y0) * (a1 - a0) / (y1 - y0))
                else:
                    t50 = float(a1)
                break
    return {"R30": float(r30), "AGBmax": agbmax, "t50": float(t50)}


def rs_metrics(row: pd.Series, q: str = Q) -> dict[str, float]:
    """Derived metrics from one fitted CR row (dry AGB → MgC)."""
    a = float(row[f"a_{q}"])
    b = float(row[f"b_{q}"])
    c = float(row[f"c_{q}"])
    if not np.isfinite([a, b, c]).all() or b <= 0 or c <= 0 or a <= 0:
        return {"R30": np.nan, "AGBmax": np.nan, "t50": np.nan}
    agb30 = float(chapman_richards(np.array([30.0]), a, b, c)[0])
    t50 = float(row[f"t50_{q}"]) if f"t50_{q}" in row.index else (
        -np.log(1.0 - 0.5 ** (1.0 / c)) / b
    )
    return {
        "R30": agb30 / 30.0 * CARBON,
        "AGBmax": a * CARBON,
        "t50": t50,
    }


def _rmse(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    if len(yt) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    err = yp - yt
    return {
        "n": int(len(yt)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def score_metric_greco(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="inner")
    if cur.empty:
        return _rmse(np.array([]), np.array([]))

    # precompute RS metrics per row
    mets = cur.apply(rs_metrics, axis=1, result_type="expand")
    cur = pd.concat([cur, mets], axis=1)

    yt, yp = [], []
    for greco, g_nfi in nfi.groupby("GRECO"):
        sub = cur.loc[cur["greco"] == str(greco)]
        if sub.empty or sub["area"].sum() <= 0:
            continue
        nfi_m = nfi_metrics(g_nfi["age"].to_numpy(float), g_nfi["agbd"].to_numpy(float))
        if not np.isfinite(nfi_m[metric]):
            continue
        pred = float(np.average(sub[metric].to_numpy(float), weights=sub["area"].to_numpy(float)))
        yt.append(nfi_m[metric])
        yp.append(pred)
    return _rmse(np.asarray(yt), np.asarray(yp))


def score_metric_species(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="left")
    cur["area"] = cur["area"].fillna(1.0)
    mets = cur.apply(rs_metrics, axis=1, result_type="expand")
    cur = pd.concat([cur, mets], axis=1)

    yt, yp = [], []
    for tfv, g_nfi in nfi.groupby("TFV"):
        sub = cur.loc[cur["TFV"] == str(int(tfv))]
        if sub.empty:
            continue
        nfi_m = nfi_metrics(g_nfi["age"].to_numpy(float), g_nfi["agbd"].to_numpy(float))
        if not np.isfinite(nfi_m[metric]):
            continue
        w = sub["area"].to_numpy(float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        pred = float(np.average(sub[metric].to_numpy(float), weights=w))
        yt.append(nfi_m[metric])
        yp.append(pred)
    return _rmse(np.asarray(yt), np.asarray(yp))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Directory with baseline_age_factor_vs_nfi_summary.csv from the sweep script",
    )
    p.add_argument(
        "--nfi-greco",
        type=Path,
        default=Path(
            "/path/to/Inventaire_national_forestier/"
            "AGB/growth/B_CACGK_GRECO_2005_2024.csv"
        ),
    )
    p.add_argument(
        "--nfi-species",
        type=Path,
        default=Path(
            "/path/to/Inventaire_national_forestier/"
            "AGB/growth/B_CACGK_esspre_2005_2024.csv"
        ),
    )
    p.add_argument(
        "--area",
        type=Path,
        default=Path("/path/to/Resultat_France/Data/France_TFVs_area.parquet"),
    )
    args = p.parse_args()

    nfi_g = load_nfi_greco(args.nfi_greco)
    nfi_s = load_nfi_species(args.nfi_species)
    area = load_area_weights(args.area)

    summary_in = args.sweep_dir / "baseline_age_factor_vs_nfi_summary.csv"
    base = pd.read_csv(summary_in)[["baseline_age", "factor", "curve_csv"]]

    metrics = ["R30", "AGBmax", "t50"]
    rows = []
    for _, r in base.iterrows():
        curves = pd.read_csv(r["curve_csv"])
        a0, f = int(r["baseline_age"]), float(r["factor"])
        print(f"a0={a0} f={f:.2f}")
        row = {"baseline_age": a0, "factor": f}
        for m in metrics:
            g = score_metric_greco(curves, nfi_g, area, m)
            s = score_metric_species(curves, nfi_s, area, m)
            row[f"greco_{m}_rmse"] = g["rmse"]
            row[f"greco_{m}_n"] = g["n"]
            row[f"species_{m}_rmse"] = s["rmse"]
            row[f"species_{m}_n"] = s["n"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = args.sweep_dir / "metric_rmse_vs_nfi_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    for m in metrics:
        unit = {"R30": "MgC/ha/yr", "AGBmax": "MgC/ha", "t50": "yr"}[m]
        print(f"\n=== GRECO {m} RMSE ({unit}) ===")
        piv = out.pivot(index="baseline_age", columns="factor", values=f"greco_{m}_rmse")
        piv.index.name = "a0"
        piv.columns = [f"{c:g}" for c in piv.columns]
        print(piv.round(2).to_string())
        print(f"\n=== Species {m} RMSE ({unit}) ===")
        piv = out.pivot(index="baseline_age", columns="factor", values=f"species_{m}_rmse")
        piv.index.name = "a0"
        piv.columns = [f"{c:g}" for c in piv.columns]
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
