"""R30, AGBmax and t50 from the refitted curves, scored against the NFI (08).

Uses the curve CSVs of a sweep folder (no refit) and the same GRECO / species
area-weighted aggregation as the AGB-trajectory RMSE in ``lib.sensitivity.nfi``.

  R30    = AGB(30)/30   [MgC ha⁻¹ yr⁻¹]
  AGBmax = a            [MgC ha⁻¹]
  t50    = age at 50 % of the asymptote [yr]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.fit.chapman_richards import chapman_richards
from lib.sensitivity.nfi import CARBON

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


METRICS = ["R30", "AGBmax", "t50"]
UNITS = {"R30": "MgC/ha/yr", "AGBmax": "MgC/ha", "t50": "yr"}
# Published settings (Δ% reference per swept parameter); see config.yaml.
REFERENCE_VALUES = {"tail_trim_age": "90", "r2_min": "0.2", "cr_b_min": "0.01",
                    "tail_trim_delta": "0.05", "tail_trim_patience": "1"}


def _with_rs_metrics(curves: pd.DataFrame, area: pd.DataFrame, how: str) -> pd.DataFrame:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how=how)
    if how == "left":
        cur["area"] = cur["area"].fillna(1.0)
    if cur.empty:
        return cur
    mets = cur.apply(rs_metrics, axis=1, result_type="expand")
    return pd.concat([cur.reset_index(drop=True), mets.reset_index(drop=True)], axis=1)


def metric_pairs_greco(curves: pd.DataFrame, nfi: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """One row per GRECO × metric: NFI value, area-weighted RS value, GRECO area."""
    cur = _with_rs_metrics(curves, area, "inner")
    rows = []
    for greco, g_nfi in nfi.groupby("GRECO"):
        sub = cur.loc[cur["greco"] == str(greco)] if not cur.empty else cur
        if sub.empty or sub["area"].sum() <= 0:
            continue
        nfi_m = nfi_metrics(g_nfi["age"].to_numpy(float), g_nfi["agbd"].to_numpy(float))
        w = sub["area"].to_numpy(float)
        for m in METRICS:
            v = sub[m].to_numpy(float)
            ok = np.isfinite(v)
            rs = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
            rows.append({"level": "greco", "unit": str(greco), "metric": m,
                         "nfi": nfi_m[m], "rs": rs, "weight": float(w.sum())})
    return pd.DataFrame(rows)


def metric_pairs_species(curves: pd.DataFrame, nfi: pd.DataFrame, area: pd.DataFrame) -> pd.DataFrame:
    """One row per NFI species (TFV) × metric: NFI value, area-weighted RS value, TFV area."""
    cur = _with_rs_metrics(curves, area, "left")
    rows = []
    for tfv, g_nfi in nfi.groupby("TFV"):
        sub = cur.loc[cur["TFV"] == str(int(tfv))] if not cur.empty else cur
        if sub.empty:
            continue
        nfi_m = nfi_metrics(g_nfi["age"].to_numpy(float), g_nfi["agbd"].to_numpy(float))
        w = sub["area"].to_numpy(float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        for m in METRICS:
            v = sub[m].to_numpy(float)
            ok = np.isfinite(v)
            rs = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
            rows.append({"level": "species", "unit": str(int(tfv)), "metric": m,
                         "nfi": nfi_m[m], "rs": rs, "weight": float(w.sum())})
    return pd.DataFrame(rows)


def _score_pairs(pairs: pd.DataFrame, metric: str) -> dict[str, float]:
    if pairs.empty:
        return _rmse(np.array([]), np.array([]))
    sub = pairs.loc[pairs["metric"] == metric]
    return _rmse(sub["nfi"].to_numpy(float), sub["rs"].to_numpy(float))


def score_metric_greco(curves, nfi, area, metric: str) -> dict[str, float]:
    return _score_pairs(metric_pairs_greco(curves, nfi, area), metric)


def score_metric_species(curves, nfi, area, metric: str) -> dict[str, float]:
    return _score_pairs(metric_pairs_species(curves, nfi, area), metric)


def national_values(pairs_greco: pd.DataFrame) -> dict[str, float]:
    """Area-weighted mean over the GRECOs that have both NFI and RS values (same set for both)."""
    out = {}
    for m in METRICS:
        sub = pairs_greco.loc[pairs_greco["metric"] == m]
        sub = sub.loc[np.isfinite(sub["nfi"]) & np.isfinite(sub["rs"])]
        w = sub["weight"].to_numpy(float)
        out[f"national_{m}_rs"] = float(np.average(sub["rs"], weights=w)) if len(sub) else np.nan
        out[f"national_{m}_nfi"] = float(np.average(sub["nfi"], weights=w)) if len(sub) else np.nan
    return out


def score_sweep(
    sweep_dir: Path,
    nfi_g: pd.DataFrame,
    nfi_s: pd.DataFrame,
    area: pd.DataFrame,
    *,
    reference_factor: float = 0.85,
) -> pd.DataFrame:
    """Score every run of a sweep folder; write the per-run, per-unit and summary tables.

    Writes in ``sweep_dir``:
      ``metric_rmse_vs_nfi_summary.csv``      RMSE / bias / MAE vs NFI per run and metric;
      ``metric_by_factor_long.csv``           per GRECO / species NFI and RS values;
      ``baseline_factor_sensitivity_table.csv`` national RS and NFI values, Δ% vs the
      published setting, RMSE / bias (one row per run; read by the figures).
    """
    # Either the a0 × f grid or a one-setting sweep.
    generic = sweep_dir / "fit_settings_vs_nfi_summary.csv"
    summary_path = generic if generic.is_file() else sweep_dir / "baseline_age_factor_vs_nfi_summary.csv"
    summary = pd.read_csv(summary_path, dtype={"value": str})
    if "param" not in summary.columns:
        summary["param"] = "age_baseline_factor"
        summary["value"] = summary["factor"].map(lambda v: f"{v:g}")
        summary["value_num"] = summary["factor"]

    rows, long = [], []
    for _, r in summary.iterrows():
        curves = pd.read_csv(r["curve_csv"])
        a0, f = int(r["baseline_age"]), float(r["factor"])
        print(f"  score {r['param']} = {r['value']}")
        pg = metric_pairs_greco(curves, nfi_g, area)
        ps = metric_pairs_species(curves, nfi_s, area)
        row = {"param": r["param"], "value": r["value"], "value_num": r["value_num"],
               "baseline_age": a0, "factor": f}
        for k in ("n_curves", "n_fitted", "n_gap_filled"):
            if k in r.index:
                row[k] = r[k]
        for m in METRICS:
            for lvl, pairs in (("greco", pg), ("species", ps)):
                sc = _score_pairs(pairs, m)
                for k in ("rmse", "bias", "mae", "n"):
                    row[f"{lvl}_{m}_{k}"] = sc[k]
        row.update(national_values(pg))
        row["greco_agb_rmse"] = r["greco_rmse"]
        row["species_agb_rmse"] = r["species_rmse"]
        rows.append(row)
        for pairs in (pg, ps):
            long.append(pairs.assign(param=r["param"], value=r["value"], baseline_age=a0, factor=f))

    out = pd.DataFrame(rows)
    out.to_csv(sweep_dir / "metric_rmse_vs_nfi_summary.csv", index=False)

    long_df = pd.concat(long, ignore_index=True)
    long_df = long_df[["param", "value", "baseline_age", "factor", "level", "unit", "metric", "nfi", "rs", "weight"]]
    long_df.to_csv(sweep_dir / "metric_by_factor_long.csv", index=False)

    tab = out.copy()
    refs = {"age_baseline_factor": f"{reference_factor:g}", **REFERENCE_VALUES}
    for key, g in tab.groupby(["param", "baseline_age"]):
        ref = g.loc[g["value"] == refs.get(key[0], "")]
        if ref.empty:
            continue
        for m in METRICS:
            v0 = float(ref[f"national_{m}_rs"].iloc[0])
            tab.loc[g.index, f"national_{m}_rs_dpct_vs_ref"] = 100.0 * (g[f"national_{m}_rs"] - v0) / v0
    keep = ["param", "value", "value_num", "baseline_age", "factor", "n_curves", "n_fitted", "n_gap_filled"]
    for m in METRICS:
        keep += [f"national_{m}_rs", f"national_{m}_nfi", f"national_{m}_rs_dpct_vs_ref",
                 f"greco_{m}_rmse", f"greco_{m}_bias", f"species_{m}_rmse", f"species_{m}_bias"]
    keep += ["greco_agb_rmse", "species_agb_rmse"]
    tab = tab[[c for c in keep if c in tab.columns]].sort_values(
        ["param", "baseline_age", "value_num"], na_position="first")
    tab.to_csv(sweep_dir / "baseline_factor_sensitivity_table.csv", index=False)
    print(f"  wrote {sweep_dir / 'baseline_factor_sensitivity_table.csv'}")
    return tab
