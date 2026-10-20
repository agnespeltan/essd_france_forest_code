#!/usr/bin/env python3
"""
Sweep EFDA young-stand baseline (age × factor) and score vs NFI.

For each baseline age a0 and factor f:
    AGB ← AGB − f × AGB(age=a0)
    keep ages ∈ [a0, max]
then Chapman–Richards curves are refit (shared b,c; monotone a; unweighted)
and scored against NFI BIOM_AR (MgC/ha) at GRECO and species levels.

Usage (from essd_growth_curves/):
    PYTHONPATH=. python3 scripts/sweep_baseline_factor_vs_nfi.py \\
        --config config.yaml \\
        --ages 0,1,2,3,4,5 \\
        --factors 0,0.25,0.5,0.75,0.99,1 \\
        --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.fit.chapman_richards import chapman_richards
from lib.fit.pipeline import fit_target
from lib.fit.preprocess import load_merged_stats
from lib.paths import output_dir

CARBON = 0.475
NFI_BG_FRAC = 1.0 - 0.221

DICT_AGE = {
    "0 - 10 ans ou non recensable": 5,
    "10 - 20 ans": 15,
    "20 - 30 ans": 25,
    "30 - 40 ans": 35,
    "40 - 50 ans": 45,
    "50 - 60 ans": 55,
    ">= 60 ans": 80,
    "Absence d'arbres mesurés": -10,
}

ESS_TO_TFV = {
    "Chênes pédonculé, rouvre et pubescent": 1,
    "Hêtre": 9,
    "Châtaignier": 10,
    "Pin maritime": 51,
    "Pin sylvestre": 52,
    "Sapin pectiné": 61,
    "Epicéa commun": 61,
    "Douglas": 64,
}


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_nfi_greco(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=12, skipfooter=1, engine="python", encoding="utf-8")
    df["age"] = df["CAC - CACGK"].map(DICT_AGE)
    df = df.loc[df["age"] > 0].copy()
    df = df.rename(columns={"Code  GRECO": "GRECO"})
    df = df[~df.eq("n.s.").any(axis=1)]
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    for col in (biom,):
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["agbd"] = df[biom] * NFI_BG_FRAC * CARBON
    df["GRECO"] = df["GRECO"].astype(str)
    return df[["GRECO", "age", "agbd"]].dropna()


def load_nfi_species(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=12, skipfooter=1, engine="python", encoding="utf-8")
    df = df.rename(columns={"ESSPRE - ESSD15": "ESS"})
    df = df.loc[df["ESS"].notna()].copy()
    df["age"] = df["CAC - CACGK"].map(DICT_AGE)
    df = df.loc[df["age"] > 0].copy()
    biom = "Moyenne hectare BIOM_AR (Tonne de matière sèche/ha)"
    df = df.replace("n.s.", np.nan).dropna(subset=[biom])
    df[biom] = df[biom].astype(str).str.replace(",", ".", regex=False)
    df[biom] = pd.to_numeric(df[biom], errors="coerce")
    df["agbd"] = df[biom] * NFI_BG_FRAC * CARBON
    df["TFV"] = df["ESS"].map(ESS_TO_TFV)
    df = df.dropna(subset=["TFV"])
    df["TFV"] = df["TFV"].astype(int)
    return df.groupby(["TFV", "age"], as_index=False)["agbd"].mean()


def load_area_weights(path: Path) -> pd.DataFrame:
    area = pd.read_parquet(path)
    area = area.replace({"Forest": "forest"})
    rename = {}
    if "grid" in area.columns:
        rename["grid"] = "greco"
    if "valid_pixel_count" in area.columns:
        rename["valid_pixel_count"] = "area"
    if "a_count" in area.columns:
        rename["a_count"] = "area"
    area = area.rename(columns=rename)
    if "greco" not in area.columns and "GRECO" in area.columns:
        area = area.rename(columns={"GRECO": "greco"})
    area["greco"] = area["greco"].astype(str)
    area["TFV"] = pd.to_numeric(area["TFV"], errors="coerce")
    area = area.dropna(subset=["TFV", "area"])
    area["TFV"] = area["TFV"].astype(int).astype(str)
    area = area.loc[~area["TFV"].isin(["0", "400"])]
    return area[["greco", "TFV", "area"]]


def rs_curve_mgc(row: pd.Series, ages: np.ndarray, q: str | int = "50") -> np.ndarray:
    q = str(int(q))
    a, b, c = float(row[f"a_{q}"]), float(row[f"b_{q}"]), float(row[f"c_{q}"])
    if not np.isfinite([a, b, c]).all() or b <= 0:
        return np.full_like(ages, np.nan, dtype=float)
    return chapman_richards(ages, a, b, c) * CARBON


def score_vs_nfi_greco(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    *,
    q: str | int = "50",
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="inner")
    if cur.empty:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    y_true, y_pred = [], []
    for greco, g_nfi in nfi.groupby("GRECO"):
        sub = cur.loc[cur["greco"] == str(greco)]
        if sub.empty or sub["area"].sum() <= 0:
            continue
        ages = g_nfi["age"].to_numpy(float)
        preds, weights = [], []
        for _, row in sub.iterrows():
            preds.append(rs_curve_mgc(row, ages, q=q))
            weights.append(float(row["area"]))
        mix = np.average(np.vstack(preds), axis=0, weights=np.asarray(weights))
        y_true.append(g_nfi["agbd"].to_numpy(float))
        y_pred.append(mix)

    if not y_true:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    err = yp - yt
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
    }


def score_vs_nfi_species(
    curves: pd.DataFrame,
    nfi: pd.DataFrame,
    area: pd.DataFrame,
    *,
    q: str | int = "50",
) -> dict[str, float]:
    cur = curves.loc[curves["gap_filled"] == 0].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur = cur.merge(area, on=["greco", "TFV"], how="left")
    cur["area"] = cur["area"].fillna(1.0)

    y_true, y_pred = [], []
    for tfv, g_nfi in nfi.groupby("TFV"):
        sub = cur.loc[cur["TFV"] == str(int(tfv))]
        if sub.empty:
            continue
        ages = g_nfi["age"].to_numpy(float)
        preds, weights = [], []
        for _, row in sub.iterrows():
            preds.append(rs_curve_mgc(row, ages, q=q))
            weights.append(float(row["area"]))
        w = np.asarray(weights)
        if w.sum() <= 0:
            w = np.ones_like(w)
        mix = np.average(np.vstack(preds), axis=0, weights=w)
        y_true.append(g_nfi["agbd"].to_numpy(float))
        y_pred.append(mix)

    if not y_true:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    err = yp - yt
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-6)) * 100),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ages", type=str, default="0,1,2,3,4,5", help="Baseline ages a0")
    p.add_argument("--factors", type=str, default="0,0.25,0.5,0.75,0.99,1.0")
    p.add_argument(
        "--tapers",
        type=str,
        default="",
        help="Comma-separated taper years T (linear fade). Empty = no taper (legacy).",
    )
    p.add_argument("--workers", type=int, default=None)
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
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    ages = [int(a) for a in _parse_floats(args.ages)]
    factors = _parse_floats(args.factors)
    tapers = _parse_floats(args.tapers) if args.tapers.strip() else [None]
    out = args.out_dir or (output_dir(cfg) / "baseline_age_factor_sweep")
    out.mkdir(parents=True, exist_ok=True)

    nfi_g = load_nfi_greco(args.nfi_greco)
    nfi_s = load_nfi_species(args.nfi_species)
    area = load_area_weights(args.area)
    print(f"NFI GRECO: {len(nfi_g)}, NFI species: {len(nfi_s)}, area: {len(area)}")
    print(f"Ages a0: {ages}")
    print(f"Factors f: {factors}")
    print(f"Tapers T: {tapers}")
    print(f"Writing to {out}")

    rows = []
    for a0 in ages:
        for T in tapers:
            for f in factors:
                t_tag = "none" if T is None else f"{T:g}"
                print(f"\n=== a0={a0}, f={f}, T={t_tag} (keep age>={a0}) ===")
                cfg["age_baseline_age"] = a0
                cfg["age_baseline_factor"] = f
                if T is None:
                    cfg.pop("baseline_taper_years", None)
                else:
                    cfg["baseline_taper_years"] = T
                df = load_merged_stats(cfg)
                curves = fit_target(cfg, df, "agb")
                curve_path = out / f"growth_curves_agb_a{a0}_f{f:.2f}_T{t_tag}.csv"
                curves.to_csv(curve_path, index=False)

                g = score_vs_nfi_greco(curves, nfi_g, area)
                s = score_vs_nfi_species(curves, nfi_s, area)
                row = {
                    "baseline_age": a0,
                    "factor": f,
                    "taper_years": np.nan if T is None else float(T),
                    "n_curves": len(curves),
                    "n_gap_filled": int((curves["gap_filled"] == 1).sum()) if "gap_filled" in curves.columns else 0,
                    "greco_n": g["n"],
                    "greco_rmse": g["rmse"],
                    "greco_mae": g["mae"],
                    "greco_bias": g["bias"],
                    "greco_mape": g["mape"],
                    "species_n": s["n"],
                    "species_rmse": s["rmse"],
                    "species_mae": s["mae"],
                    "species_bias": s["bias"],
                    "species_mape": s["mape"],
                    "curve_csv": str(curve_path),
                }
                rows.append(row)
                print(
                    f"  GRECO   RMSE={g['rmse']:.2f} MAE={g['mae']:.2f} "
                    f"bias={g['bias']:+.2f} MAPE={g['mape']:.1f}% (n={g['n']})"
                )
                print(
                    f"  Species RMSE={s['rmse']:.2f} MAE={s['mae']:.2f} "
                    f"bias={s['bias']:+.2f} MAPE={s['mape']:.1f}% (n={s['n']})"
                )

    summary = pd.DataFrame(rows)
    summary_path = out / "baseline_age_factor_vs_nfi_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary["rank_score"] = summary["greco_rmse"] + summary["species_rmse"]
    best = summary.sort_values("rank_score").iloc[0]
    print("\n=== BEST (min GRECO_RMSE + species_RMSE) ===")
    t_best = best["taper_years"]
    t_str = "none" if not np.isfinite(t_best) else f"{t_best:g}"
    print(
        f"a0={int(best['baseline_age'])}  f={best['factor']}  T={t_str}  "
        f"GRECO_RMSE={best['greco_rmse']:.2f}  species_RMSE={best['species_rmse']:.2f}  "
        f"GRECO_bias={best['greco_bias']:+.2f}"
    )
    print(f"Summary → {summary_path}")

    # Heatmaps: if single a0 and tapers present, plot f × T; else a0 × f
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if summary["baseline_age"].nunique() == 1 and summary["taper_years"].notna().any():
        piv_g = summary.pivot(index="taper_years", columns="factor", values="greco_rmse")
        piv_s = summary.pivot(index="taper_years", columns="factor", values="species_rmse")
        ylab = "taper T (yr)"
    else:
        piv_g = summary.pivot(index="baseline_age", columns="factor", values="greco_rmse")
        piv_s = summary.pivot(index="baseline_age", columns="factor", values="species_rmse")
        ylab = "baseline age a0"
    for ax, piv, title in (
        (axes[0], piv_g, "GRECO RMSE (MgC/ha)"),
        (axes[1], piv_s, "Species RMSE (MgC/ha)"),
    ):
        im = ax.imshow(piv.to_numpy(float), aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{i:g}" for i in piv.index])
        ax.set_xlabel("factor f")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_path = out / "baseline_age_factor_vs_nfi.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure → {fig_path}")


if __name__ == "__main__":
    main()
