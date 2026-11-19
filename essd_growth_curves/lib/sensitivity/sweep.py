"""Refit the growth curves with changed settings and score them against the NFI (08).

Two kinds of sweep:

* ``run_baseline_grid``: young-stand baseline age a0 × factor f (× optional taper T).
  For each run, AGB ← AGB − f × AGB(age = a0) and ages < a0 are dropped, then the
  curves are refit. Writes ``baseline_age_factor_vs_nfi_summary.csv`` and an RMSE heat map.
* ``run_setting_sweep``: one config setting at a time (e.g. ``tail_trim_age``,
  ``r2_min``, ``tail_trim_delta``, ``tail_trim_patience``, ``cr_b_min``,
  ``age_baseline_factor``), everything else from the config. Writes
  ``fit_settings_vs_nfi_summary.csv``.

Each run also writes its curves CSV, so ``lib.sensitivity.metrics.score_sweep`` can
score R30 / AGBmax / t50 without refitting.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import excluded_tfv_set
from lib.constants import FIT_QUANTILES, TARGET_COLUMN_PREFIX
from lib.fit.chapman_richards import trim_decreasing_tail, trim_settings
from lib.fit.pipeline import _unique_tfv_greco_pairs, fit_target
from lib.fit.preprocess import filter_tfv_greco, load_merged_stats
from lib.sensitivity.nfi import score_vs_nfi_greco, score_vs_nfi_species

def _parse_value(v):
    """Config value → float, or None for ``none`` / ``null`` (setting switched off)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ("none", "null"):
            return None
    return float(v)


def _tag(v) -> str:
    return "none" if v is None else f"{v:g}"


def trim_diagnostic(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Age points removed by the configured tail trim, per TFV × GRECO × quantile."""
    target = TARGET_COLUMN_PREFIX["agb"]
    trim = trim_settings(cfg)
    rows = []
    for tfv, greco in _unique_tfv_greco_pairs(df, excluded_tfv_set(cfg)):
        filt = filter_tfv_greco(df, target, tfv, greco)
        if filt.empty:
            continue
        ages = filt["age"].to_numpy(float)
        for tau in FIT_QUANTILES:
            y = filt[f"{target}_{int(100 * tau)}th"].to_numpy(float)
            x_kept, _ = trim_decreasing_tail(
                ages, y, x_threshold=trim["age"], delta=trim["delta"], patience=trim["patience"]
            )
            removed = np.setdiff1d(np.sort(ages[np.isfinite(ages)]), x_kept)
            rows.append({
                "TFV": tfv, "greco": greco, "quantile": int(100 * tau),
                "n_points": int(np.isfinite(ages).sum()),
                "n_points_above_trim_age": int((ages > trim["age"]).sum()),
                "n_removed": int(len(removed)),
                "cut_age": float(removed.min()) if len(removed) else np.nan,
                "max_age": float(np.nanmax(ages)),
            })
    return pd.DataFrame(rows)


def run_setting_sweep(
    base: dict,
    settings: dict[str, list],
    out: Path,
    nfi_g: pd.DataFrame,
    nfi_s: pd.DataFrame,
    area: pd.DataFrame,
    *,
    with_trim_diagnostic: bool = False,
) -> pd.DataFrame:
    """Refit once per value of each setting (``{key: [v1, v2, ...]}``; ``None`` = off)."""
    out.mkdir(parents=True, exist_ok=True)
    df_base = load_merged_stats(base)

    if with_trim_diagnostic:
        diag = trim_diagnostic(base, df_base)
        diag.to_csv(out / "tail_trim_diagnostic.csv", index=False)
        q50 = diag.loc[diag["quantile"] == 50]
        print(
            f"Trim diagnostic (q50): {int((q50['n_removed'] > 0).sum())}/{len(q50)} strata trimmed, "
            f"{int(q50['n_removed'].sum())}/{int(q50['n_points'].sum())} age points removed, "
            f"median cut age {q50['cut_age'].median():.0f} yr"
        )

    rows = []
    for key, vals in settings.items():
        for v in [_parse_value(x) for x in vals]:
            cfg = copy.deepcopy(base)
            cfg[key] = v
            print(f"\n=== {key} = {_tag(v)} ===")
            # Only the baseline-correction keys change the preprocessed input.
            df = load_merged_stats(cfg) if key.startswith("age_baseline") else df_base
            curves = fit_target(cfg, df, "agb")
            path = out / f"growth_curves_agb_{key}_{_tag(v)}.csv"
            curves.to_csv(path, index=False)
            g = score_vs_nfi_greco(curves, nfi_g, area)
            s = score_vs_nfi_species(curves, nfi_s, area)
            rows.append({
                "param": key,
                "value": _tag(v),
                "value_num": np.nan if v is None else float(v),
                # columns expected by lib.sensitivity.metrics.score_sweep
                "baseline_age": float(cfg.get("age_baseline_age", 2)),
                "factor": float(cfg.get("age_baseline_factor", 0.85)),
                "n_curves": len(curves),
                "n_fitted": int((curves["gap_filled"] == 0).sum()),
                "n_gap_filled": int((curves["gap_filled"] != 0).sum()),
                "greco_rmse": g["rmse"], "greco_bias": g["bias"],
                "species_rmse": s["rmse"], "species_bias": s["bias"],
                "curve_csv": str(path),
            })
            print(f"  fitted={rows[-1]['n_fitted']} gap-filled={rows[-1]['n_gap_filled']} "
                  f"GRECO RMSE={g['rmse']:.2f} species RMSE={s['rmse']:.2f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "fit_settings_vs_nfi_summary.csv", index=False)
    print(f"Summary → {out / 'fit_settings_vs_nfi_summary.csv'}")
    return summary


def run_baseline_grid(
    cfg: dict,
    ages: list[int],
    factors: list[float],
    tapers: list[float | None],
    out: Path,
    nfi_g: pd.DataFrame,
    nfi_s: pd.DataFrame,
    area: pd.DataFrame,
) -> pd.DataFrame:
    """Refit for every baseline age a0 × factor f (× taper T; ``None`` = no taper)."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
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
    plt.close(fig)
    print(f"Figure → {fig_path}")

    return summary
