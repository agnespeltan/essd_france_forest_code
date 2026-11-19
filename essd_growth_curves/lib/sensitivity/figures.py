"""SI figures of the growth-curve sensitivity tests (09).

Inputs are the sweep folders written by 08 (``baseline_factor_sensitivity_table.csv``
and ``metric_by_factor_long.csv`` in each).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("AGBmax", r"AGB$_{max}$ (MgC ha$^{-1}$)"),
    ("t50", r"t$_{50}$ (yr)"),
    ("R30", r"R$_{30}$ (MgC ha$^{-1}$ yr$^{-1}$)"),
]
PARAMS = {  # key -> (axis label, published value)
    "tail_trim_delta": (r"trim tolerance $\delta$ (fraction)", "0.05"),
    "tail_trim_patience": ("trim patience (points)", "1"),
    "tail_trim_age": ("tail-trim age (yr)", "90"),
    "r2_min": (r"R$^2$ threshold", "0.2"),
    "cr_b_min": (r"lower bound on b (yr$^{-1}$)", "0.01"),
    "age_baseline_factor": ("baseline factor f", "0.85"),
}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8983"


def load(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs, longs = [], []
    for d in dirs:
        for name in ("baseline_factor_sensitivity_table.csv",):
            t = pd.read_csv(d / name, dtype={"value": str})
            tabs.append(t)
        longs.append(pd.read_csv(d / "metric_by_factor_long.csv", dtype={"value": str}))
    tab = pd.concat(tabs, ignore_index=True)
    long = pd.concat(longs, ignore_index=True)
    # The default run appears in every sweep; keep one per (param, value).
    tab = tab.drop_duplicates(["param", "value"]).reset_index(drop=True)
    return tab, long



def fig_fit_settings(
    sweep_dirs: list[Path], table: Path, out: Path, params: list[str] | None = None
) -> pd.DataFrame:
    """SI figure + combined table: one row of panels per swept setting, columns = metric.

    Each panel shows the national area-weighted RS value (band = GRECO 25–75 %),
    the NFI national value (dashed) and the published setting (dotted). The table
    keeps every setting; ``params`` only selects the rows drawn.
    """
    tab, long = load(sweep_dirs)
    all_params = [k for k in PARAMS if k in set(tab["param"])]
    tab["param"] = pd.Categorical(tab["param"], all_params, ordered=True)
    tab = tab.sort_values(["param", "value_num"], na_position="first")
    table.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(table, index=False)
    print(f"Table → {table}")
    params = [k for k in (params or all_params) if k in all_params]

    fig, axes = plt.subplots(len(params), 3, figsize=(11, 3.1 * len(params)), squeeze=False)
    for i, key in enumerate(params):
        xlabel, ref = PARAMS[key]
        t = tab.loc[tab["param"] == key]
        vals = t["value"].tolist()
        x = np.arange(len(vals))  # categorical: "none" sits first
        lg = long.loc[(long["param"] == key) & (long["level"] == "greco")]
        for j, (m, label) in enumerate(METRICS):
            ax = axes[i, j]
            q = lg.loc[lg["metric"] == m].groupby("value")["rs"].quantile([0.25, 0.75]).unstack()
            q = q.reindex(vals)
            ax.fill_between(x, q[0.25], q[0.75], color=BLUE, alpha=0.15, lw=0, label="RS, GRECO 25–75 %")
            ax.plot(x, t[f"national_{m}_rs"], color=BLUE, lw=2, marker="o", ms=5, label="RS, national")
            ax.axhline(t[f"national_{m}_nfi"].median(), color=INK, lw=1.2, ls="--", label="NFI, national")
            if ref in vals:
                ax.axvline(vals.index(ref), color=MUTED, lw=1, ls=":")
            for k in (0, len(x) - 1):
                d = t[f"national_{m}_rs_dpct_vs_ref"].iloc[k]
                if np.isfinite(d):
                    ax.annotate(f"{d:+.1f} %", (x[k], t[f"national_{m}_rs"].iloc[k]),
                                textcoords="offset points", xytext=(4 if k == 0 else -4, 7),
                                ha="left" if k == 0 else "right", fontsize=8, color=MUTED)
            ax.set_xticks(x)
            ax.set_xticklabels(vals)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.margins(y=0.15)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")

    return tab


def fig_baseline_factor(sweep_dir: Path, out: Path, reference_factor: float = 0.85) -> Path:
    """SI figure for the baseline factor f.

    Top row: national area-weighted RS metric vs f (band = GRECO 25–75 %), NFI national
    value dashed. Bottom row: RMSE vs NFI at GRECO and species level.
    """
    tab = pd.read_csv(sweep_dir / "baseline_factor_sensitivity_table.csv").sort_values("factor")
    long = pd.read_csv(sweep_dir / "metric_by_factor_long.csv")
    band = (
        long.loc[long["level"] == "greco"]
        .groupby(["metric", "factor"])["rs"]
        .quantile([0.25, 0.75])
        .unstack()
    )
    f = tab["factor"].to_numpy(float)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for j, (m, label) in enumerate(METRICS):
        ax = axes[0, j]
        lo = band.loc[m][0.25].reindex(f).to_numpy(float)
        hi = band.loc[m][0.75].reindex(f).to_numpy(float)
        ax.fill_between(f, lo, hi, color=BLUE, alpha=0.15, lw=0, label="RS, GRECO 25–75 %")
        ax.plot(f, tab[f"national_{m}_rs"], color=BLUE, lw=2, marker="o", ms=5, label="RS, national")
        ax.axhline(tab[f"national_{m}_nfi"].median(), color=INK, lw=1.2, ls="--", label="NFI, national")
        ax.set_ylabel(label)
        # Δ% at the ends of the range, relative to the reference factor
        for k in (0, len(f) - 1):
            d = tab[f"national_{m}_rs_dpct_vs_ref"].iloc[k]
            ax.annotate(f"{d:+.1f} %", (f[k], tab[f"national_{m}_rs"].iloc[k]),
                        textcoords="offset points", xytext=(4 if k == 0 else -4, 7),
                        ha="left" if k == 0 else "right", fontsize=8, color=MUTED)

        ax = axes[1, j]
        ax.plot(f, tab[f"greco_{m}_rmse"], color=BLUE, lw=2, marker="o", ms=5, label="GRECO")
        ax.plot(f, tab[f"species_{m}_rmse"], color=ORANGE, lw=2, marker="s", ms=5, label="Species")
        ax.set_ylabel(f"RMSE vs NFI ({label.split('(')[1]}")
        ax.set_xlabel("baseline factor f")

    for ax in axes.ravel():
        ax.axvline(reference_factor, color=MUTED, lw=1, ls=":")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.margins(y=0.15)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    l1 = [f"RMSE, {x}" for x in l1]
    fig.legend(h0 + h1, l0 + l1, loc="upper center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    axes[0, 0].annotate(f"f = {reference_factor:g}", (reference_factor, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(3, -10),
                        textcoords="offset points", fontsize=8, color=MUTED)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")

    return out
