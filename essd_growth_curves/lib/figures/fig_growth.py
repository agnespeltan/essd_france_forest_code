"""Paper figures: Fig. 2, SI1, SI2 (self-contained; used by 05_make_figures.py).

Adapted from ``Figures/Fig_age_distribution.py`` panels for AGB growth curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from lib.figures.constants import (
    BUBBLE_ALPHA,
    DICT_GROUP_EN,
    FS_O,
    FS_T,
    GRECO_GROUP_COLORS,
    GRECO_GROUP_ORDER,
    GRECO_GROUPS,
    GRECO_LETTER_COLORS,
    GRECO_LETTERS,
    REGIONAL_PANELS,
    RS_CARBON,
    RS_ENV_HI_Q,
    RS_ENV_LO_Q,
    RS_MAIN_Q,
    TFV_GROUP,
    TFV_GROUP_COLORS,
    TFV_GROUP_ORDER,
    VALIDATION_SPECIES,
)
from lib.figures.cr import (
    area_weighted_mean_std,
    chapman_richards,
    cr_nfi,
    fit_cr3,
    param_d,
    t50_from_abcd,
)
from lib.figures.data import (
    load_area_tfv_greco,
    load_greco_gdf,
    load_nfi_fits,
    load_nfi_greco_series,
    load_nfi_species_series,
    load_robinson,
    load_rs_curves,
    metrics_from_nfi_greco,
    metrics_from_nfi_species,
    metrics_from_robinson,
    metrics_from_rs,
)

AGES = np.linspace(0, 150, 151)
VALIDATION_AGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 115)]


def _ensure_d(ddf: pd.DataFrame) -> pd.DataFrame:
    for q in (RS_MAIN_Q, RS_ENV_LO_Q, RS_ENV_HI_Q, "5", "95"):
        col = f"d_{q}"
        if col not in ddf.columns:
            ddf[col] = 0.0
    return ddf


def _plot_all_greco_fitted_curves(ax, ddf, tfv, y_label_cap=150):
    end_x = 120
    end_points = []
    for greco in GRECO_LETTERS:
        rs_row = ddf[(ddf["TFV"] == tfv) & (ddf["greco"] == greco)]
        if rs_row.empty:
            continue
        row = rs_row.iloc[0]
        params_c = (
            row[f"a_{RS_MAIN_Q}"],
            row[f"b_{RS_MAIN_Q}"],
            row[f"c_{RS_MAIN_Q}"],
            param_d(row, RS_MAIN_Q),
        )
        if not np.all(np.isfinite(params_c)):
            continue
        biomass_c = chapman_richards(AGES, *params_c) * RS_CARBON
        color = GRECO_LETTER_COLORS.get(greco, "gray")
        params_lo = (
            row.get(f"a_{RS_ENV_LO_Q}"),
            row.get(f"b_{RS_ENV_LO_Q}"),
            row.get(f"c_{RS_ENV_LO_Q}"),
            param_d(row, RS_ENV_LO_Q),
        )
        params_hi = (
            row.get(f"a_{RS_ENV_HI_Q}"),
            row.get(f"b_{RS_ENV_HI_Q}"),
            row.get(f"c_{RS_ENV_HI_Q}"),
            param_d(row, RS_ENV_HI_Q),
        )
        if np.all(np.isfinite(params_lo)) and np.all(np.isfinite(params_hi)):
            lo = chapman_richards(AGES, *params_lo) * RS_CARBON
            hi = chapman_richards(AGES, *params_hi) * RS_CARBON
            ax.fill_between(AGES, np.minimum(lo, hi), np.maximum(lo, hi), color=color, alpha=0.1, linewidth=0, zorder=1)
        ax.plot(AGES, biomass_c, color=color, linewidth=1.0, zorder=2)
        end_y = float(chapman_richards(end_x, *params_c) * RS_CARBON)
        if np.isfinite(end_y):
            end_points.append((end_y, greco, params_c))
    end_points.sort(reverse=True)
    for i, (end_y, greco, params) in enumerate(end_points):
        x_offset = [0, 5, 10][i % 3]
        if end_y > y_label_cap:
            x_vals = np.linspace(0, 150, 1501)
            y_vals = chapman_richards(x_vals, *params) * RS_CARBON
            idx = np.where(y_vals >= y_label_cap)[0]
            label_x = x_vals[idx[0]] if len(idx) else end_x - x_offset
            label_y = y_label_cap
        else:
            label_x = end_x - x_offset
            label_y = float(chapman_richards(label_x, *params) * RS_CARBON)
        ax.text(label_x, label_y, greco, color="black", fontsize=7, va="center", ha="right", fontweight="bold", zorder=6)


def _plot_nfi_validation(ax, dfi_growth, tfv):
    dfi = dfi_growth[dfi_growth["TFV"] == tfv].sort_values("age")
    if dfi.empty:
        return
    ages, medians, err_lo, err_hi = [], [], [], []
    for start, end in VALIDATION_AGE_BINS:
        bin_data = dfi[(dfi["age"] >= start) & (dfi["age"] <= end)]
        if bin_data.empty:
            continue
        w = bin_data["weight"] / bin_data["weight"].sum()
        ages.append(start + 27.5 if end == 115 else start + 5)
        medians.append(np.average(bin_data["agbd"], weights=w))
        err_lo.append(np.average(bin_data["agbd_errorm"], weights=w))
        err_hi.append(np.average(bin_data["agbd_errorp"], weights=w))
    if len(ages) < 3:
        return
    ages = np.asarray(ages, float)
    medians = np.asarray(medians, float)
    p_mean = fit_cr3(ages, medians)
    if p_mean is None:
        return
    p_lo = fit_cr3(ages, np.asarray(err_lo, float), p0=p_mean)
    p_hi = fit_cr3(ages, np.asarray(err_hi, float), p0=p_mean)
    grid = np.linspace(0, 125, 200)
    if p_lo is not None and p_hi is not None:
        lo = cr_nfi(grid, *p_lo)
        hi = cr_nfi(grid, *p_hi)
        ax.fill_between(grid, np.minimum(lo, hi), np.maximum(lo, hi), color="black", alpha=0.12, linewidth=0, zorder=4)
    ax.plot(grid, cr_nfi(grid, *p_mean), color="black", linewidth=1.8, zorder=5, alpha=0.95)


def _style_validation_ax(ax, title, show_ylabel=True, ylim=(0, 160), yticks=None):
    ax.set_title(title, fontsize=FS_T, pad=8)
    ax.set_xlim(0, 125)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"], fontsize=FS_O)
    if yticks is None:
        yticks = [ylim[0], 0.5 * (ylim[0] + ylim[1]), ylim[1]]
    ax.set_yticks(yticks)
    labels = [str(int(t)) if float(t).is_integer() else f"{t:g}" for t in yticks]
    if show_ylabel:
        ax.set_ylabel(r"AGB (MgC ha$^{-1}$)", fontsize=FS_O)
        ax.set_yticklabels(labels, fontsize=FS_O)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Age (years)", fontsize=FS_O)
    ax.tick_params(axis="both", labelsize=FS_O)
    ax.set_box_aspect(1)


def _plot_greco_map(ax, gdf, title="Large ecological regions"):
    colors = [GRECO_LETTER_COLORS.get(str(x), "lightgray") for x in gdf["GRECO"]]
    gdf.plot(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    offsets = {
        "E": (100000, 0),
        "H": (0, -20000),
        "J": (0, 30000),
        "C": (-30000, -20000),
        "D": (0, 130000),
        "I": (0, -80000),
        "K": (-100000, 0),
    }
    for _, row in gdf.iterrows():
        dx, dy = offsets.get(row["GRECO"], (0, 0))
        ax.annotate(
            row["GRECO"],
            xy=(row.geometry.centroid.x + dx, row.geometry.centroid.y + dy),
            ha="center",
            va="center",
            color="black",
            fontsize=7,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=FS_T, pad=8)


def _dot_whisker(ax, x, mu, sd, color, marker="o", markersize=7):
    if not np.isfinite(mu):
        return
    sd = 0.0 if not np.isfinite(sd) else float(sd)
    ax.errorbar(
        [x],
        [mu],
        yerr=[[sd], [sd]],
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        ecolor=color,
        capsize=3.5,
        elinewidth=1.2,
        markersize=markersize,
        zorder=4,
    )


def _legend_source_handles(sources, markers, markersize=8):
    return [
        Line2D(
            [0],
            [0],
            marker=markers[s],
            color="0.5",
            markerfacecolor="0.5",
            markeredgecolor="0.5",
            markersize=markersize,
            linestyle="none",
            label=lab,
        )
        for s, lab in sources
    ]


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------


def plot_figure2(ctx: dict[str, Any], fig_dir: Path) -> Path:
    ddf = ctx["ddf"]
    dfi = ctx["dfi_species"]
    nfi_g = ctx["nfi_greco_series"]
    gdf = ctx["greco_gdf"]
    gc_met = ctx["gc_met_tfv"]

    fig = plt.figure(figsize=(15.8, 7.8))
    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.08, 0.92],
        height_ratios=[1, 1],
        hspace=0.34,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    gs_map = gs[0, 2].subgridspec(2, 1, height_ratios=[2.6, 0.45], hspace=0.05)
    ax_map = fig.add_subplot(gs_map[0, 0])
    ax_map_leg = fig.add_subplot(gs_map[1, 0])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    gs_met_wrap = gs[:, 4].subgridspec(3, 1, height_ratios=[0.01, 1.0, 0.05], hspace=0)
    gs_met = gs_met_wrap[1, 0].subgridspec(3, 1, hspace=0.22)
    ax_met = [fig.add_subplot(gs_met[i, 0]) for i in range(3)]

    ylim = (0, 160)
    yticks = [0, 50, 100, 150]
    has_nfi_species = dfi is not None and not dfi.empty
    has_nfi_greco = nfi_g is not None and not nfi_g.empty
    legend_val = [Line2D([0], [0], color="gray", linewidth=1.0, label="RS")]
    if has_nfi_species:
        legend_val.append(Line2D([0], [0], color="black", linewidth=1.8, label="NFI"))
    for ax, (tfv, title) in zip([ax_a, ax_b], VALIDATION_SPECIES):
        _plot_all_greco_fitted_curves(ax, ddf, tfv, y_label_cap=150)
        if has_nfi_species:
            _plot_nfi_validation(ax, dfi, tfv)
        _style_validation_ax(ax, title, show_ylabel=(ax is ax_a), ylim=ylim, yticks=yticks)
        ax.legend(handles=legend_val, loc="upper left", fontsize=11, frameon=False)

    _plot_greco_map(ax_map, gdf)
    ax_map_leg.axis("off")
    ax_map_leg.legend(
        handles=[
            mpatches.Patch(facecolor=c, edgecolor="none", label=lab)
            for lab, c in GRECO_GROUP_COLORS.items()
        ],
        loc="center",
        fontsize=11,
        frameon=False,
        ncol=1,
    )

    legend_reg = [
        Line2D([0], [0], color=TFV_GROUP_COLORS[g], linewidth=1.0, label=g)
        for g in TFV_GROUP_ORDER
    ]
    if has_nfi_greco:
        legend_reg.append(Line2D([0], [0], color="black", linewidth=1.4, label="NFI"))
    for ax, (region, title) in zip([ax_d, ax_e, ax_f], REGIONAL_PANELS):
        for tfv in ddf["TFV"].unique():
            for _, row in ddf[ddf["TFV"] == tfv].iterrows():
                if GRECO_GROUPS.get(row["greco"]) != region:
                    continue
                grp = TFV_GROUP.get(float(tfv))
                color = TFV_GROUP_COLORS.get(grp)
                if color is None:
                    continue
                y = chapman_richards(
                    AGES,
                    row[f"a_{RS_MAIN_Q}"],
                    row[f"b_{RS_MAIN_Q}"],
                    row[f"c_{RS_MAIN_Q}"],
                    param_d(row, RS_MAIN_Q),
                ) * RS_CARBON
                ax.plot(AGES, y, color=color, linewidth=1.0, alpha=0.3)
        if has_nfi_greco:
            for greco in [g for g, grp in GRECO_GROUPS.items() if grp == region]:
                sub = nfi_g[nfi_g["GRECO"] == greco].sort_values("age")
                if sub.empty:
                    continue
                popt = fit_cr3(sub["age"].to_numpy(float), sub["agbd"].to_numpy(float))
                if popt is None:
                    continue
                ax.plot(AGES, cr_nfi(AGES, *popt), color="black", linewidth=1.4, zorder=10)
        ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=FS_T, zorder=12)
        ax.set_xlim(0, 125)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"], fontsize=FS_O)
        ax.set_yticks(yticks)
        ax.set_xlabel("Age (years)", fontsize=FS_O)
        ax.tick_params(axis="both", labelsize=FS_O)
        ax.set_box_aspect(1)
    ax_d.set_ylabel(r"AGB (MgC ha$^{-1}$)", fontsize=FS_O)
    ax_d.set_yticklabels([str(t) for t in yticks], fontsize=FS_O)
    ax_e.set_yticklabels([])
    ax_f.set_yticklabels([])

    fig.canvas.draw()
    pos_d, pos_f = ax_d.get_position(), ax_f.get_position()
    fig.legend(
        handles=legend_reg,
        loc="lower center",
        bbox_to_anchor=(0.5 * (pos_d.x0 + pos_f.x1), pos_d.y1 + 0.002),
        bbox_transform=fig.transFigure,
        ncol=5,
        fontsize=11,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    # Metrics g–i
    sources = ["RS GRECO-TFV"]
    if "NFI species" in set(gc_met["source"].dropna()):
        sources.append("NFI species")
    markers = {"RS GRECO-TFV": "D", "NFI species": "o"}
    param_specs = [
        ("agb_max", r"$AGB_{max}$ (MgC ha$^{-1}$)"),
        ("r30", r"$R_{30}$ (MgC ha$^{-1}$ yr$^{-1}$)"),
        ("t50", r"$t_{50}$ (years)"),
    ]
    tfv_order = [g for g in TFV_GROUP_ORDER if g in set(gc_met["tfv_group"].dropna())]
    met_stats: dict[str, list] = {p: [] for p, _ in param_specs}
    for grp in tfv_order:
        sub_g = gc_met[gc_met["tfv_group"] == grp]
        for src in sources:
            sub = sub_g[sub_g["source"] == src]
            if sub.empty:
                continue
            for param, _ in param_specs:
                mu, sd = area_weighted_mean_std(sub[param].values, sub["area"].values)
                if np.isfinite(mu):
                    met_stats[param].append((grp, src, mu, 0.0 if not np.isfinite(sd) else sd))

    for i, (ax, (param, ylab)) in enumerate(zip(ax_met, param_specs)):
        for xi, grp in enumerate(tfv_order):
            color = TFV_GROUP_COLORS.get(grp, "0.4")
            for si, src in enumerate(sources):
                pts = [t for t in met_stats[param] if t[0] == grp and t[1] == src]
                if not pts:
                    continue
                _, _, mu, sd = pts[0]
                xpos = xi + (si - (len(sources) - 1) / 2) * 0.18
                _dot_whisker(ax, xpos, mu, sd, color, marker=markers[src], markersize=7)
        ax.set_xticks(np.arange(len(tfv_order)))
        ax.set_xlim(-0.5, len(tfv_order) - 0.5)
        if i == len(param_specs) - 1:
            ax.set_xticklabels(
                [g.replace(" ", "\n", 1) if " " in g else g for g in tfv_order],
                fontsize=FS_O,
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="center", va="top", rotation_mode="anchor")
            ax.tick_params(axis="x", pad=12)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel(ylab, fontsize=FS_O)
        ax.tick_params(axis="y", labelsize=FS_O)
        extents = [v for _, _, mu, sd in met_stats[param] for v in (mu - sd, mu + sd) if np.isfinite(v)]
        if extents:
            data_lo, data_hi = min(extents), max(extents)
            span = max(data_hi - data_lo, 1e-9)
            lo, hi = data_lo - 0.08 * span, data_hi + 0.08 * span
            if data_lo >= 0:
                lo = max(lo, 0.0)
            lo, hi = min(lo, data_lo), max(hi, data_hi)
            ticks = np.asarray(MaxNLocator(nbins=4, min_n_ticks=3).tick_values(lo, hi), float)
            ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
            ax.set_ylim(lo, hi)
            ax.set_yticks(ticks)

    legend_met = [("RS GRECO-TFV", "RS")]
    if "NFI species" in sources:
        legend_met.append(("NFI species", "NFI"))
    ax_met[0].legend(
        handles=_legend_source_handles(legend_met, markers),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_met),
        fontsize=11,
        frameon=False,
    )

    # Panel letters
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    y_row1 = pos_a.y1
    for ax, lab, dx in ((ax_a, "a", 0.042), (ax_b, "b", 0.018), (ax_map, "c", 0.018)):
        p = ax.get_position()
        fig.text(p.x0 - dx, y_row1, lab, transform=fig.transFigure, fontsize=14, fontweight="bold", va="bottom")
    pos_d = ax_d.get_position()
    y_row2 = pos_d.y1 + 0.002
    for ax, lab, dx in ((ax_d, "d", 0.042), (ax_e, "e", 0.018), (ax_f, "f", 0.018)):
        p = ax.get_position()
        fig.text(p.x0 - dx, y_row2, lab, transform=fig.transFigure, fontsize=14, fontweight="bold", va="bottom")
    gap = y_row1 - ax_met[0].get_position().y1
    for ax, lab in zip(ax_met, "ghi"):
        p = ax.get_position()
        y = y_row1 if lab == "g" else p.y1 + gap
        fig.text(p.x0 - 0.018, y, lab, transform=fig.transFigure, fontsize=14, fontweight="bold", va="bottom")

    out = fig_dir / "Figure_2_agb_growth_species_regions_metrics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure SI1
# ---------------------------------------------------------------------------


def plot_figure_si1(ctx: dict[str, Any], fig_dir: Path) -> Path:
    ddf = ctx["ddf"].copy()
    area = ctx["area"]
    ddf["agb_max"] = pd.to_numeric(ddf[f"a_{RS_MAIN_Q}"], errors="coerce") * RS_CARBON
    ddf["t50"] = ddf.apply(
        lambda r: t50_from_abcd(
            r[f"a_{RS_MAIN_Q}"],
            r[f"b_{RS_MAIN_Q}"],
            r[f"c_{RS_MAIN_Q}"],
            param_d(r, RS_MAIN_Q),
        ),
        axis=1,
    )
    ddf["r30"] = ddf.apply(
        lambda r: chapman_richards(
            30,
            r[f"a_{RS_MAIN_Q}"],
            r[f"b_{RS_MAIN_Q}"],
            r[f"c_{RS_MAIN_Q}"],
            param_d(r, RS_MAIN_Q),
        )
        * RS_CARBON
        / 30.0,
        axis=1,
    )
    ddf["r2"] = pd.to_numeric(ddf.get("R2_50", ddf.get("R2_75")), errors="coerce")
    # Metrics: all GRECO with a curve (incl. gap-filled). Weights = mask area of those GRECO.
    merged = ddf.merge(area, on=["TFV", "greco"], how="inner")
    # Order / bubble size: full TFV mask area (all GRECO, not restricted to fitted rows).
    area_tfv = (
        area.groupby("TFV", observed=True)["area"]
        .sum()
        .rename("area_full")
        .reset_index()
    )

    rows = []
    for tfv, grp in merged.groupby("TFV", observed=True):
        tfv_group = TFV_GROUP.get(float(tfv))
        if tfv_group in (None, "Other", "Forest"):
            continue
        w = grp["area"].to_numpy(float)
        if np.nansum(w) <= 0:
            continue
        full = float(area_tfv.loc[area_tfv["TFV"] == int(tfv), "area_full"].sum())
        rec = {
            "TFV": int(tfv),
            "tfv_group": tfv_group,
            "area": full if full > 0 else float(np.nansum(w)),
        }
        for col in ("agb_max", "t50", "r30", "r2"):
            vals = grp[col].to_numpy(float)
            ok = np.isfinite(vals) & np.isfinite(w) & (w > 0)
            if not ok.any():
                rec[f"{col}_center"] = np.nan
                rec[f"{col}_sd"] = np.nan
                continue
            # Dot = area-weighted mean; bar = ± area-weighted SD across GRECO.
            mu, sd = area_weighted_mean_std(vals[ok], w[ok])
            rec[f"{col}_center"] = mu
            rec[f"{col}_sd"] = 0.0 if not np.isfinite(sd) else float(sd)
        rows.append(rec)
    stats = pd.DataFrame(rows).sort_values("area", ascending=True)
    if stats.empty:
        raise RuntimeError("SI1: no TFV stats to plot")

    n = len(stats)
    fig = plt.figure(figsize=(14, max(8, 0.32 * n)))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    # (key, xlabel, lo_floor, panel) — xlim covers full mean±SD; ticks via MaxNLocator.
    cfg = [
        ("agb_max", r"$AGB_{max}$ (MgC ha$^{-1}$)", 0.0, "a"),
        ("t50", r"$t_{50}$ (years)", 0.0, "b"),
        ("r30", r"$R_{30}$ (MgC ha$^{-1}$ yr$^{-1}$)", 0.0, "c"),
        ("r2", r"$R^2$", None, "d"),
    ]
    tfv_order = stats["TFV"].tolist()
    y_map = {t: i for i, t in enumerate(tfv_order)}
    area_max = float(stats["area"].max())
    bubble = lambda a: 50.0 + 1000.0 * (a / area_max) ** 0.7
    y_pad = 0.5

    for i, (ax, (pkey, xlab, lo_floor, plab)) in enumerate(zip(axes, cfg)):
        centers = stats[f"{pkey}_center"].to_numpy(float)
        sds = stats[f"{pkey}_sd"].to_numpy(float)
        data_lo = float(np.nanmin(centers - sds))
        data_hi = float(np.nanmax(centers + sds))
        span = max(data_hi - data_lo, 1e-9)
        lo = data_lo - 0.06 * span
        hi = data_hi + 0.06 * span
        if lo_floor is not None and data_lo >= lo_floor:
            lo = max(lo, lo_floor)
        # Keep full whiskers so the mean stays centered on the bar.
        lo = min(lo, data_lo)
        hi = max(hi, data_hi)
        if hi <= lo:
            hi = lo + span
        ticks = np.asarray(MaxNLocator(nbins=4, min_n_ticks=3).tick_values(lo, hi), float)
        ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
        for row in stats.itertuples():
            center = getattr(row, f"{pkey}_center")
            sd = getattr(row, f"{pkey}_sd")
            if not (np.isfinite(center) and np.isfinite(sd)):
                continue
            y = y_map[row.TFV]
            c = TFV_GROUP_COLORS.get(row.tfv_group, "0.5")
            ax.plot([center - sd, center + sd], [y, y], color=c, linewidth=2.4, zorder=2, solid_capstyle="round")
            ax.scatter(center, y, s=bubble(row.area), c=c, alpha=BUBBLE_ALPHA, edgecolors="none", zorder=3, clip_on=False)
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlab, fontsize=FS_O)
        ax.set_yticks(np.arange(len(tfv_order)))
        if i == 0:
            ax.set_yticklabels(
                [DICT_GROUP_EN.get(float(t), str(int(t))) for t in tfv_order],
                fontsize=FS_O,
                va="center",
            )
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_ylim(-y_pad, len(tfv_order) - 1 + y_pad)
        ax.tick_params(axis="x", labelsize=FS_O)
        ax.text(-0.22 if i == 0 else -0.08, 1.10, plab, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    area_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.6",
            markeredgecolor="none",
            alpha=BUBBLE_ALPHA,
            markersize=np.sqrt(bubble(a)) / 3.0,
            label=f"{a:g} Mha",
            linestyle="none",
        )
        for a in (0.5, 1.0, 5.0)
    ]
    group_handles = [
        Line2D([0], [0], color=c, linewidth=2.4, label=g)
        for g, c in TFV_GROUP_COLORS.items()
        if g in set(stats["tfv_group"])
    ]
    leg = axes[-1].legend(handles=area_handles, title="Area", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=FS_O, title_fontsize=FS_O, frameon=False)
    fig.add_artist(leg)
    axes[-1].legend(handles=group_handles, title="Composition", loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=FS_O, title_fontsize=FS_O, frameon=False)

    out = fig_dir / "Figure_SI1_growth_metrics_by_species_class.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure SI2
# ---------------------------------------------------------------------------


def plot_figure_si2(ctx: dict[str, Any], fig_dir: Path) -> Path:
    area = ctx["area"]
    rs = ctx["rs_greco_tfv"].copy()
    nfi = ctx["nfi_greco_metrics"].copy() if ctx.get("nfi_greco_metrics") is not None else pd.DataFrame()
    rob = ctx["robinson_metrics"].copy() if ctx.get("robinson_metrics") is not None else pd.DataFrame()

    rs = rs.merge(area, on=["greco", "TFV"], how="left")
    rs["area"] = pd.to_numeric(rs["area"], errors="coerce").fillna(0.0)
    # Collapse TFV×GRECO → GRECO (area-weighted)
    rows = []
    for g, sub in rs.groupby("greco", observed=True):
        ww = sub["area"].to_numpy(float)
        if ww.sum() <= 0:
            ww = np.ones(len(sub), float)
        rec = {
            "source": "RS",
            "greco": g,
            "greco_group": GRECO_GROUPS.get(g, np.nan),
            "area": float(ww.sum()),
        }
        for p in ("agb_max", "r30", "t50"):
            v = sub[p].to_numpy(float)
            ok = np.isfinite(v) & (ww > 0)
            rec[p] = float(np.average(v[ok], weights=ww[ok])) if ok.any() else np.nan
        rows.append(rec)
    rs_g = pd.DataFrame(rows)

    greco_area = area.groupby("greco", observed=True)["area"].sum().reset_index()
    if not nfi.empty and "greco" in nfi.columns:
        nfi = nfi.merge(greco_area, on="greco", how="left")
        nfi["area"] = pd.to_numeric(nfi["area"], errors="coerce").fillna(0.0)
        nfi["source"] = "NFI"
    else:
        nfi = pd.DataFrame()
    if not rob.empty and "greco" in rob.columns:
        rob = rob.merge(greco_area, on="greco", how="left")
        rob["area"] = pd.to_numeric(rob["area"], errors="coerce").fillna(0.0)
    else:
        rob = pd.DataFrame()

    plot = {"RS": rs_g}
    if not nfi.empty:
        plot["NFI"] = nfi
    if not rob.empty:
        plot["Robinson"] = rob
    markers = {"RS": "D", "NFI": "o", "Robinson": "s"}
    n_src = len(plot)
    # Center sources around 0 with even spacing
    src_list = list(plot.keys())
    offsets = {
        s: (i - (n_src - 1) / 2) * 0.18 for i, s in enumerate(src_list)
    }
    groups = [
        g
        for g in GRECO_GROUP_ORDER
        if any(g in set(df["greco_group"].dropna()) for df in plot.values() if not df.empty)
    ]
    specs = [
        ("agb_max", r"$AGB_{max}$ (MgC ha$^{-1}$)"),
        ("r30", r"$R_{30}$ (MgC ha$^{-1}$ yr$^{-1}$)"),
        ("t50", r"$t_{50}$ (years)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (param, ylab) in zip(axes, specs):
        extents: list[float] = []
        for xi, grp in enumerate(groups):
            color = GRECO_GROUP_COLORS.get(grp, "0.4")
            for src in src_list:
                df = plot[src]
                if df.empty:
                    continue
                sub = df.loc[df["greco_group"] == grp, [param, "area"]].copy()
                sub[param] = pd.to_numeric(sub[param], errors="coerce")
                sub["area"] = pd.to_numeric(sub["area"], errors="coerce").fillna(0.0)
                sub = sub.dropna(subset=[param])
                if sub.empty:
                    continue
                # Dot = area-weighted mean; whisker = ± area-weighted SD across GRECO.
                mu, sd = area_weighted_mean_std(sub[param].to_numpy(float), sub["area"].to_numpy(float))
                if not np.isfinite(mu):
                    continue
                sd = 0.0 if not np.isfinite(sd) else float(sd)
                extents.extend([mu - sd, mu + sd])
                _dot_whisker(
                    ax,
                    xi + offsets[src],
                    mu,
                    sd,
                    color,
                    marker=markers[src],
                    markersize=7,
                )
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([g.replace(" ", "\n") if " " in g else g for g in groups], fontsize=FS_O - 1)
        ax.set_ylabel(ylab, fontsize=FS_O)
        ax.tick_params(axis="y", labelsize=FS_O)
        if extents:
            data_lo, data_hi = min(extents), max(extents)
            span = max(data_hi - data_lo, 1e-9)
            lo, hi = data_lo - 0.08 * span, data_hi + 0.08 * span
            if data_lo >= 0:
                lo = max(lo, 0.0)
            lo, hi = min(lo, data_lo), max(hi, data_hi)
            ticks = np.asarray(MaxNLocator(nbins=4, min_n_ticks=3).tick_values(lo, hi), float)
            ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
            ax.set_ylim(lo, hi)
            ax.set_yticks(ticks)

    legend_src = [("RS", "RS")]
    if "NFI" in plot:
        legend_src.append(("NFI", "NFI"))
    if "Robinson" in plot:
        legend_src.append(("Robinson", r"Robinson $\it{et\,al.}$"))
    fig.legend(
        handles=_legend_source_handles(legend_src, markers),
        loc="upper center",
        ncol=len(legend_src),
        frameon=False,
        fontsize=FS_O,
        bbox_to_anchor=(0.5, 1.10),
    )
    fig.tight_layout(w_pad=2.2)
    out = fig_dir / "Figure_SI2_growth_metrics_by_ecological_region.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _build_tfv_metrics(ddf, area, nfi_species) -> pd.DataFrame:
    rs = metrics_from_rs(ddf, "RS GRECO-TFV")
    rs = rs.loc[rs["tfv_group"].isin(TFV_GROUP_ORDER)].copy()
    rs = rs.merge(area, on=["greco", "TFV"], how="left")
    rs["area"] = pd.to_numeric(rs["area"], errors="coerce").fillna(0.0)
    # Collapse to TFV-level for Fig. 2 g–i
    rows = []
    for tfv, sub in rs.groupby("TFV", observed=True):
        ww = sub["area"].to_numpy(float)
        if ww.sum() <= 0:
            ww = np.ones(len(sub), float)
        rec = {
            "source": "RS GRECO-TFV",
            "TFV": float(tfv),
            "tfv_group": sub["tfv_group"].iloc[0],
            "area": float(ww.sum()),
        }
        for p in ("agb_max", "r30", "t50"):
            v = sub[p].to_numpy(float)
            ok = np.isfinite(v) & (ww > 0)
            rec[p] = float(np.average(v[ok], weights=ww[ok])) if ok.any() else np.nan
        rows.append(rec)
    rs_tfv = pd.DataFrame(rows)
    if nfi_species is None or nfi_species.empty:
        return rs_tfv
    nfi = metrics_from_nfi_species(nfi_species)
    if nfi.empty:
        return rs_tfv
    nfi = nfi.loc[nfi["tfv_group"].isin(TFV_GROUP_ORDER)].copy()
    tfv_area = area.groupby("TFV", observed=True)["area"].sum().reset_index()
    nfi = nfi.merge(tfv_area, on="TFV", how="left")
    nfi["area"] = pd.to_numeric(nfi["area"], errors="coerce").fillna(0.0)
    return pd.concat([rs_tfv, nfi], ignore_index=True)


def run_figures(cfg: dict[str, Any], figure_dir: Path | None = None, **_kw) -> Path:
    """Load inputs and write Fig. 2, SI1, SI2 PNGs."""
    if figure_dir is not None:
        fig_dir = Path(figure_dir).expanduser()
    elif cfg.get("figure_dir"):
        fig_dir = Path(cfg["figure_dir"]).expanduser()
    else:
        raise KeyError("figure_dir missing (set in config or pass figure_dir=)")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig. 2: local fits only. SI1/SI2: include gap-filled curves for full coverage.
    # NFI / Robinson paths are optional — figures still run RS-only if unset/missing.
    ddf_local = _ensure_d(load_rs_curves(cfg, include_gap_filled=False))
    ddf_si = _ensure_d(load_rs_curves(cfg, include_gap_filled=True))
    area = load_area_tfv_greco(cfg)
    gdf = load_greco_gdf(cfg)
    dfi = load_nfi_species_series(cfg)
    nfi_g_series = load_nfi_greco_series(cfg)
    nfi_sp, nfi_greco = load_nfi_fits(cfg)
    cp2 = load_robinson(cfg)

    if dfi.empty and nfi_g_series.empty and nfi_sp.empty and nfi_greco.empty:
        print("  Note: NFI inputs missing — Fig. 2/SI2 without NFI overlays")
    if cp2.empty:
        print("  Note: Robinson/CP2 missing — SI2 without Robinson")

    nfi_greco_met = (
        metrics_from_nfi_greco(nfi_greco) if not nfi_greco.empty else pd.DataFrame()
    )
    robinson_met = metrics_from_robinson(cp2) if not cp2.empty else pd.DataFrame()

    ctx_main = {
        "ddf": ddf_local,
        "area": area,
        "greco_gdf": gdf,
        "dfi_species": dfi,
        "nfi_greco_series": nfi_g_series,
        "gc_met_tfv": _build_tfv_metrics(ddf_local, area, nfi_sp),
        "rs_greco_tfv": metrics_from_rs(ddf_local, "RS GRECO-TFV"),
        "nfi_greco_metrics": nfi_greco_met,
        "robinson_metrics": robinson_met,
    }
    ctx_si = {
        **ctx_main,
        "ddf": ddf_si,
        "gc_met_tfv": _build_tfv_metrics(ddf_si, area, nfi_sp),
        "rs_greco_tfv": metrics_from_rs(ddf_si, "RS GRECO-TFV"),
    }

    written = []
    for name, fn, ctx in (
        ("Figure 2", plot_figure2, ctx_main),
        ("Figure SI1", plot_figure_si1, ctx_si),
        ("Figure SI2", plot_figure_si2, ctx_si),
    ):
        path = fn(ctx, fig_dir)
        print(f"  {name}: {path}")
        written.append(path)
    return fig_dir
