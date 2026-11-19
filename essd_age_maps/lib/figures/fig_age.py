"""Paper figures: Fig. 3, SI3–SI6 (adapted from Figures/Fig_age_distribution.py)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from lib.figures.constants import (
    BUBBLE_ALPHA,
    ESS_TO_MATCH,
    FS_O,
    FS_T,
    GAMI_FILL,
    GAMI_LINE,
    GRECO_DX,
    GRECO_GROUP_COLORS,
    GRECO_GROUP_ORDER,
    MATCH_LABEL,
    MATCH_ORDER,
    MATCH_TO_TFVS,
    NFI_EDGE,
    RS_FILL,
    RS_LINE,
    TFV_GROUP,
    TFV_GROUP_COLORS,
    TFV_LABEL,
    TFV_GROUP_ORDER,
)
from lib.figures.data import annotate_groups, calc_stats, load_figure_context


def _add_panel_label(ax, label, x=-0.08, y=1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")


def _auto_xlim(stats, cols=("q25", "q75"), pad=0.05, step=10):
    lo = min(stats[c].min() for c in cols if c in stats)
    hi = max(stats[c].max() for c in cols if c in stats)
    rng = hi - lo if hi > lo else step
    lo = np.floor((lo - pad * rng) / step) * step
    hi = np.ceil((hi + pad * rng) / step) * step
    return float(lo), float(hi)


def _age_cmap(vmax: float):
    def adjust_color(color, amount=0.5):
        c = np.array(to_rgba(color))
        c = 1 - (1 - c) * (1 - amount) if amount > 0 else c * (1 + amount)
        c[3] = 1
        return np.clip(c, 0, 1)

    base = ["mediumpurple", "skyblue", "yellowgreen", "gold", "indianred"]
    colors = []
    for color in base:
        colors.extend([adjust_color(color, 0.5), color, adjust_color(color, -0.5)])
    bounds = [0, 30, 50, 80, 150, vmax]
    positions = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i] / vmax, bounds[i + 1] / vmax
        positions.extend([start, (start + end) / 2, end])
    positions[-1] = 1.0
    cmap = LinearSegmentedColormap.from_list("map_grad", list(zip(positions, colors)))
    cmap.set_bad((0, 0, 0, 0))
    cbar_pos = [i * 0.2 + j * 0.1 for i in range(5) for j in [0, 1, 2]]
    cbar_cmap = LinearSegmentedColormap.from_list("cbar_grad", list(zip(cbar_pos, colors)))
    return cmap, cbar_cmap


def _weighted_age_kde_curve(ages, weights, age_grid, bw_yr=5.0):
    ages = np.asarray(ages, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(ages) & np.isfinite(weights) & (weights > 0)
    ages, weights = ages[valid], weights[valid]
    if len(ages) < 2:
        return np.zeros_like(age_grid, dtype=float)
    kde = gaussian_kde(ages, weights=weights)
    data_std = float(np.sqrt(kde._data_covariance[0, 0]))
    if data_std > 0 and bw_yr > 0:
        kde.set_bandwidth(bw_method=float(bw_yr) / data_std)
    return kde(age_grid) * weights.sum()


def _hist_ages_areas(df, min_age=0):
    if df is None or df.empty:
        return np.array([]), np.array([])
    g = df.loc[df["age"] >= min_age].groupby("age", observed=True)["area"].sum().reset_index()
    return g["age"].to_numpy(dtype=float), g["area"].to_numpy(dtype=float)


def _kde_envelope(df_lo, df_main, df_hi, age_grid, min_age=0, bw=5.0):
    lo = _weighted_age_kde_curve(*_hist_ages_areas(df_lo, min_age), age_grid, bw_yr=bw)
    main = _weighted_age_kde_curve(*_hist_ages_areas(df_main, min_age), age_grid, bw_yr=bw)
    hi = _weighted_age_kde_curve(*_hist_ages_areas(df_hi, min_age), age_grid, bw_yr=bw)
    stack = np.stack([lo, main, hi], axis=0)
    return stack.min(axis=0), main, stack.max(axis=0)


def _inset_cbar(fig, ax, mappable, title, tick_vals, tick_labs, fs_o=FS_O):
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes(
        [
            pos.x0 + 0.88 * pos.width,
            pos.y0 + 0.3 * pos.height,
            0.025 * pos.width,
            0.55 * pos.height,
        ]
    )
    cax.patch.set_facecolor("white")
    cax.patch.set_alpha(0.8)
    cbar = plt.colorbar(mappable, cax=cax, orientation="vertical", ticks=tick_vals)
    cbar.ax.tick_params(labelsize=fs_o)
    cbar.ax.set_yticklabels(tick_labs, fontsize=fs_o)
    cbar.ax.set_title(title, fontsize=FS_T, pad=6)
    return cax


# ---------------------------------------------------------------------------
# Fig. 3
# ---------------------------------------------------------------------------

def _plot_rs_gami_nfi(ax, ctx, rs_ymax=0.3):
    age_grid = np.linspace(0, 220, 400)
    rs_lo, rs_main, rs_hi = _kde_envelope(ctx["rs_lo"], ctx["rs_main"], ctx["rs_hi"], age_grid)
    g_lo, g_main, g_hi = _kde_envelope(
        ctx["gami_lo"], ctx["gami_mean"], ctx["gami_hi"], age_grid, min_age=1, bw=5.0
    )
    ax.fill_between(age_grid, rs_lo, rs_hi, color=RS_FILL, alpha=0.22, linewidth=0, zorder=1)
    ax.plot(age_grid, rs_main, color=RS_LINE, linewidth=1.4, zorder=2)
    ax.fill_between(age_grid, g_lo, g_hi, color=GAMI_FILL, alpha=0.22, linewidth=0, zorder=1)
    ax.plot(age_grid, g_main, color=GAMI_LINE, linewidth=1.4, zorder=2)

    nfi = ctx.get("nfi_forest")
    has_nfi = nfi is not None and not nfi.empty
    nfi_y = None
    if has_nfi:
        bw = 10.0
        a = nfi["age"].to_numpy(dtype=float)
        w = nfi["area"].to_numpy(dtype=float)
        s = nfi["area_err"].to_numpy(dtype=float)
        valid = (a > 0) & (w > 0)
        a, w, s = a[valid], w[valid], s[valid]
        K = np.exp(-0.5 * ((age_grid[:, None] - a[None, :]) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        nfi_y = K @ w
        nfi_std = np.sqrt(K**2 @ s**2)
        ax.fill_between(
            age_grid, np.maximum(0, nfi_y - nfi_std), nfi_y + nfi_std, color=NFI_EDGE, alpha=0.18, linewidth=0
        )
        ax.plot(age_grid, nfi_y, color=NFI_EDGE, linewidth=1.4)

    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Age (years)", fontsize=FS_O)
    ax.set_ylabel(r"Area (Mha yr$^{-1}$)", fontsize=FS_O)
    ticks = np.arange(0, rs_ymax + 0.1, 0.1)
    ax.set_ylim(0, ticks[-1])
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=FS_O)
    handles = [
        Line2D([0], [0], color=RS_LINE, linewidth=1.4, label="Ours"),
        Line2D([0], [0], color=GAMI_LINE, linewidth=1.4, label="GAMI"),
    ]
    if has_nfi:
        handles.append(Line2D([0], [0], color=NFI_EDGE, linewidth=1.4, label="NFI"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=FS_O)
    bits = [
        f"Ours: {float(np.trapz(rs_main, age_grid)):.2f} Mha",
        f"GAMI: {float(np.trapz(g_main, age_grid)):.2f} Mha",
    ]
    if nfi_y is not None:
        bits.append(f"NFI: {float(np.trapz(nfi_y, age_grid)):.2f} Mha")
    print("KDE integrals — " + " | ".join(bits))


def _plot_age_map(fig, ax, ctx):
    """Match ``Fig_age_distribution._plot_fig2s_age_map``."""
    masked, bounds = ctx["masked_age"], ctx["age_bounds"]
    vmax = float(np.max(masked))
    cmap, cbar_cmap = _age_cmap(vmax)
    ctx["france_gdf"].plot(ax=ax, color="white", edgecolor="none", zorder=1)
    ax.imshow(
        masked,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="none",
        aspect="equal",
        zorder=2,
    )
    ctx["greco_gdf"].plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _inset_cbar(
        fig,
        ax,
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cbar_cmap),
        "Age",
        np.linspace(0, 1, 6),
        ["0 yrs", "30 yrs", "50 yrs", "80 yrs", "150 yrs", f"{int(vmax)} yrs"],
    )


def _tfv_greco_stats(rs_main: pd.DataFrame) -> pd.DataFrame:
    d = annotate_groups(rs_main)
    d = d[(d["age"] >= 0) & d["tfv_group"].notna() & d["greco_group"].notna()]
    d = d[~d["tfv_group"].isin(["Other", "Forest"])]
    return (
        d.groupby(["tfv_group", "greco_group"], observed=True)
        .apply(calc_stats, include_groups=False)
        .reset_index()
    )


def _plot_tfv_greco_bubbles(ax, stats_df, xlim=None, xticks=None, show_legend=True):
    stats = stats_df[~stats_df["tfv_group"].isin(["Other", "Forest"])].copy()
    tfv_order = [t for t in TFV_GROUP_ORDER if t in stats["tfv_group"].unique()][::-1]
    y_map = {t: i for i, t in enumerate(tfv_order)}
    area_max = float(stats["area"].max()) if len(stats) else 1.0
    bubble = lambda a: 50.0 + 1000.0 * (a / area_max) ** 0.7
    if xlim is None:
        xlim = _auto_xlim(stats)
        xlim = (max(0.0, xlim[0]), xlim[1])
    if xticks is None:
        xticks = list(range(int(xlim[0]), int(xlim[1]) + 1, 50))

    for row in stats.itertuples():
        if row.tfv_group not in y_map or row.greco_group not in GRECO_DX:
            continue
        y = y_map[row.tfv_group] + GRECO_DX[row.greco_group]
        c = GRECO_GROUP_COLORS[row.greco_group]
        ax.plot([row.q25, row.q75], [y, y], color=c, linewidth=2.4, zorder=2)
        ax.scatter(row.q50, y, s=bubble(row.area), c=c, alpha=BUBBLE_ALPHA, edgecolors="none", zorder=3)

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks], fontsize=FS_O)
    ax.set_xlabel("Age (years)", fontsize=FS_O)
    ax.set_yticks(range(len(tfv_order)))
    ax.set_yticklabels([t.replace(" ", "\n", 1) for t in tfv_order], fontsize=FS_O)
    ax.set_ylim(-0.6, max(len(tfv_order) - 0.4, 0.4))
    if show_legend:
        leg_kw = dict(fontsize=FS_O, title_fontsize=FS_O, frameon=False)
        area_h = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.6",
                markeredgecolor="none",
                markersize=np.sqrt(bubble(a)) / 3.0,
                label=f"{a:.0f} Mha",
                alpha=BUBBLE_ALPHA,
            )
            for a in (1.0, 5.0, 20.0)
        ]
        reg_h = [
            Line2D([0], [0], color=GRECO_GROUP_COLORS[g], linewidth=2.4, label=g)
            for g in GRECO_GROUP_ORDER
            if g in stats["greco_group"].unique()
        ]
        leg1 = ax.legend(
            handles=area_h,
            title="Area",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            **leg_kw,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=reg_h,
            title="Region",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0,
            **leg_kw,
        )


def plot_figure3(ctx: dict[str, Any], fig_dir: Path) -> Path:
    """Fig. 3 layout from ``Fig_age_distribution`` (fig2 simplified / Age_hist_figure3_paper)."""
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.4, "width_ratios": [1, 1.5, 1]}
    )
    _plot_rs_gami_nfi(ax_a, ctx)
    cax = _plot_age_map(fig, ax_b, ctx)
    stats = _tfv_greco_stats(ctx["rs_main"])
    _plot_tfv_greco_bubbles(ax_c, stats)
    for ax in (ax_a, ax_c):
        ax.set_box_aspect(1)

    # Reposition: keep width_ratios widths, align tops, extra gap b→c (original script).
    fig.canvas.draw()
    pa, pb, pc = [ax.get_position() for ax in (ax_a, ax_b, ax_c)]
    wa, wb, wc = pa.width, pb.width, pc.width
    gap_ab, gap_bc = 0.04, 0.09
    x0 = pa.x0
    h = max(pa.height, pb.height, pc.height)
    top = max(pa.y0 + pa.height, pb.y0 + pb.height, pc.y0 + pc.height)
    y = top - h
    map_dx, map_dy = -0.02, -0.04
    ax_a.set_position([x0, y, wa, h])
    ax_b.set_position([x0 + wa + gap_ab + map_dx, y + map_dy, wb, h])
    ax_c.set_position([x0 + wa + wb + gap_ab + gap_bc, y, wc, h])
    fig.canvas.draw()
    bpos = ax_b.get_position()
    if cax is not None:
        cax.set_position(
            [
                bpos.x0 + 0.88 * bpos.width,
                bpos.y0 + 0.3 * bpos.height,
                0.025 * bpos.width,
                0.55 * bpos.height,
            ]
        )

    # Panel labels at a shared figure-y (converted to each axes fraction).
    fig.canvas.draw()
    label_y_fig = max(ax.get_position().y1 for ax in (ax_a, ax_b, ax_c)) + 0.01
    for ax, lbl, lx in zip((ax_a, ax_b, ax_c), "abc", (-0.22, -0.04, -0.22)):
        pos = ax.get_position()
        ly = (label_y_fig - pos.y0) / pos.height
        ax.text(
            lx,
            ly,
            lbl,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    out = fig_dir / "Figure_3_forest_age_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# SI3 — unc / CV / Ours−GAMI maps
# ---------------------------------------------------------------------------

def _plot_unc_map(fig, ax, masked, bounds, greco_gdf, france_gdf):
    france_gdf.plot(ax=ax, color="white", edgecolor="none", zorder=1)
    finite = masked.compressed() if np.ma.isMaskedArray(masked) else masked[np.isfinite(masked)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 100.0
    vmax = max(10.0, math.ceil(vmax / 10.0) * 10.0)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0, 0, 0, 0))
    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="none",
        aspect="equal",
        zorder=2,
    )
    greco_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _inset_cbar(fig, ax, im, "Uncertainty", [0, vmax], ["0%", f"{int(vmax)}%"])


def _plot_cv_map(fig, ax, masked, bounds, greco_gdf, france_gdf):
    france_gdf.plot(ax=ax, color="white", edgecolor="none", zorder=1)
    finite = masked.compressed() if np.ma.isMaskedArray(masked) else masked[np.isfinite(masked)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 100.0
    vmax = max(10.0, min(200.0, math.ceil(vmax / 10.0) * 10.0))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0, 0, 0, 0))
    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="none",
        aspect="equal",
        zorder=2,
    )
    greco_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _inset_cbar(fig, ax, im, "CV", [0, vmax], ["0%", f"{int(vmax)}%"])


def _plot_diff_map(fig, ax, masked, bounds, greco_gdf, france_gdf):
    france_gdf.plot(ax=ax, color="white", edgecolor="none", zorder=1)
    finite = masked.compressed() if np.ma.isMaskedArray(masked) else masked[np.isfinite(masked)]
    if finite.size:
        p_lo, p_hi = np.nanpercentile(finite, [2, 98])
    else:
        p_lo, p_hi = -50.0, 50.0
    vmin = min(math.floor(float(p_lo) / 10.0) * 10.0, -10.0)
    vmax = max(math.ceil(float(p_hi) / 10.0) * 10.0, 10.0)
    cmap = plt.cm.Spectral.copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="none",
        aspect="equal",
        zorder=2,
    )
    greco_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _inset_cbar(
        fig, ax, im, "Ours − GAMI", [vmin, 0, vmax], [f"{vmin:.0f} yr", "0", f"+{vmax:.0f} yr"]
    )


def plot_figure_si3(ctx: dict[str, Any], fig_dir: Path) -> Path | None:
    if "masked_cv" not in ctx or "masked_diff" not in ctx:
        print("  Skip SI3: set figures.age_cv_5km and figures.ours_minus_gami_5km")
        return None
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.30}
    )
    _plot_unc_map(fig, ax_a, ctx["masked_unc"], ctx["unc_bounds"], ctx["greco_gdf"], ctx["france_gdf"])
    _plot_cv_map(fig, ax_b, ctx["masked_cv"], ctx["cv_bounds"], ctx["greco_gdf"], ctx["france_gdf"])
    _plot_diff_map(fig, ax_c, ctx["masked_diff"], ctx["diff_bounds"], ctx["greco_gdf"], ctx["france_gdf"])
    for ax, lbl in zip((ax_a, ax_b, ax_c), "abc"):
        _add_panel_label(ax, lbl, x=-0.08)
    out = fig_dir / "Figure_SI3_age_uncertainty_cv_gami_diff.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# SI4 / SI5 — KDEs per species / GRECO
# ---------------------------------------------------------------------------

def _nfi_kde(ages, weights, weight_errs, age_grid, bw_yr=20.0):
    ages = np.asarray(ages, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_errs = np.asarray(weight_errs, dtype=float)
    valid = np.isfinite(ages) & (ages > 0) & np.isfinite(weights) & (weights > 0)
    a, w = ages[valid], weights[valid]
    s = weight_errs[valid] if len(weight_errs) == len(ages) else np.zeros_like(w)
    s = np.where(np.isfinite(s), s, 0.0)
    if len(a) == 0:
        z = np.zeros_like(age_grid)
        return z, z, z
    K = np.exp(-0.5 * ((age_grid[:, None] - a[None, :]) / bw_yr) ** 2) / (bw_yr * np.sqrt(2 * np.pi))
    y = K @ w
    std = np.sqrt(K**2 @ s**2)
    return np.maximum(0.0, y - std), y, y + std


def _plot_kde_by_class(
    nfi_df,
    class_col,
    title_map,
    out_path: Path,
    rs_getter,
    gami_getter=None,
    class_order=None,
    ncols=4,
    age_max=200,
):
    if class_order is not None:
        classes = [c for c in class_order if (nfi_df[class_col] == c).any()]
    else:
        classes = sorted(nfi_df[class_col].dropna().unique().tolist())
    n = len(classes)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    age_grid = np.linspace(0, age_max, 300)
    legend_handles = [
        Line2D([0], [0], color=RS_LINE, linewidth=1.4, label="Ours"),
        Line2D([0], [0], color=GAMI_LINE, linewidth=1.4, label="GAMI"),
        Line2D([0], [0], color=NFI_EDGE, linewidth=1.4, label="NFI"),
    ]
    for i, cls in enumerate(classes):
        ax = axes[i // ncols, i % ncols]
        sub = nfi_df.loc[nfi_df[class_col] == cls]
        nfi_lo, nfi_y, nfi_hi = _nfi_kde(
            sub["age"].values, sub["area"].values, sub["area_err"].values, age_grid
        )
        nfi_area = float(sub["area"].sum())
        ymax = float(np.nanmax(nfi_hi)) if len(nfi_hi) else 0.0

        rs_lo, rs_main, rs_hi = rs_getter(cls)
        rs_area = float(rs_main["area"].sum()) if rs_main is not None and len(rs_main) else 0.0
        kde_lo = _weighted_age_kde_curve(*_hist_ages_areas(rs_lo), age_grid)
        kde_main = _weighted_age_kde_curve(*_hist_ages_areas(rs_main), age_grid)
        kde_hi = _weighted_age_kde_curve(*_hist_ages_areas(rs_hi), age_grid)
        lo = np.minimum(kde_lo, np.minimum(kde_main, kde_hi))
        hi = np.maximum(kde_lo, np.maximum(kde_main, kde_hi))
        ax.fill_between(age_grid, lo, hi, color=RS_FILL, alpha=0.22, linewidth=0)
        ax.plot(age_grid, kde_main, color=RS_LINE, linewidth=1.4)
        ymax = max(ymax, float(np.nanmax(hi)) if len(hi) else 0.0)

        gami_area = None
        if gami_getter is not None:
            g_lo, g_mean, g_hi = gami_getter(cls)
            gami_area = float(g_mean["area"].sum()) if g_mean is not None and len(g_mean) else 0.0
            glo = _weighted_age_kde_curve(*_hist_ages_areas(g_lo, min_age=1), age_grid, bw_yr=5.0)
            gmean = _weighted_age_kde_curve(*_hist_ages_areas(g_mean, min_age=1), age_grid, bw_yr=5.0)
            ghi = _weighted_age_kde_curve(*_hist_ages_areas(g_hi, min_age=1), age_grid, bw_yr=5.0)
            ax.fill_between(
                age_grid,
                np.minimum(glo, np.minimum(gmean, ghi)),
                np.maximum(glo, np.maximum(gmean, ghi)),
                color=GAMI_FILL,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(age_grid, gmean, color=GAMI_LINE, linewidth=1.4)
            ymax = max(ymax, float(np.nanmax(ghi)) if len(ghi) else 0.0)

        ax.fill_between(age_grid, nfi_lo, nfi_hi, color=NFI_EDGE, alpha=0.18, linewidth=0)
        ax.plot(age_grid, nfi_y, color=NFI_EDGE, linewidth=1.5)
        ax.set_xlim(0, age_max)
        ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)
        ax.set_title(title_map.get(cls, str(cls)), fontsize=9, pad=2)
        bits = [f"NFI={nfi_area:.2f}", f"RS={rs_area:.2f}"]
        if gami_area is not None:
            bits.append(f"GAMI={gami_area:.2f}")
        ax.text(0.98, 0.95, "  ".join(bits) + " Mha", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.25")
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel(r"Area (Mha yr$^{-1}$)", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Age (years)", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_figure_si4(ctx: dict[str, Any], fig_dir: Path) -> Path | None:
    nfi = ctx.get("nfi_ess")
    if nfi is None or nfi.empty:
        print("  Skip SI4: set figures.nfi_surf_esspre")
        return None
    nfi = nfi.copy()
    nfi["match"] = nfi["ESS"].map(ESS_TO_MATCH)
    nfi = nfi.loc[nfi["match"].notna()].copy()
    nfi = (
        nfi.groupby(["match", "age"], observed=True)
        .agg(area=("area", "sum"), area_err=("area_err", lambda x: float(np.sqrt(np.sum(np.asarray(x) ** 2)))))
        .reset_index()
    )
    zz_lo, zz_main, zz_hi = ctx["rs_lo"], ctx["rs_main"], ctx["rs_hi"]
    empty = zz_main.iloc[0:0]

    def rs_for_match(key):
        tfvs = MATCH_TO_TFVS.get(key, [])
        if not tfvs:
            return empty, empty, empty
        return (
            zz_lo.loc[zz_lo["tfv"].isin(tfvs)],
            zz_main.loc[zz_main["tfv"].isin(tfvs)],
            zz_hi.loc[zz_hi["tfv"].isin(tfvs)],
        )

    gami_getter = None
    if "gami_by_tfv" in ctx:
        gbt = ctx["gami_by_tfv"]
        gbt_lo = ctx.get("gami_by_tfv_lo", gbt)
        gbt_hi = ctx.get("gami_by_tfv_hi", gbt)

        def gami_for_match(key):
            tfvs = MATCH_TO_TFVS.get(key, [])
            if not tfvs:
                return empty, empty, empty
            return (
                gbt_lo.loc[gbt_lo["tfv"].isin(tfvs)],
                gbt.loc[gbt["tfv"].isin(tfvs)],
                gbt_hi.loc[gbt_hi["tfv"].isin(tfvs)],
            )

        gami_getter = gami_for_match

    out = fig_dir / "Figure_SI4_age_kde_by_species.png"
    return _plot_kde_by_class(
        nfi,
        "match",
        MATCH_LABEL,
        out,
        rs_for_match,
        gami_getter=gami_getter,
        class_order=MATCH_ORDER,
        ncols=4,
    )


def plot_figure_si5(ctx: dict[str, Any], fig_dir: Path) -> Path | None:
    nfi = ctx.get("nfi_greco")
    if nfi is None or nfi.empty:
        print("  Skip SI5: set figures.nfi_surf_greco")
        return None
    order = sorted(nfi["GRECO"].dropna().unique().tolist())
    titles = {
        g: f"{g}: {nfi.loc[nfi['GRECO'] == g, 'GRECO_name'].iloc[0]}"
        if (nfi["GRECO"] == g).any()
        else str(g)
        for g in order
    }
    zz_lo = annotate_groups(ctx["rs_lo"])
    zz_main = annotate_groups(ctx["rs_main"])
    zz_hi = annotate_groups(ctx["rs_hi"])

    def rs_for_greco(g):
        return (
            zz_lo.loc[zz_lo["GRECO"] == g],
            zz_main.loc[zz_main["GRECO"] == g],
            zz_hi.loc[zz_hi["GRECO"] == g],
        )

    gami_getter = None
    if "gami_by_greco" in ctx:
        gbg = ctx["gami_by_greco"]
        gbg_lo = ctx.get("gami_by_greco_lo", gbg)
        gbg_hi = ctx.get("gami_by_greco_hi", gbg)

        def gami_for_greco(g):
            return (
                gbg_lo.loc[gbg_lo["GRECO"] == g],
                gbg.loc[gbg["GRECO"] == g],
                gbg_hi.loc[gbg_hi["GRECO"] == g],
            )

        gami_getter = gami_for_greco

    out = fig_dir / "Figure_SI5_age_kde_by_greco.png"
    return _plot_kde_by_class(
        nfi, "GRECO", titles, out, rs_for_greco, gami_getter=gami_getter, class_order=order, ncols=4
    )


# ---------------------------------------------------------------------------
# SI6 — species bubbles + Ours−GAMI
# ---------------------------------------------------------------------------

def _tfv_stats_from_hist(hist_df: pd.DataFrame) -> pd.DataFrame:
    df = hist_df.copy()
    if "area" not in df.columns:
        df["area"] = df["count"] * 0.09 / 1e6
    df = df[(df["tfv"] > 0) & (df["age"] >= 0)].copy()
    df["tfv"] = df["tfv"].astype(int)
    stats = df.groupby("tfv", observed=True).apply(calc_stats, include_groups=False).reset_index()
    stats["tfv"] = stats["tfv"].astype(int)
    stats["tfv_group"] = stats["tfv"].map(lambda t: TFV_GROUP.get(int(t)))
    return stats


def _bubble_xlim_xticks(stats_df, *, lo_floor=None, pad=0.06):
    """Content-tight xlim + sparse ticks (same approach as ``essd_growth_curves`` SI1)."""
    q25 = stats_df["q25"].to_numpy(float)
    q75 = stats_df["q75"].to_numpy(float)
    ok = np.isfinite(q25) & np.isfinite(q75)
    if not ok.any():
        return (0.0, 1.0), np.asarray([0.0, 1.0])
    data_lo = float(np.nanmin(q25[ok]))
    data_hi = float(np.nanmax(q75[ok]))
    span = max(data_hi - data_lo, 1e-9)
    lo = data_lo - pad * span
    hi = data_hi + pad * span
    if lo_floor is not None and data_lo >= lo_floor:
        lo = max(lo, lo_floor)
    lo = min(lo, data_lo)
    hi = max(hi, data_hi)
    if hi <= lo:
        hi = lo + span
    ticks = np.asarray(MaxNLocator(nbins=4, min_n_ticks=3).tick_values(lo, hi), float)
    ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
    return (lo, hi), ticks


def _plot_tfv_bubbles_all(
    ax,
    stats_df,
    xlim=None,
    xticks=None,
    tfv_order=None,
    xlabel="Age (years)",
    show_legend=True,
    show_ylabel=True,
    area_max=None,
    lo_floor=None,
):
    """Horizontal Q25–Q75 / Q50 bubbles — layout matched to growth-curves SI1."""
    stats = stats_df[~stats_df["tfv_group"].isin(["Other", "Forest"])].copy()
    if tfv_order is None:
        stats = stats.sort_values("area", ascending=True)
        tfv_order = stats["tfv"].astype(int).tolist()
    else:
        tfv_order = [int(t) for t in tfv_order]
        stats = stats[stats["tfv"].isin(tfv_order)].copy()
    y_map = {t: i for i, t in enumerate(tfv_order)}
    if area_max is None:
        area_max = float(stats["area"].max()) if len(stats) else 1.0
    bubble = lambda a: 50.0 + 1000.0 * (a / max(area_max, 1e-12)) ** 0.7

    if xlim is None:
        xlim, auto_ticks = _bubble_xlim_xticks(stats, lo_floor=lo_floor)
    else:
        auto_ticks = None
    if xticks is None:
        xticks = auto_ticks if auto_ticks is not None else _bubble_xlim_xticks(stats, lo_floor=lo_floor)[1]

    for row in stats.itertuples():
        tfv = int(row.tfv)
        if tfv not in y_map:
            continue
        y = y_map[tfv]
        c = TFV_GROUP_COLORS.get(row.tfv_group, "0.5")
        ax.plot(
            [row.q25, row.q75],
            [y, y],
            color=c,
            linewidth=2.4,
            zorder=2,
            solid_capstyle="round",
        )
        ax.scatter(
            row.q50,
            y,
            s=bubble(row.area),
            c=c,
            alpha=BUBBLE_ALPHA,
            edgecolors="none",
            zorder=3,
            clip_on=False,
        )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=FS_O)
    ax.tick_params(axis="x", labelsize=FS_O)
    ax.set_yticks(np.arange(len(tfv_order)))
    y_pad = 0.5
    if show_ylabel:
        ax.set_yticklabels(
            [TFV_LABEL.get(float(t), str(int(t))) for t in tfv_order],
            fontsize=FS_O,
            va="center",
        )
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_ylim(-y_pad, max(len(tfv_order) - 1, 0) + y_pad)
    if show_legend:
        area_h = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.6",
                markeredgecolor="none",
                markersize=np.sqrt(bubble(a)) / 3.0,
                label=f"{a:g} Mha",
                alpha=BUBBLE_ALPHA,
                linestyle="none",
            )
            for a in (1.0, 5.0, 20.0)
        ]
        grp_h = [
            Line2D([0], [0], color=c, linewidth=2.4, label=g)
            for g, c in TFV_GROUP_COLORS.items()
            if g in set(stats["tfv_group"])
        ]
        leg1 = ax.legend(
            handles=area_h,
            title="Area",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=FS_O,
            title_fontsize=FS_O,
            frameon=False,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=grp_h,
            title="Composition",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            fontsize=FS_O,
            title_fontsize=FS_O,
            frameon=False,
        )
    return tfv_order


def plot_figure_si6(ctx: dict[str, Any], fig_dir: Path) -> Path:
    ours = _tfv_stats_from_hist(ctx["rs_main"])
    ours = ours[ours["tfv_group"].notna() & ~ours["tfv_group"].isin(["Other", "Forest"])]
    # Same as growth-curves SI1: sort by area ascending → largest TFV at top.
    ours = ours.sort_values("area", ascending=True).reset_index(drop=True)
    order = ours["tfv"].astype(int).tolist()
    area_max = float(ours["area"].max()) if len(ours) else 1.0
    area_by_tfv = ours.set_index("tfv")["area"].astype(float)
    n = len(order)

    if "diff_tfv_stats" in ctx:
        diff = ctx["diff_tfv_stats"].copy()
        if "tfv_group" not in diff.columns:
            diff["tfv_group"] = diff["tfv"].map(lambda t: TFV_GROUP.get(int(t)))
        # Bubble size / y-order from Ours area (not the Ours∩GAMI mask area).
        diff["tfv"] = diff["tfv"].astype(int)
        diff["area"] = diff["tfv"].map(area_by_tfv)
        diff = diff.dropna(subset=["area"])
        fig, (ax_a, ax_b) = plt.subplots(
            1,
            2,
            figsize=(12, max(6, 0.28 * n)),
            gridspec_kw={"wspace": 0.18, "width_ratios": [1.15, 1.25]},
        )
        _plot_tfv_bubbles_all(
            ax_a,
            ours,
            tfv_order=order,
            show_legend=False,
            area_max=area_max,
            lo_floor=0.0,
        )
        _plot_tfv_bubbles_all(
            ax_b,
            diff,
            tfv_order=order,
            xlabel=r"Ours $-$ GAMI (years)",
            show_legend=True,
            show_ylabel=False,
            area_max=area_max,
            lo_floor=None,
        )
        ax_b.axvline(0.0, color="0.5", linestyle="--", linewidth=0.9, zorder=1)
        ax_a.set_title("Ours", fontsize=FS_O, pad=6)
        ax_b.set_title("Ours − GAMI (30 m)", fontsize=FS_O, pad=6)
        for ax, lbl, lx in zip((ax_a, ax_b), "ab", (-0.22, -0.08)):
            _add_panel_label(ax, lbl, x=lx)
    else:
        print("  SI6 panel (b) skipped (set figures.ours_minus_gami_tfv_csv); writing Ours-only panel")
        fig, ax_a = plt.subplots(figsize=(7, max(6, 0.28 * n)))
        _plot_tfv_bubbles_all(ax_a, ours, tfv_order=order, show_legend=True, area_max=area_max, lo_floor=0.0)
        ax_a.set_title("Ours", fontsize=FS_O, pad=6)
        _add_panel_label(ax_a, "a", x=-0.22)

    out = fig_dir / "Figure_SI6_age_bubbles_species_gami_diff.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def run_figures(cfg: dict[str, Any], figure_dir: Path | None = None, **_kw) -> Path:
    """Load inputs and write Fig. 3, SI3–SI6 PNGs."""
    if figure_dir is not None:
        fig_dir = Path(figure_dir).expanduser()
    elif cfg.get("figure_dir"):
        fig_dir = Path(cfg["figure_dir"]).expanduser()
    else:
        raise KeyError("figure_dir missing (set in config or pass figure_dir=)")
    fig_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_figure_context(cfg)
    for name, fn in (
        ("Figure 3", plot_figure3),
        ("Figure SI3", plot_figure_si3),
        ("Figure SI4", plot_figure_si4),
        ("Figure SI5", plot_figure_si5),
        ("Figure SI6", plot_figure_si6),
    ):
        path = fn(ctx, fig_dir)
        if path is not None:
            print(f"  {name}: {path}")
    return fig_dir
