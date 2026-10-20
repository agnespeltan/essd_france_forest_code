"""ESSD rotation figures: Fig. 6, SI12, SI13 — match Fig_disturbances style."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from lib.constants import DICT_GROUP_EN, DICT_GROUP_TFVS, TFV_GROUP_COLORS, TFV_GROUP_ORDER

ROT = "rotation_period"
ROT_E = "rotation_period_err"
NATIONAL_CI_Z_95 = 1.96

_DOT_MARKERSIZE = 11
_DOT_MARKER_ALPHA = 1.0
_SI_DOT_MARKERSIZE = 5
_SI_ERROR_CAPSIZE = 0
_SI_ERROR_LW = 0.9
_AXIS_TICK_LABELSIZE = 14
_MAP_CBAR_TICK_FONTSIZE = 14
_MAP_CBAR_X = 0.9
_MAP_CBAR_Y = 0.5
_MAP_CBAR_WIDTH = 0.03
_MAP_CBAR_HEIGHT = 0.4
_SI_FIG_HEIGHT = 9.0
_SI_PANEL_WIDTH = 2.9
_SI_YTICK_LABELSIZE = 9
_SI_TICK_LABELSIZE = 11
_SI_YLABEL_FONTSIZE = 11
_SI_LEGEND_FONTSIZE = 11
_SI_PANEL_LABEL_FONTSIZE = 14
_SI_LEGEND_MARKERSIZE = 7
_HARVEST_MARKER = "o"
_NATURAL_MARKER = "s"
_DEFAULT_FRANCE_OUTLINE = Path("/home/apelliss/Documents/IGN_donnees/France_1km.shp")
_SI_LEGEND_ELEMENTS = [
    Line2D(
        [0],
        [0],
        marker=_HARVEST_MARKER,
        linestyle="None",
        markerfacecolor="gray",
        markeredgecolor="gray",
        markersize=_SI_LEGEND_MARKERSIZE,
        alpha=_DOT_MARKER_ALPHA,
        label="Harvest",
    ),
    Line2D(
        [0],
        [0],
        marker=_NATURAL_MARKER,
        linestyle="None",
        markerfacecolor="gray",
        markeredgecolor="gray",
        markersize=_SI_LEGEND_MARKERSIZE,
        alpha=_DOT_MARKER_ALPHA,
        label="Natural",
    ),
]


def _ser_num_from_codeser(codeser: object) -> int | float:
    if pd.isna(codeser):
        return np.nan
    s = str(codeser).strip()
    return (ord(s[0].upper()) - ord("A") + 1) * 100 + int(s[1:])


def _load_trends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tfv = pd.to_numeric(df["TFV"], errors="coerce")
    keep = tfv.notna() & (tfv > 0) & (tfv < 500) & (tfv != 400)
    out = df.loc[keep].copy()
    out["TFV"] = tfv.loc[keep].astype(int)
    out["ser"] = pd.to_numeric(out["ser"], errors="coerce").astype(int)
    out["total_area"] = pd.to_numeric(out.get("total_area", 0.0), errors="coerce").fillna(0.0)
    out[ROT] = pd.to_numeric(out[ROT], errors="coerce")
    out[ROT_E] = pd.to_numeric(out[ROT_E], errors="coerce")
    out["TFV_group"] = out["TFV"].map(DICT_GROUP_TFVS)
    return out


def _rotation_mean_subset(df: pd.DataFrame) -> pd.DataFrame:
    work = df[[ROT, "total_area"]].copy()
    return work[np.isfinite(work[ROT]) & (work[ROT] > 0) & (work["total_area"] > 0)]


def _rotation_se_subset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [ROT, ROT_E, "total_area", "TFV"]
    work = df[cols].copy()
    return work[
        np.isfinite(work[ROT])
        & (work[ROT] > 0)
        & np.isfinite(work[ROT_E])
        & (work[ROT_E] >= 0)
        & (work["total_area"] > 0)
    ]


def _tfv_block_se(df: pd.DataFrame) -> float:
    work = _rotation_se_subset(df)
    if work.empty:
        return 0.0
    w_sum = float(work["total_area"].sum())
    if w_sum <= 0:
        return 0.0
    acc = 0.0
    for _, g in work.groupby("TFV"):
        wg = g["total_area"].to_numpy(dtype=float)
        sg = g[ROT_E].to_numpy(dtype=float)
        acc += float(np.sum(wg * sg) ** 2)
    return float(np.sqrt(acc) / w_sum)


def rotation_weighted_stats(df: pd.DataFrame) -> dict[str, float]:
    mean_work = _rotation_mean_subset(df)
    if mean_work.empty:
        return {"mean": float("nan"), "std": 0.0}
    mean = float(np.average(mean_work[ROT], weights=mean_work["total_area"]))
    return {"mean": mean, "std": _tfv_block_se(df)}


def national_area_weighted_mean_stderr(df: pd.DataFrame) -> tuple[float, float]:
    st = rotation_weighted_stats(df)
    return st["mean"], st["std"]


def _fmt_ci95(mean: float, se: float, mean_nd: int = 0) -> str:
    if not np.isfinite(mean):
        return "n/a"
    half = NATIONAL_CI_Z_95 * (se if np.isfinite(se) else 0.0)
    return f"{mean:.{mean_nd}f} [{mean - half:.{mean_nd}f}, {mean + half:.{mean_nd}f}]"


def _normalize_agent(series: pd.Series) -> pd.Series:
    a = series.astype(str).str.lower()
    return a.map(
        {
            "1": "natural",
            "1.0": "natural",
            "2": "natural",
            "2.0": "natural",
            "3": "harvest",
            "3.0": "harvest",
            "unplanned": "natural",
            "natural": "natural",
            "harvest": "harvest",
        }
    ).fillna(a)


def _ser_rotation_agg(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ser, g in df.groupby("ser"):
        st = rotation_weighted_stats(g)
        rows.append(
            {
                "ser": int(ser),
                ROT: st["mean"],
                ROT_E: st["std"],
                "total_area": float(g["total_area"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["ser", ROT, ROT_E, "total_area", "rel_unc"])
    out["rel_unc"] = np.where(
        np.isfinite(out[ROT]) & (out[ROT] > 0) & np.isfinite(out[ROT_E]),
        100.0 * out[ROT_E] / np.abs(out[ROT]),
        np.nan,
    )
    return out


def _category_color(item, group_col: str = "TFV") -> str:
    if group_col == "TFV_group":
        return TFV_GROUP_COLORS.get(item, "gray")
    return TFV_GROUP_COLORS.get(DICT_GROUP_TFVS.get(float(item), "Other"), "gray")


def _place_subplot_label(ax, label: str, *, style: str = "auto") -> None:
    if style == "si":
        x, y, ha, va = 0.0, 1.02, "left", "bottom"
    elif style == "map" or (style == "auto" and not ax.axison):
        x, y, ha, va = 0.02, 0.97, "left", "top"
    else:
        x, y, ha, va = -0.02, 1.06, "right", "bottom"
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va=va,
        ha=ha,
        clip_on=False,
        zorder=20,
    )


def _place_fig_map_title(fig, ax, title: str, y_pad: float = 0.012) -> None:
    if not title:
        return
    pos = ax.get_position()
    fig.text(
        0.5 * (pos.x0 + pos.x1),
        pos.y1 + y_pad,
        title,
        fontsize=16,
        ha="center",
        va="bottom",
        clip_on=False,
    )


def _attach_map_colorbar(ax, mappable, ticks, labels) -> None:
    fig = ax.get_figure()
    pos = ax.get_position()
    fig_x = pos.x0 + _MAP_CBAR_X * pos.width
    fig_y = pos.y0 + _MAP_CBAR_Y * pos.height - _MAP_CBAR_HEIGHT * pos.height / 2
    cbar_ax = fig.add_axes(
        [fig_x, fig_y, _MAP_CBAR_WIDTH * pos.width, _MAP_CBAR_HEIGHT * pos.height]
    )
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels, fontsize=_MAP_CBAR_TICK_FONTSIZE)
    cbar.ax.tick_params(labelsize=_MAP_CBAR_TICK_FONTSIZE)


def _overlay_france_outline(ax, france_gdf: gpd.GeoDataFrame | None) -> None:
    if france_gdf is None or france_gdf.empty:
        return
    france_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=11)


def _plot_ser_choropleth(
    ax,
    gdf: gpd.GeoDataFrame,
    column: str,
    *,
    cmap: str,
    tick_fmt,
    title: str = "",
    panel_label: str | None = None,
    vmin=None,
    vmax=None,
    france_gdf: gpd.GeoDataFrame | None = None,
):
    if gdf.empty or column not in gdf.columns or gdf[column].dropna().empty:
        ax.set_axis_off()
        return
    valid = gdf[column].dropna()
    vmin = float(valid.min()) if vmin is None else float(vmin)
    vmax = float(valid.max()) if vmax is None else float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        missing_kwds={"color": "lightgray"},
        legend=False,
    )
    _overlay_france_outline(ax, france_gdf)
    ax.set_axis_off()
    ax._map_cbar_spec = {
        "mappable": ax.collections[0],
        "ticks": [vmin, vmax],
        "labels": [tick_fmt(vmin), tick_fmt(vmax)],
    }
    ax._map_title = title
    if panel_label:
        _place_subplot_label(ax, panel_label, style="map")


def _set_group_xticklabels(ax, labels: list[str]) -> None:
    wrapped = [str(lab).replace(" ", "\n") for lab in labels]
    ax.set_xticklabels(wrapped, fontsize=_AXIS_TICK_LABELSIZE)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="top")
    ax.tick_params(axis="x", pad=6, labelsize=_AXIS_TICK_LABELSIZE)


def _adaptive_value_lim(means: list[float], ses: list[float], *, pad_frac: float = 0.12):
    lo = min(m - s for m, s in zip(means, ses) if np.isfinite(m))
    hi = max(m + s for m, s in zip(means, ses) if np.isfinite(m))
    span = hi - lo
    pad = pad_frac * span if span > 0 else max(abs(hi), abs(lo), 1.0) * pad_frac
    return lo - pad, hi + pad


def _plot_group_dots(ax, harvest: pd.DataFrame, group_order: list[str], *, panel_label: str = "b"):
    positions = np.arange(len(group_order))
    means, ses = [], []
    for pos, item in zip(positions, group_order):
        st = rotation_weighted_stats(harvest.loc[harvest["TFV_group"] == item])
        means.append(st["mean"])
        ses.append(st["std"])
        if not np.isfinite(st["mean"]):
            continue
        color = _category_color(item, "TFV_group")
        ax.errorbar(
            pos,
            st["mean"],
            yerr=st["std"],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=_DOT_MARKERSIZE,
            capsize=3.5,
            elinewidth=1.2,
            zorder=4,
        )
    ax.set_xticks(positions)
    _set_group_xticklabels(ax, group_order)
    ax.set_ylabel("Rotation period (years)", fontsize=16)
    ax.tick_params(axis="y", labelsize=_AXIS_TICK_LABELSIZE)
    y_lo, y_hi = _adaptive_value_lim(means, ses, pad_frac=0.12)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if panel_label:
        _place_subplot_label(ax, panel_label, style="auto")


def _si_tfv_category_ylim(n_tfvs: int) -> tuple[float, float]:
    """Category-axis lim with first TFV at the bottom (same as essd_disturbances SI)."""
    n = int(n_tfvs)
    return (-0.6, n - 0.4)


def _style_si_tfv_axis(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _plot_tfv_horizontal(ax, harvest: pd.DataFrame, tfv_order: list[int]) -> None:
    """Horizontal SI panel matching essd_disturbances Figure_SI8/SI10 style."""
    if not tfv_order:
        ax.set_axis_off()
        return
    positions = np.arange(len(tfv_order))
    means, ses = [], []
    ax._si_horizontal = True
    for pos, tfv in zip(positions, tfv_order):
        st = rotation_weighted_stats(harvest.loc[harvest["TFV"] == tfv])
        means.append(st["mean"])
        ses.append(st["std"])
        if not np.isfinite(st["mean"]):
            continue
        color = _category_color(tfv, "TFV")
        ax.errorbar(
            st["mean"],
            pos,
            xerr=st["std"],
            fmt=_HARVEST_MARKER,
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            alpha=_DOT_MARKER_ALPHA,
            markersize=_SI_DOT_MARKERSIZE,
            capsize=_SI_ERROR_CAPSIZE,
            elinewidth=_SI_ERROR_LW,
            zorder=4,
        )
    labels = [DICT_GROUP_EN.get(float(t), str(int(t))) for t in tfv_order]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=_SI_YTICK_LABELSIZE)
    ax.set_ylim(*_si_tfv_category_ylim(len(tfv_order)))
    ax.set_ylabel("")
    ax.set_xlabel("Rotation period (years)", fontsize=_SI_YLABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=_SI_TICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", labelleft=True, labelsize=_SI_YTICK_LABELSIZE, length=3)
    if any(np.isfinite(m) for m in means):
        x_lo, x_hi = _adaptive_value_lim(means, ses, pad_frac=0.10)
        ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    _style_si_tfv_axis(ax)


def _finalize_si_tfv_figure(fig, ax, save_path: Path, sorted_tfvs: list[int]) -> None:
    """Match essd_disturbances ``_finalize_si_tfv_vertical_figure`` for one panel."""
    _style_si_tfv_axis(ax)
    _place_subplot_label(ax, "a", style="si")
    y_positions = np.arange(len(sorted_tfvs))
    labels = [DICT_GROUP_EN.get(float(t), str(int(t))) for t in sorted_tfvs]
    ax.set_yticks(y_positions)
    ax.set_ylim(*_si_tfv_category_ylim(len(sorted_tfvs)))
    ax.set_yticklabels(labels, fontsize=_SI_YTICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", labelleft=True, labelsize=_SI_YTICK_LABELSIZE, length=3)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.28, right=0.98, wspace=0.18)
    # Re-apply after layout (tight_layout can drop y-tick labels).
    ax.set_yticks(y_positions)
    ax.set_ylim(*_si_tfv_category_ylim(len(sorted_tfvs)))
    ax.set_yticklabels(labels, fontsize=_SI_YTICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", labelleft=True, labelsize=_SI_YTICK_LABELSIZE, length=3)
    bbox_top = ax.get_position().y1
    fig.legend(
        handles=_SI_LEGEND_ELEMENTS,
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_top + 0.02),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=_SI_LEGEND_FONTSIZE,
        frameon=False,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.20)


def _build_sorted_tfvs(*frames: pd.DataFrame) -> list[int]:
    """TFV order for SI13: increasing TFV×SER ``total_area`` (natural + harvest, once per cell).

    Ascending area → smallest at the bottom of a normal y-axis. Only labeled TFVs kept.
    """
    parts: list[pd.DataFrame] = []
    for df in frames:
        if df is None or df.empty or "total_area" not in df.columns:
            continue
        cols = [c for c in ("TFV", "ser", "total_area") if c in df.columns]
        if "TFV" not in cols or "total_area" not in cols:
            continue
        parts.append(df[cols].drop_duplicates(subset=[c for c in ("TFV", "ser") if c in cols]))
    if not parts:
        return []
    area = pd.concat(parts, ignore_index=True)
    if "ser" in area.columns:
        area = area.drop_duplicates(subset=["TFV", "ser"])
    else:
        area = area.drop_duplicates(subset=["TFV"])
    by_tfv = area.groupby("TFV", as_index=True)["total_area"].sum().sort_values(ascending=True)
    return [int(t) for t in by_tfv.index if float(t) in DICT_GROUP_EN]


def print_manuscript_summary(harvest: pd.DataFrame, ser_gdf: gpd.GeoDataFrame) -> None:
    names = ser_gdf[["ser_num"]].copy()
    names["NomSER"] = ser_gdf["NomSER"] if "NomSER" in ser_gdf.columns else ""
    names = names.drop_duplicates("ser_num")
    H = harvest.merge(names, left_on="ser", right_on="ser_num", how="left")
    rot = H.loc[np.isfinite(H[ROT]) & (H[ROT] > 0)]
    nat, nat_se = national_area_weighted_mean_stderr(rot)
    print("\n=== Manuscript rotation (95% CI = 1.96×SE) ===")
    print(f"National: {_fmt_ci95(nat, nat_se, mean_nd=0)}")
    for label, mask in [
        ("Corsica", H["NomSER"].astype(str).str.contains("Corse", case=False, na=False)),
        (
            "southern Alps",
            H["NomSER"]
            .astype(str)
            .str.contains(
                "Alpes internes du Sud|Alpes externes du Sud|Alpes intermédiaires du Sud",
                case=False,
                na=False,
            ),
        ),
        ("Les Landes", H["NomSER"].astype(str).str.contains("Landes de Gascogne", case=False, na=False)),
    ]:
        st = rotation_weighted_stats(H.loc[mask])
        print(f"{label}: {_fmt_ci95(st['mean'], st['std'], mean_nd=0)}")
    print("TFV groups:")
    for g in TFV_GROUP_ORDER:
        st = rotation_weighted_stats(H.loc[H["TFV_group"] == g])
        print(f"  {g}: {_fmt_ci95(st['mean'], st['std'], mean_nd=0)}")
    print("Species:")
    for tfv_i, lab in [(19, "poplar"), (51, "maritime pine"), (63, "larch"), (9, "beech")]:
        st = rotation_weighted_stats(H.loc[H["TFV"] == tfv_i])
        print(f"  {lab}: {_fmt_ci95(st['mean'], st['std'], mean_nd=0)}")


def run_figures(
    trends_path: str | Path,
    figure_dir: str | Path,
    *,
    ser_shapefile: str | Path,
    greco_shapefile: str | Path | None = None,
    france_outline: str | Path | None = None,
) -> Path:
    """Write Fig. 6, SI12, SI13 and print manuscript values."""
    del greco_shapefile
    fig_dir = Path(figure_dir).expanduser()
    fig_dir.mkdir(parents=True, exist_ok=True)

    trend = _load_trends(Path(trends_path))
    trend["agent"] = _normalize_agent(trend["agent"])
    harvest = trend.loc[trend["agent"].eq("harvest")].copy()
    natural = trend.loc[trend["agent"].eq("natural")].copy()
    if harvest.empty or harvest[ROT].notna().sum() == 0:
        raise ValueError(
            f"No harvest rows with rotation_period in {trends_path}. "
            "Re-run 01_attach_rotation.py after fixing agent labels."
        )

    ser = gpd.read_file(ser_shapefile)
    if "ser_num" not in ser.columns:
        if "codeser" not in ser.columns:
            raise KeyError("SER shapefile needs ser_num or codeser")
        ser["ser_num"] = ser["codeser"].map(_ser_num_from_codeser)

    outline_path = Path(france_outline) if france_outline else _DEFAULT_FRANCE_OUTLINE
    france_gdf = gpd.read_file(outline_path) if outline_path.is_file() else None
    if france_gdf is not None and ser.crs is not None and france_gdf.crs != ser.crs:
        france_gdf = france_gdf.to_crs(ser.crs)

    ser_agg = _ser_rotation_agg(harvest)
    merged = ser.merge(ser_agg, left_on="ser_num", right_on="ser", how="left")

    present = set(
        harvest.loc[np.isfinite(harvest[ROT]) & (harvest[ROT] > 0), "TFV_group"].dropna()
    )
    group_order = [g for g in TFV_GROUP_ORDER if g in present]
    tfv_order = _build_sorted_tfvs(natural, harvest)

    # Figure 6 — SER map + TFV-group dots (original layout)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 5.0),
        gridspec_kw={"width_ratios": [1.0, 0.42, 1.35], "wspace": 0.12},
    )
    ax_map, ax_sp, ax_dots = axes
    ax_sp.set_axis_off()
    _plot_ser_choropleth(
        ax_map,
        merged,
        ROT,
        cmap="viridis",
        tick_fmt=lambda v: f"{v:.0f} yr",
        title="Rotation period",
        panel_label="a",
        france_gdf=france_gdf,
    )
    _plot_group_dots(ax_dots, harvest, group_order, panel_label="b")
    fig.canvas.draw()
    if getattr(ax_map, "_map_cbar_spec", None):
        spec = ax_map._map_cbar_spec
        _attach_map_colorbar(ax_map, spec["mappable"], spec["ticks"], spec["labels"])
        del ax_map._map_cbar_spec
    _place_fig_map_title(fig, ax_map, "Rotation period")
    out6 = fig_dir / "Figure_6_rotation_period.png"
    fig.savefig(out6, dpi=300, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)

    # Figure SI12 — relative uncertainty map (single panel, no empty column)
    fig, ax_h = plt.subplots(figsize=(7.5, 5.0))
    vmax = float(merged["rel_unc"].dropna().max()) if merged["rel_unc"].notna().any() else 3.0
    _plot_ser_choropleth(
        ax_h,
        merged,
        "rel_unc",
        cmap="hot_r",
        tick_fmt=lambda v: f"{v:.0f}%",
        title="Rotation period",
        panel_label="a",
        vmin=0.0,
        vmax=vmax,
        france_gdf=france_gdf,
    )
    fig.canvas.draw()
    plt.tight_layout(pad=0.4)
    if getattr(ax_h, "_map_cbar_spec", None):
        spec = ax_h._map_cbar_spec
        _attach_map_colorbar(ax_h, spec["mappable"], spec["ticks"], spec["labels"])
        del ax_h._map_cbar_spec
    _place_fig_map_title(fig, ax_h, "Rotation period")
    out_si12 = fig_dir / "Figure_SI12_relative_uncertainty_rotation_period.png"
    fig.savefig(out_si12, dpi=300, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)

    # Figure SI13 — by TFV (same horizontal SI style as disturbance SI8/SI10)
    fig, ax = plt.subplots(figsize=(_SI_PANEL_WIDTH + 2.2, _SI_FIG_HEIGHT))
    _plot_tfv_horizontal(ax, harvest, tfv_order)
    out_si13 = fig_dir / "Figure_SI13_rotation_period_by_species_category.png"
    _finalize_si_tfv_figure(fig, ax, out_si13, tfv_order)
    plt.close(fig)

    print_manuscript_summary(harvest, ser)
    for p in (out6, out_si12, out_si13):
        print(f"  wrote {p}")
    return fig_dir
