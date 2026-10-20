"""Build disturbance metric table and run one-way η² partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.config import output_path
from lib.figures.prepare_table import build_figure_table, _norm_agent
from lib.variance.eta2 import eta2_arrays, eta2_equal, perm_p_eta2, summarize_factor_row

CARBON_FRACTION = 0.475
PATCH_PIXEL_AREA_HA = 0.09
YEAR_REF = 2020

TFV_GROUP_ORDER = [
    "Pure deciduous",
    "Pure conifers",
    "Mixed deciduous",
    "Mixed conifers",
]

DICT_GROUP_TFVS = {
    0.0: "Other",
    1.0: "Pure deciduous",
    6.0: "Pure deciduous",
    9.0: "Pure deciduous",
    10.0: "Pure deciduous",
    14.0: "Pure deciduous",
    19.0: "Pure deciduous",
    49.0: "Pure deciduous",
    100.0: "Mixed deciduous",
    111.0: "Mixed deciduous",
    51.0: "Pure conifers",
    52.0: "Pure conifers",
    53.0: "Pure conifers",
    57.0: "Pure conifers",
    58.0: "Pure conifers",
    81.0: "Pure conifers",
    80.0: "Mixed conifers",
    61.0: "Pure conifers",
    63.0: "Pure conifers",
    64.0: "Pure conifers",
    91.0: "Pure conifers",
    90.0: "Mixed conifers",
    200.0: "Mixed conifers",
    222.0: "Mixed conifers",
    310.0: "Mixed deciduous",
    320.0: "Mixed conifers",
    400.0: "Other",
    401.0: "Mixed deciduous",
    402.0: "Mixed conifers",
    403.0: "Mixed deciduous",
}

AGBD_LOSS = "agbd_lost_conservative"
REL_AGBD_LOSS = "relative_agbd_lost_conservative"
RATE_X_SEV = "rate_x_severity"
ROTATION = "rotation_period"


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _fitted_or_fallback(
    trend,
    intercept,
    slope,
    intercept_se,
    slope_se,
    median,
    median_se,
    *,
    year_start: int,
    invalid_below: float,
    fallback_floor: float,
):
    dt = float(YEAR_REF - year_start)
    keep = trend.astype(str).to_numpy() != "no trend"
    a = _as_float(intercept)
    b = _as_float(slope)
    sa = _as_float(intercept_se)
    sb = np.maximum(_as_float(slope_se), 0.0)
    med = _as_float(median)
    sm = np.maximum(_as_float(median_se), 0.0)
    fitted = a + b * dt
    fitted_se = np.sqrt(sa**2 + (sb * dt) ** 2)
    keep = keep & np.isfinite(fitted) & (fitted >= invalid_below)
    fallback = np.where(np.isfinite(med), med, fallback_floor)
    fallback = np.maximum(fallback, fallback_floor)
    level = np.where(keep, fitted, fallback)
    level_se = np.maximum(np.where(keep, fitted_se, sm), 0.0)
    slope_hat = np.where(keep, b, 0.0)
    cov = np.where(keep, dt * sb**2, 0.0)
    return level, level_se, slope_hat, sb, cov


def _row_ratio_pct(slope, level):
    slope = np.asarray(slope, dtype=float)
    level = np.asarray(level, dtype=float)
    out = np.full(slope.shape, np.nan, dtype=float)
    ok = np.isfinite(slope) & np.isfinite(level) & (level > 0)
    out[ok] = 100.0 * slope[ok] / level[ok]
    return out


def _row_ratio_pct_se(slope, slope_se, level, level_se, cov=None):
    slope = np.asarray(slope, dtype=float)
    slope_se = np.asarray(slope_se, dtype=float)
    level = np.asarray(level, dtype=float)
    level_se = np.asarray(level_se, dtype=float)
    cov = np.zeros_like(slope) if cov is None else np.asarray(cov, dtype=float)
    out = np.full(slope.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(slope)
        & np.isfinite(slope_se)
        & np.isfinite(level)
        & np.isfinite(level_se)
        & (level > 0)
        & (slope_se >= 0)
        & (level_se >= 0)
    )
    s, ss, L, sL, c = slope[ok], slope_se[ok], level[ok], level_se[ok], cov[ok]
    cap = ss * sL
    c = np.clip(c, -cap, cap)
    var = (
        (100.0 / L) ** 2 * ss**2
        + (100.0 * s / L**2) ** 2 * sL**2
        - 2.0 * (100.0 / L) * (100.0 * s / L**2) * c
    )
    out[ok] = np.sqrt(np.maximum(var, 0.0))
    return out


def add_row_metrics(df: pd.DataFrame, *, year_start: int) -> pd.DataFrame:
    """Attach 2020 levels, trends, severity products and SEs (same as figures)."""
    out = df.copy()
    level, level_se, slope, slope_se, cov = _fitted_or_fallback(
        out["trend_disturbance_rate"],
        out["intercept_disturbance_rate"],
        out["slope_disturbance_rate"],
        out["intercept_std_disturbance_rate"],
        out["slope_std_disturbance_rate"],
        out["median_disturbance_rate"],
        out["median_std_disturbance_rate"],
        year_start=year_start,
        invalid_below=0.0,
        fallback_floor=0.0,
    )
    out["disturbance_rate_2020"] = level
    out["disturbance_rate_2020_error"] = level_se
    out["d_area_2020_perc"] = level * 100.0
    out["d_area_2020_error_perc"] = level_se * 100.0
    out["slope_pp"] = slope * 100.0
    out["slope_pp_err"] = slope_se * 100.0
    out["d_area_level_slope_cov"] = cov * 100.0**2
    out["d_area_evolution_perc"] = _row_ratio_pct(out["slope_pp"], out["d_area_2020_perc"])
    out["d_area_evolution_perc_se"] = _row_ratio_pct_se(
        out["slope_pp"],
        out["slope_pp_err"],
        out["d_area_2020_perc"],
        out["d_area_2020_error_perc"],
        cov=out["d_area_level_slope_cov"],
    )

    level_px, level_px_se, slope_px, slope_px_se, cov_px = _fitted_or_fallback(
        out["trend_mean_patch_size"],
        out["intercept_mean_patch_size"],
        out["slope_mean_patch_size"],
        out["intercept_std_mean_patch_size"],
        out["slope_std_mean_patch_size"],
        out["median_mean_patch_size"],
        out["median_std_mean_patch_size"],
        year_start=year_start,
        invalid_below=1.0,
        fallback_floor=1.0,
    )
    ha = PATCH_PIXEL_AREA_HA
    out["mean_patch_size_2020"] = level_px * ha
    out["mean_patch_size_2020_error"] = level_px_se * ha
    out["slope_patch_ha"] = slope_px * ha
    out["slope_patch_ha_err"] = slope_px_se * ha
    out["patch_level_slope_cov"] = cov_px * ha**2
    out["mean_patch_size_evolution_perc"] = _row_ratio_pct(
        out["slope_patch_ha"], out["mean_patch_size_2020"]
    )
    out["mean_patch_size_evolution_perc_se"] = _row_ratio_pct_se(
        out["slope_patch_ha"],
        out["slope_patch_ha_err"],
        out["mean_patch_size_2020"],
        out["mean_patch_size_2020_error"],
        cov=out["patch_level_slope_cov"],
    )

    if AGBD_LOSS in out.columns:
        # CSV is Mg/ha biomass → MgC ha⁻¹ (same as figures)
        out[AGBD_LOSS] = (
            pd.to_numeric(out[AGBD_LOSS], errors="coerce").fillna(0.0) * CARBON_FRACTION
        ).clip(lower=0.0)
        err_col = f"{AGBD_LOSS}_error"
        if err_col in out.columns:
            out[err_col] = (
                pd.to_numeric(out[err_col], errors="coerce").fillna(0.0) * CARBON_FRACTION
            ).clip(lower=0.0)
        sev = out[AGBD_LOSS].to_numpy(dtype=float)
        sev_se = (
            pd.to_numeric(out.get(err_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        rate = out["disturbance_rate_2020"].to_numpy(dtype=float)
        rate_se = out["disturbance_rate_2020_error"].to_numpy(dtype=float)
        out[RATE_X_SEV] = np.maximum(sev * rate, 0.0)
        out[f"{RATE_X_SEV}_error"] = np.sqrt((sev * rate_se) ** 2 + (rate * sev_se) ** 2)

    return out


def _load_joined(cfg: dict[str, Any], *, rotation_csv: Path | None) -> pd.DataFrame:
    trends_path = Path(cfg.get("trends_csv") or output_path(cfg, "trends")).expanduser()
    sev_path = Path(cfg.get("severity_csv") or output_path(cfg, "severity")).expanduser()
    if not trends_path.is_file():
        raise FileNotFoundError(f"Trends CSV not found: {trends_path}")
    if not sev_path.is_file():
        raise FileNotFoundError(f"Severity CSV not found: {sev_path}")

    trends = pd.read_csv(trends_path)
    sev = pd.read_csv(sev_path)
    # Preserve trends gap_filled (prepare_table prefers severity's column)
    trends_gap = trends[["TFV", "ser", "agent", "gap_filled"]].copy()
    trends_gap["agent"] = _norm_agent(trends_gap["agent"])
    trends_gap["TFV"] = trends_gap["TFV"].astype(str)
    trends_gap["ser"] = pd.to_numeric(trends_gap["ser"], errors="coerce").astype("Int64")
    trends_gap = trends_gap.rename(columns={"gap_filled": "gap_filled_trends"})

    table = build_figure_table(trends, sev)
    table = table.merge(trends_gap, on=["TFV", "ser", "agent"], how="left")
    table["gap_filled_severity"] = pd.to_numeric(table.get("gap_filled"), errors="coerce")

    if rotation_csv is not None:
        rot_path = Path(rotation_csv).expanduser()
        if not rot_path.is_file():
            raise FileNotFoundError(f"Rotation CSV not found: {rot_path}")
        rot = pd.read_csv(rot_path)
        rot["agent"] = _norm_agent(rot["agent"])
        rot["TFV"] = rot["TFV"].astype(str)
        rot["ser"] = pd.to_numeric(rot["ser"], errors="coerce").astype("Int64")
        cols = ["TFV", "ser", "agent", ROTATION, "rotation_period_err", "gap_filled"]
        cols = [c for c in cols if c in rot.columns]
        rot = rot[cols].rename(columns={"gap_filled": "gap_filled_rotation"})
        table = table.merge(rot, on=["TFV", "ser", "agent"], how="left")

    return table


def prepare_analysis_frame(
    cfg: dict[str, Any],
    *,
    rotation_csv: Path | None = None,
) -> pd.DataFrame:
    """SER×TFV×agent cells with derived metrics (gap fills still present for masking)."""
    df = _load_joined(cfg, rotation_csv=rotation_csv)
    year_start = int(cfg.get("year_start", 1999))
    df = add_row_metrics(df, year_start=year_start)

    df["TFV"] = pd.to_numeric(df["TFV"], errors="coerce")
    df = df.loc[df["TFV"].notna() & (df["TFV"] > 0) & (df["TFV"] < 500) & (df["TFV"] != 400)].copy()
    df["TFV"] = df["TFV"].astype(int)
    df["ser"] = pd.to_numeric(df["ser"], errors="coerce")
    df = df.loc[df["ser"].notna()].copy()
    df["ser"] = df["ser"].astype(int)
    df["agent"] = df["agent"].astype(str).str.strip().str.lower()
    df = df.loc[df["agent"].isin(("harvest", "natural"))].copy()
    df["tfv_group"] = df["TFV"].map(DICT_GROUP_TFVS)
    df = df.loc[df["tfv_group"].isin(TFV_GROUP_ORDER)].copy()
    df["total_area"] = pd.to_numeric(df.get("total_area", 0.0), errors="coerce").fillna(0.0)
    df = df.loc[df["total_area"] > 0].copy()
    df = df.drop_duplicates(subset=["ser", "TFV", "agent"], keep="first")
    return df.reset_index(drop=True)


def _param_spec(df: pd.DataFrame) -> list[tuple[str, str, bool, str, str]]:
    """(value_col, label, agent_ok, se_col, gap_col)."""
    specs = [
        ("d_area_2020_perc", "Rate", True, "d_area_2020_error_perc", "gap_filled_trends"),
        ("d_area_evolution_perc", "Rate trend", True, "d_area_evolution_perc_se", "gap_filled_trends"),
        ("mean_patch_size_2020", "Patch size", True, "mean_patch_size_2020_error", "gap_filled_trends"),
        (
            "mean_patch_size_evolution_perc",
            "Patch trend",
            True,
            "mean_patch_size_evolution_perc_se",
            "gap_filled_trends",
        ),
        (AGBD_LOSS, "Abs. severity", True, f"{AGBD_LOSS}_error", "gap_filled_severity"),
        (REL_AGBD_LOSS, "Rel. severity", True, f"{REL_AGBD_LOSS}_error", "gap_filled_severity"),
        (RATE_X_SEV, "AGB intensity", True, f"{RATE_X_SEV}_error", "gap_filled_severity"),
        (ROTATION, "Rotation", False, "rotation_period_err", "gap_filled_rotation"),
    ]
    out = []
    for col, lab, agent_ok, se, gap in specs:
        if col not in df.columns:
            continue
        out.append((col, lab, agent_ok, se if se in df.columns else "", gap))
    return out


def _mask_local(df: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    if gap_col and gap_col in df.columns:
        g = pd.to_numeric(df[gap_col], errors="coerce")
        return df.loc[g.fillna(-1).eq(0)].copy()
    # fallback: combined gap_filled
    if "gap_filled" in df.columns:
        g = pd.to_numeric(df["gap_filled"], errors="coerce")
        return df.loc[g.fillna(-1).eq(0)].copy()
    return df.copy()


def compute_disturbance_eta2(
    cfg: dict[str, Any],
    *,
    rotation_csv: Path | str | None = None,
    n_mc: int = 1000,
    n_perm: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    One-way η² for species (TFV), region (SER), and agent.

    Equal weight per SER×TFV×agent cell. Gap-filled rows excluded per metric source.
    Rotation: harvest only; agent factor skipped. MC CI from row SEs; permutation p.
    """
    rot = Path(rotation_csv).expanduser() if rotation_csv else None
    if rot is None:
        cand = Path(cfg["output_dir"]).expanduser() / "disturbance_trends_with_rotation.csv"
        if cand.is_file():
            rot = cand

    base = prepare_analysis_frame(cfg, rotation_csv=rot)
    specs = _param_spec(base)
    if not specs:
        raise RuntimeError("No variance metrics found in joined disturbance table")

    rng_mc = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 42)
    rows: list[dict[str, Any]] = []
    mc_ser_draws: list[list[float]] = []
    mc_tfv_draws: list[list[float]] = []
    mc_agent_draws: list[list[float]] = []

    for col, lab, agent_ok, se_col, gap_col in specs:
        work = _mask_local(base, gap_col)
        if col == ROTATION:
            work = work.loc[work["agent"].eq("harvest")].copy()
        work = work.loc[np.isfinite(pd.to_numeric(work[col], errors="coerce"))].copy()
        if work.empty:
            continue

        point = {
            "ser": 100.0 * eta2_equal(work, col, "ser"),
            "tfv": 100.0 * eta2_equal(work, col, "TFV"),
        }
        pvals = {
            "ser": perm_p_eta2(work, col, "ser", n_perm=n_perm, rng=rng_perm),
            "tfv": perm_p_eta2(work, col, "TFV", n_perm=n_perm, rng=rng_perm),
        }
        if agent_ok:
            point["agent"] = 100.0 * eta2_equal(work, col, "agent")
            pvals["agent"] = perm_p_eta2(work, col, "agent", n_perm=n_perm, rng=rng_perm)

        y = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if se_col and se_col in work.columns:
            se = np.maximum(
                pd.to_numeric(work[se_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                0.0,
            )
        else:
            se = np.zeros(len(work), dtype=float)
        groups = {"ser": work["ser"].to_numpy(), "tfv": work["TFV"].to_numpy()}
        if agent_ok:
            groups["agent"] = work["agent"].to_numpy()

        w = np.ones(len(work), dtype=float)
        ser_d, tfv_d, agent_d = [], [], []
        for _ in range(n_mc):
            yp = y + rng_mc.normal(0.0, se)
            if col in {
                "d_area_2020_perc",
                "mean_patch_size_2020",
                AGBD_LOSS,
                RATE_X_SEV,
                ROTATION,
            }:
                yp = np.maximum(yp, 0.0)
            if col == REL_AGBD_LOSS:
                yp = np.clip(yp, 0.0, 100.0)
            es = eta2_arrays(yp, w, groups["ser"])
            et = eta2_arrays(yp, w, groups["tfv"])
            if np.isfinite(es):
                ser_d.append(100.0 * es)
            if np.isfinite(et):
                tfv_d.append(100.0 * et)
            if agent_ok:
                ea = eta2_arrays(yp, w, groups["agent"])
                if np.isfinite(ea):
                    agent_d.append(100.0 * ea)

        ci = {
            "ser": (
                (float(np.percentile(ser_d, 2.5)), float(np.percentile(ser_d, 97.5)))
                if len(ser_d) >= 10
                else (np.nan, np.nan)
            ),
            "tfv": (
                (float(np.percentile(tfv_d, 2.5)), float(np.percentile(tfv_d, 97.5)))
                if len(tfv_d) >= 10
                else (np.nan, np.nan)
            ),
        }
        if agent_ok:
            ci["agent"] = (
                (float(np.percentile(agent_d, 2.5)), float(np.percentile(agent_d, 97.5)))
                if len(agent_d) >= 10
                else (np.nan, np.nan)
            )

        mc_ser_draws.append(ser_d)
        mc_tfv_draws.append(tfv_d)
        if agent_ok:
            mc_agent_draws.append(agent_d)

        rows.append(
            summarize_factor_row(
                parameter=lab,
                column=col,
                agent_factor=agent_ok,
                point=point,
                ci=ci,
                pvals=pvals,
                n_cells=len(work),
            )
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("Variance table empty after filtering")

    def _mean_ci_aligned(draw_lists: list[list[float]]) -> tuple[float, float]:
        if not draw_lists:
            return float("nan"), float("nan")
        n = min(len(d) for d in draw_lists)
        if n < 10:
            return float("nan"), float("nan")
        means = [float(np.mean([d[i] for d in draw_lists])) for i in range(n)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(lo), float(hi)

    avg_s = float(table["eta2_ser_pct"].mean())
    avg_t = float(table["eta2_tfv_pct"].mean())
    agent_rows = table.loc[table["agent_factor"]]
    avg_a = float(agent_rows["eta2_agent_pct"].mean()) if not agent_rows.empty else float("nan")
    s_lo, s_hi = _mean_ci_aligned(mc_ser_draws)
    t_lo, t_hi = _mean_ci_aligned(mc_tfv_draws)
    a_lo, a_hi = _mean_ci_aligned(mc_agent_draws)

    summary = {
        "n_metrics": int(len(table)),
        "area_weighted": False,
        "n_mc": n_mc,
        "n_perm": n_perm,
        "avg_eta2_ser_pct": avg_s,
        "avg_eta2_ser_lo": s_lo,
        "avg_eta2_ser_hi": s_hi,
        "avg_eta2_tfv_pct": avg_t,
        "avg_eta2_tfv_lo": t_lo,
        "avg_eta2_tfv_hi": t_hi,
        "avg_eta2_agent_pct": avg_a,
        "avg_eta2_agent_lo": a_lo,
        "avg_eta2_agent_hi": a_hi,
        "rotation_csv": str(rot) if rot else None,
    }
    return table, summary


def format_eta2_summary(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    """Printable block for the terminal / manuscript notes."""

    def _fmt(point, lo, hi, p=None):
        if not np.isfinite(point):
            base = "nan"
        elif not (np.isfinite(lo) and np.isfinite(hi)):
            base = f"{point:.1f}"
        else:
            base = f"{point:.1f} [{lo:.1f}-{hi:.1f}]"
        if p is None or not np.isfinite(p):
            return base
        if p < 0.001:
            return f"{base} p<0.001"
        return f"{base} p={p:.3f}"

    lines = [
        "ONE-WAY η² — SER vs TFV vs agent (EQUAL WEIGHT PER CELL; gap-filled excluded)",
        f"MC n={summary['n_mc']} | perm n={summary['n_perm']}",
        f"{'parameter':<14} {'SER':>28} {'TFV':>28} {'agent':>28}",
    ]
    for _, row in table.iterrows():
        a = (
            _fmt(row["eta2_agent_pct"], row["eta2_agent_lo"], row["eta2_agent_hi"], row["p_agent"])
            if row["agent_factor"]
            else "n/a (harvest-only)"
        )
        lines.append(
            f"{row['parameter']:<14} "
            f"{_fmt(row['eta2_ser_pct'], row['eta2_ser_lo'], row['eta2_ser_hi'], row['p_ser']):>28} "
            f"{_fmt(row['eta2_tfv_pct'], row['eta2_tfv_lo'], row['eta2_tfv_hi'], row['p_tfv']):>28} "
            f"{a:>28}"
        )
    lines.append(
        f"{'Mean':<14} "
        f"{_fmt(summary['avg_eta2_ser_pct'], summary['avg_eta2_ser_lo'], summary['avg_eta2_ser_hi']):>28} "
        f"{_fmt(summary['avg_eta2_tfv_pct'], summary['avg_eta2_tfv_lo'], summary['avg_eta2_tfv_hi']):>28} "
        f"{_fmt(summary['avg_eta2_agent_pct'], summary['avg_eta2_agent_lo'], summary['avg_eta2_agent_hi']):>28}"
    )
    who = [("SER", summary["avg_eta2_ser_pct"]), ("TFV", summary["avg_eta2_tfv_pct"])]
    if np.isfinite(summary["avg_eta2_agent_pct"]):
        who.append(("agent", summary["avg_eta2_agent_pct"]))
    who.sort(key=lambda x: -x[1] if np.isfinite(x[1]) else -np.inf)
    lines.append("→ mean η² ranking: " + " > ".join(f"{k} ({v:.1f}%)" for k, v in who))
    lines.append("Rotation: harvest only; agent η² skipped. Gap-filled values excluded.")
    return "\n".join(lines)
