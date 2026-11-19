"""Bootstrap uncertainty of the Chapman–Richards growth curves.

For each fitted (not gap-filled) TFV × GRECO, the stratum's age points are resampled
with replacement — the 25th / 50th / 75th AGB percentiles of one age stay together —
and refit with the production ``fit_quantiles`` (shared b, c; monotone a; same tail
trim and young-stand baseline). Each draw is therefore one consistent set of three
quantile curves.

Two kinds of uncertainty are kept apart:

* **fit** (σ_fit): spread of the bootstrap draws of the q50 curve — how well the
  typical curve of the stratum is known. Used for comparisons between strata (η²).
* **spread** (σ_spread): between-stand AGB variability, (q75 − q25) / 1.349 of the
  published curves. Zero for t50 because the quantile curves share (b, c).
* **total**: σ_total = √(σ_fit² + σ_spread²); and an empirical "combined" 25/75 %
  from drawing, in each bootstrap draw, one stand value
  q50 + z·(q75 − q25)/1.349 with z ~ N(0, 1), using that draw's own curves.

Metrics are in dry biomass (as the curve parameters); multiply AGBmax / R30 by the
carbon fraction for MgC.
"""

from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from lib.constants import FIT_QUANTILES, TARGET_COLUMN_PREFIX
from lib.fit.chapman_richards import b_min_setting, chapman_richards, fit_quantiles, trim_settings
from lib.fit.preprocess import filter_tfv_greco

METRICS = ("AGBmax", "R30", "t50")
QS = tuple(int(round(100 * q)) for q in FIT_QUANTILES)  # (25, 50, 75)
IQR_TO_SD = 1.349  # IQR of a normal distribution in units of σ


def curve_metrics(a: float, b: float, c: float) -> dict[str, float]:
    if not (np.isfinite([a, b, c]).all() and a > 0 and b > 0 and c > 0):
        return {"AGBmax": np.nan, "R30": np.nan, "t50": np.nan}
    return {
        "AGBmax": a,
        "R30": float(chapman_richards(np.array([30.0]), a, b, c)[0]) / 30.0,
        "t50": float(-np.log(1.0 - 0.5 ** (1.0 / c)) / b),
    }


def _draw_row(res: dict[float, dict[str, Any]]) -> dict[str, float] | None:
    if len(res) != len(FIT_QUANTILES):
        return None
    out: dict[str, float] = {}
    for tau, r in res.items():
        a, b, c = r["params"]
        q = int(round(100 * tau))
        out[f"a_{q}"] = a
        out["b"], out["c"] = b, c
        for m, v in curve_metrics(a, b, c).items():
            out[f"{m}_{q}"] = v
    return out


def bootstrap_pair(
    df: pd.DataFrame,
    target: str,
    tfv: str,
    greco: Any,
    *,
    n_boot: int,
    seed: int,
    trim: dict[str, Any] | None,
    b_min: float | None = None,
) -> pd.DataFrame:
    """All bootstrap draws for one stratum (long table; empty if not fittable)."""
    filt = filter_tfv_greco(df, target, tfv, greco)
    if filt.empty or int(filt["total_age_points"].iloc[0]) < 3:
        return pd.DataFrame()
    # Stable per-stratum seed so results do not depend on worker scheduling.
    rng = np.random.default_rng([seed, zlib.crc32(f"{tfv}|{greco}".encode())])
    n = len(filt)
    rows = []
    for i in range(n_boot):
        sample = filt.iloc[rng.integers(0, n, n)]
        try:
            row = _draw_row(fit_quantiles(sample, target, FIT_QUANTILES, trim=trim, b_min=b_min))
        except Exception:  # noqa: BLE001 - a failed draw is recorded as missing
            row = None
        rows.append({"draw": i, "ok": row is not None, **(row or {})})
    out = pd.DataFrame(rows)
    out.insert(0, "greco", greco)
    out.insert(0, "TFV", str(tfv))
    return out


def run_bootstrap(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    pairs: list[tuple[str, Any]],
    *,
    n_boot: int = 200,
    seed: int = 0,
    target_key: str = "agb",
) -> pd.DataFrame:
    target = TARGET_COLUMN_PREFIX[target_key]
    trim = trim_settings(cfg)
    b_min = b_min_setting(cfg)
    workers = int(cfg.get("workers", 4))
    print(f"Bootstrap: {len(pairs)} strata × B={n_boot}, workers={workers}, trim={trim}")
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(bootstrap_pair, df, target, t, g, n_boot=n_boot, seed=seed, trim=trim, b_min=b_min): (t, g)
            for t, g in pairs
        }
        for k, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            if not r.empty:
                frames.append(r)
            if k % 10 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(draws: pd.DataFrame, curves: pd.DataFrame, *, seed: int = 1) -> pd.DataFrame:
    """Per-stratum uncertainty columns, aligned on ``curves`` (TFV, greco).

    Gap-filled rows copy the statistics of their donor (same TFV, GRECO =
    ``fill_source``) and get ``boot_source`` = that donor.
    """
    rng = np.random.default_rng(seed)
    ok = draws.loc[draws["ok"]].copy()
    ok["TFV"] = ok["TFV"].astype(str)
    ok["greco"] = ok["greco"].astype(str)
    stats = []
    for (tfv, greco), d in ok.groupby(["TFV", "greco"]):
        rec: dict[str, Any] = {"TFV": tfv, "greco": greco, "boot_n": len(d)}
        for q in QS:
            rec[f"a_se_{q}"] = float(d[f"a_{q}"].std(ddof=1))
        rec["b_se"] = float(d["b"].std(ddof=1))
        rec["c_se"] = float(d["c"].std(ddof=1))
        z = rng.standard_normal(len(d))
        for m in METRICS:
            v = d[f"{m}_50"].to_numpy(float)
            rec[f"{m}_se_fit"] = float(np.nanstd(v, ddof=1))
            rec[f"{m}_ci95_lo_fit"], rec[f"{m}_ci95_hi_fit"] = np.nanpercentile(v, [2.5, 97.5])
            width = np.abs(d[f"{m}_75"].to_numpy(float) - d[f"{m}_25"].to_numpy(float))
            comb = v + z * width / IQR_TO_SD
            rec[f"{m}_q25_combined"], rec[f"{m}_q75_combined"] = np.nanpercentile(comb, [25, 75])
        stats.append(rec)
    stats = pd.DataFrame(stats)

    cur = curves[["TFV", "greco", "gap_filled", "fill_source", "a_25", "a_50", "a_75", "b_50", "c_50"]].copy()
    cur["TFV"] = cur["TFV"].astype(str)
    cur["greco"] = cur["greco"].astype(str)
    cur["boot_source"] = np.where(cur["gap_filled"] == 0, cur["greco"], cur["fill_source"].astype(str))
    out = cur.merge(
        stats.rename(columns={"greco": "boot_source"}), on=["TFV", "boot_source"], how="left"
    )

    # Between-stand spread from the published quantile curves (σ ≈ IQR / 1.349).
    for m in METRICS:
        lo = np.array([curve_metrics(a, b, c)[m] for a, b, c in zip(out["a_25"], out["b_50"], out["c_50"])])
        hi = np.array([curve_metrics(a, b, c)[m] for a, b, c in zip(out["a_75"], out["b_50"], out["c_50"])])
        out[f"{m}_sd_spread"] = np.abs(hi - lo) / IQR_TO_SD
        out[f"{m}_sd_total"] = np.sqrt(out[f"{m}_se_fit"] ** 2 + out[f"{m}_sd_spread"] ** 2)

    cols = ["TFV", "greco", "boot_source", "boot_n", *[f"a_se_{q}" for q in QS], "b_se", "c_se"]
    for m in METRICS:
        cols += [f"{m}_se_fit", f"{m}_ci95_lo_fit", f"{m}_ci95_hi_fit",
                 f"{m}_sd_spread", f"{m}_sd_total", f"{m}_q25_combined", f"{m}_q75_combined"]
    return out[cols]
