"""Load figure inputs from ``config.figures`` (no hardcoded machine paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from lib.constants import GRECO_GROUPS, PIXEL_AREA_MHA, TFV_GROUP
from lib.figures.constants import DICT_AGE


def _fig(cfg: dict[str, Any]) -> dict[str, Any]:
    figs = cfg.get("figures")
    if not isinstance(figs, dict) or not figs:
        raise KeyError("config.figures mapping is required for make_figures")
    return figs


def _req(figs: dict[str, Any], key: str) -> Path:
    if key not in figs or not figs[key]:
        raise KeyError(f"Missing figures.{key} in config")
    return Path(figs[key]).expanduser()


def _optional_path(figs: dict[str, Any], key: str) -> Path | None:
    """Resolve figures.<key> if set and present on disk; else None."""
    raw = figs.get(key)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def load_age_hist_csv(path: Path | str) -> pd.DataFrame:
    """RS age histogram; attach area (Mha). Remap TFV 0→100, 400→401."""
    df = pd.read_csv(path)
    df = df.loc[df["age"].astype(str) != "NODATA"].copy()
    df["age"] = df["age"].astype(int)
    if "tfv" in df.columns:
        df["tfv"] = pd.to_numeric(df["tfv"], errors="coerce")
        df["tfv"] = df["tfv"].replace({0: 100, 400: 401})
        group_cols = [c for c in ("tfv", "ser", "age") if c in df.columns]
        df = df.groupby(group_cols, as_index=False, observed=True)["count"].sum()
    if "count" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "count"})
    df["area"] = df["count"] * PIXEL_AREA_MHA
    return df


def load_gami_class_hist(path: Path | str, *, id_key: str) -> pd.DataFrame:
    """GAMI age hist NPZ (by TFV or GRECO) → long table ``id_key, age, count, area``.

    NPZ keys: ``tfv`` or ``greco``, ``age``, ``counts`` [n_class × n_age], optional
    ``pixel_area_ha``.
    """
    p = Path(path).expanduser()
    if p.suffix.lower() == ".csv":
        df = load_age_hist_csv(p)
        if id_key == "GRECO" and "GRECO" not in df.columns and "greco" in df.columns:
            df = df.rename(columns={"greco": "GRECO"})
        return df

    data = np.load(p, allow_pickle=True)
    pix_ha = float(data["pixel_area_ha"]) if "pixel_area_ha" in data.files else 0.09
    ages = np.asarray(data["age"])
    counts = np.asarray(data["counts"])
    if "tfv" in data.files:
        ids = [int(x) for x in np.asarray(data["tfv"]).tolist()]
        out_key = "tfv"
    elif "greco" in data.files:
        ids = [str(x) for x in np.asarray(data["greco"]).tolist()]
        out_key = "GRECO"
    else:
        raise KeyError(f"No tfv/greco key in {p}")
    if id_key not in (out_key, "tfv", "GRECO"):
        pass  # keep out_key from file
    rows = []
    for i, cid in enumerate(ids):
        c = counts[i]
        m = c > 0
        if not np.any(m):
            continue
        rows.append(
            pd.DataFrame(
                {
                    out_key: cid,
                    "age": ages[m].astype(int),
                    "count": c[m].astype(np.int64),
                    "area": c[m].astype(np.float64) * pix_ha / 1e6,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=[out_key, "age", "count", "area"])
    return pd.concat(rows, ignore_index=True)


def load_ours_minus_gami_tfv_stats(path: Path | str) -> pd.DataFrame:
    """Build TFV bubble stats (q25/q50/q75/area) from Ours−GAMI diff hist NPZ."""
    p = Path(path).expanduser()
    data = np.load(p, allow_pickle=True)
    edges = np.asarray(data["bin_edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.asarray(data["counts_all"])
    pix_ha = float(data["pixel_area_ha"]) if "pixel_area_ha" in data.files else 0.09
    rows = []
    for i, tfv in enumerate(np.asarray(data["tfv"]).tolist()):
        w = counts[i].astype(float)
        if not np.any(w > 0):
            continue
        q25, q50, q75 = weighted_quantiles(centers, w)
        rows.append(
            {
                "tfv": int(tfv),
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
                "area": float(w.sum() * pix_ha / 1e6),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["tfv_group"] = out["tfv"].map(lambda t: TFV_GROUP.get(int(t)))
    return out


def load_gami_national_hist(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GAMI national hist CSV with value / value_lower / value_upper columns."""
    simon = pd.read_csv(path)
    simon = simon[simon["age"].astype(str) != "NODATA"].copy()
    simon["age"] = simon["age"].astype(int)
    mean = simon[["age", "value"]].rename(columns={"value": "count"}).copy()
    lo = simon[["age", "value_lower"]].rename(columns={"value_lower": "count"}).copy()
    hi = simon[["age", "value_upper"]].rename(columns={"value_upper": "count"}).copy()
    for df in (mean, lo, hi):
        df["area"] = df["count"] * PIXEL_AREA_MHA
        # age=0 is nodata/non-forest in GAMI national hist
        m = df["age"] > 0
        df.drop(df.index[~m], inplace=True)
    return mean, lo, hi


def load_nfi_aires_age(path: Path | str) -> pd.DataFrame:
    """National NFI forest area by age (Aires_age-style CSV)."""
    nfi = pd.read_csv(path, header=10, delimiter=";", skipfooter=1, engine="python")
    st = "ST (milliers d'hectares)"
    err = "ST - ½ largeur de l'IC 95 (milliers d'hectares)"
    nfi = nfi.loc[nfi[st].astype(str).str.strip() != "n.s."].copy()
    nfi[st] = nfi[st].astype(str).str.replace(" ", "", regex=False).astype(float)
    nfi["area"] = nfi[st] * 1e3 / 1e6
    nfi["area_err"] = 2.0 * pd.to_numeric(nfi[err], errors="coerce").fillna(0.0) * 1e3 / 1e6
    nfi["age"] = nfi["CAC - CAC"].map(DICT_AGE)
    out = nfi.loc[nfi["age"].notna() & (nfi["age"] >= 0), ["age", "area", "area_err"]].copy()
    return out


def load_nfi_surf_cacgk(
    path: Path | str, class_col: str, class_out: str, header: int
) -> pd.DataFrame:
    """NFI surface × CACGK age table."""
    df = pd.read_csv(path, header=header, delimiter=";", skipfooter=1, engine="python")
    st_col = "ST (milliers d'hectares)"
    err_col = "ST - ½ largeur de l'IC 95 (milliers d'hectares)"
    df = df.loc[df[st_col].astype(str).str.strip() != "n.s."].copy()
    df[st_col] = df[st_col].astype(str).str.replace(" ", "", regex=False).astype(float)
    df[err_col] = pd.to_numeric(df[err_col], errors="coerce").fillna(0.0)
    out = pd.DataFrame(
        {
            class_out: df[class_col].astype(str).str.strip(),
            "age": df["CAC - CACGK"].map(DICT_AGE),
            "area": df[st_col] * 1e3 / 1e6,
            "area_err": 2.0 * df[err_col] * 1e3 / 1e6,
        }
    )
    return out.loc[out["age"].notna() & (out["age"] >= 0)].copy()


def weighted_quantiles(values, weights, probs=(0.25, 0.5, 0.75)):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.array([np.nan] * len(probs), dtype=float)
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(probs, cdf, values)


def calc_stats(x: pd.DataFrame) -> pd.Series:
    w = x["count"].to_numpy() if "count" in x.columns else x["area"].to_numpy()
    ages = x["age"].to_numpy()
    q25, q50, q75 = weighted_quantiles(ages, w)
    area = float(x["area"].sum()) if "area" in x.columns else float(w.sum() * PIXEL_AREA_MHA)
    return pd.Series({"q25": q25, "q50": q50, "q75": q75, "area": area})


def annotate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add GRECO letter, greco_group, tfv_group."""
    out = df.copy()
    if "GRECO" not in out.columns:
        ser_col = "ser" if "ser" in out.columns else ("ser_num" if "ser_num" in out.columns else None)
        if ser_col is not None:
            out["GRECO"] = out[ser_col] // 100
            out["GRECO"] = out["GRECO"].map(
                lambda x: chr(64 + int(x)) if 1 <= int(x) <= 26 else "Z"
            )
        else:
            out["GRECO"] = "A"
    out["GRECO"] = out["GRECO"].astype(str)
    out["greco_group"] = out["GRECO"].map(GRECO_GROUPS)
    if "tfv" in out.columns:
        out["tfv_group"] = out["tfv"].map(lambda t: TFV_GROUP.get(int(t)) if pd.notna(t) else None)
    if "area" not in out.columns and "count" in out.columns:
        out["area"] = out["count"] * PIXEL_AREA_MHA
    return out


def load_masked_age_5km(age_5km: Path, tree_cover: Path):
    """5 km age map (incl. age=0) masked where tree cover ≤ 10%."""
    with rasterio.open(age_5km) as src_age:
        age = src_age.read(1).astype(np.float64)
        bounds = src_age.bounds
        transform = src_age.transform
        crs = src_age.crs
        nodata = src_age.nodata
    with rasterio.open(tree_cover) as src_tc:
        tc = src_tc.read(1).astype(np.float64)
        if tc.shape != age.shape:
            tc_res = np.empty_like(age)
            reproject(
                source=rasterio.band(src_tc, 1),
                destination=tc_res,
                src_transform=src_tc.transform,
                src_crs=src_tc.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            tc = tc_res
    if nodata is not None:
        age[age == nodata] = np.nan
    tc_mask = np.isnan(tc) | (tc <= 0.10)
    # Keep age=0 (recent clearcuts); only hide nodata / non-forest.
    masked = np.ma.masked_where((age == -1) | (age < 0) | np.isnan(age) | tc_mask, age)
    return masked, bounds


def load_masked_metric_5km(metric_tif: Path, tree_cover: Path, *, as_percent: bool = False):
    """Generic 5 km metric (rel unc / CV / diff) with tree-cover mask."""
    with rasterio.open(metric_tif) as src:
        data = src.read(1).astype(np.float64)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    with rasterio.open(tree_cover) as src_tc:
        tc = src_tc.read(1).astype(np.float64)
        if tc.shape != data.shape:
            tc_res = np.empty_like(data)
            reproject(
                source=rasterio.band(src_tc, 1),
                destination=tc_res,
                src_transform=src_tc.transform,
                src_crs=src_tc.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            tc = tc_res
    if nodata is not None:
        data[data == nodata] = np.nan
    tc_mask = np.isnan(tc) | (tc <= 0.10)
    masked = np.ma.masked_where(~np.isfinite(data) | (data == -1) | tc_mask, data)
    if as_percent:
        masked = np.ma.abs(masked) * 100.0
    return masked, bounds


def ensure_rel_unc_5km(cfg: dict[str, Any], cache_dir: Path) -> Path:
    """|(p75−p25)/p50| at 5 km from age_lo/main/hi_5km."""
    figs = _fig(cfg)
    if figs.get("rel_unc_5km"):
        return Path(figs["rel_unc_5km"]).expanduser()
    out = cache_dir / "age_rel_unc_5km.tif"
    if out.exists():
        return out
    main_p = Path(figs.get("age_main_5km") or figs["age_main"]).expanduser()
    lo_p = Path(figs.get("age_lo_5km") or figs["age_lo"]).expanduser()
    hi_p = Path(figs.get("age_hi_5km") or figs["age_hi"]).expanduser()
    with rasterio.open(main_p) as s0, rasterio.open(lo_p) as s1, rasterio.open(hi_p) as s2:
        main = s0.read(1).astype(np.float64)
        lo = s1.read(1).astype(np.float64)
        hi = s2.read(1).astype(np.float64)
        profile = s0.profile.copy()
        nodata = s0.nodata
    for arr in (main, lo, hi):
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr < 0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (hi - lo) / main
    # age=0 with zero IQR → 0% relative unc; otherwise leave non-finite as nodata
    zero_main = (main == 0) & np.isfinite(lo) & np.isfinite(hi) & (hi == lo)
    rel[zero_main] = 0.0
    rel[~np.isfinite(rel)] = np.nan
    profile.update(dtype="float32", nodata=-1.0, count=1, compress="deflate")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_arr = np.where(np.isfinite(rel), rel, -1.0).astype(np.float32)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(out_arr, 1)
    return out


def load_figure_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Bundle all inputs needed by ``run_figures``."""
    figs = _fig(cfg)
    rs_main = load_age_hist_csv(_req(figs, "rs_hist_main"))
    rs_lo = load_age_hist_csv(_req(figs, "rs_hist_lo"))
    rs_hi = load_age_hist_csv(_req(figs, "rs_hist_hi"))

    gami_hist = _req(figs, "gami_hist")
    gami_mean, gami_lo, gami_hi = load_gami_national_hist(gami_hist)

    # NFI paths are optional — omit/unset → RS(+GAMI)-only Fig. 3a; skip SI4/SI5.
    nfi_aires = _optional_path(figs, "nfi_aires_age")
    nfi_forest = (
        load_nfi_aires_age(nfi_aires)
        if nfi_aires is not None
        else pd.DataFrame(columns=["age", "area", "area_err"])
    )
    nfi_ess_path = _optional_path(figs, "nfi_surf_esspre")
    nfi_ess = (
        load_nfi_surf_cacgk(nfi_ess_path, "ESSPRE - ESSD15", "ESS", header=10)
        if nfi_ess_path is not None
        else pd.DataFrame(columns=["ESS", "age", "area", "area_err"])
    )
    nfi_greco_path = _optional_path(figs, "nfi_surf_greco")
    if nfi_greco_path is not None:
        nfi_greco = load_nfi_surf_cacgk(
            nfi_greco_path, "Code  GRECO", "GRECO", header=11
        )
        raw_g = pd.read_csv(
            nfi_greco_path, header=11, delimiter=";", skipfooter=1, engine="python"
        )
        name_map = (
            raw_g[["Code  GRECO", "Nom  GRECO"]]
            .drop_duplicates()
            .assign(
                **{
                    "Code  GRECO": lambda d: d["Code  GRECO"].astype(str).str.strip(),
                    "Nom  GRECO": lambda d: d["Nom  GRECO"].astype(str).str.strip(),
                }
            )
            .set_index("Code  GRECO")["Nom  GRECO"]
            .to_dict()
        )
        nfi_greco["GRECO_name"] = nfi_greco["GRECO"].map(name_map)
    else:
        nfi_greco = pd.DataFrame(columns=["GRECO", "age", "area", "area_err", "GRECO_name"])

    if nfi_forest.empty and nfi_ess.empty and nfi_greco.empty:
        print("  NFI CSVs unset/missing — Fig. 3a without NFI; SI4/SI5 skipped")

    france = gpd.read_file(_req(figs, "france_shapefile"))
    greco = gpd.read_file(Path(cfg["greco_shapefile"]).expanduser())

    age_5km = Path(figs.get("age_main_5km") or figs["age_main"]).expanduser()
    tree_cover = _req(figs, "tree_cover")
    masked_age, age_bounds = load_masked_age_5km(age_5km, tree_cover)

    cache = Path(cfg.get("output_dir", ".")).expanduser() / "_figure_cache"
    rel_unc_path = ensure_rel_unc_5km(cfg, cache)
    masked_unc, unc_bounds = load_masked_metric_5km(rel_unc_path, tree_cover, as_percent=True)

    ctx: dict[str, Any] = {
        "rs_main": rs_main,
        "rs_lo": rs_lo,
        "rs_hi": rs_hi,
        "gami_mean": gami_mean,
        "gami_lo": gami_lo,
        "gami_hi": gami_hi,
        "nfi_forest": nfi_forest,
        "nfi_ess": nfi_ess,
        "nfi_greco": nfi_greco,
        "france_gdf": france,
        "greco_gdf": greco,
        "masked_age": masked_age,
        "age_bounds": age_bounds,
        "masked_unc": masked_unc,
        "unc_bounds": unc_bounds,
        "tree_cover": tree_cover,
        "figs": figs,
        "cache_dir": cache,
    }

    # Optional SI3 panels (b, c)
    if figs.get("age_cv_5km"):
        ctx["masked_cv"], ctx["cv_bounds"] = load_masked_metric_5km(
            Path(figs["age_cv_5km"]).expanduser(), tree_cover, as_percent=True
        )
    if figs.get("ours_minus_gami_5km"):
        ctx["masked_diff"], ctx["diff_bounds"] = load_masked_metric_5km(
            Path(figs["ours_minus_gami_5km"]).expanduser(), tree_cover, as_percent=False
        )
    if figs.get("ours_minus_gami_tfv_csv"):
        p = Path(figs["ours_minus_gami_tfv_csv"]).expanduser()
        ctx["diff_tfv_stats"] = (
            load_ours_minus_gami_tfv_stats(p)
            if p.suffix == ".npz"
            else pd.read_csv(p)
        )
    if figs.get("gami_hist_by_tfv"):
        ctx["gami_by_tfv"] = load_gami_class_hist(
            figs["gami_hist_by_tfv"], id_key="tfv"
        )
        if figs.get("gami_hist_by_tfv_lo"):
            ctx["gami_by_tfv_lo"] = load_gami_class_hist(
                figs["gami_hist_by_tfv_lo"], id_key="tfv"
            )
        if figs.get("gami_hist_by_tfv_hi"):
            ctx["gami_by_tfv_hi"] = load_gami_class_hist(
                figs["gami_hist_by_tfv_hi"], id_key="tfv"
            )
    if figs.get("gami_hist_by_greco"):
        ctx["gami_by_greco"] = load_gami_class_hist(
            figs["gami_hist_by_greco"], id_key="GRECO"
        )
        if figs.get("gami_hist_by_greco_lo"):
            ctx["gami_by_greco_lo"] = load_gami_class_hist(
                figs["gami_hist_by_greco_lo"], id_key="GRECO"
            )
        if figs.get("gami_hist_by_greco_hi"):
            ctx["gami_by_greco_hi"] = load_gami_class_hist(
                figs["gami_hist_by_greco_hi"], id_key="GRECO"
            )

    return ctx
