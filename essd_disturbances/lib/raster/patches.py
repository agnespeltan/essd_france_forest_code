"""Build annual disturbance patch stats from the multilayer agent stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from lib.config import output_path
from lib.raster.geometry import agent_band_index, polygon_to_raster_with_bbox, read_ser_polygons
from lib.raster.jobs import run_polygon_jobs

_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)

# TFVs pooled into the synthetic "Forest" aggregate (exclude open/other codes)
_FOREST_TFV = frozenset(
    {
        1, 6, 9, 10, 14, 19, 49, 100, 111, 51, 52, 53, 57, 58, 81, 80, 61, 63, 64,
        91, 90, 200, 222, 310, 320, 401, 402, 403,
    }
)
_FOREST_TFV_ARR = np.fromiter(_FOREST_TFV, dtype=np.int64)


def _patch_stats_from_sizes(sizes: np.ndarray) -> tuple[int, float, float]:
    """patch_count, mean, std from component sizes (same variance as before)."""
    n = int(sizes.size)
    if n == 0:
        return 0, 0.0, 0.0
    total = float(sizes.sum())
    mean = total / n
    var = float(np.dot(sizes, sizes) / n) - (mean * mean)
    std = float(np.sqrt(var)) if var > 0 else 0.0
    return n, mean, std


def group_patch_stats(
    tfv: np.ndarray,
    agent: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Vectorized pixel counts + 8-connected patch stats per (TFV, agent).

    Encodes groups with ``np.unique``, then one ``ndimage.label`` per group
    (required for equal-value connectivity) and ``bincount`` for sizes.
    """
    cols = [
        "TFV",
        "agent_class",
        "pixel_count",
        "patch_count",
        "mean_patch_size",
        "std_patch_size",
    ]
    if not np.any(mask):
        return pd.DataFrame(columns=cols)

    tfv_m = tfv[mask].astype(np.int64, copy=False)
    agent_m = agent[mask].astype(np.int64, copy=False)
    # Pack (tfv, agent) — agent is 1 or 3
    packed = (tfv_m << 2) | agent_m
    uniq_p, inv = np.unique(packed, return_inverse=True)
    n_g = int(uniq_p.size)
    pixel_count = np.bincount(inv, minlength=n_g)

    code = np.zeros(tfv.shape, dtype=np.int32)
    code[mask] = inv + 1

    patch_count = np.zeros(n_g, dtype=np.int32)
    mean_size = np.zeros(n_g, dtype=np.float64)
    std_size = np.zeros(n_g, dtype=np.float64)

    for g in range(n_g):
        labeled, nfeat = ndimage.label(code == (g + 1), structure=_CONNECTIVITY_8)
        if nfeat == 0:
            continue
        sizes = np.bincount(labeled.ravel(), minlength=nfeat + 1)[1 : nfeat + 1].astype(
            np.float64, copy=False
        )
        pc, mu, sd = _patch_stats_from_sizes(sizes)
        patch_count[g] = pc
        mean_size[g] = mu
        std_size[g] = sd

    return pd.DataFrame(
        {
            "TFV": (uniq_p >> 2).astype(np.int64),
            "agent_class": (uniq_p & 3).astype(np.int64),
            "pixel_count": pixel_count,
            "patch_count": patch_count,
            "mean_patch_size": mean_size,
            "std_patch_size": std_size,
        }
    )


def agent_only_patch_stats(agent: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Patch stats grouped by agent only (Forest aggregate)."""
    cols = [
        "agent_class",
        "pixel_count",
        "patch_count",
        "mean_patch_size",
        "std_patch_size",
    ]
    if not np.any(mask):
        return pd.DataFrame(columns=cols)

    agent_m = agent[mask].astype(np.int64, copy=False)
    uniq, inv = np.unique(agent_m, return_inverse=True)
    n_g = int(uniq.size)
    pixel_count = np.bincount(inv, minlength=n_g)

    code = np.zeros(agent.shape, dtype=np.int32)
    code[mask] = inv + 1

    patch_count = np.zeros(n_g, dtype=np.int32)
    mean_size = np.zeros(n_g, dtype=np.float64)
    std_size = np.zeros(n_g, dtype=np.float64)

    for g in range(n_g):
        labeled, nfeat = ndimage.label(code == (g + 1), structure=_CONNECTIVITY_8)
        if nfeat == 0:
            continue
        sizes = np.bincount(labeled.ravel(), minlength=nfeat + 1)[1 : nfeat + 1].astype(
            np.float64, copy=False
        )
        pc, mu, sd = _patch_stats_from_sizes(sizes)
        patch_count[g] = pc
        mean_size[g] = mu
        std_size[g] = sd

    return pd.DataFrame(
        {
            "agent_class": uniq,
            "pixel_count": pixel_count,
            "patch_count": patch_count,
            "mean_patch_size": mean_size,
            "std_patch_size": std_size,
        }
    )


def _process_ser_patches(
    *,
    tfv_path: str,
    agent_path: str,
    agent_first_year: int,
    agent_nodata: int,
    year_start: int,
    year_end: int,
    polygon,
    square_id,
    include_forest: bool,
) -> pd.DataFrame:
    polygon_raster, _, bbox_window = polygon_to_raster_with_bbox(polygon, tfv_path)

    with rasterio.open(tfv_path) as tfv_src:
        tfv = tfv_src.read(1, window=bbox_window)
        base_mask = (tfv != tfv_src.nodata) & (polygon_raster == 1)
        tfv_on = tfv[base_mask].astype(np.int64, copy=False)
        if tfv_on.size:
            tfv_bincount = np.bincount(tfv_on)
        else:
            tfv_bincount = np.zeros(1, dtype=np.int64)
        forest_pixel_count = int(np.isin(tfv_on, _FOREST_TFV_ARR).sum())

    results: list[pd.DataFrame] = []
    with rasterio.open(agent_path) as agent_src:
        nodata = agent_src.nodata if agent_src.nodata is not None else agent_nodata
        for year in range(year_start, year_end + 1):
            band = agent_band_index(year, agent_first_year)
            if band < 1 or band > agent_src.count:
                continue
            agent = agent_src.read(band, window=bbox_window)
            # Collapse fire (2) into natural (1); keep harvest (3)
            agent_work = np.where(agent == 2, 1, agent)
            mask_valid = base_mask & (agent != nodata) & np.isin(agent_work, [1, 3])

            summary = group_patch_stats(tfv, agent_work, mask_valid)
            if summary.empty:
                continue

            summary["year"] = year
            summary["grid"] = square_id
            # vectorized valid_pixel_count lookup
            tfv_ids = summary["TFV"].to_numpy(dtype=np.int64)
            vpc = np.zeros(len(summary), dtype=np.int64)
            ok = (tfv_ids >= 0) & (tfv_ids < len(tfv_bincount))
            vpc[ok] = tfv_bincount[tfv_ids[ok]]
            summary["valid_pixel_count"] = vpc
            results.append(summary)

            if include_forest:
                forest_mask = mask_valid & np.isin(tfv, _FOREST_TFV_ARR)
                forest_summary = agent_only_patch_stats(agent_work, forest_mask)
                if not forest_summary.empty:
                    forest_summary["TFV"] = "Forest"
                    forest_summary["year"] = year
                    forest_summary["grid"] = square_id
                    forest_summary["valid_pixel_count"] = forest_pixel_count
                    results.append(forest_summary)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["TFV"] = out["TFV"].astype(str)
    out["agent_class"] = out["agent_class"].astype(str)
    return out


def build_patches(cfg: dict[str, Any], *, max_polygons: int | None = None) -> Path:
    """Build patches parquet from multilayer agent + TFV + SER shapefile."""
    rasters = cfg["rasters"]
    tfv_path = str(Path(rasters["tfv"]).expanduser())
    agent_path = str(Path(rasters["agent_multilayer"]).expanduser())
    ser_shp = Path(cfg["ser_shapefile"]).expanduser()

    agent_first_year = int(cfg.get("agent_first_year", 1985))
    agent_nodata = int(cfg.get("agent_nodata", 255))
    year_start = int(cfg["year_start"])
    year_end = int(cfg["year_end"])
    workers = int(cfg.get("workers", 4))
    include_forest = bool(cfg.get("include_forest_aggregate", True))

    with rasterio.open(tfv_path) as src:
        raster_crs = src.crs
    if not ser_shp.is_file():
        raise FileNotFoundError(f"ser_shapefile required: {ser_shp}")
    gdf = read_ser_polygons(ser_shp, raster_crs)
    if max_polygons is not None:
        gdf = gdf.iloc[: int(max_polygons)].copy()

    jobs = [
        {
            "tfv_path": tfv_path,
            "agent_path": agent_path,
            "agent_first_year": agent_first_year,
            "agent_nodata": agent_nodata,
            "year_start": year_start,
            "year_end": year_end,
            "polygon": row.geometry,
            "square_id": row["square_id"],
            "include_forest": include_forest,
        }
        for _, row in gdf.iterrows()
    ]

    out = output_path(cfg, "patches")
    cfg["patches_parquet"] = str(out)
    return run_polygon_jobs(
        worker=_process_ser_patches,
        jobs=jobs,
        workers=workers,
        output_path=out,
        desc="build_patches",
    )
