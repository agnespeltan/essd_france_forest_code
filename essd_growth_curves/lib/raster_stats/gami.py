"""Build GAMI-age GRECO growth statistics (EFDA-free pixels only)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio

from lib.config import excluded_tfv_set
from lib.paths import greco_shapefile, output_dir, stats_path
from lib.raster_stats.agent_stack import (
    efda_free_mask_from_year,
    resolve_latest_year_path,
)
from lib.raster_stats.common import (
    build_mask_raster_map,
    build_stat_raster_map,
    calculate_summary_statistics,
    is_excluded_tfv,
    polygon_to_raster_with_bbox,
    read_greco_polygons,
    run_polygon_jobs,
)


def process_gami_polygon(
    raster_paths: dict[str, str],
    selected_stat_names: list[str],
    polygon,
    square_id: Any,
    excluded_tfv: list[str],
    include_year: bool = True,
) -> pd.DataFrame:
    """Summarize AGB by GAMI age and TFV (EFDA-free pixels only)."""
    reference = raster_paths["TFV"]
    polygon_raster, _, bbox_window = polygon_to_raster_with_bbox(polygon, reference)

    with rasterio.open(raster_paths["TFV"]) as src_tfv, \
         rasterio.open(raster_paths["age_gami"]) as src_age:

        tfv = src_tfv.read(1, window=bbox_window, masked=True)
        age = src_age.read(1, window=bbox_window, masked=True)
        efda_free = efda_free_mask_from_year(raster_paths["year_efda"], bbox_window)

        valid = (
            efda_free
            & (~age.mask)
            & (age > 0)
            & (~tfv.mask)
            & (polygon_raster == 1)
        )
        tfv_m = ma.masked_array(tfv, mask=~valid)

    data_dict: dict[str, ma.MaskedArray] = {}
    for name in selected_stat_names:
        with rasterio.open(raster_paths[name]) as src:
            data_dict[name] = ma.masked_array(
                src.read(1, window=bbox_window, masked=True), mask=~valid
            )

    frames: list[pd.DataFrame] = []
    vals = np.asarray(tfv_m)
    ok = (~np.ma.getmaskarray(tfv_m)) & np.isfinite(vals)
    for tfv_code in np.unique(vals[ok]):
        if is_excluded_tfv(tfv_code, excluded_tfv):
            continue
        sel = valid & (tfv_m == tfv_code)
        if not np.any(sel):
            continue
        summary = calculate_summary_statistics(
            data_dict={k: v[sel] for k, v in data_dict.items()},
            age_values=age[sel],
            group_vars={},
            grid=square_id,
            tfv=tfv_code,
            unique_id_count=None,
            severity=None,
            include_year=include_year,
        )
        if not summary.empty:
            frames.append(summary)

    del age, data_dict, tfv_m, valid
    gc.collect()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_gami_stats(cfg: dict[str, Any], *, max_polygons: int | None = None) -> Path:
    """Write growth_stats_gami_greco.parquet (path from config)."""
    year_path = resolve_latest_year_path(cfg)
    mask_paths = build_mask_raster_map(cfg)
    mask_paths["year_efda"] = year_path
    stat_paths = build_stat_raster_map(cfg)

    required = ["TFV", "age_gami", "year_efda"]
    missing = [k for k in required if k not in mask_paths]
    if missing:
        raise KeyError(f"GAMI builder missing rasters: {missing}")
    if not stat_paths:
        raise KeyError("No AGB raster configured (rasters.agb).")

    raster_paths = {
        **{k: str(v) for k, v in mask_paths.items()},
        **{k: str(v) for k, v in stat_paths.items()},
    }
    print(f"GAMI EFDA-free mask from: {year_path}")

    with rasterio.open(raster_paths["TFV"]) as src:
        raster_crs = src.crs
    gdf = read_greco_polygons(greco_shapefile(cfg), raster_crs)
    if max_polygons is not None:
        gdf = gdf.iloc[: max(0, int(max_polygons))].copy()

    jobs = [
        {
            "raster_paths": raster_paths,
            "selected_stat_names": list(stat_paths.keys()),
            "polygon": row.geometry,
            "square_id": row["square_id"],
            "excluded_tfv": sorted(excluded_tfv_set(cfg)),
            "include_year": True,
        }
        for _, row in gdf.iterrows()
    ]

    out = stats_path(cfg, "gami")
    output_dir(cfg).mkdir(parents=True, exist_ok=True)
    run_polygon_jobs(
        worker=process_gami_polygon,
        jobs=jobs,
        workers=int(cfg.get("workers", 4)),
        output_path=out,
        desc="build-gami",
    )
    return out
