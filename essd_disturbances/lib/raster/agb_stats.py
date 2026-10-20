"""Build AGB quantile stats by disturbance year-window from latest year+agent maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from lib.config import output_path
from lib.raster.agent_stack import (
    AGENT_NODATA,
    YEAR_NODATA,
    resolve_latest_agent_path,
    resolve_latest_year_path,
)
from lib.raster.geometry import (
    parse_inclusive_year_range,
    polygon_to_raster_with_bbox,
    read_ser_polygons,
)
from lib.raster.jobs import run_polygon_jobs
from lib.raster.patches import _FOREST_TFV, _FOREST_TFV_ARR


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _process_ser_agb(
    *,
    tfv_path: str,
    agb_path: str,
    latest_year_path: str,
    latest_agent_path: str,
    windows: list[tuple[str, np.ndarray]],
    before_range: str,
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

    with rasterio.open(agb_path) as agb_src:
        agb = agb_src.read(1, window=bbox_window).astype(float)
        agb_nodata = agb_src.nodata
        if agb_nodata is not None:
            agb = np.where(agb == agb_nodata, np.nan, agb)

    with rasterio.open(latest_year_path) as ly_src:
        latest = ly_src.read(1, window=bbox_window).astype(np.int32)
        ly_nodata = ly_src.nodata if ly_src.nodata is not None else YEAR_NODATA
        latest = np.where(latest == ly_nodata, -1, latest)

    with rasterio.open(latest_agent_path) as la_src:
        agent = la_src.read(1, window=bbox_window)
        la_nodata = la_src.nodata if la_src.nodata is not None else AGENT_NODATA

    results: list[pd.DataFrame] = []
    for label, years in windows:
        year_set = set(int(y) for y in years)
        mask_valid = (
            base_mask
            & (agent != la_nodata)
            & np.isin(agent, [1, 2, 3])
            & np.isin(latest, list(year_set))
            & np.isfinite(agb)
        )
        # Before cohort = pre-disturbance reference: drop AGB == 0 pixels
        if label == before_range:
            mask_valid = mask_valid & (agb > 0)
        if not np.any(mask_valid):
            continue

        tfv_v = tfv[mask_valid]
        agent_v = agent[mask_valid].astype(np.int64, copy=False)
        # 1+2 → unplanned, 3 → harvest
        agent_class = np.where(agent_v == 3, "harvest", "unplanned")
        agb_v = agb[mask_valid]

        data = pd.DataFrame(
            {
                "TFV": tfv_v,
                "agent_class": agent_class,
                "agb": agb_v,
            }
        )

        rows = []
        for (tfv_val, ac), g in data.groupby(["TFV", "agent_class"], sort=False):
            stats = _quantile_summary(g["agb"].to_numpy(dtype=float))
            tfv_i = int(tfv_val)
            vpc = int(tfv_bincount[tfv_i]) if 0 <= tfv_i < len(tfv_bincount) else 0
            stats.update(
                {
                    "TFV": tfv_val,
                    "agent_class": ac,
                    "year_range": label,
                    "grid": square_id,
                    "valid_pixel_count": vpc,
                }
            )
            rows.append(stats)
        if rows:
            results.append(pd.DataFrame(rows))

        if include_forest:
            forest = data.loc[data["TFV"].isin(list(_FOREST_TFV))]
            if not forest.empty:
                f_rows = []
                for ac, g in forest.groupby("agent_class", sort=False):
                    stats = _quantile_summary(g["agb"].to_numpy(dtype=float))
                    stats.update(
                        {
                            "TFV": "Forest",
                            "agent_class": ac,
                            "year_range": label,
                            "grid": square_id,
                            "valid_pixel_count": forest_pixel_count,
                        }
                    )
                    f_rows.append(stats)
                results.append(pd.DataFrame(f_rows))

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out["TFV"] = out["TFV"].astype(str)
    return out


def build_agb_stats(cfg: dict[str, Any], *, max_polygons: int | None = None) -> Path:
    """Build AGB year-range stats parquet (before/after windows)."""
    rasters = cfg["rasters"]
    tfv_path = str(Path(rasters["tfv"]).expanduser())
    agb_path = str(Path(rasters["agb"]).expanduser())
    ser_shp = Path(cfg["ser_shapefile"]).expanduser()
    latest_year_path = str(resolve_latest_year_path(cfg))
    latest_agent_path = str(resolve_latest_agent_path(cfg))

    workers = int(cfg.get("workers", 4))
    include_forest = bool(cfg.get("include_forest_aggregate", True))

    before_range = str(cfg["before_range"])
    windows: list[tuple[str, np.ndarray]] = []
    for key in ("after_range", "before_range"):
        label = str(cfg[key])
        _, _, years = parse_inclusive_year_range(label)
        windows.append((label, np.asarray(years, dtype=np.int32)))

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
            "agb_path": agb_path,
            "latest_year_path": latest_year_path,
            "latest_agent_path": latest_agent_path,
            "windows": windows,
            "before_range": before_range,
            "polygon": row.geometry,
            "square_id": row["square_id"],
            "include_forest": include_forest,
        }
        for _, row in gdf.iterrows()
    ]

    out = output_path(cfg, "agb_stats")
    cfg["agb_stats_parquet"] = str(out)
    return run_polygon_jobs(
        worker=_process_ser_agb,
        jobs=jobs,
        workers=workers,
        output_path=out,
        desc="build_agb_stats",
    )
