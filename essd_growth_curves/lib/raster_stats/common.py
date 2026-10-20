"""Shared polygon / percentile helpers for EFDA and GAMI builders."""

from __future__ import annotations

import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

# Config raster key → column prefix in stats parquets.
# Short names are historical (French "biomasse"); meaning is documented in the README.
STAT_COLUMN_NAMES = {
    "agb": "biomasseM",  # AGB (Mg/ha); column prefix kept for parquet schema
}

DEFAULT_STAT_KEYS = tuple(STAT_COLUMN_NAMES.keys())


def normalize_tfv_label(tfv: Any) -> str:
    """Canonical string label for a TFV code or aggregate name."""
    if tfv is None or (isinstance(tfv, float) and not np.isfinite(tfv)):
        return "nan"
    if isinstance(tfv, str):
        if tfv == "forest":
            return "forest"
        try:
            return str(int(float(tfv)))
        except ValueError:
            return tfv
    if isinstance(tfv, (float, np.floating, int, np.integer)):
        return str(int(tfv))
    return str(tfv)


def is_excluded_tfv(tfv: Any, excluded: Iterable[str]) -> bool:
    return normalize_tfv_label(tfv) in {str(x) for x in excluded}


def calculate_summary_statistics(
    data_dict: dict[str, np.ndarray],
    age_values: np.ndarray,
    group_vars: dict[str, np.ndarray] | None = None,
    *,
    grid: Any = None,
    tfv: Any = None,
    unique_id_count: int | None = None,
    severity: Any = None,
    include_year: bool = True,
) -> pd.DataFrame:
    """Percentiles / count / min / max / sum per age (and optional group vars)."""
    group_vars = group_vars or {}
    data = pd.DataFrame()
    # Column name "year" is historical: EFDA writes calendar disturbance year;
    # GAMI writes stand age (years). Fit preprocess renames/derives `age`.
    if include_year and age_values is not None:
        data["year"] = np.asarray(age_values).ravel()

    for var, values in group_vars.items():
        data[var] = np.asarray(values).ravel()
    for key, values in data_dict.items():
        data[key] = np.asarray(values).ravel()

    required_cols = list(group_vars.keys())
    if include_year and "year" in data.columns:
        required_cols.append("year")
    if required_cols:
        data = data.dropna(subset=required_cols)
    if data.empty:
        return pd.DataFrame()

    group_cols = list(group_vars.keys())
    if include_year and "year" in data.columns:
        group_cols.append("year")

    def safe_percentile(x: pd.Series, q: float) -> float:
        if x.notna().sum() > 0:
            return float(np.nanpercentile(x, q))
        return np.nan

    quantiles = list(range(5, 100, 5))
    agg_dict: dict[str, tuple[str, Any]] = {}
    for key in data_dict:
        agg_dict[f"{key}_count"] = (key, lambda x: int(x.notna().sum()))
        agg_dict[f"{key}_min"] = (key, lambda x: x.min(skipna=True))
        agg_dict[f"{key}_max"] = (key, lambda x: x.max(skipna=True))
        agg_dict[f"{key}_sum"] = (key, lambda x: x.sum(skipna=True))
    for q in quantiles:
        for key in data_dict:
            agg_dict[f"{key}_{q}th"] = (key, lambda x, q=q: safe_percentile(x, q))

    if group_cols:
        summary = data.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    else:
        # Single-row aggregate if somehow no grouping columns
        summary = data.agg(**{k: (c, f) for k, (c, f) in agg_dict.items()})
        summary = summary.to_frame().T

    summary["grid"] = grid  # GRECO region id (from shapefile GRECO / ser_num)
    summary["severity"] = severity
    summary["TFV"] = normalize_tfv_label(tfv)
    summary["unique_ID_count_RS"] = unique_id_count
    return summary


def polygon_to_raster_with_bbox(polygon, reference_raster_path: str | Path):
    """Rasterize a polygon within a grid-aligned bbox of the reference raster."""
    with rasterio.open(reference_raster_path) as ref_raster:
        transform = ref_raster.transform
        resolution = (transform.a, -transform.e)
        raster_bounds = ref_raster.bounds
        polygon_bounds = polygon.bounds

        def snap_to_grid(value, res, offset, snap_function):
            snapped_value = snap_function((value - offset) / res) * res + offset
            return np.round(snapped_value / res) * res

        aligned_bounds = (
            max(raster_bounds.left, snap_to_grid(polygon_bounds[0], resolution[0], transform.c, np.floor)),
            max(raster_bounds.bottom, snap_to_grid(polygon_bounds[1], resolution[1], transform.f, np.floor)),
            min(raster_bounds.right, snap_to_grid(polygon_bounds[2], resolution[0], transform.c, np.ceil)),
            min(raster_bounds.top, snap_to_grid(polygon_bounds[3], resolution[1], transform.f, np.ceil)),
        )
        if aligned_bounds[0] >= aligned_bounds[2] or aligned_bounds[1] >= aligned_bounds[3]:
            raise ValueError("Aligned bounds result in an invalid or empty bounding box.")

        bbox_window = from_bounds(*aligned_bounds, transform=transform)
        height = int(np.round((aligned_bounds[3] - aligned_bounds[1]) / resolution[1]))
        width = int(np.round((aligned_bounds[2] - aligned_bounds[0]) / resolution[0]))
        bbox_transform = rasterio.windows.transform(bbox_window, transform)
        polygon_raster = rasterize(
            [(polygon, 1)],
            out_shape=(height, width),
            transform=bbox_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
    return polygon_raster, bbox_transform, bbox_window


def read_greco_polygons(shapefile_path: str | Path, raster_crs) -> gpd.GeoDataFrame:
    """Load GRECO polygons and dissolve by region id (column ``square_id``)."""
    gdf = gpd.read_file(shapefile_path)
    if "ser_num" in gdf.columns:
        gdf = gdf.rename(columns={"ser_num": "square_id"})
    elif "GRECO" in gdf.columns:
        gdf = gdf.rename(columns={"GRECO": "square_id"})
    elif "square_id" not in gdf.columns:
        raise ValueError(
            f"Shapefile must contain GRECO, ser_num, or square_id columns: {shapefile_path}"
        )
    # Internal name square_id; written to parquet as column ``grid`` (= GRECO id).
    gdf = gdf.dissolve(by="square_id").reset_index()
    return gdf.to_crs(raster_crs)


def build_stat_raster_map(cfg: dict[str, Any]) -> dict[str, Path]:
    """
    Map legacy stats column names → absolute raster paths.

    Uses config keys under rasters.* (e.g. agb → biomasseM column prefix).
    """
    from lib.paths import raster_path

    out: dict[str, Path] = {}
    for config_key, column_name in STAT_COLUMN_NAMES.items():
        try:
            path = raster_path(cfg, config_key)
        except KeyError:
            continue
        if not path.is_file():
            print(f"Note: skipping missing raster rasters.{config_key}: {path}")
            continue
        out[column_name] = path
    return out


def build_mask_raster_map(cfg: dict[str, Any]) -> dict[str, Path]:
    """Supporting rasters needed for masks / IDs (legacy keys)."""
    from lib.paths import raster_path

    mapping = {
        "TFV": "tfv",
        "age_gami": "gami_age_mean",
    }
    out: dict[str, Path] = {}
    for key, config_key in mapping.items():
        try:
            out[key] = raster_path(cfg, config_key)
        except KeyError:
            continue
    # Year map comes from 00_materialize_year (resolve_latest_year_path), not here.
    return out


def run_polygon_jobs(
    *,
    worker: Callable[..., pd.DataFrame],
    jobs: list[dict[str, Any]],
    workers: int,
    output_path: Path,
    desc: str,
) -> pd.DataFrame:
    """Run per-polygon workers and write one parquet (GRECO-scale: small N)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    frames: list[pd.DataFrame] = []
    total = len(jobs)
    print(f"{desc}: {total} polygons, workers={workers}")

    if total == 0:
        empty = pd.DataFrame()
        empty.to_parquet(output_path, index=False)
        return empty

    if workers <= 1 or total == 1:
        for i, job in enumerate(jobs, start=1):
            df = worker(**job)
            if df is not None and not df.empty:
                frames.append(df)
            print(f"{desc}: {i}/{total}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, **job): idx for idx, job in enumerate(jobs)}
            done = 0
            for future in as_completed(futures):
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
                done += 1
                print(f"{desc}: {done}/{total}")

    if not frames:
        result = pd.DataFrame()
    else:
        result = pd.concat(frames, ignore_index=True)
        result["TFV"] = result["TFV"].astype(str)

    result.to_parquet(output_path, index=False)
    print(f"{desc}: wrote {len(result)} rows → {output_path}")
    del frames
    gc.collect()
    return result


def read_window_masked(path: str | Path, window, valid_mask: np.ndarray) -> ma.MaskedArray:
    with rasterio.open(path) as src:
        data = src.read(1, window=window, masked=True)
    return ma.masked_array(data, mask=~valid_mask)
