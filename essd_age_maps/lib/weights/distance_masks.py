"""Distance-based GRECO weight rasters (tanh soft edges)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from lib.config import raster_path, weights_dir
from lib.constants import WEIGHT_NODATA
from lib.naming import weight_filename


def create_distance_normalized_mask(polygon, reference_raster, max_distance_pixels):
    """Windowed tanh weight mask for one GRECO polygon."""
    transform = reference_raster.transform
    raster_bounds = reference_raster.bounds
    polygon_bounds = polygon.bounds
    pixel_size = abs(transform.a)
    buffer_dist = max_distance_pixels * pixel_size

    def snap_to_grid(value, resolution, offset, snap_func):
        return np.round(snap_func((value - offset) / resolution) * resolution + offset, 6)

    resolution = (transform.a, -transform.e)
    aligned_bounds = (
        max(
            raster_bounds.left,
            snap_to_grid(polygon_bounds[0] - buffer_dist, resolution[0], transform.c, np.floor),
        ),
        max(
            raster_bounds.bottom,
            snap_to_grid(polygon_bounds[1] - buffer_dist, resolution[1], transform.f, np.floor),
        ),
        min(
            raster_bounds.right,
            snap_to_grid(polygon_bounds[2] + buffer_dist, resolution[0], transform.c, np.ceil),
        ),
        min(
            raster_bounds.top,
            snap_to_grid(polygon_bounds[3] + buffer_dist, resolution[1], transform.f, np.ceil),
        ),
    )
    if aligned_bounds[0] >= aligned_bounds[2] or aligned_bounds[1] >= aligned_bounds[3]:
        return None, None, None

    window = from_bounds(*aligned_bounds, transform=transform)
    window_transform = rasterio.windows.transform(window, transform)
    polygon_raster = rasterize(
        [(polygon, 1)],
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    binary_mask = polygon_raster.astype(bool)
    dist_inside = distance_transform_edt(binary_mask)
    dist_outside = distance_transform_edt(~binary_mask)
    distance_raster = (dist_inside - dist_outside).astype(np.float32)
    normalized_dist = np.clip(distance_raster / max_distance_pixels, -1, 1)
    weights = 0.5 + 0.5 * np.tanh(normalized_dist * 2)
    weights[weights < 0.02] = 0.0
    weights[weights > 0.98] = 1.0
    weights_int = (weights * 1000).astype(np.int32)
    return weights_int, window, window_transform


def build_greco_weights(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Build ``greco_{id}_weights.tif`` for each GRECO under weights_dir."""
    shapefile_path = Path(cfg["greco_shapefile"]).expanduser()
    reference_raster_path = raster_path(cfg, "agb")
    out_dir = weights_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_distance_km = float(cfg.get("max_distance_km", 30))

    with rasterio.open(reference_raster_path) as ref_src:
        raster_crs = ref_src.crs
        height, width = ref_src.shape
        pixel_width = abs(ref_src.transform.a)
        max_distance_pixels = max(1, int(max_distance_km * 1000 / pixel_width))
        profile = ref_src.profile.copy()
        profile.update(
            dtype="int32",
            nodata=WEIGHT_NODATA,
            compress="deflate",
            predictor=2,
            count=1,
        )

        print(f"Reference raster: {width} x {height} pixels")
        print(f"Pixel resolution: {pixel_width:.2f} m")
        print(f"Max distance: {max_distance_km} km ({max_distance_pixels} pixels)")

        gdf = gpd.read_file(shapefile_path)
        if "GRECO" in gdf.columns:
            greco_col = "GRECO"
        elif "square_id" in gdf.columns:
            greco_col = "square_id"
        elif "ser_num" in gdf.columns:
            greco_col = "ser_num"
        else:
            raise ValueError("Could not find GRECO ID column in shapefile")

        gdf = gdf.to_crs(raster_crs)
        gdf_dissolved = gdf.dissolve(by=greco_col).reset_index()
        print(f"Found {len(gdf_dissolved)} unique GRECO regions")

        for _, row in tqdm(
            gdf_dissolved.iterrows(), total=len(gdf_dissolved), desc="GRECO weights"
        ):
            greco_id = row[greco_col]
            out_path = out_dir / weight_filename(str(greco_id))
            if out_path.exists() and not force:
                print(f"  Skip existing: {out_path.name}")
                continue

            weights, window, _ = create_distance_normalized_mask(
                row.geometry, ref_src, max_distance_pixels
            )
            if weights is None:
                print(f"  Skipping GRECO {greco_id}: invalid bounds")
                continue

            weights_with_nodata = weights.copy()
            weights_with_nodata[weights == 0] = WEIGHT_NODATA
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(weights_with_nodata, 1, window=window)
            print(f"  Saved: {out_path.name}")

    print(f"Weights directory: {out_dir}")
    return out_dir
