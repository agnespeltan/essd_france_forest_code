"""Polygon window helpers for GRECO tiles."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds


def create_polygon_raster(
    polygon,
    reference: str | rasterio.DatasetReader,
):
    """Create raster mask for polygon aligned to a reference raster path or open dataset."""
    own = False
    if isinstance(reference, (str, bytes)):
        ref_raster = rasterio.open(reference)
        own = True
    else:
        ref_raster = reference

    try:
        transform = ref_raster.transform
        raster_bounds = ref_raster.bounds
        polygon_bounds = polygon.bounds

        def snap_to_grid(value, resolution, offset, snap_func):
            return np.round(
                snap_func((value - offset) / resolution) * resolution + offset, 6
            )

        resolution = (transform.a, -transform.e)
        aligned_bounds = (
            max(
                raster_bounds.left,
                snap_to_grid(polygon_bounds[0], resolution[0], transform.c, np.floor),
            ),
            max(
                raster_bounds.bottom,
                snap_to_grid(polygon_bounds[1], resolution[1], transform.f, np.floor),
            ),
            min(
                raster_bounds.right,
                snap_to_grid(polygon_bounds[2], resolution[0], transform.c, np.ceil),
            ),
            min(
                raster_bounds.top,
                snap_to_grid(polygon_bounds[3], resolution[1], transform.f, np.ceil),
            ),
        )
        if aligned_bounds[0] >= aligned_bounds[2] or aligned_bounds[1] >= aligned_bounds[3]:
            return None, None

        bbox_window = from_bounds(*aligned_bounds, transform=transform)
        bbox_transform = rasterio.windows.transform(bbox_window, transform)
        polygon_raster = rasterize(
            [(polygon, 1)],
            out_shape=(int(bbox_window.height), int(bbox_window.width)),
            transform=bbox_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
    finally:
        if own:
            ref_raster.close()

    return polygon_raster, bbox_window


def load_shapefile(shapefile_path, raster_crs):
    """Load GRECO polygons; dissolve by square_id."""
    gdf = gpd.read_file(shapefile_path)
    if "ser_num" in gdf.columns:
        gdf = gdf.rename(columns={"ser_num": "square_id"})
    elif "GRECO" in gdf.columns:
        gdf = gdf.rename(columns={"GRECO": "square_id"})
    elif "square_id" not in gdf.columns:
        raise ValueError("Shapefile needs GRECO / ser_num / square_id column")
    return gdf.dissolve(by="square_id").reset_index().to_crs(raster_crs)
