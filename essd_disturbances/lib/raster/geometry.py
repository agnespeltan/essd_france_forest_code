"""Polygon window helpers for SER-level raster stats."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.windows import from_bounds
import rasterio


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


def codeser_to_ser_num(codeser) -> int:
    """Map IGN ``codeser`` (e.g. ``A11``) to numeric SER id (``111``)."""
    s = str(codeser).strip()
    if not s:
        raise ValueError(f"Empty codeser: {codeser!r}")
    letter = s[0].upper()
    if not ("A" <= letter <= "Z"):
        raise ValueError(f"Unexpected codeser letter in {codeser!r}")
    greco = ord(letter) - 64  # A→1
    return greco * 100 + int(s[1:])


def read_ser_polygons(shapefile_path: str | Path, raster_crs) -> gpd.GeoDataFrame:
    """Load SER polygons from shapefile; dissolve by numeric ``square_id``.

    Accepts ``ser_num``, ``square_id``, or IGN ``codeser`` (``A11`` → ``111``).
    """
    path = Path(shapefile_path)
    if not path.is_file():
        raise FileNotFoundError(f"SER shapefile not found: {path}")

    gdf = gpd.read_file(path)
    if "ser_num" in gdf.columns:
        gdf = gdf.rename(columns={"ser_num": "square_id"})
    elif "square_id" in gdf.columns:
        pass
    elif "codeser" in gdf.columns:
        gdf = gdf.copy()
        gdf["square_id"] = gdf["codeser"].map(codeser_to_ser_num)
    else:
        raise ValueError(
            f"Shapefile must contain ser_num, square_id, or codeser: {path}"
        )

    gdf = gdf.dissolve(by="square_id").reset_index()
    return gdf.to_crs(raster_crs)


def parse_inclusive_year_range(label: str) -> tuple[int, int, list[int]]:
    """Parse ``YYYY-YYYY`` as inclusive bounds → (start, end, list of years)."""
    parts = str(label).split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-YYYY year range, got {label!r}")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError(f"Invalid year range {label!r}")
    return start, end, list(range(start, end + 1))


def agent_band_index(year: int, agent_first_year: int) -> int:
    """1-based GeoTIFF band for calendar year."""
    return int(year - agent_first_year + 1)
