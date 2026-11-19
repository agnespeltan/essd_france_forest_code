"""Census forest TFV×SER area from the TFV mask × SER polygons (not disturbance-conditioned)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from lib.config import excluded_tfv_set, output_path
from lib.constants import GROUPS_BG, GROUPS_FT, PIXEL_AREA_MHA
from lib.raster.geometry import polygon_to_raster_with_bbox, read_ser_polygons


def _greco_letter(ser: pd.Series) -> pd.Series:
    g = (ser.astype(int) // 100).clip(lower=0)
    return g.map(lambda x: chr(64 + x) if 1 <= x <= 26 else "Z")


def census_forest_tfv_ser(
    tfv_path: str | Path,
    ser_shapefile: str | Path,
    excluded: frozenset[str],
) -> pd.DataFrame:
    """Return one row per SER×TFV with mask pixels > 0 (after exclusions).

    Columns: ``ser``, ``TFV`` (str), ``n_pixels``, ``total_area`` (Mha),
    ``FT``, ``GRECO``, ``BG``.
    """
    tfv_path = Path(tfv_path).expanduser()
    ser_shp = Path(ser_shapefile).expanduser()
    if not tfv_path.is_file():
        raise FileNotFoundError(f"TFV raster not found: {tfv_path}")
    if not ser_shp.is_file():
        raise FileNotFoundError(f"SER shapefile not found: {ser_shp}")

    with rasterio.open(tfv_path) as src:
        gdf = read_ser_polygons(ser_shp, src.crs)
        nodata = src.nodata

    rows: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        ser = int(row["square_id"])
        try:
            poly_raster, _, bbox_window = polygon_to_raster_with_bbox(
                row.geometry, tfv_path
            )
        except ValueError:
            continue
        with rasterio.open(tfv_path) as src:
            tfv = src.read(1, window=bbox_window)
        if tfv.shape != poly_raster.shape:
            # Defensive crop to overlapping shape
            h = min(tfv.shape[0], poly_raster.shape[0])
            w = min(tfv.shape[1], poly_raster.shape[1])
            tfv = tfv[:h, :w]
            poly_raster = poly_raster[:h, :w]
        inside = tfv[poly_raster == 1]
        if nodata is not None:
            inside = inside[inside != nodata]
        if inside.size == 0:
            continue
        u, c = np.unique(inside, return_counts=True)
        for tv, n in zip(u.tolist(), c.tolist()):
            n = int(n)
            if n <= 0:
                continue
            tfv_s = str(int(tv))
            if tfv_s.lower() in excluded:
                continue
            rows.append(
                {
                    "ser": ser,
                    "TFV": tfv_s,
                    "n_pixels": n,
                    "total_area": float(n) * PIXEL_AREA_MHA,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["GRECO"] = _greco_letter(out["ser"])
    out["FT"] = out["TFV"].map(GROUPS_FT)
    out["BG"] = out["GRECO"].map(GROUPS_BG)
    return out.sort_values(["ser", "TFV"], kind="mergesort").reset_index(drop=True)


def build_forest_area(cfg: dict[str, Any]) -> Path:
    """Census mask forest area and write ``forest_tfv_ser_area.parquet``."""
    excluded = excluded_tfv_set(cfg)
    rasters = cfg["rasters"]
    tfv_path = Path(rasters["tfv"]).expanduser()
    ser_shp = Path(cfg["ser_shapefile"]).expanduser()
    table = census_forest_tfv_ser(tfv_path, ser_shp, excluded)
    out = output_path(cfg, "forest_area")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    cfg["forest_area_parquet"] = str(out)
    return out


def load_or_build_forest_area(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load forest-area parquet, or build it if missing."""
    path_raw = cfg.get("forest_area_parquet")
    if not path_raw:
        try:
            path = output_path(cfg, "forest_area")
        except KeyError:
            path = Path(cfg["output_dir"]).expanduser() / "forest_tfv_ser_area.parquet"
    else:
        path = Path(path_raw).expanduser()

    if path.is_file():
        df = pd.read_parquet(path)
    else:
        build_forest_area(cfg)
        df = pd.read_parquet(Path(cfg["forest_area_parquet"]).expanduser())

    df = df.copy()
    df["TFV"] = df["TFV"].astype(str)
    df["ser"] = pd.to_numeric(df["ser"], errors="coerce").astype(int)
    if "total_area" not in df.columns and "n_pixels" in df.columns:
        df["total_area"] = df["n_pixels"].astype(float) * PIXEL_AREA_MHA
    if "GRECO" not in df.columns:
        df["GRECO"] = _greco_letter(df["ser"])
    if "FT" not in df.columns:
        df["FT"] = df["TFV"].map(GROUPS_FT)
    if "BG" not in df.columns:
        df["BG"] = df["GRECO"].map(GROUPS_BG)
    return df
