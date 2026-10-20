"""Weighted national age map from per-GRECO tiles + distance weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from lib.config import age_tile_path, raster_path, weighted_age_path, weights_dir
from lib.constants import AGE_NODATA, WEIGHT_NODATA
from lib.invert.polygons import load_shapefile
from lib.naming import weight_filename


def create_weighted_age_map(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """``weighted_age = sum(age_i * w_i) / sum(w_i)`` tiled over France."""
    out_path = weighted_age_path(cfg)
    if out_path.exists() and not force:
        print(f"Skip existing: {out_path}")
        return out_path

    shapefile_path = Path(cfg["greco_shapefile"]).expanduser()
    age_map_folder = age_tile_path(cfg, "A").parent
    w_dir = weights_dir(cfg)
    tile_size = int(cfg.get("tile_size", 16384))

    with rasterio.open(raster_path(cfg, "agb")) as src:
        polygons = load_shapefile(shapefile_path, src.crs)
    greco_ids = [str(x) for x in sorted(polygons["square_id"].unique())]

    missing: list[str] = []
    for gid in greco_ids:
        ap = age_tile_path(cfg, gid)
        wp = w_dir / weight_filename(gid)
        if not ap.exists():
            missing.append(str(ap))
        if not wp.exists():
            missing.append(str(wp))
    if missing:
        raise FileNotFoundError("Missing input file(s):\n  " + "\n  ".join(missing))

    first = age_tile_path(cfg, greco_ids[0])
    with rasterio.open(first) as ref_src:
        profile = ref_src.profile.copy()
        profile.update(dtype="int32", nodata=AGE_NODATA, compress="deflate", predictor=2)
        height, width = ref_src.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating weighted age map from {len(greco_ids)} GRECO regions…")
    print(f"  tiles:   {age_map_folder}")
    print(f"  weights: {w_dir}")
    print(f"  output:  {out_path}")

    total_valid = 0
    num_row = (height + tile_size - 1) // tile_size
    num_col = (width + tile_size - 1) // tile_size

    age_srcs: list[rasterio.DatasetReader] = []
    weight_srcs: list[rasterio.DatasetReader] = []
    try:
        for gid in greco_ids:
            age_srcs.append(rasterio.open(age_tile_path(cfg, gid)))
            weight_srcs.append(rasterio.open(w_dir / weight_filename(gid)))

        with rasterio.open(out_path, "w", **profile) as dst:
            pbar = tqdm(total=num_row * num_col, desc="Blend tiles", unit="tile")
            for row_start in range(0, height, tile_size):
                for col_start in range(0, width, tile_size):
                    row_end = min(row_start + tile_size, height)
                    col_end = min(col_start + tile_size, width)
                    th, tw = row_end - row_start, col_end - col_start
                    window = Window(col_start, row_start, tw, th)

                    weighted_sum = np.zeros((th, tw), dtype=np.float32)
                    total_weights = np.zeros((th, tw), dtype=np.float32)

                    for age_src, weight_src in zip(age_srcs, weight_srcs):
                        try:
                            age_data = age_src.read(1, window=window)
                            weight_data = weight_src.read(1, window=window)
                            if age_data.shape != (th, tw):
                                age_data = age_src.read(
                                    1, window=window, out_shape=(th, tw)
                                )
                            if weight_data.shape != (th, tw):
                                weight_data = weight_src.read(
                                    1, window=window, out_shape=(th, tw)
                                )

                            # Skip empty weight windows early
                            if weight_src.nodata is not None:
                                w_probe = weight_data != weight_src.nodata
                            else:
                                w_probe = weight_data != WEIGHT_NODATA
                            w_probe &= weight_data > 0
                            if not np.any(w_probe):
                                continue

                            if age_src.nodata is not None and age_src.nodata != AGE_NODATA:
                                age_data[age_data == age_src.nodata] = AGE_NODATA
                            age_data = np.nan_to_num(
                                age_data,
                                nan=AGE_NODATA,
                                posinf=AGE_NODATA,
                                neginf=AGE_NODATA,
                            )
                            age_data[age_data < 0] = AGE_NODATA

                            weight_data = weight_data.astype(np.float32, copy=False)
                            if weight_src.nodata is not None:
                                weight_data[weight_data == weight_src.nodata] = 0
                            weight_data[weight_data == WEIGHT_NODATA] = 0
                            weight_data = np.nan_to_num(
                                weight_data, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            weight_data[weight_data < 0] = 0

                            valid = (age_data >= 0) & (weight_data > 0)
                            weighted_sum[valid] += age_data[valid] * weight_data[valid]
                            total_weights[valid] += weight_data[valid]
                        except Exception as exc:  # noqa: BLE001
                            print(f"    Error reading window: {exc}")

                    out_tile = np.full((th, tw), AGE_NODATA, dtype=np.int32)
                    valid = total_weights > 0
                    if np.any(valid):
                        with np.errstate(invalid="ignore", divide="ignore"):
                            wa = weighted_sum[valid] / total_weights[valid]
                            wa = np.nan_to_num(
                                wa, nan=AGE_NODATA, posinf=AGE_NODATA, neginf=AGE_NODATA
                            )
                            wa = np.clip(wa, AGE_NODATA, np.iinfo(np.int32).max)
                            out_tile[valid] = np.round(wa).astype(np.int32)

                    dst.write(out_tile, 1, window=window)
                    total_valid += int(np.sum(out_tile >= 0))
                    pbar.update(1)
                    pbar.set_postfix(valid=f"{total_valid:,}")
            pbar.close()
    finally:
        for src in age_srcs + weight_srcs:
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass

    cov = 100.0 * total_valid / (height * width)
    print(f"Weighted age map saved: {out_path} ({total_valid:,} valid, {cov:.1f}%)")
    return out_path
