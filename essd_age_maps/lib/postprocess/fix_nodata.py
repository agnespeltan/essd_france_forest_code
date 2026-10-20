"""Force age==0 → nodata where AGB==0 and disturbance year ∉ keep window.

Default keep window is {2019, 2020, 2021} (``disturbance_keep_years`` / 
``EXCLUDED_DIST_YEARS``). Requires a latest-year map with years through the
window max (``latest_disturbance_year_le2021_30m.tif``).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from lib.config import fixed_age_path, raster_path, weighted_age_path
from lib.constants import AGE_NODATA
from lib.raster.disturbance_year import ensure_disturbance_year, keep_years

N_ROW_BLOCKS = 10


def _print_progress(current: int, total: int, width: int = 30) -> None:
    ratio = 0.0 if total == 0 else current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def build_masks(age, dist, agb, age_nodata, dist_nodata, agb_nodata, keep: frozenset[int]):
    valid_age = np.isfinite(age)
    if age_nodata is not None and np.isfinite(age_nodata):
        valid_age &= age != age_nodata
    valid_age &= age != AGE_NODATA

    valid_dist = np.isfinite(dist)
    if dist_nodata is not None and np.isfinite(dist_nodata):
        valid_dist &= dist != dist_nodata

    valid_agb = np.isfinite(agb)
    if agb_nodata is not None and np.isfinite(agb_nodata):
        valid_agb &= agb != agb_nodata

    # Keep age==0 when latest disturbance year is in the keep window.
    in_keep = np.isin(dist, list(keep))
    dist_allows_change = (~valid_dist) | (~in_keep)
    return valid_age & (age == 0) & dist_allows_change & valid_agb & (agb == 0)


def fix_nodata(cfg: dict[str, Any], *, force: bool = False) -> Any:
    """Write fixed age map under output_dir."""
    keep = keep_years(cfg)
    # Ensure latest-year map includes the keep window (e.g. le2021 for 2019–2021).
    ensure_disturbance_year(cfg, force=False)

    age_path = weighted_age_path(cfg)
    # Allow override: pin an existing weighted map
    if cfg.get("rasters", {}).get("age_weighted"):
        from pathlib import Path

        age_path = Path(cfg["rasters"]["age_weighted"]).expanduser()
    dist_path = raster_path(cfg, "disturbance_year")
    agb_path = raster_path(cfg, "agb")
    out_path = fixed_age_path(cfg)

    if out_path.exists() and not force:
        print(f"Skip existing: {out_path}")
        return out_path

    with rasterio.open(age_path) as age_src, rasterio.open(dist_path) as dist_src, rasterio.open(
        agb_path
    ) as agb_src:
        if (
            age_src.width != dist_src.width
            or age_src.height != dist_src.height
            or age_src.transform != dist_src.transform
            or age_src.crs != dist_src.crs
        ):
            raise ValueError("Age and disturbance rasters are not aligned.")
        if (
            age_src.width != agb_src.width
            or age_src.height != agb_src.height
            or age_src.transform != agb_src.transform
            or age_src.crs != agb_src.crs
        ):
            raise ValueError("Age and AGB rasters are not aligned.")

        profile = age_src.profile.copy()
        age_nodata = age_src.nodata
        dist_nodata = dist_src.nodata
        agb_nodata = agb_src.nodata
        width, height = age_src.width, age_src.height
        output_dtype = profile["dtype"]
        profile["count"] = 1
        output_nodata = profile.get("nodata", age_nodata)
        if output_nodata is None:
            output_nodata = AGE_NODATA
            profile["nodata"] = output_nodata

        out_path.parent.mkdir(parents=True, exist_ok=True)
        block_h = max(1, height // N_ROW_BLOCKS)
        total_forced = 0

        with rasterio.open(out_path, "w", **profile) as out_dst:
            for i in range(N_ROW_BLOCKS):
                row0 = i * block_h
                row1 = (i + 1) * block_h if i < N_ROW_BLOCKS - 1 else height
                if row1 <= row0:
                    continue
                read_win = Window(0, row0, width, row1 - row0)
                age_block = age_src.read(1, window=read_win).astype(np.float32)
                dist_block = dist_src.read(1, window=read_win)
                agb_block = agb_src.read(1, window=read_win).astype(np.float32)
                force_mask = build_masks(
                    age_block,
                    dist_block,
                    agb_block,
                    age_nodata,
                    dist_nodata,
                    agb_nodata,
                    keep,
                )
                total_forced += int(force_mask.sum())
                age_block[force_mask] = output_nodata
                to_write = age_block
                if np.issubdtype(np.dtype(output_dtype), np.integer):
                    to_write = np.rint(to_write)
                out_dst.write(to_write.astype(output_dtype), 1, window=read_win)
                _print_progress(i + 1, N_ROW_BLOCKS)

    years = ", ".join(str(y) for y in sorted(keep))
    print(f"Keep age==0 when disturbance year in {{{years}}}")
    print(f"Disturbance year map: {dist_path}")
    print(f"Forced nodata pixels (age==0 & AGB==0): {total_forced}")
    print(f"Wrote: {out_path}")
    return out_path
