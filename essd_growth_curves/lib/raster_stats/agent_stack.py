"""Agent multilayer → single-band latest disturbance year (DEFLATE cache)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
import rasterio
from rasterio.windows import Window

from lib.paths import features_root, output_dir, raster_path, resolve_under

# Nodata for the derived single-band year GeoTIFF (calendar years are 1985–2019).
YEAR_NODATA = 0


def agent_stack_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read agent-stack knobs from config (with defaults)."""
    return {
        "first_year": int(cfg.get("agent_first_year", 1985)),
        "max_year": int(cfg.get("agent_max_year", cfg.get("reference_year", 2020))),
        "nodata": int(cfg.get("agent_nodata", 255)),
        "min_valid_value": int(cfg.get("agent_min_valid_value", 1)),
    }


def default_latest_year_path(cfg: dict[str, Any], max_year: int | None = None) -> Path:
    """Path for the derived latest-year GeoTIFF (single output)."""
    if max_year is None:
        max_year = int(agent_stack_options(cfg)["max_year"])
    rasters = cfg.get("rasters") or {}
    if "disturbance_year" in rasters:
        return resolve_under(features_root(cfg), rasters["disturbance_year"])
    name = cfg.get(
        "latest_year_raster",
        f"latest_disturbance_year_lt{int(max_year)}_30m.tif",
    )
    return resolve_under(output_dir(cfg), name)


def resolve_latest_year_path(cfg: dict[str, Any]) -> Path:
    """Return existing year GeoTIFF, or raise (run ``00_materialize_year.py`` first)."""
    path = default_latest_year_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Latest-year raster not found: {path}\n"
            "Run: python3 00_materialize_year.py --config config.yaml"
        )
    return path


def materialize_latest_year_raster(
    stack_path: str | Path,
    out_path: str | Path,
    *,
    first_year: int = 1985,
    max_year: int = 2020,
    nodata: int = 255,
    min_valid_value: int = 1,
    block_size: int = 1024,
) -> Path:
    """
    Block-wise reduce agent multilayer → one DEFLATE single-band year GeoTIFF.

    Per tile: **one** multi-band GDAL read (all years < max_year), then fill
    newest→oldest in NumPy (early exit when the tile is full). ~35× fewer
    reads than band-at-a-time. Peak RAM ≈ block² × n_bands bytes (~35 MB at 1024).
    """
    stack_path = Path(stack_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(stack_path) as src:
            bs = min(int(block_size), src.width, src.height)
            # Prefer power-of-two tile size for TIFF tiling.
            for cand in (1024, 512, 256):
                if cand <= bs:
                    bs = cand
                    break
            profile = {
                "driver": "GTiff",
                "height": src.height,
                "width": src.width,
                "count": 1,
                "dtype": "uint16",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": YEAR_NODATA,
                "compress": "deflate",
                "zlevel": 2,  # faster write; still DEFLATE
                "predictor": 2,
                "tiled": True,
                "blockxsize": bs,
                "blockysize": bs,
                "BIGTIFF": "IF_SAFER",
                "NUM_THREADS": "ALL_CPUS",
            }
            # Newest first: indexes order matches stack axis 0 after read.
            band_years = [
                (b, first_year + b - 1)
                for b in range(1, src.count + 1)
                if first_year + b - 1 < max_year
            ]
            band_years.reverse()
            if not band_years:
                raise ValueError(
                    f"No agent bands with year < {max_year} "
                    f"(first_year={first_year}, count={src.count})"
                )
            indexes = [b for b, _ in band_years]
            years = np.asarray([y for _, y in band_years], dtype=np.uint16)

            print(
                f"Materializing latest year < {max_year} "
                f"({len(indexes)} bands / 1 multi-read per tile, "
                f"tiles {bs}×{bs}, DEFLATE z=2)\n"
                f"  {stack_path}\n  → {out_path}",
                flush=True,
            )
            n_blocks = 0
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    # (n_bands, h, w) — single GDAL call
                    stack = src.read(indexes, window=window)
                    n_b, height, width = stack.shape
                    latest = np.zeros((height, width), dtype=np.uint16)
                    empty = np.ones((height, width), dtype=bool)

                    for i in range(n_b):
                        if not empty.any():
                            break
                        band = stack[i]
                        valid = empty & (band != nodata) & (band >= min_valid_value)
                        latest[valid] = years[i]
                        empty[valid] = False

                    latest[empty] = YEAR_NODATA
                    dst.write(latest, 1, window=window)
                    del stack, latest, empty
                    n_blocks += 1
                    if n_blocks % 200 == 0:
                        print(f"  … {n_blocks} tiles", flush=True)

    tmp_path.replace(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out_path


def materialize_from_config(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Build or reuse the latest-year GeoTIFF from config."""
    opts = agent_stack_options(cfg)
    out = default_latest_year_path(cfg, int(opts["max_year"]))
    stack = raster_path(cfg, "agent_multilayer")
    force = force or bool(cfg.get("force_rebuild_latest_year", False))
    if (
        out.is_file()
        and not force
        and out.stat().st_mtime >= stack.stat().st_mtime
    ):
        print(f"Using existing: {out}")
        return out
    return materialize_latest_year_raster(
        stack,
        out,
        first_year=int(opts["first_year"]),
        max_year=int(opts["max_year"]),
        nodata=int(opts["nodata"]),
        min_valid_value=int(opts["min_valid_value"]),
        block_size=int(cfg.get("latest_year_block_size", 1024)),
    )


def read_latest_year_window(
    path: str | Path,
    window: Window,
    *,
    nodata: int = YEAR_NODATA,
) -> ma.MaskedArray:
    """Read a window from the year GeoTIFF (uint16, nodata masked)."""
    with rasterio.open(path) as src:
        data = src.read(1, window=window)
        nd = src.nodata if src.nodata is not None else nodata
    return ma.masked_array(data, mask=(data == nd))


def efda_free_mask_from_year(
    path: str | Path,
    window: Window,
    *,
    nodata: int = YEAR_NODATA,
) -> np.ndarray:
    """True where year raster is nodata (= no disturbance before reference_year)."""
    with rasterio.open(path) as src:
        data = src.read(1, window=window)
        nd = src.nodata if src.nodata is not None else nodata
    return data == nd
