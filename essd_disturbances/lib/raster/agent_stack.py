"""Agent multilayer → single-band latest year + latest agent (DEFLATE caches).

Same approach as essd_growth_curves: one multi-band read per tile, fill
newest→oldest. Year and agent maps are written in the same pass so they stay
pixel-aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# Nodata for the derived single-band year GeoTIFF
YEAR_NODATA = 0
# Nodata for the derived single-band agent GeoTIFF (matches agent stack)
AGENT_NODATA = 255


def agent_stack_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read agent-stack knobs from config (with defaults)."""
    return {
        "first_year": int(cfg.get("agent_first_year", 1985)),
        # Inclusive last calendar year kept in the year / agent maps
        "max_year": int(cfg.get("agent_last_year", cfg.get("year_end", 2024))),
        "nodata": int(cfg.get("agent_nodata", 255)),
        "min_valid_value": int(cfg.get("agent_min_valid_value", 1)),
    }


def default_latest_year_path(cfg: dict[str, Any], max_year: int | None = None) -> Path:
    """Path for the derived latest-year GeoTIFF."""
    if max_year is None:
        max_year = int(agent_stack_options(cfg)["max_year"])
    rasters = cfg.get("rasters") or {}
    if rasters.get("disturbance_year"):
        return Path(rasters["disturbance_year"]).expanduser()
    name = cfg.get(
        "latest_year_raster",
        f"latest_disturbance_year_le{int(max_year)}_30m.tif",
    )
    return Path(cfg["output_dir"]).expanduser() / name


def default_latest_agent_path(cfg: dict[str, Any], max_year: int | None = None) -> Path:
    """Path for the derived latest-agent GeoTIFF (agent at the latest year)."""
    if max_year is None:
        max_year = int(agent_stack_options(cfg)["max_year"])
    rasters = cfg.get("rasters") or {}
    if rasters.get("disturbance_agent"):
        return Path(rasters["disturbance_agent"]).expanduser()
    name = cfg.get(
        "latest_agent_raster",
        f"latest_disturbance_agent_le{int(max_year)}_30m.tif",
    )
    return Path(cfg["output_dir"]).expanduser() / name


def resolve_latest_year_path(cfg: dict[str, Any]) -> Path:
    """Return existing year GeoTIFF, or raise (run ``00`` first)."""
    path = default_latest_year_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Latest-year raster not found: {path}\n"
            "Run: python3 00_materialize_latest_year.py --config config.yaml"
        )
    return path


def resolve_latest_agent_path(cfg: dict[str, Any]) -> Path:
    """Return existing agent GeoTIFF, or raise (run ``00`` first)."""
    path = default_latest_agent_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Latest-agent raster not found: {path}\n"
            "Run: python3 00_materialize_latest_year.py --config config.yaml"
        )
    return path


def _block_size_for(src: rasterio.DatasetReader, block_size: int) -> int:
    bs = min(int(block_size), src.width, src.height)
    for cand in (1024, 512, 256):
        if cand <= bs:
            return cand
    return bs


def materialize_latest_maps(
    stack_path: str | Path,
    year_path: str | Path,
    agent_path: str | Path,
    *,
    first_year: int = 1985,
    max_year: int = 2024,
    nodata: int = 255,
    min_valid_value: int = 1,
    block_size: int = 1024,
) -> tuple[Path, Path]:
    """
    Block-wise reduce agent multilayer → year + agent DEFLATE GeoTIFFs.

    Keeps the latest valid agent year with ``first_year <= year <= max_year``.
    Per tile: one multi-band GDAL read, then fill newest→oldest in NumPy.
    Year and agent stay pixel-aligned (same fill mask).
    """
    stack_path = Path(stack_path)
    year_path = Path(year_path)
    agent_path = Path(agent_path)
    year_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.parent.mkdir(parents=True, exist_ok=True)

    year_tmp = year_path.with_suffix(year_path.suffix + ".tmp")
    agent_tmp = agent_path.with_suffix(agent_path.suffix + ".tmp")
    for tmp in (year_tmp, agent_tmp):
        if tmp.exists():
            tmp.unlink()

    with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(stack_path) as src:
            bs = _block_size_for(src, block_size)
            year_profile = {
                "driver": "GTiff",
                "height": src.height,
                "width": src.width,
                "count": 1,
                "dtype": "uint16",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": YEAR_NODATA,
                "compress": "deflate",
                "zlevel": 2,
                "predictor": 2,
                "tiled": True,
                "blockxsize": bs,
                "blockysize": bs,
                "BIGTIFF": "IF_SAFER",
                "NUM_THREADS": "ALL_CPUS",
            }
            agent_profile = {
                **year_profile,
                "dtype": "uint8",
                "nodata": AGENT_NODATA,
                "predictor": 1,
            }
            # Newest first; inclusive max_year
            band_years = [
                (b, first_year + b - 1)
                for b in range(1, src.count + 1)
                if first_year <= first_year + b - 1 <= max_year
            ]
            band_years.reverse()
            if not band_years:
                raise ValueError(
                    f"No agent bands with {first_year} <= year <= {max_year} "
                    f"(count={src.count})"
                )
            indexes = [b for b, _ in band_years]
            years = np.asarray([y for _, y in band_years], dtype=np.uint16)

            print(
                f"Materializing latest year+agent <= {max_year} "
                f"({len(indexes)} bands / 1 multi-read per tile, "
                f"tiles {bs}×{bs}, DEFLATE z=2)\n"
                f"  {stack_path}\n"
                f"  → {year_path}\n"
                f"  → {agent_path}",
                flush=True,
            )
            n_blocks = 0
            with (
                rasterio.open(year_tmp, "w", **year_profile) as year_dst,
                rasterio.open(agent_tmp, "w", **agent_profile) as agent_dst,
            ):
                for _, window in year_dst.block_windows(1):
                    stack = src.read(indexes, window=window)
                    n_b, height, width = stack.shape
                    latest_year = np.zeros((height, width), dtype=np.uint16)
                    latest_agent = np.full(
                        (height, width), AGENT_NODATA, dtype=np.uint8
                    )
                    empty = np.ones((height, width), dtype=bool)

                    for i in range(n_b):
                        if not empty.any():
                            break
                        band = stack[i]
                        valid = empty & (band != nodata) & (band >= min_valid_value)
                        latest_year[valid] = years[i]
                        latest_agent[valid] = band[valid].astype(np.uint8, copy=False)
                        empty[valid] = False

                    latest_year[empty] = YEAR_NODATA
                    year_dst.write(latest_year, 1, window=window)
                    agent_dst.write(latest_agent, 1, window=window)
                    del stack, latest_year, latest_agent, empty
                    n_blocks += 1
                    if n_blocks % 200 == 0:
                        print(f"  … {n_blocks} tiles", flush=True)

    year_tmp.replace(year_path)
    agent_tmp.replace(agent_path)
    print(
        f"Wrote {year_path} ({year_path.stat().st_size / 1e6:.1f} MB)\n"
        f"Wrote {agent_path} ({agent_path.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    return year_path, agent_path


def materialize_latest_year_raster(
    stack_path: str | Path,
    out_path: str | Path,
    *,
    first_year: int = 1985,
    max_year: int = 2024,
    nodata: int = 255,
    min_valid_value: int = 1,
    block_size: int = 1024,
    agent_out_path: str | Path | None = None,
) -> Path:
    """Back-compat wrapper: materialize year (+ agent next to it if unset)."""
    if agent_out_path is None:
        # sibling: latest_disturbance_year_… → latest_disturbance_agent_…
        p = Path(out_path)
        agent_out_path = p.with_name(
            p.name.replace("latest_disturbance_year_", "latest_disturbance_agent_")
        )
        if agent_out_path == p:
            agent_out_path = p.with_name(f"latest_disturbance_agent_le{max_year}_30m.tif")
    year_path, _ = materialize_latest_maps(
        stack_path,
        out_path,
        agent_out_path,
        first_year=first_year,
        max_year=max_year,
        nodata=nodata,
        min_valid_value=min_valid_value,
        block_size=block_size,
    )
    return year_path


def materialize_from_config(cfg: dict[str, Any], *, force: bool = False) -> tuple[Path, Path]:
    """Build or reuse the latest-year and latest-agent GeoTIFFs from config."""
    opts = agent_stack_options(cfg)
    max_year = int(opts["max_year"])
    year_out = default_latest_year_path(cfg, max_year)
    agent_out = default_latest_agent_path(cfg, max_year)
    stack = Path(cfg["rasters"]["agent_multilayer"]).expanduser()
    if not stack.is_file():
        raise FileNotFoundError(f"Agent multilayer not found: {stack}")
    force = force or bool(cfg.get("force_rebuild_latest_year", False))
    stack_mtime = stack.stat().st_mtime
    year_ok = (
        year_out.is_file()
        and not force
        and year_out.stat().st_mtime >= stack_mtime
    )
    agent_ok = (
        agent_out.is_file()
        and not force
        and agent_out.stat().st_mtime >= stack_mtime
    )
    if year_ok and agent_ok:
        print(f"Using existing:\n  {year_out}\n  {agent_out}")
        return year_out, agent_out
    return materialize_latest_maps(
        stack,
        year_out,
        agent_out,
        first_year=int(opts["first_year"]),
        max_year=max_year,
        nodata=int(opts["nodata"]),
        min_valid_value=int(opts["min_valid_value"]),
        block_size=int(cfg.get("latest_year_block_size", 1024)),
    )
