"""Latest disturbance-year map capped to the age-map keep window (2019–2021).

Builds ``latest_disturbance_year_le{max_year}_30m.tif`` from the agent multilayer
with ``year <= max(disturbance_keep_years)`` (default 2021). The nodata-fix step
then keeps age==0 only when that latest year is in {2019, 2020, 2021}.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from lib.config import features_root, output_dir, resolve_under
from lib.constants import EXCLUDED_DIST_YEARS


def keep_years(cfg: dict[str, Any]) -> frozenset[int]:
    """Years for which age==0 & AGB==0 is retained (recent clearcuts)."""
    raw = cfg.get("disturbance_keep_years")
    if raw is None:
        return frozenset(EXCLUDED_DIST_YEARS)
    return frozenset(int(y) for y in raw)


def keep_max_year(cfg: dict[str, Any]) -> int:
    return max(keep_years(cfg))


def default_year_filename(cfg: dict[str, Any]) -> str:
    return f"latest_disturbance_year_le{keep_max_year(cfg)}_30m.tif"


def default_agent_filename(cfg: dict[str, Any]) -> str:
    return f"latest_disturbance_agent_le{keep_max_year(cfg)}_30m.tif"


def year_path(cfg: dict[str, Any]) -> Path:
    rasters = cfg.get("rasters") or {}
    raw = rasters.get("disturbance_year") or default_year_filename(cfg)
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return resolve_under(output_dir(cfg), p)


def agent_stack_path(cfg: dict[str, Any]) -> Path:
    rasters = cfg.get("rasters") or {}
    if not rasters.get("agent_multilayer"):
        raise KeyError(
            "Missing rasters.agent_multilayer in config "
            "(needed to materialize disturbance_year for 2019–2021 keep window)"
        )
    raw = rasters["agent_multilayer"]
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return resolve_under(features_root(cfg), p)


def _load_disturbances_materialize():
    """Load ``materialize_latest_maps`` from sibling ``essd_disturbances`` (no lib clash)."""
    dist_stack = (
        Path(__file__).resolve().parents[2].parent
        / "essd_disturbances"
        / "lib"
        / "raster"
        / "agent_stack.py"
    )
    if not dist_stack.is_file():
        raise FileNotFoundError(
            f"Cannot find essd_disturbances agent_stack at {dist_stack}"
        )
    spec = importlib.util.spec_from_file_location("essd_dist_agent_stack", dist_stack)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {dist_stack}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.materialize_latest_maps


def ensure_disturbance_year(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Build or reuse the le{max_keep} latest-year GeoTIFF used by ``03_finalize_age_map``."""
    out = year_path(cfg)
    max_year = keep_max_year(cfg)
    stack = agent_stack_path(cfg)
    if not stack.is_file():
        raise FileNotFoundError(f"Agent multilayer not found: {stack}")

    if out.is_file() and not force and out.stat().st_mtime >= stack.stat().st_mtime:
        print(f"Using existing disturbance year (<= {max_year}): {out}")
        return out

    agent_out = output_dir(cfg) / default_agent_filename(cfg)
    materialize_latest_maps = _load_disturbances_materialize()
    year_path_out, _ = materialize_latest_maps(
        stack,
        out,
        agent_out,
        first_year=int(cfg.get("agent_first_year", 1985)),
        max_year=max_year,
        nodata=int(cfg.get("agent_nodata", 255)),
        min_valid_value=int(cfg.get("agent_min_valid_value", 1)),
        block_size=int(cfg.get("latest_year_block_size", 1024)),
    )
    # Point config at the product we just wrote (relative name under output_dir).
    cfg.setdefault("rasters", {})["disturbance_year"] = year_path_out.name
    return year_path_out
