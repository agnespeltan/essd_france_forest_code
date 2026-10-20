"""Output filename helpers (clear ESSD-style names)."""

from __future__ import annotations

from typing import Any


def _opt_suffix(cfg: dict[str, Any]) -> str:
    """Only non-default product flags appear in the filename."""
    parts: list[str] = []
    if cfg.get("use_thinning"):
        parts.append("thinning")
    # Default product is not_corrected; only tag the alternate
    if cfg.get("not_corrected") is False:
        parts.append("corrected")
    if cfg.get("use_forest_aggregates"):
        parts.append("forest")
    return ("_" + "_".join(parts)) if parts else ""


def age_tile_filename(cfg: dict[str, Any], greco_id: str) -> str:
    """Per-GRECO age tile, e.g. ``forest_age_greco_A_q50.tif``."""
    q = int(cfg.get("quantile", 50))
    return f"forest_age_greco_{greco_id}_q{q}{_opt_suffix(cfg)}.tif"


def weighted_age_filename(cfg: dict[str, Any]) -> str:
    """Blended national age (before nodata fix), e.g. ``forest_age_blend_q50.tif``."""
    q = int(cfg.get("quantile", 50))
    return f"forest_age_blend_q{q}{_opt_suffix(cfg)}.tif"


def fixed_age_filename(cfg: dict[str, Any]) -> str:
    """Final national age map, e.g. ``forest_age_q50.tif``."""
    q = int(cfg.get("quantile", 50))
    return f"forest_age_q{q}{_opt_suffix(cfg)}.tif"


def weight_filename(greco_id: str) -> str:
    return f"greco_{greco_id}_weights.tif"
