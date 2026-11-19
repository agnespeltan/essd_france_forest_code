"""Output filename helpers (clear ESSD-style names)."""

from __future__ import annotations

from typing import Any


# Growth-curve quantile → age-map label. Inverting the 25th-percentile AGB curve
# gives OLDER ages (upper map); the 75th-percentile curve gives YOUNGER ages (lower).
AGE_LABEL = {50: "central", 25: "upper", 75: "lower"}


def age_label(quantile: int) -> str:
    """``50 → central``, ``25 → upper``, ``75 → lower``."""
    q = int(quantile)
    if q not in AGE_LABEL:
        raise ValueError(f"quantile must be one of {sorted(AGE_LABEL)}, got {q}")
    return AGE_LABEL[q]


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
    """Per-GRECO age tile, e.g. ``forest_age_greco_A_central.tif``."""
    lab = age_label(cfg.get("quantile", 50))
    return f"forest_age_greco_{greco_id}_{lab}{_opt_suffix(cfg)}.tif"


def weighted_age_filename(cfg: dict[str, Any]) -> str:
    """Blended national age (before nodata fix), e.g. ``forest_age_blend_central.tif``."""
    lab = age_label(cfg.get("quantile", 50))
    return f"forest_age_blend_{lab}{_opt_suffix(cfg)}.tif"


def fixed_age_filename(cfg: dict[str, Any]) -> str:
    """Final national age map: ``forest_age_central.tif`` (q50), ``_upper`` (q25), ``_lower`` (q75)."""
    lab = age_label(cfg.get("quantile", 50))
    return f"forest_age_{lab}{_opt_suffix(cfg)}.tif"


def weight_filename(greco_id: str) -> str:
    return f"greco_{greco_id}_weights.tif"
