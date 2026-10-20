"""Resolve absolute paths from the YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_under(root: Path, maybe_relative: str | Path) -> Path:
    """If path is absolute, return it; otherwise join under root."""
    path = Path(maybe_relative).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def features_root(cfg: dict[str, Any]) -> Path:
    return _as_path(cfg["features_root"])


def output_dir(cfg: dict[str, Any]) -> Path:
    return _as_path(cfg["output_dir"])


def greco_shapefile(cfg: dict[str, Any]) -> Path:
    return _as_path(cfg["greco_shapefile"])


def raster_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve one raster entry from cfg['rasters'][key]."""
    rasters = cfg.get("rasters") or {}
    if key not in rasters:
        raise KeyError(f"Missing rasters.{key} in config")
    return resolve_under(features_root(cfg), rasters[key])


def stats_path(cfg: dict[str, Any], which: str) -> Path:
    """Resolve a stats parquet path (efda | gami)."""
    name = (cfg.get("stats") or {})[which]
    return resolve_under(output_dir(cfg), name)


def curves_path(cfg: dict[str, Any], target: str) -> Path:
    """Resolve a fitted-curves CSV path for a fit target."""
    name = (cfg.get("curves") or {})[target]
    return resolve_under(output_dir(cfg), name)


def raster_dict(cfg: dict[str, Any]) -> dict[str, Path]:
    """All configured rasters as absolute paths."""
    return {key: raster_path(cfg, key) for key in (cfg.get("rasters") or {})}
