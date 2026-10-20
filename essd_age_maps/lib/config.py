"""Load and validate YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lib.naming import age_tile_filename, fixed_age_filename, weighted_age_filename


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_under(root: Path, maybe_relative: str | Path) -> Path:
    """If path is absolute, return it; otherwise join under root."""
    path = Path(maybe_relative).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and apply defaults."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    cfg.setdefault("quantile", 50)
    cfg.setdefault("use_thinning", False)
    cfg.setdefault("not_corrected", True)
    cfg.setdefault("include_bef", True)
    cfg.setdefault("use_forest_aggregates", False)
    cfg.setdefault("max_distance_km", 30)
    cfg.setdefault("tile_size", 16384)  # 02 blend spatial tiles
    cfg.setdefault("block_size", 2048)  # 01 weight-masked invert blocks
    # Years for which age==0 & AGB==0 is kept (recent clearcuts); also sets le{max} year map.
    cfg.setdefault("disturbance_keep_years", [2019, 2020, 2021])
    cfg.setdefault("agent_first_year", 1985)
    cfg.setdefault("workers", 1)
    cfg.setdefault("low_memory", True)
    cfg.setdefault("force", False)
    cfg.setdefault(
        "outputs",
        {
            # Under output_dir (Data_clean), same as other essd_* products
            "weights_dir": "GRECO",
            "tiles_dir": "Age_tiles",
            "weighted": None,
            "fixed": None,
        },
    )
    cfg.setdefault("growth_curves_csv", "growth_curves_agb_greco.csv")
    cfg["_config_path"] = str(config_path)
    return cfg


def data_root(cfg: dict[str, Any]) -> Path | None:
    if not cfg.get("data_root"):
        return None
    return _as_path(cfg["data_root"])


def features_root(cfg: dict[str, Any]) -> Path:
    if cfg.get("features_root"):
        return _as_path(cfg["features_root"])
    root = data_root(cfg)
    if root is not None:
        return root / "Features_30m"
    raise KeyError("Missing features_root (or data_root) in config")


def output_dir(cfg: dict[str, Any]) -> Path:
    return _as_path(cfg["output_dir"])


def figure_dir(cfg: dict[str, Any]) -> Path | None:
    if not cfg.get("figure_dir"):
        return None
    return _as_path(cfg["figure_dir"])


def weights_dir(cfg: dict[str, Any]) -> Path:
    """GRECO soft-weight GeoTIFFs (default: output_dir/GRECO)."""
    outs = cfg.get("outputs", {})
    name = outs.get("weights_dir", "GRECO")
    p = Path(name)
    return p.resolve() if p.is_absolute() else resolve_under(output_dir(cfg), p)


def tiles_dir(cfg: dict[str, Any]) -> Path:
    outs = cfg.get("outputs", {})
    name = outs.get("tiles_dir", "Age_tiles")
    p = Path(name)
    return p.resolve() if p.is_absolute() else resolve_under(output_dir(cfg), p)


def weighted_age_path(cfg: dict[str, Any]) -> Path:
    outs = cfg.get("outputs", {})
    name = outs.get("weighted") or weighted_age_filename(cfg)
    p = Path(name)
    return p.resolve() if p.is_absolute() else resolve_under(output_dir(cfg), p)


def fixed_age_path(cfg: dict[str, Any]) -> Path:
    outs = cfg.get("outputs", {})
    name = outs.get("fixed") or fixed_age_filename(cfg)
    p = Path(name)
    return p.resolve() if p.is_absolute() else resolve_under(output_dir(cfg), p)


def age_tile_path(cfg: dict[str, Any], greco_id: str) -> Path:
    return tiles_dir(cfg) / age_tile_filename(cfg, greco_id)


def growth_curves_path(cfg: dict[str, Any]) -> Path:
    """Fitted CR curves CSV (default: essd_growth_curves product under output_dir)."""
    name = cfg.get("growth_curves_csv", "growth_curves_agb_greco.csv")
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    # Prefer Data_clean, then data_root/Data
    cand = resolve_under(output_dir(cfg), p)
    if cand.is_file():
        return cand
    root = data_root(cfg)
    if root is not None:
        alt = resolve_under(root / "Data", p)
        if alt.is_file():
            return alt
    return cand


def raster_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``rasters.<key>`` under features_root (disturbance_year → output_dir)."""
    rasters = cfg.get("rasters") or {}
    if key not in rasters or not rasters[key]:
        raise KeyError(f"Missing rasters.{key} in config")
    raw = rasters[key]
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    # Shared ESSD year map lives in Data_clean
    if key == "disturbance_year":
        return resolve_under(output_dir(cfg), p)
    return resolve_under(features_root(cfg), p)
