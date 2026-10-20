"""Load and validate YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lib.constants import DEFAULT_EXCLUDED_TFV


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and apply defaults."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    cfg.setdefault("year_start", 1999)
    cfg.setdefault("year_end", 2024)
    cfg.setdefault("min_years_ser", 10)
    cfg.setdefault("min_years_mk", 4)
    cfg.setdefault("before_range", "2021-2024")
    cfg.setdefault("after_range", "2016-2019")
    cfg.setdefault("agent_first_year", 1985)
    cfg.setdefault("agent_last_year", 2024)
    cfg.setdefault("agent_nodata", 255)
    cfg.setdefault("agent_min_valid_value", 1)
    cfg.setdefault("latest_year_block_size", 1024)
    cfg.setdefault("force_rebuild_latest_year", False)
    cfg.setdefault("workers", 4)
    cfg.setdefault("include_forest_aggregate", True)
    # ser_shapefile is required for 00/01 builders
    cfg.setdefault("excluded_tfv", sorted(DEFAULT_EXCLUDED_TFV))
    cfg.setdefault(
        "outputs",
        {
            "patches": "disturbance_patches.parquet",
            "agb_stats": "disturbance_agb_stats.parquet",
            "trends": "disturbance_trends.csv",
            "severity": "disturbance_severity.csv",
        },
    )
    # Resolve intermediate parquet paths (explicit override wins)
    out_dir = Path(cfg.get("output_dir", ".")).expanduser()
    outs = cfg["outputs"]
    cfg.setdefault("patches_parquet", str(out_dir / outs.get("patches", "disturbance_patches.parquet")))
    cfg.setdefault(
        "agb_stats_parquet", str(out_dir / outs.get("agb_stats", "disturbance_agb_stats.parquet"))
    )
    cfg["_config_path"] = str(config_path)
    return cfg


def excluded_tfv_set(cfg: dict[str, Any]) -> frozenset[str]:
    """Return excluded TFV labels as lowercase strings."""
    return frozenset(str(x).lower() for x in cfg.get("excluded_tfv", DEFAULT_EXCLUDED_TFV))


def output_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve an outputs.<key> filename under output_dir."""
    out_dir = Path(cfg["output_dir"]).expanduser()
    name = cfg.get("outputs", {}).get(key)
    if not name:
        raise KeyError(f"Missing outputs.{key} in config")
    return out_dir / name
