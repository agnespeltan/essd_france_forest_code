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

    cfg.setdefault("workers", 4)
    cfg.setdefault("reference_year", 2020)
    cfg.setdefault("min_age_gami", 35)
    cfg.setdefault("age_baseline_age", 2)
    cfg.setdefault("age_baseline_factor", 0.85)
    cfg.setdefault("r2_min", 0.2)
    cfg.setdefault("r2_75_min", 0.2)  # legacy alias; prefer r2_min
    cfg.setdefault("excluded_tfv", sorted(DEFAULT_EXCLUDED_TFV))
    cfg.setdefault("fit_targets", ["agb"])
    cfg.setdefault(
        "stats",
        {
            "efda": "growth_stats_efda_greco.parquet",
            "gami": "growth_stats_gami_greco.parquet",
        },
    )
    cfg.setdefault(
        "curves",
        {
            "agb": "growth_curves_agb_greco.csv",
        },
    )
    cfg["_config_path"] = str(config_path)
    return cfg


def excluded_tfv_set(cfg: dict[str, Any]) -> frozenset[str]:
    """Return excluded TFV labels as strings."""
    return frozenset(str(x) for x in cfg.get("excluded_tfv", DEFAULT_EXCLUDED_TFV))
