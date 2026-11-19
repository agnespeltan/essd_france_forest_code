"""Load and validate YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lib.constants import (
    DEFAULT_AGE_OFFSET,
    DEFAULT_MAX_AGE13,
    DEFAULT_MIN_COUNT,
    DEFAULT_TFV_ALIASES,
    PACKAGE_DATA,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and apply defaults."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    cfg.setdefault("min_count", DEFAULT_MIN_COUNT)
    cfg.setdefault("age_offset", DEFAULT_AGE_OFFSET)
    cfg.setdefault("max_age13", DEFAULT_MAX_AGE13)
    cfg.setdefault("include_mixtures", True)
    cfg.setdefault("tfv_aliases", [list(p) for p in DEFAULT_TFV_ALIASES])
    cfg.setdefault(
        "outputs",
        {
            "ages_tfv": "age_coupe_TFV_NFI.csv",
            "ages_greco_tfv": "age_coupe_greco_TFV_NFI.csv",
            "ages_ser_tfv": "age_coupe_ser_TFV_NFI.csv",
            "ages_ser_forest": "age_coupe_ser_forest_NFI.csv",
            "ages_greco": "age_coupe_greco_NFI.csv",
            "ages_national": "age_coupe_NFI.csv",
            "harvest_age_table": "mean_harvest_age.csv",
            "trends_with_harvest_age": "disturbance_trends_with_harvest_age.csv",
            "trends_with_harvest_age_size_freq": (
                "disturbance_trends_with_harvest_age_and_size_frequency.csv"
            ),
            "trends_combined": (
                "disturbance_trends_with_harvest_age_size_frequency_and_severity.csv"
            ),
            "patch_size_mle": "patch_size_selected_mle.csv",
            "severity": "disturbance_severity.csv",
        },
    )

    # Back-compat: pre-rename output keys (rotation → harvest_age) in older configs.
    outputs = cfg["outputs"]
    for old, new in (
        ("rotation_table", "harvest_age_table"),
        ("trends_with_rotation", "trends_with_harvest_age"),
        ("trends_with_rotation_size_freq", "trends_with_harvest_age_size_freq"),
    ):
        if old in outputs and new not in outputs:
            outputs[new] = outputs.pop(old)

    species = cfg.setdefault("species", {})
    species.setdefault("espar_to_tfv", str(PACKAGE_DATA / "espar_to_tfv.csv"))
    species.setdefault("espar_corr_age", str(PACKAGE_DATA / "espar_corr_age.csv"))

    cfg["_config_path"] = str(config_path)
    return cfg


def output_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve an outputs.<key> filename under output_dir."""
    out_dir = Path(cfg["output_dir"]).expanduser()
    name = cfg.get("outputs", {}).get(key)
    if not name:
        raise KeyError(f"Missing outputs.{key} in config")
    return out_dir / name


def tfv_alias_pairs(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    """Return (target_tfv, source_tfv) pairs from config."""
    pairs: list[tuple[int, int]] = []
    for item in cfg.get("tfv_aliases", DEFAULT_TFV_ALIASES):
        if len(item) != 2:
            raise ValueError(f"tfv_aliases entries must be [target, source], got {item}")
        pairs.append((int(item[0]), int(item[1])))
    return pairs
