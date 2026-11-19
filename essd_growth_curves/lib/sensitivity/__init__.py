"""Sensitivity of the growth curves to the fit settings, scored against the NFI (08, 09).

Config block ``sensitivity`` (see config.example.yaml); the NFI tables and TFV areas
come from ``figures.nfi_greco_csv``, ``figures.nfi_species_csv`` and ``figures.area_parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.paths import output_dir

BASELINE_SWEEP = "baseline_factor"  # folder of the a0 × f grid
NFI_KEYS = ("nfi_greco_csv", "nfi_species_csv", "area_parquet")


def sensitivity_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    sens = cfg.get("sensitivity")
    if not sens:
        raise SystemExit("config has no 'sensitivity' block (see config.example.yaml)")
    return sens


def sensitivity_dir(cfg: dict[str, Any]) -> Path:
    """Root folder of the sweeps (``sensitivity.out_dir``, relative to output_dir)."""
    sub = Path(sensitivity_cfg(cfg).get("out_dir", "sensitivity")).expanduser()
    return sub if sub.is_absolute() else output_dir(cfg) / sub


def sweep_names(cfg: dict[str, Any]) -> list[str]:
    """Sweep folders in run order: the a0 × f grid first, then each settings sweep."""
    sens = sensitivity_cfg(cfg)
    names = [BASELINE_SWEEP] if sens.get("baseline_grid") else []
    return names + list((sens.get("settings") or {}).keys())


def nfi_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    figs = cfg.get("figures") or {}
    missing = [k for k in NFI_KEYS if not figs.get(k)]
    if missing:
        raise SystemExit(f"sensitivity tests need figures.{', figures.'.join(missing)} in the config")
    return {k: Path(figs[k]).expanduser() for k in NFI_KEYS}
