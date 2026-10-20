#!/usr/bin/env python3
"""Run harvest ages → attach rotation → figures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config, output_path
from lib.figures.fig_rotation import run_figures
from lib.nfi.harvest_ages import build_harvest_ages
from lib.rotation.attach import attach_rotation

_DEFAULT_FIG_DIR = Path("/home/apelliss/Documents/Resultat_France/Figures/ESSD")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--skip-ages",
        action="store_true",
        help="Skip 00; reuse existing age_coupe_*.csv under output_dir",
    )
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip 02 figures",
    )
    p.add_argument("--figure-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    if not args.skip_ages:
        print("=== 00 build harvest ages ===")
        build_harvest_ages(cfg)
    print("=== 01 attach rotation ===")
    attach_rotation(cfg)

    if args.skip_figures:
        return

    print("=== 02 make figures ===")
    os.environ.setdefault("MPLBACKEND", "Agg")
    fig_dir = (
        Path(args.figure_dir).expanduser()
        if args.figure_dir is not None
        else Path(cfg.get("figure_dir", _DEFAULT_FIG_DIR)).expanduser()
    )
    ser_shp = cfg.get("ser_shapefile")
    if not ser_shp:
        raise KeyError("ser_shapefile is required for figures")
    run_figures(
        output_path(cfg, "trends_with_rotation"),
        fig_dir,
        ser_shapefile=ser_shp,
        greco_shapefile=cfg.get("greco_shapefile"),
    )


if __name__ == "__main__":
    main()
