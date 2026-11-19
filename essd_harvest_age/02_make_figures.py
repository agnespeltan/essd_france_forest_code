#!/usr/bin/env python3
"""Plot mean-harvest-age figures from trends+harvest age CSV.

Example::

  python3 02_make_figures.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config, output_path
from lib.figures.fig_harvest_age import run_figures



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="PNG output dir (default: figure_dir in the config)",
    )
    p.add_argument(
        "--trends",
        type=Path,
        default=None,
        help="Override trends+harvest age CSV (default: outputs.trends_with_harvest_age)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    trends_path = (
        Path(args.trends).expanduser()
        if args.trends is not None
        else output_path(cfg, "trends_with_harvest_age")
    )
    if not trends_path.is_file():
        raise FileNotFoundError(
            f"Trends+harvest age not found: {trends_path}\nRun 01_attach_harvest_age.py first."
        )

    fig_dir = (
        Path(args.figure_dir).expanduser()
        if args.figure_dir is not None
        else Path(cfg["figure_dir"]).expanduser()
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    ser_shp = cfg.get("ser_shapefile")
    if not ser_shp:
        raise KeyError("ser_shapefile is required for figures")

    print(f"Plotting from {trends_path}")
    print(f"  figures → {fig_dir}")
    out = run_figures(
        trends_path,
        fig_dir,
        ser_shapefile=ser_shp,
        france_outline=cfg.get("france_outline_shapefile"),
    )
    print(f"Done. Figures in {out}")


if __name__ == "__main__":
    main()
