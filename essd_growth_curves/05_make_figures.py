#!/usr/bin/env python3
"""Write paper growth-curve figures (Fig. 2, SI1, SI2).

Self-contained plotting in ``lib/figures/`` (does not call
``Figures/Fig_age_distribution.py``).

Example::

  python3 05_make_figures.py --config config.yaml
  python3 05_make_figures.py --figure-dir /path/to/Figures/ESSD
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.figures.fig_growth import run_figures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="PNG output dir (default: config figure_dir)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Enable matplotlib interactive mode (default: Agg / off)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.figure_dir is not None:
        fig_dir = Path(args.figure_dir).expanduser()
    elif cfg.get("figure_dir"):
        fig_dir = Path(cfg["figure_dir"]).expanduser()
    else:
        raise KeyError("figure_dir missing (set in config or pass --figure-dir)")

    if not args.interactive:
        os.environ.setdefault("MPLBACKEND", "Agg")

    print(f"Plotting growth figures → {fig_dir}")
    out = run_figures(cfg, figure_dir=fig_dir, interactive=bool(args.interactive))
    print(f"Done. Figures in {out}")


if __name__ == "__main__":
    main()
