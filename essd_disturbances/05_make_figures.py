#!/usr/bin/env python3
"""Join trends+severity and write the disturbance figures (Fig. 4–5, SI7–SI11).

Plotting lives in ``lib/figures/fig_disturbances.py``; every input path and
``year_start`` come from the config. η² is computed only by ``07_compute_variance.py``.

Example::

  python3 05_make_figures.py --config config.yaml
  python3 05_make_figures.py --prepare-only
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
from lib.figures.fig_disturbances import run_figures
from lib.figures.prepare_table import write_figure_table

_REQUIRED_FOR_FIGURES = ("figure_dir", "ser_shapefile", "france_outline_shapefile",
                         "kowalski_loss_csv", "kowalski_grid_shp", "year_start")


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
        "--prepare-only",
        action="store_true",
        help="Only write the joined CSV; do not plot",
    )
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse existing joined CSV under output_dir",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Enable matplotlib interactive mode (default: Agg / off)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"]).expanduser()
    table_name = cfg.get("outputs", {}).get(
        "figure_table", "disturbance_trends_for_figures.csv"
    )
    table_path = out_dir / table_name

    if not args.skip_prepare:
        table_path = write_figure_table(cfg, out_name=table_name)
        print(f"Wrote figure table: {table_path} ({table_path.stat().st_size / 1e6:.2f} MB)")
    elif not table_path.is_file():
        raise FileNotFoundError(
            f"Joined table not found: {table_path}\n"
            "Run without --skip-prepare first."
        )

    if args.prepare_only:
        return

    if args.figure_dir is not None:
        cfg["figure_dir"] = str(args.figure_dir)
    missing = [k for k in _REQUIRED_FOR_FIGURES if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in {args.config}")
    fig_dir = Path(cfg["figure_dir"]).expanduser()

    if not args.interactive:
        os.environ.setdefault("MPLBACKEND", "Agg")

    print(f"Plotting from {table_path}")
    print(f"  figures → {fig_dir}")
    out = run_figures(
        table_path,
        fig_dir,
        year_start=int(cfg["year_start"]),
        ser_shapefile=cfg["ser_shapefile"],
        france_outline_shp=cfg["france_outline_shapefile"],
        kowalski_loss_csv=cfg["kowalski_loss_csv"],
        kowalski_grid_shp=cfg["kowalski_grid_shp"],
        interactive=bool(args.interactive),
    )
    print(f"Done. Figures in {out}")


if __name__ == "__main__":
    main()
