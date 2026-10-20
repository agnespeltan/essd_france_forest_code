#!/usr/bin/env python3
"""Join trends+severity and write disturbance combined figures (self-contained).

Does **not** call ``Figures/Fig_disturbances.py``. Plotting lives in
``lib/figures/fig_disturbances.py``. No rotation period — those panels stay off.

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

_DEFAULT_FIG_DIR = Path(
    "/home/apelliss/Documents/Resultat_France/Figures/ESSD"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="PNG output dir (default: config figure_dir or Figures/ESSD)",
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

    fig_dir = (
        Path(args.figure_dir).expanduser()
        if args.figure_dir is not None
        else Path(cfg.get("figure_dir", _DEFAULT_FIG_DIR)).expanduser()
    )

    if not args.interactive:
        os.environ.setdefault("MPLBACKEND", "Agg")

    ser_shp = cfg.get("ser_shapefile")
    greco_shp = cfg.get("greco_shapefile")
    kw = {"interactive": bool(args.interactive)}
    if ser_shp:
        kw["ser_shapefile"] = ser_shp
    if greco_shp:
        kw["greco_shapefile"] = greco_shp
    if cfg.get("kowalski_loss_csv"):
        kw["kowalski_loss_csv"] = cfg["kowalski_loss_csv"]
    if cfg.get("kowalski_grid_shp"):
        kw["kowalski_grid_shp"] = cfg["kowalski_grid_shp"]

    print(f"Plotting from {table_path}")
    print(f"  figures → {fig_dir}")
    out = run_figures(table_path, fig_dir, **kw)
    print(f"Done. Figures in {out}")


if __name__ == "__main__":
    main()
