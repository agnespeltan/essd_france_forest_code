#!/usr/bin/env python3
"""Materialize latest disturbance year capped to the age-map keep window.

Default keep years are 2019–2021 → writes
``output_dir/latest_disturbance_year_le2021_30m.tif`` from
``rasters.agent_multilayer`` (years ≤ 2021 only).

Used by ``03_fix_nodata`` so age==0 is kept for clearcuts in 2019–2021.

Example::

  python3 00b_materialize_disturbance_year.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster.disturbance_year import ensure_disturbance_year, keep_max_year, keep_years


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    years = sorted(keep_years(cfg))
    print(f"Keep window: {years} → materialize year <= {keep_max_year(cfg)}")
    out = ensure_disturbance_year(cfg, force=args.force or bool(cfg.get("force")))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
