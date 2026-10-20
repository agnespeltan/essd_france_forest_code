#!/usr/bin/env python3
"""Build single-band latest disturbance year + agent GeoTIFFs from the agent stack.

Reads rasters.agent_multilayer, writes two co-registered DEFLATE GeoTIFFs
(default under output_dir):
  - latest_disturbance_year_le2024_30m.tif   (uint16)
  - latest_disturbance_agent_le2024_30m.tif  (uint8; agent at that year)

Run this before 03_build_agb_stats.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster.agent_stack import materialize_from_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML config (default: ./config.yaml)",
    )
    p.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Inclusive latest year (default: config agent_last_year). "
        "Use 2021 for age-map nodata fix (keep clearcuts in 2019–2021).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if outputs already exist and are newer than the stack",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_year is not None:
        cfg["agent_last_year"] = int(args.max_year)
    year_out, agent_out = materialize_from_config(cfg, force=args.force)
    print(f"Done:\n  {year_out}\n  {agent_out}")


if __name__ == "__main__":
    main()
