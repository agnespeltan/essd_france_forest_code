#!/usr/bin/env python3
"""Build single-band latest disturbance year GeoTIFF (year < reference_year).

Reads rasters.agent_multilayer, writes one DEFLATE uint16 GeoTIFF
(default: output_dir/latest_disturbance_year_lt2020_30m.tif).
Run this before 01_build_efda / 02_build_gami.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster_stats.agent_stack import materialize_from_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML config (default: ./config.yaml)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the output already exists and is newer than the stack",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out = materialize_from_config(cfg, force=args.force)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
