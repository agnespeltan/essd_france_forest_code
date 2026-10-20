#!/usr/bin/env python3
"""Run materialize → patches → trends → AGB stats → severity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster.agent_stack import materialize_from_config
from lib.raster.agb_stats import build_agb_stats
from lib.raster.patches import build_patches
from lib.severity.pipeline import compute_severity
from lib.trends.pipeline import fit_trends


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-polygons", type=int, default=None, help="Smoke test: first N SER polygons")
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip 00–01 and 03; use existing rasters/parquets (still runs 02 trends + 04)",
    )
    p.add_argument(
        "--force-year",
        action="store_true",
        help="Rebuild latest-year + latest-agent GeoTIFFs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers

    if not args.skip_build:
        print("=== 00 materialize latest year + agent ===")
        year_out, agent_out = materialize_from_config(cfg, force=args.force_year)
        print(year_out)
        print(agent_out)
        print("=== 01 build patches ===")
        print(build_patches(cfg, max_polygons=args.max_polygons))
    print("=== 02 fit trends ===")
    print(fit_trends(cfg))
    if not args.skip_build:
        print("=== 03 build AGB stats ===")
        print(build_agb_stats(cfg, max_polygons=args.max_polygons))
    print("=== 04 compute severity ===")
    print(compute_severity(cfg))


if __name__ == "__main__":
    main()
