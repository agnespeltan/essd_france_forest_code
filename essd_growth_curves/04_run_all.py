#!/usr/bin/env python3
"""Run materialize year → EFDA → GAMI → fit in order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.fit.pipeline import fit_growth_curves
from lib.raster_stats.agent_stack import materialize_from_config
from lib.raster_stats.efda import build_efda_stats
from lib.raster_stats.gami import build_gami_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML config (default: ./config.yaml)",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-polygons", type=int, default=None, help="Smoke test: first N GRECO polygons")
    p.add_argument("--force-year", action="store_true", help="Rebuild latest-year GeoTIFF")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers

    print("=== 00 materialize year ===")
    print(materialize_from_config(cfg, force=args.force_year))
    print("=== 01 build EFDA ===")
    print(build_efda_stats(cfg, max_polygons=args.max_polygons))
    print("=== 02 build GAMI ===")
    print(build_gami_stats(cfg, max_polygons=args.max_polygons))
    print("=== 03 fit curves ===")
    for target, path in fit_growth_curves(cfg).items():
        print(f"{target}: {path}")


if __name__ == "__main__":
    main()
