#!/usr/bin/env python3
"""Build EFDA GRECO growth statistics (disturbance-year ages, young stands)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster_stats.efda import build_efda_stats


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
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    out = build_efda_stats(cfg, max_polygons=args.max_polygons)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
