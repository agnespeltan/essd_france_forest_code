#!/usr/bin/env python3
"""Invert AGB→age per GRECO, only where that GRECO's weight > 0.

Requires ``00_build_greco_weights.py`` (or existing ``outputs.weights_dir``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.invert.tiles import build_all_age_tiles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--greco",
        type=str,
        default=None,
        help="Single GRECO id (e.g. A). Default: all, sequentially.",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Smoke test: process only the first N weight-bbox blocks",
    )
    p.add_argument(
        "--max-polygons",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --max-blocks
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    max_blocks = args.max_blocks if args.max_blocks is not None else args.max_polygons
    paths = build_all_age_tiles(
        cfg,
        force=args.force or bool(cfg.get("force")),
        max_blocks=max_blocks,
        greco=args.greco,
    )
    for path in paths:
        print(f"Done: {path}")


if __name__ == "__main__":
    main()
