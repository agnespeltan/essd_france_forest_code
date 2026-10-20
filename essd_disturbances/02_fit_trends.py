#!/usr/bin/env python3
"""Fit Mann–Kendall disturbance trends (SER → GRECO → France)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.trends.pipeline import fit_trends


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML config (default: ./config.yaml)",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    path = fit_trends(cfg)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
