#!/usr/bin/env python3
"""Compute conservative AGB disturbance severity (SER → GRECO → France)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.severity.pipeline import compute_severity


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
    path = compute_severity(cfg)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
