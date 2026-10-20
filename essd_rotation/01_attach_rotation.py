#!/usr/bin/env python3
"""Cascade-fill NFI rotation onto disturbance trends.

Cascade: TFV×SER → TFV×GRECO → national TFV. Alias-at-SER → ``gap_filled=3``.
Writes ``rotation_periods.csv`` and ``disturbance_trends_with_rotation.csv``.
Natural-agent rows get NaN rotation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.rotation.attach import attach_rotation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    print("=== 01 attach rotation ===")
    attach_rotation(cfg)


if __name__ == "__main__":
    main()
