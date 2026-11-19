#!/usr/bin/env python3
"""Cascade-fill NFI mean harvest age onto disturbance trends.

Cascade: TFV×SER → TFV×GRECO → national TFV. Alias-at-SER → ``harvest_age_gap_filled=3``.
Preserves trends ``gap_filled`` (rate 0/1/2). Writes ``mean_harvest_age.csv`` and
``disturbance_trends_with_harvest_age.csv``. Natural-agent rows get NaN harvest age.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.harvest_age.attach import attach_harvest_age


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    print("=== 01 attach mean harvest age ===")
    attach_harvest_age(cfg)


if __name__ == "__main__":
    main()
