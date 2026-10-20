#!/usr/bin/env python3
"""Aggregate NFI *harvested* tree ages → stratified harvest-age CSVs.

Keeps only trees with ``coupe==1`` (cut between successive visits), then maps
ESPAR→TFV via package ``data/espar_to_tfv.csv`` (pure + mixture 80/90/100) and
``data/espar_corr_age.csv``. No Excel at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.nfi.harvest_ages import build_harvest_ages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    print("=== 00 build harvest ages ===")
    build_harvest_ages(cfg)


if __name__ == "__main__":
    main()
