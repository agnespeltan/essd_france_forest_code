#!/usr/bin/env python3
"""Census SER×TFV forest area from the TFV mask (independent of disturbances)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.raster.forest_area import build_forest_area


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    path = build_forest_area(cfg)
    import pandas as pd

    df = pd.read_parquet(path)
    print(f"Done: {path} ({len(df)} SER×TFV keys)")


if __name__ == "__main__":
    main()
