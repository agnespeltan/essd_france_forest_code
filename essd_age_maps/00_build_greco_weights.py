#!/usr/bin/env python3
"""Build GRECO distance weight rasters (tanh soft edges)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.weights.distance_masks import build_greco_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--force", action="store_true", help="Rebuild existing weights")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = build_greco_weights(cfg, force=args.force or bool(cfg.get("force")))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
