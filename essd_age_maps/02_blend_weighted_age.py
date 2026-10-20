#!/usr/bin/env python3
"""Blend per-GRECO age tiles with distance weights into a national age map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.blend.weighted import create_weighted_age_map
from lib.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = create_weighted_age_map(cfg, force=args.force or bool(cfg.get("force")))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
