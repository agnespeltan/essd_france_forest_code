#!/usr/bin/env python3
"""Fit Chapman–Richards growth curves from EFDA + GAMI stats parquets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.fit.pipeline import fit_growth_curves


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML config (default: ./config.yaml)",
    )
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    written = fit_growth_curves(cfg)
    for target, path in written.items():
        print(f"Done [{target}]: {path}")


if __name__ == "__main__":
    main()
