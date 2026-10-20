#!/usr/bin/env python3
"""Rebuild ``forest_age_q*_hist.csv``, ``*_5km.tif``, and ``*_cv_5km.tif`` from fixed age maps.

Example::

  python3 03b_build_summaries.py --force
  python3 03b_build_summaries.py --quantile 50 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.postprocess.summaries import build_all_summaries, build_summaries_for_quantile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--quantile", type=int, default=None, help="Single quantile (default: 25,50,75)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.quantile is not None:
        outs = [build_summaries_for_quantile(cfg, args.quantile, force=args.force, n_jobs=args.n_jobs)]
    else:
        outs = build_all_summaries(cfg, force=args.force, n_jobs=args.n_jobs)
    for hist, km, cv in outs:
        print(f"  {hist.name} | {km.name} | {cv.name}")
    print("Done.")


if __name__ == "__main__":
    main()
