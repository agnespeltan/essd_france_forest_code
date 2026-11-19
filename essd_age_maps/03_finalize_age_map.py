#!/usr/bin/env python3
"""Final age map and its figure inputs, per growth-curve quantile.

1. Fix nodata: age == 0 → nodata where AGB == 0, unless the latest disturbance
   year is in the keep window (config ``disturbance_keep_years``, default
   2019–2021) → ``forest_age_<label>.tif``. Builds
   ``latest_disturbance_year_le2021_30m.tif`` first if it is missing.
2. Summaries from that map → ``forest_age_<label>_hist.csv``, ``_5km.tif``
   (median) and ``_cv_5km.tif`` (spatial CV). Skipped when already newer than the map.

Labels: 50 central, 25 upper (older ages), 75 lower (younger ages).

Examples::

  python3 03_finalize_age_map.py                    # config quantile
  python3 03_finalize_age_map.py --all-quantiles    # 50, 25 and 75
  python3 03_finalize_age_map.py --skip-fix --all-quantiles --force   # rebuild summaries only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.naming import age_label
from lib.postprocess.fix_nodata import fix_nodata
from lib.postprocess.summaries import build_summaries_for_quantile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    q = p.add_mutually_exclusive_group()
    q.add_argument("--quantile", type=int, default=None,
                   help="Curve quantile: 50 central, 25 upper, 75 lower (default: config)")
    q.add_argument("--all-quantiles", action="store_true", help="Run 50, 25 and 75")
    p.add_argument("--skip-fix", action="store_true", help="Only rebuild the summaries")
    p.add_argument("--skip-summaries", action="store_true", help="Only fix nodata")
    p.add_argument("--force", action="store_true", help="Rebuild outputs that already exist")
    p.add_argument("--n-jobs", type=int, default=4, help="Threads for the histogram")
    args = p.parse_args()

    cfg = load_config(args.config)
    force = args.force or bool(cfg.get("force"))
    if args.all_quantiles:
        quantiles = [50, 25, 75]
    else:
        quantiles = [int(args.quantile if args.quantile is not None else cfg.get("quantile", 50))]

    for quantile in quantiles:
        qcfg = dict(cfg, quantile=quantile)
        print(f"=== quantile {quantile} ({age_label(quantile)}) ===")
        if not args.skip_fix:
            print(f"  fixed map: {fix_nodata(qcfg, force=force)}")
        if not args.skip_summaries:
            hist, km, cv = build_summaries_for_quantile(qcfg, quantile, force=force, n_jobs=args.n_jobs)
            print(f"  summaries: {hist.name} | {km.name} | {cv.name}")
    print("Done.")


if __name__ == "__main__":
    main()
