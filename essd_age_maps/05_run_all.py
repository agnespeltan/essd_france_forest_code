#!/usr/bin/env python3
"""Run the full age-map pipeline locally.

Default: 00 weights → 01 all GRECO tiles (sequential) → 02 blend → 03 fix_nodata.
Optional: --with-figures runs figure generation after compute.

Examples::

  python3 05_run_all.py
  python3 05_run_all.py --skip-weights          # reuse existing GRECO weights
  python3 05_run_all.py --with-figures
  python3 05_run_all.py --max-blocks 20         # smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.blend.weighted import create_weighted_age_map
from lib.config import load_config
from lib.invert.tiles import build_all_age_tiles, list_greco_ids
from lib.postprocess.fix_nodata import fix_nodata
from lib.weights.distance_masks import build_greco_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Polygons per GRECO tile (default from config; keep 1 on laptops)",
    )
    p.add_argument(
        "--greco",
        type=str,
        default=None,
        help="Single GRECO id for tiles (default: all, sequentially)",
    )
    p.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Smoke: first N weight-bbox blocks per GRECO tile",
    )
    p.add_argument("--max-polygons", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--force", action="store_true", help="Rebuild existing products")
    p.add_argument(
        "--quantile",
        type=int,
        default=None,
        help="CR quantile (25/50/75); default from config",
    )
    p.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip 00; use existing greco_*_weights.tif",
    )
    p.add_argument(
        "--skip-tiles",
        action="store_true",
        help="Skip 01; use existing age tiles",
    )
    p.add_argument("--skip-blend", action="store_true", help="Skip 02 blend")
    p.add_argument("--skip-fix", action="store_true", help="Skip 03 fix_nodata")
    p.add_argument(
        "--with-figures",
        action="store_true",
        help="Also run figures (04) after compute",
    )
    p.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="PNG output dir when --with-figures (default: config figure_dir)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.quantile is not None:
        cfg["quantile"] = int(args.quantile)
    cfg.setdefault("low_memory", True)
    if int(cfg.get("workers", 1)) > 1 and cfg.get("low_memory", True):
        print(
            f"Note: low_memory=True → using workers=1 (requested {cfg['workers']})"
        )
        cfg["workers"] = 1
    force = args.force or bool(cfg.get("force"))

    print("=== Local age-map pipeline ===")
    print(f"config: {args.config}")
    print(f"output_dir: {cfg.get('output_dir')}")
    print(f"workers={cfg.get('workers')} low_memory={cfg.get('low_memory')}")

    if not args.skip_weights:
        print("=== 00 build GRECO weights ===")
        print(build_greco_weights(cfg, force=force))
    else:
        print("=== 00 skipped (--skip-weights) ===")

    if not args.skip_tiles:
        ids = [args.greco] if args.greco else list_greco_ids(cfg)
        print(f"=== 01 build age tiles ({len(ids)} GRECO, sequential) ===")
        for i, gid in enumerate(ids, start=1):
            print(f"--- GRECO {gid} ({i}/{len(ids)}) ---")
            path = build_all_age_tiles(
                cfg,
                force=force,
                max_blocks=args.max_blocks
                if args.max_blocks is not None
                else args.max_polygons,
                greco=gid,
            )[0]
            print(path)
    else:
        print("=== 01 skipped (--skip-tiles) ===")

    if not args.skip_blend:
        print("=== 02 blend weighted age ===")
        print(create_weighted_age_map(cfg, force=force))
    else:
        print("=== 02 skipped (--skip-blend) ===")

    if not args.skip_fix:
        print("=== 03 fix nodata ===")
        print(fix_nodata(cfg, force=force))
    else:
        print("=== 03 skipped (--skip-fix) ===")

    if args.with_figures:
        print("=== 04 make figures ===")
        os.environ.setdefault("MPLBACKEND", "Agg")
        from lib.figures.fig_age import run_figures

        out = run_figures(cfg, figure_dir=args.figure_dir)
        print(f"Figures: {out}")

    print("=== Done (local) ===")


if __name__ == "__main__":
    main()
