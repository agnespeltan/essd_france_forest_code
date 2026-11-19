#!/usr/bin/env python3
"""Bootstrap uncertainty of the growth curves (see lib/fit/bootstrap.py).

Resamples the age points of every fitted TFV × GRECO (B draws), refits, and writes

  growth_curves_agb_greco_bootstrap.csv          every draw (long)
  growth_curves_agb_greco_uncertainty.csv        per-stratum SE / CI / spread / total

With ``--merge`` the uncertainty columns are also added to the published curves
CSV (a backup ``*.pre_bootstrap.csv`` is written first; existing columns are kept).

Example::

  python3 07_bootstrap_curves.py --config config.yaml --n-boot 200 --merge
  python3 07_bootstrap_curves.py --max-strata 5 --n-boot 20     # smoke test
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.fit.bootstrap import run_bootstrap, summarise  # noqa: E402
from lib.fit.preprocess import load_merged_stats  # noqa: E402
from lib.paths import curves_path, output_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-strata", type=int, default=None, help="Smoke test: first N strata only")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--merge", action="store_true", help="Add the columns to the published curves CSV")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    out_dir = args.out_dir or output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves_csv = curves_path(cfg, "agb")
    curves = pd.read_csv(curves_csv)
    fitted = curves.loc[curves["gap_filled"] == 0, ["TFV", "greco"]]
    pairs = [(str(t), g) for t, g in zip(fitted["TFV"], fitted["greco"])]
    if args.max_strata:
        pairs = pairs[: args.max_strata]

    df = load_merged_stats(cfg)
    draws = run_bootstrap(cfg, df, pairs, n_boot=args.n_boot, seed=args.seed)
    draws_csv = out_dir / "growth_curves_agb_greco_bootstrap.csv"
    draws.to_csv(draws_csv, index=False)
    print(f"Draws → {draws_csv} ({len(draws)} rows, {draws['ok'].mean():.1%} converged)")

    unc = summarise(draws, curves, seed=args.seed + 1)
    if args.max_strata:
        unc = unc.dropna(subset=["boot_n"])
    unc_csv = out_dir / "growth_curves_agb_greco_uncertainty.csv"
    unc.to_csv(unc_csv, index=False)
    print(f"Uncertainty → {unc_csv} ({len(unc)} rows, {unc['boot_n'].notna().sum()} with bootstrap)")

    if args.merge:
        if args.max_strata:
            raise SystemExit("--merge refused with --max-strata")
        backup = curves_csv.with_suffix(".pre_bootstrap.csv")
        if not backup.exists():
            shutil.copy2(curves_csv, backup)
        new_cols = [c for c in unc.columns if c not in ("TFV", "greco")]
        base = curves.drop(columns=[c for c in new_cols if c in curves.columns])
        key = base[["TFV", "greco"]].astype(str)
        merged = pd.concat([base, key.merge(unc.astype({"TFV": str, "greco": str}), on=["TFV", "greco"],
                                            how="left")[new_cols]], axis=1)
        assert len(merged) == len(curves)
        merged.to_csv(curves_csv, index=False)
        print(f"Merged {len(new_cols)} columns into {curves_csv} (backup {backup.name})")


if __name__ == "__main__":
    main()
