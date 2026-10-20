#!/usr/bin/env python3
"""One-way η² variance partitioning for disturbance (+ optional rotation) metrics.

Equal weight per species × small ecological region × agent cell. Gap-filled values
excluded. 95% CIs from Monte Carlo perturbation within each row SE (n=1000).
Significance from label-permutation tests (n=1000). Rotation (harvest only) is
excluded from the agent factor.

Example::

  python3 07_compute_variance.py --config config.yaml
  python3 07_compute_variance.py --rotation /path/to/disturbance_trends_with_rotation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.variance.pipeline import compute_disturbance_eta2, format_eta2_summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--rotation",
        type=Path,
        default=None,
        help="Trends+rotation CSV (default: output_dir/disturbance_trends_with_rotation.csv if present)",
    )
    p.add_argument("--n-mc", type=int, default=1000, help="Monte Carlo draws for 95%% CI")
    p.add_argument("--n-perm", type=int, default=1000, help="Permutations for p-values")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: output_dir/disturbance_eta2_variance.csv)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    print(
        f"Computing disturbance η² (equal weight, gap-filled excluded, "
        f"n_mc={args.n_mc}, n_perm={args.n_perm})…"
    )
    table, summary = compute_disturbance_eta2(
        cfg,
        rotation_csv=args.rotation,
        n_mc=args.n_mc,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    out = args.out
    if out is None:
        out = Path(cfg["output_dir"]).expanduser() / "disturbance_eta2_variance.csv"
    else:
        out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {summary_path}")
    print()
    print(format_eta2_summary(table, summary))


if __name__ == "__main__":
    main()
