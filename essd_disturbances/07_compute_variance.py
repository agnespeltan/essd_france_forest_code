#!/usr/bin/env python3
"""One-way η² variance partitioning (+ two-way species vs region partition) for disturbance (+ optional mean harvest age) metrics.

Equal weight per species × small ecological region × agent cell. Gap-filled values
excluded. 95% CIs from Monte Carlo perturbation within each row SE (n=1000).
Significance from label-permutation tests (n=1000). Mean harvest age (harvest only) is
excluded from the agent factor.

Example::

  python3 07_compute_variance.py --config config.yaml
  python3 07_compute_variance.py --harvest-age /path/to/disturbance_trends_with_harvest_age.csv
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
from lib.variance.pipeline import compute_disturbance_eta2, compute_disturbance_eta2_twoway, format_eta2_summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--harvest-age",
        "--rotation",  # pre-rename alias
        dest="harvest_age",
        type=Path,
        default=None,
        help="Trends+harvest age CSV (default: output_dir/disturbance_trends_with_harvest_age.csv if present)",
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
        harvest_age_csv=args.harvest_age,
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

    two = compute_disturbance_eta2_twoway(
        cfg, harvest_age_csv=args.harvest_age, n_mc=args.n_mc, seed=args.seed
    )
    two_out = out.with_name(out.stem.replace("_variance", "") + "_twoway.csv")
    two.to_csv(two_out, index=False)
    print(f"\nWrote {two_out}")
    print("Two-way partition, species vs region (% of variance after agent; 95% MC CI):")
    for _, r in two.iterrows():
        print(
            f"  {r['parameter']:16s} species {r['unique_species']:5.1f} "
            f"[{r['unique_species_lo']:.1f}, {r['unique_species_hi']:.1f}] | region "
            f"{r['unique_region']:5.1f} [{r['unique_region_lo']:.1f}, {r['unique_region_hi']:.1f}] | "
            f"shared {r['shared']:5.1f} | residual {r['residual']:5.1f} | adj species "
            f"{r['unique_species_adj']:5.1f} region {r['unique_region_adj']:5.1f}"
        )


if __name__ == "__main__":
    main()
