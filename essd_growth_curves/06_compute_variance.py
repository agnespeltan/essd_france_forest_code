#!/usr/bin/env python3
"""Compute equal-weight one-way η² for growth-curve metrics (AGBmax, R30, t50).

Factors: species category (TFV) vs ecological region (GRECO letter).
Gap-filled curves excluded by default. 95% CIs: Monte Carlo perturbation within
each value's 25th–75th percentile range (n=1000). Significance: label permutation
(n=1000).

Example::

  python3 06_compute_variance.py --config config.yaml
  python3 06_compute_variance.py --n-mc 1000 --n-perm 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.paths import output_dir
from lib.variance.eta2 import compute_eta2_table, format_eta2_paragraph


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--n-mc", type=int, default=1000, help="Monte Carlo draws for 95%% CI")
    p.add_argument("--n-boot", type=int, default=None, help=argparse.SUPPRESS)  # alias
    p.add_argument("--n-perm", type=int, default=1000, help="Permutations for p-values")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-gap-filled",
        action="store_true",
        help="Include gap-filled curves (default: exclude, as in manuscript)",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy flag (= default behaviour)
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: output_dir/growth_curves_eta2_variance.csv)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    include_gap = bool(args.include_gap_filled) and not args.local_only
    n_mc = args.n_mc if args.n_boot is None else args.n_boot
    print(
        f"Computing η² (equal weight, include_gap_filled={include_gap}, "
        f"n_mc={n_mc}, n_perm={args.n_perm})…"
    )
    table = compute_eta2_table(
        cfg,
        include_gap_filled=include_gap,
        n_mc=n_mc,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    out = args.out
    if out is None:
        out = output_dir(cfg) / "growth_curves_eta2_variance.csv"
    else:
        out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}")

    print("\n=== η² summary (% of variance; ± = half 95% MC CI) ===")
    for _, r in table.iterrows():
        print(
            f"  {r['metric_label']:<7}  species {r['eta2_species_pct']:.1f}±{r['eta2_species_pm']:.1f}% "
            f"(p={r['p_species']:.3f})  |  region {r['eta2_region_pct']:.1f}±{r['eta2_region_pm']:.1f}% "
            f"(p={r['p_region']:.3f})  n={r['n_pairs']}"
        )
    sp = table["eta2_species_pct"]
    rg = table["eta2_region_pct"]
    import numpy as np

    sp_pm = 1.96 * float(sp.std(ddof=1)) / np.sqrt(len(sp))
    rg_pm = 1.96 * float(rg.std(ddof=1)) / np.sqrt(len(rg))
    print(
        f"  Mean     species {sp.mean():.1f}±{sp_pm:.1f}%  |  region {rg.mean():.1f}±{rg_pm:.1f}%"
    )

    print("\n=== Results text ===")
    print(format_eta2_paragraph(table))


if __name__ == "__main__":
    main()
