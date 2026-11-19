#!/usr/bin/env python3
"""Sensitivity of the growth curves to the fit settings, scored against the NFI.

Runs every sweep of the ``sensitivity`` config block:

* ``baseline_grid``: young-stand baseline age a0 × factor f (folder ``baseline_factor``);
* ``settings``: one setting at a time per named sweep, e.g. ``tail_trim_age``,
  ``r2_min``, ``tail_trim_delta``, ``tail_trim_patience``, ``cr_b_min``.

For each run the curves are refit (other settings from the config) and scored against
the NFI: AGB trajectories (RMSE per GRECO and species) and R30, AGBmax, t50 (national
area-weighted values, Δ% vs the published setting, RMSE / bias). Outputs go to
``<output_dir>/<sensitivity.out_dir>/<sweep>/``; 09 draws the SI figures.

Needs the NFI tables and TFV areas: ``figures.nfi_greco_csv``, ``figures.nfi_species_csv``,
``figures.area_parquet``.

Examples::

  python3 08_sensitivity_tests.py                       # all sweeps (a few hours)
  python3 08_sensitivity_tests.py --only b_min          # one sweep
  python3 08_sensitivity_tests.py --score-only          # re-score existing curves, no refit
  python3 08_sensitivity_tests.py --trim-diagnostic     # + points removed by the tail trim
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# No GUI backend: the a0 × f heat map is drawn while fit threads run.
os.environ.setdefault("MPLBACKEND", "Agg")

from lib.config import load_config  # noqa: E402
from lib.fit.preprocess import load_merged_stats  # noqa: E402
from lib.sensitivity import (  # noqa: E402
    BASELINE_SWEEP,
    nfi_paths,
    sensitivity_cfg,
    sensitivity_dir,
    sweep_names,
)
from lib.sensitivity.metrics import score_sweep  # noqa: E402
from lib.sensitivity.nfi import load_area_weights, load_nfi_greco, load_nfi_species  # noqa: E402
from lib.sensitivity.sweep import run_baseline_grid, run_setting_sweep, trim_diagnostic  # noqa: E402

SUMMARIES = ("fit_settings_vs_nfi_summary.csv", "baseline_age_factor_vs_nfi_summary.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--only", nargs="+", default=None, metavar="SWEEP",
                   help="Sweeps to run (names from the config; 'baseline_factor' = a0 × f grid)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Refit sweeps that already have results")
    p.add_argument("--score-only", action="store_true",
                   help="No refit: re-score the curve CSVs already in each sweep folder")
    p.add_argument("--trim-diagnostic", action="store_true",
                   help="Also write tail_trim_diagnostic.csv (points removed by the default trim)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    sens = sensitivity_cfg(cfg)
    root = sensitivity_dir(cfg)
    names = sweep_names(cfg)
    if args.only:
        unknown = sorted(set(args.only) - set(names))
        if unknown:
            p.error(f"unknown sweep(s) {unknown}; config has {names}")
        names = [n for n in names if n in args.only]

    paths = nfi_paths(cfg)
    nfi_g = load_nfi_greco(paths["nfi_greco_csv"])
    nfi_s = load_nfi_species(paths["nfi_species_csv"])
    area = load_area_weights(paths["area_parquet"])
    ref_f = float(sens.get("reference_factor", cfg.get("age_baseline_factor", 0.85)))
    print(f"NFI: {len(nfi_g)} GRECO rows, {len(nfi_s)} species rows; area: {len(area)} rows")
    print(f"Sweeps: {names} → {root}")

    if args.trim_diagnostic:
        root.mkdir(parents=True, exist_ok=True)
        diag = trim_diagnostic(cfg, load_merged_stats(cfg))
        diag.to_csv(root / "tail_trim_diagnostic.csv", index=False)
        print(f"Wrote {root / 'tail_trim_diagnostic.csv'}")

    for name in names:
        out = root / name
        print(f"\n##### {name}")
        done = any((out / s).is_file() for s in SUMMARIES)
        if args.score_only:
            if not done:
                print(f"  no results in {out}: run without --score-only first")
                continue
        elif done and not args.force:
            print("  results exist (use --force to refit); scoring only")
        elif name == BASELINE_SWEEP:
            grid = sens["baseline_grid"]
            run_baseline_grid(
                cfg,
                [int(a) for a in grid.get("ages", [cfg.get("age_baseline_age", 2)])],
                [float(f) for f in grid["factors"]],
                [None if t is None else float(t) for t in (grid.get("tapers") or [None])],
                out, nfi_g, nfi_s, area,
            )
        else:
            run_setting_sweep(cfg, sens["settings"][name], out, nfi_g, nfi_s, area)
        score_sweep(out, nfi_g, nfi_s, area, reference_factor=ref_f)

    print("\nDone. Figures: python3 09_make_sensitivity_figures.py")


if __name__ == "__main__":
    main()
