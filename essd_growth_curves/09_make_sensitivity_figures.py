#!/usr/bin/env python3
"""SI figures of the growth-curve sensitivity tests (inputs from 08).

Writes to ``figure_dir``:
  Figure_SI_baseline_factor_sensitivity.png   AGBmax, t50, R30 and RMSE vs NFI against f;
  Figure_SI_fit_settings_sensitivity.png      one row per swept setting (f, trim, R², b bound);
and ``fit_settings_sensitivity_table.csv`` (every run of every sweep) in the sensitivity folder.

Examples::

  python3 09_make_sensitivity_figures.py
  python3 09_make_sensitivity_figures.py --params tail_trim_age r2_min age_baseline_factor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from lib.config import load_config  # noqa: E402
from lib.sensitivity import BASELINE_SWEEP, sensitivity_cfg, sensitivity_dir, sweep_names  # noqa: E402
from lib.sensitivity.figures import PARAMS, fig_baseline_factor, fig_fit_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--figure-dir", type=Path, default=None, help="Default: figure_dir in the config")
    p.add_argument("--params", nargs="+", default=None, choices=list(PARAMS),
                   help="Rows of the fit-settings figure, in order (default: every swept setting)")
    args = p.parse_args()

    cfg = load_config(args.config)
    sens = sensitivity_cfg(cfg)
    root = sensitivity_dir(cfg)
    fig_dir = args.figure_dir or cfg.get("figure_dir")
    if not fig_dir:
        p.error("figure_dir missing in config (or pass --figure-dir)")
    fig_dir = Path(fig_dir).expanduser()

    dirs = [root / n for n in sweep_names(cfg)]
    dirs = [d for d in dirs if (d / "baseline_factor_sensitivity_table.csv").is_file()]
    if not dirs:
        raise SystemExit(f"no scored sweeps in {root}: run 08_sensitivity_tests.py first")

    if (root / BASELINE_SWEEP) in dirs:
        ref_f = float(sens.get("reference_factor", cfg.get("age_baseline_factor", 0.85)))
        fig_baseline_factor(root / BASELINE_SWEEP, fig_dir / "Figure_SI_baseline_factor_sensitivity.png", ref_f)

    fig_fit_settings(
        dirs,
        root / "fit_settings_sensitivity_table.csv",
        fig_dir / "Figure_SI_fit_settings_sensitivity.png",
        args.params or sens.get("figure_params"),
    )


if __name__ == "__main__":
    main()
