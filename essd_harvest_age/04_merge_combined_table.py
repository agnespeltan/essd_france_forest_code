#!/usr/bin/env python3
"""Merge trends+harvest age with size-frequency MLE and severity into Data_clean CSVs.

Keeps rate ``gap_filled`` and ``harvest_age_gap_filled``; renames MLE/severity
``gap_filled`` → ``mle_gap_filled`` / ``severity_gap_filled``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.harvest_age.merge import merge_combined_tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument(
        "--mle-csv",
        type=Path,
        default=None,
        help="Override patch_size_selected_mle.csv path",
    )
    p.add_argument(
        "--severity-csv",
        type=Path,
        default=None,
        help="Override disturbance_severity.csv path",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mle_csv is not None:
        cfg["patch_size_mle_csv"] = str(args.mle_csv.expanduser())
    if args.severity_csv is not None:
        cfg["severity_csv"] = str(args.severity_csv.expanduser())

    print("=== 04 merge combined table ===")
    merge_combined_tables(cfg)


if __name__ == "__main__":
    main()
