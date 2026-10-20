#!/usr/bin/env python3
"""Set age==0 to nodata where AGB==0 unless disturbance year is in the keep window.

Default keep window: 2019–2021 (config ``disturbance_keep_years``). Ensures
``latest_disturbance_year_le2021_30m.tif`` exists before fixing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import load_config
from lib.postprocess.fix_nodata import fix_nodata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = fix_nodata(cfg, force=args.force or bool(cfg.get("force")))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
