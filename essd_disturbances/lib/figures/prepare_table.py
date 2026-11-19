"""Join trends + severity into a table for ``lib.figures.fig_disturbances``.

No mean harvest age (NFI harvest-age) — those panels stay off when plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lib.config import output_path

_AGENT_MAP = {
    "1": "natural",
    "2": "natural",
    "3": "harvest",
    "natural": "natural",
    "unplanned": "natural",
    "harvest": "harvest",
}

_SEV_RENAME = {
    "agb_lost": "agbd_lost_conservative",
    "agb_lost_error": "agbd_lost_conservative_error",
    "relative_agb_lost": "relative_agbd_lost_conservative",
    "relative_agb_lost_error": "relative_agbd_lost_conservative_error",
}


def _norm_agent(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().map(_AGENT_MAP)


def build_figure_table(
    trends: pd.DataFrame,
    severity: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join severity onto trends; rename severity cols for ``run_figures``."""
    tr = trends.copy()
    sev = severity.copy()

    tr["TFV"] = tr["TFV"].astype(str)
    sev["TFV"] = sev["TFV"].astype(str)
    tr["ser"] = pd.to_numeric(tr["ser"], errors="coerce").astype("Int64")
    sev["ser"] = pd.to_numeric(sev["ser"], errors="coerce").astype("Int64")
    tr["agent"] = _norm_agent(tr["agent"])
    sev["agent"] = _norm_agent(sev["agent"])

    sev = sev.rename(columns=_SEV_RENAME)
    # Also expose non-conservative aliases used by clipping helpers
    if "relative_agbd_lost_conservative" in sev.columns:
        sev["relative_agbd_lost"] = sev["relative_agbd_lost_conservative"]
        sev["relative_agbd_lost_error"] = sev["relative_agbd_lost_conservative_error"]
    if "agbd_lost_conservative" in sev.columns:
        sev["agbd_lost"] = sev["agbd_lost_conservative"]

    keys = ["TFV", "ser", "agent"]
    sev_cols = keys + [
        c
        for c in (
            "agbd_lost_conservative",
            "agbd_lost_conservative_error",
            "relative_agbd_lost_conservative",
            "relative_agbd_lost_conservative_error",
            "relative_agbd_lost",
            "relative_agbd_lost_error",
            "agbd_lost",
            "gap_filled",
        )
        if c in sev.columns
    ]
    # Prefer severity gap_filled when both exist
    if "gap_filled" in tr.columns and "gap_filled" in sev.columns:
        tr = tr.drop(columns=["gap_filled"])

    out = tr.merge(sev[sev_cols], on=keys, how="left")
    out["agent"] = out["agent"].astype(str)
    return out


def write_figure_table(cfg: dict[str, Any], *, out_name: str | None = None) -> Path:
    """Load pipeline CSVs from config, write joined figure table under output_dir."""
    out_dir = Path(cfg["output_dir"]).expanduser()
    trends_path = cfg.get("trends_csv") or output_path(cfg, "trends")
    sev_path = cfg.get("severity_csv") or output_path(cfg, "severity")
    trends_path = Path(trends_path).expanduser()
    sev_path = Path(sev_path).expanduser()
    if not trends_path.is_file():
        raise FileNotFoundError(f"Trends CSV not found: {trends_path}")
    if not sev_path.is_file():
        raise FileNotFoundError(f"Severity CSV not found: {sev_path}")

    table = build_figure_table(pd.read_csv(trends_path), pd.read_csv(sev_path))
    name = out_name or cfg.get("outputs", {}).get(
        "figure_table", "disturbance_trends_for_figures.csv"
    )
    out = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out
