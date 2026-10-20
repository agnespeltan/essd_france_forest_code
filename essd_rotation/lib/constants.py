"""Shared constants for the NFI rotation-period pipeline."""

from __future__ import annotations

# Package data/ lives next to lib/
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA = PACKAGE_ROOT / "data"

DEFAULT_MIN_COUNT = 5
DEFAULT_AGE_OFFSET = 3
DEFAULT_MAX_AGE13 = 500

# Fill level on joined harvest rows
GAP_FILLED_SER = 0        # matched TFV×SER (direct NFI)
GAP_FILLED_GRECO = 1      # same TFV × GRECO
GAP_FILLED_NATIONAL = 2   # same TFV national
GAP_FILLED_ALIAS_SER = 3  # alias target filled at SER from donor species

# Sparse TFV ← denser source (order matters). TFV 200 is NFI-built (80∪90), not an alias.
DEFAULT_TFV_ALIASES: list[tuple[int, int]] = [
    (401, 100),  # open deciduous ← mixed deciduous
    (402, 200),  # open conifers ← mixed conifers
    (111, 100),
    (222, 200),
    (310, 100),
    (320, 200),
    (403, 401),  # after 401←100
    (58, 200),   # mountain/stone pines (sparse NFI)
]

TFV_GROUP_ORDER = [
    "Pure deciduous",
    "Pure conifers",
    "Mixed deciduous",
    "Mixed conifers",
]

TFV_GROUP_COLORS = {
    "Pure deciduous": "forestgreen",
    "Pure conifers": "mediumvioletred",
    "Mixed deciduous": "#1565C0",
    "Mixed conifers": "#E65100",
}

DICT_GROUP_TFVS = {
    0.0: "Other",
    1.0: "Pure deciduous",
    6.0: "Pure deciduous",
    9.0: "Pure deciduous",
    10.0: "Pure deciduous",
    14.0: "Pure deciduous",
    19.0: "Pure deciduous",
    49.0: "Pure deciduous",
    100.0: "Mixed deciduous",
    111.0: "Mixed deciduous",
    51.0: "Pure conifers",
    52.0: "Pure conifers",
    53.0: "Pure conifers",
    57.0: "Pure conifers",
    58.0: "Pure conifers",
    81.0: "Pure conifers",
    80.0: "Mixed conifers",
    61.0: "Pure conifers",
    63.0: "Pure conifers",
    64.0: "Pure conifers",
    91.0: "Pure conifers",
    90.0: "Mixed conifers",
    200.0: "Mixed conifers",
    222.0: "Mixed conifers",
    310.0: "Mixed deciduous",
    320.0: "Mixed conifers",
    400.0: "Other",
    401.0: "Mixed deciduous",
    402.0: "Mixed conifers",
    403.0: "Mixed deciduous",
}

DICT_GROUP_EN = {
    1.0: "Deciduous oak",
    6.0: "Evergreen oak",
    9.0: "Beech",
    10.0: "Chestnut",
    14.0: "Locust",
    19.0: "Poplar",
    49.0: "Other deciduous",
    100.0: "Mixed deciduous",
    111.0: "Mixed deciduous\nin island",
    51.0: "Maritime pine",
    52.0: "Scotch pine",
    53.0: "Austrian pine",
    57.0: "Aleppo pine",
    58.0: "Mountain\n& stone pines",
    81.0: "Other pines",
    80.0: "Mixed pines",
    61.0: "Fir & spruce",
    63.0: "Larch",
    64.0: "Douglas",
    91.0: "Other non pine",
    90.0: "Mixed non pine",
    200.0: "Mixed conifers",
    222.0: "Mixed conifers\nin island",
    310.0: "Mainly deciduous",
    320.0: "Mainly conifers",
    401.0: "Open deciduous",
    402.0: "Open conifers",
    403.0: "Open mixed",
}
