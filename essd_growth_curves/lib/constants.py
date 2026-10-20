"""Shared scientific constants for the growth-curve pipeline."""

# TFV labels never fitted or written to published curve tables.
DEFAULT_EXCLUDED_TFV = frozenset({"0", "400", "forest", "504", "506"})

# Quantiles fitted for each TFV × GRECO (as fractions).
FIT_QUANTILES = (0.25, 0.50, 0.75)

# ---------------------------------------------------------------------------
# Gap-fill geography
# ---------------------------------------------------------------------------
# Local fits are kept only if min(R2_25, R2_50, R2_75) > r2_min (config).
# Missing TFV×GRECO with area > 0 are then filled in this order:
#
#   1. GROUP   — same GRECO_GROUPS entry; donor = largest-area kept fit
#   2. CLIMATE — GRECO_CLIMATE_DONOR[target letter] if that donor has a kept fit
#   3. else leave missing
#
GRECO_GROUPS = {
    # Oceanic
    "A": "Oceanic",
    "B": "Oceanic",
    "C": "Oceanic",
    "F": "Oceanic",
    # Mountainous
    "D": "Mountainous",
    "E": "Mountainous",
    "G": "Mountainous",
    "H": "Mountainous",
    "I": "Mountainous",
    # Mediterranean
    "J": "Mediterranean",
    "K": "Mediterranean",
}

# Rule 2 donors (target GRECO letter → climate-neighbor letter).
GRECO_CLIMATE_DONOR = {
    "J": "F",  # Mediterranean → Sud-Ouest océanique
    "K": "F",
    "B": "G",  # Oceanic leftovers (e.g. spruce) → Massif Central
    "F": "G",
}

# Map fit target names (config) → column prefix in the stats parquets.
TARGET_COLUMN_PREFIX = {
    "agb": "biomasseM",
}
