"""Shared constants for the age-map pipeline."""

from __future__ import annotations

AGE_NODATA = -1
WEIGHT_NODATA = -9999
PIXEL_AREA_MHA = 0.09 / 1e6  # 30 m pixel → Mha

# Default keep window for age==0 (override with config disturbance_keep_years).
# Requires latest_disturbance_year_le2021_30m.tif (years ≤ 2021).
EXCLUDED_DIST_YEARS = frozenset({2019, 2020, 2021})

# GRECO letter → large ecological region (paper Fig. 3 panel c)
GRECO_GROUPS = {
    "A": "Oceanic",
    "B": "Oceanic",
    "C": "Oceanic",
    "F": "Oceanic",
    "D": "Mountainous",
    "E": "Mountainous",
    "G": "Mountainous",
    "H": "Mountainous",
    "I": "Mountainous",
    "J": "Mediterranean",
    "K": "Mediterranean",
}

TFV_GROUP = {
    1: "Pure deciduous",
    6: "Pure deciduous",
    9: "Pure deciduous",
    10: "Pure deciduous",
    14: "Pure deciduous",
    19: "Pure deciduous",
    49: "Pure deciduous",
    100: "Mixed deciduous",
    111: "Mixed deciduous",
    51: "Pure conifers",
    52: "Pure conifers",
    53: "Pure conifers",
    57: "Pure conifers",
    58: "Pure conifers",
    61: "Pure conifers",
    63: "Pure conifers",
    64: "Pure conifers",
    81: "Pure conifers",
    80: "Mixed conifers",
    90: "Mixed conifers",
    91: "Pure conifers",
    200: "Mixed conifers",
    222: "Mixed conifers",
    310: "Mixed deciduous",
    320: "Mixed conifers",
    401: "Mixed deciduous",
    402: "Mixed conifers",
    403: "Mixed deciduous",
}

TFV_GROUP_COLORS = {
    "Pure deciduous": "forestgreen",
    "Pure conifers": "mediumvioletred",
    "Mixed deciduous": "skyblue",
    "Mixed conifers": "orange",
}

TFV_GROUP_ORDER = [
    "Pure deciduous",
    "Mixed deciduous",
    "Pure conifers",
    "Mixed conifers",
]

GRECO_GROUP_COLORS = {
    "Oceanic": "darkturquoise",
    "Mountainous": "peru",
    "Mediterranean": "mediumorchid",
}

GRECO_GROUP_ORDER = ["Oceanic", "Mountainous", "Mediterranean"]
