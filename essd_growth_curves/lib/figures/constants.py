"""Shared labels / colours for growth-curve paper figures."""

from __future__ import annotations

from lib.constants import GRECO_GROUPS  # re-export (gap-fill / figures share groups)

RS_MAIN_Q = "50"
RS_ENV_LO_Q = "25"
RS_ENV_HI_Q = "75"

# Curve panels (a–f) historically used 0.47; metric panels use 0.475.
RS_CARBON_CURVES = 0.47
RS_CARBON_METRICS = 0.475
NFI_SOIL_FRAC = 0.221
NFI_CARBON = 0.475

GRECO_LETTER_COLORS = {
    "A": "darkturquoise",
    "B": "darkturquoise",
    "C": "darkturquoise",
    "F": "darkturquoise",
    "D": "peru",
    "E": "peru",
    "G": "peru",
    "H": "peru",
    "I": "peru",
    "J": "mediumorchid",
    "K": "mediumorchid",
}

GRECO_GROUP_COLORS = {
    "Oceanic": "darkturquoise",
    "Mountainous": "peru",
    "Mediterranean": "mediumorchid",
}

TFV_GROUP = {
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
    61.0: "Pure conifers",
    63.0: "Pure conifers",
    64.0: "Pure conifers",
    81.0: "Pure conifers",
    80.0: "Mixed conifers",
    90.0: "Mixed conifers",
    91.0: "Pure conifers",
    200.0: "Mixed conifers",
    222.0: "Mixed conifers",
    310.0: "Mixed deciduous",
    320.0: "Mixed conifers",
    401.0: "Mixed deciduous",
    402.0: "Mixed conifers",
    403.0: "Mixed deciduous",
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

GRECO_GROUP_ORDER = ["Oceanic", "Mountainous", "Mediterranean"]

DICT_GROUP_EN = {
    0.0: "Clear cut",
    1.0: "Deciduous oak",
    6.0: "Evergreen oak",
    9.0: "Beech",
    10.0: "Chestnut",
    14.0: "Locust",
    19.0: "Poplar",
    49.0: "Other deciduous",
    100.0: "Mixed deciduous",
    111.0: "Mixed deciduous in island",
    51.0: "Maritime pine",
    52.0: "Scotch pine",
    53.0: "Austrian pine",
    57.0: "Aleppo pine",
    58.0: "Mountain & stone pines",
    81.0: "Other pines",
    80.0: "Mixed pines",
    61.0: "Fir & spruce",
    63.0: "Larch",
    64.0: "Douglas",
    91.0: "Other non pine",
    90.0: "Mixed non pine",
    200.0: "Mixed conifers",
    222.0: "Mixed conifers in island",
    310.0: "Mainly deciduous",
    320.0: "Mainly conifers",
    400.0: "Open clear cut",
    401.0: "Open deciduous",
    402.0: "Open conifers",
    403.0: "Open mixed",
}

VALIDATION_SPECIES = [
    (1, "Deciduous oak"),
    (61, "Fir & spruce"),
]

REGIONAL_PANELS = [
    ("Oceanic", "Oceanic"),
    ("Mountainous", "Mountainous"),
    ("Mediterranean", "Mediterranean"),
]

GRECO_LETTERS = list("ABCDEFGHIJK")

FS_T = 12
FS_O = 11
BUBBLE_ALPHA = 0.55
