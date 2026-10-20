"""Shared constants for the disturbance trends / severity pipeline."""

from __future__ import annotations

DEFAULT_EXCLUDED_TFV = frozenset({"0", "400", "forest", "504", "506"})

# agent_class in patches parquet → publishable agent label
AGENT_MAP_NUMERIC = {1: "natural", 2: "natural", 3: "harvest"}

# agent_class in AGB stats parquet → publishable agent label
AGENT_MAP_STRING = {"unplanned": "natural", "harvest": "harvest", "natural": "natural"}

# Forest-type groups (TFV → broad class)
GROUPS_FT = {
    "0": "D",
    "1": "D",
    "6": "D",
    "9": "D",
    "10": "D",
    "14": "D",
    "19": "D",
    "49": "D",
    "100": "D",
    "111": "D",
    "51": "C",
    "52": "C",
    "53": "C",
    "57": "C",
    "58": "C",
    "81": "C",
    "80": "C",
    "61": "C",
    "63": "C",
    "64": "C",
    "91": "C",
    "90": "C",
    "200": "C",
    "222": "C",
    "310": "D",
    "320": "C",
    "400": "D",
    "401": "D",
    "402": "C",
    "403": "D",
    "Forest": "Forest",
}

# Biogeographic groups from GRECO letter (C kept as Semi-continental only)
GROUPS_BG = {
    "A": "Oceanic",
    "B": "Oceanic",
    "C": "Semi-continental",
    "F": "Oceanic",
    "D": "Mountainous",
    "E": "Mountainous",
    "G": "Mountainous",
    "H": "Mountainous",
    "I": "Mountainous",
    "J": "Mediterranean",
    "K": "Mediterranean",
}

PIXEL_AREA_MHA = 0.09 / 1e6  # 30 m pixel → Mha

GAP_FILLED_SER = 0
GAP_FILLED_GRECO = 1
GAP_FILLED_NATIONAL = 2

METRICS = (
    "disturbance_rate",
    "patch_count",
    "mean_patch_size",
    "std_patch_size",
)
