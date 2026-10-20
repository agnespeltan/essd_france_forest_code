"""Colours, labels, and species-match tables for age-map figures."""

from __future__ import annotations

from lib.constants import (
    GRECO_GROUP_COLORS,
    GRECO_GROUP_ORDER,
    GRECO_GROUPS,
    TFV_GROUP,
    TFV_GROUP_COLORS,
    TFV_GROUP_ORDER,
)

FS_T, FS_O = 12, 11
RS_FILL = "#4393c3"
RS_LINE = "#2166ac"
GAMI_FILL = "#f4a582"
GAMI_LINE = "#d6604d"
NFI_EDGE = "#4d9221"
BUBBLE_ALPHA = 0.55
GRECO_DX = {"Oceanic": -0.24, "Mountainous": 0.0, "Mediterranean": 0.24}

DICT_AGE = {
    "0-5 ans": 2.5,
    "5-10 ans": 7.5,
    "10-15 ans": 12.5,
    "15-20 ans": 17.5,
    "20-25 ans": 22.5,
    "25-30 ans": 27.5,
    "30-35 ans": 32.5,
    "35-40 ans": 37.5,
    "40-50 ans": 45,
    "50-60 ans": 55,
    "60-70 ans": 65,
    "70-80 ans": 75,
    "80-100 ans": 90,
    "100-120 ans": 110,
    "120-140 ans": 130,
    "140-160 ans": 150,
    "160-180 ans": 170,
    "180-200 ans": 190,
    "200-240 ans": 220,
    ">= 240 ans": 260,
    "0 - 10 ans ou non recensable": 5,
    "10 - 20 ans": 15,
    "20 - 30 ans": 25,
    "30 - 40 ans": 35,
    "40 - 50 ans": 45,
    "50 - 60 ans": 55,
    ">= 60 ans": 80,
    "Non défini": -1,
    "Absence d'arbres mesurés": -1,
    "Momentanément déboisé": -1,
}

TFV_LABEL = {
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
    401.0: "Open deciduous",
    402.0: "Open conifers",
    403.0: "Open mixed",
}

# NFI ESSPRE ↔ BD Forêt TFV match (SI4)
SPECIES_MATCH = [
    {
        "key": "oak",
        "label": "Deciduous oak",
        "ess": ["Chênes pédonculé, rouvre et pubescent"],
        "tfvs": [1],
    },
    {"key": "beech", "label": "Beech", "ess": ["Hêtre"], "tfvs": [9]},
    {"key": "chestnut", "label": "Chestnut", "ess": ["Châtaignier"], "tfvs": [10]},
    {"key": "mar_pine", "label": "Maritime pine", "ess": ["Pin maritime"], "tfvs": [51]},
    {"key": "scots_pine", "label": "Scotch pine", "ess": ["Pin sylvestre"], "tfvs": [52]},
    {"key": "douglas", "label": "Douglas", "ess": ["Douglas"], "tfvs": [64]},
    {
        "key": "fir_spruce",
        "label": "Fir & spruce",
        "ess": ["Sapin pectiné", "Epicéa commun"],
        "tfvs": [61],
    },
    {
        "key": "other_dec",
        "label": "Other deciduous",
        "ess": ["Autres feuillus", "Charme", "Frêne"],
        "tfvs": [100, 49, 6, 14, 19, 111, 310, 401, 403],
    },
    {
        "key": "other_con",
        "label": "Other conifers",
        "ess": ["Autres résineux"],
        "tfvs": [200, 81, 91, 53, 57, 58, 63, 80, 90, 222, 320, 402],
    },
]

ESS_TO_MATCH = {ess: m["key"] for m in SPECIES_MATCH for ess in m["ess"]}
MATCH_TO_TFVS = {m["key"]: m["tfvs"] for m in SPECIES_MATCH}
MATCH_LABEL = {m["key"]: m["label"] for m in SPECIES_MATCH}
MATCH_ORDER = [m["key"] for m in SPECIES_MATCH]

__all__ = [
    "FS_T",
    "FS_O",
    "RS_FILL",
    "RS_LINE",
    "GAMI_FILL",
    "GAMI_LINE",
    "NFI_EDGE",
    "BUBBLE_ALPHA",
    "GRECO_DX",
    "DICT_AGE",
    "TFV_LABEL",
    "SPECIES_MATCH",
    "ESS_TO_MATCH",
    "MATCH_TO_TFVS",
    "MATCH_LABEL",
    "MATCH_ORDER",
    "GRECO_GROUPS",
    "GRECO_GROUP_COLORS",
    "GRECO_GROUP_ORDER",
    "TFV_GROUP",
    "TFV_GROUP_COLORS",
    "TFV_GROUP_ORDER",
]
