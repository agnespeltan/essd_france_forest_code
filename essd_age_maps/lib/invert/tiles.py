"""Per-GRECO age tiles: CR invert only where that GRECO's weight > 0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from lib.config import (
    age_tile_path,
    growth_curves_path,
    raster_path,
    tiles_dir,
    weights_dir,
)
from lib.constants import AGE_NODATA, WEIGHT_NODATA
from lib.invert.chapman_richards import chapman_richards_vectorized
from lib.invert.polygons import load_shapefile
from lib.naming import weight_filename


def build_param_index(
    growth_df: pd.DataFrame,
    quantile: int,
    target_greco_id: str,
    use_forest_aggregates: bool,
) -> dict[Any, tuple[float, float, float]]:
    """Map TFV (or ``'forest'``) → ``(a, b, c)`` for one GRECO."""
    a_col, b_col, c_col = f"a_{quantile}", f"b_{quantile}", f"c_{quantile}"
    sub = growth_df[growth_df["greco"].astype(str) == str(target_greco_id)]
    index: dict[Any, tuple[float, float, float]] = {}
    for _, row in sub.iterrows():
        tfv = row["TFV"]
        key: Any = "forest" if str(tfv) == "forest" else int(tfv)
        index[key] = (float(row[a_col]), float(row[b_col]), float(row[c_col]))
    if use_forest_aggregates and "forest" not in index:
        raise KeyError(f"No forest CR params for GRECO {target_greco_id}")
    return index


def _build_tfv_lut(
    param_index: dict[Any, tuple[float, float, float]],
    use_forest_aggregates: bool,
    max_code: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer TFV → (a, b, c) arrays; ``valid`` marks codes with params."""
    a = np.full(max_code, np.nan, dtype=np.float64)
    b = np.full(max_code, np.nan, dtype=np.float64)
    c = np.full(max_code, np.nan, dtype=np.float64)
    if use_forest_aggregates:
        params = param_index.get("forest")
        if params is None:
            raise KeyError("forest params missing")
        aa, bb, cc = params
        for code in range(max_code):
            a[code], b[code], c[code] = aa, bb, cc
        return a, b, c, np.isfinite(a)

    for code in range(max_code):
        params = param_index.get(code)
        if params is None:
            fallback = 100 if code < 50 or code in (310, 111) else 200
            params = param_index.get(fallback)
        if params is not None:
            a[code], b[code], c[code] = params
    return a, b, c, np.isfinite(a)


def _weight_positive(weight: np.ndarray, weight_nodata) -> np.ndarray:
    """Boolean mask: soft-edge weight contributes to the blend."""
    m = np.isfinite(weight) & (weight > 0)
    if weight_nodata is not None:
        m &= weight != weight_nodata
    m &= weight != WEIGHT_NODATA
    return m


def _bbox_of_positive_weights(
    weight_src: rasterio.DatasetReader,
    block_size: int,
) -> tuple[int, int, int, int] | None:
    """Return (row0, col0, row1, col1) exclusive end, or None if empty."""
    height, width = weight_src.height, weight_src.width
    nodata = weight_src.nodata
    r0, c0, r1, c1 = height, width, 0, 0
    found = False
    for row in range(0, height, block_size):
        h = min(block_size, height - row)
        for col in range(0, width, block_size):
            w = min(block_size, width - col)
            win = Window(col, row, w, h)
            data = weight_src.read(1, window=win)
            if not np.any(_weight_positive(data, nodata)):
                continue
            found = True
            r0 = min(r0, row)
            c0 = min(c0, col)
            r1 = max(r1, row + h)
            c1 = max(c1, col + w)
    if not found:
        return None
    return r0, c0, r1, c1


def _init_nodata_file(
    out_path: Path, profile: dict, height: int, width: int, block: int = 4096
) -> None:
    """Create output GeoTIFF filled with AGE_NODATA."""
    with rasterio.open(out_path, "w", **profile) as dst:
        for row0 in range(0, height, block):
            h = min(block, height - row0)
            for col0 in range(0, width, block):
                w = min(block, width - col0)
                fill = np.full((h, w), AGE_NODATA, dtype=np.int32)
                dst.write(fill, 1, window=Window(col0, row0, w, h))


def _invert_block(
    agb: np.ndarray,
    tfv: np.ndarray,
    weight: np.ndarray,
    weight_nodata,
    agb_nodata,
    tfv_nodata,
    lut_a: np.ndarray,
    lut_b: np.ndarray,
    lut_c: np.ndarray,
) -> np.ndarray:
    """CR invert for one block; nodata where weight ≤ 0 or invalid inputs."""
    age = np.full(agb.shape, AGE_NODATA, dtype=np.int32)
    wmask = _weight_positive(weight, weight_nodata)

    agb_f = agb.astype(np.float64, copy=False)
    if agb_nodata is not None:
        agb_ok = np.isfinite(agb_f) & (agb_f != agb_nodata)
    else:
        agb_ok = np.isfinite(agb_f)

    tfv_i = tfv.astype(np.int32, copy=False)
    if tfv_nodata is not None:
        tfv_ok = tfv_i != tfv_nodata
    else:
        tfv_ok = np.ones(tfv_i.shape, dtype=bool)
    tfv_i = tfv_i.copy()
    tfv_i[tfv_i == 0] = 100
    tfv_i[tfv_i == 400] = 401
    tfv_ok &= (tfv_i > 0) & (tfv_i < 500) & (tfv_i < len(lut_a))

    valid = wmask & agb_ok & tfv_ok
    if not np.any(valid):
        return age

    flat_idx = np.flatnonzero(valid)
    codes = tfv_i.ravel()[flat_idx]
    params_ok = np.isfinite(lut_a[codes])
    if not np.any(params_ok):
        return age
    flat_idx = flat_idx[params_ok]
    codes = codes[params_ok]

    for code in np.unique(codes):
        sel = codes == code
        pix = flat_idx[sel]
        aa, bb, cc = float(lut_a[code]), float(lut_b[code]), float(lut_c[code])
        ages = chapman_richards_vectorized(agb_f.ravel()[pix], aa, bb, cc)
        age.ravel()[pix] = ages
    return age


def list_greco_ids(cfg: dict[str, Any]) -> list[str]:
    """GRECO ids from shapefile (sorted)."""
    agb_path = raster_path(cfg, "agb")
    shapefile_path = Path(cfg["greco_shapefile"]).expanduser()
    with rasterio.open(agb_path) as src:
        polygons = load_shapefile(shapefile_path, src.crs)
    return [str(x) for x in sorted(polygons["square_id"].unique())]


def build_age_tile(
    cfg: dict[str, Any],
    target_greco_id: str,
    *,
    force: bool = False,
    max_blocks: int | None = None,
    max_polygons: int | None = None,  # alias kept for old CLI
) -> Path:
    """Build age tile for one GRECO: only pixels with weight > 0."""
    if max_blocks is None and max_polygons is not None:
        max_blocks = max_polygons

    agb_path = raster_path(cfg, "agb")
    species_path = raster_path(cfg, "tfv")
    weight_path = weights_dir(cfg) / weight_filename(str(target_greco_id))
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Missing weight raster for GRECO {target_greco_id}: {weight_path}\n"
            "Run 00_build_greco_weights.py first (or set outputs.weights_dir)."
        )

    growth_csv = growth_curves_path(cfg)
    quantile = int(cfg.get("quantile", 50))
    use_forest = bool(cfg.get("use_forest_aggregates", False))
    block_size = int(cfg.get("block_size", 2048))
    out_folder = tiles_dir(cfg)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = age_tile_path(cfg, target_greco_id)

    if out_path.exists() and not force:
        print(f"Skip existing: {out_path}")
        return out_path

    growth_df = pd.read_csv(growth_csv)
    if not use_forest and "forest" in growth_df.TFV.astype(str).values:
        growth_df = growth_df[growth_df.TFV.astype(str) != "forest"].copy()
        growth_df["TFV"] = growth_df["TFV"].astype(int)
    param_index = build_param_index(growth_df, quantile, target_greco_id, use_forest)
    lut_a, lut_b, lut_c, _ = _build_tfv_lut(param_index, use_forest)

    with rasterio.open(agb_path) as agb_src:
        height, width = agb_src.shape
        profile = agb_src.profile.copy()
        profile.update(dtype="int32", nodata=AGE_NODATA, compress="deflate", predictor=2)
        agb_nodata = agb_src.nodata

    print(
        f"Processing GRECO {target_greco_id} → {out_path.name} "
        f"(weight-masked, block_size={block_size})"
    )
    print(f"  weights: {weight_path}")

    with rasterio.open(weight_path) as weight_src:
        if weight_src.height != height or weight_src.width != width:
            raise ValueError(
                f"Weight raster shape {weight_src.shape} != AGB {(height, width)}"
            )
        bbox = _bbox_of_positive_weights(weight_src, block_size)
        weight_nodata = weight_src.nodata

    if bbox is None:
        print("  No positive weights — writing empty nodata tile")
        _init_nodata_file(out_path, profile, height, width, block=block_size)
        return out_path

    r0, c0, r1, c1 = bbox
    print(f"  weight bbox rows[{r0}:{r1}] cols[{c0}:{c1}]")

    blocks: list[Window] = []
    for row in range(r0, r1, block_size):
        h = min(block_size, r1 - row)
        for col in range(c0, c1, block_size):
            w = min(block_size, c1 - col)
            blocks.append(Window(col, row, w, h))
    if max_blocks is not None:
        blocks = blocks[: max(0, int(max_blocks))]
        print(f"  smoke: first {len(blocks)} blocks only (skip full-France nodata fill)")
        # Sparse create — only written blocks matter for smoke
        with rasterio.open(out_path, "w", **profile) as dst:
            for win in blocks:
                fill = np.full((int(win.height), int(win.width)), AGE_NODATA, dtype=np.int32)
                dst.write(fill, 1, window=win)
    else:
        # Full France nodata base so blend never sees garbage outside the mask
        _init_nodata_file(out_path, profile, height, width, block=max(block_size, 4096))

    valid_pixels = 0
    with rasterio.open(agb_path) as agb_src, rasterio.open(
        species_path
    ) as tfv_src, rasterio.open(weight_path) as weight_src, rasterio.open(
        out_path, "r+"
    ) as dst:
        tfv_nodata = tfv_src.nodata
        for win in tqdm(blocks, desc=f"GRECO {target_greco_id}", unit="block"):
            weight = weight_src.read(1, window=win)
            if not np.any(_weight_positive(weight, weight_nodata)):
                continue
            agb = agb_src.read(1, window=win)
            tfv = tfv_src.read(1, window=win)
            age = _invert_block(
                agb,
                tfv,
                weight,
                weight_nodata,
                agb_nodata,
                tfv_nodata,
                lut_a,
                lut_b,
                lut_c,
            )
            n = int(np.sum(age != AGE_NODATA))
            if n == 0:
                continue
            dst.write(age, 1, window=win)
            valid_pixels += n

    print(f"Saved {out_path} ({valid_pixels:,} age pixels where weight>0)")
    return out_path


def build_all_age_tiles(
    cfg: dict[str, Any],
    *,
    force: bool = False,
    max_blocks: int | None = None,
    max_polygons: int | None = None,
    greco: str | None = None,
) -> list[Path]:
    """Build one or all GRECO age tiles (weight-masked)."""
    ids = [greco] if greco else list_greco_ids(cfg)
    return [
        build_age_tile(
            cfg,
            gid,
            force=force,
            max_blocks=max_blocks,
            max_polygons=max_polygons,
        )
        for gid in ids
    ]
