"""Build figure inputs from fixed age maps: ``*_hist.csv``, ``*_5km.tif``, ``*_cv_5km.tif``."""

from __future__ import annotations

import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from lib.config import fixed_age_path, features_root, output_dir, raster_path, resolve_under
from lib.constants import AGE_NODATA
from lib.naming import age_label

_AGE_MULT = 1
_SER_MULT = 10_000
_TFV_MULT = 1_000_000_000


def _pack_tfv_ser_age(tfv, ser, age):
    return (
        tfv.astype(np.int64) * _TFV_MULT
        + ser.astype(np.int64) * _SER_MULT
        + age.astype(np.int64) * _AGE_MULT
    )


def _unpack_tfv_ser_age(packed_keys):
    packed_keys = np.asarray(packed_keys, dtype=np.int64)
    return (
        packed_keys // _TFV_MULT,
        (packed_keys % _TFV_MULT) // _SER_MULT,
        (packed_keys % _SER_MULT) // _AGE_MULT,
    )


def _windows(width: int, height: int, tile: int):
    for row in range(0, height, tile):
        for col in range(0, width, tile):
            yield Window(
                col, row, min(tile, width - col), min(tile, height - row)
            )


def _count_batch(args):
    age_path, tfv_path, ser_path, age_nodata, specs = args
    local = Counter()
    with (
        rasterio.open(age_path) as age_src,
        rasterio.open(tfv_path) as tfv_src,
        rasterio.open(ser_path) as ser_src,
    ):
        for col_off, row_off, w, h in specs:
            win = Window(col_off, row_off, w, h)
            age = age_src.read(1, window=win)
            tfv = tfv_src.read(1, window=win)
            ser = ser_src.read(1, window=win)
            valid = np.ones(age.shape, dtype=bool)
            if age_nodata is not None:
                valid &= age != age_nodata
            valid &= age != AGE_NODATA
            valid &= np.isfinite(age) & np.isfinite(tfv) & np.isfinite(ser)
            if not np.any(valid):
                continue
            packed = _pack_tfv_ser_age(tfv[valid], ser[valid], age[valid])
            keys, counts = np.unique(packed, return_counts=True)
            local.update({int(k): int(c) for k, c in zip(keys, counts)})
    return local


def build_age_hist(
    age_tif: Path,
    tfv_tif: Path,
    ser_tif: Path,
    out_csv: Path,
    *,
    tile_size: int = 4096,
    n_jobs: int = 4,
) -> Path:
    """Write ``tfv,ser,age,count`` histogram CSV for one age map."""
    with rasterio.open(age_tif) as src:
        width, height = src.width, src.height
        age_nodata = src.nodata

    specs = [
        (int(w.col_off), int(w.row_off), int(w.width), int(w.height))
        for w in _windows(width, height, tile_size)
    ]
    batch = max(1, (len(specs) + n_jobs - 1) // n_jobs)
    batches = [specs[i : i + batch] for i in range(0, len(specs), batch)]
    args_list = [(str(age_tif), str(tfv_tif), str(ser_tif), age_nodata, b) for b in batches]

    counts: Counter = Counter()
    if n_jobs <= 1 or len(batches) == 1:
        for a in tqdm(args_list, desc=f"hist {age_tif.name}"):
            counts.update(_count_batch(a))
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(batches))) as pool:
            futs = [pool.submit(_count_batch, a) for a in args_list]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"hist {age_tif.name}"):
                counts.update(fut.result())

    keys = np.fromiter(counts.keys(), dtype=np.int64)
    vals = np.fromiter(counts.values(), dtype=np.int64)
    tfv, ser, age = _unpack_tfv_ser_age(keys)
    df = pd.DataFrame({"tfv": tfv, "ser": ser, "age": age, "count": vals})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows → {out_csv}")
    return out_csv


def build_age_5km(age_tif: Path, out_tif: Path) -> Path:
    """Median aggregate 30 m age → 5 km GeoTIFF (gdalwarp)."""
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(age_tif) as src:
        src_nodata = src.nodata if src.nodata is not None else AGE_NODATA
    cmd = [
        "gdalwarp",
        "-tr",
        "5000",
        "5000",
        "-r",
        "med",
        "-srcnodata",
        str(src_nodata),
        "-dstnodata",
        str(src_nodata),
        "-overwrite",
        str(age_tif),
        str(out_tif),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Wrote 5 km → {out_tif}")
    return out_tif


def build_age_cv_5km(age_tif: Path, ref_5km: Path, out_tif: Path) -> Path:
    """Spatial age CV at 5 km: σ/μ of valid 30 m ages (age > 0) inside each 5 km cell."""
    import warnings
    from rasterio.windows import from_bounds

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_5km) as ref:
        out_h, out_w = ref.shape
        dst_transform = ref.transform
        dst_profile = ref.profile.copy()

    out = np.full((out_h, out_w), np.nan, dtype=np.float32)
    with rasterio.open(age_tif) as src:
        nodata = src.nodata
        for i in range(out_h):
            if i % 50 == 0:
                print(f"  spatial CV 5 km: row {i}/{out_h}", flush=True)
            top = dst_transform.f + i * dst_transform.e
            bottom = top + dst_transform.e
            left = dst_transform.c
            right = dst_transform.c + out_w * dst_transform.a
            win = from_bounds(left, bottom, right, top, transform=src.transform)
            row = src.read(1, window=win, boundless=True, fill_value=nodata).astype(np.float32)
            if nodata is not None:
                row[row == nodata] = np.nan
            row[row <= 0] = np.nan

            x_edges = dst_transform.c + np.arange(out_w + 1) * dst_transform.a
            col_edges = np.clip(
                np.round((x_edges - src.transform.c) / src.transform.a - win.col_off).astype(int),
                0,
                row.shape[1],
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for j in range(out_w):
                    c0, c1 = col_edges[j], col_edges[j + 1]
                    if c1 <= c0:
                        continue
                    valid = row[:, c0:c1]
                    valid = valid[np.isfinite(valid)]
                    if valid.size < 2:
                        continue
                    mu = float(np.mean(valid))
                    if mu <= 0:
                        continue
                    out[i, j] = float(np.std(valid, ddof=0)) / mu

    dst_profile.update(dtype="float32", nodata=-1.0, count=1, compress="deflate")
    out_write = np.where(np.isfinite(out), out, -1.0).astype(np.float32)
    with rasterio.open(out_tif, "w", **dst_profile) as dst:
        dst.write(out_write, 1)
    print(f"Wrote CV 5 km → {out_tif}")
    return out_tif


def ser_raster_path(cfg: dict[str, Any]) -> Path:
    rasters = cfg.get("rasters") or {}
    if rasters.get("ser"):
        p = Path(rasters["ser"]).expanduser()
        return p.resolve() if p.is_absolute() else resolve_under(features_root(cfg), p)
    return resolve_under(features_root(cfg), "Human_30m/France_ser_30m.tif")


def build_summaries_for_quantile(
    cfg: dict[str, Any],
    quantile: int,
    *,
    force: bool = False,
    n_jobs: int = 4,
) -> tuple[Path, Path, Path]:
    """Rebuild ``forest_age_<label>_hist.csv``, ``_5km.tif`` and ``_cv_5km.tif`` (label: central/upper/lower)."""
    cfg = dict(cfg)
    cfg["quantile"] = int(quantile)
    age_tif = fixed_age_path(cfg)
    if not age_tif.is_file():
        raise FileNotFoundError(f"Missing fixed age map: {age_tif}")

    out_dir = output_dir(cfg)
    lab = age_label(quantile)
    hist_csv = out_dir / f"forest_age_{lab}_hist.csv"
    age_5km = out_dir / f"forest_age_{lab}_5km.tif"
    age_cv = out_dir / f"forest_age_{lab}_cv_5km.tif"
    tfv = raster_path(cfg, "tfv")
    ser = ser_raster_path(cfg)

    if force or not hist_csv.is_file() or hist_csv.stat().st_mtime < age_tif.stat().st_mtime:
        build_age_hist(age_tif, tfv, ser, hist_csv, n_jobs=n_jobs)
    else:
        print(f"Skip hist (up to date): {hist_csv}")

    if force or not age_5km.is_file() or age_5km.stat().st_mtime < age_tif.stat().st_mtime:
        build_age_5km(age_tif, age_5km)
    else:
        print(f"Skip 5 km (up to date): {age_5km}")

    if force or not age_cv.is_file() or age_cv.stat().st_mtime < age_tif.stat().st_mtime:
        build_age_cv_5km(age_tif, age_5km, age_cv)
    else:
        print(f"Skip CV 5 km (up to date): {age_cv}")

    return hist_csv, age_5km, age_cv


def build_all_summaries(
    cfg: dict[str, Any],
    quantiles: tuple[int, ...] = (25, 50, 75),
    *,
    force: bool = False,
    n_jobs: int = 4,
) -> list[tuple[Path, Path, Path]]:
    return [
        build_summaries_for_quantile(cfg, q, force=force, n_jobs=n_jobs) for q in quantiles
    ]
