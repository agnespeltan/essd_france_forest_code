#!/usr/bin/env python3
"""Build the Zenodo deposit tables with species and region codes in every CSV.

Each code is written twice: our numeric id (``*_num``) and the real IGN code,
plus the English species name:

    TFV_num  TFV         TFV_name       GRECO_num  GRECO  SER_num  SER
    51       FF2-51-51   Maritime pine  6          F      621      F21

No values are computed here: the 2020 levels and slope types come from
essd_disturbances (02, lib/levels.py) and R30 from essd_growth_curves (03). This
script only adds the codes, orders the columns, zips and checks.

Reads the pipeline outputs in ``Data_clean`` (left untouched, the pipelines
keep reading numeric ``TFV`` / ``ser``) and rewrites only the deposit files:
the four product zips and the two lookup CSVs (and, with --age-map, age_map.zip).

    python essd_package/build_deposit.py [--data-dir /path/to/Data_clean]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
ESSD_DIST = REPO / "essd_disturbances"


# zip name -> (source CSV in data dir, CSV name inside the zip, has SER columns)
PRODUCTS = {
    "disturbance_rate_patches.zip": ("disturbance_trends.csv", "disturbance_trends.csv", True),
    "disturbance_severity.zip": ("disturbance_severity.csv", "disturbance_severity.csv", True),
    "mean_harvest_age.zip": ("mean_harvest_age.csv", "mean_harvest_age.csv", True),
    "growth_curves.zip": ("growth_curves_agb_greco.csv", "growth_curves_agb_greco.csv", False),
}

CODE_COLS_SER = ["TFV_num", "TFV", "TFV_name", "GRECO_num", "GRECO", "SER_num", "SER"]
CODE_COLS_GRECO = ["TFV_num", "TFV", "TFV_name", "GRECO_num", "GRECO"]


def greco_num(letter: pd.Series) -> pd.Series:
    return letter.astype(str).str.upper().map(lambda c: ord(c) - 64).astype(int)


def load_lookups(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tfv = pd.read_csv(data_dir / "tfv_codes.csv")
    if "TFV_num" not in tfv.columns:  # first run: original lookup layout
        tfv = tfv.rename(columns={"TFV": "TFV_num", "bd_foret_code_tfv": "TFV", "name": "TFV_name"})
    tfv["TFV_num"] = tfv["TFV_num"].astype(int)
    # Footnote markers belong to the paper table, not to the data.
    tfv["TFV_name"] = tfv["TFV_name"].astype(str).str.replace(r"[‡†]", "", regex=True).str.strip()

    ser = pd.read_csv(data_dir / "ser_codes.csv")
    if "SER_num" not in ser.columns:
        ser = ser.rename(columns={"ser": "SER_num", "codeser": "SER", "name": "SER_name"})
    ser["SER_num"] = ser["SER_num"].astype(int)
    ser["GRECO_num"] = greco_num(ser["GRECO"])

    tfv = tfv[["TFV_num", "TFV", "TFV_name"] + [c for c in tfv.columns if c not in ("TFV_num", "TFV", "TFV_name")]]
    ser = ser[["SER_num", "SER", "GRECO_num", "GRECO", "SER_name"]]
    assert tfv["TFV_num"].is_unique and ser["SER_num"].is_unique
    return tfv, ser


def add_codes(df: pd.DataFrame, tfv: pd.DataFrame, ser: pd.DataFrame, has_ser: bool) -> pd.DataFrame:
    n0 = len(df)
    out = df.rename(columns={"TFV": "TFV_num", "ser": "SER_num", "greco": "GRECO"})
    out["TFV_num"] = out["TFV_num"].astype(int)
    out = out.merge(tfv[["TFV_num", "TFV", "TFV_name"]], on="TFV_num", how="left", validate="m:1")
    if has_ser:
        out["SER_num"] = out["SER_num"].astype(int)
        out = out.drop(columns=["GRECO"], errors="ignore")
        out = out.merge(ser[["SER_num", "SER", "GRECO_num", "GRECO"]], on="SER_num", how="left", validate="m:1")
        lead = CODE_COLS_SER
    else:
        out["GRECO_num"] = greco_num(out["GRECO"])
        lead = CODE_COLS_GRECO
    for gap in ("harvest_age_gap_filled", "rotation_gap_filled"):
        if gap in out.columns:
            out = out.rename(columns={gap: "gap_filled"})
    if "agent" in out.columns:
        lead = lead + ["agent"]
    if "trend_disturbance_rate" in out.columns:
        level_cols = [f"slope_type_{m}" for m in SLOPE_METRICS]
        level_cols += [c for m in SLOPE_METRICS for c in (f"{m}_2020", f"{m}_2020_error")]
        _require(out, level_cols, "essd_disturbances/02_fit_trends.py")
        lead = lead + level_cols
    if "b_50" in out.columns:
        _require(out, [f"R30_{q}" for q in (25, 50, 75)], "essd_growth_curves/03_fit_curves.py")
    out = out[lead + [c for c in out.columns if c not in lead]]

    # Checks: nothing lost, every code resolved, nested regions consistent.
    assert len(out) == n0, "row count changed"
    missing = out[lead].isna().any()
    assert not missing.any(), f"unresolved codes: {list(missing[missing].index)}"
    if has_ser:
        assert not out.duplicated(["TFV_num", "SER_num", "agent"]).any(), "duplicate keys"
        assert (out["GRECO"] == out["SER"].str[0]).all(), "GRECO != first letter of SER"
        assert (out["GRECO_num"] == out["SER_num"] // 100).all(), "GRECO_num != SER_num // 100"
    else:
        assert not out.duplicated(["TFV_num", "GRECO"]).any(), "duplicate keys"
    return out


SLOPE_METRICS = ("disturbance_rate", "mean_patch_size")


def _require(df: pd.DataFrame, cols: list[str], producer: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing {missing}: rerun {producer}")


def _dist_config() -> dict:
    cfg_path = ESSD_DIST / "config.yaml"
    return (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.is_file() else {}


def write_zip(df: pd.DataFrame, zip_path: Path, inner_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / inner_name
        df.to_csv(csv, index=False)
        tmp_zip = Path(tmp) / zip_path.name
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv, arcname=inner_name)
        shutil.move(str(tmp_zip), zip_path)


# Age maps, same names in Data_clean and in the deposit (essd_age_maps/lib/naming.py).
# Inverting the 25th-percentile AGB curve gives OLDER ages (upper), the 75th YOUNGER (lower).
AGE_FILES = ("forest_age_central.tif", "forest_age_upper.tif", "forest_age_lower.tif")


def _crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            crc = zlib.crc32(block, crc)
    return crc


def write_age_zip(data_dir: Path, backup_dir: Path | None = None) -> None:
    """Rebuild age_map.zip (tifs stored, already compressed).

    Reports, for each tif, whether it differs from the member of the existing
    age_map.zip with the same name, then checks the new zip against the tifs.
    """
    zip_path = data_dir / "age_map.zip"
    crcs = {name: _crc32(data_dir / name) for name in AGE_FILES}
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            published = {i.filename: i.CRC for i in zf.infolist()}
        for name in AGE_FILES:
            ref = published.get(name)
            state = "new" if ref is None else ("unchanged" if crcs[name] == ref else "CHANGED")
            print(f"  {name}: {state} vs previous age_map.zip")
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        tmp_zip = Path(tmp) / zip_path.name
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for name in AGE_FILES:
                zf.write(data_dir / name, arcname=name)
        with zipfile.ZipFile(tmp_zip) as zf:
            assert {i.filename: i.CRC for i in zf.infolist()} == crcs, "zip members differ from the tifs"
        if zip_path.is_file() and backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(zip_path), backup_dir / "age_map.previous.zip")
        shutil.move(str(tmp_zip), zip_path)
    print(f"age_map.zip: {', '.join(AGE_FILES)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Data_clean folder (default: output_dir of essd_disturbances/config.yaml)")
    p.add_argument("--age-map", action="store_true",
                   help="Also rebuild age_map.zip (~900 MB) with the published file names")
    p.add_argument("--backup-dir", type=Path, default=None,
                   help="Where to move the previous age_map.zip (default: overwrite)")
    args = p.parse_args()
    if args.data_dir is None:
        out = _dist_config().get("output_dir")
        if not out:
            p.error("--data-dir is required (no output_dir in essd_disturbances/config.yaml)")
        args.data_dir = Path(out)
    data_dir = args.data_dir.expanduser()
    if args.age_map:
        write_age_zip(data_dir, args.backup_dir)

    tfv, ser = load_lookups(data_dir)
    for zip_name, (src, inner, has_ser) in PRODUCTS.items():
        # keep_default_na=False: the slope_type label "null" must stay a string
        raw = pd.read_csv(data_dir / src, float_precision="round_trip",
                          keep_default_na=False, na_values=[""])
        df = add_codes(raw, tfv, ser, has_ser)
        write_zip(df, data_dir / zip_name, inner)
        extra = f", per agent {df['agent'].value_counts().to_dict()}" if "agent" in df.columns else ""
        print(f"{zip_name}: {inner} {len(df)} rows × {df.shape[1]} cols{extra}")
        for m in SLOPE_METRICS:
            col = f"slope_type_{m}"
            if col in df.columns:
                print(f"  {col}:\n" + pd.crosstab(df[col], df["agent"]).to_string())
    old = data_dir / "rotation_periods.zip"  # renamed product (now mean_harvest_age.zip)
    if old.is_file():
        old.unlink()
        print("removed old rotation_periods.zip")

    tfv.to_csv(data_dir / "tfv_codes.csv", index=False)
    ser.to_csv(data_dir / "ser_codes.csv", index=False)
    print(f"tfv_codes.csv: {len(tfv)} rows; ser_codes.csv: {len(ser)} rows")


if __name__ == "__main__":
    main()
