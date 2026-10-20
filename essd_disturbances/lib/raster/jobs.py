"""Parallel SER polygon job runner."""

from __future__ import annotations

import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def run_polygon_jobs(
    *,
    worker: Callable[..., pd.DataFrame],
    jobs: list[dict[str, Any]],
    workers: int,
    output_path: Path,
    desc: str,
) -> Path:
    """Run per-polygon workers and write one parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and replace only on success (avoid wiping prior output on crash).
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    frames: list[pd.DataFrame] = []
    total = len(jobs)
    print(f"{desc}: {total} polygons, workers={workers}", flush=True)

    if total == 0:
        pd.DataFrame().to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)
        return output_path

    if workers <= 1 or total == 1:
        for i, job in enumerate(jobs, start=1):
            df = worker(**job)
            if df is not None and not df.empty:
                frames.append(df)
            if i % 10 == 0 or i == total:
                print(f"{desc}: {i}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, **job): idx for idx, job in enumerate(jobs)}
            done = 0
            for future in as_completed(futures):
                df = future.result()
                if df is not None and not df.empty:
                    frames.append(df)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"{desc}: {done}/{total}", flush=True)

    if not frames:
        result = pd.DataFrame()
    else:
        result = pd.concat(frames, ignore_index=True)
        if "TFV" in result.columns:
            result["TFV"] = result["TFV"].astype(str)

    result.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"{desc}: wrote {len(result)} rows → {output_path}", flush=True)
    del frames
    gc.collect()
    return output_path
