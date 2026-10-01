"""Eindhoven Centraal pedestrian trajectories adapter.

Default real-world reference dataset for reactmetric.

Source:
    Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., & Toschi, F.
    "Data-driven physics-based modeling of pedestrian dynamics - dataset:
    Pedestrian trajectories at Eindhoven train station". Zenodo.
    https://zenodo.org/records/13784588, DOI: 10.5281/zenodo.13784588.

License:
    Creative Commons Attribution 4.0 International (CC-BY-4.0).

The full Zenodo record is about 4.6 GB and partitioned into 10-day parquet files,
so the wheel ships only a tiny schema-compatible sample for tutorials/tests. Use
``download_eindhoven`` or ``load_eindhoven(download=True)`` to fetch full files
into a local cache.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import urllib.request
from importlib import resources
from typing import Dict, Optional, Sequence

import numpy as np

from ..core.trajectory import Trajectory, TrajectorySet
from .mapping import mappings

ZENODO_RECORD_ID = 13784588
ZENODO_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_HTML_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
DOI = "10.5281/zenodo.13784588"
LICENSE = "CC-BY-4.0"

_DAY_FILES: Dict[str, str] = {
    "01_10": "Eindhoven_centraal_trajectories_days_01_10.parquet",
    "11_20": "Eindhoven_centraal_trajectories_days_11_20.parquet",
    "21_30": "Eindhoven_centraal_trajectories_days_21_30.parquet",
    "31_40": "Eindhoven_centraal_trajectories_days_31_40.parquet",
    "41_50": "Eindhoven_centraal_trajectories_days_41_50.parquet",
    "51_60": "Eindhoven_centraal_trajectories_days_51_60.parquet",
}


def citation() -> str:
    """Human-readable attribution string for docs and generated reports."""
    return (
        "Pouw, C. A. S., van der Vleuten, G. G. M., Corbetta, A., & Toschi, F. "
        f"({DOI}). Pedestrian trajectories at Eindhoven Centraal. Zenodo. "
        f"License: {LICENSE}."
    )


def _cache_dir(cache_dir: Optional[str] = None) -> str:
    root = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "reactmetric")
    path = os.path.join(root, "eindhoven")
    os.makedirs(path, exist_ok=True)
    return path


def _record_metadata() -> Dict:
    with urllib.request.urlopen(ZENODO_RECORD_URL, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def eindhoven_file_manifest() -> Dict[str, Dict[str, object]]:
    """Return Zenodo file metadata keyed by filename."""
    record = _record_metadata()
    return {
        item["key"]: {
            "size": item["size"],
            "checksum": item.get("checksum"),
            "url": item["links"]["self"],
        }
        for item in record["files"]
    }


def download_eindhoven(
    days: str = "01_10",
    cache_dir: Optional[str] = None,
    force: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download one 10-day Eindhoven parquet shard from Zenodo into a cache.

    This is intentionally explicit because each shard is hundreds of MB and the
    full dataset is about 4.6 GB.
    """
    if days not in _DAY_FILES:
        raise ValueError(f"days must be one of {tuple(_DAY_FILES)}; got {days!r}")
    filename = _DAY_FILES[days]
    cache = _cache_dir(cache_dir)
    path = os.path.join(cache, filename)
    if os.path.exists(path) and not force:
        return path

    manifest = eindhoven_file_manifest()
    if filename not in manifest:
        raise RuntimeError(f"{filename} not found in Zenodo record {ZENODO_RECORD_ID}")
    url = str(manifest[filename]["url"])
    tmp = f"{path}.part"
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=chunk_size)
    os.replace(tmp, path)
    return path


def _eindhoven_mapping():
    return mappings.eindhoven


def _from_rows(rows: Sequence[Dict[str, float]], min_length: int, dt: float) -> TrajectorySet:
    mapping = _eindhoven_mapping()
    id_col = mapping.id
    time_col = mapping.time
    x_col, y_col = mapping.coords[:2]
    pos_scale = mapping.position_scale
    time_scale = mapping.time_scale

    grouped: Dict[int, list] = {}
    for row in rows:
        tid = int(row[id_col])
        grouped.setdefault(tid, []).append(row)

    tracks = {}
    for tid, group in grouped.items():
        group = sorted(group, key=lambda r: float(r[time_col]))
        if len(group) < min_length:
            continue
        positions = np.array(
            [
                [float(r[x_col]) * pos_scale, float(r[y_col]) * pos_scale]
                for r in group
            ],
            dtype=float,
        )
        times = np.array([float(r[time_col]) * time_scale for r in group], dtype=float)
        tracks[tid] = Trajectory(positions, dt=dt, id=tid, times=times)
    if not tracks:
        raise ValueError(f"no Eindhoven tracks >= {min_length} frames")
    return TrajectorySet(tracks)


def _load_limited_parquet(
    path: str,
    max_tracks: int,
    min_length: int,
    dt: float,
    batch_size: int = 100_000,
) -> TrajectorySet:
    """Stream a large Eindhoven parquet shard while keeping only max_tracks IDs."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Memory-bounded Eindhoven loading with max_tracks requires pyarrow. "
            "Install it with: pip install 'reactmetric[datasets]'"
        ) from e

    mapping = _eindhoven_mapping()
    id_col = mapping.id
    time_col = mapping.time
    x_col, y_col = mapping.coords[:2]
    cols = [time_col, id_col, x_col, y_col]
    parquet = pq.ParquetFile(path)
    missing = set(cols) - set(parquet.schema.names)
    if missing:
        raise ValueError(f"Eindhoven parquet missing columns: {sorted(missing)}")

    selected = set()
    rows = []
    empty_after_selection = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=cols):
        table = batch.to_pydict()
        ids = table[id_col]
        for tid in ids:
            if len(selected) >= max_tracks:
                break
            selected.add(int(tid))
        if not selected:
            continue
        times = table[time_col]
        xs = table[x_col]
        ys = table[y_col]
        kept = 0
        for i, tid in enumerate(ids):
            tid = int(tid)
            if tid in selected:
                kept += 1
                rows.append(
                    {
                        time_col: times[i],
                        id_col: tid,
                        x_col: xs[i],
                        y_col: ys[i],
                    }
                )
        if len(selected) >= max_tracks:
            if kept == 0:
                empty_after_selection += 1
            else:
                empty_after_selection = 0
            if empty_after_selection >= 1:
                break
    return _from_rows(rows, min_length=min_length, dt=dt)


def load_eindhoven_sample(
    min_length: Optional[int] = None,
    dt: Optional[float] = None,
) -> TrajectorySet:
    """Load the tiny bundled Eindhoven-format sample used by quickstart/tests.

    Column/unit assumptions come from ``mappings.eindhoven``. Use
    ``load_eindhoven(download=True)`` for the full Zenodo data.
    """
    mapping = _eindhoven_mapping()
    min_length = mapping.min_length if min_length is None else min_length
    dt = mapping.dt if dt is None else dt
    sample = resources.files("reactmetric.datasets.data").joinpath("eindhoven_sample.csv")
    with sample.open(newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    return _from_rows(rows, min_length=min_length, dt=dt)


def load_eindhoven(
    path: Optional[str] = None,
    days: str = "01_10",
    cache_dir: Optional[str] = None,
    download: bool = False,
    min_length: Optional[int] = None,
    max_tracks: Optional[int] = None,
    dt: Optional[float] = None,
) -> TrajectorySet:
    """Load Eindhoven Centraal trajectories from parquet, cache, or bundled sample.

    Mapping (columns, mm/ms scales, default dt) is ``mappings.eindhoven``.

    Args:
        path: Direct path to a Zenodo parquet shard or CSV in Eindhoven schema.
        days: Which 10-day shard to download/load when ``path`` is not provided.
        cache_dir: Optional cache root; defaults to ``~/.cache/reactmetric``.
        download: If True, fetch the requested shard from Zenodo when absent.
        min_length: Drop trajectories shorter than this many frames.
        max_tracks: Optional cap after loading, useful for quick demos.
        dt: Expected nominal frame interval (10 Hz = 0.1s).

    Returns:
        TrajectorySet with positions in meters and ids from the mapping id column.
    """
    mapping = _eindhoven_mapping()
    min_length = mapping.min_length if min_length is None else min_length
    dt = mapping.dt if dt is None else dt

    if path is None:
        cached = os.path.join(_cache_dir(cache_dir), _DAY_FILES[days])
        if os.path.exists(cached):
            path = cached
        elif download:
            path = download_eindhoven(days=days, cache_dir=cache_dir)
        else:
            return load_eindhoven_sample(min_length=min_length, dt=dt)

    if path.endswith(".csv"):
        # Prefer the unified mapping-driven loader for ordinary CSV.
        from .load import load as load_mapped

        data = load_mapped(path, mapping, max_tracks=max_tracks, dt=dt, min_length=min_length)
        return data
    elif max_tracks is not None:
        data = _load_limited_parquet(
            path=path,
            max_tracks=max_tracks,
            min_length=min_length,
            dt=dt,
        )
    else:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Full Eindhoven parquet loading requires pandas + pyarrow: "
                "pip install 'reactmetric[datasets]'"
            ) from e
        cols = [mapping.time, mapping.id, *mapping.coords]
        try:
            df = pd.read_parquet(path, columns=cols)
        except ImportError as e:
            raise ImportError(
                "Full Eindhoven parquet loading requires a parquet engine. "
                "Install the dataset extra with: pip install 'reactmetric[datasets]' "
                "or install pyarrow directly in this environment."
            ) from e
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Eindhoven parquet missing columns: {sorted(missing)}")
        rows = df[cols].to_dict("records")
        data = _from_rows(rows, min_length=min_length, dt=dt)

    if max_tracks is not None and len(data) > max_tracks:
        data = TrajectorySet({tid: data[tid] for tid in data.ids[:max_tracks]})
    return data
