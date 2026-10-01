"""Unified tabular dataset loading driven by DatasetMapping."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..core.trajectory import Trajectory, TrajectorySet
from .mapping import DatasetMapping, MappingLike, mappings


def _read_table(path: str, columns=None):
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Tabular dataset loading requires pandas: pip install 'reactmetric[datasets]'"
        ) from e

    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith((".parquet", ".pq")):
        try:
            return pd.read_parquet(path, columns=columns)
        except ImportError as e:
            raise ImportError(
                "Parquet loading requires pyarrow: pip install 'reactmetric[datasets]'"
            ) from e
        except TypeError:
            # some engines reject columns=; fall back then select
            df = pd.read_parquet(path)
            return df[columns] if columns is not None else df
    raise ValueError(f"unsupported tabular file type for {path!r} (use .csv or .parquet)")


def _coerce_time_seconds(series, time_scale: float):
    """Convert numeric or datetime timestamps to float seconds."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Tabular dataset loading requires pandas: pip install 'reactmetric[datasets]'"
        ) from e

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) * time_scale
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        raise ValueError(f"time column has {n_bad} unparseable timestamp values")
    # Relative seconds keep dt inference stable regardless of epoch offset.
    return (parsed - parsed.min()).dt.total_seconds().astype(float) * time_scale


def _apply_scales(df, mapping: DatasetMapping):
    frame = df.copy()
    for col in mapping.coords:
        frame[col] = frame[col].astype(float) * mapping.position_scale
    if mapping.time is not None and mapping.time in frame.columns:
        frame[mapping.time] = _coerce_time_seconds(frame[mapping.time], mapping.time_scale)
    return frame


def _from_scaled_dataframe(df, mapping: DatasetMapping) -> TrajectorySet:
    tracks = {}
    for key, sub in df.groupby(mapping.id):
        sub = sub.sort_values(mapping.time) if mapping.time else sub
        if len(sub) < mapping.min_length:
            continue
        positions = sub[list(mapping.coords)].to_numpy(dtype=float)
        times = (
            sub[mapping.time].to_numpy(dtype=float) if mapping.time is not None else None
        )
        tracks[key] = Trajectory(positions, dt=mapping.dt, id=key, times=times)
    if not tracks:
        raise ValueError(
            f"no tracks >= {mapping.min_length} frames after applying mapping {mapping.name!r}"
        )
    return TrajectorySet(tracks)


def _load_whitespace_ethucy(path: str, mapping: DatasetMapping) -> TrajectorySet:
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"expected whitespace 'frame ped x y' rows in {path}")
    frames, peds, xs = raw[:, 0], raw[:, 1], raw[:, 2]
    ys = raw[:, 4] if raw.shape[1] >= 8 else raw[:, 3]
    tracks = {}
    for ped in np.unique(peds):
        mask = peds == ped
        order = np.argsort(frames[mask])
        pos = np.column_stack([xs[mask][order], ys[mask][order]]) * mapping.position_scale
        if len(pos) >= mapping.min_length:
            tracks[int(ped)] = Trajectory(pos, dt=mapping.dt, id=int(ped))
    if not tracks:
        raise ValueError(f"no tracks >= {mapping.min_length} frames found in {path}")
    return TrajectorySet(tracks)


def load(
    path: str,
    mapping: MappingLike,
    *,
    max_tracks: Optional[int] = None,
    dt: Optional[float] = None,
    min_length: Optional[int] = None,
    use_3d: bool = False,
) -> TrajectorySet:
    """Load a local trajectory file using an explicit DatasetMapping.

    Args:
        path: Local CSV/parquet path, or ETH/UCY whitespace text file.
        mapping: Registry name (``\"eindhoven\"``, ``\"tracks\"``, ...), a
            :class:`DatasetMapping`, or a dict with the same fields.
        max_tracks: Optional cap on number of tracks retained.
        dt / min_length: Optional overrides of mapping fields for this load only.
        use_3d: For the ``tracks`` mapping, switch to the 3D coordinate triple
            stored in ``mapping.extras['coords_3d']``.
    """
    resolved = mappings.resolve(mapping)
    overrides = {}
    if dt is not None:
        overrides["dt"] = dt
    if min_length is not None:
        overrides["min_length"] = min_length
    if use_3d:
        coords_3d = resolved.extras.get("coords_3d")
        if not coords_3d:
            raise ValueError(f"mapping {resolved.name!r} has no extras['coords_3d']")
        overrides["coords"] = tuple(coords_3d)
    if overrides:
        resolved = resolved.with_updates(**overrides)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if resolved.format == "whitespace_ethucy":
        data = _load_whitespace_ethucy(path, resolved)
    else:
        cols = [resolved.id, *resolved.coords]
        if resolved.time is not None:
            cols = [resolved.id, resolved.time, *resolved.coords]
        # de-dupe while preserving order
        seen = set()
        ordered = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        df = _read_table(path, columns=ordered)
        missing = [c for c in ordered if c not in df.columns]
        if missing:
            raise ValueError(
                f"file {path} missing columns {missing} required by mapping {resolved.name!r}"
            )
        df = _apply_scales(df[ordered], resolved)
        data = _from_scaled_dataframe(df, resolved)

    if max_tracks is not None and len(data) > max_tracks:
        data = TrajectorySet({tid: data[tid] for tid in data.ids[:max_tracks]})
    return data


def load_dataset(
    name: str,
    *,
    path: Optional[str] = None,
    max_tracks: Optional[int] = None,
    dt: Optional[float] = None,
    **loader_kwargs,
) -> TrajectorySet:
    """Load a named default dataset using its registered mapping.

    Thin convenience over the specialized loaders (Eindhoven download/cache,
    Tracks local file, ETH/UCY scene resolution) while keeping mappings as the
    single place column/unit assumptions live.
    """
    if name in {"eindhoven", "eindhoven-sample"}:
        from . import eindhoven as eh

        if name == "eindhoven-sample":
            data = eh.load_eindhoven_sample(
                min_length=mappings.eindhoven.min_length,
                dt=dt or mappings.eindhoven.dt,
            )
            if max_tracks is not None and len(data) > max_tracks:
                data = TrajectorySet({tid: data[tid] for tid in data.ids[:max_tracks]})
            return data
        return eh.load_eindhoven(
            path=path,
            max_tracks=max_tracks,
            dt=dt or mappings.eindhoven.dt,
            min_length=mappings.eindhoven.min_length,
            **loader_kwargs,
        )
    if name == "tracks":
        from . import tracks as tr

        return tr.load_tracks(
            path=path,
            max_tracks=max_tracks,
            dt=dt,
            **loader_kwargs,
        )
    if name == "ethucy":
        from . import ethucy as eu

        if path is not None:
            source = path
        else:
            source = loader_kwargs.pop("scene", "eth")
        return eu.load_eth_ucy(
            source,
            root=loader_kwargs.pop("root", None),
            dt=dt,
            min_length=mappings.ethucy.min_length,
            **loader_kwargs,
        )
    if name == "synthetic":
        from .synthetic import synthetic_regime_changes

        data, _ = synthetic_regime_changes(
            n=loader_kwargs.pop("n", 300),
            seed=loader_kwargs.pop("seed", 0),
            dt=dt or 0.1,
        )
        if max_tracks is not None and len(data) > max_tracks:
            data = TrajectorySet({tid: data[tid] for tid in data.ids[:max_tracks]})
        return data
    raise ValueError(
        f"unknown dataset {name!r}; choose from eindhoven, eindhoven-sample, "
        "tracks, ethucy, synthetic — or call load(path, mapping=...)"
    )
