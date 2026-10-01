"""Trajectory and TrajectorySet: the ND data model REACT consumes.

A Trajectory is a (T, D) array of positions sampled at a fixed interval ``dt``
(or with explicit ``times``). D is arbitrary: 1 for a scalar series, 2 for floor
coordinates, 3 for 3D pose, etc. Everything downstream (references, forecasters,
the metric) is written against this interface, so the library is domain-agnostic.

Input interoperability follows the Argoverse 2 tabular convention: a tidy long
DataFrame keyed by ``track_id`` with a time column and one column per coordinate.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..constants import DEFAULT_DT


class Trajectory:
    """A single agent's path: positions (T, D) at frame interval ``dt``."""

    __slots__ = ("positions", "dt", "id", "times")

    def __init__(
        self,
        positions: np.ndarray,
        dt: float = DEFAULT_DT,
        id: Optional[object] = None,
        times: Optional[np.ndarray] = None,
    ):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions[:, None]
        if positions.ndim != 2:
            raise ValueError(f"positions must be (T, D); got shape {positions.shape}")
        self.positions = positions
        self.id = id
        if times is not None:
            times = np.asarray(times, dtype=float)
            if len(times) != len(positions):
                raise ValueError("times must have one entry per frame")
            self.times = times
            if len(times) > 1:
                dt = float(np.median(np.diff(times)))
        else:
            self.times = None
        self.dt = float(dt)

    @property
    def T(self) -> int:
        return self.positions.shape[0]

    @property
    def D(self) -> int:
        return self.positions.shape[1]

    def __len__(self) -> int:
        return self.T

    def velocity(self, window: int = 1) -> np.ndarray:
        """Causal finite-difference velocity over ``window`` frames, (T, D).

        velocity[t] = (x[t] - x[t - window]) / (window * dt) for t >= window;
        earlier frames repeat the first valid estimate.
        """
        w = int(window)
        v = np.zeros_like(self.positions)
        if self.T > w:
            v[w:] = (self.positions[w:] - self.positions[:-w]) / (w * self.dt)
            v[:w] = v[w]
        return v

    @classmethod
    def from_dataframe(
        cls,
        df,
        coords: Sequence[str] = ("x", "y"),
        time: Optional[str] = None,
        dt: float = DEFAULT_DT,
        id: Optional[object] = None,
    ) -> Trajectory:
        """Build one trajectory from a (already single-track) DataFrame."""
        frame = df
        if time is not None:
            frame = frame.sort_values(time)
        positions = frame[list(coords)].to_numpy(dtype=float)
        times = frame[time].to_numpy(dtype=float) if time is not None else None
        return cls(positions, dt=dt, id=id, times=times)

    def __repr__(self) -> str:
        return f"Trajectory(id={self.id!r}, T={self.T}, D={self.D}, dt={self.dt})"


class TrajectorySet:
    """An ordered collection of trajectories keyed by id."""

    def __init__(self, tracks: Dict[object, Trajectory]):
        self._tracks: Dict[object, Trajectory] = dict(tracks)

    @property
    def ids(self) -> List[object]:
        return list(self._tracks.keys())

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, key: object) -> Trajectory:
        return self._tracks[key]

    def __contains__(self, key: object) -> bool:
        return key in self._tracks

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._tracks.values())

    def items(self):
        return self._tracks.items()

    def as_arrays(self) -> Dict[object, np.ndarray]:
        """Mapping id -> (T, D) positions (the form internal helpers expect)."""
        return {k: v.positions for k, v in self._tracks.items()}

    @property
    def dt(self) -> float:
        for t in self._tracks.values():
            return t.dt
        return DEFAULT_DT

    @property
    def D(self) -> int:
        for t in self._tracks.values():
            return t.D
        return 0

    def split(
        self, fraction: float = 0.2, seed: int = 0
    ) -> Tuple[TrajectorySet, TrajectorySet]:
        """Seeded split into (calibration, evaluation) by track.

        The first ``fraction`` of a shuffled id list becomes the calibration
        set (for fitting references / steady-state thresholds); the rest is the
        evaluation set. Splitting by track keeps events within a track together.
        """
        rng = np.random.default_rng(seed)
        ids = np.array(self.ids, dtype=object)
        rng.shuffle(ids)
        n_cal = max(1, int(round(len(ids) * fraction)))
        cal_ids = set(ids[:n_cal].tolist())
        calib = {k: v for k, v in self._tracks.items() if k in cal_ids}
        evalset = {k: v for k, v in self._tracks.items() if k not in cal_ids}
        return TrajectorySet(calib), TrajectorySet(evalset)

    @classmethod
    def from_dataframe(
        cls,
        df,
        id: str = "track_id",
        coords: Sequence[str] = ("x", "y"),
        time: Optional[str] = None,
        dt: float = DEFAULT_DT,
    ) -> TrajectorySet:
        """Build a TrajectorySet from a tidy long DataFrame (Argoverse 2 style).

        Each row is one (track, time) observation. ``id`` groups rows into
        tracks; ``coords`` are the position columns; ``time`` (optional) orders
        rows and sets dt.
        """
        tracks: Dict[object, Trajectory] = {}
        for key, sub in df.groupby(id):
            tracks[key] = Trajectory.from_dataframe(
                sub, coords=coords, time=time, dt=dt, id=key
            )
        return cls(tracks)

    @classmethod
    def from_arrays(
        cls, arrays: Dict[object, np.ndarray], dt: float = DEFAULT_DT
    ) -> TrajectorySet:
        return cls({k: Trajectory(v, dt=dt, id=k) for k, v in arrays.items()})

    def __repr__(self) -> str:
        return f"TrajectorySet(n={len(self)}, D={self.D}, dt={self.dt})"
