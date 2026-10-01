"""Reference climatology: the information baseline REACT scores against.

`Climatology.fit(...)` is the user-facing factory; it dispatches to the global,
kinematic, or position implementation and returns a Reference. The kinematic and
position modes use 2D spatial structure; for other dimensions they fall back to
the global marginal climatology automatically.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Optional, Sequence

from ..constants import DEFAULT_DT
from .base import Reference
from .climatology import GlobalClimatology, fit_global_climatology
from .kinematic import KinematicClimatology, fit_kinematic_climatology
from .position import PositionClimatology, fit_position_climatology


class Climatology:
    """Factory + (de)serialization for the REACT reference."""

    @staticmethod
    def fit(
        data,
        leads: Sequence[int],
        mode: str = "kinematic",
        k: int = 4,
        dt: Optional[float] = None,
        seed: int = 0,
        calib_ids: Optional[Sequence[object]] = None,
        **mode_kwargs,
    ) -> Reference:
        """Fit a reference on a calibration TrajectorySet (or id->array mapping).

        Args:
            data: TrajectorySet or dict[id -> (T, D) array].
            leads: forecast leads to fit references for.
            mode: 'kinematic' | 'global' | 'position'.
            k: GMM components for the global/position mixtures.
            dt: frame interval; inferred from a TrajectorySet if omitted.
            mode_kwargs: forwarded to the mode-specific fitter.
        """
        if hasattr(data, "as_arrays"):
            arrays = data.as_arrays()
            dt = dt if dt is not None else data.dt
        else:
            arrays = dict(data)
            dt = dt if dt is not None else DEFAULT_DT
        ids = list(calib_ids) if calib_ids is not None else list(arrays.keys())
        dim = next(iter(arrays.values())).shape[1]

        if mode in ("kinematic", "position") and dim != 2:
            warnings.warn(
                f"reference mode '{mode}' requires 2D positions (got D={dim}); "
                f"falling back to 'global'.",
                stacklevel=2,
            )
            mode = "global"

        if mode == "global":
            return fit_global_climatology(arrays, ids, leads, dt=dt, k=k, seed=seed)
        if mode == "kinematic":
            return fit_kinematic_climatology(arrays, ids, leads, dt=dt, seed=seed, **mode_kwargs)
        if mode == "position":
            return fit_position_climatology(
                arrays, ids, leads, dt=dt, k=k, seed=seed, **mode_kwargs
            )
        raise ValueError(f"unknown reference mode '{mode}'")


def save(reference: Reference, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(reference, f)


def load(path: str) -> Reference:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "Reference",
    "Climatology",
    "GlobalClimatology",
    "KinematicClimatology",
    "PositionClimatology",
    "fit_global_climatology",
    "fit_kinematic_climatology",
    "fit_position_climatology",
    "save",
    "load",
]
