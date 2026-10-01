"""Shared kinematic helpers and a light noise-calibration object for baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ols_velocity(points: np.ndarray, dt: float) -> np.ndarray:
    """Per-axis OLS slope (velocity) of k>=2 sequential positions."""
    k = len(points)
    i = np.arange(k, dtype=float)
    ic = i - i.mean()
    denom = float(np.sum(ic**2))
    if denom <= 0:
        return np.zeros(points.shape[1])
    return (ic[:, None] * (points - points.mean(axis=0))).sum(axis=0) / (denom * dt)


def slope_variance_factor(k: int) -> float:
    """1 / sum_i (i - ibar)^2 for k equally spaced points (units: per dt^2)."""
    i = np.arange(k, dtype=float)
    ic = i - i.mean()
    denom = float(np.sum(ic**2))
    return 1.0 / denom if denom > 0 else np.inf


@dataclass
class CalibratedNoise:
    """Lightweight observation-noise / speed summary estimated from data."""

    measurement_noise: float = 0.05  # meters (per-axis position std)
    mean_speed: float = 1.2  # m/s

    @classmethod
    def estimate(cls, data, window: int = 5) -> CalibratedNoise:
        """Estimate measurement noise (high-frequency jitter) and mean speed.

        Measurement noise is taken from the second-difference of positions
        (acceleration residual), which is dominated by observation noise at high
        frequency. Mean speed is the median finite-difference speed.
        """
        arrays = data.as_arrays() if hasattr(data, "as_arrays") else dict(data)
        dt = data.dt if hasattr(data, "dt") else 0.1
        accel_res = []
        speeds = []
        for obs in arrays.values():
            if len(obs) < window + 2:
                continue
            second = obs[2:] - 2 * obs[1:-1] + obs[:-2]
            accel_res.append(second[np.isfinite(second).all(axis=1)])
            v = (obs[window:] - obs[:-window]) / (window * dt)
            sp = np.hypot(v[:, 0], v[:, 1]) if obs.shape[1] >= 2 else np.abs(v[:, 0])
            speeds.append(sp[np.isfinite(sp)])
        if accel_res:
            res = np.concatenate(accel_res)
            # second difference of white noise has variance 6 * sigma^2 per axis
            sigma = float(np.sqrt(max(np.mean(res**2) / 6.0, 1e-6)))
        else:
            sigma = 0.05
        mean_speed = float(np.median(np.concatenate(speeds))) if speeds else 1.2
        return cls(measurement_noise=sigma, mean_speed=mean_speed)


def stack_tracks(data, max_tracks: int = 0, seed: int = 0, segment: int = 0):
    """Pad a TrajectorySet (or id -> array mapping) for batched filtering.

    Returns (obs, lengths, dt): obs is (N, T_max, D) with each track's last
    position repeated past its end, and lengths holds each track's T. Tracks with
    non-finite positions or fewer than 4 frames are dropped. With ``segment`` > 0,
    tracks longer than that are cut into consecutive pieces of at most ``segment``
    frames (each piece restarts the filter), which keeps padding small.
    """
    arrays = data.as_arrays() if hasattr(data, "as_arrays") else dict(data)
    dt = float(data.dt) if hasattr(data, "dt") else 0.1
    tracks = [np.asarray(a, float) for a in arrays.values()]
    tracks = [a for a in tracks if len(a) >= 4 and np.isfinite(a).all()]
    if max_tracks and len(tracks) > max_tracks:
        pick = np.random.default_rng(seed).choice(len(tracks), max_tracks, replace=False)
        tracks = [tracks[i] for i in sorted(pick)]
    if segment:
        tracks = [a[i : i + segment] for a in tracks for i in range(0, len(a), segment)]
        tracks = [a for a in tracks if len(a) >= 4]
    if not tracks:
        raise ValueError("no usable tracks to fit on")
    lengths = np.array([len(a) for a in tracks])
    obs = np.empty((len(tracks), lengths.max(), tracks[0].shape[1]))
    for i, a in enumerate(tracks):
        obs[i, : len(a)] = a
        obs[i, len(a) :] = a[-1]
    return obs, lengths, dt
