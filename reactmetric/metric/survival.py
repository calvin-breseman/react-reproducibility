"""Survival statistics for REACT recovery times.

Recovery time is right-censored (a track can end, or the next regime change can
arrive, before the model recovers), so the natural summary is a Kaplan-Meier
product-limit estimator. Confidence intervals resample TRACKS (clusters), since
events within a track are not independent.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def kaplan_meier(
    durations: np.ndarray, observed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Product-limit estimator of the recovery-time survival function.

    Censored events (no recovery before the window ended) contribute risk time
    without a recovery. The atom at zero (anticipated events) appears as S(0) < 1.

    Returns:
        (times, survival): survival[i] = S(times[i]), a step function.
    """
    if len(durations) == 0:
        return np.array([0.0]), np.array([1.0])
    order = np.argsort(durations)
    durations = np.asarray(durations, dtype=float)[order]
    observed = np.asarray(observed, dtype=bool)[order]

    times = [0.0]
    surv = [1.0]
    s = 1.0
    n_at_risk = len(durations)
    i = 0
    while i < len(durations):
        t = durations[i]
        d = 0
        c = 0
        while i < len(durations) and durations[i] == t:
            if observed[i]:
                d += 1
            else:
                c += 1
            i += 1
        if d > 0 and n_at_risk > 0:
            s *= 1.0 - d / n_at_risk
            times.append(t)
            surv.append(s)
        n_at_risk -= d + c
    return np.array(times), np.array(surv)


def km_median(times: np.ndarray, surv: np.ndarray) -> float:
    """Median survival time: first time where S drops to or below 0.5."""
    below = np.where(surv <= 0.5)[0]
    return float(times[below[0]]) if len(below) else float("nan")


def cluster_bootstrap_ci(
    measurements: List,
    stat_fn: Callable[[List], float],
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap CI for a statistic of the measurements, resampling TRACKS."""
    by_track: Dict[object, List] = {}
    for m in measurements:
        by_track.setdefault(m.id, []).append(m)
    track_ids = list(by_track.keys())
    if len(track_ids) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(track_ids))
    stats = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(track_ids), replace=True)
        sample = [m for j in sample_idx for m in by_track[track_ids[j]]]
        try:
            v = stat_fn(sample)
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(stats, 100 * alpha)),
        float(np.percentile(stats, 100 * (1 - alpha))),
    )
