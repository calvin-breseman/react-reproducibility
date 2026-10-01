"""Built-in kinematic changepoint detector (the default EventSource).

Detects "commitment events" -- regime changes in a person's motion -- by running
a PELT changepoint search on standardized kinematic features (speed and turn
rate), then classifies each changepoint into the taxonomy (stop/start,
speed_change, turn_onset/exit/change, sharp_turn, both) from pre/post effect
sizes. Each event also carries an information-theoretic ``detection_floor_frames``
-- the irreducible quickest-detection delay an optimal online detector would
incur for that change -- which feeds Oracle C.

This is a clean re-implementation (no dependency on the internal pipeline). For
users who already have a CPD stack, ``RupturesDetector`` and ``from_times``
provide alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from ..core.trajectory import Trajectory, TrajectorySet
from .base import Event


@dataclass
class SegmentationConfig:
    velocity_window: int = 5
    penalty: float = 12.0           # PELT penalty (per changepoint), in cost units
    min_size: int = 8               # minimum frames between changepoints
    stop_speed: float = 0.2         # m/s; below this is "stopped"
    speed_change_z: float = 1.5     # |dspeed| / sigma to call a speed change
    turn_rate_z: float = 1.5        # |dturn| / sigma to call a turn
    sharp_turn_rate: float = 1.0    # rad/s; abs turn rate for 'sharp_turn'
    detection_arl: float = 50.0     # nominal average-run-length for the floor


EVENT_SENSITIVITY_PRESETS: Dict[str, SegmentationConfig] = {
    "high": SegmentationConfig(
        velocity_window=5,
        penalty=8.0,
        min_size=8,
        speed_change_z=1.25,
        turn_rate_z=1.25,
    ),
    "medium": SegmentationConfig(),
    "low": SegmentationConfig(
        velocity_window=10,
        penalty=80.0,
        min_size=50,
        speed_change_z=2.5,
        turn_rate_z=2.0,
        sharp_turn_rate=1.25,
    ),
    "very-low": SegmentationConfig(
        velocity_window=15,
        penalty=160.0,
        min_size=80,
        speed_change_z=3.0,
        turn_rate_z=3.0,
        sharp_turn_rate=1.5,
    ),
}


def segmentation_config(
    preset: str = "medium",
    **overrides,
) -> SegmentationConfig:
    """Build a kinematic detector config from an application-level preset.

    Sensitivity decreases from ``high`` to ``very-low``. Keyword overrides can
    tune any ``SegmentationConfig`` field after the preset is applied.
    """
    if preset not in EVENT_SENSITIVITY_PRESETS:
        raise ValueError(
            f"preset must be one of {tuple(EVENT_SENSITIVITY_PRESETS)}; got {preset!r}"
        )
    return replace(EVENT_SENSITIVITY_PRESETS[preset], **overrides)


def _pelt(signal: np.ndarray, penalty: float, min_size: int) -> List[int]:
    """PELT for piecewise-constant multivariate mean with L2 cost.

    Returns interior changepoint indices (1..n-1).
    """
    n = len(signal)
    if n < 2 * min_size:
        return []
    cs = np.vstack([np.zeros(signal.shape[1]), np.cumsum(signal, axis=0)])
    cs2 = np.concatenate([[0.0], np.cumsum((signal**2).sum(axis=1))])

    def seg_cost(s: int, e: int) -> float:
        length = e - s
        total_sq = cs2[e] - cs2[s]
        seg_sum = cs[e] - cs[s]
        return float(total_sq - (seg_sum**2).sum() / length)

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last_cp = [0] * (n + 1)
    R = [0]
    for e in range(min_size, n + 1):
        best, best_s = np.inf, 0
        for s in R:
            if e - s < min_size:
                continue
            c = F[s] + seg_cost(s, e) + penalty
            if c < best:
                best, best_s = c, s
        F[e] = best
        last_cp[e] = best_s
        R = [s for s in R if F[s] + seg_cost(s, e) <= F[e]]
        R.append(e)

    cps: List[int] = []
    e = n
    while e > 0:
        s = last_cp[e]
        if s > 0:
            cps.append(s)
        e = s
    return sorted(cps)


class KinematicChangepoints:
    """Detect kinematic regime changes; the default REACT EventSource."""

    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()

    @classmethod
    def from_preset(cls, preset: str = "medium", **overrides) -> KinematicChangepoints:
        """Construct a detector using an application-level sensitivity preset."""
        return cls(segmentation_config(preset, **overrides))

    # ---- feature extraction ----

    def _features(self, obs: np.ndarray, dt: float):
        w = self.config.velocity_window
        t = len(obs)
        if t <= w + 1:
            return None
        vel = (obs[w:] - obs[:-w]) / (w * dt)
        speed = np.hypot(vel[:, 0], vel[:, 1]) if obs.shape[1] >= 2 else np.abs(vel[:, 0])
        if obs.shape[1] >= 2:
            heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
            turn_rate = np.gradient(heading) / dt
        else:
            turn_rate = np.zeros_like(speed)
        return speed, turn_rate, w

    def detect(self, traj: Trajectory) -> List[Event]:
        obs = traj.positions
        dt = traj.dt
        feats = self._features(obs, dt)
        if feats is None:
            return []
        speed, turn_rate, w = feats
        cfg = self.config

        sp_sigma = max(np.std(np.diff(speed)) / np.sqrt(2), 1e-3)
        tr_sigma = max(np.std(np.diff(turn_rate)) / np.sqrt(2), 1e-3)
        signal = np.column_stack([speed / sp_sigma, turn_rate / tr_sigma])
        cps = _pelt(signal, cfg.penalty, cfg.min_size)

        events: List[Event] = []
        bounds = [0] + cps + [len(signal)]
        for bi in range(1, len(bounds) - 1):
            cp = bounds[bi]
            pre_lo, post_hi = bounds[bi - 1], bounds[bi + 1]
            t0 = cp + w  # map feature index back to trajectory frame
            ev = self._classify(
                speed, turn_rate, cp, pre_lo, post_hi, sp_sigma, tr_sigma, dt, traj.id
            )
            ev.t0 = int(t0)
            events.append(ev)
        return events

    def _classify(self, speed, turn_rate, cp, lo, hi, sp_sigma, tr_sigma, dt, tid) -> Event:
        cfg = self.config
        pre_sp, post_sp = speed[lo:cp], speed[cp:hi]
        pre_tr, post_tr = turn_rate[lo:cp], turn_rate[cp:hi]
        d_speed = float(np.mean(post_sp) - np.mean(pre_sp))
        d_turn = float(np.mean(post_tr) - np.mean(pre_tr))
        speed_z = abs(d_speed) / sp_sigma
        turn_z = abs(d_turn) / tr_sigma

        is_speed = speed_z >= cfg.speed_change_z
        is_turn = turn_z >= cfg.turn_rate_z

        pre_mean_sp, post_mean_sp = float(np.mean(pre_sp)), float(np.mean(post_sp))
        if is_speed and post_mean_sp < cfg.stop_speed <= pre_mean_sp:
            etype = "stop"
        elif is_speed and pre_mean_sp < cfg.stop_speed <= post_mean_sp:
            etype = "start"
        elif is_turn and is_speed:
            etype = "both"
        elif is_turn:
            if abs(float(np.mean(post_tr))) >= cfg.sharp_turn_rate:
                etype = "sharp_turn"
            elif abs(float(np.mean(pre_tr))) < cfg.turn_rate_z * tr_sigma:
                etype = "turn_onset"
            elif abs(float(np.mean(post_tr))) < cfg.turn_rate_z * tr_sigma:
                etype = "turn_exit"
            else:
                etype = "turn_change"
        elif is_speed:
            etype = "speed_change"
        else:
            etype = "speed_change" if speed_z >= turn_z else "turn_change"

        if is_speed and is_turn:
            series = "both"
        elif is_turn:
            series = "heading"
        else:
            series = "speed"

        floor = self._detection_floor(speed_z, turn_z, len(post_sp))
        return Event(
            id=tid, t0=cp, event_type=etype, series=series,
            speed_level_z=speed_z, heading_slope_z=turn_z,
            d_speed_level=d_speed, d_turn_rate=d_turn,
            detection_floor_frames=floor,
        )

    def _detection_floor(self, speed_z: float, turn_z: float, post_len: int) -> float:
        """Quickest-detection delay bound: log(ARL) / KL_per_frame.

        For a per-frame standardized mean shift z, KL ~= z^2 / 2, so the optimal
        CUSUM expected delay scales as log(ARL) / (z^2/2). The most detectable
        channel sets the floor.
        """
        z = max(speed_z, turn_z)
        if z <= 1e-6:
            return float(post_len)
        kl = 0.5 * z**2
        floor = np.log(self.config.detection_arl) / kl
        return float(max(1.0, min(floor, post_len)))

    def detect_set(self, ts: TrajectorySet) -> Dict[object, List[Event]]:
        return {tid: self.detect(traj) for tid, traj in ts.items()}

    def calibrate_penalty(
        self, calib: TrajectorySet, target_false_rate: float = 0.02, max_iter: int = 12
    ) -> float:
        """Bisection on the PELT penalty to hit a target changepoints-per-frame rate.

        ``target_false_rate`` is the desired density of detected changepoints per
        frame on the calibration set (a proxy for the operational false-alarm
        rate). Returns and stores the calibrated penalty.
        """
        def rate_for(pen: float) -> float:
            self.config.penalty = pen
            n_cp = 0
            n_frames = 0
            for traj in calib:
                evs = self.detect(traj)
                n_cp += len(evs)
                n_frames += max(traj.T, 1)
            return n_cp / max(n_frames, 1)

        lo, hi = 1.0, 200.0
        for _ in range(max_iter):
            mid = np.sqrt(lo * hi)
            r = rate_for(mid)
            if r > target_false_rate:
                lo = mid
            else:
                hi = mid
        self.config.penalty = np.sqrt(lo * hi)
        return self.config.penalty

    def calibrate_events_per_track(
        self,
        calib: TrajectorySet,
        target_events_per_track: float,
        max_iter: int = 12,
    ) -> float:
        """Tune the PELT penalty to an interpretable event count per track."""
        if target_events_per_track <= 0:
            raise ValueError("target_events_per_track must be positive")
        mean_frames = np.mean([max(traj.T, 1) for traj in calib])
        target_rate = target_events_per_track / max(float(mean_frames), 1.0)
        return self.calibrate_penalty(calib, target_false_rate=target_rate, max_iter=max_iter)

    def calibrate_events_per_minute(
        self,
        calib: TrajectorySet,
        target_events_per_minute: float,
        max_iter: int = 12,
        dt: Optional[float] = None,
    ) -> float:
        """Tune the PELT penalty to an operational event rate per trajectory minute."""
        if target_events_per_minute <= 0:
            raise ValueError("target_events_per_minute must be positive")
        if dt is None:
            dt = float(np.mean([traj.dt for traj in calib]))
        target_rate = target_events_per_minute * dt / 60.0
        return self.calibrate_penalty(calib, target_false_rate=target_rate, max_iter=max_iter)
