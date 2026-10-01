"""Event model and the EventSource protocol.

An Event is one regime change ("commitment event") on one trajectory at frame
``t0``. The default source is the kinematic changepoint detector, but any object
implementing EventSource -- including a bring-your-own timestamp adapter or a
ruptures plugin -- can supply events to REACT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, runtime_checkable

import numpy as np

from ..constants import DEFAULT_DT
from ..core.trajectory import Trajectory, TrajectorySet


@dataclass
class Event:
    """A regime change on one trajectory."""

    id: object                       # trajectory id
    t0: int                          # changepoint frame (hindsight-localized)
    event_type: str = "unknown"      # taxonomy label
    series: str = ""                 # which channel moved: 'speed' | 'heading' | 'both'

    # Standardized effect sizes (per-sample sigma units).
    heading_level_z: float = 0.0
    heading_slope_z: float = 0.0
    speed_level_z: float = 0.0
    speed_slope_z: float = 0.0

    # Physical effect sizes.
    d_turn_rate: float = 0.0
    d_speed_slope: float = 0.0
    d_heading_level: float = 0.0
    d_speed_level: float = 0.0

    detection_floor_frames: float = float("inf")
    t0_std: float = float("nan")

    def floor_seconds(self, dt: float = DEFAULT_DT) -> float:
        return self.detection_floor_frames * dt


@runtime_checkable
class EventSource(Protocol):
    """Anything that can produce events for trajectories."""

    def detect(self, traj: Trajectory) -> List[Event]:
        ...

    def detect_set(self, ts: TrajectorySet) -> Dict[object, List[Event]]:
        ...


class FixedEvents:
    """Bring-your-own events: a precomputed mapping id -> list of t0 frames.

    Use this when you already know where the disruptions are (e.g. labeled
    maneuvers) and want to pre-empt the built-in detector.
    """

    def __init__(self, times: Dict[object, object], event_type: str = "user"):
        self._times: Dict[object, List[int]] = {
            k: [int(t) for t in np.atleast_1d(v)] for k, v in times.items()
        }
        self._event_type = event_type

    def detect(self, traj: Trajectory) -> List[Event]:
        return [
            Event(id=traj.id, t0=t, event_type=self._event_type)
            for t in self._times.get(traj.id, [])
        ]

    def detect_set(self, ts: TrajectorySet) -> Dict[object, List[Event]]:
        return {tid: self.detect(traj) for tid, traj in ts.items()}


def from_times(times: Dict[object, object], event_type: str = "user") -> FixedEvents:
    """Build a bring-your-own EventSource from a mapping id -> t0 frame(s)."""
    return FixedEvents(times, event_type=event_type)


def events_by_track(events: List[Event]) -> Dict[object, List[Event]]:
    by: Dict[object, List[Event]] = {}
    for e in events:
        by.setdefault(e.id, []).append(e)
    return by


def flatten_events(detected: Dict[object, List[Event]]) -> List[Event]:
    out: List[Event] = []
    for evs in detected.values():
        out.extend(evs)
    return out
