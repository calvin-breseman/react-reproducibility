"""Shared constants for reactmetric.

The library is domain-agnostic, but spatial (2D/3D) trajectory forecasting is
the first-class use case, so the defaults below match the 10 Hz pedestrian
setting REACT was developed on. Override per call where your domain differs.
"""

from __future__ import annotations

# Default frame interval (seconds). 10 fps is the Standard Cognition Tracks /
# ETH-UCY convention. Trajectories carry their own dt; this is only a fallback.
DEFAULT_DT: float = 0.1

# Numerical floor added to covariance diagonals before inversion.
COV_REGULARIZATION: float = 1e-6

# Event taxonomy produced by the kinematic changepoint detector. Kept stable so
# downstream code (and saved results) can rely on these labels.
EVENT_TYPES = (
    "stop",
    "start",
    "speed_change",
    "turn_onset",
    "turn_exit",
    "turn_change",
    "sharp_turn",
    "both",
)


def react_at_lead(lead) -> str:
    """Canonical label for the metric at forecast lead ``h`` (e.g. ``REACT@15``).

    REACT without a lead is undefined: the information score is always taken at
    a specific h-step-ahead forecast. Prefer this helper anywhere metrics or
    plots are labeled.
    """
    if lead is None:
        return "REACT"
    return f"REACT@{int(lead)}"


def react_frames_ylabel(lead) -> str:
    """Axis label for recovery time in frames at a given lead."""
    return f"{react_at_lead(lead)} (frames)"
