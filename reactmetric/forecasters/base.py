"""The Forecaster interface: predictive distributions at every valid issue frame.

Every evaluated model -- yours, the built-in baselines, the oracles -- is wrapped
behind one method so the NLL / information-score computation lives in exactly one
place and no model can gain or lose from incidental likelihood-evaluation
differences.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory


@runtime_checkable
class Forecaster(Protocol):
    """Produce predictive Gaussians for a trajectory, or None if too short."""

    name: str

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        ...


class BaseForecaster:
    """Convenience base class carrying a ``name`` attribute."""

    name: str = "base"

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        raise NotImplementedError
