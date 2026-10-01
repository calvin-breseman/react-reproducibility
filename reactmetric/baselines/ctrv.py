"""CTRV (Constant Turn Rate and Velocity) baseline via an EKF.

A reference implementation of the CTRV motion model that is the workhorse single-
model tracker for agents that move and turn. State is [x, y, v, theta, omega];
the filter runs over the observed history and then extrapolates with covariance
propagation. The position block of the propagated covariance (plus measurement
noise) is the predictive Gaussian. 2D only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster
from ._kinematics import CalibratedNoise


def _ctrv_step(state: np.ndarray, dt: float) -> np.ndarray:
    x, y, v, th, om = state
    if abs(om) > 1e-5:
        x = x + v / om * (np.sin(th + om * dt) - np.sin(th))
        y = y + v / om * (-np.cos(th + om * dt) + np.cos(th))
    else:
        x = x + v * np.cos(th) * dt
        y = y + v * np.sin(th) * dt
    th = th + om * dt
    return np.array([x, y, v, th, om])


def _jacobian(state: np.ndarray, dt: float, eps: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian of the CTRV step (robust near omega -> 0)."""
    f0 = _ctrv_step(state, dt)
    J = np.zeros((5, 5))
    for i in range(5):
        ds = np.zeros(5)
        ds[i] = eps
        J[:, i] = (_ctrv_step(state + ds, dt) - f0) / eps
    return J


class CTRV(BaseForecaster):
    def __init__(
        self,
        calibration: Optional[CalibratedNoise] = None,
        measurement_noise: Optional[float] = None,
        history: int = 10,
        horizon: int = 15,
        sigma_a: float = 0.6,        # linear-acceleration process noise (m/s^2)
        sigma_yaw: float = 0.8,      # yaw-acceleration process noise (rad/s^2)
        temperature: float = 1.0,    # covariance scale (kept explicit, never auto-tuned)
        name: str = "CTRV",
    ):
        if measurement_noise is None:
            measurement_noise = calibration.measurement_noise if calibration else 0.05
        self.measurement_noise = float(measurement_noise)
        self.history = int(history)
        self.horizon = int(horizon)
        self.sigma_a = float(sigma_a)
        self.sigma_yaw = float(sigma_yaw)
        self.temperature = float(temperature)
        self.name = name

    def _Q(self, dt: float) -> np.ndarray:
        q = np.zeros((5, 5))
        q[2, 2] = (self.sigma_a * dt) ** 2
        q[3, 3] = (0.5 * self.sigma_yaw * dt**2) ** 2
        q[4, 4] = (self.sigma_yaw * dt) ** 2
        return q

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        if traj.D != 2:
            return None
        obs = traj.positions
        dt = traj.dt
        T, L, H = traj.T, self.history, self.horizon
        if T < L + H:
            return None

        sigma_r = self.measurement_noise
        R = np.eye(2) * sigma_r**2
        Hmat = np.zeros((2, 5))
        Hmat[0, 0] = Hmat[1, 1] = 1.0
        Q = self._Q(dt)

        v0 = (obs[3] - obs[0]) / (3 * dt)
        sp = float(np.hypot(*v0))
        th = float(np.arctan2(v0[1], v0[0]))
        state = np.array([obs[0, 0], obs[0, 1], sp, th, 0.0])
        P = np.diag([sigma_r**2, sigma_r**2, 0.5**2, 0.3**2, 0.3**2])

        issue_frames, means_out, covs_out = [], [], []
        for t in range(1, T):
            F = _jacobian(state, dt)
            state = _ctrv_step(state, dt)
            P = F @ P @ F.T + Q

            if L <= t <= T - H:
                f_means = np.empty((H, 2))
                f_covs = np.empty((H, 2, 2))
                s_f = state.copy()
                P_f = P.copy()
                for h in range(H):
                    f_means[h] = s_f[:2]
                    f_covs[h] = P_f[:2, :2] * self.temperature + R
                    if h < H - 1:
                        Ff = _jacobian(s_f, dt)
                        s_f = _ctrv_step(s_f, dt)
                        P_f = Ff @ P_f @ Ff.T + Q
                issue_frames.append(t)
                means_out.append(f_means)
                covs_out.append(f_covs)

            S = Hmat @ P @ Hmat.T + R
            K = P @ Hmat.T @ np.linalg.inv(S)
            resid = obs[t] - Hmat @ state
            state = state + K @ resid
            P = (np.eye(5) - K @ Hmat) @ P

        if not issue_frames:
            return None
        return Predictive(
            issue_frames=np.array(issue_frames, dtype=int),
            means=np.array(means_out),
            covs=np.array(covs_out),
            horizon=H,
        )
