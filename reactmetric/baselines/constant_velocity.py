"""Constant-velocity (persistence) baseline: a Kalman filter.

The standard constant-velocity tracker (Bar-Shalom, Li & Kirubarajan 2001, ch. 6):
the state is position and velocity per axis, the velocity drifts as discrete white
noise acceleration with standard deviation ``accel_noise`` (m/s^2), and positions
are observed with per-axis standard deviation ``measurement_noise`` (m). The
predictive Gaussian at each lead is the filter's own propagated covariance plus
measurement noise, so the spread grows with lead as the motion model implies.

The two noise levels are the model's only parameters. ``ConstantVelocity.fit``
estimates both by maximum likelihood on the filter's one-step innovations
(Harvey 1989, ch. 3), the standard way to tune a Kalman filter from data.
Calibration at longer leads then follows from the model and is not tuned.
Works in any dimension (axes are independent and share one noise model).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster
from ._kinematics import stack_tracks

_LOG_2PI = float(np.log(2.0 * np.pi))


def _cv_model(dt: float, accel_noise: float):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = accel_noise**2 * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])
    return F, Q


def _gain_sequence(n: int, dt: float, accel_noise: float, measurement_noise: float):
    """Prior covariances P_t|t-1 and gains for t = 2 .. n-1.

    The covariance recursion does not depend on the data, so every track with the
    same initialisation shares it. Initialisation is two-point differencing at t = 1.
    """
    F, Q = _cv_model(dt, accel_noise)
    R = measurement_noise**2
    P = np.array([[R, R / dt], [R / dt, 2.0 * R / dt**2]])
    priors = np.zeros((n, 2, 2))
    gains = np.zeros((n, 2))
    for t in range(2, n):
        P = F @ P @ F.T + Q
        priors[t] = P
        S = P[0, 0] + R
        K = P[:, 0] / S
        gains[t] = K
        P = P - np.outer(K, P[0, :])
    return priors, gains


def _filter_states(obs: np.ndarray, dt: float, gains: np.ndarray):
    """Prior means x_t|t-1 for t >= 2. obs (..., T, D) -> (..., T, 2, D)."""
    T = obs.shape[-2]
    x = np.stack([obs[..., 1, :], (obs[..., 1, :] - obs[..., 0, :]) / dt], axis=-2)
    priors = np.zeros(obs.shape[:-2] + (T, 2, obs.shape[-1]))
    for t in range(2, T):
        x = np.stack([x[..., 0, :] + dt * x[..., 1, :], x[..., 1, :]], axis=-2)
        priors[..., t, :, :] = x
        innov = obs[..., t, :] - x[..., 0, :]
        x = x + gains[t][:, None] * innov[..., None, :]
    return priors


class ConstantVelocity(BaseForecaster):
    def __init__(
        self,
        history: int = 10,
        horizon: int = 15,
        measurement_noise: float = 0.05,
        accel_noise: float = 0.5,
        name: str = "ConstantVelocity",
    ):
        self.history = int(history)
        self.horizon = int(horizon)
        self.measurement_noise = float(measurement_noise)
        self.accel_noise = float(accel_noise)
        self.name = name

    @classmethod
    def fit(
        cls,
        data,
        history: int = 10,
        horizon: int = 15,
        max_tracks: int = 2000,
        seed: int = 0,
        name: str = "ConstantVelocity",
    ) -> ConstantVelocity:
        """Maximum-likelihood noise levels from one-step innovations on ``data``."""
        from scipy.optimize import minimize

        obs, lengths, dt = stack_tracks(data, max_tracks=max_tracks, seed=seed)
        N, T, D = obs.shape
        live = np.arange(T)[None, :] < lengths[:, None]  # (N, T)
        live[:, :2] = False

        def nll(theta):
            sa, sr = np.exp(theta)
            priors, gains = _gain_sequence(T, dt, sa, sr)
            x = _filter_states(obs, dt, gains)[..., 0, :]  # (N, T, D)
            S = priors[:, 0, 0] + sr**2  # (T,)
            r2 = ((obs - x) ** 2).sum(axis=-1)  # (N, T)
            ll = -0.5 * (D * (_LOG_2PI + np.log(S))[None, :] + r2 / S[None, :])
            return -float(ll[live].sum()) / live.sum()

        res = minimize(
            nll,
            np.log([0.5, 0.02]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400},
        )
        sa, sr = np.exp(res.x)
        model = cls(
            history=history,
            horizon=horizon,
            measurement_noise=float(sr),
            accel_noise=float(sa),
            name=name,
        )
        model.fit_result_ = {
            "nll_per_frame": float(res.fun),
            "n_frames": int(live.sum()),
            "n_tracks": int(N),
            "converged": bool(res.success),
        }
        return model

    def effective_memory(self, dt: float = 0.1) -> float:
        """Frames of history carrying (1 - 1/e) of the velocity estimate's weight.

        The steady-state filter forgets old observations geometrically; this is the
        number of frames over which the weights fall by a factor e.
        """
        _, gains = _gain_sequence(400, dt, self.accel_noise, self.measurement_noise)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        K = gains[-1]
        A = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ F
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        return float(-1.0 / np.log(rho)) if 0 < rho < 1 else float("inf")

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        obs = np.asarray(traj.positions, float)
        dt = traj.dt
        t, dim = obs.shape
        L, H = max(self.history, 2), self.horizon
        if t < L + H:
            return None
        F, Q = _cv_model(dt, self.accel_noise)
        R = self.measurement_noise**2
        P_prior, gains = _gain_sequence(t, dt, self.accel_noise, self.measurement_noise)
        x_prior = _filter_states(obs, dt, gains)  # (T, 2, D)

        # issue frame ti sees observations up to ti - 1; column 0 targets ti itself.
        issue = np.arange(L, t - H + 1)
        steps = np.arange(H) * dt
        means = x_prior[issue, 0, None, :] + steps[None, :, None] * x_prior[issue, 1, None, :]
        var = np.empty((len(issue), H))
        for i, ti in enumerate(issue):
            P = P_prior[ti]
            for h in range(H):
                var[i, h] = P[0, 0] + R
                P = F @ P @ F.T + Q
        covs = var[:, :, None, None] * np.eye(dim)[None, None]
        return Predictive(issue_frames=issue, means=means, covs=covs, horizon=H)
