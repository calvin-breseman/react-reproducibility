"""Interacting Multiple Model (IMM) baseline: dwell + CV + left/right turns.

The standard adaptive baseline from the target-tracking literature, ported from
the internal pipeline. Each mode is a Kalman filter with its own dynamics; mode
probabilities evolve via a Markov transition matrix and the mixture is
moment-matched to a single Gaussian per forecast step. 2D only.

The predictive covariance is the filter's own propagated covariance plus
measurement noise. ``IMM.fit`` sets the noise levels, the stop mode's velocity
decay, the turn rate and the mode persistence by maximum likelihood on the
filter's one-step innovations (the IMM likelihood is the mode mixture of the
per-mode innovation densities; Bar-Shalom, Li & Kirubarajan 2001, sec. 11.6).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..constants import COV_REGULARIZATION
from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster
from ..scoring.nll import LOG_2PI
from ._kinematics import CalibratedNoise, stack_tracks


def _cv_F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _ct_F(omega: float, dt: float) -> np.ndarray:
    if abs(omega) < 1e-9:
        return _cv_F(dt)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array(
        [
            [1, 0, s / omega, -(1 - c) / omega],
            [0, 1, (1 - c) / omega, s / omega],
            [0, 0, c, -s],
            [0, 0, s, c],
        ]
    )


def _white_accel_Q(sigma_a: float, dt: float) -> np.ndarray:
    q_pp = dt**4 / 4.0
    q_pv = dt**3 / 2.0
    q_vv = dt**2
    Q1 = np.array([[q_pp, q_pv], [q_pv, q_vv]]) * sigma_a**2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return Q


def _stop_F(velocity_decay: float) -> np.ndarray:
    F = np.eye(4)
    F[2, 2] = float(velocity_decay)
    F[3, 3] = float(velocity_decay)
    return F


class IMM(BaseForecaster):
    def __init__(
        self,
        calibration: Optional[CalibratedNoise] = None,
        measurement_noise: Optional[float] = None,
        history: int = 10,
        horizon: int = 15,
        modes: Tuple[str, ...] = ("stop", "cv", "turn_left", "turn_right"),
        turn_rate: float = 0.6,
        sigma_a_cv: float = 0.4,
        sigma_a_ct: float = 0.7,
        sigma_a_stop: float = 0.15,
        stop_velocity_decay: float = 0.15,
        p_stay: float = 0.97,
        measurement_noise_stop: Optional[float] = None,
        p_stay_stop: Optional[float] = None,
        name: str = "IMM",
    ):
        if measurement_noise is None:
            measurement_noise = calibration.measurement_noise if calibration else 0.05
        self.measurement_noise = float(measurement_noise)
        self.history = int(history)
        self.horizon = int(horizon)
        self.turn_rate = float(turn_rate)
        self.sigma_a_cv = float(sigma_a_cv)
        self.sigma_a_ct = float(sigma_a_ct)
        self.sigma_a_stop = float(sigma_a_stop)
        self.stop_velocity_decay = float(stop_velocity_decay)
        self.p_stay = float(p_stay)
        # The stop mode may carry its own sensor noise (trackers jitter less on a person
        # standing still) and its own persistence (people stay stopped far longer than
        # they hold a turn). Unset, both equal the shared values.
        self.measurement_noise_stop = float(
            self.measurement_noise if measurement_noise_stop is None else measurement_noise_stop
        )
        self.p_stay_stop = float(self.p_stay if p_stay_stop is None else p_stay_stop)
        self.name = name

        spec = []
        for m in modes:
            if m == "stop":
                spec.append(("stop", 0.0))
            elif m == "cv":
                spec.append(("cv", 0.0))
            elif m == "turn_left":
                spec.append(("turn_left", self.turn_rate))
            elif m == "turn_right":
                spec.append(("turn_right", -self.turn_rate))
            else:
                raise ValueError(f"unknown IMM mode '{m}'")
        self._mode_specs = spec
        n = len(spec)
        stay = np.array([self.p_stay_stop if k == "stop" else self.p_stay for k, _ in spec])
        self._M = np.repeat(((1.0 - stay) / (n - 1))[:, None], n, axis=1)
        np.fill_diagonal(self._M, stay)
        noise = [
            self.measurement_noise_stop if k == "stop" else self.measurement_noise for k, _ in spec
        ]
        self._R = np.array([np.eye(2) * r**2 for r in noise])  # (m, 2, 2) per-mode sensor noise

    _FIT_PARAMS = (
        "measurement_noise",
        "sigma_a_cv",
        "sigma_a_ct",
        "sigma_a_stop",
        "stop_velocity_decay",
        "p_stay",
        "turn_rate",
        "measurement_noise_stop",
        "p_stay_stop",
    )

    def innovation_loglik(self, obs: np.ndarray, lengths: np.ndarray, dt: float) -> np.ndarray:
        """Per-track sum of one-step log predictive densities, batched over tracks.

        Mirrors the filter in ``predict`` exactly (initialisation, mixing, update);
        obs is (N, T, 2) padded, lengths gives each track's T.
        """
        N, T, _ = obs.shape
        sigma_r = self.measurement_noise
        R_obs = self._R + np.eye(2) * COV_REGULARIZATION  # (m, 2, 2)
        Fs, Qs = self._mode_matrices(dt)
        F = np.stack(Fs)  # (m, 4, 4)
        Q = np.stack(Qs)
        m = len(Fs)
        M = self._M
        v0 = (obs[:, 3] - obs[:, 0]) / (3 * dt)
        x0 = np.concatenate([obs[:, 0], v0], axis=1)  # (N, 4)
        P0 = np.diag([sigma_r**2, sigma_r**2, 0.5**2, 0.5**2])
        xs = np.repeat(x0[:, None], m, axis=1)  # (N, m, 4)
        Ps = np.broadcast_to(P0, (N, m, 4, 4)).copy()
        mu = np.full((N, m), 1.0 / m)
        total = np.zeros(N)
        for t in range(1, T):
            cbar = np.maximum(mu @ M, 1e-12)  # (N, m)
            w = M[None] * mu[:, :, None] / cbar[:, None, :]  # (N, i, j)
            xm = np.einsum("nij,nia->nja", w, xs)
            d = xs[:, :, None, :] - xm[:, None, :, :]  # (N, i, j, 4)
            Pm = np.einsum("nij,niab->njab", w, Ps) + np.einsum("nij,nija,nijb->njab", w, d, d)
            xs = np.einsum("jab,njb->nja", F, xm)
            Ps = np.einsum("jab,njbc,jdc->njad", F, Pm, F) + Q[None]
            S = Ps[:, :, :2, :2] + R_obs[None]  # (N, m, 2, 2)
            Si = np.linalg.inv(S)
            r = obs[:, t, None, :] - xs[:, :, :2]  # (N, m, 2)
            K = np.einsum("njab,njbc->njac", Ps[:, :, :, :2], Si)  # (N, m, 4, 2)
            xs = xs + np.einsum("njab,njb->nja", K, r)
            Ps = Ps - np.einsum("njab,njbc->njac", K, Ps[:, :, :2, :])  # (I - KH) P
            lik = (
                -0.5 * np.einsum("nja,njab,njb->nj", r, Si, r)
                - 0.5 * np.linalg.slogdet(S)[1]
                - LOG_2PI
            )
            log_mu = np.log(cbar) + lik
            peak = log_mu.max(axis=1, keepdims=True)
            step = peak[:, 0] + np.log(np.exp(log_mu - peak).sum(axis=1))
            total += np.where(t < lengths, step, 0.0)
            mu = np.exp(log_mu - peak)
            mu /= mu.sum(axis=1, keepdims=True)
        return total

    # Wide default bounds: the unconstrained fit. Pass tighter ones to keep the
    # modes physical (the one-step likelihood alone lets a turn mode become a
    # catch-all high-noise mode, whose noise then compounds over longer leads).
    FIT_BOUNDS = {
        "measurement_noise": (0.005, 0.1),
        "sigma_a_cv": (0.0, 20.0),
        "sigma_a_ct": (0.0, 50.0),
        "sigma_a_stop": (0.0, 5.0),
        "stop_velocity_decay": (0.0, 1.0),
        "p_stay": (0.5, 0.9999),
        "turn_rate": (0.0, 3.0),
        "measurement_noise_stop": (0.005, 0.1),
        "p_stay_stop": (0.5, 0.9999),
    }

    @classmethod
    def fit(
        cls,
        data,
        bounds: Optional[dict] = None,
        max_tracks: int = 400,
        segment: int = 250,
        seed: int = 0,
        maxiter: int = 20,
        **kwargs,
    ) -> IMM:
        """Maximum-likelihood IMM parameters from one-step innovations on ``data``.

        Fits measurement noise, the three process-noise levels, the stop mode's
        velocity decay, the turn rate and the mode persistence p_stay, each within
        ``bounds`` (name -> (low, high); unspecified names use ``FIT_BOUNDS``; set
        low == high to hold a parameter fixed). Other constructor arguments (modes,
        history, horizon, name) pass through.
        """
        from scipy.optimize import minimize

        obs, lengths, dt = stack_tracks(data, max_tracks=max_tracks, seed=seed, segment=segment)
        n_frames = float((lengths - 1).sum())
        box = {**cls.FIT_BOUNDS, **(bounds or {})}
        lo = np.array([box[k][0] for k in cls._FIT_PARAMS], float)
        hi = np.array([box[k][1] for k in cls._FIT_PARAMS], float)
        span = hi - lo
        free = span > 0

        def unpack(u):
            x = lo.copy()
            x[free] = lo[free] + np.clip(u, 0.0, 1.0) * span[free]
            return dict(zip(cls._FIT_PARAMS, map(float, x)))

        def nll(u):
            model = cls(**unpack(u), **kwargs)
            return -float(model.innovation_loglik(obs, lengths, dt).sum()) / n_frames

        start = cls(measurement_noise=0.02, **kwargs)
        x0 = np.clip([getattr(start, k) for k in cls._FIT_PARAMS], lo, hi)
        u0 = (x0[free] - lo[free]) / span[free]
        # Powell: derivative-free with bounds. L-BFGS-B's finite-difference line search
        # stalled here well short of the optimum.
        res = minimize(
            nll,
            u0,
            method="Powell",
            bounds=[(0.0, 1.0)] * int(free.sum()),
            options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-7},
        )
        params = unpack(res.x)
        model = cls(**params, **kwargs)
        u_full = np.full(len(lo), 0.5)
        u_full[free] = res.x
        at_bound = [
            k for k, u, f in zip(cls._FIT_PARAMS, u_full, free) if f and (u < 1e-3 or u > 1 - 1e-3)
        ]
        model.fit_result_ = {
            "nll_per_frame": float(res.fun),
            "n_frames": int(n_frames),
            "n_tracks": int(len(lengths)),
            "converged": bool(res.success),
            "n_evals": int(res.nfev),
            "at_bound": at_bound,
            "bounds": {k: list(box[k]) for k in cls._FIT_PARAMS},
        }
        return model

    def _mode_matrices(self, dt: float):
        Fs, Qs = [], []
        for kind, omega in self._mode_specs:
            if kind == "stop":
                Fs.append(_stop_F(self.stop_velocity_decay))
                Qs.append(_white_accel_Q(self.sigma_a_stop, dt))
            else:
                Fs.append(_ct_F(omega, dt))
                Qs.append(_white_accel_Q(self.sigma_a_cv if kind == "cv" else self.sigma_a_ct, dt))
        return Fs, Qs

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        if traj.D != 2:
            return None
        obs = traj.positions
        dt = traj.dt
        T, L, H = traj.T, self.history, self.horizon
        if T < L + H:
            return None

        sigma_r = self.measurement_noise
        R_modes = self._R
        R_obs = R_modes + np.eye(2) * COV_REGULARIZATION
        Hmat = np.zeros((2, 4))
        Hmat[0, 0] = Hmat[1, 1] = 1.0
        Fs, Qs = self._mode_matrices(dt)
        n_modes = len(self._mode_specs)
        M = self._M

        v0 = (obs[3] - obs[0]) / (3 * dt)
        x0 = np.array([obs[0, 0], obs[0, 1], v0[0], v0[1]])
        P0 = np.diag([sigma_r**2, sigma_r**2, 0.5**2, 0.5**2])
        xs = [x0.copy() for _ in range(n_modes)]
        Ps = [P0.copy() for _ in range(n_modes)]
        mu = np.full(n_modes, 1.0 / n_modes)

        issue_frames, means_out, covs_out = [], [], []
        for t in range(1, T):
            cbar = np.maximum(M.T @ mu, 1e-12)
            mix_w = (M * mu[:, None]) / cbar[None, :]
            xs_mixed, Ps_mixed = [], []
            for j in range(n_modes):
                xm = sum(mix_w[i, j] * xs[i] for i in range(n_modes))
                Pm = sum(
                    mix_w[i, j] * (Ps[i] + np.outer(xs[i] - xm, xs[i] - xm)) for i in range(n_modes)
                )
                xs_mixed.append(xm)
                Ps_mixed.append(Pm)
            for j in range(n_modes):
                xs[j] = Fs[j] @ xs_mixed[j]
                Ps[j] = Fs[j] @ Ps_mixed[j] @ Fs[j].T + Qs[j]

            if L <= t <= T - H:
                f_means = np.empty((H, 2))
                f_covs = np.empty((H, 2, 2))
                xs_f = [x.copy() for x in xs]
                Ps_f = [P.copy() for P in Ps]
                mu_f = M.T @ mu
                for h in range(H):
                    xm = sum(mu_f[j] * xs_f[j] for j in range(n_modes))
                    Pm = sum(
                        mu_f[j] * (Ps_f[j] + np.outer(xs_f[j] - xm, xs_f[j] - xm))
                        for j in range(n_modes)
                    )
                    f_means[h] = xm[:2]
                    # observation noise of whichever mode the person is in
                    f_covs[h] = Pm[:2, :2] + np.einsum("j,jab->ab", mu_f, R_modes)
                    if h < H - 1:
                        for j in range(n_modes):
                            xs_f[j] = Fs[j] @ xs_f[j]
                            Ps_f[j] = Fs[j] @ Ps_f[j] @ Fs[j].T + Qs[j]
                        mu_f = M.T @ mu_f
                issue_frames.append(t)
                means_out.append(f_means)
                covs_out.append(f_covs)

            z = obs[t]
            likelihoods = np.empty(n_modes)
            for j in range(n_modes):
                S = Hmat @ Ps[j] @ Hmat.T + R_obs[j]
                S_inv = np.linalg.inv(S)
                resid = z - Hmat @ xs[j]
                K = Ps[j] @ Hmat.T @ S_inv
                xs[j] = xs[j] + K @ resid
                Ps[j] = (np.eye(4) - K @ Hmat) @ Ps[j]
                _, log_det = np.linalg.slogdet(S)
                likelihoods[j] = -0.5 * float(resid @ S_inv @ resid) - 0.5 * log_det - LOG_2PI

            log_mu = np.log(np.maximum(M.T @ mu, 1e-300)) + likelihoods
            log_mu -= log_mu.max()
            mu = np.exp(log_mu)
            mu /= mu.sum()

        if not issue_frames:
            return None
        return Predictive(
            issue_frames=np.array(issue_frames, dtype=int),
            means=np.array(means_out),
            covs=np.array(covs_out),
            horizon=H,
        )
