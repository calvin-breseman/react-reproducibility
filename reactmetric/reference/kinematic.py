"""Kinematic-conditional displacement climatology (2D).

p(displacement | speed, heading): the empirical conditional climatology -- among
observed people moving this fast in this direction, what did their lead-step
futures look like. This is NOT a constant-velocity model; it is the body-frame
conditional slice (rotation invariance: heading only sets the frame; translation
invariance: pooled over positions), with speed a continuous covariate. Each
per-(speed, lead) future is a single anisotropic Gaussian (maximum entropy given
its conditional mean and covariance).

Because speed/heading come from a causal lagging window, the reference is still
surprised by genuine maneuvers; only trivially-predictable persistence is
conditioned out. A separate isotropic stopped regime handles people standing still.

How the two regimes are weighted is set by ``stop_model``:

* ``"ramp"`` (default): a hand-set linear ramp from ``stop_speed`` to
  ``stop_speed + transition_band``, with each regime fitted on its side of the
  threshold.
* ``"fitted"``: the probability of the moving regime is a smooth function of speed
  estimated from data, jointly with both regimes, by EM (a mixture of experts with
  kernel-smoothed M-steps over the speed grid). A person who looks stopped but
  starts moving within the lead is then accounted for at the rate it happens in the
  calibration data, instead of by where a threshold happens to sit.

Heading is a 2D notion, so this mode requires D == 2; the facade falls back to the
global climatology for other dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..constants import DEFAULT_DT
from ..scoring.nll import (
    LOG_2PI,
    GaussianMixtureND,
    eps_star_from_cov,
    gaussian_nll,
    supported_distortion_from_cov,
)
from .base import (
    CV_VELOCITY_WINDOW,
    MAX_FIT_SAMPLES,
    cv_nll_trace_from_cov,
    fit_cv_residual_cov,
    regularize_cov,
)


def _weighted_moments(samples: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = weights / max(float(weights.sum()), 1e-12)
    mu = (w[:, None] * samples).sum(axis=0)
    d = samples - mu
    cov = np.einsum("n,ni,nj->ij", w, d, d)
    return mu, cov


@dataclass
class KinematicClimatology:
    """Conditional displacement reference p(displacement | speed, heading), 2D."""

    leads: Tuple[int, ...]
    dim: int
    dt: float
    velocity_window: int
    stop_speed: float
    speed_kernel_bw: float
    transition_band: float
    speed_grid: Dict[int, np.ndarray]
    mu_grid: Dict[int, np.ndarray]
    cov_grid: Dict[int, np.ndarray]
    stop_cov: Dict[int, np.ndarray]
    marginal_cov: Dict[int, np.ndarray]
    cv_covs: Dict[int, np.ndarray]
    n_samples: Dict[int, int] = field(default_factory=dict)
    n_calib_tracks: int = 0
    stop_model: str = "ramp"
    move_weight_grid: Dict[int, np.ndarray] = field(default_factory=dict)
    em_loglik: Dict[int, list] = field(default_factory=dict)

    def move_weight(self, lead: int, speed) -> np.ndarray:
        """Probability of the moving regime at this causal speed."""
        speed = np.asarray(speed, dtype=float)
        if self.stop_model == "fitted":
            return np.interp(speed, self.speed_grid[lead], self.move_weight_grid[lead])
        return np.clip((speed - self.stop_speed) / max(self.transition_band, 1e-9), 0.0, 1.0)

    def _predict_moving(self, lead: int, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grid = self.speed_grid[lead]
        mu_grid = self.mu_grid[lead]
        cov_grid = self.cov_grid[lead]
        mu = np.empty((len(speed), 2))
        mu[:, 0] = np.interp(speed, grid, mu_grid[:, 0])
        mu[:, 1] = np.interp(speed, grid, mu_grid[:, 1])
        c00 = np.interp(speed, grid, cov_grid[:, 0, 0])
        c01 = np.interp(speed, grid, cov_grid[:, 0, 1])
        c11 = np.interp(speed, grid, cov_grid[:, 1, 1])
        cov = np.empty((len(speed), 2, 2))
        cov[:, 0, 0] = c00
        cov[:, 0, 1] = c01
        cov[:, 1, 0] = c01
        cov[:, 1, 1] = c11
        return mu, cov

    def nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        t = len(obs)
        w = self.velocity_window
        if lead not in self.speed_grid or t <= lead + w:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead + w, t)
        anchors = frames - lead
        vel = (obs[anchors] - obs[anchors - w]) / (w * self.dt)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        disp = obs[frames] - obs[anchors]
        cth, sth = np.cos(heading), np.sin(heading)
        disp_body = np.column_stack(
            [cth * disp[:, 0] + sth * disp[:, 1], -sth * disp[:, 0] + cth * disp[:, 1]]
        )
        mu, cov = self._predict_moving(lead, speed)
        nll_move = gaussian_nll(disp_body - mu, cov)
        nll_stop = gaussian_nll(disp, self.stop_cov[lead])
        lam = self.move_weight(lead, speed)
        eps = 1e-12
        logp_move = np.log(lam + eps) - nll_move
        logp_stop = np.log(1.0 - lam + eps) - nll_stop
        peak = np.maximum(logp_move, logp_stop)
        logp = peak + np.log(np.exp(logp_move - peak) + np.exp(logp_stop - peak))
        nll = -logp
        ok = np.isfinite(nll)
        return frames[ok].astype(int), nll[ok]

    def cv_nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        return cv_nll_trace_from_cov(obs, lead, self.dt, self.cv_covs[lead])

    def score_trace(self, traj, lead: int, scoring: str = "nll"):
        if scoring == "nll":
            return self.nll_trace(traj, lead)
        from ..scoring.crps import gaussian_score

        obs = np.asarray(traj.positions if hasattr(traj, "positions") else traj, dtype=float)
        t = len(obs)
        w = self.velocity_window
        if lead not in self.speed_grid or t <= lead + w:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead + w, t)
        anchors = frames - lead
        vel = (obs[anchors] - obs[anchors - w]) / (w * self.dt)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        mu_b, cov_b = self._predict_moving(lead, speed)
        c, s = np.cos(heading), np.sin(heading)
        rot = np.zeros((len(frames), 2, 2))
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        mu_world = np.einsum("nij,nj->ni", rot, mu_b)
        cov_world = np.einsum("nij,njk,nlk->nil", rot, cov_b, rot)
        mu_abs = obs[anchors] + mu_world
        resid = obs[frames] - mu_abs
        score = gaussian_score(resid, cov_world, scoring)
        ok = np.isfinite(score)
        return frames[ok].astype(int), score[ok]

    def conditional_gaussian(
        self, lead: int, speed: float, heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_b, cov_b = self._predict_moving(lead, np.array([float(speed)]))
        mu_b, cov_b = mu_b[0], cov_b[0]
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        return rot @ mu_b, rot @ cov_b @ rot.T

    def displacement_prior(self, lead: int, speed: float, heading: float) -> GaussianMixtureND:
        mu_w, cov_w = self.conditional_gaussian(lead, float(speed), float(heading))
        lam = float(self.move_weight(lead, float(speed)))
        eps = 1e-12
        weights = np.array([lam + eps, 1.0 - lam + eps])
        means = np.vstack([mu_w, np.zeros(2)])
        covs = np.stack([cov_w, self.stop_cov[lead]])
        return GaussianMixtureND(weights=weights, means=means, covs=covs)

    def cv_residual_cov(self, lead: int) -> np.ndarray:
        return self.cv_covs.get(lead, np.zeros((2, 2)))

    def eps_star(self, lead: int, d_meters: float) -> float:
        return eps_star_from_cov(self.marginal_cov[lead], d_meters)

    def supported_distortion(self, lead: int) -> float:
        return supported_distortion_from_cov(self.marginal_cov[lead])

    def metadata(self) -> Dict:
        out = {
            "mode": "kinematic",
            "dim": self.dim,
            "velocity_window": self.velocity_window,
            "stop_speed": self.stop_speed,
            "speed_kernel_bw": self.speed_kernel_bw,
            "transition_band": self.transition_band,
            "stop_model": self.stop_model,
            "n_calib_tracks": self.n_calib_tracks,
            "leads": list(self.leads),
            "per_lead": {},
        }
        for lead in self.leads:
            cov = self.marginal_cov[lead]
            out["per_lead"][str(lead)] = {
                "n_samples": int(self.n_samples.get(lead, 0)),
                "mixture_covariance": cov.tolist(),
                "differential_entropy_gaussian_nats": float(
                    1.0 + LOG_2PI + 0.5 * np.log(np.linalg.det(cov))
                ),
                "cv_residual_covariance": self.cv_covs[lead].tolist(),
            }
        return out


def fit_kinematic_climatology(
    arrays: Dict[object, np.ndarray],
    calib_ids: Sequence[object],
    leads: Sequence[int],
    dt: float = DEFAULT_DT,
    velocity_window: int = CV_VELOCITY_WINDOW,
    stop_speed: float = 0.2,
    speed_kernel_bw: float = 0.2,
    n_speed_knots: int = 16,
    transition_band: float = 0.15,
    seed: int = 0,
    stop_model: str = "ramp",
    em_iters: int = 60,
) -> KinematicClimatology:
    leads_t = tuple(sorted({int(l) for l in leads}))
    w = int(velocity_window)
    rng = np.random.default_rng(seed)
    speed_grid: Dict[int, np.ndarray] = {}
    mu_grid: Dict[int, np.ndarray] = {}
    cov_grid: Dict[int, np.ndarray] = {}
    stop_cov: Dict[int, np.ndarray] = {}
    marginal_cov: Dict[int, np.ndarray] = {}
    cv_covs: Dict[int, np.ndarray] = {}
    n_samples: Dict[int, int] = {}
    move_weight_grid: Dict[int, np.ndarray] = {}
    em_loglik: Dict[int, list] = {}
    if stop_model not in ("ramp", "fitted"):
        raise ValueError(f"stop_model must be 'ramp' or 'fitted'; got {stop_model!r}")

    for lead in leads_t:
        sp_chunks, body_chunks, world_chunks = [], [], []
        for tid in calib_ids:
            obs = arrays[tid]
            t = len(obs)
            if t <= lead + w:
                continue
            idx = np.arange(w, t - lead)
            vel = (obs[idx] - obs[idx - w]) / (w * dt)
            disp = obs[idx + lead] - obs[idx]
            finite = np.isfinite(vel).all(axis=1) & np.isfinite(disp).all(axis=1)
            if not finite.any():
                continue
            vel, disp = vel[finite], disp[finite]
            sp = np.hypot(vel[:, 0], vel[:, 1])
            hd = np.arctan2(vel[:, 1], vel[:, 0])
            cth, sth = np.cos(hd), np.sin(hd)
            body = np.column_stack(
                [cth * disp[:, 0] + sth * disp[:, 1], -sth * disp[:, 0] + cth * disp[:, 1]]
            )
            sp_chunks.append(sp)
            body_chunks.append(body)
            world_chunks.append(disp)

        if not sp_chunks:
            raise ValueError(f"No kinematic samples at lead {lead}")
        speeds = np.concatenate(sp_chunks)
        body = np.concatenate(body_chunks, axis=0)
        world = np.concatenate(world_chunks, axis=0)
        if len(speeds) > MAX_FIT_SAMPLES:
            sel = rng.choice(len(speeds), MAX_FIT_SAMPLES, replace=False)
            speeds, body, world = speeds[sel], body[sel], world[sel]
        if len(speeds) < 100:
            raise ValueError(f"Only {len(speeds)} kinematic samples at lead {lead}")

        if stop_model == "fitted":
            grid, mug, covg, scov, wgrid, trace = _fit_mixture_em(
                speeds, body, world, stop_speed, transition_band, speed_kernel_bw,
                n_speed_knots, em_iters,
            )
            speed_grid[lead], mu_grid[lead], cov_grid[lead] = grid, mug, covg
            stop_cov[lead], move_weight_grid[lead], em_loglik[lead] = scov, wgrid, trace
            marginal_cov[lead] = regularize_cov(np.cov(body.T))
            cv_covs[lead] = fit_cv_residual_cov(arrays, calib_ids, lead, dt)
            n_samples[lead] = int(len(speeds))
            continue

        moving = speeds >= stop_speed
        stopped_world = world[~moving]
        if len(stopped_world) >= 20:
            scov = (stopped_world.T @ stopped_world) / len(stopped_world)
        else:
            scov = (world.T @ world) / len(world)
        stop_cov[lead] = regularize_cov(scov)

        sm_sp, sm_body = speeds[moving], body[moving]
        if len(sm_sp) < 50:
            sm_sp, sm_body = speeds, body
        lo = float(max(stop_speed, np.quantile(sm_sp, 0.01)))
        hi = float(np.quantile(sm_sp, 0.99))
        if hi <= lo:
            hi = lo + 1e-3
        grid = np.linspace(lo, hi, n_speed_knots)
        mug = np.empty((n_speed_knots, 2))
        covg = np.empty((n_speed_knots, 2, 2))
        for gi, sg in enumerate(grid):
            wts = np.exp(-0.5 * ((sm_sp - sg) / max(speed_kernel_bw, 1e-6)) ** 2)
            if wts.sum() < 1e-6:
                wts = np.ones_like(sm_sp)
            mu, cov = _weighted_moments(sm_body, wts)
            mug[gi] = mu
            covg[gi] = regularize_cov(cov)

        speed_grid[lead] = grid
        mu_grid[lead] = mug
        cov_grid[lead] = covg
        marginal_cov[lead] = regularize_cov(np.cov(body.T))
        cv_covs[lead] = fit_cv_residual_cov(arrays, calib_ids, lead, dt)
        n_samples[lead] = int(len(speeds))

    return KinematicClimatology(
        leads=leads_t, dim=2, dt=dt, velocity_window=w, stop_speed=float(stop_speed),
        speed_kernel_bw=float(speed_kernel_bw), transition_band=float(transition_band),
        speed_grid=speed_grid, mu_grid=mu_grid, cov_grid=cov_grid, stop_cov=stop_cov,
        marginal_cov=marginal_cov, cv_covs=cv_covs, n_samples=n_samples,
        stop_model=stop_model, move_weight_grid=move_weight_grid, em_loglik=em_loglik,
        n_calib_tracks=len(list(calib_ids)),
    )


def _fit_mixture_em(speeds, body, world, stop_speed, transition_band, bw, n_knots, iters):
    """Best of several EM starts (mixtures have local optima; keep the highest likelihood)."""
    starts = [(stop_speed, transition_band), (0.1, 0.1), (0.4, 0.3), (0.05, 0.6)]
    fits = [
        _fit_mixture_em_once(speeds, body, world, s0, b0, bw, n_knots, iters) for s0, b0 in starts
    ]
    return max(fits, key=lambda f: f[-1][-1])


def _fit_mixture_em_once(speeds, body, world, stop_speed, transition_band, bw, n_knots, iters):
    """Stopped/moving mixture of experts over speed, by EM.

    p(d | s) = (1 - pi(s)) N(d_world; 0, S_stop) + pi(s) N(d_body; mu(s), S(s)).
    E-step: responsibilities. M-step: S_stop from stopped responsibilities; mu(s),
    S(s) and pi(s) as kernel-smoothed (bandwidth ``bw``) weighted moments on a speed
    grid spanning all speeds, interpolated between knots. Initialised from the ramp.
    """
    n_knots = max(int(n_knots), 24)
    grid = np.linspace(0.0, float(np.quantile(speeds, 0.995)), n_knots)
    K = np.exp(-0.5 * ((speeds[None, :] - grid[:, None]) / max(bw, 1e-6)) ** 2)  # (G, N)
    K_sum = np.maximum(K.sum(axis=1), 1e-12)
    ramp = np.clip((speeds - stop_speed) / max(transition_band, 1e-9), 0.0, 1.0)
    r_move = np.clip(ramp, 0.02, 0.98)
    mug = np.zeros((n_knots, 2))
    covg = np.zeros((n_knots, 2, 2))
    trace = []

    def interp(values):
        return np.stack([np.interp(speeds, grid, values[:, j]) for j in range(values.shape[1])], 1)

    for _ in range(int(iters)):
        r_stop = 1.0 - r_move
        scov = regularize_cov((r_stop[:, None] * world).T @ world / max(r_stop.sum(), 1e-12))
        wgrid = np.clip((K * r_move[None, :]).sum(axis=1) / K_sum, 1e-4, 1 - 1e-4)
        for g in range(n_knots):
            wts = K[g] * r_move
            if wts.sum() < 1e-8:
                wts = K[g]
            mu, cov = _weighted_moments(body, wts)
            mug[g] = mu
            covg[g] = regularize_cov(cov)
        pi = np.interp(speeds, grid, wgrid)
        mu_i = interp(mug)
        cov_i = interp(covg.reshape(n_knots, 4)).reshape(-1, 2, 2)
        lp_move = np.log(pi) - gaussian_nll(body - mu_i, cov_i)
        lp_stop = np.log(1.0 - pi) - gaussian_nll(world, scov)
        peak = np.maximum(lp_move, lp_stop)
        ll = peak + np.log(np.exp(lp_move - peak) + np.exp(lp_stop - peak))
        trace.append(float(ll.mean()))
        r_move = np.exp(lp_move - ll)
        if len(trace) > 5 and abs(trace[-1] - trace[-2]) < 1e-6:
            break
    return grid, mug.copy(), covg.copy(), scov, wgrid, trace


def _fit_mixture_em_avg(steps, lats, world, prior_grid, prior_pdf, n_nodes, w, dt, stop_speed,
                        transition_band, bw, n_knots, iters):
    """Best of several EM starts on the rounding-averaged input (highest final likelihood)."""
    omega, speed, heading = _expand_nodes(steps, lats, prior_grid, prior_pdf, n_nodes, w, dt)
    body = _to_body(world, heading)
    starts = [(stop_speed, transition_band), (0.1, 0.1), (0.4, 0.3), (0.05, 0.6)]
    fits = [
        _fit_mixture_em_avg_once(omega, speed, body, world, s0, b0, bw, n_knots, iters)
        for s0, b0 in starts
    ]
    # label identification: the stopped regime must own zero speed (pi(0) < 0.5); without this the
    # averaged fit can converge to a degenerate solution (broad "stopped" catch-all, pi ~ 0.9)
    # of nearly equal likelihood at short leads.
    ok = [f for f in fits if f[4][0] < 0.5] or fits
    return max(ok, key=lambda f: f[-1][-1])


def _fit_mixture_em_avg_once(omega, speed, body, world, stop_speed, transition_band, bw, n_knots,
                             iters):
    """Stopped/moving mixture of experts by EM when each frame is a mixture over quadrature nodes.

    p(d | input) = sum_j omega_j [(1 - pi(s_j)) N(d_world; 0, S_stop) + pi(s_j) N(d_body_j; mu(s_j), S(s_j))].
    E-step: joint responsibility of (node, regime). M-step as in ``_fit_mixture_em_once`` with
    each (frame, node) row carrying its responsibility as weight.
    """
    n, mm = omega.shape
    sp = speed.ravel()
    bd = body.reshape(-1, 2)
    n_knots = max(int(n_knots), 24)
    grid = np.linspace(0.0, float(np.quantile(sp, 0.995)), n_knots)
    kern = np.exp(-0.5 * ((sp[None, :] - grid[:, None]) / max(bw, 1e-6)) ** 2)
    ramp = np.clip((sp - stop_speed) / max(transition_band, 1e-9), 0.0, 1.0)
    rm = np.clip(ramp, 0.02, 0.98)
    rho_m = omega.ravel() * rm
    rho_s = omega.ravel() * (1.0 - rm)
    feats = np.stack([np.ones(len(sp)), bd[:, 0], bd[:, 1], bd[:, 0] ** 2, bd[:, 0] * bd[:, 1],
                      bd[:, 1] ** 2], axis=1)
    lo = np.log(omega.ravel() + 1e-300)
    trace = []
    mug = np.zeros((n_knots, 2))
    covg = np.zeros((n_knots, 2, 2))

    def interp(values):
        return np.stack([np.interp(sp, grid, values[:, j]) for j in range(values.shape[1])], 1)

    for _ in range(int(iters)):
        rs_frame = rho_s.reshape(n, mm).sum(axis=1)
        scov = regularize_cov((rs_frame[:, None] * world).T @ world / max(rs_frame.sum(), 1e-12))
        wgrid = np.clip((kern @ rho_m) / np.maximum(kern @ (rho_m + rho_s), 1e-12), 1e-4, 1 - 1e-4)
        wk = kern * rho_m[None, :]
        tot = wk.sum(axis=1)
        weak = tot < 1e-8
        if weak.any():
            wk[weak] = kern[weak]
            tot = wk.sum(axis=1)
        m = wk @ feats / tot[:, None]  # (G, 6): E1, Ex, Ey, Exx, Exy, Eyy
        mug = m[:, 1:3]
        covg = np.empty((n_knots, 2, 2))
        covg[:, 0, 0] = m[:, 3] - m[:, 1] ** 2
        covg[:, 0, 1] = covg[:, 1, 0] = m[:, 4] - m[:, 1] * m[:, 2]
        covg[:, 1, 1] = m[:, 5] - m[:, 2] ** 2
        covg = np.stack([regularize_cov(c) for c in covg])
        pi = np.interp(sp, grid, wgrid)
        mu_i = interp(mug)
        cov_i = interp(covg.reshape(n_knots, 4)).reshape(-1, 2, 2)
        lp_m = lo + np.log(pi) - gaussian_nll(bd - mu_i, cov_i)
        nll_s = gaussian_nll(world, scov)
        lp_s = lo + np.log(1.0 - pi) - np.repeat(nll_s, mm)
        both = np.stack([lp_m.reshape(n, mm), lp_s.reshape(n, mm)], axis=0)
        peak = both.max(axis=(0, 2))
        ll = peak + np.log(np.exp(both - peak[None, :, None]).sum(axis=(0, 2)))
        trace.append(float(ll.mean()))
        rho_m = np.exp(lp_m - np.repeat(ll, mm))
        rho_s = np.exp(lp_s - np.repeat(ll, mm))
        if len(trace) > 5 and abs(trace[-1] - trace[-2]) < 1e-6:
            break
    return grid, mug.copy(), covg.copy(), scov, wgrid, trace
