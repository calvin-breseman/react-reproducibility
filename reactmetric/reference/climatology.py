"""Global (unconditional) displacement climatology.

p(dx | lead): the marginal law of lead-step displacements over the calibration
tracks, as a Gaussian mixture with the fixed floor component. Event-insensitive
(it conditions on nothing the maneuver changes), so it is the natural reference
for commitment-event detection.
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
    supported_distortion_from_cov,
)
from .base import (
    cv_nll_trace_from_cov,
    displacement_samples,
    fit_cv_residual_cov,
    fit_gmm,
    with_floor,
)


@dataclass
class GlobalClimatology:
    """Per-lead displacement mixtures + CV residual reference + metadata."""

    leads: Tuple[int, ...]
    dim: int
    dt: float
    models: Dict[int, GaussianMixtureND]
    cv_covs: Dict[int, np.ndarray]
    n_samples: Dict[int, int] = field(default_factory=dict)
    k: int = 0
    n_calib_tracks: int = 0

    def nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = traj.positions if hasattr(traj, "positions") else np.asarray(traj)
        t = len(obs)
        if t <= lead:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead, t)
        disp = obs[lead:] - obs[:-lead]
        nll = self.models[lead].nll(disp)
        ok = np.isfinite(nll)
        return frames[ok], nll[ok]

    def cv_nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = traj.positions if hasattr(traj, "positions") else np.asarray(traj)
        return cv_nll_trace_from_cov(obs, lead, self.dt, self.cv_covs[lead])

    def score_trace(self, traj, lead: int, scoring: str = "nll"):
        if scoring == "nll":
            return self.nll_trace(traj, lead)
        from ..scoring.crps import gaussian_score

        obs = traj.positions if hasattr(traj, "positions") else np.asarray(traj)
        t = len(obs)
        if t <= lead:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead, t)
        gm = self.models[lead]
        mu = obs[:-lead] + gm.mixture_mean
        resid = obs[lead:] - mu
        score = gaussian_score(resid, gm.mixture_covariance, scoring)
        ok = np.isfinite(score)
        return frames[ok], score[ok]

    def displacement_prior(self, lead: int, speed: float, heading: float) -> GaussianMixtureND:
        return self.models[lead]

    def cv_residual_cov(self, lead: int) -> np.ndarray:
        return self.cv_covs.get(lead, np.zeros((self.dim, self.dim)))

    def eps_star(self, lead: int, d_meters: float) -> float:
        return eps_star_from_cov(self.models[lead].mixture_covariance, d_meters)

    def supported_distortion(self, lead: int) -> float:
        return supported_distortion_from_cov(self.models[lead].mixture_covariance)

    def metadata(self) -> Dict:
        out = {
            "mode": "global",
            "dim": self.dim,
            "k": self.k,
            "n_calib_tracks": self.n_calib_tracks,
            "leads": list(self.leads),
            "per_lead": {},
        }
        for lead in self.leads:
            cov = self.models[lead].mixture_covariance
            out["per_lead"][str(lead)] = {
                "n_samples": int(self.n_samples.get(lead, 0)),
                "mixture_covariance": cov.tolist(),
                "differential_entropy_gaussian_nats": float(
                    0.5 * self.dim * (1.0 + LOG_2PI) + 0.5 * np.log(np.linalg.det(cov))
                ),
                "cv_residual_covariance": self.cv_covs[lead].tolist(),
            }
        return out


def fit_global_climatology(
    arrays: Dict[object, np.ndarray],
    calib_ids: Sequence[object],
    leads: Sequence[int],
    dt: float = DEFAULT_DT,
    k: int = 4,
    seed: int = 0,
) -> GlobalClimatology:
    models: Dict[int, GaussianMixtureND] = {}
    cv_covs: Dict[int, np.ndarray] = {}
    n_samples: Dict[int, int] = {}
    leads_t = tuple(sorted({int(l) for l in leads}))
    dim = next(iter(arrays.values())).shape[1]

    for lead in leads_t:
        samples = displacement_samples(arrays, calib_ids, lead, seed=seed)
        if len(samples) < 100:
            raise ValueError(
                f"Only {len(samples)} displacement samples at lead {lead}; "
                f"cannot fit climatology"
            )
        gmm = fit_gmm(samples, k=k, seed=seed)
        models[lead] = with_floor(gmm, lead, dt)
        n_samples[lead] = len(samples)
        cv_covs[lead] = fit_cv_residual_cov(arrays, calib_ids, lead, dt)

    return GlobalClimatology(
        leads=leads_t, dim=dim, dt=dt, models=models, cv_covs=cv_covs,
        n_samples=n_samples, k=k, n_calib_tracks=len(list(calib_ids)),
    )
