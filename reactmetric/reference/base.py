"""The reference (climatology) protocol and shared fitting helpers.

The reference is the "knowing nothing about this person beyond population
statistics" density REACT scores against. The information score is

    s_m(f) = nll_ref(f) - nll_model(f)   (nats)

a pointwise log Bayes factor. The reference conditions ONLY on event-continuous
quantities (the anchor position and the lead, or a causal kinematic estimate) so
that it is not itself surprised away by the very maneuvers REACT measures.

Design invariants ported from the internal pipeline:
- A fixed broad floor component (excluded from EM) bounds the reference NLL above,
  so rare large displacements cannot explode it and mechanically flatter
  surprised heavy-tailed models.
- The persistence (constant-velocity) reference is provided as a secondary,
  "value-added over naive extrapolation" baseline.
"""

from __future__ import annotations

from typing import Dict, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..constants import COV_REGULARIZATION
from ..scoring.nll import GaussianMixtureND

# Floor component: fixed weight, never fitted by EM.
FLOOR_WEIGHT = 0.02
FLOOR_SIGMA_SPEED = 4.0   # m/s; floor sigma per lead = max(this * l * dt, FLOOR_SIGMA_MIN)
FLOOR_SIGMA_MIN = 0.5     # m

CV_VELOCITY_WINDOW = 5
MAX_FIT_SAMPLES = 200_000


@runtime_checkable
class Reference(Protocol):
    """Common interface REACT consumes from any climatology mode."""

    leads: Tuple[int, ...]
    dim: int

    def nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def cv_nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def displacement_prior(self, lead: int, speed: float, heading: float) -> GaussianMixtureND:
        ...

    def cv_residual_cov(self, lead: int) -> np.ndarray:
        ...

    def eps_star(self, lead: int, d_meters: float) -> float:
        ...

    def supported_distortion(self, lead: int) -> float:
        ...

    def metadata(self) -> Dict:
        ...


# ---------------------------------------------------------------------------
# Sampling + fitting helpers
# ---------------------------------------------------------------------------

def displacement_samples(
    arrays: Dict[object, np.ndarray],
    track_ids: Sequence[object],
    lead: int,
    max_samples: int = MAX_FIT_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Pooled lead-step displacements x[t + lead] - x[t] over the tracks."""
    chunks = []
    for tid in track_ids:
        obs = arrays[tid]
        if len(obs) <= lead:
            continue
        d = obs[lead:] - obs[:-lead]
        finite = np.isfinite(d).all(axis=1)
        if finite.any():
            chunks.append(d[finite])
    if not chunks:
        return np.empty((0, next(iter(arrays.values())).shape[1]))
    pooled = np.concatenate(chunks, axis=0)
    if len(pooled) > max_samples:
        rng = np.random.default_rng(seed)
        pooled = pooled[rng.choice(len(pooled), max_samples, replace=False)]
    return pooled


def fit_gmm(samples: np.ndarray, k: int, seed: int) -> GaussianMixtureND:
    """Fit a k-component full-covariance GMM with sklearn."""
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(
        n_components=k, covariance_type="full", reg_covar=1e-6, random_state=seed, n_init=2
    )
    gm.fit(samples)
    return GaussianMixtureND(weights=gm.weights_, means=gm.means_, covs=gm.covariances_)


def with_floor(gmm: GaussianMixtureND, lead: int, dt: float) -> GaussianMixtureND:
    """Append the fixed broad floor component to a fitted mixture."""
    dim = gmm.dim
    sigma = max(FLOOR_SIGMA_SPEED * lead * dt, FLOOR_SIGMA_MIN)
    weights = np.concatenate([gmm.weights * (1.0 - FLOOR_WEIGHT), [FLOOR_WEIGHT]])
    means = np.vstack([gmm.means, np.zeros((1, dim))])
    covs = np.concatenate([gmm.covs, (np.eye(dim) * sigma**2)[None]], axis=0)
    return GaussianMixtureND(weights=weights, means=means, covs=covs)


def fit_cv_residual_cov(
    arrays: Dict[object, np.ndarray],
    track_ids: Sequence[object],
    lead: int,
    dt: float,
    window: int = CV_VELOCITY_WINDOW,
    max_samples: int = MAX_FIT_SAMPLES,
) -> np.ndarray:
    """Empirical covariance of constant-velocity extrapolation residuals."""
    w = window
    chunks = []
    dim = next(iter(arrays.values())).shape[1]
    for tid in track_ids:
        obs = arrays[tid]
        t = len(obs)
        if t <= lead + w:
            continue
        anchors = obs[w : t - lead]
        vel = (obs[w : t - lead] - obs[: t - lead - w]) / (w * dt)
        resid = obs[lead + w :] - (anchors + vel * (lead * dt))
        finite = np.isfinite(resid).all(axis=1)
        if finite.any():
            chunks.append(resid[finite])
    if not chunks:
        return np.eye(dim) * COV_REGULARIZATION
    pooled = np.concatenate(chunks, axis=0)
    if len(pooled) > max_samples:
        pooled = pooled[:max_samples]
    cov = np.cov(pooled.T)
    return np.atleast_2d(cov)


def regularize_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float).copy()
    dim = cov.shape[0]
    idx = np.arange(dim)
    cov[idx, idx] += COV_REGULARIZATION
    if not np.all(np.isfinite(cov)) or float(np.linalg.det(cov)) <= 0.0:
        return np.eye(dim) * max(COV_REGULARIZATION, 1e-3)
    return cov


def cv_nll_trace_from_cov(
    obs: np.ndarray, lead: int, dt: float, cov: np.ndarray, window: int = CV_VELOCITY_WINDOW
) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity (persistence) reference NLL trace at the lead."""
    from ..scoring.nll import gaussian_nll

    t = len(obs)
    w = window
    if t <= lead + w:
        return np.array([], dtype=int), np.array([])
    frames = np.arange(lead + w, t)
    anchors = obs[w : t - lead]
    vel = (obs[w : t - lead] - obs[: t - lead - w]) / (w * dt)
    pred = anchors + vel * (lead * dt)
    resid = obs[lead + w :] - pred
    nll = gaussian_nll(resid, cov)
    ok = np.isfinite(nll)
    return frames[ok], nll[ok]
