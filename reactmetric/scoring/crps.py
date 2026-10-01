"""Calibration-robust scoring: CRPS and the energy score.

The log score is calibration-sensitive by design (that is the point), but some
users cannot supply trustworthy covariances. CRPS / energy score are strictly
proper but far gentler on covariance misspecification, so REACT offers them as an
opt-in variant via ``scoring=``.

Both the model and the reference are scored with the SAME rule on a per-frame
Gaussian predictive (mean + covariance), so the information score s = score_ref -
score_model remains a like-for-like comparison. (For mixture references the
per-frame conditional Gaussian / moment-matched Gaussian is used; this is the
documented behaviour of the robust variant.)

All scores are negatively oriented (lower is better), matching the NLL sign
convention so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def gaussian_crps(resid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Sum of per-axis univariate Gaussian CRPS for zero-mean residuals.

    CRPS(N(0, sigma^2), r) = sigma * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)),
    z = r / sigma. Summed over independent axes (uses marginal variances).
    """
    resid = np.atleast_2d(np.asarray(resid, dtype=float))
    cov = np.asarray(cov, dtype=float)
    n, dim = resid.shape
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (n, dim, dim))
    var = np.einsum("nii->ni", cov)
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    z = resid / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - INV_SQRT_PI)
    return crps.sum(axis=1)


def gaussian_energy_score(
    resid: np.ndarray, cov: np.ndarray, n_samples: int = 100, seed: int = 0
) -> np.ndarray:
    """Monte-Carlo energy score ES = E||X - y|| - 0.5 E||X - X'|| per row.

    X ~ N(mean, cov); here residuals are y - mean, so X is sampled as N(0, cov).
    """
    resid = np.atleast_2d(np.asarray(resid, dtype=float))
    cov = np.asarray(cov, dtype=float)
    n, dim = resid.shape
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (n, dim, dim))
    rng = np.random.default_rng(seed)
    reg = cov + np.eye(dim) * 1e-9
    try:
        chol = np.linalg.cholesky(reg)                       # (n, D, D)
    except np.linalg.LinAlgError:
        chol = np.zeros_like(reg)
        for i in range(n):
            try:
                chol[i] = np.linalg.cholesky(reg[i])
            except np.linalg.LinAlgError:
                chol[i] = np.diag(np.sqrt(np.diag(reg[i])))
    eps = rng.standard_normal((n, n_samples, dim))
    draws = np.einsum("nde,nse->nsd", chol, eps)             # (n, S, D), zero-mean
    term1 = np.linalg.norm(draws - resid[:, None, :], axis=2).mean(axis=1)
    perm = rng.permutation(n_samples)
    term2 = 0.5 * np.linalg.norm(draws - draws[:, perm, :], axis=2).mean(axis=1)
    return term1 - term2


def gaussian_score(
    resid: np.ndarray, cov: np.ndarray, scoring: str, seed: int = 0
) -> np.ndarray:
    """Per-row score under the requested rule (lower is better)."""
    if scoring == "nll":
        from .nll import gaussian_nll

        return gaussian_nll(resid, cov)
    if scoring == "crps":
        return gaussian_crps(resid, cov)
    if scoring == "energy":
        return gaussian_energy_score(resid, cov, seed=seed)
    raise ValueError(f"unknown scoring rule '{scoring}'")


def predictive_score_matrix(pred, observations: np.ndarray, scoring: str) -> np.ndarray:
    """Per-(issue, step) score of realized observations under a Predictive."""
    observations = np.asarray(observations, dtype=float)
    n, hh, dim = pred.means.shape
    t = len(observations)
    targets = pred.issue_frames[:, None] + np.arange(hh)[None, :]
    valid = targets < t
    targets_clipped = np.minimum(targets, t - 1)
    diff = observations[targets_clipped] - pred.means
    flat = gaussian_score(diff.reshape(-1, dim), pred.covs.reshape(-1, dim, dim), scoring)
    return np.where(valid, flat.reshape(n, hh), np.nan)
