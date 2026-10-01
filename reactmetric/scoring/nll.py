"""Log-score (negative log-likelihood) primitives, generalized to N dimensions.

The internal pedestrian pipeline hardcoded closed-form 2x2 covariance inverses
everywhere. Here the same math is written for arbitrary state dimension D with
a fast path for D == 2 (the dominant case), so REACT works on 1D series, 2D
floor positions, 3D poses, or anything else.

A single Gaussian NLL evaluator (`gaussian_nll`) and a frozen Gaussian mixture
(`GaussianMixtureND`) cover every density REACT scores against, including the
mixture posteriors produced by the climatology-seeded oracle.
"""

from __future__ import annotations

import numpy as np

from ..constants import COV_REGULARIZATION

LOG_2PI = float(np.log(2.0 * np.pi))


def _solve_quadratic_2d(resid: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Mahalanobis distance and log-det for batched 2x2 covs."""
    a = cov[:, 0, 0] + COV_REGULARIZATION
    b = cov[:, 0, 1]
    c = cov[:, 1, 0]
    d = cov[:, 1, 1] + COV_REGULARIZATION
    det = a * d - b * c
    dx, dy = resid[:, 0], resid[:, 1]
    mahal = (d * dx * dx - (b + c) * dx * dy + a * dy * dy) / det
    return mahal, np.log(det)


def gaussian_nll(resid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """NLL of zero-mean residuals under a (shared or per-row) covariance.

    Args:
        resid: (N, D) residuals (realized minus predicted).
        cov: (D, D) shared covariance, or (N, D, D) per-row covariances.

    Returns:
        (N,) negative log-likelihoods in nats.
    """
    resid = np.atleast_2d(np.asarray(resid, dtype=float))
    cov = np.asarray(cov, dtype=float)
    n, dim = resid.shape
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (n, dim, dim))

    if dim == 2:
        mahal, logdet = _solve_quadratic_2d(resid, cov)
        return 0.5 * mahal + 0.5 * logdet + dim * 0.5 * LOG_2PI

    # General D: regularize, batched solve + slogdet.
    reg = cov + np.eye(dim) * COV_REGULARIZATION
    sign, logdet = np.linalg.slogdet(reg)
    sol = np.linalg.solve(reg, resid[:, :, None])[:, :, 0]   # (N, D)
    mahal = np.einsum("nd,nd->n", resid, sol)
    bad = sign <= 0
    out = 0.5 * mahal + 0.5 * logdet + dim * 0.5 * LOG_2PI
    out[bad] = np.nan
    return out


def eps_star_from_cov(cov: np.ndarray, d_meters: float) -> float:
    """Necessary information (nats) to localize at RMS error ``d`` under N(., cov).

    Gaussian rate-distortion R(D) = 0.5 * log(det(Sigma) / det(Sigma_D)) with the
    isotropic target Sigma_D = (d^2 / D) I so that E||err||^2 = d^2. Clamped at 0.
    """
    cov = np.asarray(cov, dtype=float)
    dim = cov.shape[0]
    det_clim = float(np.linalg.det(cov))
    det_task = (d_meters**2 / dim) ** dim
    if det_task <= 0:
        return 0.0
    return max(0.0, 0.5 * np.log(det_clim / det_task))


def supported_distortion_from_cov(cov: np.ndarray) -> float:
    """RMS localization scale where a N(., cov) reference needs zero nats."""
    cov = np.asarray(cov, dtype=float)
    dim = cov.shape[0]
    det_clim = float(np.linalg.det(cov))
    return float(np.sqrt(dim * det_clim ** (1.0 / dim)))


class GaussianMixtureND:
    """Frozen D-dimensional Gaussian mixture with vectorized log-pdf.

    Parameters are plain arrays so the object pickles without sklearn. A fast
    path for D == 2 mirrors the internal closed-form evaluator; higher D uses
    batched linear algebra.
    """

    __slots__ = ("weights", "means", "covs", "_inv", "_log_norm", "dim")

    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray):
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covs = np.asarray(covs, dtype=float)
        self.dim = self.means.shape[1]
        self._precompute()

    def _precompute(self) -> None:
        cov = self.covs.copy()
        idx = np.arange(self.dim)
        cov[:, idx, idx] += COV_REGULARIZATION
        self._inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        self._log_norm = -0.5 * logdet - self.dim * 0.5 * LOG_2PI + np.log(
            np.clip(self.weights, 1e-300, None)
        )

    def __getstate__(self):
        return {"weights": self.weights, "means": self.means, "covs": self.covs}

    def __setstate__(self, state):
        self.weights = state["weights"]
        self.means = state["means"]
        self.covs = state["covs"]
        self.dim = self.means.shape[1]
        self._precompute()

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Vectorized mixture log-density. x: (N, D) -> (N,)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        diff = x[:, None, :] - self.means[None, :, :]            # (N, C, D)
        m = np.einsum("ncj,cjk,nck->nc", diff, self._inv, diff)  # (N, C)
        comp = self._log_norm[None, :] - 0.5 * m
        peak = comp.max(axis=1, keepdims=True)
        return peak[:, 0] + np.log(np.exp(comp - peak).sum(axis=1))

    def nll(self, x: np.ndarray) -> np.ndarray:
        return -self.log_pdf(x)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(len(self.weights), size=int(n), p=self.weights)
        out = np.empty((int(n), self.dim), dtype=float)
        for j in range(len(self.weights)):
            mask = comp == j
            if np.any(mask):
                out[mask] = rng.multivariate_normal(
                    self.means[j], self.covs[j], size=int(np.sum(mask))
                )
        return out

    @property
    def mixture_mean(self) -> np.ndarray:
        return self.weights @ self.means

    @property
    def mixture_covariance(self) -> np.ndarray:
        """Law of total variance over components."""
        mu = self.mixture_mean
        cov = np.zeros((self.dim, self.dim))
        for w, m, c in zip(self.weights, self.means, self.covs):
            d = (m - mu)[:, None]
            cov += w * (c + d @ d.T)
        return cov


def fuse_with_gaussian(
    prior: GaussianMixtureND, mu: np.ndarray, cov: np.ndarray
) -> GaussianMixtureND:
    """Gaussian-sum Bayesian update of a mixture prior with one Gaussian observation.

    Treats ``prior`` (over the displacement) as the prior and N(mu, cov) -- a
    post-change constant-velocity displacement estimate -- as a soft observation.
    The posterior is again a mixture:

        Sigma_j' = (Sigma_j^-1 + cov^-1)^-1
        mu_j'    = Sigma_j' (Sigma_j^-1 mu_j + cov^-1 mu)
        w_j'     proportional to w_j * N(mu_j; mu, Sigma_j + cov)

    As cov -> inf the update is inert (posterior == prior), so a forecast with no
    usable post-change information collapses to the prior exactly and its
    information score is identically zero.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    dim = prior.dim
    if not np.all(np.isfinite(cov)) or float(np.linalg.det(cov)) > 1e12:
        return prior
    reg = np.eye(dim) * COV_REGULARIZATION
    inv_obs = np.linalg.inv(cov + reg)
    c = len(prior.weights)
    new_means = np.empty((c, dim))
    new_covs = np.empty((c, dim, dim))
    log_w = np.empty(c)
    for j in range(c):
        sigma_j = prior.covs[j] + reg
        inv_j = np.linalg.inv(sigma_j)
        sigma_post = np.linalg.inv(inv_j + inv_obs)
        new_means[j] = sigma_post @ (inv_j @ prior.means[j] + inv_obs @ mu)
        new_covs[j] = sigma_post
        s = sigma_j + cov
        diff = prior.means[j] - mu
        _, logdet = np.linalg.slogdet(s)
        mahal = float(diff @ np.linalg.solve(s, diff))
        log_w[j] = np.log(max(prior.weights[j], 1e-300)) - 0.5 * (logdet + mahal)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return GaussianMixtureND(weights=w, means=new_means, covs=new_covs)
