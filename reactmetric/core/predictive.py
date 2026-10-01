"""Predictive: a forecaster's per-track predictive distributions over the future.

For issue frame ``issue_frames[i]``, ``means[i, h]`` / ``covs[i, h]`` describe the
predictive distribution of the position at frame ``issue_frames[i] + h`` (lead
h+1, h = 0..H-1). This is the ND generalization of the internal TrackForecasts.

`gaussian` is the direct port (you supply means + covariances). `isotropic` is a
diagonal shortcut. `from_samples` is the new ingestion path for ensemble /
multimodal / K-sampled-trajectory forecasters: samples are converted to a
Gaussian predictive (moment matching, optionally KDE-smoothed) so they ride the
same fast log-score path as everything else.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..scoring.nll import gaussian_nll


class Predictive:
    """Gaussian predictive distributions for one track from one forecaster."""

    __slots__ = ("issue_frames", "means", "covs", "horizon")

    def __init__(
        self,
        issue_frames: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        horizon: Optional[int] = None,
    ):
        self.issue_frames = np.asarray(issue_frames, dtype=int)
        self.means = np.asarray(means, dtype=float)
        self.covs = np.asarray(covs, dtype=float)
        if self.means.ndim != 3:
            raise ValueError(f"means must be (n, H, D); got {self.means.shape}")
        if self.covs.ndim != 4:
            raise ValueError(f"covs must be (n, H, D, D); got {self.covs.shape}")
        self.horizon = int(horizon if horizon is not None else self.means.shape[1])

    # ---- constructors -----------------------------------------------------

    @classmethod
    def gaussian(
        cls, issue_frames: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> Predictive:
        """Direct constructor from predictive means (n, H, D) and covs (n, H, D, D)."""
        return cls(issue_frames, means, covs)

    @classmethod
    def isotropic(
        cls, issue_frames: np.ndarray, means: np.ndarray, sigma
    ) -> Predictive:
        """Diagonal/isotropic shortcut.

        ``sigma`` may be a scalar, a per-dim vector (D,), or a per-(issue, lead)
        array broadcastable to (n, H) or (n, H, D).
        """
        means = np.asarray(means, dtype=float)
        n, h, dim = means.shape
        sigma = np.asarray(sigma, dtype=float)
        var = np.empty((n, h, dim))
        if sigma.ndim == 0:
            var[...] = sigma**2
        elif sigma.shape == (dim,):
            var[...] = (sigma**2)[None, None, :]
        elif sigma.shape == (n, h):
            var[...] = (sigma**2)[:, :, None]
        elif sigma.shape == (n, h, dim):
            var[...] = sigma**2
        else:
            raise ValueError(f"sigma shape {sigma.shape} not broadcastable to {means.shape}")
        covs = np.zeros((n, h, dim, dim))
        idx = np.arange(dim)
        covs[:, :, idx, idx] = var
        return cls(issue_frames, means, covs)

    @classmethod
    def from_samples(
        cls,
        issue_frames: np.ndarray,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
        method: str = "moment",
    ) -> Predictive:
        """Build a Gaussian predictive from samples (n, H, S, D).

        ``method='moment'`` moment-matches each (issue, lead) sample cloud to a
        Gaussian. ``method='kde'`` adds a Silverman-rule bandwidth to the
        covariance, i.e. a kernel-smoothed (and slightly more conservative)
        density. ``weights`` (n, H, S) optionally weights samples (e.g. mode
        probabilities for K-trajectory forecasts).
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 4:
            raise ValueError(f"samples must be (n, H, S, D); got {samples.shape}")
        n, h, s, dim = samples.shape
        if weights is None:
            w = np.full((n, h, s), 1.0 / s)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / np.clip(w.sum(axis=2, keepdims=True), 1e-12, None)
        means = np.einsum("nhs,nhsd->nhd", w, samples)
        diff = samples - means[:, :, None, :]
        covs = np.einsum("nhs,nhsd,nhse->nhde", w, diff, diff)
        if method == "kde":
            # Silverman bandwidth factor per effective sample size, isotropic.
            factor = (4.0 / (dim + 2.0)) ** (1.0 / (dim + 4.0)) * s ** (-1.0 / (dim + 4.0))
            idx = np.arange(dim)
            var_diag = covs[:, :, idx, idx]
            covs = covs.copy()
            covs[:, :, idx, idx] = var_diag * (1.0 + factor**2)
        elif method != "moment":
            raise ValueError(f"unknown from_samples method '{method}'")
        return cls(issue_frames, means, covs)

    # ---- properties -------------------------------------------------------

    @property
    def D(self) -> int:
        return self.means.shape[2]

    def __len__(self) -> int:
        return len(self.issue_frames)

    def at_lead(self, lead: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(target_frames, means, covs) for the lead-step forecast (1-indexed)."""
        h = int(lead)
        if not 1 <= h <= self.horizon:
            raise ValueError(f"lead {h} outside horizon 1..{self.horizon}")
        target = self.issue_frames + (h - 1)
        return target, self.means[:, h - 1], self.covs[:, h - 1]

    # ---- scoring ----------------------------------------------------------

    def nll_matrix(self, observations: np.ndarray) -> np.ndarray:
        """Per-(issue, step) Gaussian NLL of the realized observations, (n, H).

        NaN where the target frame is beyond the track or the covariance is not
        positive-definite.
        """
        observations = np.asarray(observations, dtype=float)
        n, hh, dim = self.means.shape
        t = len(observations)
        targets = self.issue_frames[:, None] + np.arange(hh)[None, :]
        valid = targets < t
        targets_clipped = np.minimum(targets, t - 1)
        diff = observations[targets_clipped] - self.means              # (n, H, D)
        flat_resid = diff.reshape(-1, dim)
        flat_cov = self.covs.reshape(-1, dim, dim)
        nll = gaussian_nll(flat_resid, flat_cov).reshape(n, hh)
        out = np.where(valid, nll, np.nan)
        return out

    def nll(self, traj, scoring: str = "nll") -> np.ndarray:
        """Per-(issue, step) score against a Trajectory (see scoring rules)."""
        observations = traj.positions if hasattr(traj, "positions") else np.asarray(traj)
        if scoring == "nll":
            return self.nll_matrix(observations)
        from ..scoring.crps import predictive_score_matrix

        return predictive_score_matrix(self, observations, scoring=scoring)


def collapse_nll_by_target(
    issue_frames: np.ndarray,
    nll: np.ndarray,
    lead: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse a per-(issue, step) NLL matrix to a target-frame trace.

    With ``lead=None``, trace[f] averages over all leads predicting frame f
    (defined only where all H leads exist, so the statistic is identically
    distributed across frames). With an integer lead h, trace[f] is the NLL of
    the h-step-ahead forecast of frame f (issued at f - h + 1).

    Target-frame indexing means a maneuver at t0 elevates the trace at t0, not H
    frames before any evidence exists.
    """
    n, hh = nll.shape
    if n == 0:
        return np.array([], dtype=int), np.array([])
    issue = np.asarray(issue_frames, dtype=int)
    first, last = int(issue[0]), int(issue[-1])

    if lead is not None:
        h = int(lead)
        if not 1 <= h <= hh:
            raise ValueError(f"lead {h} outside forecast horizon 1..{hh}")
        frames = issue + (h - 1)
        vals = nll[:, h - 1]
        ok = ~np.isnan(vals)
        return frames[ok], vals[ok]

    candidate_frames = np.arange(first + hh - 1, last + 1)
    if len(candidate_frames) == 0:
        return np.array([], dtype=int), np.array([])
    acc = np.zeros(len(candidate_frames))
    count = np.zeros(len(candidate_frames), dtype=int)
    for h in range(hh):
        rows = candidate_frames - h - first
        vals = nll[rows, h]
        ok = ~np.isnan(vals)
        acc[ok] += vals[ok]
        count[ok] += 1
    full = count == hh
    return candidate_frames[full], acc[full] / hh
