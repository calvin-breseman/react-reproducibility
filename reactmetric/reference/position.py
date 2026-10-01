"""Position-conditioned climatology (2D): cell mixtures shrunk to the global law.

Captures environment structure (people move differently in different places) but
is spatially data-hungry, so each cell mixture is shrunk toward the global
mixture by n / (n + n0). Rate-distortion anchors and the CV reference delegate to
the embedded global climatology, whose covariance defines the task thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..constants import DEFAULT_DT
from ..scoring.nll import GaussianMixtureND
from .base import fit_gmm, with_floor
from .climatology import GlobalClimatology, fit_global_climatology


@dataclass
class PositionClimatology:
    """Position-binned reference; a full Reference via delegation to global."""

    global_clim: GlobalClimatology
    cell_m: float
    min_cell_samples: int
    shrinkage_n0: float
    cells: Dict[Tuple[int, int, int], GaussianMixtureND]
    cell_counts: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    @property
    def leads(self) -> Tuple[int, ...]:
        return self.global_clim.leads

    @property
    def dim(self) -> int:
        return self.global_clim.dim

    @property
    def dt(self) -> float:
        return self.global_clim.dt

    def nll_trace(self, traj, lead: int) -> Tuple[np.ndarray, np.ndarray]:
        obs = traj.positions if hasattr(traj, "positions") else np.asarray(traj)
        t = len(obs)
        if t <= lead:
            return np.array([], dtype=int), np.array([])
        frames = np.arange(lead, t)
        anchors = obs[:-lead]
        disp = obs[lead:] - anchors
        glob = self.global_clim.models[lead]
        nll = np.empty(len(frames))
        ix = np.floor(anchors[:, 0] / self.cell_m).astype(int)
        iy = np.floor(anchors[:, 1] / self.cell_m).astype(int)
        for i in range(len(frames)):
            key = (lead, int(ix[i]), int(iy[i]))
            local = self.cells.get(key)
            g = glob.log_pdf(disp[i : i + 1])[0]
            if local is None:
                nll[i] = -g
            else:
                n = self.cell_counts[key]
                lam = n / (n + self.shrinkage_n0)
                l = local.log_pdf(disp[i : i + 1])[0]
                peak = max(l, g)
                nll[i] = -(peak + np.log(lam * np.exp(l - peak) + (1.0 - lam) * np.exp(g - peak)))
        ok = np.isfinite(nll)
        return frames[ok], nll[ok]

    def cv_nll_trace(self, traj, lead: int):
        return self.global_clim.cv_nll_trace(traj, lead)

    def score_trace(self, traj, lead: int, scoring: str = "nll"):
        if scoring == "nll":
            return self.nll_trace(traj, lead)
        return self.global_clim.score_trace(traj, lead, scoring)

    def displacement_prior(self, lead: int, speed: float, heading: float):
        return self.global_clim.displacement_prior(lead, speed, heading)

    def cv_residual_cov(self, lead: int) -> np.ndarray:
        return self.global_clim.cv_residual_cov(lead)

    def eps_star(self, lead: int, d_meters: float) -> float:
        return self.global_clim.eps_star(lead, d_meters)

    def supported_distortion(self, lead: int) -> float:
        return self.global_clim.supported_distortion(lead)

    def metadata(self) -> Dict:
        out = dict(self.global_clim.metadata())
        out["mode"] = "position"
        out["cell_m"] = self.cell_m
        out["min_cell_samples"] = self.min_cell_samples
        out["shrinkage_n0"] = self.shrinkage_n0
        out["n_cells"] = len(self.cells)
        return out


def fit_position_climatology(
    arrays: Dict[object, np.ndarray],
    calib_ids: Sequence[object],
    leads: Sequence[int],
    dt: float = DEFAULT_DT,
    k: int = 4,
    seed: int = 0,
    cell_m: float = 2.0,
    min_cell_samples: int = 500,
    shrinkage_n0: float = 1000.0,
) -> PositionClimatology:
    global_clim = fit_global_climatology(arrays, calib_ids, leads, dt=dt, k=k, seed=seed)
    cells: Dict[Tuple[int, int, int], GaussianMixtureND] = {}
    counts: Dict[Tuple[int, int, int], int] = {}
    for lead in global_clim.leads:
        buckets: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for tid in calib_ids:
            obs = arrays[tid]
            if len(obs) <= lead:
                continue
            anchors = obs[:-lead]
            disp = obs[lead:] - anchors
            ix = np.floor(anchors[:, 0] / cell_m).astype(int)
            iy = np.floor(anchors[:, 1] / cell_m).astype(int)
            for i in range(len(disp)):
                if np.isfinite(disp[i]).all():
                    buckets.setdefault((int(ix[i]), int(iy[i])), []).append(disp[i])
        for (cx, cy), rows in buckets.items():
            if len(rows) < min_cell_samples:
                continue
            samples = np.array(rows)
            kk = min(k, max(1, len(samples) // 200))
            gmm = fit_gmm(samples, k=kk, seed=seed)
            cells[(lead, cx, cy)] = with_floor(gmm, lead, dt)
            counts[(lead, cx, cy)] = len(samples)
    return PositionClimatology(
        global_clim=global_clim, cell_m=cell_m, min_cell_samples=min_cell_samples,
        shrinkage_n0=shrinkage_n0, cells=cells, cell_counts=counts,
    )
