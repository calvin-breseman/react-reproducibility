"""Trajectron++ as a REACT forecaster (kept outside the package, like the other experiment code).

Same interface as ``reactmetric.baselines.gru.GRUPosition``: ``predict(traj) -> Predictive``, and
``predict_mixture(traj)`` for the un-collapsed 25-component mixture. Needs a clone of
https://github.com/StanfordASL/Trajectron-plus-plus (``--tpp-dir`` / ``TPP_DIR``) and a trained
single-agent (edge_encoding false) model; the Eindhoven run-1 checkpoint (epoch 30) is stored in
``../artifacts/eindhoven/trajectron_run1`` (``--tpp-ckpt`` / ``TPP_CKPT`` pick another epoch).

Conventions (identical to the GRU baselines): ``issue_frames[i] = t + 1`` with t the last observed
frame; ``means[i, h-1]`` forecasts frame ``issue_frames[i] + h - 1`` (lead h); the model sees
frames t-10..t;
issue frames run from ``history + 1`` to ``T - horizon``. Tracks must be at 10 Hz, in metres.

Moment matching: ``predict`` collapses each mixture to its mean and covariance (analytically), which
penalises multimodal forecasts in log score. ``predict_mixture`` keeps all components. The
decoder is deterministic (``gmm_mode=True``), so predictions are reproducible.

Features follow the clone's experiments/pedestrians/process_data.py: state = [pos, vel, acc] at
dt = 0.1 s, velocity and acceleration by causal backward differences (``derivative_of``).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from typing import Optional

import numpy as np

from reactmetric.core.predictive import Predictive
from reactmetric.core.trajectory import Trajectory
from reactmetric.forecasters.base import BaseForecaster

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(HERE), "artifacts", "eindhoven", "trajectron_run1")
DT = 0.1
HT = 10  # history steps (states t-10..t are fed)
PH = 15  # prediction horizon (REACT leads 1..15)
ATTENTION_RADIUS = 1.0
STANDARDIZATION = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        # finite-difference acceleration is noisy (std ~ 4-5 m/s^2 at 10 Hz)
        "acceleration": {"x": {"mean": 0, "std": 5}, "y": {"mean": 0, "std": 5}},
    }
}
STD6 = np.array([ATTENTION_RADIUS, ATTENTION_RADIUS, 1.0, 1.0, 5.0, 5.0], dtype=np.float64)

_clone = {}


def _load_clone(tpp_dir: Optional[str] = None):
    """Put the clone's ``trajectron/`` directory on sys.path and import what we need."""
    if _clone:
        return
    tpp_dir = tpp_dir or os.environ.get("TPP_DIR")
    if not tpp_dir or not os.path.isdir(os.path.join(tpp_dir, "trajectron")):
        raise FileNotFoundError(
            "Trajectron++ clone not found: pass tpp_dir / --tpp-dir or set TPP_DIR "
            f"(got {tpp_dir!r})"
        )
    path = os.path.join(os.path.abspath(tpp_dir), "trajectron")
    if path not in sys.path:
        sys.path.insert(0, path)
    from environment import Environment, Scene, derivative_of

    _clone.update(Environment=Environment, Scene=Scene, derivative_of=derivative_of)


def _make_env():
    Environment, Scene = _clone["Environment"], _clone["Scene"]
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): ATTENTION_RADIUS}
    env.scenes = [Scene(timesteps=1, dt=DT, name="dummy")]  # carries dt = 0.1 for the integrator
    return env


def _state_array(positions, dt=DT):
    """(T, 2) metres -> (T, 6) [x, y, vx, vy, ax, ay], position centred on the first frame."""
    d = _clone["derivative_of"]
    p = np.asarray(positions, dtype=np.float64)
    p = p - p[0]
    x, y = p[:, 0], p[:, 1]
    vx, vy = d(x, dt), d(y, dt)
    ax, ay = d(vx, dt), d(vy, dt)
    return np.stack([x, y, vx, vy, ax, ay], axis=1)


def _featurize(P, idx, ht=HT, ph=PH, with_labels=False):
    """Vectorised get_node_timestep_data for single-node scenes: idx = last observed rows of P."""
    import torch

    idx = np.asarray(idx)
    x = P[idx[:, None] + np.arange(-ht, 1)[None, :]]  # (B, ht+1, 6)
    rel = np.zeros((len(idx), 1, 6))
    rel[:, 0, 0:2] = x[:, -1, 0:2]
    x_st = (x - rel) / STD6
    first = torch.zeros(len(idx), dtype=torch.long)
    x_t = torch.tensor(x, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    return first, x_t, None, x_st_t, None, None, None, None, None


def _latest_checkpoint(model_dir: str) -> int:
    eps = [
        int(m.group(1))
        for p in glob.glob(os.path.join(model_dir, "model_registrar-*.pt"))
        if (m := re.search(r"model_registrar-(\d+)\.pt$", p))
    ]
    if not eps:
        raise FileNotFoundError(f"no model_registrar-*.pt in {model_dir}")
    return max(eps)


class TrajectronForecaster(BaseForecaster):
    """Single-agent pedestrian Trajectron++ (dynamics integration, K=25 latent modes)."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        checkpoint: Optional[int] = None,
        tpp_dir: Optional[str] = None,
        batch_size: int = 512,
        threads: Optional[int] = None,
        name: str = "TrajectronPP",
        min_issue: Optional[int] = None,
    ):
        _load_clone(tpp_dir)
        import torch
        from model.model_registrar import ModelRegistrar
        from model.trajectron import Trajectron

        if threads is not None:
            torch.set_num_threads(int(threads))
        self._torch = torch
        self.name = name
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if checkpoint is None and os.environ.get("TPP_CKPT"):
            checkpoint = int(os.environ["TPP_CKPT"])
        self.model_dir = model_dir
        self.checkpoint = (
            int(checkpoint) if checkpoint is not None else _latest_checkpoint(model_dir)
        )
        self.batch_size = int(batch_size)
        with open(os.path.join(model_dir, "config.json")) as f:
            hp = json.load(f)
        if hp.get("edge_encoding", True):
            raise ValueError("this adapter supports single-agent (edge_encoding=false) models only")
        self.history = int(hp["maximum_history_length"])
        self.horizon = int(hp["prediction_horizon"])
        if (self.history, self.horizon) != (HT, PH):
            warnings.warn(
                f"model history/horizon ({self.history},{self.horizon}) != ({HT},{PH})",
                stacklevel=2,
            )
        self.min_issue = int(min_issue) if min_issue is not None else self.history + 1
        if self.min_issue < self.history + 1:
            raise ValueError("min_issue must be >= history + 1")
        reg = ModelRegistrar(model_dir, "cpu")
        reg.load_models(self.checkpoint)  # evaluate.py order: load, then build Trajectron
        stg = Trajectron(reg, hp, None, "cpu")
        stg.set_environment(_make_env())  # dummy scene carries dt = 0.1 for the integrator
        stg.set_annealing_params()
        self._nt = stg.env.NodeType.PEDESTRIAN
        self._model = stg.node_models_dict[self._nt]
        self._warned = set()

    # ------------------------------------------------------------------ core
    def _batches(self, traj: Trajectory):
        """Yield (issue (b,), log_pis (b,K), mus (b,H,K,2), covs (b,H,K,2,2)) chunks (world frame).
        Returns None (not a generator) if the track cannot be forecast."""
        if traj.D != 2:
            return None
        if abs(traj.dt - DT) > 0.02 and "dt" not in self._warned:
            self._warned.add("dt")
            warnings.warn(
                f"{self.name}: trajectory dt={traj.dt:.3f}s but the model was trained at {DT}s; "
                "resample to 10 Hz first (forecast steps are 0.1 s).",
                stacklevel=2,
            )
        t_obs = np.arange(
            self.min_issue - 1, traj.T - self.horizon
        )  # last observed frame; issue = t_obs + 1
        if len(t_obs) == 0:
            return None
        pos = np.asarray(traj.positions, dtype=float)
        P = _state_array(pos, DT)  # centred on frame 0 (translation invariant)
        origin = pos[0]
        torch = self._torch

        def gen():
            with torch.no_grad():
                for i in range(0, len(t_obs), self.batch_size):
                    tt = t_obs[i : i + self.batch_size]
                    b = _featurize(P, tt, ht=self.history, ph=self.horizon, with_labels=False)
                    d = self._model.predict_dist(b[1], b[3], b[0], self.horizon, gmm_mode=True)
                    yield (
                        tt + 1,
                        d.log_pis[0, :, 0, :]
                        .numpy()
                        .astype(np.float64),  # (B, K), constant over steps
                        d.mus[0].numpy().astype(np.float64) + origin,  # (B, H, K, 2)
                        d.get_covariance_matrix()[0].numpy().astype(np.float64),
                    )  # (B, H, K, 2, 2)

        return gen()

    def predict_mixture(self, traj: Trajectory):
        """Un-collapsed predictive: dict(issue_frames, weights (n,K), means (n,H,K,2),
        covs (n,H,K,2,2)).
        Memory is n*H*K*(2+4) doubles -- fine for one track, big for very long ones."""
        g = self._batches(traj)
        if g is None:
            return None
        parts = list(g)
        return dict(
            issue_frames=np.concatenate([p[0] for p in parts]),
            weights=np.exp(np.concatenate([p[1] for p in parts])),
            means=np.concatenate([p[2] for p in parts]),
            covs=np.concatenate([p[3] for p in parts]),
        )

    @staticmethod
    def collapse(lp, mu, cv):
        """Moment-match mixtures: lp (b,K) log weights, mu (b,H,K,2), cv (b,H,K,2,2)
        -> (b,H,2), (b,H,2,2)."""
        w = np.exp(lp)
        w = w / w.sum(-1, keepdims=True)  # (b, K)
        mean = np.einsum("bk,bhkd->bhd", w, mu)
        diff = mu - mean[:, :, None, :]
        cov = np.einsum("bk,bhkij->bhij", w, cv) + np.einsum("bk,bhki,bhkj->bhij", w, diff, diff)
        return mean, 0.5 * (cov + np.swapaxes(cov, -1, -2))

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        g = self._batches(traj)
        if g is None:
            return None
        issue, means, covs = [], [], []
        for iss, lp, mu, cv in g:
            m, c = self.collapse(lp, mu, cv)
            issue.append(iss)
            means.append(m)
            covs.append(c)
        return Predictive(
            issue_frames=np.concatenate(issue),
            means=np.concatenate(means),
            covs=np.concatenate(covs),
            horizon=self.horizon,
        )


def add_cli_args(ap):
    """--tpp-dir / --tpp-ckpt for experiment scripts (defaults from TPP_DIR / TPP_CKPT)."""
    ap.add_argument(
        "--tpp-dir",
        default=os.environ.get("TPP_DIR"),
        help="Trajectron++ clone (default: $TPP_DIR); enables the Trajectron entries",
    )
    ap.add_argument(
        "--tpp-model",
        default=DEFAULT_MODEL_DIR,
        help="trained single-agent model directory (default: %(default)s)",
    )
    ap.add_argument(
        "--tpp-ckpt",
        type=int,
        default=int(os.environ["TPP_CKPT"]) if os.environ.get("TPP_CKPT") else None,
        help="checkpoint epoch (default: $TPP_CKPT, else the latest in the model directory)",
    )
