"""Pretrained pedestrian GRU baselines (extras=[torch]).

Two models trained on a large corpus of retail pedestrian motion for the
original REACT research:

* ``GRUPosition`` — history of normalized (x, y); checkpoint
  ``gru_position_best.pt``
* ``GRUOrientation`` — normalized (x, y) plus sin/cos heading; checkpoint
  ``gru_orientation_best.pt``

Both ship with their weights under ``reactmetric.baselines.checkpoints`` and
are registered as default comparison baselines when torch is installed.

Retrained / scratch checkpoints from ``scripts/train_gru_baselines.py`` expect
inputs on an approximately **10 Hz** (``dt=0.1``) frame grid — the training
script regrids Avolta (~6.25 Hz) and Eindhoven onto that rate before windowing.
Pass trajectories that are already near 10 Hz (e.g. Eindhoven) or resample
upstream before calling ``predict``.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Optional, Tuple

import numpy as np

from ..core.predictive import Predictive
from ..core.trajectory import Trajectory
from ..forecasters.base import BaseForecaster

_CHECKPOINT_NAMES = {
    "position": "gru_position_best.pt",
    "orientation": "gru_orientation_best.pt",
}


def default_checkpoint_path(variant: str = "position") -> str:
    """Absolute path to a shipped pretrained GRU checkpoint."""
    if variant not in _CHECKPOINT_NAMES:
        raise ValueError(f"variant must be one of {tuple(_CHECKPOINT_NAMES)}; got {variant!r}")
    name = _CHECKPOINT_NAMES[variant]
    # Prefer importlib.resources (installed package), fall back to source tree.
    try:
        pkg = resources.files("reactmetric.baselines.checkpoints")
        target = pkg.joinpath(name)
        with resources.as_file(target) as path:
            return str(path)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "checkpoints", name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Shipped GRU checkpoint not found for variant={variant!r}: {path}"
            ) from None
        return path


def _compute_heading(positions: np.ndarray) -> np.ndarray:
    t = len(positions)
    headings = np.zeros(t)
    for i in range(t - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            headings[i] = np.arctan2(dy, dx)
        elif i > 0:
            headings[i] = headings[i - 1]
    headings[-1] = headings[-2] if t >= 2 else 0.0
    return headings


def _normalize_trajectory(
    positions: np.ndarray, headings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    origin = positions[0].copy()
    translated = positions - origin
    initial_heading = float(headings[0])
    c, s = np.cos(-initial_heading), np.sin(-initial_heading)
    rotation_matrix = np.array([[c, -s], [s, c]])
    normalized_positions = (rotation_matrix @ translated.T).T
    normalized_headings = headings - initial_heading
    normalized_headings = np.arctan2(
        np.sin(normalized_headings), np.cos(normalized_headings)
    )
    return normalized_positions, normalized_headings, initial_heading


class _GRUTrajectoryNet:  # pragma: no cover - thin torch wrapper
    """Architecture matching TGF_2026 ``GRUTrajectoryPredictor`` checkpoints."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        forecast_horizon: int,
        dropout: float = 0.1,
    ):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, forecast_horizon * 4),
                )
                self.forecast_horizon = forecast_horizon

            def forward(self, x):
                import torch

                _, hidden = self.encoder(x)
                h_final = hidden[-1]
                output = self.decoder(h_final).view(-1, self.forecast_horizon, 4)
                mean = output[:, :, :2]
                log_var = torch.clamp(output[:, :, 2:], min=-10.0, max=4.0)
                return mean, log_var

        self.net = Net()


class _GRUBase(BaseForecaster):
    """Shared loader/predict path for position and orientation GRUs."""

    variant: str = "position"
    input_key: str = "input_position"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        history: Optional[int] = None,
        horizon: Optional[int] = None,
        device: str = "cpu",
        name: Optional[str] = None,
    ):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "GRU baselines require torch: pip install 'reactmetric[torch]'"
            ) from e

        path = checkpoint or default_checkpoint_path(self.variant)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        state = (
            ckpt.get("model_state_dict", ckpt)
            if isinstance(ckpt, dict)
            else ckpt
        )

        self.history = int(history if history is not None else cfg.get("history_length", 30))
        self.horizon = int(
            horizon if horizon is not None else cfg.get("forecast_horizon", 15)
        )
        hidden = int(cfg.get("hidden_size", 64))
        num_layers = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))
        input_dim = 2 if self.variant == "position" else 4

        wrap = _GRUTrajectoryNet(
            input_dim=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            forecast_horizon=self.horizon,
            dropout=dropout,
        )
        self.device = device
        self.name = name or (
            "GRUPosition" if self.variant == "position" else "GRUOrientation"
        )
        self.checkpoint_path = path
        self.model = wrap.net.to(device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, traj: Trajectory) -> Optional[Predictive]:
        if traj.D != 2:
            return None
        import torch

        obs = traj.positions
        t, history, horizon = traj.T, self.history, self.horizon
        if t < history + horizon:
            return None

        issue = np.arange(history, t - horizon + 1)
        inputs, origins, init_headings = [], [], []
        for ti in issue:
            hist = obs[ti - history : ti]
            headings = _compute_heading(hist)
            norm_hist, norm_headings, init_heading = _normalize_trajectory(hist, headings)
            if self.input_key == "input_position":
                inp = norm_hist
            else:
                sin_h = np.sin(norm_headings)[:, None]
                cos_h = np.cos(norm_headings)[:, None]
                inp = np.concatenate([norm_hist, sin_h, cos_h], axis=1)
            inputs.append(inp)
            origins.append(hist[0])
            init_headings.append(init_heading)

        x = torch.tensor(np.array(inputs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean_norm, log_var = self.model(x)
            mean_norm = mean_norm.cpu().numpy()
            var = np.exp(log_var.cpu().numpy())

        n = len(issue)
        means = np.empty((n, horizon, 2))
        covs = np.zeros((n, horizon, 2, 2))
        for i in range(n):
            ih = init_headings[i]
            c, s = np.cos(-ih), np.sin(-ih)
            # Match TGF_2026 GRUForecaster denormalization (row-vector @ rot).
            rot = np.array([[c, -s], [s, c]])
            means[i] = mean_norm[i, :horizon] @ rot + origins[i]
            # The network's variances live in the heading-aligned frame (along-track,
            # cross-track). With world = normalized @ rot for row vectors, the world
            # covariance is rot.T @ diag(var) @ rot, the same map applied to the means.
            covs[i] = np.einsum("ji,hj,jk->hik", rot, var[i, :horizon], rot)
        return Predictive(issue_frames=issue, means=means, covs=covs, horizon=horizon)


class GRUPosition(_GRUBase):
    """Pretrained position-history GRU (2-D normalized coordinates)."""

    variant = "position"
    input_key = "input_position"


class GRUOrientation(_GRUBase):
    """Pretrained orientation-augmented GRU (position + sin/cos heading)."""

    variant = "orientation"
    input_key = "input_orientation"


# Back-compat alias: bare ``GRU`` means the position model.
GRU = GRUPosition
