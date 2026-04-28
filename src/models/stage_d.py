"""
src/models/stage_d.py

Stage D — Reactive Closed-Loop Controller.

Per-step residual policy mapping a 33-dim robot state to a 12-dim
all-joint correction. The architecture is whole-body; whether
support-leg outputs receive any training signal is controlled by the
per-joint loss weights baked into the trainer (and saved in the
checkpoint for reproducibility — see :class:`StageDBundle`).

The ``state_33d`` input is z-score normalized before the forward pass
using a :class:`~src.data.dataset.StateNormalizer` fit on the training
split. The normalizer is serialized inline with the model so a single
``stage_d.pt`` file contains everything inference needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import STATE_DIM, StateNormalizer

STAGE_D_INPUT_DIM = STATE_DIM       # 33
STAGE_D_OUTPUT_DIM = 12
HIDDEN_DIM = 128
N_HIDDEN_LAYERS = 3


def _build_mlp(in_dim: int, out_dim: int,
               hidden_dim: int = HIDDEN_DIM,
               n_hidden: int = N_HIDDEN_LAYERS) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class StageDPolicy(nn.Module):
    """33 → 12 unbounded MLP."""

    def __init__(self):
        super().__init__()
        self.net = _build_mlp(STAGE_D_INPUT_DIM, STAGE_D_OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class StageDBundle:
    """Policy + normalizer + joint weights, packaged for one-file checkpoint.

    ``joint_weights`` is the per-joint loss weighting used at training
    time. It does not affect inference, but is saved with the checkpoint
    so eval can describe the training regime.
    """

    policy: StageDPolicy
    normalizer: StateNormalizer
    joint_weights: torch.Tensor  # (12,) float32

    @classmethod
    def fresh(cls, normalizer: StateNormalizer,
              joint_weights: np.ndarray) -> "StageDBundle":
        if joint_weights.shape != (STAGE_D_OUTPUT_DIM,):
            raise ValueError(
                f"joint_weights must have shape (12,), got {joint_weights.shape}")
        return cls(
            policy=StageDPolicy(),
            normalizer=normalizer,
            joint_weights=torch.as_tensor(joint_weights, dtype=torch.float32),
        )

    def to(self, device) -> "StageDBundle":
        self.policy.to(device)
        self.joint_weights = self.joint_weights.to(device)
        return self

    def eval(self) -> "StageDBundle":
        self.policy.eval()
        return self

    def train(self, mode: bool = True) -> "StageDBundle":
        self.policy.train(mode)
        return self

    def parameters(self):
        return self.policy.parameters()

    def state_dict(self) -> dict:
        norm = self.normalizer
        if norm.mean is None or norm.std is None:
            raise RuntimeError("StateNormalizer must be fit before saving")
        return {
            "policy":          self.policy.state_dict(),
            "normalizer_mean": norm.mean.astype(np.float32).tolist(),
            "normalizer_std":  norm.std.astype(np.float32).tolist(),
            "joint_weights":   self.joint_weights.detach().cpu()
                                   .numpy().astype(np.float32).tolist(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.policy.load_state_dict(sd["policy"])
        norm = StateNormalizer()
        norm.mean = np.asarray(sd["normalizer_mean"], dtype=np.float32)
        norm.std  = np.asarray(sd["normalizer_std"],  dtype=np.float32)
        if norm.mean.shape != (STATE_DIM,) or norm.std.shape != (STATE_DIM,):
            raise ValueError(
                f"Normalizer in checkpoint has wrong shape: "
                f"mean={norm.mean.shape}, std={norm.std.shape}")
        self.normalizer = norm
        jw = np.asarray(sd["joint_weights"], dtype=np.float32)
        if jw.shape != (STAGE_D_OUTPUT_DIM,):
            raise ValueError(
                f"joint_weights in checkpoint has wrong shape: {jw.shape}")
        self.joint_weights = torch.as_tensor(jw, dtype=torch.float32)


def infer_stage_d(
    bundle: StageDBundle,
    state_33d: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Single-sample inference helper for the 500 Hz control loop.

    Takes a raw (un-normalized) (33,) state, applies the bundled
    normalizer, runs the policy under ``no_grad``, and returns a (12,)
    numpy array of predicted joint deltas in original units.
    """
    if state_33d.shape != (STAGE_D_INPUT_DIM,):
        raise ValueError(
            f"state_33d must have shape (33,), got {state_33d.shape}")
    normed = bundle.normalizer.transform(state_33d)
    x = torch.from_numpy(normed).to(device).unsqueeze(0)
    with torch.no_grad():
        y = bundle.policy(x).squeeze(0).cpu().numpy()
    return y.astype(np.float32)


def load_stage_d_bundle(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> StageDBundle:
    """Load a checkpoint produced by ``train_stage_d.py`` into a ready-to-use bundle."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bundle_sd = ckpt["bundle_state_dict"]

    bundle = StageDBundle(
        policy=StageDPolicy(),
        normalizer=StateNormalizer(),
        joint_weights=torch.zeros(STAGE_D_OUTPUT_DIM, dtype=torch.float32),
    )
    bundle.load_state_dict(bundle_sd)
    bundle.to(device).eval()
    return bundle
