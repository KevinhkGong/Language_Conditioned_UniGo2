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


# ──────────────────────────────────────────────────────────────────────
# Action-chunked variant — opt-in alongside the existing single-step path
# ──────────────────────────────────────────────────────────────────────
#
# ``StageDChunkedPolicy`` is an MLP whose final layer expands to ``K * 12``
# units, reshaped into ``(K, 12)`` chunks of K future residuals. It is a
# strict architectural addition — ``StageDPolicy`` and ``StageDBundle`` are
# not modified, and a single-step training run still produces a checkpoint
# that loads through the existing ``StageDBundle.load_state_dict``.
#
# At deployment, ``StageDChunkedInference`` (in src.policy.stage_d_runtime)
# wraps this with a rolling chunk buffer and exponential-decay temporal
# ensembling so the per-step interface ``predict(state) -> (12,)`` is
# preserved across single-step and chunked variants.

CHUNKED_HIDDEN_DIMS: tuple = (128, 256, 256)


class StageDChunkedPolicy(nn.Module):
    """33 → (K, 12) MLP. Predicts K future residuals from the current state.

    ``forward`` accepts batched ``(B, 33)`` or unbatched ``(33,)`` input
    and returns ``(B, K, 12)`` or ``(K, 12)`` respectively. Hidden layer
    sizes default to ``CHUNKED_HIDDEN_DIMS = (128, 256, 256)``.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        chunk_size: int = 25,
        hidden_dims: tuple = CHUNKED_HIDDEN_DIMS,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.state_dim = int(state_dim)
        self.chunk_size = int(chunk_size)
        self.action_dim = STAGE_D_OUTPUT_DIM   # 12

        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        flat = self.net(state)  # (..., K*12)
        return flat.view(*flat.shape[:-1], self.chunk_size, self.action_dim)


@dataclass
class StageDChunkedBundle:
    """Chunked policy + normalizer + joint weights + chunk_size, one-file ckpt.

    Mirrors :class:`StageDBundle` but for the chunked architecture. The
    single-step bundle is unchanged — chunked checkpoints carry the
    ``chunk_size`` field so the inference side can verify alignment.
    """

    policy: StageDChunkedPolicy
    normalizer: StateNormalizer
    joint_weights: torch.Tensor  # (12,) float32
    chunk_size: int

    @classmethod
    def fresh(
        cls,
        normalizer: StateNormalizer,
        joint_weights: np.ndarray,
        chunk_size: int,
    ) -> "StageDChunkedBundle":
        if joint_weights.shape != (STAGE_D_OUTPUT_DIM,):
            raise ValueError(
                f"joint_weights must have shape (12,), got {joint_weights.shape}")
        return cls(
            policy=StageDChunkedPolicy(chunk_size=chunk_size),
            normalizer=normalizer,
            joint_weights=torch.as_tensor(joint_weights, dtype=torch.float32),
            chunk_size=int(chunk_size),
        )

    def to(self, device) -> "StageDChunkedBundle":
        self.policy.to(device)
        self.joint_weights = self.joint_weights.to(device)
        return self

    def eval(self) -> "StageDChunkedBundle":
        self.policy.eval()
        return self

    def train(self, mode: bool = True) -> "StageDChunkedBundle":
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
            "chunk_size":      int(self.chunk_size),
        }

    def load_state_dict(self, sd: dict) -> None:
        ck = int(sd.get("chunk_size", self.chunk_size))
        if ck != self.chunk_size:
            raise ValueError(
                f"checkpoint chunk_size={ck} does not match "
                f"policy chunk_size={self.chunk_size}")
        self.policy.load_state_dict(sd["policy"])
        norm = StateNormalizer()
        norm.mean = np.asarray(sd["normalizer_mean"], dtype=np.float32)
        norm.std = np.asarray(sd["normalizer_std"], dtype=np.float32)
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


def load_stage_d_chunked_bundle(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
    chunk_size: int | None = None,
) -> StageDChunkedBundle:
    """Load a chunked checkpoint into a ready-to-use bundle.

    If ``chunk_size`` is None, infer it from the checkpoint's top-level
    ``chunk_size`` field (preferred) or from ``bundle_state_dict.chunk_size``.
    Raises ValueError on mismatch.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bundle_sd = ckpt["bundle_state_dict"]
    inferred = int(
        ckpt.get("chunk_size", bundle_sd.get("chunk_size", 0)) or 0)
    if inferred <= 0:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has no chunk_size; "
            "is this a single-step bundle?")
    if chunk_size is not None and int(chunk_size) != inferred:
        raise ValueError(
            f"requested chunk_size={chunk_size} does not match checkpoint "
            f"chunk_size={inferred}")

    bundle = StageDChunkedBundle(
        policy=StageDChunkedPolicy(chunk_size=inferred),
        normalizer=StateNormalizer(),
        joint_weights=torch.zeros(STAGE_D_OUTPUT_DIM, dtype=torch.float32),
        chunk_size=inferred,
    )
    bundle.load_state_dict(bundle_sd)
    bundle.to(device).eval()
    return bundle
