"""
src/policy/stage_d_runtime.py

Deployment wrapper for the trained Stage D residual policy.

Architecture (per training_data_spec_v3.md §2.1):
    33 → 128 → 128 → 128 → 12, ReLU, no output activation.

The training script (scripts/train_stage_d.py) writes a single ``stage_d.pt``
checkpoint that bundles the policy weights, the per-dim z-score normalizer
fit on the train split, and the joint-loss weights used during training.
``load_stage_d_bundle`` already handles unpacking — this class wraps it for
hot-path inference at 500 Hz.

The forward path pre-allocates a (1, 33) GPU tensor reused across calls so
the per-step cost is one host→device copy of 33 floats plus one MLP forward.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.data.dataset import STATE_DIM
from src.models.stage_d import STAGE_D_OUTPUT_DIM, load_stage_d_bundle

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


class StageDInference:
    """Hot-path inference wrapper for the Stage D residual policy."""

    def __init__(
        self,
        ckpt_path: PathLike = "models/stage_d_combined/stage_d.pt",
        device: str = "cuda",
    ):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Stage D checkpoint not found: {ckpt_path}")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            device = "cpu"
        self._device = torch.device(device)
        self._ckpt_path = ckpt_path

        self._bundle = load_stage_d_bundle(ckpt_path, device=self._device)

        norm = self._bundle.normalizer
        if norm.mean is None or norm.std is None:
            raise RuntimeError(
                f"Stage D checkpoint at {ckpt_path} has no normalizer stats."
            )
        self._mean = torch.from_numpy(
            norm.mean.astype(np.float32)).to(self._device)
        self._std = torch.from_numpy(
            norm.std.astype(np.float32)).to(self._device)

        # Pre-allocated input tensor reused across calls.
        self._x = torch.zeros(
            (1, STATE_DIM), dtype=torch.float32, device=self._device)

        logger.info(
            f"StageDInference loaded from {ckpt_path} on {self._device}")

    @property
    def checkpoint_path(self) -> Path:
        return self._ckpt_path

    @torch.no_grad()
    def predict(self, state_33: np.ndarray) -> np.ndarray:
        """(33,) raw state → (12,) residual joint delta."""
        if state_33.shape != (STATE_DIM,):
            raise ValueError(
                f"state must have shape ({STATE_DIM},), got {state_33.shape}"
            )
        # Copy raw state into the pre-allocated tensor, normalize on device.
        self._x.copy_(torch.from_numpy(
            np.ascontiguousarray(state_33, dtype=np.float32)).unsqueeze(0))
        normed = (self._x - self._mean) / self._std
        y = self._bundle.policy(normed).squeeze(0)
        return y.detach().cpu().numpy().astype(np.float32)
