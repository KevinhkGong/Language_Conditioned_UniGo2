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
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.data.dataset import STATE_DIM
from src.models.stage_d import (
    STAGE_D_OUTPUT_DIM,
    StageDChunkedBundle,
    load_stage_d_bundle,
    load_stage_d_chunked_bundle,
)

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


class StageDChunkedInference:
    """Action-chunked Stage D residual policy with temporal ensembling.

    Wraps a ``StageDChunkedBundle`` checkpoint behind the same per-step
    interface as :class:`StageDInference`: ``predict(state) -> (12,)``.
    Internally maintains a deque of the most recent chunks and ensembles
    their predictions for the current control step using exponential
    decay weights ``w_i = exp(-decay_rate * i)`` where ``i`` is the number
    of steps since the chunk was predicted.

    Lifecycle:
      - ``__init__`` loads the chunked bundle. If loading fails and
        ``fallback_to_single_step`` is True with a valid single-step
        ``fallback_checkpoint_path``, the instance silently delegates
        every ``predict()`` to a wrapped :class:`StageDInference`.
      - ``reset()`` clears the chunk buffer and the step counter; call at
        each active-phase entry to prevent leakage of stale chunks across
        phase transitions.
      - ``predict(state)`` predicts a fresh chunk, inserts it into the
        buffer (skipping insertion on NaN), then weights all alive past
        chunks for the current step and returns the ensembled (12,)
        residual. Falls back to zero residual if no chunks are alive.

    Robustness notes:
      - Chunks containing any NaN are not inserted into the buffer (they
        also wouldn't have made it into training data, so we treat the
        prediction as unreliable rather than poison the ensemble).
      - The weighted average is normalised by the sum of weights, so the
        magnitude is bounded regardless of how many chunks contribute.
    """

    def __init__(
        self,
        checkpoint_path: PathLike = "models/stage_d_combined/stage_d.pt",
        chunk_size: int = 25,
        decay_rate: float = 0.1,
        device: str = "cpu",
        fallback_to_single_step: bool = False,
        fallback_checkpoint_path: Optional[PathLike] = None,
    ):
        self._ckpt_path = Path(checkpoint_path)
        self._chunk_size = int(chunk_size)
        if self._chunk_size < 1:
            raise ValueError(
                f"chunk_size must be >= 1, got {self._chunk_size}")
        self._decay_rate = float(decay_rate)
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested for chunked Stage D but unavailable; "
                "falling back to CPU.")
            device = "cpu"
        self._device = torch.device(device)
        self._fallback: Optional[StageDInference] = None

        # Pre-compute exponential decay weights for offsets [0, K).
        offsets = np.arange(self._chunk_size, dtype=np.float64)
        self._weights = np.exp(-self._decay_rate * offsets).astype(np.float32)

        # Try to load the chunked bundle. On failure, optionally fall back.
        try:
            self._bundle = load_stage_d_chunked_bundle(
                self._ckpt_path, device=self._device,
                chunk_size=self._chunk_size,
            )
        except Exception as e:
            if fallback_to_single_step and fallback_checkpoint_path is not None:
                logger.warning(
                    f"Chunked checkpoint at {self._ckpt_path} failed to load "
                    f"({e}); falling back to single-step "
                    f"{fallback_checkpoint_path}.")
                self._fallback = StageDInference(
                    ckpt_path=fallback_checkpoint_path,
                    device=device,
                )
                self._bundle = None
            else:
                raise

        if self._bundle is not None:
            norm = self._bundle.normalizer
            if norm.mean is None or norm.std is None:
                raise RuntimeError(
                    f"Chunked checkpoint at {self._ckpt_path} has no "
                    "normalizer stats.")
            self._mean = torch.from_numpy(
                norm.mean.astype(np.float32)).to(self._device)
            self._std = torch.from_numpy(
                norm.std.astype(np.float32)).to(self._device)
            self._x = torch.zeros(
                (1, STATE_DIM), dtype=torch.float32, device=self._device)
            logger.info(
                f"StageDChunkedInference loaded from {self._ckpt_path} "
                f"(chunk_size={self._chunk_size}, decay_rate={self._decay_rate}) "
                f"on {self._device}")

        # Ensemble buffer: deque of (predicted_at_step, chunk_array).
        self._chunk_buffer: deque = deque(maxlen=self._chunk_size)
        self._step_counter: int = 0

    @property
    def checkpoint_path(self) -> Path:
        return self._ckpt_path

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def is_fallback(self) -> bool:
        return self._fallback is not None

    def reset(self) -> None:
        """Clear the chunk buffer and reset the step counter.

        Call at active-phase entry to prevent stale chunks from a previous
        phase's state distribution from contributing to the new phase's
        ensembled output. Cheap and idempotent; safe to call always.
        """
        self._chunk_buffer.clear()
        self._step_counter = 0
        if self._fallback is not None and hasattr(self._fallback, "reset"):
            try:
                self._fallback.reset()
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(f"fallback.reset() raised: {e}")

    @torch.no_grad()
    def predict(self, state_33: np.ndarray) -> np.ndarray:
        """(33,) raw state → (12,) ensembled residual joint delta."""
        if self._fallback is not None:
            return self._fallback.predict(state_33)

        if state_33.shape != (STATE_DIM,):
            raise ValueError(
                f"state must have shape ({STATE_DIM},), got {state_33.shape}"
            )

        # 1. Predict a fresh K×12 chunk from the current state.
        self._x.copy_(torch.from_numpy(
            np.ascontiguousarray(state_33, dtype=np.float32)).unsqueeze(0))
        normed = (self._x - self._mean) / self._std
        new_chunk = self._bundle.policy(normed).squeeze(0).detach().cpu()
        new_chunk_np = new_chunk.numpy().astype(np.float32)
        # new_chunk_np shape: (K, 12)

        # 2. Insert into buffer unless the chunk contains a NaN — refuse
        #    to poison the ensemble with bad predictions.
        if np.all(np.isfinite(new_chunk_np)):
            self._chunk_buffer.append((self._step_counter, new_chunk_np))
        else:
            logger.warning(
                "Chunked Stage D produced non-finite values; chunk skipped.")

        # 3. Weighted-average all alive past chunks' predictions for the
        #    current step. Offset = how many control steps ago the chunk
        #    was emitted; chunks older than K steps are silently dropped
        #    by the deque's maxlen but we still gate on offset < K.
        current_t = self._step_counter
        weighted_sum = np.zeros(STAGE_D_OUTPUT_DIM, dtype=np.float64)
        weight_total = 0.0
        for predicted_at, chunk in self._chunk_buffer:
            offset = current_t - predicted_at
            if 0 <= offset < self._chunk_size:
                w = float(self._weights[offset])
                weighted_sum += w * chunk[offset]
                weight_total += w

        self._step_counter += 1

        if weight_total < 1e-9:
            # No alive contributions (e.g., first step had a NaN). Return zero
            # residual rather than NaN so the controller's downstream sum into
            # target_q is well-defined.
            return np.zeros(STAGE_D_OUTPUT_DIM, dtype=np.float32)
        return (weighted_sum / weight_total).astype(np.float32)
