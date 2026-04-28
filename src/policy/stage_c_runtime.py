"""
src/policy/stage_c_runtime.py

Deployment wrapper for the trained Stage C model (StandoffMLP + WaypointMLP).

The training script (scripts/train_stage_c.py) writes a single bundle file
(e.g. ``models/stage_c/stage_c.pt``) with state_dict layout::

    {"bundle_state_dict": {"standoff": ..., "waypoint": ...}, ...}

Stage C inputs are raw ``[target_pos_base (3), interaction_onehot (2)]`` —
the training pipeline applies no normalization, so neither does this wrapper.
The architecture is fixed at 5 → 64 → 64 → 64 → {3, 9} (ReLU) per
training_data_spec_v3.md §2.1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from src.models.stage_c import StageCBundle

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


class StageCInference:
    """Runtime wrapper around a trained ``StageCBundle``.

    Both heads share the bundle checkpoint, so a single path is sufficient.
    The ``standoff_ckpt_path`` and ``waypoint_ckpt_path`` arguments are kept
    for API symmetry; if both are given they must point at the same file.
    """

    def __init__(
        self,
        standoff_ckpt_path: PathLike = "models/stage_c/stage_c.pt",
        waypoint_ckpt_path: PathLike = "models/stage_c/stage_c.pt",
        device: str = "cuda",
    ):
        standoff_ckpt_path = Path(standoff_ckpt_path)
        waypoint_ckpt_path = Path(waypoint_ckpt_path)
        if standoff_ckpt_path != waypoint_ckpt_path:
            raise ValueError(
                "Stage C is shipped as a single bundle checkpoint; "
                "standoff_ckpt_path and waypoint_ckpt_path must match. "
                f"Got {standoff_ckpt_path} vs {waypoint_ckpt_path}."
            )
        if not standoff_ckpt_path.is_file():
            raise FileNotFoundError(
                f"Stage C checkpoint not found: {standoff_ckpt_path}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            device = "cpu"
        self._device = torch.device(device)
        self._ckpt_path = standoff_ckpt_path

        ckpt = torch.load(
            str(standoff_ckpt_path),
            map_location=self._device,
            weights_only=False,
        )
        bundle_sd = ckpt["bundle_state_dict"]

        self._bundle = StageCBundle.fresh()
        self._bundle.load_state_dict(bundle_sd)
        self._bundle.to(self._device).eval()

        logger.info(f"StageCInference loaded from {standoff_ckpt_path} on {self._device}")

    @property
    def checkpoint_path(self) -> Path:
        return self._ckpt_path

    @staticmethod
    def _build_input(
        target_pos_base: np.ndarray,
        interaction: str,
    ) -> np.ndarray:
        if target_pos_base.shape != (3,):
            raise ValueError(
                f"target_pos_base must have shape (3,), got {target_pos_base.shape}"
            )
        if interaction == "press":
            onehot = np.array([1.0, 0.0], dtype=np.float32)
        elif interaction == "push":
            onehot = np.array([0.0, 1.0], dtype=np.float32)
        else:
            raise ValueError(
                f"interaction must be 'press' or 'push', got {interaction!r}"
            )
        return np.concatenate(
            [target_pos_base.astype(np.float32), onehot]
        ).astype(np.float32)

    @torch.no_grad()
    def predict_standoff(
        self,
        target_pos_base: np.ndarray,
        interaction: str = "press",
    ) -> np.ndarray:
        """5 → 3 standoff offset ``[dx, dy, dyaw]`` in metres / radians."""
        x = self._build_input(target_pos_base, interaction)
        x_t = torch.from_numpy(x).to(self._device).unsqueeze(0)
        y = self._bundle.standoff(x_t).squeeze(0).cpu().numpy()
        return y.astype(np.float32)

    @torch.no_grad()
    def predict_waypoints(
        self,
        target_pos_base: np.ndarray,
        interaction: str = "press",
    ) -> Dict[str, np.ndarray]:
        """5 → 9 → ``{"lift", "extend", "press"}`` FR ``[hip, thigh, calf]``."""
        x = self._build_input(target_pos_base, interaction)
        x_t = torch.from_numpy(x).to(self._device).unsqueeze(0)
        wp = self._bundle.waypoint(x_t).squeeze(0).cpu().numpy().astype(np.float32)
        return {
            "lift":   wp[0:3].copy(),
            "extend": wp[3:6].copy(),
            "press":  wp[6:9].copy(),
        }
