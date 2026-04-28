"""
src/models/stage_c.py

Stage C — Contact Planner (per-episode regression).

Two independent small MLPs share the same 5-dim input (target position in
base frame at standoff, concatenated with a 2-dim interaction one-hot)
but predict different things and are trained against different label
availability:

  - StandoffMLP  : 5 → 3  ``(dx, dy, dyaw)`` standoff pose offset.
                   Only ~half of the collected episodes carry this label
                   (legacy sessions did not record it), so training must
                   mask NaN rows.
  - WaypointMLP  : 5 → 9  FR leg waypoints ``[lift | extend | press]``,
                   each 3 joint angles (hip, thigh, calf). Present in
                   every successful episode.

Keeping the heads separate keeps the training path obvious: the
waypoint head sees every sample, the standoff head sees only the
labeled subset. A shared trunk would couple their optimization in a
way that is not useful given the 15-episode scale.

The pair is bundled via :class:`StageCBundle` for checkpoint save/load
and for inference via :func:`infer_stage_c`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


STAGE_C_INPUT_DIM = 5
STANDOFF_OUTPUT_DIM = 3     # dx, dy, dyaw
WAYPOINT_OUTPUT_DIM = 9     # lift_fr (3) + extend_fr (3) + press_fr (3)
HIDDEN_DIM = 64
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


class StandoffMLP(nn.Module):
    """5 → 3 (dx, dy, dyaw). Unbounded regression."""

    def __init__(self):
        super().__init__()
        self.net = _build_mlp(STAGE_C_INPUT_DIM, STANDOFF_OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaypointMLP(nn.Module):
    """5 → 9 FR joint waypoints [lift | extend | press]. Unbounded."""

    def __init__(self):
        super().__init__()
        self.net = _build_mlp(STAGE_C_INPUT_DIM, WAYPOINT_OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class StageCBundle:
    """
    Pair of (StandoffMLP, WaypointMLP) treated as one checkpoint unit.

    Provides ``to / eval / train / parameters / state_dict /
    load_state_dict`` that forward to both members.
    """

    standoff: StandoffMLP
    waypoint: WaypointMLP

    @classmethod
    def fresh(cls) -> "StageCBundle":
        return cls(standoff=StandoffMLP(), waypoint=WaypointMLP())

    def to(self, device) -> "StageCBundle":
        self.standoff.to(device)
        self.waypoint.to(device)
        return self

    def eval(self) -> "StageCBundle":
        self.standoff.eval()
        self.waypoint.eval()
        return self

    def train(self, mode: bool = True) -> "StageCBundle":
        self.standoff.train(mode)
        self.waypoint.train(mode)
        return self

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.standoff.parameters()
        yield from self.waypoint.parameters()

    def state_dict(self) -> dict:
        return {
            "standoff": self.standoff.state_dict(),
            "waypoint": self.waypoint.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.standoff.load_state_dict(sd["standoff"])
        self.waypoint.load_state_dict(sd["waypoint"])


def infer_stage_c(
    bundle: StageCBundle,
    target_pos_base: np.ndarray,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Inference helper: given a (3,) button position in base frame, produce
    the Stage C outputs as numpy arrays.

    Always assumes press (interaction_onehot = [1, 0]).
    """
    if target_pos_base.shape != (3,):
        raise ValueError(
            f"target_pos_base must have shape (3,), got {target_pos_base.shape}")

    device = torch.device(device)
    bundle.to(device).eval()

    interaction = np.array([1.0, 0.0], dtype=np.float32)
    x = np.concatenate([target_pos_base.astype(np.float32), interaction])
    x_t = torch.from_numpy(x).to(device).unsqueeze(0)  # (1, 5)

    with torch.no_grad():
        standoff = bundle.standoff(x_t).squeeze(0).cpu().numpy()
        wp_flat  = bundle.waypoint(x_t).squeeze(0).cpu().numpy()

    return {
        "standoff_offset":    standoff.astype(np.float32),
        "lift_waypoint_fr":   wp_flat[0:3].astype(np.float32),
        "extend_waypoint_fr": wp_flat[3:6].astype(np.float32),
        "press_waypoint_fr":  wp_flat[6:9].astype(np.float32),
    }
