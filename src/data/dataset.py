"""
src/data/dataset.py

Dataset loader for Stage C (per-episode contact planner) and Stage D
(per-step residual controller) training data, collected as HDF5 episode
files by ``src/data/recorder.py``.

Schema reference: docs/training_data_spec.md v2.1 + v2 additive fields.

Nothing in this module runs the robot or touches live data; it only
reads HDF5 files written by the recorder.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Stage D state vector layout (33 dims)
# ──────────────────────────────────────────────────────────────────────
# Matches exactly the concatenation order used by the Stage D policy.
# Referenced by StateNormalizer and anywhere a slice needs a human label.

STATE_LAYOUT: list[tuple[int, int, str]] = [
    (0,   3,  "fr_joint_pos"),          # joint_pos_actual[t, 0:3]
    (3,   6,  "fr_joint_vel"),          # joint_vel_actual[t, 0:3]
    (6,   9,  "foot_to_target_error"),  # foot_to_target_error[t]
    (9,  12,  "nominal_waypoint_fr"),   # nominal_waypoint_fr[t]
    (12, 13,  "phase_progress"),        # phase_progress[t]
    (13, 15,  "imu_rp"),                # imu_rpy[t, 0:2]  (roll, pitch only)
    (15, 27,  "joint_tau_est"),         # joint_tau_est[t]  (all 12)
    (27, 30,  "imu_gyro"),              # imu_gyro[t]
    (30, 33,  "imu_accel"),             # imu_accel[t]
]
STATE_DIM: int = 33


# Defaults for pre-edit metadata fields. See docstring of load_episode.
_DEFAULT_PRESS_OFFSET_X     = 0.593
_DEFAULT_PRESS_OFFSET_Y     = -0.047
_DEFAULT_Y_SIT_STAND_DRIFT  = 0.0
_DEFAULT_NAV_EXTRA_FORWARD  = 0.10
_DEFAULT_COLLECTION_MODE    = "jacobian_pid"

_REQUIRED_PER_STEP = (
    "timestamp", "joint_pos_actual", "joint_vel_actual", "joint_pos_cmd",
    "fr_foot_pos_fk", "target_pos_base", "foot_to_target_error", "imu_rpy",
    "phase_label", "phase_progress", "nominal_waypoint_fr",
    "jacobian_pid_delta",
)
_REQUIRED_V2_PER_STEP = (
    "joint_tau_est", "imu_gyro", "imu_accel", "achieved_delta_fr",
)
_REQUIRED_PHASE_TRANSITIONS = ("lift_end", "extend_end", "hold_end")


# ──────────────────────────────────────────────────────────────────────
# Sample dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StageCSample:
    """One episode's worth of Stage C training data."""
    episode_id: str
    target_pos_base: np.ndarray         # (3,)
    interaction_onehot: np.ndarray      # (2,) — [1,0]=press, [0,1]=push
    standoff_offset: np.ndarray         # (3,) — may contain nan
    lift_waypoint_fr: np.ndarray        # (3,)
    extend_waypoint_fr: np.ndarray      # (3,)
    press_waypoint_fr: np.ndarray       # (3,)


@dataclass
class StageDSample:
    """One time step's worth of Stage D training data."""
    state_33d: np.ndarray               # (33,) input
    achieved_delta_fr: np.ndarray       # (3,) label — kept for backward compat
    achieved_delta_q: np.ndarray        # (12,) all-joint label; [0:3] == fr
    episode_id: str
    phase: int                          # 0=lift, 1=extend, 2=hold
    data_format_version: str            # "v2" or "v3"


# ──────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────

def _decode(x):
    """Decode bytes → str; leave everything else alone."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def _get_attr(attrs: h5py.AttributeManager, key: str, default=None):
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, np.ndarray):
        return v
    return _decode(v)


def _attr_bool(attrs: h5py.AttributeManager, key: str, default: bool = False) -> bool:
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, np.ndarray):
        return bool(v.item())
    return bool(v)


def _attr_float(attrs: h5py.AttributeManager, key: str, default: float) -> float:
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, np.ndarray):
        return float(v.item())
    return float(v)


def _attr_str(attrs: h5py.AttributeManager, key: str, default: str = "") -> str:
    if key not in attrs:
        return default
    return str(_decode(attrs[key]))


def _attr_array(attrs: h5py.AttributeManager, key: str,
                shape: tuple[int, ...], default=None) -> np.ndarray | None:
    if key not in attrs:
        return default
    v = np.asarray(attrs[key], dtype=np.float32)
    if v.shape != shape:
        return default
    return v


def _compute_achieved_delta_q(joint_pos_actual: np.ndarray,
                              joint_pos_cmd:    np.ndarray) -> np.ndarray:
    """``joint_pos_actual[t+1] - joint_pos_cmd[t]`` for all 12 joints.

    Last row duplicates the second-to-last to preserve (T, 12) shape; T==1
    falls back to zeros (no t+1 available).
    """
    actual = np.asarray(joint_pos_actual, dtype=np.float32)
    cmd    = np.asarray(joint_pos_cmd,    dtype=np.float32)
    T = actual.shape[0]
    if T == 0:
        return np.zeros((0, 12), dtype=np.float32)
    out = np.zeros((T, 12), dtype=np.float32)
    if T == 1:
        return out
    out[:-1] = actual[1:] - cmd[:-1]
    out[-1]  = out[-2]
    return out


def _detect_format_version(attrs: h5py.AttributeManager) -> str:
    """Return "v3" if any wholebody marker is present, else "v2"."""
    for key in ("gain_schedule", "collection_mode"):
        v = _attr_str(attrs, key, default="")
        if "wholebody" in v.lower():
            return "v3"
    return "v2"


# ──────────────────────────────────────────────────────────────────────
# load_episode
# ──────────────────────────────────────────────────────────────────────

def load_episode(path: Path) -> dict | None:
    """
    Open one HDF5 episode file and return a parsed dict, or None if the
    file cannot be used for training.

    Returns None when:
      - the file cannot be opened
      - ``success_target`` is False or missing
      - per-step T==0 (empty episode)
      - any of phase_transitions/{lift_end, extend_end, hold_end} is missing
      - any v2 additive per-step dataset is missing (predates v2 entirely)

    Missing v2 *metadata* attributes (``standoff_dxdy_commanded``,
    ``press_offset_x``, ``press_offset_y``, ``nav_extra_forward_m``,
    ``y_sit_stand_drift_comp``, ``collection_mode``) are filled silently
    with the pre-edit defaults baked into this module.
    """
    path = Path(path)
    try:
        f = h5py.File(path, "r")
    except (OSError, IOError) as e:
        logger.warning(f"Cannot open {path.name}: {e}")
        return None

    try:
        attrs = f.attrs

        if not _attr_bool(attrs, "success_target", default=False):
            logger.warning(f"Skip {path.name}: success_target missing or False")
            return None

        if "per_step" not in f:
            logger.warning(f"Skip {path.name}: no per_step group")
            return None
        per_step = f["per_step"]

        missing = [k for k in _REQUIRED_PER_STEP if k not in per_step]
        if missing:
            logger.warning(f"Skip {path.name}: missing per_step fields {missing}")
            return None

        missing_v2 = [k for k in _REQUIRED_V2_PER_STEP if k not in per_step]
        if missing_v2:
            logger.warning(
                f"Skip {path.name}: pre-v2 episode, missing {missing_v2}")
            return None

        T = per_step["timestamp"].shape[0]
        if T == 0:
            logger.warning(f"Skip {path.name}: empty per_step (T=0)")
            return None

        if "phase_transitions" not in f:
            logger.warning(f"Skip {path.name}: no phase_transitions group")
            return None
        pt_group = f["phase_transitions"]
        missing_pt = [k for k in _REQUIRED_PHASE_TRANSITIONS
                      if k not in pt_group]
        if missing_pt:
            logger.warning(
                f"Skip {path.name}: missing phase_transitions {missing_pt}")
            return None

        # Core per-step arrays. Read into memory once.
        joint_pos_actual     = np.asarray(per_step["joint_pos_actual"],
                                          dtype=np.float32)
        joint_pos_cmd        = np.asarray(per_step["joint_pos_cmd"],
                                          dtype=np.float32)
        joint_vel_actual     = np.asarray(per_step["joint_vel_actual"],
                                          dtype=np.float32)
        foot_to_target_error = np.asarray(per_step["foot_to_target_error"],
                                          dtype=np.float32)
        nominal_waypoint_fr  = np.asarray(per_step["nominal_waypoint_fr"],
                                          dtype=np.float32)
        imu_rpy              = np.asarray(per_step["imu_rpy"], dtype=np.float32)
        phase_label          = np.asarray(per_step["phase_label"], dtype=np.int64)
        phase_progress       = np.asarray(per_step["phase_progress"],
                                          dtype=np.float32)
        joint_tau_est        = np.asarray(per_step["joint_tau_est"],
                                          dtype=np.float32)
        imu_gyro             = np.asarray(per_step["imu_gyro"], dtype=np.float32)
        imu_accel            = np.asarray(per_step["imu_accel"], dtype=np.float32)
        achieved_delta_fr    = np.asarray(per_step["achieved_delta_fr"],
                                          dtype=np.float32)

        # 12-dim achieved delta — all joints, computed at load time.
        # achieved_delta_q[t] = joint_pos_actual[t+1] - joint_pos_cmd[t]; the
        # last row duplicates the second-to-last to keep (T, 12) alignment
        # (matches the recorder's last-step behavior for the FR slice).
        # The FR slice [0:3] is then overwritten with the on-disk
        # achieved_delta_fr to guarantee bit-exact equality, including the
        # NaN edges that the recorder may have produced.
        achieved_delta_q = _compute_achieved_delta_q(
            joint_pos_actual, joint_pos_cmd)
        achieved_delta_q[:, 0:3] = achieved_delta_fr

        # NaN masks — foot_to_target_error is nan when grounding was
        # unavailable at that step; achieved_delta_fr is nan at torn edges.
        error_nan_mask    = np.isnan(foot_to_target_error).any(axis=1)
        delta_nan_mask    = np.isnan(achieved_delta_fr).any(axis=1)
        drop_mask         = error_nan_mask | delta_nan_mask
        valid_mask        = ~drop_mask

        # Phase-transition joint vectors (12-dim → we slice FR 0:3 later).
        pt = {k: np.asarray(pt_group[k], dtype=np.float32)
              for k in _REQUIRED_PHASE_TRANSITIONS}

        # Root-level metadata.
        episode_id = _attr_str(attrs, "episode_id", default=path.stem)
        target_pos_base = _attr_array(
            attrs, "target_pos_base_at_standoff", (3,),
            default=np.full(3, np.nan, dtype=np.float32))

        standoff_dxdy = _attr_array(
            attrs, "standoff_dxdy_commanded", (3,),
            default=np.full(3, np.nan, dtype=np.float32))

        press_offset_x = _attr_float(
            attrs, "press_offset_x", _DEFAULT_PRESS_OFFSET_X)
        press_offset_y = _attr_float(
            attrs, "press_offset_y", _DEFAULT_PRESS_OFFSET_Y)
        nav_extra_forward_m = _attr_float(
            attrs, "nav_extra_forward_m", _DEFAULT_NAV_EXTRA_FORWARD)
        y_sit_stand_drift_comp = _attr_float(
            attrs, "y_sit_stand_drift_comp", _DEFAULT_Y_SIT_STAND_DRIFT)
        collection_mode = _attr_str(
            attrs, "collection_mode", default=_DEFAULT_COLLECTION_MODE)
        data_format_version = _detect_format_version(attrs)

        out = {
            "episode_id": episode_id,
            "path": path,
            "T": T,
            # Metadata
            "success_target": True,  # by construction we got here
            "target_pos_base": target_pos_base,
            "standoff_dxdy_commanded": standoff_dxdy,
            "press_offset_x": press_offset_x,
            "press_offset_y": press_offset_y,
            "nav_extra_forward_m": nav_extra_forward_m,
            "y_sit_stand_drift_comp": y_sit_stand_drift_comp,
            "collection_mode": collection_mode,
            "data_format_version": data_format_version,
            # Phase transitions (12-dim each)
            "phase_transitions": pt,
            # Per-step arrays
            "joint_pos_actual": joint_pos_actual,
            "joint_pos_cmd": joint_pos_cmd,
            "joint_vel_actual": joint_vel_actual,
            "foot_to_target_error": foot_to_target_error,
            "nominal_waypoint_fr": nominal_waypoint_fr,
            "imu_rpy": imu_rpy,
            "phase_label": phase_label,
            "phase_progress": phase_progress,
            "joint_tau_est": joint_tau_est,
            "imu_gyro": imu_gyro,
            "imu_accel": imu_accel,
            "achieved_delta_fr": achieved_delta_fr,
            "achieved_delta_q": achieved_delta_q,
            # Masks
            "valid_mask": valid_mask,
            "error_nan_mask": error_nan_mask,
            "delta_nan_mask": delta_nan_mask,
        }

        logger.info(
            f"Loaded {path.name}: T={T}, valid_steps={int(valid_mask.sum())}, "
            f"mode={collection_mode}, format={data_format_version}")
        return out

    finally:
        f.close()


# ──────────────────────────────────────────────────────────────────────
# Stage D state assembly
# ──────────────────────────────────────────────────────────────────────

def _build_state_33d(ep: dict, t: int) -> np.ndarray:
    """Assemble the 33-dim state vector for time step t from a loaded episode."""
    s = np.empty(STATE_DIM, dtype=np.float32)
    s[0:3]   = ep["joint_pos_actual"][t, 0:3]
    s[3:6]   = ep["joint_vel_actual"][t, 0:3]
    s[6:9]   = ep["foot_to_target_error"][t]
    s[9:12]  = ep["nominal_waypoint_fr"][t]
    s[12]    = ep["phase_progress"][t]
    s[13:15] = ep["imu_rpy"][t, 0:2]
    s[15:27] = ep["joint_tau_est"][t]
    s[27:30] = ep["imu_gyro"][t]
    s[30:33] = ep["imu_accel"][t]
    return s


# ──────────────────────────────────────────────────────────────────────
# Sample iterators
# ──────────────────────────────────────────────────────────────────────

def iter_stage_c_samples(episodes_dir: Path,
                         success_only: bool = True) -> list[StageCSample]:
    """Walk ``episodes_dir`` and build one StageCSample per loadable episode."""
    episodes_dir = Path(episodes_dir)
    paths = sorted(episodes_dir.glob("episode_*.h5"))
    samples: list[StageCSample] = []
    for p in paths:
        ep = load_episode(p)
        if ep is None:
            # load_episode already enforces success_target when success_only
            # would care; an override for success_only=False is handled below.
            if not success_only:
                # We can't recover a skipped episode here — load_episode
                # hard-fails on all non-success_target reasons too. That's
                # fine: a failed load means unusable data regardless.
                pass
            continue
        samples.append(_stage_c_sample_from_episode(ep))
    return samples


def _stage_c_sample_from_episode(ep: dict) -> StageCSample:
    pt = ep["phase_transitions"]
    return StageCSample(
        episode_id=ep["episode_id"],
        target_pos_base=ep["target_pos_base"].astype(np.float32),
        interaction_onehot=np.array([1.0, 0.0], dtype=np.float32),
        standoff_offset=ep["standoff_dxdy_commanded"].astype(np.float32),
        lift_waypoint_fr=pt["lift_end"][0:3].astype(np.float32),
        extend_waypoint_fr=pt["extend_end"][0:3].astype(np.float32),
        press_waypoint_fr=pt["hold_end"][0:3].astype(np.float32),
    )


def iter_stage_d_samples(
    episodes_dir: Path,
    success_only: bool = True,
    phases: tuple[int, ...] = (0, 1, 2),
) -> Iterator[tuple[StageDSample, str]]:
    """
    Generator yielding (StageDSample, episode_id) for every valid step in
    every loadable episode under ``episodes_dir``.

    A step is emitted only if its ``phase_label`` is in ``phases`` AND
    neither ``foot_to_target_error`` nor ``achieved_delta_fr`` is NaN.
    """
    episodes_dir = Path(episodes_dir)
    paths = sorted(episodes_dir.glob("episode_*.h5"))
    phase_set = set(int(p) for p in phases)
    for p in paths:
        ep = load_episode(p)
        if ep is None:
            continue
        yield from _iter_stage_d_from_episode(ep, phase_set)


def _iter_stage_d_from_episode(
    ep: dict, phase_set: set[int]
) -> Iterator[tuple[StageDSample, str]]:
    ep_id = ep["episode_id"]
    valid = ep["valid_mask"]
    phase_labels = ep["phase_label"]
    achieved_fr = ep["achieved_delta_fr"]
    achieved_q  = ep["achieved_delta_q"]
    fmt_version = ep["data_format_version"]
    T = ep["T"]
    dropped_phase = 0
    dropped_nan = 0
    for t in range(T):
        ph = int(phase_labels[t])
        if ph not in phase_set:
            dropped_phase += 1
            continue
        if not valid[t]:
            dropped_nan += 1
            continue
        state = _build_state_33d(ep, t)
        sample = StageDSample(
            state_33d=state,
            achieved_delta_fr=achieved_fr[t].astype(np.float32),
            achieved_delta_q=achieved_q[t].astype(np.float32),
            episode_id=ep_id,
            phase=ph,
            data_format_version=fmt_version,
        )
        yield sample, ep_id
    logger.debug(
        f"Episode {ep_id}: {T} steps → dropped_phase={dropped_phase} "
        f"dropped_nan={dropped_nan}")


# ──────────────────────────────────────────────────────────────────────
# Torch datasets
# ──────────────────────────────────────────────────────────────────────

class StageCDataset(Dataset):
    """Wraps a list of StageCSample objects."""

    def __init__(self, samples: list[StageCSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        input_vec = np.concatenate(
            [s.target_pos_base, s.interaction_onehot]).astype(np.float32)
        return {
            "input":              torch.from_numpy(input_vec),
            "standoff_offset":    torch.from_numpy(
                s.standoff_offset.astype(np.float32)),
            "lift_waypoint_fr":   torch.from_numpy(
                s.lift_waypoint_fr.astype(np.float32)),
            "extend_waypoint_fr": torch.from_numpy(
                s.extend_waypoint_fr.astype(np.float32)),
            "press_waypoint_fr":  torch.from_numpy(
                s.press_waypoint_fr.astype(np.float32)),
            "episode_id":         s.episode_id,
        }


class StageDDataset(Dataset):
    """Wraps a materialized list of StageDSample objects."""

    def __init__(self, samples: list[StageDSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "state":               torch.from_numpy(s.state_33d.astype(np.float32)),
            "delta_fr":            torch.from_numpy(
                s.achieved_delta_fr.astype(np.float32)),
            "delta_q":             torch.from_numpy(
                s.achieved_delta_q.astype(np.float32)),
            "episode_id":          s.episode_id,
            "phase":               torch.as_tensor(s.phase, dtype=torch.long),
            "data_format_version": s.data_format_version,
        }


# ──────────────────────────────────────────────────────────────────────
# Splitting and builder helpers
# ──────────────────────────────────────────────────────────────────────

def split_episodes_train_val(
    episode_paths: list[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Deterministic episode-level split. Both stages must share this split."""
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    paths = list(episode_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = int(round(len(paths) * val_fraction))
    n_train = len(paths) - n_val
    return paths[:n_train], paths[n_train:]


def _list_episode_paths(episodes_dir: Path) -> list[Path]:
    d = Path(episodes_dir)
    if not d.is_dir():
        # Treat missing dirs as empty rather than raising — multi-dir
        # callers may pass a v3 path that does not exist yet.
        return []
    return sorted(d.glob("episode_*.h5"))


def _list_episode_paths_multi(
    episodes_dirs: "list[Path] | Path",
) -> list[Path]:
    if isinstance(episodes_dirs, (str, Path)):
        episodes_dirs = [Path(episodes_dirs)]
    paths: list[Path] = []
    for d in episodes_dirs:
        paths.extend(_list_episode_paths(Path(d)))
    return sorted(paths)


def build_stage_c_datasets(
    episodes_dirs: "list[Path] | Path",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[StageCDataset, StageCDataset]:
    """Build Stage C train/val datasets from one or more episode dirs.

    ``episodes_dirs`` may be a single Path (legacy) or a list. Missing
    directories are treated as empty.
    """
    paths = _list_episode_paths_multi(episodes_dirs)
    train_paths, val_paths = split_episodes_train_val(
        paths, val_fraction=val_fraction, seed=seed)
    train_samples = _collect_stage_c(train_paths)
    val_samples = _collect_stage_c(val_paths)
    return StageCDataset(train_samples), StageCDataset(val_samples)


def build_stage_d_datasets(
    episodes_dirs: "list[Path] | Path",
    val_fraction: float = 0.2,
    seed: int = 42,
    phases: tuple[int, ...] = (0, 1, 2),
    format_filter: str | None = None,
) -> tuple[StageDDataset, StageDDataset]:
    """Build Stage D train/val datasets from one or more episode dirs.

    ``episodes_dirs`` may be a single Path (legacy) or a list. Missing
    directories are treated as empty. ``format_filter`` of ``"v2"`` /
    ``"v3"`` keeps only episodes whose detected ``data_format_version``
    matches; ``None`` keeps all.
    """
    if format_filter is not None and format_filter not in ("v2", "v3"):
        raise ValueError(
            f"format_filter must be None, 'v2', or 'v3'; got {format_filter!r}")

    paths = _list_episode_paths_multi(episodes_dirs)
    train_paths, val_paths = split_episodes_train_val(
        paths, val_fraction=val_fraction, seed=seed)
    train_samples = _collect_stage_d(train_paths, phases, format_filter)
    val_samples   = _collect_stage_d(val_paths,   phases, format_filter)
    return StageDDataset(train_samples), StageDDataset(val_samples)


def _collect_stage_c(paths: list[Path]) -> list[StageCSample]:
    out: list[StageCSample] = []
    for p in paths:
        ep = load_episode(p)
        if ep is None:
            continue
        out.append(_stage_c_sample_from_episode(ep))
    return out


def _collect_stage_d(
    paths: list[Path],
    phases: tuple[int, ...],
    format_filter: str | None = None,
) -> list[StageDSample]:
    phase_set = set(int(p) for p in phases)
    out: list[StageDSample] = []
    for p in paths:
        ep = load_episode(p)
        if ep is None:
            continue
        if format_filter is not None and ep["data_format_version"] != format_filter:
            continue
        for sample, _ in _iter_stage_d_from_episode(ep, phase_set):
            out.append(sample)
    return out


# ──────────────────────────────────────────────────────────────────────
# State normalizer
# ──────────────────────────────────────────────────────────────────────

class StateNormalizer:
    """Per-dim z-score normalization for Stage D state vectors."""

    def __init__(self):
        self.mean: np.ndarray | None = None  # (33,) float32
        self.std:  np.ndarray | None = None  # (33,) float32

    def fit(self, samples: list[StageDSample]) -> None:
        if not samples:
            raise ValueError("Cannot fit StateNormalizer on empty sample list")
        X = np.stack([s.state_33d for s in samples], axis=0).astype(np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard near-constant dims so transform doesn't blow up.
        std = np.where(std < 1e-6, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def _check_fit(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("StateNormalizer is not fit yet")

    def transform(self, state: np.ndarray) -> np.ndarray:
        self._check_fit()
        return ((state - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, state: np.ndarray) -> np.ndarray:
        self._check_fit()
        return (state * self.std + self.mean).astype(np.float32)

    def save(self, path: Path) -> None:
        self._check_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean":   self.mean.tolist(),
            "std":    self.std.tolist(),
            "layout": [[a, b, n] for (a, b, n) in STATE_LAYOUT],
            "dim":    STATE_DIM,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StateNormalizer":
        with open(path) as f:
            payload = json.load(f)
        obj = cls()
        obj.mean = np.asarray(payload["mean"], dtype=np.float32)
        obj.std = np.asarray(payload["std"], dtype=np.float32)
        if obj.mean.shape != (STATE_DIM,) or obj.std.shape != (STATE_DIM,):
            raise ValueError(
                f"Normalizer in {path} has wrong shape: "
                f"mean={obj.mean.shape}, std={obj.std.shape}")
        return obj
