"""
scripts/inspect_v4_calibration.py

Read-only diagnostic that opens the N most recent v3 episode HDF5 files and
prints a summary of metadata correctness, standoff drift across the session,
and rear-leg tracking quality during extend+hold. Used as a sanity check
before committing to a primary collection session.

The script never writes to disk and never touches the robot. Output is
plain ASCII to stdout. Files that fail to open are skipped with a warning.

CLI:
    python scripts/inspect_v4_calibration.py
        [--data-dir data/real/stage_d_v3]
        [--num-episodes 5]

Sections:
    1. Per-episode metadata table — gravity_ff state, support gains, gain
       schedule, intrinsics version, success flags.
    2. Standoff drift across the recent episodes — x/y/z range and
       monotonic-trend detection.
    3. Rear-leg tracking quality — max and mean |q_actual - q_cmd| per
       support joint during phases {lift, extend, hold}.
    4. Overall verdict — READY / ATTENTION / INVESTIGATE.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# Repo root on sys.path so this script can be invoked directly without an
# installed package, mirroring the validate_episode_v* pattern.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.recorder import PHASE_TO_LABEL  # canonical mapping

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("data/real/stage_d_v3")
DEFAULT_NUM_EPISODES = 5

# Active-phase integer labels read from src/data/recorder.PHASE_TO_LABEL so
# we don't drift if the mapping ever changes.
LIFT_LBL   = PHASE_TO_LABEL["lift"]
EXTEND_LBL = PHASE_TO_LABEL["extend"]
HOLD_LBL   = PHASE_TO_LABEL["hold"]
ACTIVE_LABELS = (LIFT_LBL, EXTEND_LBL, HOLD_LBL)

# Standard Go2 per-leg ordering: FR=[0:3], FL=[3:6], RR=[6:9], RL=[9:12].
# Support tripod's thigh/calf joints — these are the ones that bear body
# weight against gravity during lift→retract and that gravity FF is meant
# to keep tracked. Hips are excluded from the tracking summary because
# their gravity moment is small under our simple model.
SUPPORT_JOINTS: List[Tuple[str, int]] = [
    ("FL_thi",  4),
    ("FL_clf",  5),
    ("RR_thi",  7),
    ("RR_clf",  8),
    ("RL_thi", 10),
    ("RL_clf", 11),
]

# Tracking-quality thresholds for flagging.
MAX_ERR_FLAG_RAD  = 0.10   # any single-step residual sag above this → flag
MEAN_ERR_FLAG_RAD = 0.05   # sustained tracking error above this → flag
MAX_ERR_FAIL_RAD  = 0.15   # severe — drives "INVESTIGATE" verdict

# Standoff drift thresholds.
DRIFT_X_FLAG_M = 0.03   # 3 cm spread across recent episodes


# ──────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────

def _list_recent_episodes(data_dir: Path, n: int) -> List[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("episode_*.h5"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return files[:n]


def _attr(attrs: h5py.AttributeManager, key: str, default: Any = "MISSING") -> Any:
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return v.item()
        return v
    return v


def _attr_str(attrs: h5py.AttributeManager, key: str) -> str:
    v = _attr(attrs, key, default="MISSING")
    if isinstance(v, np.ndarray):
        return str(v.tolist())
    return str(v)


def _attr_bool(attrs: h5py.AttributeManager, key: str) -> str:
    if key not in attrs:
        return "MISSING"
    v = attrs[key]
    if isinstance(v, np.ndarray):
        return "True" if bool(v.item()) else "False"
    return "True" if bool(v) else "False"


def _attr_float(attrs: h5py.AttributeManager, key: str,
                fmt: str = "{:.3f}") -> str:
    if key not in attrs:
        return "MISSING"
    v = attrs[key]
    if isinstance(v, np.ndarray):
        v = v.item()
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _attr_3vec(attrs: h5py.AttributeManager, key: str) -> Optional[np.ndarray]:
    if key not in attrs:
        return None
    v = np.asarray(attrs[key]).astype(np.float64).flatten()
    if v.shape != (3,):
        return None
    return v


# ──────────────────────────────────────────────
# Per-episode loaders
# ──────────────────────────────────────────────

def _load_metadata_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            return {
                "file":                       path.name,
                "episode_id":                 _attr_str(a, "episode_id"),
                "gravity_ff_enabled":         _attr_bool(a, "gravity_ff_enabled"),
                "ff_body_mass_kg":            _attr_float(a, "gravity_ff_body_mass_kg"),
                "kp_support_soft":            _attr_float(a, "kp_support_soft"),
                "kd_support_soft":            _attr_float(a, "kd_support_soft"),
                "gain_schedule":              _attr_str(a, "gain_schedule"),
                "camera_intrinsics_version":  _attr_str(a, "camera_intrinsics_version"),
                "success_audio_live":         _attr_bool(a, "success_audio_live"),
                "success_target":             _attr_bool(a, "success_target"),
                "color_detected":             _attr_str(a, "color_detected"),
            }
    except (OSError, IOError) as e:
        logger.warning(f"Cannot open {path.name}: {e}")
        return None


def _load_standoff(path: Path) -> Optional[np.ndarray]:
    try:
        with h5py.File(path, "r") as f:
            return _attr_3vec(f.attrs, "target_pos_base_at_standoff")
    except (OSError, IOError) as e:
        logger.warning(f"Cannot open {path.name} for standoff: {e}")
        return None


def _load_tracking(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (joint_pos_actual, joint_pos_cmd, phase_label) for one episode.

    Skips the file if any required field is missing or the per_step group is
    empty. Returns None on any irregularity.
    """
    try:
        with h5py.File(path, "r") as f:
            if "per_step" not in f:
                logger.warning(f"{path.name}: no per_step group — skipping")
                return None
            ps = f["per_step"]
            for k in ("joint_pos_actual", "joint_pos_cmd", "phase_label"):
                if k not in ps:
                    logger.warning(f"{path.name}: missing {k} — skipping")
                    return None
            actual = np.asarray(ps["joint_pos_actual"], dtype=np.float32)
            cmd    = np.asarray(ps["joint_pos_cmd"],    dtype=np.float32)
            phase  = np.asarray(ps["phase_label"],      dtype=np.int64)
            if actual.shape[0] == 0:
                logger.warning(f"{path.name}: T=0 — skipping")
                return None
            return actual, cmd, phase
    except (OSError, IOError) as e:
        logger.warning(f"Cannot open {path.name} for tracking: {e}")
        return None


# ──────────────────────────────────────────────
# Section 1 — Metadata
# ──────────────────────────────────────────────

def _print_metadata_section(rows: List[Dict[str, str]]) -> int:
    print("─" * 100)
    print("Section 1 — Metadata")
    print("─" * 100)
    if not rows:
        print("(no episodes loaded)")
        return 0

    cols = [
        ("episode_id",                 16),
        ("gravity_ff_enabled",         18),
        ("ff_body_mass_kg",            16),
        ("kp_support_soft",            15),
        ("kd_support_soft",            15),
        ("gain_schedule",              30),
        ("camera_intrinsics_version",  26),
        ("success_audio_live",         18),
        ("success_target",             14),
        ("color_detected",             14),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("  ".join("-" * w for _, w in cols))

    missing_count = 0
    for r in rows:
        line_parts = []
        for name, w in cols:
            val = str(r.get(name, "MISSING"))
            if val == "MISSING":
                missing_count += 1
            line_parts.append(f"{val:<{w}}")
        print("  ".join(line_parts))

    if missing_count > 0:
        print(f"\n  >> {missing_count} MISSING attribute(s) across the table above.")
    return missing_count


# ──────────────────────────────────────────────
# Section 2 — Standoff drift
# ──────────────────────────────────────────────

def _print_standoff_section(
    paths: List[Path], rows: List[Dict[str, str]]
) -> Tuple[float, bool]:
    """Return (x_range_m, monotonic_trend_flag)."""
    print()
    print("─" * 100)
    print("Section 2 — Standoff drift (target_pos_base_at_standoff)")
    print("─" * 100)

    standoffs: List[Tuple[str, np.ndarray]] = []
    for p, r in zip(paths, rows):
        s = _load_standoff(p)
        if s is None:
            print(f"  {r.get('episode_id', p.name):<16}  MISSING target_pos_base_at_standoff")
            continue
        standoffs.append((r.get("episode_id", p.name), s))

    if not standoffs:
        print("(no standoff data available)")
        return 0.0, False

    print(f"  {'episode':<16}  {'x':>9}  {'y':>9}  {'z':>9}")
    for ep_id, s in standoffs:
        print(f"  {ep_id:<16}  {s[0]:+9.4f}  {s[1]:+9.4f}  {s[2]:+9.4f}")

    arr = np.stack([s for _, s in standoffs], axis=0)  # (N, 3)
    x_range = float(arr[:, 0].max() - arr[:, 0].min())
    y_range = float(arr[:, 1].max() - arr[:, 1].min())
    z_range = float(arr[:, 2].max() - arr[:, 2].min())
    means   = arr.mean(axis=0)

    print()
    print(f"  range:  x={x_range*100:5.2f}cm  y={y_range*100:5.2f}cm  z={z_range*100:5.2f}cm")
    print(f"  mean :  x={means[0]:+.4f}  y={means[1]:+.4f}  z={means[2]:+.4f}")

    # Detect monotonic trend in chronological order. Files are passed newest-
    # first; reverse so we iterate oldest→newest for trend interpretation.
    x_chrono = arr[::-1, 0]
    diffs = np.diff(x_chrono)
    is_mono_inc = bool(np.all(diffs > 0)) and len(diffs) >= 2
    is_mono_dec = bool(np.all(diffs < 0)) and len(diffs) >= 2
    trend_flag = is_mono_inc or is_mono_dec

    if trend_flag:
        direction = "increasing" if is_mono_inc else "decreasing"
        magnitude = abs(x_chrono[-1] - x_chrono[0])
        print(f"  >> DRIFT DETECTED: x is monotonically {direction} across "
              f"{len(x_chrono)} episodes ({magnitude*100:.2f} cm total).")
    elif x_range > DRIFT_X_FLAG_M:
        print(f"  >> NOTE: x range > {DRIFT_X_FLAG_M*100:.0f} cm "
              f"({x_range*100:.2f} cm) — variability without monotonic trend.")
    return x_range, trend_flag


# ──────────────────────────────────────────────
# Section 3 — Rear-leg tracking quality
# ──────────────────────────────────────────────

def _episode_tracking_stats(
    actual: np.ndarray, cmd: np.ndarray, phase: np.ndarray,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Per-joint (max, mean) of |actual - cmd| during {lift, extend, hold}."""
    mask = np.isin(phase, ACTIVE_LABELS)
    if not mask.any():
        return None
    err = np.abs(actual[mask] - cmd[mask])  # (T_active, 12)
    out: Dict[str, Tuple[float, float]] = {}
    for name, idx in SUPPORT_JOINTS:
        col = err[:, idx]
        out[name] = (float(col.max()), float(col.mean()))
    return out


def _print_tracking_section(
    paths: List[Path], rows: List[Dict[str, str]]
) -> Tuple[int, int]:
    """Return (n_flag_warnings, n_severe_failures)."""
    print()
    print("─" * 100)
    print("Section 3 — Rear-leg tracking quality (|q_actual - q_cmd| during "
          "lift+extend+hold)")
    print("─" * 100)

    flag_warns = 0
    severe = 0
    for p, r in zip(paths, rows):
        loaded = _load_tracking(p)
        if loaded is None:
            print(f"  {r.get('episode_id', p.name):<16}  (no tracking data)")
            continue
        actual, cmd, phase = loaded
        stats = _episode_tracking_stats(actual, cmd, phase)
        if stats is None:
            print(f"  {r.get('episode_id', p.name):<16}  (no active-phase steps)")
            continue
        ep_id = r.get("episode_id", p.name)
        parts = []
        for name, _idx in SUPPORT_JOINTS:
            mx, mn = stats[name]
            tag = ""
            if mx > MAX_ERR_FAIL_RAD:
                tag = "*"
                severe += 1
            elif mx > MAX_ERR_FLAG_RAD or mn > MEAN_ERR_FLAG_RAD:
                tag = "!"
                flag_warns += 1
            parts.append(f"{name} max={mx:.3f} mean={mn:.3f}{tag}")
        print(f"  {ep_id:<16}  " + " | ".join(parts))

    print()
    print(f"  thresholds: max-error flag {MAX_ERR_FLAG_RAD:.2f} rad (!),  "
          f"mean-error flag {MEAN_ERR_FLAG_RAD:.2f} rad (!),  "
          f"severe failure {MAX_ERR_FAIL_RAD:.2f} rad (*)")
    return flag_warns, severe


# ──────────────────────────────────────────────
# Section 4 — Verdict
# ──────────────────────────────────────────────

def _print_verdict(
    *,
    metadata_missing: int,
    drift_flag: bool,
    drift_range_m: float,
    tracking_flags: int,
    tracking_severe: int,
    n_episodes_loaded: int,
) -> None:
    print()
    print("─" * 100)
    print("Section 4 — Verdict")
    print("─" * 100)

    if n_episodes_loaded == 0:
        print("  INVESTIGATE BEFORE PROCEEDING — no episodes could be loaded.")
        return

    metadata_bad = metadata_missing >= n_episodes_loaded * 2
    drift_bad    = drift_flag or drift_range_m > DRIFT_X_FLAG_M * 2
    tracking_bad = tracking_severe > 0

    severe_count = sum([metadata_bad, drift_bad, tracking_bad])

    if severe_count >= 1:
        print("  INVESTIGATE BEFORE PROCEEDING — see flagged items above:")
        if metadata_bad:
            print(f"    - {metadata_missing} MISSING metadata fields across "
                  f"{n_episodes_loaded} episodes")
        if drift_bad:
            if drift_flag:
                print(f"    - monotonic standoff drift detected")
            else:
                print(f"    - standoff x range {drift_range_m*100:.2f} cm "
                      f"(> {DRIFT_X_FLAG_M*100*2:.0f} cm)")
        if tracking_bad:
            print(f"    - {tracking_severe} severe tracking failure(s) "
                  f"(max error > {MAX_ERR_FAIL_RAD:.2f} rad)")
        return

    minor_count = (
        (1 if metadata_missing > 0 else 0)
        + (1 if drift_range_m > DRIFT_X_FLAG_M else 0)
        + (1 if tracking_flags > 0 else 0)
    )
    if minor_count > 0:
        print("  ATTENTION: drift or tracking concerns — see flagged items above.")
        if metadata_missing > 0:
            print(f"    - {metadata_missing} MISSING metadata field(s) "
                  f"across {n_episodes_loaded} episodes")
        if drift_range_m > DRIFT_X_FLAG_M:
            print(f"    - standoff x range {drift_range_m*100:.2f} cm "
                  f"(> {DRIFT_X_FLAG_M*100:.0f} cm)")
        if tracking_flags > 0:
            print(f"    - {tracking_flags} tracking warning(s) "
                  f"(max > {MAX_ERR_FLAG_RAD:.2f} rad or mean > "
                  f"{MEAN_ERR_FLAG_RAD:.2f} rad)")
        return

    print(f"  READY FOR PRIMARY COLLECTION — {n_episodes_loaded} recent "
          f"episodes pass all checks.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect the N most recent v3 collection episodes for "
                    "metadata correctness, standoff drift, and rear-leg "
                    "tracking quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Episode HDF5 directory.",
    )
    p.add_argument(
        "--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
        help="How many recent episodes (by mtime) to inspect.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    print("=" * 100)
    print(f"v4 calibration inspection — directory: {args.data_dir}")
    print(f"                            episodes : {args.num_episodes} most recent")
    print("=" * 100)

    paths = _list_recent_episodes(args.data_dir, args.num_episodes)
    if not paths:
        print(f"No episode_*.h5 files found in {args.data_dir}.")
        return 1

    rows: List[Dict[str, str]] = []
    kept_paths: List[Path] = []
    for p in paths:
        row = _load_metadata_row(p)
        if row is None:
            continue
        rows.append(row)
        kept_paths.append(p)

    if not rows:
        print(f"No episode files in {args.data_dir} could be opened.")
        return 1

    metadata_missing = _print_metadata_section(rows)
    drift_range, drift_flag = _print_standoff_section(kept_paths, rows)
    tracking_flags, tracking_severe = _print_tracking_section(kept_paths, rows)
    _print_verdict(
        metadata_missing=metadata_missing,
        drift_flag=drift_flag,
        drift_range_m=drift_range,
        tracking_flags=tracking_flags,
        tracking_severe=tracking_severe,
        n_episodes_loaded=len(rows),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
