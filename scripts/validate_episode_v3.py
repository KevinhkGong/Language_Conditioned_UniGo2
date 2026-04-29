#!/usr/bin/env python
"""
scripts/validate_episode_v3.py

Per-episode HDF5 sanity check for Stage D v3 (whole-body) files. Reuses
the v2 validator (which itself wraps the v2.1 validator) and layers v3
additive checks on top:

  - Root attr ``gain_schedule`` present and starts with ``"wholebody"``.
  - Root attr ``collection_mode`` == ``"wholebody_guided"`` (overrides the
    v2 acceptance set, which permits hand_guided).
  - per_step/joint_kp_used present, shape (T, 12), float32.
  - During extend + hold: support-leg (joints 3..11) KP values within
    ±5 of KP_SUPPORT_SOFT (= 45). Outside this band → WARN.
  - During lift: support-leg KP values within ±5 of KP_STABLE (= 100).
    Outside this band → WARN.
  - INFO: support-leg achieved_delta_q std during extend+hold should be
    ≥0.04 rad on at least one rear-hip or rear-thigh dim — that's the
    signal that the whole-body demonstration was active. Lower than
    that → WARN that the demonstration may have been too cautious or
    the gain schedule was too stiff.

Usage:
    python scripts/validate_episode_v3.py path/to/episode_wholebody_*.h5

Exit codes:
    0 — all hard checks pass (warnings do not fail)
    1 — at least one hard check fails
    2 — file could not be opened
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running as `python scripts/validate_episode_v3.py …` by putting the
# repo root on sys.path. Without this, the cross-script imports below resolve
# only when the file is invoked as `python -m scripts.validate_episode_v3`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse v2 (which reuses v2.1).
from scripts.validate_episode_v2 import (
    validate_episode_v2,
    _print_v2_items,
)
from scripts.validate_episode import _attr_to_str

from src.planner.heuristic_contact_wholebody import (
    KP_SUPPORT_SOFT,
)
from src.planner.heuristic_contact import KP_STABLE


# Phase id constants from src/data/recorder.py (PHASE_TO_LABEL).
PHASE_LIFT, PHASE_EXTEND, PHASE_HOLD = 0, 1, 2

# Tolerance band on observed KP values vs the prescribed schedule.
# ±5 covers the discrete transition steps where the gain has just been
# changed but a stale low_cmd snapshot may have leaked through.
KP_TOLERANCE = 5.0

# Whole-body activity signal: rear-hip or rear-thigh achieved-delta std
# during extend+hold should clear this threshold to call the
# demonstration "active". Lower → WARN.
SUPPORT_DELTA_STD_MIN = 0.04

SUPPORT_JOINT_INDICES = list(range(3, 12))  # FL, RL, RR
REAR_JOINTS_FOR_SIGNAL = (6, 7, 9, 10)      # RR_hip, RR_thigh, RL_hip, RL_thigh


def _fr_indices() -> list[int]:
    return [0, 1, 2]


def _check_kp_band(
    arr: np.ndarray,
    expected: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """Return (mean, min, max) over the supplied joint slice."""
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.min()), float(arr.max())


def validate_episode_v3(path: Path):
    """
    Run v2 validator, then layer v3 checks on top.

    Returns:
        v21_res: CheckResult from the underlying v2.1 base validator.
        v2_items: list[(label, status, detail)] from the v2 validator.
        v3_items: list[(label, status, detail)] for v3-specific checks.
            status ∈ {"PASS","FAIL","WARN","INFO"}.
    """
    v21_res, v2_items = validate_episode_v2(path)
    v3_items: list[tuple[str, str, str]] = []

    if not path.exists():
        v3_items.append(("File exists", "FAIL", f"{path}"))
        return v21_res, v2_items, v3_items

    try:
        f = h5py.File(path, "r")
    except Exception as e:
        v3_items.append(("HDF5 open", "FAIL", f"{e}"))
        return v21_res, v2_items, v3_items

    try:
        # ── gain_schedule root attr ──
        has_gs = "gain_schedule" in f.attrs
        v3_items.append(("root attr 'gain_schedule' present",
                         "PASS" if has_gs else "FAIL", ""))
        gs_str = None
        if has_gs:
            gs_str = _attr_to_str(f.attrs["gain_schedule"])
            ok = gs_str.startswith("wholebody")
            v3_items.append((
                "gain_schedule starts with 'wholebody'",
                "PASS" if ok else "FAIL",
                f"got '{gs_str}'",
            ))

        # ── collection_mode override for v3 ──
        # The v2 validator already required collection_mode ∈
        # {jacobian_pid, hand_guided}. v3 widens that set, so v2's check
        # FAILS on a wholebody_guided file. We check the v3-specific
        # required value here and demote the v2 false-fail.
        if "collection_mode" in f.attrs:
            cm_str = _attr_to_str(f.attrs["collection_mode"])
            ok_cm = cm_str == "wholebody_guided"
            v3_items.append((
                "collection_mode == 'wholebody_guided'",
                "PASS" if ok_cm else "FAIL",
                f"got '{cm_str}'",
            ))
            # Demote the v2 acceptance-set fail to INFO so the overall
            # PASS judgment isn't blocked.
            for i, (label, status, detail) in enumerate(v2_items):
                if label.startswith("collection_mode in") and status == "FAIL":
                    v2_items[i] = (label, "INFO", detail
                                   + "  (v3 file: superseded by v3 check)")

        # ── per_step/joint_kp_used presence + shape ──
        if "per_step" not in f:
            v3_items.append(("per_step group exists", "FAIL", ""))
            return v21_res, v2_items, v3_items

        grp = f["per_step"]
        present_kp = "joint_kp_used" in grp
        v3_items.append(("per_step/joint_kp_used present",
                         "PASS" if present_kp else "FAIL", ""))
        if not present_kp:
            return v21_res, v2_items, v3_items

        kp = grp["joint_kp_used"][()]
        T_ref = grp["timestamp"].shape[0] if "timestamp" in grp else None

        ok_dim = kp.ndim == 2 and kp.shape[1] == 12
        v3_items.append(("joint_kp_used shape (T, 12)",
                         "PASS" if ok_dim else "FAIL",
                         f"shape={kp.shape}"))
        ok_dtype = kp.dtype == np.float32
        v3_items.append(("joint_kp_used dtype is float32",
                         "PASS" if ok_dtype else "FAIL",
                         f"got {kp.dtype}"))
        if T_ref is not None:
            ok_T = kp.shape[0] == T_ref
            v3_items.append((
                f"joint_kp_used length == T ({T_ref})",
                "PASS" if ok_T else "FAIL", f"got {kp.shape[0]}"))

        if "phase_label" not in grp:
            v3_items.append(("phase_label present", "FAIL", ""))
            return v21_res, v2_items, v3_items

        phase = grp["phase_label"][()]
        if not ok_dim:
            return v21_res, v2_items, v3_items

        # ── KP values vs prescribed schedule ──
        compliance_mask = (phase == PHASE_EXTEND) | (phase == PHASE_HOLD)
        lift_mask       = phase == PHASE_LIFT

        if int(compliance_mask.sum()) > 0:
            sup = kp[compliance_mask][:, SUPPORT_JOINT_INDICES]
            mean, lo, hi = _check_kp_band(sup, KP_SUPPORT_SOFT, KP_TOLERANCE)
            band_ok = abs(mean - KP_SUPPORT_SOFT) <= KP_TOLERANCE
            v3_items.append((
                f"support KP during extend+hold ≈ {KP_SUPPORT_SOFT:.0f} (±{KP_TOLERANCE:.0f})",
                "PASS" if band_ok else "WARN",
                f"mean={mean:.1f} min={lo:.1f} max={hi:.1f} "
                f"(n={int(compliance_mask.sum())} steps)",
            ))
        else:
            v3_items.append((
                "support KP during extend+hold",
                "WARN", "no extend/hold steps logged"))

        if int(lift_mask.sum()) > 0:
            sup = kp[lift_mask][:, SUPPORT_JOINT_INDICES]
            mean, lo, hi = _check_kp_band(sup, KP_STABLE, KP_TOLERANCE)
            band_ok = abs(mean - KP_STABLE) <= KP_TOLERANCE
            v3_items.append((
                f"support KP during lift ≈ {KP_STABLE:.0f} (±{KP_TOLERANCE:.0f})",
                "PASS" if band_ok else "WARN",
                f"mean={mean:.1f} min={lo:.1f} max={hi:.1f} "
                f"(n={int(lift_mask.sum())} steps)",
            ))
        else:
            v3_items.append((
                "support KP during lift",
                "WARN", "no lift steps logged"))

        # ── support-leg activity signal (info-level) ──
        # Per-joint achieved delta during extend+hold. The v3 dataset
        # loader reconstructs delta_q at load time, but here we approximate
        # it from the per-step joint_pos_actual / joint_pos_cmd that are
        # already on disk.
        if ("joint_pos_actual" in grp and "joint_pos_cmd" in grp
                and int(compliance_mask.sum()) > 1):
            jpa = grp["joint_pos_actual"][()]
            jpc = grp["joint_pos_cmd"][()]
            delta = np.zeros_like(jpa)
            delta[:-1] = jpa[1:] - jpc[:-1]
            delta[-1]  = delta[-2]
            sub = delta[compliance_mask]

            # Per-joint std on the rear-hip/rear-thigh dims, find the max.
            rear_stds = [(j, float(sub[:, j].std())) for j in REAR_JOINTS_FOR_SIGNAL]
            best_j, best_std = max(rear_stds, key=lambda jv: jv[1])
            joint_label = {6: "RR_hip", 7: "RR_thigh", 9: "RL_hip", 10: "RL_thigh"}[best_j]

            if best_std >= SUPPORT_DELTA_STD_MIN:
                v3_items.append((
                    f"rear-leg achieved-delta std ≥ {SUPPORT_DELTA_STD_MIN:.2f} rad",
                    "PASS",
                    f"max at {joint_label}: std={best_std:.4f}",
                ))
            else:
                v3_items.append((
                    f"rear-leg achieved-delta std ≥ {SUPPORT_DELTA_STD_MIN:.2f} rad",
                    "WARN",
                    f"max at {joint_label}: std={best_std:.4f} — "
                    "demonstration may have been too cautious or gains too stiff",
                ))
            # Also report all four for visibility.
            joints_summary = ", ".join(
                f"{{6:'RR_hip',7:'RR_thigh',9:'RL_hip',10:'RL_thigh'}}[{j}]={s:.4f}"
                for j, s in rear_stds)
            v3_items.append((
                "per-rear-joint extend+hold std (info)",
                "INFO",
                "; ".join(f"j{j}={s:.4f}" for j, s in rear_stds),
            ))
        else:
            v3_items.append(("rear-leg activity signal",
                             "WARN", "insufficient extend/hold steps to compute"))

    finally:
        f.close()

    return v21_res, v2_items, v3_items


def _print_items(items, header: str) -> None:
    print(f"  {header}")
    for label, status, detail in items:
        mark = {"PASS": "✓", "FAIL": "✗",
                "WARN": "!", "INFO": "·"}.get(status, "?")
        line = f"  [{mark}] {label}"
        if detail:
            line += f"  — {detail}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate a single Stage D v3 (whole-body) episode HDF5.")
    ap.add_argument("path", type=str)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    print(f"Validating (v3): {path}")
    print("-" * 72)
    print("Part 1 — v2.1 base schema")
    print("-" * 72)
    v21_res, v2_items, v3_items = validate_episode_v3(path)
    v21_res.print()

    print()
    print("-" * 72)
    print("Part 2 — v2 additive schema")
    print("-" * 72)
    _print_v2_items(v2_items)

    print()
    print("-" * 72)
    print("Part 3 — v3 additive schema")
    print("-" * 72)
    _print_items(v3_items, header="")

    v2_hard_fail = any(st == "FAIL" for _, st, _ in v2_items)
    v3_hard_fail = any(st == "FAIL" for _, st, _ in v3_items)
    overall_ok = v21_res.pass_all() and not v2_hard_fail and not v3_hard_fail
    print()
    print("OVERALL (v3): " + ("PASS" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
