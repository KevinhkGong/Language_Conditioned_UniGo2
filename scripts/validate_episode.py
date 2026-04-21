#!/usr/bin/env python
"""
scripts/validate_episode.py

Per-episode HDF5 sanity check implementing the checklist from
training_data_spec.md v2.1, Section 11.1.

Usage:
    python scripts/validate_episode.py path/to/episode.h5

Exit codes:
    0 — all checks pass
    1 — at least one check fails
    2 — file could not be opened
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ──────────────────────────────────────────────
# Expected schema (spec Sections 5.5, 5.6, 5.7, 5.8)
# ──────────────────────────────────────────────

REQUIRED_ROOT_ATTRS = [
    # Episode identification and conditions
    "episode_id",
    "target_pos_base_at_standoff",
    "target_pos_world_initial",
    "perturbation_commanded",
    "k_p",
    "button_height_m",
    "notes",
    "press_mode",
    "target_offset_xyz",
    "contact_step",
    "duration_s",
    # v2.1 audio / success fields
    "success_fk",
    "success_audio_live",
    "audio_detection_time_s",
    "contact_method",
    "color_detected",
    "color_expected",
    "success_target",
    "audio_transcript",
    "audio_sample_rate",
]

REQUIRED_PER_STEP_FIELDS = [
    "timestamp",
    "joint_pos_actual",
    "joint_vel_actual",
    "joint_pos_cmd",
    "fr_foot_pos_fk",
    "target_pos_base",
    "foot_to_target_error",
    "imu_rpy",
    "phase_label",
    "phase_progress",
    "nominal_waypoint_fr",
    "jacobian_pid_delta",
]

REQUIRED_GROUNDING_FIELDS = [
    "timestamp",
    "position_base",
    "confidence",
    "depth_m",
    "position_base_valid",
]

MANDATORY_PHASE_TRANSITIONS = ["lift_end", "extend_end", "hold_end"]
VALID_CONTACT_METHODS = {"fk", "audio", "foot_force", "none"}


class CheckResult:
    """Accumulates pass/fail status with labels for final printing."""

    def __init__(self):
        self.items: list[tuple[str, bool, str]] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.items.append((label, ok, detail))
        return ok

    def pass_all(self) -> bool:
        return all(ok for _, ok, _ in self.items)

    def print(self) -> None:
        for label, ok, detail in self.items:
            mark = "✓" if ok else "✗"
            line = f"  [{mark}] {label}"
            if detail:
                line += f"  — {detail}"
            print(line)
        print()
        print("OVERALL: " + ("PASS" if self.pass_all() else "FAIL"))


def _attr_to_str(v):
    """Normalize HDF5 attribute (bytes, numpy scalar, etc.) to Python str."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.decode("utf-8", errors="replace")
    return str(v)


def validate_episode(path: Path) -> CheckResult:
    res = CheckResult()

    if not path.exists():
        res.check(f"File exists at {path}", False)
        return res

    try:
        f = h5py.File(path, "r")
    except Exception as e:
        res.check(f"HDF5 open", False, f"Exception: {e}")
        return res

    try:
        # ── 1. Root attributes present ─────────────────────────────────────
        for attr in REQUIRED_ROOT_ATTRS:
            res.check(f"root attr '{attr}' present", attr in f.attrs)

        # ── 2. per_step group + fields ────────────────────────────────────
        has_per_step = "per_step" in f
        res.check("per_step group exists", has_per_step)
        T = None
        if has_per_step:
            grp = f["per_step"]
            for field in REQUIRED_PER_STEP_FIELDS:
                exists = field in grp
                res.check(f"per_step/{field} exists", exists)
                if exists and T is None:
                    T = grp[field].shape[0]
                if exists and T is not None and grp[field].shape[0] != T:
                    res.check(
                        f"per_step/{field} length consistent (T={T})",
                        False,
                        f"got {grp[field].shape[0]}",
                    )
                else:
                    res.check(
                        f"per_step/{field} length consistent (T={T})", True)

            # NaN checks for mandatory-finite fields
            for mandatory_finite in ["joint_pos_actual", "joint_vel_actual",
                                     "joint_pos_cmd", "imu_rpy"]:
                if mandatory_finite in grp:
                    arr = grp[mandatory_finite][()]
                    has_nan = bool(np.any(~np.isfinite(arr)))
                    res.check(
                        f"per_step/{mandatory_finite} has no NaN/Inf",
                        not has_nan,
                    )

        # ── 3. grounding group + fields ───────────────────────────────────
        has_grounding = "grounding" in f
        res.check("grounding group exists", has_grounding)
        if has_grounding:
            gg = f["grounding"]
            G = None
            for field in REQUIRED_GROUNDING_FIELDS:
                exists = field in gg
                res.check(f"grounding/{field} exists", exists)
                if exists and G is None:
                    G = gg[field].shape[0]
                if exists and G is not None and gg[field].shape[0] != G:
                    res.check(
                        f"grounding/{field} length consistent (G={G})",
                        False,
                        f"got {gg[field].shape[0]}",
                    )
                else:
                    res.check(
                        f"grounding/{field} length consistent (G={G})", True)

        # ── 4. phase_transitions mandatory keys ───────────────────────────
        has_pt = "phase_transitions" in f
        res.check("phase_transitions group exists", has_pt)
        if has_pt:
            pt = f["phase_transitions"]
            for key in MANDATORY_PHASE_TRANSITIONS:
                present = key in pt
                res.check(f"phase_transitions/{key} present", present)
                if present:
                    shape_ok = pt[key].shape == (12,)
                    res.check(
                        f"phase_transitions/{key} shape == (12,)",
                        shape_ok,
                        f"got {pt[key].shape}",
                    )

        # ── 5. audio dataset length vs duration_s * sample_rate ±10% ──────
        has_audio = "audio" in f
        res.check("audio dataset present", has_audio)
        if has_audio:
            audio = f["audio"]
            res.check(
                "audio dtype is float32",
                audio.dtype == np.float32,
                f"got {audio.dtype}",
            )
            duration_s = float(f.attrs.get("duration_s", 0.0))
            sr = int(f.attrs.get("audio_sample_rate", 0))
            expected = duration_s * sr
            actual = audio.shape[0]
            # ±10% tolerance
            if expected > 0:
                lo, hi = expected * 0.9, expected * 1.1
                ok = lo <= actual <= hi
                res.check(
                    "audio length matches duration_s * sample_rate (±10%)",
                    ok,
                    f"expected≈{expected:.0f} got {actual}",
                )
            else:
                res.check(
                    "audio length matches duration_s * sample_rate (±10%)",
                    False,
                    "duration_s or sample_rate is zero",
                )

        # ── 6. contact_method is in allowed set ───────────────────────────
        if "contact_method" in f.attrs:
            cm = _attr_to_str(f.attrs["contact_method"])
            res.check(
                f"contact_method in {sorted(VALID_CONTACT_METHODS)}",
                cm in VALID_CONTACT_METHODS,
                f"got '{cm}'",
            )

        # ── 7. audio_detection_time_s valid when success_audio_live True ──
        if ("success_audio_live" in f.attrs
                and "audio_detection_time_s" in f.attrs):
            succ_aud = bool(f.attrs["success_audio_live"])
            det_t    = float(f.attrs["audio_detection_time_s"])
            dur_s    = float(f.attrs.get("duration_s", 0.0))
            if succ_aud:
                ok_finite = np.isfinite(det_t)
                res.check(
                    "audio_detection_time_s finite when success_audio_live",
                    ok_finite,
                    f"got {det_t}",
                )
                if ok_finite:
                    ok_range = 0.0 <= det_t <= dur_s + 1.0  # allow small slack
                    res.check(
                        "audio_detection_time_s within episode duration",
                        ok_range,
                        f"det_t={det_t:.2f}, duration_s={dur_s:.2f}",
                    )

        # ── 8. jacobian_pid_delta not identically zero for perturbed eps ──
        if ("perturbation_commanded" in f.attrs
                and "per_step" in f
                and "jacobian_pid_delta" in f["per_step"]):
            pert = np.asarray(f.attrs["perturbation_commanded"], dtype=np.float32)
            perturbed = float(np.linalg.norm(pert)) > 0.005
            if perturbed:
                jpd = f["per_step/jacobian_pid_delta"][()]
                max_abs = float(np.max(np.abs(jpd))) if jpd.size > 0 else 0.0
                nonzero = max_abs > 1e-5
                res.check(
                    "jacobian_pid_delta not identically zero "
                    "(perturbed episode)",
                    nonzero,
                    f"max |delta| = {max_abs:.3e}",
                )

        # ── 9. episode duration sanity ────────────────────────────────────
        if "duration_s" in f.attrs:
            dur = float(f.attrs["duration_s"])
            ok = 8.0 < dur < 40.0
            res.check(
                "8s < duration_s < 40s",
                ok,
                f"got {dur:.2f}s",
            )
    finally:
        f.close()

    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a single Stage D episode HDF5.")
    ap.add_argument("path", type=str, help="Path to episode .h5 file")
    args = ap.parse_args()

    path = Path(args.path)
    print(f"Validating: {path}")
    print("-" * 72)

    res = validate_episode(path)
    res.print()
    return 0 if res.pass_all() else 1


if __name__ == "__main__":
    sys.exit(main())