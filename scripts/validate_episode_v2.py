#!/usr/bin/env python
"""
scripts/validate_episode_v2.py

Per-episode HDF5 sanity check for Stage D v2 files. Reuses the full v2.1
validator (scripts/validate_episode.py) and layers on the four new v2
per-step fields plus the collection_mode root attr.

Usage:
    python scripts/validate_episode_v2.py path/to/episode_guided_*.h5

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

# Reuse the v2.1 validator verbatim; its checks still apply to v2 files.
from scripts.validate_episode import (
    validate_episode as validate_episode_v21,
    _attr_to_str,
)


V2_REQUIRED_PER_STEP_FIELDS = [
    ("joint_tau_est",     (None, 12)),
    ("imu_gyro",          (None, 3)),
    ("imu_accel",         (None, 3)),
    ("achieved_delta_fr", (None, 3)),
]

VALID_COLLECTION_MODES = {"jacobian_pid", "hand_guided"}


def _relax_v21_audio_length_check(v21_res, path: Path) -> None:
    """
    The v2.1 validator gates audio length at ±10% of duration_s * sample_rate.
    That is too strict for collect_guided.py output: the AudioRecorder stream
    runs from start_episode() (before heuristic.execute() is called) to
    end_episode() (after it returns), which wraps StandDown, ReleaseMode,
    the control-loop window itself, gain zeroing, and SelectMode (whose
    3 s timeout often fires). duration_s measures only the inner control-loop
    window, so audio buffers are typically 1.5–2.5x the naive expected length.
    The extra samples are silent and don't affect AudioLiveDetector (separate
    stream) or Whisper post-processing (long-audio-tolerant). We remove the
    strict item from the v2.1 CheckResult and replace it with a relaxed
    [0.9x, 3.0x] band check. The v2.1 file itself is unchanged.
    """
    label_prefix = "audio length matches duration_s * sample_rate"
    v21_res.items = [
        item for item in v21_res.items
        if not item[0].startswith(label_prefix)
    ]
    try:
        with h5py.File(path, "r") as f:
            if "audio" not in f:
                return
            audio_data = f["audio"]
            duration_s = float(f.attrs.get("duration_s", 0.0))
            audio_sample_rate = int(f.attrs.get("audio_sample_rate", 0))
            expected_audio_samples = duration_s * audio_sample_rate
            actual_audio_samples = audio_data.shape[0]
            ratio = (actual_audio_samples / expected_audio_samples
                     if expected_audio_samples > 0 else 0)
            ok = 0.9 <= ratio <= 3.0
            v21_res.check(
                "audio length within [0.9x, 3.0x] of duration_s * sample_rate",
                ok,
                f"expected≈{int(expected_audio_samples)} got {actual_audio_samples} "
                f"(ratio={ratio:.2f}x)",
            )
    except Exception as e:
        v21_res.check("audio length (v2 relaxed)", False, f"exception: {e}")


def validate_episode_v2(path: Path):
    """
    Run the v2.1 validator, then apply the v2-only checks.

    Returns: (v21_res, v2_items)
        v21_res: CheckResult from validate_episode
        v2_items: list[(label, ok_or_warn, detail)]; ok_or_warn ∈ {"PASS","FAIL","WARN"}
    """
    v21_res = validate_episode_v21(path)
    # v2-specific relaxation of the audio-length tolerance; see helper docstring.
    _relax_v21_audio_length_check(v21_res, path)
    v2_items: list[tuple[str, str, str]] = []

    if not path.exists():
        v2_items.append(("File exists", "FAIL", f"{path}"))
        return v21_res, v2_items

    try:
        f = h5py.File(path, "r")
    except Exception as e:
        v2_items.append(("HDF5 open", "FAIL", f"{e}"))
        return v21_res, v2_items

    try:
        # ── per_step v2 fields present + shape ──
        if "per_step" not in f:
            v2_items.append(("per_step group exists", "FAIL", ""))
            return v21_res, v2_items
        grp = f["per_step"]

        T_ref = None
        if "timestamp" in grp:
            T_ref = grp["timestamp"].shape[0]

        for name, (_T, last_dim) in V2_REQUIRED_PER_STEP_FIELDS:
            present = name in grp
            v2_items.append((f"per_step/{name} present",
                             "PASS" if present else "FAIL", ""))
            if not present:
                continue

            arr = grp[name]
            if len(arr.shape) != 2:
                v2_items.append((f"per_step/{name} is 2-D",
                                 "FAIL", f"shape={arr.shape}"))
                continue
            ok_last = arr.shape[1] == last_dim
            v2_items.append((f"per_step/{name} shape[-1] == {last_dim}",
                             "PASS" if ok_last else "FAIL",
                             f"got {arr.shape[1]}"))
            if T_ref is not None:
                ok_T = arr.shape[0] == T_ref
                v2_items.append((f"per_step/{name} length == T ({T_ref})",
                                 "PASS" if ok_T else "FAIL",
                                 f"got {arr.shape[0]}"))
            ok_dtype = arr.dtype == np.float32
            v2_items.append((f"per_step/{name} dtype is float32",
                             "PASS" if ok_dtype else "FAIL",
                             f"got {arr.dtype}"))
            has_nan = bool(np.any(~np.isfinite(arr[()])))
            v2_items.append((f"per_step/{name} has no NaN/Inf",
                             "PASS" if not has_nan else "FAIL", ""))

        # ── collection_mode root attr ──
        has_cm = "collection_mode" in f.attrs
        v2_items.append(("root attr 'collection_mode' present",
                         "PASS" if has_cm else "FAIL", ""))
        cm_str = None
        if has_cm:
            cm_str = _attr_to_str(f.attrs["collection_mode"])
            ok_cm = cm_str in VALID_COLLECTION_MODES
            v2_items.append((
                f"collection_mode in {sorted(VALID_COLLECTION_MODES)}",
                "PASS" if ok_cm else "FAIL",
                f"got '{cm_str}'",
            ))

        # ── hand_guided only: achieved_delta_fr should show non-trivial
        # motion during extend+hold, because the human is moving the paw.
        # This is a WARNING, not a hard failure — an episode where the
        # operator froze won't pass but is still valid training data.
        if cm_str == "hand_guided" \
                and "achieved_delta_fr" in grp \
                and "phase_label" in grp:
            delta = grp["achieved_delta_fr"][()]
            labels = grp["phase_label"][()]
            # PHASE_TO_LABEL: lift=0, extend=1, hold=2 (from recorder.py)
            active_mask = (labels == 1) | (labels == 2)
            n_active = int(active_mask.sum())
            if n_active == 0:
                v2_items.append(("achieved_delta_fr: extend+hold coverage",
                                 "WARN",
                                 "no extend/hold steps were logged"))
            else:
                # "Non-trivial" = per-step max abs delta across FR joints > 0.05 rad
                per_step_max = np.max(np.abs(delta[active_mask]), axis=1)
                frac_moving = float((per_step_max > 0.05).sum() / n_active)
                if frac_moving >= 0.10:
                    v2_items.append((
                        "achieved_delta_fr non-trivial during extend+hold (≥10%)",
                        "PASS",
                        f"{frac_moving*100:.1f}% of {n_active} active steps",
                    ))
                else:
                    v2_items.append((
                        "achieved_delta_fr non-trivial during extend+hold (≥10%)",
                        "WARN",
                        f"{frac_moving*100:.1f}% of {n_active} active steps "
                        "— operator may have barely moved the paw",
                    ))
    finally:
        f.close()

    return v21_res, v2_items


def _print_v2_items(items):
    for label, status, detail in items:
        mark = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
        line = f"  [{mark}] {label}"
        if detail:
            line += f"  — {detail}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate a single Stage D v2 episode HDF5.")
    ap.add_argument("path", type=str)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    print(f"Validating (v2): {path}")
    print("-" * 72)
    print("Part 1 — v2.1 base schema")
    print("-" * 72)
    v21_res, v2_items = validate_episode_v2(path)
    v21_res.print()

    print()
    print("-" * 72)
    print("Part 2 — v2 additive schema")
    print("-" * 72)
    _print_v2_items(v2_items)

    v2_hard_fail = any(st == "FAIL" for _, st, _ in v2_items)
    v2_ok = (not v2_hard_fail)
    overall_ok = v21_res.pass_all() and v2_ok
    print()
    print("OVERALL (v2): " + ("PASS" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
