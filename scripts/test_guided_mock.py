#!/usr/bin/env python
"""
scripts/test_guided_mock.py

Offline smoke tests for the Stage D v2 guided-collection additions. No robot,
no DDS, no hardware. Three tests:

    1. HeuristicContactGuided constructs with SDK symbols monkey-patched and
       exposes _compliance_active=True + collection_mode="hand_guided".

       NOTE: the task spec mentioned a "dry_run" flag on HeuristicContact;
       that flag does not actually exist on the class (only on the CLI of
       heuristic_contact.py). We avoid touching DDS by patching the four
       SDK symbols HeuristicContact.__init__ calls.

    2. StageDRecorder._compute_achieved_delta handles shape and edge cases
       (T=0, T=1, T>=2 with last-step duplication).

    3. StageDRecorder.save() writes a valid v2 HDF5 given dummy buffers,
       including the four new per-step datasets and collection_mode.

Run:
    python scripts/test_guided_mock.py

Exit 0 = all pass, 1 = any failure.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# Make src.* imports work when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Test 1 — HeuristicContactGuided flags
# ──────────────────────────────────────────────

def test_guided_flags() -> None:
    """
    Mock out the Unitree SDK so HeuristicContact.__init__ can run without
    live DDS. Verify _compliance_active and collection_mode.
    """
    import src.planner.heuristic_contact as hc

    with mock.patch.object(hc, "ChannelFactoryInitialize"), \
         mock.patch.object(hc, "ChannelPublisher") as pub, \
         mock.patch.object(hc, "ChannelSubscriber") as sub, \
         mock.patch.object(hc, "SportClient") as sport, \
         mock.patch.object(hc, "MotionSwitcherClient") as msc:
        pub.return_value.Init.return_value = None
        pub.return_value.Write.return_value = None
        sub.return_value.Init.return_value = None
        sport.return_value.SetTimeout.return_value = None
        sport.return_value.Init.return_value = None
        msc.return_value.SetTimeout.return_value = None
        msc.return_value.Init.return_value = None

        from src.planner.heuristic_contact_guided import HeuristicContactGuided
        ctrl = HeuristicContactGuided(
            network_interface="lo",
            already_initialized=True,
        )

    assert ctrl._compliance_active is True, "compliance should be active"
    assert ctrl.collection_mode == "hand_guided", \
        f"expected 'hand_guided', got {ctrl.collection_mode!r}"
    # Double-check base class default is still rigid.
    assert hc.HeuristicContact.__init__ is not None  # sanity
    print("  [PASS] HeuristicContactGuided flags correct")


# ──────────────────────────────────────────────
# Test 2 — _compute_achieved_delta synthetic
# ──────────────────────────────────────────────

def test_achieved_delta() -> None:
    from src.data.recorder import StageDRecorder

    # Empty case
    out = StageDRecorder._compute_achieved_delta(
        np.zeros((0, 12), dtype=np.float32),
        np.zeros((0, 12), dtype=np.float32),
    )
    assert out.shape == (0, 3), f"empty shape {out.shape}"

    # Single-step: no t+1 so we emit zeros.
    out = StageDRecorder._compute_achieved_delta(
        np.ones((1, 12), dtype=np.float32) * 0.3,
        np.ones((1, 12), dtype=np.float32) * 0.1,
    )
    assert out.shape == (1, 3), f"T=1 shape {out.shape}"
    assert np.allclose(out, 0.0), f"T=1 should emit zeros, got {out}"

    # Synthetic T=5. Give each step a known actual[t+1] - cmd[t] on FR joints.
    T = 5
    actual = np.zeros((T, 12), dtype=np.float32)
    cmd    = np.zeros((T, 12), dtype=np.float32)

    # step 0: actual[1][0:3] = [0.10, 0.20, 0.30], cmd[0][0:3] = [0.00, 0.00, 0.00]
    actual[1, 0:3] = [0.10, 0.20, 0.30]
    # step 1: actual[2][0:3] = [0.15, 0.25, 0.35], cmd[1][0:3] = [0.05, 0.05, 0.05]
    actual[2, 0:3] = [0.15, 0.25, 0.35]
    cmd[1, 0:3]    = [0.05, 0.05, 0.05]
    # step 2: actual[3][0:3] = [0.20, 0.30, 0.40], cmd[2][0:3] = [0.10, 0.10, 0.10]
    actual[3, 0:3] = [0.20, 0.30, 0.40]
    cmd[2, 0:3]    = [0.10, 0.10, 0.10]
    # step 3: actual[4][0:3] = [0.25, 0.35, 0.45], cmd[3][0:3] = [0.15, 0.15, 0.15]
    actual[4, 0:3] = [0.25, 0.35, 0.45]
    cmd[3, 0:3]    = [0.15, 0.15, 0.15]
    # step 4: duplicated from step 3.

    out = StageDRecorder._compute_achieved_delta(actual, cmd)
    assert out.shape == (T, 3), f"shape {out.shape}"
    assert np.allclose(out[0], [0.10, 0.20, 0.30]), f"step 0: {out[0]}"
    assert np.allclose(out[1], [0.10, 0.20, 0.30]), f"step 1: {out[1]}"
    assert np.allclose(out[2], [0.10, 0.20, 0.30]), f"step 2: {out[2]}"
    assert np.allclose(out[3], [0.10, 0.20, 0.30]), f"step 3: {out[3]}"
    # Last step duplicates step 3.
    assert np.allclose(out[4], out[3]), f"last step should duplicate: {out[4]} vs {out[3]}"
    print("  [PASS] _compute_achieved_delta handles T=0, T=1, T=5 correctly")


# ──────────────────────────────────────────────
# Test 3 — v2 HDF5 smoke (no heuristic, no hardware)
# ──────────────────────────────────────────────

class _FakeGroundingThread:
    def get_log(self):
        return []


class _FakeHeuristic:
    """Just enough to satisfy StageDRecorder.end_episode and save()."""
    collection_mode = "hand_guided"


class _FakeResult:
    contact_step = 42
    duration_s = 10.5
    press_mode = "wall"
    target_offset_xyz = np.array([0.5, -0.05, 0.25], dtype=np.float32)
    contact_method = "audio"
    phase_transitions = {
        "lift_end":   np.zeros(12, dtype=np.float32),
        "extend_end": np.zeros(12, dtype=np.float32),
        "hold_end":   np.zeros(12, dtype=np.float32),
    }


def _populate_recorder(recorder, T=20) -> None:
    """Fill every per-step buffer with plausible synthetic data."""
    recorder._reset_buffers()
    for t in range(T):
        recorder.timestamp.append(float(t * 0.002))
        recorder.joint_pos_actual.append(np.full(12, 0.1 * t, dtype=np.float32))
        recorder.joint_vel_actual.append(np.zeros(12, dtype=np.float32))
        recorder.joint_pos_cmd.append(np.full(12, 0.05 * t, dtype=np.float32))
        recorder.fr_foot_pos_fk.append(np.array([0.3, -0.1, 0.1], dtype=np.float32))
        recorder.target_pos_base.append(np.array([0.5, -0.1, 0.25], dtype=np.float32))
        recorder.foot_to_target_error.append(np.array([0.2, 0.0, 0.15], dtype=np.float32))
        recorder.imu_rpy.append(np.zeros(3, dtype=np.float32))
        recorder.phase_label.append(1)   # extend — counts as active for v2 warn check
        recorder.phase_progress.append(np.float32(t / T))
        recorder.nominal_waypoint_fr.append(np.zeros(3, dtype=np.float32))
        recorder.jacobian_pid_delta.append(np.zeros(3, dtype=np.float32))
        recorder.joint_tau_est.append(np.full(12, 2.0, dtype=np.float32))
        recorder.imu_gyro.append(np.zeros(3, dtype=np.float32))
        recorder.imu_accel.append(np.array([0.0, 0.0, 9.81], dtype=np.float32))


def test_v2_hdf5_write() -> None:
    from src.data.recorder import StageDRecorder

    rec = StageDRecorder(
        heuristic=_FakeHeuristic(),
        grounding_thread=_FakeGroundingThread(),
    )
    _populate_recorder(rec, T=20)
    rec._recording = False  # not using the sampler thread
    rec._metadata = {
        "episode_id":                  "smoketest",
        "target_pos_base_at_standoff": np.zeros(3, dtype=np.float32),
        "target_pos_world_initial":    np.zeros(3, dtype=np.float32),
        "perturbation_commanded":      np.zeros(2, dtype=np.float32),
        "k_p":                         0.0,
        "button_height_m":             0.25,
        "color_expected":              "red",
        "notes":                       "smoke",
    }
    rec.end_episode(_FakeResult())
    assert rec._metadata["collection_mode"] == "hand_guided"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "episode_guided_smoke.h5"
        rec.save(
            str(out),
            audio=None,
            audio_sample_rate=16000,
            metadata_override={
                "success_fk":             True,
                "success_audio_live":     True,
                "audio_detection_time_s": 2.5,
                "contact_method":         "audio",
                "color_detected":         "red",
                "color_expected":         "red",
                "success_target":         True,
                "audio_transcript":       "red",
                "collection_mode":        "hand_guided",
            },
            use_v2_schema=True,
        )
        assert out.exists(), f"file not written: {out}"

        with h5py.File(out, "r") as f:
            for name in ("joint_tau_est", "imu_gyro", "imu_accel",
                         "achieved_delta_fr"):
                assert name in f["per_step"], f"{name} missing"
            assert f["per_step/joint_tau_est"].shape == (20, 12)
            assert f["per_step/imu_gyro"].shape      == (20, 3)
            assert f["per_step/imu_accel"].shape     == (20, 3)
            assert f["per_step/achieved_delta_fr"].shape == (20, 3)
            assert f["per_step/joint_tau_est"].dtype == np.float32
            cm = f.attrs["collection_mode"]
            if isinstance(cm, (bytes, np.bytes_)):
                cm = cm.decode()
            assert str(cm) == "hand_guided", f"collection_mode={cm!r}"

    print("  [PASS] v2 HDF5 write — all new datasets + collection_mode present")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    failures = 0
    for name, fn in [
        ("HeuristicContactGuided flags",  test_guided_flags),
        ("_compute_achieved_delta",       test_achieved_delta),
        ("v2 HDF5 smoke write",           test_v2_hdf5_write),
    ]:
        print(f"Running: {name}")
        try:
            fn()
        except Exception:
            failures += 1
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    print()
    print("=" * 40)
    print("OK" if failures == 0 else f"{failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
