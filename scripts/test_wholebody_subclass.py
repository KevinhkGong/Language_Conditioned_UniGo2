#!/usr/bin/env python
"""
scripts/test_wholebody_subclass.py

Offline smoke tests for the Stage D v3 whole-body collection additions.
No robot, no DDS, no hardware. Three tests:

    1. HeuristicContactWholeBody constructs (with SDK symbols
       monkey-patched), exposes _compliance_active=True, collection_mode
       == "wholebody_guided", gain_schedule == "wholebody_v1_kpsupport45".

    2. _send_cmd writes the correct per-motor KP for each phase:
         lift           → FR @ KP_FR=40,           support @ KP_STABLE=100
         extend         → FR @ KP_FR_COMPLIANT=15, support @ KP_SUPPORT_SOFT=45
         hold           → FR @ KP_FR_COMPLIANT=15, support @ KP_SUPPORT_SOFT=45
         retract_curl   → FR @ KP_FR=40,           support @ KP_STABLE=100
         lower_to_sit   → all motors @ KP_SOFT=25

    3. StageDRecorder.save() writes a valid v3 HDF5 given dummy buffers
       — including the new per_step/joint_kp_used dataset and the
       gain_schedule root attr — when the heuristic exposes a
       gain_schedule attribute.

Run:
    python scripts/test_wholebody_subclass.py

Exit 0 = all pass, 1 = any failure.
"""

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
# Helpers
# ──────────────────────────────────────────────

class _FakeMotorCmd:
    """Per-motor command struct stub. Attributes mirror the SDK shape."""
    def __init__(self):
        self.q   = 0.0
        self.dq  = 0.0
        self.tau = 0.0
        self.kp  = 0.0
        self.kd  = 0.0
        self.mode = 0


class _FakeLowCmd:
    """LowCmd_ stub with motor_cmd[i] returning per-motor structs."""
    def __init__(self, n_motors: int = 20):
        self.motor_cmd = [_FakeMotorCmd() for _ in range(n_motors)]
        self.head = [0, 0]
        self.level_flag = 0
        self.gpio = 0
        self.crc = 0


def _patch_dds_and_construct():
    """Monkey-patch SDK symbols, then construct HeuristicContactWholeBody."""
    import src.planner.heuristic_contact as hc

    patches = (
        mock.patch.object(hc, "ChannelFactoryInitialize"),
        mock.patch.object(hc, "ChannelPublisher"),
        mock.patch.object(hc, "ChannelSubscriber"),
        mock.patch.object(hc, "SportClient"),
        mock.patch.object(hc, "MotionSwitcherClient"),
    )
    enters = [p.start() for p in patches]
    # Wire harmless return values where the constructor reads them.
    pub, sub, sport, msc = enters[1], enters[2], enters[3], enters[4]
    pub.return_value.Init.return_value = None
    pub.return_value.Write.return_value = None
    sub.return_value.Init.return_value = None
    sport.return_value.SetTimeout.return_value = None
    sport.return_value.Init.return_value = None
    msc.return_value.SetTimeout.return_value = None
    msc.return_value.Init.return_value = None

    from src.planner.heuristic_contact_wholebody import HeuristicContactWholeBody
    ctrl = HeuristicContactWholeBody(
        network_interface="lo",
        already_initialized=True,
    )
    return ctrl, patches


def _stop_patches(patches) -> None:
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass


# ──────────────────────────────────────────────
# Test 1 — flags
# ──────────────────────────────────────────────

def test_wholebody_flags() -> None:
    ctrl, patches = _patch_dds_and_construct()
    try:
        assert ctrl._compliance_active is True, \
            "compliance should be active"
        assert ctrl.collection_mode == "wholebody_guided", \
            f"collection_mode={ctrl.collection_mode!r}"
        assert ctrl.gain_schedule == "wholebody_v1_kpsupport45", \
            f"gain_schedule={ctrl.gain_schedule!r}"
    finally:
        _stop_patches(patches)
    print("  [PASS] HeuristicContactWholeBody flags correct")


# ──────────────────────────────────────────────
# Test 2 — _send_cmd KP per phase
# ──────────────────────────────────────────────

def _kps_after_send(ctrl, phase: str) -> tuple[list[float], list[float]]:
    """Return (kp[0:12], kd[0:12]) after invoking _send_cmd for phase."""
    ctrl._phase = phase
    # Replace the per-motor structs and the publisher / CRC with fakes
    # so _send_cmd writes into observable attributes without touching DDS.
    ctrl._low_cmd = _FakeLowCmd(n_motors=20)
    ctrl._crc = mock.MagicMock()
    ctrl._crc.Crc.return_value = 0
    ctrl._pub = mock.MagicMock()
    ctrl._pub.Write.return_value = None

    target_q = [0.0] * 12
    ctrl._send_cmd(target_q)
    kp = [ctrl._low_cmd.motor_cmd[i].kp for i in range(12)]
    kd = [ctrl._low_cmd.motor_cmd[i].kd for i in range(12)]
    return kp, kd


def test_wholebody_send_cmd_gains() -> None:
    from src.planner.heuristic_contact import (
        KP_FR, KD_FR,
        KP_FR_COMPLIANT, KD_FR_COMPLIANT,
        KP_STABLE, KD_STABLE,
        KP_SOFT, KD_SOFT,
    )
    from src.planner.heuristic_contact_wholebody import (
        KP_SUPPORT_SOFT, KD_SUPPORT_SOFT,
    )

    ctrl, patches = _patch_dds_and_construct()
    try:
        # Cases keyed by phase: (FR_kp_expected, support_kp_expected,
        #                        FR_kd_expected, support_kd_expected)
        cases = {
            "lift":         (KP_FR,           KP_STABLE,
                             KD_FR,           KD_STABLE),
            "extend":       (KP_FR_COMPLIANT, KP_SUPPORT_SOFT,
                             KD_FR_COMPLIANT, KD_SUPPORT_SOFT),
            "hold":         (KP_FR_COMPLIANT, KP_SUPPORT_SOFT,
                             KD_FR_COMPLIANT, KD_SUPPORT_SOFT),
            "retract_curl": (KP_FR,           KP_STABLE,
                             KD_FR,           KD_STABLE),
            # lower_to_sit: ALL motors at KP_SOFT.
            "lower_to_sit": (KP_SOFT,         KP_SOFT,
                             KD_SOFT,         KD_SOFT),
        }

        for phase, (fr_kp, sup_kp, fr_kd, sup_kd) in cases.items():
            kp, kd = _kps_after_send(ctrl, phase)
            for i in (0, 1, 2):
                assert kp[i] == fr_kp, \
                    f"phase={phase} FR joint {i}: kp={kp[i]} expected {fr_kp}"
                assert kd[i] == fr_kd, \
                    f"phase={phase} FR joint {i}: kd={kd[i]} expected {fr_kd}"
            for i in range(3, 12):
                assert kp[i] == sup_kp, \
                    f"phase={phase} support joint {i}: kp={kp[i]} expected {sup_kp}"
                assert kd[i] == sup_kd, \
                    f"phase={phase} support joint {i}: kd={kd[i]} expected {sup_kd}"
    finally:
        _stop_patches(patches)
    print("  [PASS] _send_cmd KP/KD correct across all phases")


# ──────────────────────────────────────────────
# Test 3 — v3 HDF5 round-trip
# ──────────────────────────────────────────────

class _FakeGroundingThread:
    def get_log(self):
        return []


class _FakeWholeBodyHeuristic:
    """Just enough to satisfy StageDRecorder.end_episode and save()."""
    collection_mode = "wholebody_guided"
    gain_schedule   = "wholebody_v1_kpsupport45"


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
        recorder.phase_label.append(1)   # extend
        recorder.phase_progress.append(np.float32(t / T))
        recorder.nominal_waypoint_fr.append(np.zeros(3, dtype=np.float32))
        recorder.jacobian_pid_delta.append(np.zeros(3, dtype=np.float32))
        recorder.joint_tau_est.append(np.full(12, 2.0, dtype=np.float32))
        recorder.imu_gyro.append(np.zeros(3, dtype=np.float32))
        recorder.imu_accel.append(np.array([0.0, 0.0, 9.81], dtype=np.float32))
        # v3: prescribed extend-phase gains — FR at 15, support at 45.
        kp_row = np.array(
            [15.0, 15.0, 15.0] + [45.0] * 9, dtype=np.float32)
        recorder.joint_kp_used.append(kp_row)


def test_v3_hdf5_write() -> None:
    from src.data.recorder import StageDRecorder

    rec = StageDRecorder(
        heuristic=_FakeWholeBodyHeuristic(),
        grounding_thread=_FakeGroundingThread(),
    )
    _populate_recorder(rec, T=20)
    rec._recording = False
    rec._metadata = {
        "episode_id":                  "wholebody_smoke",
        "target_pos_base_at_standoff": np.zeros(3, dtype=np.float32),
        "target_pos_world_initial":    np.zeros(3, dtype=np.float32),
        "perturbation_commanded":      np.zeros(2, dtype=np.float32),
        "k_p":                         0.0,
        "button_height_m":             0.25,
        "color_expected":              "red",
        "notes":                       "wholebody smoke",
    }
    rec.end_episode(_FakeResult())
    assert rec._metadata["collection_mode"] == "wholebody_guided"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "episode_wholebody_smoke.h5"
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
                "collection_mode":        "wholebody_guided",
            },
            use_v2_schema=True,
        )
        assert out.exists(), f"file not written: {out}"

        with h5py.File(out, "r") as f:
            # v2 datasets still present.
            for name in ("joint_tau_est", "imu_gyro", "imu_accel",
                         "achieved_delta_fr"):
                assert name in f["per_step"], f"{name} missing"
            # v3 dataset.
            assert "joint_kp_used" in f["per_step"], "joint_kp_used missing"
            kp = f["per_step/joint_kp_used"][()]
            assert kp.shape == (20, 12), f"kp shape={kp.shape}"
            assert kp.dtype == np.float32
            # Spot-check values.
            assert np.allclose(kp[:, 0:3], 15.0)
            assert np.allclose(kp[:, 3:12], 45.0)
            # v3 root attr.
            assert "gain_schedule" in f.attrs, "gain_schedule attr missing"
            gs = f.attrs["gain_schedule"]
            if isinstance(gs, (bytes, np.bytes_)):
                gs = gs.decode()
            assert str(gs) == "wholebody_v1_kpsupport45", f"gain_schedule={gs!r}"
            cm = f.attrs["collection_mode"]
            if isinstance(cm, (bytes, np.bytes_)):
                cm = cm.decode()
            assert str(cm) == "wholebody_guided", f"collection_mode={cm!r}"

    print("  [PASS] v3 HDF5 write — joint_kp_used + gain_schedule present")


# ──────────────────────────────────────────────
# Bonus — v2 path stays clean (no joint_kp_used, no gain_schedule)
# ──────────────────────────────────────────────

class _FakeV2Heuristic:
    collection_mode = "hand_guided"
    # No gain_schedule attribute on purpose.


def test_v2_hdf5_unchanged() -> None:
    """Sanity: a heuristic without gain_schedule must produce a v2-shaped file."""
    from src.data.recorder import StageDRecorder

    rec = StageDRecorder(
        heuristic=_FakeV2Heuristic(),
        grounding_thread=_FakeGroundingThread(),
    )
    _populate_recorder(rec, T=10)
    rec._recording = False
    rec._metadata = {
        "episode_id":                  "v2_smoke",
        "target_pos_base_at_standoff": np.zeros(3, dtype=np.float32),
        "target_pos_world_initial":    np.zeros(3, dtype=np.float32),
        "perturbation_commanded":      np.zeros(2, dtype=np.float32),
        "k_p":                         0.0,
        "button_height_m":             0.25,
        "color_expected":              "red",
        "notes":                       "v2 smoke",
    }
    rec.end_episode(_FakeResult())

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "episode_v2_smoke.h5"
        rec.save(
            str(out),
            audio=None,
            audio_sample_rate=16000,
            metadata_override={"collection_mode": "hand_guided"},
            use_v2_schema=True,
        )
        with h5py.File(out, "r") as f:
            assert "joint_kp_used" not in f["per_step"], \
                "v2 file must not contain joint_kp_used"
            assert "gain_schedule" not in f.attrs, \
                "v2 file must not contain gain_schedule attr"
            assert "achieved_delta_fr" in f["per_step"]
    print("  [PASS] v2 path unchanged when heuristic has no gain_schedule")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    failures = 0
    for name, fn in [
        ("HeuristicContactWholeBody flags",  test_wholebody_flags),
        ("_send_cmd KP per phase",           test_wholebody_send_cmd_gains),
        ("v3 HDF5 smoke write",              test_v3_hdf5_write),
        ("v2 HDF5 unchanged (no leakage)",   test_v2_hdf5_unchanged),
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
