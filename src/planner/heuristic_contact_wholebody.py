"""
src/planner/heuristic_contact_wholebody.py

Whole-body compliance variant of HeuristicContactGuided for v3 collection.

Two operators work together: Person 1 keeps a hand on the FR paw exactly as
in the FR-only protocol, while Person 2 demonstrates body posture
adjustments (rear hip lower, sagittal weight shift, lateral CoM shift) by
applying gentle pressure on the rear haunches or pelvis. To make that
physically possible, the support legs (FL, RL, RR) drop to softer PD gains
during the extend + hold phases — the same window during which the FR leg
softens for guidance.

Gain schedules
──────────────

  FR leg (inherited from HeuristicContactGuided, unchanged):
      lift, retract_*       : KP_FR=40,           KD_FR=3
      extend, hold          : KP_FR_COMPLIANT=15, KD_FR_COMPLIANT=1.5
      lower_to_sit          : KP_SOFT=25,         KD_SOFT=3
      everything else       : KP_STABLE=100,      KD_STABLE=6

  Support legs (FL, RL, RR) — NEW in this subclass:
      extend, hold          : KP_SUPPORT_SOFT=45, KD_SUPPORT_SOFT=4.5
      lower_to_sit          : KP_SOFT=25,         KD_SOFT=3 (matches base)
      everything else       : KP_STABLE=100,      KD_STABLE=6

The lift phase keeps support legs at full stiffness — the body must be
rigidly supported during weight shift and FR lift. Only once the FR paw is
committed forward (extend onward) do we soften support legs. Retract
phases also keep support legs rigid: the leg withdrawal can perturb the
body and we want stable tracking back to the weight-shift pose.

Implementation note
───────────────────

We override ``_send_cmd`` rather than ``_control_loop``. The parent's
control logic (target generation, IMU correction, gates) is correct and
unchanged. Only the per-motor PD gains during a specific phase window
differ. ``_compliance_active`` is inherited as True from the parent;
this subclass simply broadens which legs the soft-gain branch covers.

Spec: Stage D v3 whole-body demonstration, April 2026.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from src.planner.heuristic_contact import (
    KP_FR, KD_FR,
    KP_FR_COMPLIANT, KD_FR_COMPLIANT,
    KP_STABLE, KD_STABLE,
    KP_SOFT,   KD_SOFT,
    FR_HIP, FR_THIGH, FR_CALF,
)
from src.planner.heuristic_contact_guided import HeuristicContactGuided

logger = logging.getLogger(__name__)


# Support-leg compliance gains — softer than KP_STABLE=100, firmer than
# KP_FR_COMPLIANT=15. Hardware-validated value pending; see
# scripts/collect_wholebody.py --cardboard-test.
KP_SUPPORT_SOFT = 55.0
KD_SUPPORT_SOFT = 5.5


# Static gravity-compensation FF torques for support tripod, in N·m.
# Indexed by leg base index in the per-leg ordering (FR=0, FL=3, RR=6, RL=9).
# Per-leg tuple is (hip, thigh, calf) feedforward torque.
#
# Source: Sport Mode rt/lowcmd capture during handshake hold (5 snapshots,
# averaged across 4 tightly-clustered captures with capture 4 outlier excluded).
# Joint positions at capture matched WEIGHT_SHIFT_POS exactly. Values are
# pose-sensitive — recapture if WEIGHT_SHIFT_POS changes.
#
# FR is excluded by design: Sport Mode commands tau=0 on FR throughout handshake.
#
# SAFETY: First robot test with --gravity-ff should be done with the button
# absent / out of reach. The FF torques are static at one pose; if the
# controller transitions through postures other than WEIGHT_SHIFT_POS during
# extend+hold, the FF values are no longer correct and may push the support
# legs in unexpected directions. Verify the support tripod holds steady at
# the new posture for ~5 seconds before any contact attempt.
SUPPORT_GRAVITY_FF = {
    3: (-0.20, +3.08, +9.70),  # FL  (hip, thigh, calf)
    6: (-0.92, +2.89, +6.89),  # RR
    9: (+2.60, +3.49, +8.51),  # RL
}


class HeuristicContactWholeBody(HeuristicContactGuided):
    """Whole-body compliance: FR + support legs soften during extend+hold."""

    # Class-level tags read by StageDRecorder. ``collection_mode`` flips
    # the dataset loader's format detection to v3. ``gain_schedule`` is
    # written as a root HDF5 attribute and identifies the gain set used
    # at collection time; it is set per-instance below so deployment-time
    # KP overrides are reflected in the metadata.
    collection_mode: str = "wholebody_guided"

    # Defaults match the historical module-level constants
    # KP_SUPPORT_SOFT / KD_SUPPORT_SOFT for backwards compatibility.
    DEFAULT_KP_SUPPORT_SOFT: float = KP_SUPPORT_SOFT  # 55.0
    DEFAULT_KD_SUPPORT_SOFT: float = KD_SUPPORT_SOFT  # 5.5

    def __init__(
        self,
        *args,
        kp_support_soft: float = DEFAULT_KP_SUPPORT_SOFT,
        kd_support_soft: float = DEFAULT_KD_SUPPORT_SOFT,
        **kwargs,
    ):
        # gravity_ff_enabled / gravity_ff_body_mass / gravity_ff_phases are
        # accepted by the base class HeuristicContact; we just let them flow
        # through **kwargs so this subclass does not need to know about them.
        super().__init__(*args, **kwargs)
        # Parent already set self._compliance_active = True. The override
        # in _send_cmd reads that flag and the current phase to broaden
        # the soft-gain branch to cover the support legs, and now also
        # reads the per-instance gain values configured here.
        self._kp_support_soft = float(kp_support_soft)
        self._kd_support_soft = float(kd_support_soft)
        # Encode the actual KP into the metadata string so downstream
        # validators / analysis can identify off-default tuning runs.
        self.gain_schedule = (
            f"wholebody_v2_kpsupport{int(self._kp_support_soft)}"
        )
        self._wholebody_banner_emitted = False
        logger.info(
            "HeuristicContactWholeBody initialised — FR + support legs "
            "soften (kp_support_soft=%.1f kd_support_soft=%.1f) during extend+hold; "
            "gravity_ff_enabled=%s.",
            self._kp_support_soft, self._kd_support_soft,
            self._gravity_ff_enabled,
        )

    def execute(self, *args, **kwargs):
        if not self._wholebody_banner_emitted:
            print("[WHOLEBODY] Both FR and support legs soften during extend+hold.")
            print("[WHOLEBODY] Person 1: hand on FR paw. Person 2: hands on rear haunches.")
            self._wholebody_banner_emitted = True
        return super().execute(*args, **kwargs)

    def _send_cmd(self, target_q: list) -> None:
        # Mirrors the parent's _send_cmd structure but extends the
        # soft-gain branch to also cover support legs (i ∉ FR set) when
        # _compliance_active is on and the phase is extend or hold.
        for i in range(12):
            self._low_cmd.motor_cmd[i].q   = target_q[i]
            self._low_cmd.motor_cmd[i].dq  = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0

            is_fr = i in (FR_HIP, FR_THIGH, FR_CALF)

            if is_fr and self._phase in (
                "lift", "extend", "hold",
                "retract_curl", "retract_rotate", "retract_extend",
            ):
                if self._compliance_active and self._phase in ("extend", "hold"):
                    self._low_cmd.motor_cmd[i].kp = KP_FR_COMPLIANT
                    self._low_cmd.motor_cmd[i].kd = KD_FR_COMPLIANT
                else:
                    self._low_cmd.motor_cmd[i].kp = KP_FR
                    self._low_cmd.motor_cmd[i].kd = KD_FR
            elif self._phase == "lower_to_sit":
                self._low_cmd.motor_cmd[i].kp = KP_SOFT
                self._low_cmd.motor_cmd[i].kd = KD_SOFT
            elif (not is_fr
                  and self._compliance_active
                  and self._phase in ("extend", "hold")):
                # NEW: support legs soften alongside FR during extend+hold.
                # Per-instance values so deployment can sweep KP_SUPPORT_SOFT
                # without editing module constants.
                self._low_cmd.motor_cmd[i].kp = self._kp_support_soft
                self._low_cmd.motor_cmd[i].kd = self._kd_support_soft
            else:
                self._low_cmd.motor_cmd[i].kp = KP_STABLE
                self._low_cmd.motor_cmd[i].kd = KD_STABLE

        # Dynamic gravity-comp FF (analytical calf + empirical thigh) is
        # implemented in HeuristicContact._maybe_apply_gravity_ff. It reads
        # gravity_ff_enabled / gravity_ff_phases / gravity_ff_body_mass off
        # self, all set on the base class. Calling the inherited helper
        # here keeps the FF policy in one place and lets the base
        # controller apply it as well.
        self._maybe_apply_gravity_ff()

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)


# ──────────────────────────────────────────────
# CLI smoke — quick sanity check that the subclass constructs.
# Opens DDS; for a pure mock check, see scripts/test_wholebody_subclass.py.
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="HeuristicContactWholeBody — construction sanity check (opens DDS).",
    )
    parser.add_argument("--interface", type=str, default=None)
    args = parser.parse_args()

    if args.interface is None:
        from src.planner.heuristic_contact import DEFAULT_INTERFACE
        args.interface = DEFAULT_INTERFACE

    print("Constructing HeuristicContactWholeBody...")
    controller = HeuristicContactWholeBody(network_interface=args.interface)
    print(f"  collection_mode    = {controller.collection_mode}")
    print(f"  gain_schedule      = {controller.gain_schedule}")
    print(f"  _compliance_active = {controller._compliance_active}")
    print(f"  KP_SUPPORT_SOFT    = {KP_SUPPORT_SOFT}")
    print(f"  KD_SUPPORT_SOFT    = {KD_SUPPORT_SOFT}")
    assert controller.collection_mode == "wholebody_guided"
    assert controller.gain_schedule   == "wholebody_v1_kpsupport45"
    assert controller._compliance_active is True
    print("OK")
