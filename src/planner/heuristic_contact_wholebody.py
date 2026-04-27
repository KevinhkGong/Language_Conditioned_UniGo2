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


class HeuristicContactWholeBody(HeuristicContactGuided):
    """Whole-body compliance: FR + support legs soften during extend+hold."""

    # Class-level tags read by StageDRecorder. ``collection_mode`` flips
    # the dataset loader's format detection to v3; ``gain_schedule`` is
    # written as a root HDF5 attribute and identifies the gain set used
    # at collection time so post-hoc analysis can correlate behavior
    # with gains across collection sessions.
    collection_mode: str = "wholebody_guided"
    gain_schedule:   str = "wholebody_v2_kpsupport55"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parent already set self._compliance_active = True. The override
        # in _send_cmd reads that flag and the current phase to broaden
        # the soft-gain branch to cover the support legs.
        self._wholebody_banner_emitted = False
        logger.info(
            "HeuristicContactWholeBody initialised — FR + support legs "
            "soften (KP_SUPPORT_SOFT=%.1f) during extend+hold.",
            KP_SUPPORT_SOFT,
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
                self._low_cmd.motor_cmd[i].kp = KP_SUPPORT_SOFT
                self._low_cmd.motor_cmd[i].kd = KD_SUPPORT_SOFT
            else:
                self._low_cmd.motor_cmd[i].kp = KP_STABLE
                self._low_cmd.motor_cmd[i].kd = KD_STABLE

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
