"""
src/planner/heuristic_contact_corrective.py

Jacobian-PID augmented heuristic for data collection and as an evaluation
baseline. Applies visual-servoing correction to the FR leg during lift,
extend, and hold phases based on live foot-to-target error from the
grounding thread.

Used in two contexts:
  1. Data collection — generates training demonstrations for Stage D.
     The per-step correction delta is the training label.
  2. Evaluation baseline — reported as Baseline 4 (Jacobian-PID reactive)
     alongside the learned Stage D (Core Method).

Retract phases are NOT corrected — they use pure heuristic logic to avoid
complications during leg withdrawal.

Spec: training_data_spec.md v2.1, Section 7.2
"""

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.planner.heuristic_contact import (
    HeuristicContact,
    FR_HIP, FR_THIGH, FR_CALF,
    CTRL_DT,
)

logger = logging.getLogger(__name__)

# Joint limits for clipping corrected commands (rad)
FR_HIP_MIN,   FR_HIP_MAX   = -0.5, +0.5
FR_THIGH_MIN, FR_THIGH_MAX = -2.5, +1.5
FR_CALF_MIN,  FR_CALF_MAX  = -2.8, +0.5

# Maximum correction magnitude per joint per step (rad)
MAX_DELTA_PER_STEP = 0.1


class HeuristicContactCorrective(HeuristicContact):
    """
    Adds Jacobian-PID correction to lift/extend/hold phases.

    Args:
        k_p: proportional gain for error-to-delta conversion (1/m).
             Typical range 0.3-1.0. Start with 0.5 and tune.
        grounding_getter: callable returning latest target_pos_base as
             np.ndarray(3,) or None. Reads from the grounding thread.
        record_correction: if True, appends each (timestamp, state, error,
             delta) tuple to self.correction_log for later retrieval.
    """

    CORRECTABLE_PHASES = ("lift", "extend", "hold")

    def __init__(
        self,
        *args,
        k_p: float = 0.5,
        grounding_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        record_correction: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_p = k_p
        self.grounding_getter = grounding_getter
        self.record_correction = record_correction
        self.correction_log: list = []
        self._correction_lock = threading.Lock()
        self.last_delta_fr = np.zeros(3, dtype=np.float32)

    def _control_loop(self):
        """
        Override the base control loop. Runs the parent logic to compute the
        nominal heuristic command, then overlays the Jacobian-PID correction
        on the FR leg in correctable phases before sending.

        Approach: let parent compute and send the nominal command. If the
        current phase is correctable, compute and send a corrected command
        immediately after. At 500 Hz the second send supersedes the first
        at the motor level. Overhead < 0.1 ms, well within 2 ms budget.
        """
        super()._control_loop()

        if self._phase not in self.CORRECTABLE_PHASES:
            self.last_delta_fr = np.zeros(3, dtype=np.float32)
            return

        if self._low_state is None or self._last_target_q is None:
            return

        target = self.grounding_getter() if self.grounding_getter else None
        if target is None:
            self.last_delta_fr = np.zeros(3, dtype=np.float32)
            return

        with self._state_lock:
            actual = [self._low_state.motor_state[i].q for i in range(12)]

        fr_joints = np.array(actual[0:3], dtype=np.float32)
        foot_pos = self._estimate_fr_foot(actual)
        error = np.asarray(target, dtype=np.float32) - foot_pos

        J = self._fr_jacobian(fr_joints)
        try:
            delta = np.linalg.pinv(J) @ (self.k_p * error)
        except np.linalg.LinAlgError:
            logger.warning("Jacobian pseudoinverse failed; skipping correction")
            return

        delta = np.clip(delta, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)

        corrected = list(self._last_target_q)
        corrected[FR_HIP]   = float(np.clip(
            corrected[FR_HIP]   + delta[0], FR_HIP_MIN,   FR_HIP_MAX))
        corrected[FR_THIGH] = float(np.clip(
            corrected[FR_THIGH] + delta[1], FR_THIGH_MIN, FR_THIGH_MAX))
        corrected[FR_CALF]  = float(np.clip(
            corrected[FR_CALF]  + delta[2], FR_CALF_MIN,  FR_CALF_MAX))

        self._last_target_q = corrected
        self._send_cmd(corrected)

        self.last_delta_fr = delta.astype(np.float32)

        if self.record_correction:
            with self._correction_lock:
                self.correction_log.append({
                    "t": time.monotonic(),
                    "phase": self._phase,
                    "phase_step": self._phase_step,
                    "fr_joints": fr_joints.tolist(),
                    "foot_pos": foot_pos.tolist(),
                    "target": np.asarray(target).tolist(),
                    "error": error.tolist(),
                    "delta": delta.tolist(),
                })

    @staticmethod
    def _fr_jacobian(fr_joints: np.ndarray) -> np.ndarray:
        """
        Analytic 3x3 Jacobian of _estimate_fr_foot w.r.t. FR joints.

        From heuristic_contact.py _estimate_fr_foot:
            x = HIP_X + L_THIGH * sin(thigh) + L_CALF * sin(thigh + calf)
            y = HIP_Y + L_THIGH * sin(hip)
            z = -L_THIGH * cos(thigh) - L_CALF * cos(thigh + calf)

        Returns:
            J: 3x3 array, rows = [dx, dy, dz], cols = [dhip, dthigh, dcalf]
        """
        L_THIGH = 0.213
        L_CALF  = 0.213
        hip, thigh, calf = fr_joints

        dx_dhip   = 0.0
        dx_dthigh = L_THIGH * np.cos(thigh) + L_CALF * np.cos(thigh + calf)
        dx_dcalf  = L_CALF  * np.cos(thigh + calf)

        dy_dhip   = L_THIGH * np.cos(hip)
        dy_dthigh = 0.0
        dy_dcalf  = 0.0

        dz_dhip   = 0.0
        dz_dthigh = L_THIGH * np.sin(thigh) + L_CALF * np.sin(thigh + calf)
        dz_dcalf  = L_CALF  * np.sin(thigh + calf)

        J = np.array([
            [dx_dhip, dx_dthigh, dx_dcalf],
            [dy_dhip, dy_dthigh, dy_dcalf],
            [dz_dhip, dz_dthigh, dz_dcalf],
        ], dtype=np.float32)
        return J


# ──────────────────────────────────────────────
# Finite-difference Jacobian verification (no robot needed)
# ──────────────────────────────────────────────
# Run with: python -m src.planner.heuristic_contact_corrective
# Checks that the analytic Jacobian agrees with finite-difference of
# _estimate_fr_foot at several representative joint configurations.
# If any config fails (err >= 1e-3), the Jacobian is wrong and the
# Jacobian-PID expert will push the leg in the wrong direction.

def _verify_jacobian() -> bool:
    """Returns True if all test configs pass, False otherwise."""
    test_joints = [
        # (label, fr_joint_vector)
        ("STAND_POS",          np.array([-0.021, +0.667, -1.359], dtype=np.float32)),
        ("wall_lift_approx",   np.array([ 0.019, -0.333, -1.359], dtype=np.float32)),
        ("wall_extend_approx", np.array([ 0.019, -1.333, -1.459], dtype=np.float32)),
        ("ground_lift_approx", np.array([ 0.019, -0.133, -1.259], dtype=np.float32)),
        ("ground_press_apx",   np.array([ 0.019, -0.333, -0.559], dtype=np.float32)),
        ("neutral",            np.array([ 0.0,   -0.8,   -0.5  ], dtype=np.float32)),
        ("edge_case",          np.array([ 0.1,   -1.0,   +0.8  ], dtype=np.float32)),
    ]

    eps = 1e-4
    all_pass = True

    print("\nFinite-difference Jacobian verification")
    print("=" * 64)
    print(f"{'label':<22} {'max|J_a - J_n|':<18} {'status'}")
    print("-" * 64)

    for label, fr in test_joints:
        J_analytic = HeuristicContactCorrective._fr_jacobian(fr)

        J_num = np.zeros((3, 3), dtype=np.float32)
        full = [0.0] * 12
        full[FR_HIP]   = float(fr[0])
        full[FR_THIGH] = float(fr[1])
        full[FR_CALF]  = float(fr[2])
        base = HeuristicContact._estimate_fr_foot(full)

        for j in range(3):
            full_p = list(full)
            full_p[j] += eps
            pert = HeuristicContact._estimate_fr_foot(full_p)
            J_num[:, j] = (pert - base) / eps

        err = float(np.max(np.abs(J_analytic - J_num)))
        ok  = err < 1e-3
        all_pass = all_pass and ok
        print(f"{label:<22} {err:.2e}           {'OK' if ok else 'FAIL'}")

    print("-" * 64)
    print(f"{'OVERALL:':<22} {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ok = _verify_jacobian()
    sys.exit(0 if ok else 1)