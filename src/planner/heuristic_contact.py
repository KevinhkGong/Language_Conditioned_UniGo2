"""
src/planner/heuristic_contact.py

Heuristic contact phase controller for wall-mounted button pressing.

Each phase uses a gate check — the next phase only starts when actual
joint positions confirm the current phase is complete. This prevents
any phase from starting before the previous one is physically done.

Execution sequence:
    1. StandDown() → ReleaseMode()
    2. sit_to_stand   — gate: all joints near STAND_POS
    3. weight_shift   — gate: FL/RL/RR hips at WEIGHT_SHIFT_POS
    4. lift           — gate: FR thigh near lift target
    5. extend         — gate: FR thigh near extend target
    6. hold           — timeout or contact detected
    7. retract        — gate: FR thigh near WEIGHT_SHIFT_POS
    8. weight_unshift — gate: all joints near STAND_POS
    9. settle         — gate: all joints near STAND_POS (tighter)
    10. lower_to_sit  — gate: all joints near SIT_POS
    11. done          — zero gains → SelectMode → RecoveryStand

Joint ordering (unitree_legged_const.py Go2):
    0=FR_hip   1=FR_thigh  2=FR_calf
    3=FL_hip   4=FL_thigh  5=FL_calf
    6=RR_hip   7=RR_thigh  8=RR_calf
    9=RL_hip  10=RL_thigh 11=RL_calf
"""

import time
import logging
import threading
import concurrent.futures
from dataclasses import dataclass

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_INTERFACE = "enx98fc84e68f1a"

CTRL_DT = 0.002   # 500 Hz

KP_STABLE = 100.0
KD_STABLE = 6.0
KP_FR     = 40.0
KD_FR     = 3.0
KP_SOFT   = 25.0
KD_SOFT   = 3.0

# Phase durations (steps at 500Hz) — used as minimum, gate check is the real trigger
STEPS_SIT_TO_STAND = 1000   # 2.0s min
STEPS_WEIGHT_SHIFT = 800    # 1.6s min
STEPS_LIFT         = 600    # 1.2s min
STEPS_EXTEND       = 600    # 1.2s min
STEPS_HOLD         = 500    # 1.0s hold after contact
STEPS_RETRACT      = 800    # 1.6s min
STEPS_WEIGHT_SHIFT_BACK = 800  # 1.6s min
STEPS_SETTLE       = 500    # 1.0s min — smooth actual → STAND_POS before sit
STEPS_LOWER_TO_SIT = 1500   # 3.0s min

# Gate timeout — max extra steps to wait for gate before forcing advance
GATE_TIMEOUT = 500   # 1.0s extra before forcing advance

CONTACT_PROXIMITY_M = 0.06
CONTACT_MAX_STEPS   = 1500

FR_HIP   = 0
FR_THIGH = 1
FR_CALF  = 2

# Joint threshold for gate checks
GATE_THRESHOLD = 0.15   # radians

# ── Poses — measured from real robot ─────────────────────────────────────────

STAND_POS = [
    -0.021, +0.667, -1.359,   # FR
    +0.025, +0.666, -1.368,   # FL
    -0.078, +0.661, -1.348,   # RR
    +0.072, +0.660, -1.347,   # RL
]

SIT_POS = [
    -0.086, +1.230, -2.769,   # FR  (measured from robot)
    +0.108, +1.230, -2.776,   # FL
    -0.423, +1.243, -2.763,   # RR
    +0.418, +1.247, -2.776,   # RL
]

WEIGHT_SHIFT_POS = list(STAND_POS)
WEIGHT_SHIFT_POS[3]  = STAND_POS[3]  - 0.30   # FL_hip inward
WEIGHT_SHIFT_POS[9]  = STAND_POS[9]  - 0.25   # RL_hip inward
WEIGHT_SHIFT_POS[6]  = STAND_POS[6]  + 0.15   # RR_hip outward
WEIGHT_SHIFT_POS[4]  = STAND_POS[4]  - 0.10   # FL_thigh lower
WEIGHT_SHIFT_POS[10] = STAND_POS[10] - 0.10   # RL_thigh lower

FR_LIFT_OFFSET   = np.array([+0.04, -1.0,  0.0])
FR_EXTEND_OFFSET = np.array([+0.04, -2.0, -0.1])


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ContactResult:
    success:            bool
    contact_step:       int
    timeout:            bool
    target_offset_xyz:  np.ndarray
    joint_trajectory:   np.ndarray
    fr_foot_trajectory: np.ndarray
    duration_s:         float


# ──────────────────────────────────────────────
# Main controller
# ──────────────────────────────────────────────

class HeuristicContact:

    def __init__(
        self,
        network_interface: str = DEFAULT_INTERFACE,
        already_initialized: bool = False,
    ):
        self._crc        = CRC()
        self._low_state  = None
        self._low_cmd    = unitree_go_msg_dds__LowCmd_()
        self._state_lock = threading.Lock()
        self._stop_flag  = threading.Event()

        self._phase        = "idle"
        self._phase_step   = 0
        self._start_pos    = list(SIT_POS)
        self._done         = False
        self._contact_step = -1
        self._joint_traj   = []
        self._fr_foot_traj = []

        # Dynamic pose storage
        self._lift_end_pos    = list(STAND_POS)
        self._hold_pos        = list(STAND_POS)
        self._retract_start   = list(STAND_POS)
        self._unshift_start   = list(STAND_POS)
        self._settle_start    = list(STAND_POS)
        self._lower_start     = list(STAND_POS)

        if not already_initialized:
            ChannelFactoryInitialize(0, network_interface)

        self._init_low_cmd()

        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._state_cb, 10)

        self._sport = SportClient()
        self._sport.SetTimeout(5.0)
        self._sport.Init()

        self._msc = MotionSwitcherClient()
        self._msc.SetTimeout(5.0)
        self._msc.Init()

        logger.info("HeuristicContact initialised.")

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def execute(
        self,
        target_offset_xyz: np.ndarray,
        contact_proximity_m: float = CONTACT_PROXIMITY_M,
    ) -> ContactResult:

        self._joint_traj        = []
        self._fr_foot_traj      = []
        self._contact_step      = -1
        self._done              = False
        self._target_offset     = target_offset_xyz
        self._contact_proximity = contact_proximity_m

        t_start = time.time()

        self._switch_to_lowlevel()
        self._wait_for_state()

        with self._state_lock:
            self._start_pos = [self._low_state.motor_state[i].q
                               for i in range(12)]

        logger.info(f"Start pose FR: hip={self._start_pos[0]:+.3f} "
                    f"thigh={self._start_pos[1]:+.3f} "
                    f"calf={self._start_pos[2]:+.3f}")

        self._phase      = "sit_to_stand"
        self._phase_step = 0
        self._stop_flag.clear()

        self._write_thread = threading.Thread(
            target=self._control_loop_thread,
            name="heuristic_contact",
            daemon=True
        )
        self._write_thread.start()

        # Generous timeout — gates ensure correctness, timeout is safety net
        timeout = 120.0
        t_wait  = 0.0
        while not self._done and t_wait < timeout:
            time.sleep(0.1)
            t_wait += 0.1

        time.sleep(0.2)
        duration = time.time() - t_start

        logger.info("Stopping control loop...")
        self._stop_flag.set()
        self._write_thread.join(timeout=2.0)

        logger.info("Zeroing gains...")
        for i in range(20):
            self._low_cmd.motor_cmd[i].kp  = 0.0
            self._low_cmd.motor_cmd[i].kd  = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)
        time.sleep(1.5)

        self._switch_to_sportmode()

        return ContactResult(
            success=self._contact_step >= 0,
            contact_step=self._contact_step,
            timeout=not self._done,
            target_offset_xyz=target_offset_xyz,
            joint_trajectory=np.array(self._joint_traj)     if self._joint_traj    else np.zeros((0, 12)),
            fr_foot_trajectory=np.array(self._fr_foot_traj) if self._fr_foot_traj else np.zeros((0, 3)),
            duration_s=duration,
        )

    # ──────────────────────────────────────────────
    # Control loop
    # ──────────────────────────────────────────────

    def _control_loop_thread(self):
        while not self._stop_flag.is_set() and not self._done:
            t0 = time.monotonic()
            self._control_loop()
            elapsed   = time.monotonic() - t0
            remaining = CTRL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _control_loop(self):
        if self._low_state is None:
            return

        self._phase_step += 1

        # Read current actual joint positions and IMU
        with self._state_lock:
            actual = [self._low_state.motor_state[i].q for i in range(12)]
            roll   = self._low_state.imu_state.rpy[0]
            pitch  = self._low_state.imu_state.rpy[1]

        target_q = list(STAND_POS)

        # ── Gate helper ───────────────────────────────────────────────
        def gate_passed(target_pos, indices, threshold=GATE_THRESHOLD):
            """True if all specified joints are close to target."""
            return all(abs(actual[i] - target_pos[i]) < threshold
                       for i in indices)

        def should_advance(min_steps, gate_ok):
            """Advance if gate passed after min steps, or force after timeout."""
            if self._phase_step >= min_steps and gate_ok:
                return True
            if self._phase_step >= min_steps + GATE_TIMEOUT:
                logger.warning(f"Phase '{self._phase}' gate timeout — forcing advance")
                return True
            return False

        # ── sit → stand ───────────────────────────────────────────────
        if self._phase == "sit_to_stand":
            alpha = min(self._phase_step / STEPS_SIT_TO_STAND, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*self._start_pos[i] + alpha*STAND_POS[i]

            gate_ok = gate_passed(STAND_POS, range(12), threshold=0.2)
            if should_advance(STEPS_SIT_TO_STAND, gate_ok):
                logger.info("✓ sit_to_stand complete")
                self._phase      = "weight_shift"
                self._phase_step = 0

        # ── weight shift ──────────────────────────────────────────────
        elif self._phase == "weight_shift":
            alpha = min(self._phase_step / STEPS_WEIGHT_SHIFT, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*STAND_POS[i] + alpha*WEIGHT_SHIFT_POS[i]

            # Gate: FL_hip, RL_hip, RR_hip have shifted
            gate_ok = gate_passed(WEIGHT_SHIFT_POS, [3, 9, 6], threshold=0.15)
            if should_advance(STEPS_WEIGHT_SHIFT, gate_ok):
                logger.info("✓ weight_shift complete")
                self._phase      = "lift"
                self._phase_step = 0

        # ── lift FR ───────────────────────────────────────────────────
        elif self._phase == "lift":
            alpha    = min(self._phase_step / STEPS_LIFT, 1.0)
            target_q = list(WEIGHT_SHIFT_POS)
            fr_s     = np.array([WEIGHT_SHIFT_POS[FR_HIP],
                                  WEIGHT_SHIFT_POS[FR_THIGH],
                                  WEIGHT_SHIFT_POS[FR_CALF]])
            fr_l     = np.array([STAND_POS[FR_HIP]   + FR_LIFT_OFFSET[0],
                                  STAND_POS[FR_THIGH] + FR_LIFT_OFFSET[1],
                                  STAND_POS[FR_CALF]  + FR_LIFT_OFFSET[2]])
            target_q[FR_HIP]   = (1-alpha)*fr_s[0] + alpha*fr_l[0]
            target_q[FR_THIGH] = (1-alpha)*fr_s[1] + alpha*fr_l[1]
            target_q[FR_CALF]  = (1-alpha)*fr_s[2] + alpha*fr_l[2]

            # Gate: FR thigh is at lift position
            gate_ok = abs(actual[FR_THIGH] - fr_l[1]) < GATE_THRESHOLD
            if should_advance(STEPS_LIFT, gate_ok):
                logger.info(f"✓ lift complete FR_thigh={actual[FR_THIGH]:+.3f}")
                self._phase        = "extend"
                self._phase_step   = 0
                self._lift_end_pos = list(target_q)

        # ── extend FR ─────────────────────────────────────────────────
        elif self._phase == "extend":
            alpha    = min(self._phase_step / STEPS_EXTEND, 1.0)
            target_q = list(self._lift_end_pos)
            fr_e     = np.array([STAND_POS[FR_HIP]   + FR_EXTEND_OFFSET[0],
                                  STAND_POS[FR_THIGH] + FR_EXTEND_OFFSET[1],
                                  STAND_POS[FR_CALF]  + FR_EXTEND_OFFSET[2]])
            fr_l     = np.array([self._lift_end_pos[FR_HIP],
                                  self._lift_end_pos[FR_THIGH],
                                  self._lift_end_pos[FR_CALF]])
            target_q[FR_HIP]   = (1-alpha)*fr_l[0] + alpha*fr_e[0]
            target_q[FR_THIGH] = (1-alpha)*fr_l[1] + alpha*fr_e[1]
            target_q[FR_CALF]  = (1-alpha)*fr_l[2] + alpha*fr_e[2]

            # Gate: FR thigh is at extend position
            gate_ok = abs(actual[FR_THIGH] - fr_e[1]) < GATE_THRESHOLD
            if should_advance(STEPS_EXTEND, gate_ok):
                logger.info(f"✓ extend complete FR_thigh={actual[FR_THIGH]:+.3f}")
                self._phase      = "hold"
                self._phase_step = 0
                self._hold_pos   = list(target_q)

        # ── hold + detect contact ─────────────────────────────────────
        elif self._phase == "hold":
            target_q = list(self._hold_pos)
            self._joint_traj.append(list(target_q))
            fr_foot = self._estimate_fr_foot(target_q)
            self._fr_foot_traj.append(fr_foot.tolist())

            dist = np.linalg.norm(fr_foot - self._target_offset)
            if dist < self._contact_proximity and self._contact_step < 0:
                self._contact_step = self._phase_step
                logger.info(f"Contact detected at step {self._contact_step} "
                            f"dist={dist:.3f}m")

            contact_done = (self._contact_step >= 0 and
                            self._phase_step >= self._contact_step + STEPS_HOLD)
            if contact_done or self._phase_step >= CONTACT_MAX_STEPS:
                logger.info(f"✓ hold complete — retracting")
                self._phase         = "retract"
                self._phase_step    = 0
                self._retract_start = list(target_q)

        # ── retract FR ────────────────────────────────────────────────
        elif self._phase == "retract":
            alpha = min(self._phase_step / STEPS_RETRACT, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*self._retract_start[i] + alpha*WEIGHT_SHIFT_POS[i]

            # Gate: FR thigh is back near WEIGHT_SHIFT_POS (retracted)
            gate_ok = abs(actual[FR_THIGH] - WEIGHT_SHIFT_POS[FR_THIGH]) < GATE_THRESHOLD
            if should_advance(STEPS_RETRACT, gate_ok):
                logger.info(f"✓ retract complete FR_thigh={actual[FR_THIGH]:+.3f}")
                self._phase         = "weight_unshift"
                self._phase_step    = 0
                self._unshift_start = list(target_q)

        # ── weight unshift ────────────────────────────────────────────
        elif self._phase == "weight_unshift":
            alpha = min(self._phase_step / STEPS_WEIGHT_SHIFT_BACK, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*self._unshift_start[i] + alpha*STAND_POS[i]

            # Gate: FL_hip, RL_hip, RR_hip back to STAND_POS
            gate_ok = gate_passed(STAND_POS, range(12), threshold=0.15)
            if should_advance(STEPS_WEIGHT_SHIFT_BACK, gate_ok):
                logger.info("✓ weight_unshift complete")
                self._phase        = "settle"
                self._phase_step   = 0
                self._settle_start = list(actual)

        # ── settle — smooth actual → STAND_POS ─────────────────────
        elif self._phase == "settle":
            alpha = min(self._phase_step / STEPS_SETTLE, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*self._settle_start[i] + alpha*STAND_POS[i]

            # Gate: all joints near STAND_POS (tighter threshold)
            gate_ok = gate_passed(STAND_POS, range(12), threshold=0.10)
            if should_advance(STEPS_SETTLE, gate_ok):
                logger.info("✓ settle complete — starting lower_to_sit from STAND_POS")
                self._phase       = "lower_to_sit"
                self._phase_step  = 0
                self._lower_start = list(STAND_POS)

        # ── lower to sit ──────────────────────────────────────────────
        elif self._phase == "lower_to_sit":
            t     = min(self._phase_step / STEPS_LOWER_TO_SIT, 1.0)
            alpha = t * t * (3 - 2 * t)   # smoothstep
            for i in range(12):
                target_q[i] = (1-alpha)*self._lower_start[i] + alpha*SIT_POS[i]

            # Gate: all joints near SIT_POS
            gate_ok = gate_passed(SIT_POS, range(12), threshold=0.2)
            if should_advance(STEPS_LOWER_TO_SIT, gate_ok):
                logger.info("✓ lower_to_sit complete")
                self._phase = "done"

        # ── done ──────────────────────────────────────────────────────
        elif self._phase == "done":
            target_q   = list(SIT_POS)
            self._done = True
            self._send_cmd(target_q)
            return

        # ── IMU balance correction — hold and retract only ────────────
        # Disabled during weight_shift/lift/extend to not fight intentional lean
        if self._phase in ("hold", "retract"):
            ROLL_GAIN  = 0.6
            PITCH_GAIN = 0.15

            roll_corr  = roll  * ROLL_GAIN
            pitch_corr = pitch * PITCH_GAIN

            target_q[3] -= roll_corr
            target_q[9] -= roll_corr
            target_q[6] += roll_corr
            target_q[4]  -= pitch_corr
            target_q[7]  -= pitch_corr
            target_q[10] -= pitch_corr

            target_q[3]  = np.clip(target_q[3],  -0.8,  0.5)
            target_q[9]  = np.clip(target_q[9],  -0.8,  0.5)
            target_q[6]  = np.clip(target_q[6],  -0.3,  0.8)
            target_q[4]  = np.clip(target_q[4],   0.3,  1.5)
            target_q[7]  = np.clip(target_q[7],   0.3,  1.5)
            target_q[10] = np.clip(target_q[10],  0.3,  1.5)

        self._send_cmd(target_q)

    def _send_cmd(self, target_q: list) -> None:
        for i in range(12):
            self._low_cmd.motor_cmd[i].q   = target_q[i]
            self._low_cmd.motor_cmd[i].dq  = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0

            if i in (FR_HIP, FR_THIGH, FR_CALF) and self._phase in (
                "lift", "extend", "hold"
            ):
                self._low_cmd.motor_cmd[i].kp = KP_FR
                self._low_cmd.motor_cmd[i].kd = KD_FR
            elif self._phase == "lower_to_sit":
                self._low_cmd.motor_cmd[i].kp = KP_SOFT
                self._low_cmd.motor_cmd[i].kd = KD_SOFT
            else:
                self._low_cmd.motor_cmd[i].kp = KP_STABLE
                self._low_cmd.motor_cmd[i].kd = KD_STABLE

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)

    # ──────────────────────────────────────────────
    # Mode switching
    # ──────────────────────────────────────────────

    def _switch_to_lowlevel(self) -> None:
        logger.info("StandDown then ReleaseMode...")
        time.sleep(1.0)

        status, result = self._msc.CheckMode()
        if result is None:
            logger.warning("CheckMode returned None — retrying...")
            time.sleep(2.0)
            status, result = self._msc.CheckMode()

        if result is None:
            raise RuntimeError("MotionSwitcher not responding.")

        while result.get('name'):
            self._sport.StandDown()
            time.sleep(3.0)
            self._msc.ReleaseMode()
            time.sleep(0.5)
            status, result = self._msc.CheckMode()
            if result is None:
                break

        logger.info("Low-Level active.")

    def _switch_to_sportmode(self) -> None:
        logger.info("Restoring Sport Mode...")
        time.sleep(1.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._msc.SelectMode, "mcf")
            try:
                ret = future.result(timeout=3.0)
                logger.info(f"SelectMode ret={ret}")
            except concurrent.futures.TimeoutError:
                logger.warning("SelectMode timed out — use controller to recover.")
                return

        time.sleep(1.5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._sport.RecoveryStand)
            try:
                ret = future.result(timeout=3.0)
                logger.info(f"RecoveryStand ret={ret}")
            except concurrent.futures.TimeoutError:
                logger.warning("RecoveryStand timed out — use controller to recover.")
                return

        time.sleep(2.0)
        logger.info("Sport Mode restored.")

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _init_low_cmd(self) -> None:
        self._low_cmd.head[0]    = 0xFE
        self._low_cmd.head[1]    = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio       = 0
        for i in range(20):
            self._low_cmd.motor_cmd[i].mode = 0x01
            self._low_cmd.motor_cmd[i].q    = 0.0
            self._low_cmd.motor_cmd[i].kp   = 0.0
            self._low_cmd.motor_cmd[i].dq   = 0.0
            self._low_cmd.motor_cmd[i].kd   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0

    def _state_cb(self, msg: LowState_) -> None:
        with self._state_lock:
            self._low_state = msg

    def _wait_for_state(self, timeout: float = 5.0) -> None:
        t = 0.0
        while self._low_state is None and t < timeout:
            time.sleep(0.1)
            t += 0.1
        if self._low_state is None:
            raise RuntimeError("No LowState received.")

    @staticmethod
    def _estimate_fr_foot(q: list) -> np.ndarray:
        L_THIGH = 0.213
        L_CALF  = 0.213
        HIP_Y   = -0.1034
        HIP_X   =  0.1934
        hip   = q[FR_HIP]
        thigh = q[FR_THIGH]
        calf  = q[FR_CALF]
        x  = HIP_X + L_THIGH * np.sin(thigh)
        y  = HIP_Y + L_THIGH * np.sin(hip)
        z  = -L_THIGH * np.cos(thigh)
        x += L_CALF * np.sin(thigh + calf)
        z -= L_CALF * np.cos(thigh + calf)
        return np.array([x, y, z], dtype=np.float32)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", type=str, default=DEFAULT_INTERFACE)
    parser.add_argument("--target-x",  type=float, default=0.35)
    parser.add_argument("--target-y",  type=float, default=-0.10)
    parser.add_argument("--target-z",  type=float, default=0.40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Heuristic Contact — Gate-Based Phase Controller")
    print("="*60)
    print(f"Interface: {args.interface}")
    print(f"Target:    ({args.target_x}, {args.target_y}, {args.target_z})")
    print("\nPhases: sit_to_stand → weight_shift → lift → extend → "
          "hold → retract → weight_unshift → settle → lower_to_sit → done")
    print("Each phase waits for gate confirmation before advancing.")
    print("\nWARNING: Robot will sit down, stand in low-level, lift FR leg.")
    input("Press Enter to continue (Ctrl+C to abort)...\n")

    controller = HeuristicContact(args.interface)
    print("✓ Connected")

    if args.dry_run:
        print("Dry-run — skipping.")
    else:
        target = np.array([args.target_x, args.target_y, args.target_z])
        result = controller.execute(target_offset_xyz=target)
        print(f"\nResult:")
        print(f"  success:      {result.success}")
        print(f"  contact_step: {result.contact_step}")
        print(f"  timeout:      {result.timeout}")
        print(f"  duration:     {result.duration_s:.2f}s")
        print(f"  traj_shape:   {result.joint_trajectory.shape}")