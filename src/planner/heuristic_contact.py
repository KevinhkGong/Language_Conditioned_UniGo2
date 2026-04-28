"""
src/planner/heuristic_contact.py

Heuristic contact phase controller for button pressing (wall or ground).

Each phase uses a gate check — the next phase only starts when actual
joint positions confirm the current phase is complete. This prevents
any phase from starting before the previous one is physically done.

Execution sequence:
    1. StandDown() → ReleaseMode()
    2. sit_to_stand   — gate: all joints near STAND_POS
    3. weight_shift   — gate: FL/RL/RR hips at WEIGHT_SHIFT_POS
    4. lift           — gate: FR thigh near lift target
    5. extend/press   — gate: FR thigh near extend/press target
    6. hold           — timeout, FK-proximity contact, or AUDIO contact
    7. retract        — gate: FR thigh near WEIGHT_SHIFT_POS
    8. weight_unshift — gate: all joints near STAND_POS
    9. settle         — gate: all joints near STAND_POS (tighter)
    10. lower_to_sit  — gate: all joints near SIT_POS
    11. done          — zero gains → SelectMode → RecoveryStand

press_mode options:
    "wall"   — FR leg swings forward and out to press a wall-mounted button
    "ground" — FR leg lifts and presses downward onto a floor button (~1.5in tall)

Joint ordering (unitree_legged_const.py Go2):
    0=FR_hip   1=FR_thigh  2=FR_calf
    3=FL_hip   4=FL_thigh  5=FL_calf
    6=RR_hip   7=RR_thigh  8=RR_calf
    9=RL_hip  10=RL_thigh 11=RL_calf

Key fixes vs previous version:
    - _retract_start now captures actual joint positions, not commanded target_q.
      This eliminates the position discontinuity at the start of retract when
      the FR leg hasn't fully reached the extend/press target.
    - FR leg keeps KP_FR (40.0) during retract phase instead of jumping to
      KP_STABLE (100.0). The sudden gain increase on a mid-air extended leg
      was causing a torque spike that destabilized the support legs.
    - Control loop no longer silently skips _send_cmd when low_state is
      momentarily stale. Instead it resends the last command, preventing
      the single-cycle torque dropout ("one damp") caused by missed publishes
      at 500 Hz under Python GIL contention.
    - settle gate threshold loosened 0.10 → 0.15 to eliminate the shake-then-
      advance behaviour caused by forcing advance mid-oscillation.
    - Foot force contact detection added as a stub (disabled by default).
      FR foot force readings were found unreliable on this hardware during
      initial testing. Enable with use_foot_force=True once characterised.

Additions for Stage C / Stage D data collection (v2.0 spec, April 21 2026):
    - self._phase_transitions: dict capturing actual joint positions at each
      gate-passed phase transition. Returned in ContactResult.
    - self.current_waypoint_fr: np.ndarray (3,) updated every 500 Hz control
      step with the current FR command. Exposed for external observer
      threads (StageDRecorder) to log as the "nominal" waypoint.
    - ContactResult now carries the phase_transitions dict.
    These are ADDITIVE only — no change to control logic, gates, timing, or gains.

Additions for audio-based contact verification (v2.1 spec, April 21 2026):
    - Optional audio_detector parameter on execute(). When provided, audio
      energy spikes during the hold phase count as contact (same role as
      FK proximity). The authoritative "button was physically pressed" signal.
    - Whichever of audio / FK-proximity / foot-force fires first sets
      self._contact_step and the detection method is logged in
      self._contact_method ("audio", "fk", "foot_force") for downstream
      metadata. Default behavior (audio_detector=None) is unchanged.
"""

import math
import time
import logging
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Set, Tuple

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

# Compliance gains for hand-guided demonstration (Stage D v2).
# When a subclass flips self._compliance_active = True, the FR leg drops to
# these gains during the extend and hold phases ONLY so the operator can
# physically guide the paw onto the button. All other legs, and the FR leg
# in every other phase (including lift and the retract sequence), keep their
# normal gains — softening the mid-air extended leg during retract would
# trigger the same kind of torque discontinuity that the v2.0 retract-curl
# fix was introduced to eliminate.
KP_FR_COMPLIANT = 15.0
KD_FR_COMPLIANT = 1.5

# Phase durations (steps at 500 Hz) — used as minimum; gate check is real trigger
STEPS_SIT_TO_STAND      = 1000   # 2.0s min
STEPS_WEIGHT_SHIFT      = 800    # 1.6s min
STEPS_LIFT              = 600    # 1.2s min
STEPS_EXTEND            = 600    # 1.2s min
STEPS_HOLD              = 500    # 1.0s hold after contact
STEPS_RETRACT_CURL      = 400    # 0.8s — calf tucks up clear of button
STEPS_RETRACT_ROTATE    = 600    # 1.2s — thigh swings back to weight shift
STEPS_RETRACT_EXTEND    = 400    # 0.8s — calf extends back to weight shift pos
STEPS_RETRACT           = 800    # 1.6s min
STEPS_WEIGHT_SHIFT_BACK = 800    # 1.6s min
STEPS_SETTLE            = 500    # 1.0s min — smooth actual → STAND_POS before sit
STEPS_LOWER_TO_SIT      = 1500   # 3.0s min

# Gate timeout — max extra steps to wait for gate before forcing advance
GATE_TIMEOUT = 500   # 1.0s extra before forcing advance

CONTACT_PROXIMITY_M = 0.06
CONTACT_MAX_STEPS   = 1500

FR_HIP   = 0
FR_THIGH = 1
FR_CALF  = 2

# Joint threshold for gate checks
GATE_THRESHOLD = 0.15   # radians

# Foot force contact detection (stub — currently unreliable on this hardware)
# Set use_foot_force=True in execute() once FR foot force is characterised.
# Threshold is the spike above baseline that indicates button contact.
# Measure baseline using print_pose.py while FR foot is lifted in the air.
FOOT_FORCE_CONTACT_THRESHOLD = 10.0   # raw units — TO BE CALIBRATED
FR_FOOT_FORCE_IDX            = 0      # index into low_state.foot_force for FR foot

# ── Poses — measured from real robot ──────────────────────────────────────────

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

# Weight-shift target posture: small lateral hip abduction + slight thigh
# lower on the FL/RL side, RR hip outward. Empirically tuned with KP_STABLE=100
# / tau=0; produces a bounded few-degree static sag during extend+hold that's
# operationally fine for baseline_2. The Stage C waypoint head was trained at
# this posture, so deployment-time waypoint predictions remain aligned.
#
# A handshake-tripod variant of this constant was tested (deeper rear squat,
# matching Sport Mode's hold pose). It was reverted because that posture is
# only stable with feedforward torque, which our base PD-only path lacks.
WEIGHT_SHIFT_POS = list(STAND_POS)
WEIGHT_SHIFT_POS[3]  = STAND_POS[3]  - 0.30   # FL_hip inward
WEIGHT_SHIFT_POS[9]  = STAND_POS[9]  - 0.25   # RL_hip inward
WEIGHT_SHIFT_POS[6]  = STAND_POS[6]  + 0.15   # RR_hip outward
WEIGHT_SHIFT_POS[4]  = STAND_POS[4]  - 0.10   # FL_thigh lower
WEIGHT_SHIFT_POS[10] = STAND_POS[10] - 0.10   # RL_thigh lower

# ── FR leg offsets — wall press ───────────────────────────────────────────────
# Leg swings forward and out to reach a wall-mounted button.
# Thigh goes strongly negative (paw-forward), calf stays near neutral.
FR_LIFT_OFFSET_WALL   = np.array([+0.04, -2.0, +0.10])
FR_EXTEND_OFFSET_WALL = np.array([+0.04, -2.0, +0.52])

# ── FR leg offsets — ground press ─────────────────────────────────────────────
# Button is ~3 inches (7.6 cm) in front of FR foot at standing, ~1.5 inches tall.
# Motion: small lift to clear ground → swing slightly forward → press straight down.
# Thigh goes moderately negative (much less than wall press — no need to reach far).
# Calf extends (positive offset) to push paw downward onto button surface.
#
# IMPORTANT: These are geometry-estimated starting values only.
# Tune empirically using print_pose.py:
#   1. Run in low-level, weight-shifted stance
#   2. Command FR leg to lift target — verify FR foot z rises above button top
#   3. Command FR leg to press target — verify FR foot lands on button center
#   4. Read joint angles from print_pose.py and update values below
FR_LIFT_OFFSET_GROUND  = np.array([+0.04, -0.8, +0.1])   # gentle lift, calf tucks
FR_PRESS_OFFSET_GROUND = np.array([+0.04, -1.0, +0.8])   # forward + calf pushes down

# How far the calf retracts during the curl step of retract
# Tucks foot clear of button before rotating leg back
FR_RETRACT_CURL_CALF_OFFSET = -0.5   # relative to current calf at press position


# ──────────────────────────────────────────────
# Dynamic gravity-compensation FF (shared by base + WholeBody subclass)
# ──────────────────────────────────────────────
#
# Approximates per-motor static gravity-comp torque at the current joint
# configuration. Applied as motor_cmd[i].tau on top of the existing PD
# command, so kp/kd only need to do tracking work — gravity load is held
# by the FF term.
#
# Phases where this fires are configurable per controller; default is the
# weight-bearing tripod set {lift, extend, hold, retract_curl, retract_rotate,
# retract_extend}. Quadripod transition phases (sit_to_stand, weight_shift,
# weight_unshift, settle, lower_to_sit) are excluded by default — they involve
# rapid posture change where a mismatched FF can fight the transition.
#
# Modeling choices, with rationale:
#
#   - Calf: full analytical J_z^T · F using the same simplified FK as
#     _estimate_fr_foot. Sign matches Sport Mode handshake captures;
#     magnitude is ~70% of Sport Mode at default body mass. Tunable via
#     gravity_ff_body_mass on the controller.
#
#   - Thigh: empirical scalar (the analytical J_z^T·F gives the wrong sign
#     in deep-squat postures because the simple model ignores body CoM
#     offset and per-segment leg masses). Sport Mode handshake captures
#     give ~+3 N·m on each support thigh at ~12 kg / 3-leg load. We scale
#     that by per-leg load relative to the same nominal so the magnitude
#     adapts to body-mass tuning, even though it stays roughly posture-
#     independent.
#
#   - Hip: 0. The simple FK has hip rotation about body-X, which doesn't
#     change foot z and so can't be reached by this gravity-only model.
#     Sport Mode hip captures are small (-1 to +3 N·m) and leg-asymmetric;
#     not worth the extra calibration effort given how tiny the moment is.
#
# FR is excluded from FF whenever it's airborne (matches Sport Mode's
# tau=0 on FR throughout the handshake).
GO2_NOMINAL_MASS_KG  = 15.0
GO2_GRAVITY          = 9.81
SUPPORT_THIGH_FF_BASELINE_NM = 3.0   # at 1/3·12kg ≈ 39N per leg (Sport Mode median)
GRAVITY_FF_DEFAULT_PHASES: Tuple[str, ...] = (
    "lift", "extend", "hold",
    "retract_curl", "retract_rotate", "retract_extend",
)


def _per_leg_gravity_ff(
    q_thigh: float,
    q_calf: float,
    w_per_leg_n: float,
) -> Tuple[float, float, float]:
    """Approximate static gravity-comp torque (hip, thigh, calf) for one leg.

    See module-level docstring above for the modeling decisions and limits.
    """
    sin_tc = math.sin(q_thigh + q_calf)

    # Hip: 0 in the simple FK (rotation axis doesn't move foot z).
    tau_hip = 0.0

    # Thigh: empirical, scaled by per-leg load relative to the nominal that
    # SUPPORT_THIGH_FF_BASELINE_NM was measured at.
    nominal_w = GO2_NOMINAL_MASS_KG * GO2_GRAVITY / 3.0
    tau_thigh = SUPPORT_THIGH_FF_BASELINE_NM * (w_per_leg_n / nominal_w)

    # Calf: analytical. Foot upward reaction with Z-Jacobian = L_C·sin(thigh+calf);
    # motor must apply opposite to resist the gravity-induced bend, hence the
    # leading negative sign.
    L_CALF = 0.213
    tau_calf = -L_CALF * sin_tc * w_per_leg_n

    return tau_hip, tau_thigh, tau_calf


def compute_gravity_ff_torques(
    q_actual: Iterable[float],
    *,
    fr_in_air: bool,
    body_mass_kg: float = GO2_NOMINAL_MASS_KG,
) -> np.ndarray:
    """Build a (12,) feed-forward torque array for the current pose.

    Indexing matches the controller's canonical FR/FL/RR/RL × hip/thigh/calf
    order. FR slot is 0 by default when ``fr_in_air`` is True (matches Sport
    Mode handshake behaviour); set ``fr_in_air=False`` for quadripod phases
    where FR should also bear weight.
    """
    q = list(q_actual)
    if len(q) < 12:
        raise ValueError(f"q_actual must have 12 entries, got {len(q)}")

    num_support = 3 if fr_in_air else 4
    if num_support <= 0 or body_mass_kg <= 0:
        return np.zeros(12, dtype=np.float32)
    w_per_leg = body_mass_kg * GO2_GRAVITY / num_support

    tau = np.zeros(12, dtype=np.float32)
    if not fr_in_air:
        tau_fr = _per_leg_gravity_ff(q[1], q[2], w_per_leg)
        tau[0:3] = tau_fr
    for base_idx in (3, 6, 9):
        tau_leg = _per_leg_gravity_ff(
            q[base_idx + 1], q[base_idx + 2], w_per_leg)
        tau[base_idx:base_idx + 3] = tau_leg
    return tau


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ContactResult:
    success:            bool
    contact_step:       int
    timeout:            bool
    press_mode:         str
    target_offset_xyz:  np.ndarray
    joint_trajectory:   np.ndarray   # (T, 12) — hold phase, actual positions
    fr_foot_trajectory: np.ndarray   # (T, 3)  — hold phase, FK estimates
    duration_s:         float
    phase_transitions:  dict         # (v2.0 spec): maps phase_name_end
                                     # → 12-dim np.float32 actual joint vector
                                     # captured at each gate-passed transition.
                                     # Mandatory keys for Stage C training:
                                     #   "lift_end", "extend_end", "hold_end"
    contact_method:     str          # NEW (v2.1 spec): which signal fired first.
                                     # One of: "fk", "audio", "foot_force", "none".
                                     # "none" means hold timed out via
                                     # CONTACT_MAX_STEPS without any contact.


# ──────────────────────────────────────────────
# Main controller
# ──────────────────────────────────────────────

class HeuristicContact:

    def __init__(
        self,
        network_interface: str = DEFAULT_INTERFACE,
        already_initialized: bool = False,
        fr_waypoints: Optional[Dict[str, np.ndarray]] = None,
        stage_d_inference: Optional[Any] = None,
        stage_d_phases: Optional[Set[str]] = None,
        grounding_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        stage_d_residual_mask: Optional[np.ndarray] = None,
        gravity_ff_enabled: bool = False,
        gravity_ff_body_mass: float = GO2_NOMINAL_MASS_KG,
        gravity_ff_phases: Optional[Set[str]] = None,
    ):
        # ── Optional deployment hooks (Friday demo) ────────────────────────
        # When ``fr_waypoints`` is provided, the lift/extend/hold FR targets
        # come from the dict instead of STAND_POS + module-level offsets.
        # When ``stage_d_inference`` is provided, a (12,) residual is added
        # to target_q every control step whose phase is in ``stage_d_phases``
        # (default {"lift", "extend", "hold"}). All four params default to
        # None / preserve baseline behavior; setting any one alone is safe.
        self._fr_waypoints: Optional[Dict[str, np.ndarray]] = fr_waypoints
        self._stage_d = stage_d_inference
        self._stage_d_phases: Set[str] = (
            set(stage_d_phases) if stage_d_phases is not None
            else {"lift", "extend", "hold"}
        )
        self._grounding_getter = grounding_getter
        # Optional element-wise multiplier applied to the (12,) Stage D
        # residual before it is added to the joint command. None means
        # "apply residual unmasked" (default behaviour). Used by deployment
        # scripts to e.g. zero rear-leg residuals for diagnostic comparison.
        if stage_d_residual_mask is not None:
            mask = np.asarray(stage_d_residual_mask, dtype=np.float32)
            if mask.shape != (12,):
                raise ValueError(
                    f"stage_d_residual_mask must be (12,), got {mask.shape}")
            self._stage_d_residual_mask: Optional[np.ndarray] = mask
        else:
            self._stage_d_residual_mask = None
        # Dynamic gravity-comp FF. When True, _send_cmd applies an analytical
        # per-leg static torque (calf) plus an empirical thigh constant during
        # the configured phases (default {lift, extend, hold, retract_*}). FR
        # is excluded while airborne. Subclasses inherit this directly via
        # super().__init__; no per-subclass logic is needed.
        self._gravity_ff_enabled = bool(gravity_ff_enabled)
        self._gravity_ff_body_mass = float(gravity_ff_body_mass)
        self._gravity_ff_phases: Set[str] = (
            set(gravity_ff_phases) if gravity_ff_phases is not None
            else set(GRAVITY_FF_DEFAULT_PHASES)
        )
        # Cached "last valid" target_pos_base for Stage D state. Filled at
        # execute() entry from self._target_offset; refreshed each step from
        # ``grounding_getter`` when one is configured. Never NaN — Stage D
        # was trained on continuous values only.
        self._stage_d_target_cache: Optional[np.ndarray] = None
        # Phase progress in [0, 1] within the current phase, set in
        # _control_loop and read by the Stage D state builder. 0.0 outside
        # of an active phase.
        self._phase_progress: float = 0.0

        self._crc        = CRC()
        self._low_state  = None
        self._low_cmd    = unitree_go_msg_dds__LowCmd_()
        self._state_lock = threading.Lock()
        self._stop_flag  = threading.Event()

        self._phase        = "idle"
        self._phase_step   = 0
        self._press_mode   = "ground"  # set properly in execute()
        self._start_pos    = list(SIT_POS)
        self._done         = False
        self._contact_step = -1
        # NEW (v2.1 spec): which signal fired first — "fk", "audio", "foot_force",
        # or "none" if hold timed out without contact. Reset per episode.
        self._contact_method = "none"
        self._joint_traj   = []
        self._fr_foot_traj = []

        # FIX: cache last sent command so we can resend on stale-state cycles
        # instead of silently skipping _send_cmd, which caused single-cycle
        # torque dropouts ("one damp") at 500 Hz under GIL contention.
        self._last_target_q: list | None = None

        # Dynamic pose storage — captured at phase transitions from actual positions
        self._weight_shift_end = list(STAND_POS)
        self._lift_end_pos    = list(STAND_POS)
        self._hold_pos        = list(STAND_POS)
        self._retract_start      = list(STAND_POS)
        self._retract_curl_end   = list(STAND_POS)
        self._retract_rotate_end = list(STAND_POS)
        self._unshift_start   = list(STAND_POS)
        self._settle_start    = list(STAND_POS)
        self._lower_start     = list(STAND_POS)

        # ── (v2.0 spec) — external observer / data collection interface ──────
        # Actual joint positions captured at each gate-passed phase transition.
        # Populated inside _control_loop at the moment should_advance() returns
        # True, before self._phase advances. Reset per episode in execute().
        # Returned in ContactResult.phase_transitions.
        self._phase_transitions: dict = {}

        # Most recent commanded FR pose [hip, thigh, calf], updated every
        # _control_loop call including during intra-phase interpolation.
        # Exposed (not underscored) so external observer threads
        # (e.g., StageDRecorder) can read it at 500 Hz as the "nominal"
        # waypoint the heuristic is currently driving toward.
        self.current_waypoint_fr = np.zeros(3, dtype=np.float32)
        # ── end v2.0 additions ───────────────────────────────────────────────

        # ── NEW (v2.1 spec) — optional audio-based contact detection ─────────
        # Set by execute() when an audio detector is passed in. Interface:
        #   - .sound_detected: bool, True once energy spike was detected
        #                      (duck-typed; we only read this attribute)
        # None means no audio detection this episode (behavior identical to v2.0).
        self._audio_detector = None
        # ── end v2.1 additions ───────────────────────────────────────────────

        # ── NEW (Stage D v2) — compliance mode for hand-guided collection ────
        # Base class default: False (rigid FR tracking, identical to v2.1).
        # HeuristicContactGuided flips this to True so _send_cmd drops the FR
        # leg to KP_FR_COMPLIANT/KD_FR_COMPLIANT during extend + hold.
        self._compliance_active: bool = False

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
        target_offset_xyz:   np.ndarray,
        press_mode:          Literal["wall", "ground"] = "ground",
        contact_proximity_m: float = CONTACT_PROXIMITY_M,
        use_foot_force:      bool  = False,
        audio_detector:      Optional[object] = None,
        target_refresh_fn:   Optional[Callable[[], Optional[np.ndarray]]] = None,  # NEW
        # NEW (calibration support): per-call offset overrides. If None,
        # fall back to module-level FR_*_OFFSET_* constants. Used by
        # scripts/calibrate_wall_standoff.py to tune offsets live between
        # attempts without editing this file.
        fr_lift_offset_wall:    Optional[np.ndarray] = None,
        fr_extend_offset_wall:  Optional[np.ndarray] = None,
        fr_lift_offset_ground:  Optional[np.ndarray] = None,
        fr_press_offset_ground: Optional[np.ndarray] = None,
    ) -> ContactResult:
        """
        Execute the full contact sequence.

        Args:
            target_offset_xyz:   3D target position in robot base frame [x, y, z].
                                 Used for FK proximity contact detection.
            press_mode:          "wall" — FR leg swings forward/out to wall button.
                                 "ground" — FR leg lifts and presses down onto floor button.
            contact_proximity_m: FK proximity threshold for contact detection (m).
            use_foot_force:      Enable foot-force contact detection (stub).
                                 Currently disabled — FR foot force unreliable on this hardware.
                                 Enable only after calibrating FOOT_FORCE_CONTACT_THRESHOLD
                                 using print_pose.py (lifted baseline vs pressed contact).
            audio_detector:      NEW (v2.1). Optional object with a `sound_detected: bool`
                                 attribute. When provided, an audio energy spike during
                                 the hold phase counts as contact alongside FK proximity
                                 and foot force. The first signal to fire sets
                                 self._contact_step and self._contact_method.
                                 Caller is responsible for lifecycle: call
                                 audio_detector.start_episode() before execute() and
                                 reset/stop after. Pass None (default) for no audio.
        """
        self._joint_traj        = []
        self._fr_foot_traj      = []
        # (v2.0 spec): reset phase transition snapshots for this episode.
        self._phase_transitions = {}
        self._contact_step      = -1
        # NEW (v2.1 spec): reset contact-method flag for this episode.
        self._contact_method    = "none"
        self._done              = False
        self._press_mode        = press_mode
        self._target_offset     = target_offset_xyz
        self._contact_proximity = contact_proximity_m
        self._use_foot_force    = use_foot_force
        self._last_target_q     = None
        # NEW (v2.1 spec): wire audio detector for this episode. None is fine.
        self._audio_detector    = audio_detector
        self._target_refresh_fn = target_refresh_fn

        # NEW (calibration support): resolve runtime offsets. If caller
        # passes explicit overrides they win; otherwise use the module-level
        # defaults. Stored on self so _control_loop can read them without
        # further plumbing.
        self._fr_lift_offset_wall_runtime = (
            np.asarray(fr_lift_offset_wall, dtype=np.float64)
            if fr_lift_offset_wall is not None
            else FR_LIFT_OFFSET_WALL
        )
        self._fr_extend_offset_wall_runtime = (
            np.asarray(fr_extend_offset_wall, dtype=np.float64)
            if fr_extend_offset_wall is not None
            else FR_EXTEND_OFFSET_WALL
        )
        self._fr_lift_offset_ground_runtime = (
            np.asarray(fr_lift_offset_ground, dtype=np.float64)
            if fr_lift_offset_ground is not None
            else FR_LIFT_OFFSET_GROUND
        )
        self._fr_press_offset_ground_runtime = (
            np.asarray(fr_press_offset_ground, dtype=np.float64)
            if fr_press_offset_ground is not None
            else FR_PRESS_OFFSET_GROUND
        )

        # Seed the Stage D target cache with the initial standoff target so
        # foot_to_target_error has a sensible value before grounding fires.
        self._stage_d_target_cache = (
            np.asarray(target_offset_xyz, dtype=np.float32).copy()
        )

        if use_foot_force:
            logger.warning(
                "Foot force contact detection enabled but currently a stub. "
                "FR foot force was unreliable during initial testing. "
                "Calibrate FOOT_FORCE_CONTACT_THRESHOLD with print_pose.py first."
            )

        if audio_detector is not None:
            logger.info("Audio-based contact detection enabled.")

        logger.info(
            f"execute() — press_mode='{press_mode}'  target={target_offset_xyz}"
        )

        t_start = time.time()

        self._switch_to_lowlevel()
        self._wait_for_state()

        with self._state_lock:
            self._start_pos = [self._low_state.motor_state[i].q for i in range(12)]

        logger.info(
            f"Start pose FR: hip={self._start_pos[0]:+.3f} "
            f"thigh={self._start_pos[1]:+.3f} "
            f"calf={self._start_pos[2]:+.3f}"
        )

        self._phase      = "sit_to_stand"
        self._phase_step = 0
        self._stop_flag.clear()

        self._write_thread = threading.Thread(
            target=self._control_loop_thread,
            name="heuristic_contact",
            daemon=True,
        )
        self._write_thread.start()

        # Generous timeout — gates ensure correctness, this is a safety net
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
            press_mode=press_mode,
            target_offset_xyz=target_offset_xyz,
            joint_trajectory=np.array(self._joint_traj)
                if self._joint_traj else np.zeros((0, 12)),
            fr_foot_trajectory=np.array(self._fr_foot_traj)
                if self._fr_foot_traj else np.zeros((0, 3)),
            duration_s=duration,
            # (v2.0 spec): return a copy so caller can safely mutate.
            phase_transitions=self._phase_transitions.copy(),
            # NEW (v2.1 spec): record which signal fired first.
            contact_method=self._contact_method,
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
        # FIX: If low_state is momentarily stale (subscriber blocked by GIL
        # or lock contention), resend the last command instead of silently
        # returning. Skipping a publish at 500 Hz causes a 2ms torque dropout
        # visible as the "one damp" shake seen at phase transitions on the robot.
        if self._low_state is None:
            if self._last_target_q is not None:
                self._send_cmd(self._last_target_q)
            return

        self._phase_step += 1

        # Read current actual joint positions and IMU under lock
        with self._state_lock:
            actual   = [self._low_state.motor_state[i].q for i in range(12)]
            roll     = self._low_state.imu_state.rpy[0]
            pitch    = self._low_state.imu_state.rpy[1]
            # Foot force — read every cycle; only acted on if use_foot_force=True
            # Index 0 = FR foot. Units are hardware-specific (not calibrated).
            fr_force = self._low_state.foot_force[FR_FOOT_FORCE_IDX]

        target_q = list(STAND_POS)

        # ── Select offsets based on press_mode ───────────────────────
        # Uses runtime overrides if execute() was called with explicit
        # offset kwargs; otherwise falls back to module-level constants
        # via the _runtime attributes set at execute() entry.
        if self._press_mode == "ground":
            lift_offset   = self._fr_lift_offset_ground_runtime
            extend_offset = self._fr_press_offset_ground_runtime
        else:
            lift_offset   = self._fr_lift_offset_wall_runtime
            extend_offset = self._fr_extend_offset_wall_runtime

        # ── Gate helpers ──────────────────────────────────────────────
        def gate_passed(target_pos, indices, threshold=GATE_THRESHOLD):
            """True if all specified joints are within threshold of target."""
            return all(
                abs(actual[i] - target_pos[i]) < threshold
                for i in indices
            )

        def should_advance(min_steps, gate_ok):
            """Advance if gate passed after min steps, or force after timeout."""
            if self._phase_step >= min_steps and gate_ok:
                return True
            if self._phase_step >= min_steps + GATE_TIMEOUT:
                logger.warning(
                    f"Phase '{self._phase}' gate timeout — forcing advance"
                )
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
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["sit_to_stand_end"] = \
                    np.array(actual, dtype=np.float32)
                # Refresh target post-stand — base may have drifted during sit_to_stand
                if self._target_refresh_fn is not None:
                    try:
                        fresh = self._target_refresh_fn()
                    except Exception as e:
                        logger.warning(f"Target refresh failed: {e}")
                        fresh = None
                    if fresh is not None:
                        old_target = self._target_offset.copy()
                        fresh_xy_only = np.asarray(fresh, dtype=np.float64).copy()
                        fresh_xy_only[2] = old_target[2]  # preserve original z
                        self._target_offset = fresh_xy_only
                        delta = np.linalg.norm(self._target_offset[:2] - old_target[:2])
                        logger.info(
                            f"Target refreshed (xy only) post-sit_to_stand: "
                            f"old=[{old_target[0]:+.3f}, {old_target[1]:+.3f}, {old_target[2]:+.3f}]  "
                            f"new=[{self._target_offset[0]:+.3f}, {self._target_offset[1]:+.3f}, {self._target_offset[2]:+.3f}]  "
                            f"shift_xy={delta*100:.1f}cm"
                        )
                self._phase      = "weight_shift"
                self._phase_step = 0

        # ── weight shift ──────────────────────────────────────────────
        elif self._phase == "weight_shift":
            alpha = min(self._phase_step / STEPS_WEIGHT_SHIFT, 1.0)
            for i in range(12):
                target_q[i] = (1-alpha)*STAND_POS[i] + alpha*WEIGHT_SHIFT_POS[i]

            gate_ok = gate_passed(WEIGHT_SHIFT_POS, [3, 9, 6], threshold=0.15)
            if should_advance(STEPS_WEIGHT_SHIFT, gate_ok):
                logger.info("✓ weight_shift complete")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["weight_shift_end"] = \
                    np.array(actual, dtype=np.float32)
                self._weight_shift_end = list(actual)
                self._phase      = "lift"
                self._phase_step = 0

        # ── lift FR ───────────────────────────────────────────────────
        elif self._phase == "lift":
            alpha    = min(self._phase_step / STEPS_LIFT, 1.0)
            self._phase_progress = alpha
            target_q = list(self._weight_shift_end)
            fr_s = np.array([
                WEIGHT_SHIFT_POS[FR_HIP],
                WEIGHT_SHIFT_POS[FR_THIGH],
                WEIGHT_SHIFT_POS[FR_CALF],
            ])
            if self._fr_waypoints is not None:
                fr_l = np.asarray(
                    self._fr_waypoints["lift"], dtype=np.float64)
            else:
                fr_l = np.array([
                    STAND_POS[FR_HIP]   + lift_offset[0],
                    STAND_POS[FR_THIGH] + lift_offset[1],
                    STAND_POS[FR_CALF]  + lift_offset[2],
                ])
            target_q[FR_HIP]   = (1-alpha)*fr_s[0] + alpha*fr_l[0]
            target_q[FR_THIGH] = (1-alpha)*fr_s[1] + alpha*fr_l[1]
            target_q[FR_CALF]  = (1-alpha)*fr_s[2] + alpha*fr_l[2]

            gate_ok = abs(actual[FR_THIGH] - fr_l[1]) < GATE_THRESHOLD
            if should_advance(STEPS_LIFT, gate_ok):
                logger.info(f"✓ lift complete  FR_thigh={actual[FR_THIGH]:+.3f}")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["lift_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase        = "extend"
                self._phase_step   = 0
                self._lift_end_pos = list(target_q)

        # ── extend / press-down FR ────────────────────────────────────
        elif self._phase == "extend":
            alpha    = min(self._phase_step / STEPS_EXTEND, 1.0)
            self._phase_progress = alpha
            target_q = list(self._lift_end_pos)
            if self._fr_waypoints is not None:
                fr_e = np.asarray(
                    self._fr_waypoints["extend"], dtype=np.float64)
            else:
                fr_e = np.array([
                    STAND_POS[FR_HIP]   + extend_offset[0],
                    STAND_POS[FR_THIGH] + extend_offset[1],
                    STAND_POS[FR_CALF]  + extend_offset[2],
                ])
            fr_l = np.array([
                self._lift_end_pos[FR_HIP],
                self._lift_end_pos[FR_THIGH],
                self._lift_end_pos[FR_CALF],
            ])
            target_q[FR_HIP]   = (1-alpha)*fr_l[0] + alpha*fr_e[0]
            target_q[FR_THIGH] = (1-alpha)*fr_l[1] + alpha*fr_e[1]
            target_q[FR_CALF]  = (1-alpha)*fr_l[2] + alpha*fr_e[2]

            gate_ok = abs(actual[FR_THIGH] - fr_e[1]) < GATE_THRESHOLD
            if should_advance(STEPS_EXTEND, gate_ok):
                logger.info(f"✓ extend complete  FR_thigh={actual[FR_THIGH]:+.3f}")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["extend_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase      = "hold"
                self._phase_step = 0
                self._hold_pos   = list(target_q)
                # When Stage C waypoints are wired up, the press waypoint
                # supersedes the extend target as the hold pose. Difference
                # is small in training data, so a step transition is fine.
                if self._fr_waypoints is not None:
                    press_fr = self._fr_waypoints["press"]
                    self._hold_pos[FR_HIP]   = float(press_fr[0])
                    self._hold_pos[FR_THIGH] = float(press_fr[1])
                    self._hold_pos[FR_CALF]  = float(press_fr[2])

        # ── hold + detect contact ─────────────────────────────────────
        elif self._phase == "hold":
            # Match recorder.PHASE_DURATION_S["hold"] = 3.0s = 1500 steps so
            # the deployment phase_progress matches what Stage D was trained on.
            self._phase_progress = min(self._phase_step / 1500.0, 1.0)
            target_q = list(self._hold_pos)
            # Record actual positions during hold for training data
            self._joint_traj.append(list(actual))
            fr_foot = self._estimate_fr_foot(actual)
            self._fr_foot_traj.append(fr_foot.tolist())

            # ── Contact detection — three parallel signals ─────────────
            # Whichever fires first wins and its method is recorded in
            # self._contact_method for downstream metadata.

            # Primary: FK proximity contact detection
            dist       = np.linalg.norm(fr_foot - self._target_offset)
            fk_contact = dist < self._contact_proximity

            # Secondary: foot force spike (stub — disabled by default)
            # FR foot force was unreliable during initial hardware testing.
            # To enable: pass use_foot_force=True to execute() and first
            # calibrate FOOT_FORCE_CONTACT_THRESHOLD with print_pose.py
            # (measure baseline with FR foot lifted, threshold above that).
            force_contact = False
            if self._use_foot_force:
                # TODO: set FOOT_FORCE_CONTACT_THRESHOLD after calibration
                force_contact = fr_force > FOOT_FORCE_CONTACT_THRESHOLD
                if force_contact:
                    logger.debug(
                        f"Foot force contact trigger: fr_force={fr_force:.1f}"
                    )

            # NEW (v2.1 spec): audio-based contact detection.
            # The audio detector runs a background thread listening for an
            # energy spike on the USB mic. When the button's sound module
            # fires, sound_detected flips True. We treat this as the
            # authoritative physical-contact signal — the button was
            # physically depressed enough to trigger its sound module.
            # Read is atomic (bool); no lock needed.
            audio_contact = False
            if self._audio_detector is not None:
                audio_contact = bool(
                    getattr(self._audio_detector, "sound_detected", False)
                )

            contact_detected = fk_contact or force_contact or audio_contact

            if contact_detected and self._contact_step < 0:
                self._contact_step = self._phase_step
                # Record which signal fired first. Priority order reflects
                # reliability: audio is physical proof, FK is geometric
                # estimate, foot force is currently unreliable.
                # Multiple signals may fire on the same step; we pick the
                # most authoritative.
                if audio_contact:
                    self._contact_method = "audio"
                elif fk_contact:
                    self._contact_method = "fk"
                elif force_contact:
                    self._contact_method = "foot_force"
                logger.info(
                    f"Contact detected at step {self._contact_step} "
                    f"dist={dist:.3f}m  fr_force={fr_force:.1f}  "
                    f"audio={audio_contact}  method={self._contact_method}"
                )

            contact_done = (
                self._contact_step >= 0
                and self._phase_step >= self._contact_step + STEPS_HOLD
            )
            if contact_done or self._phase_step >= CONTACT_MAX_STEPS:
                logger.info("✓ hold complete — retracting")
                # (v2.0 spec): capture actual at gate-passed transition.
                # This is the "press_waypoint" label for Stage C training — the
                # actual FR joint pose at the end of hold, just before retract.
                self._phase_transitions["hold_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase         = "retract_curl"
                self._phase_step    = 0
                # FIX: capture actual positions, not commanded target_q.
                # With KP_FR=40 (soft), FR leg may not fully reach hold_pos.
                # Using commanded pos as retract start creates a discontinuity
                # on the first retract step, causing a jerk on all legs.
                self._retract_start = list(actual)

        # ── retract step 1: curl calf up clear of button ──────────────
        elif self._phase == "retract_curl":
            alpha    = min(self._phase_step / STEPS_RETRACT_CURL, 1.0)
            target_q = list(self._retract_start)
            # Only move the calf — thigh stays at press position
            curl_calf_target = self._retract_start[FR_CALF] + FR_RETRACT_CURL_CALF_OFFSET
            target_q[FR_CALF] = (
                (1-alpha) * self._retract_start[FR_CALF]
                + alpha   * curl_calf_target
            )

            gate_ok = abs(actual[FR_CALF] - curl_calf_target) < GATE_THRESHOLD
            if should_advance(STEPS_RETRACT_CURL, gate_ok):
                logger.info(f"✓ retract_curl complete  FR_calf={actual[FR_CALF]:+.3f}")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["retract_curl_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase           = "retract_rotate"
                self._phase_step      = 0
                self._retract_curl_end = list(actual)

        # ── retract step 2: rotate thigh back to weight shift ─────────
        elif self._phase == "retract_rotate":
            alpha    = min(self._phase_step / STEPS_RETRACT_ROTATE, 1.0)
            target_q = list(self._retract_curl_end)
            # Only move the thigh — calf stays tucked
            target_q[FR_THIGH] = (
                (1-alpha) * self._retract_curl_end[FR_THIGH]
                + alpha   * WEIGHT_SHIFT_POS[FR_THIGH]
            )

            gate_ok = abs(actual[FR_THIGH] - WEIGHT_SHIFT_POS[FR_THIGH]) < GATE_THRESHOLD
            if should_advance(STEPS_RETRACT_ROTATE, gate_ok):
                logger.info(f"✓ retract_rotate complete  FR_thigh={actual[FR_THIGH]:+.3f}")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["retract_rotate_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase             = "retract_extend"
                self._phase_step        = 0
                self._retract_rotate_end = list(actual)

        # ── retract step 3: extend calf back to weight shift ──────────
        elif self._phase == "retract_extend":
            alpha    = min(self._phase_step / STEPS_RETRACT_EXTEND, 1.0)
            target_q = list(self._retract_rotate_end)
            target_q[FR_CALF] = (
                (1-alpha) * self._retract_rotate_end[FR_CALF]
                + alpha   * WEIGHT_SHIFT_POS[FR_CALF]
            )

            gate_ok = abs(actual[FR_CALF] - WEIGHT_SHIFT_POS[FR_CALF]) < GATE_THRESHOLD
            if should_advance(STEPS_RETRACT_EXTEND, gate_ok):
                logger.info(f"✓ retract_extend complete  FR_calf={actual[FR_CALF]:+.3f}")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["retract_extend_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase         = "weight_unshift"
                self._phase_step    = 0
                self._unshift_start = list(actual)

        # ── weight unshift ────────────────────────────────────────────
        elif self._phase == "weight_unshift":
            alpha = min(self._phase_step / STEPS_WEIGHT_SHIFT_BACK, 1.0)
            for i in range(12):
                target_q[i] = (
                    (1-alpha) * self._unshift_start[i]
                    + alpha   * STAND_POS[i]
                )

            gate_ok = gate_passed(STAND_POS, range(12), threshold=0.15)
            if should_advance(STEPS_WEIGHT_SHIFT_BACK, gate_ok):
                logger.info("✓ weight_unshift complete")
                # Note (v2.0 spec): no snapshot for weight_unshift → settle per
                # spec Section 7.1 — only 9 snapshots requested, and this one
                # is not among them.
                self._phase        = "settle"
                self._phase_step   = 0
                self._settle_start = list(actual)

        # ── settle — smooth actual → STAND_POS ───────────────────────
        elif self._phase == "settle":
            alpha = min(self._phase_step / STEPS_SETTLE, 1.0)
            for i in range(12):
                target_q[i] = (
                    (1-alpha) * self._settle_start[i]
                    + alpha   * STAND_POS[i]
                )

            # FIX: loosened threshold 0.10 → 0.15 to avoid forcing advance
            # mid-oscillation, which caused a visible single shake before sit.
            gate_ok = gate_passed(STAND_POS, range(12), threshold=0.15)
            if should_advance(STEPS_SETTLE, gate_ok):
                logger.info("✓ settle complete — starting lower_to_sit")
                # (v2.0 spec): capture actual at gate-passed transition
                self._phase_transitions["settle_end"] = \
                    np.array(actual, dtype=np.float32)
                self._phase       = "lower_to_sit"
                self._phase_step  = 0
                self._lower_start = list(STAND_POS)

        # ── lower to sit ──────────────────────────────────────────────
        elif self._phase == "lower_to_sit":
            t     = min(self._phase_step / STEPS_LOWER_TO_SIT, 1.0)
            alpha = t * t * (3 - 2 * t)   # smoothstep — avoids jerk at start/end
            for i in range(12):
                target_q[i] = (1-alpha)*self._lower_start[i] + alpha*SIT_POS[i]

            gate_ok = gate_passed(SIT_POS, range(12), threshold=0.2)
            if should_advance(STEPS_LOWER_TO_SIT, gate_ok):
                logger.info("✓ lower_to_sit complete")
                # Note (v2.0 spec): no snapshot for lower_to_sit → done per spec.
                self._phase = "done"

        # ── done ──────────────────────────────────────────────────────
        elif self._phase == "done":
            target_q   = list(SIT_POS)
            self._done = True
            # (v2.0 spec): keep current_waypoint_fr in sync on early-return
            # path for consistency with the main path below. Not strictly needed
            # (recorder only samples during lift/extend/hold) but avoids stale
            # values if a caller inspects this attribute after execute() returns.
            self.current_waypoint_fr = np.array(target_q[0:3], dtype=np.float32)
            self._send_cmd(target_q)
            return

        # ── IMU balance correction — hold and retract only ────────────
        # Disabled during weight_shift/lift/extend: those phases intentionally
        # shift the CoM and the IMU correction would fight the desired lean.
        # Re-enabled for hold and retract where the robot should be balanced.
        if self._phase in ("hold", "retract_curl", "retract_rotate", "retract_extend"):
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

        # (v2.0 spec): expose current FR command for external observer
        # threads (e.g., StageDRecorder) to log at 500 Hz as the "nominal"
        # waypoint. Updated every control step including during intra-phase
        # interpolation — not only at phase transitions.
        self.current_waypoint_fr = np.array(target_q[0:3], dtype=np.float32)

        # ── Stage D residual (deployment) ─────────────────────────────
        # When a Stage D inference handle is configured and the current
        # phase is in self._stage_d_phases, build the 33-dim state vector
        # (training_data_spec_v3.md §5.1), call the residual policy, and
        # add the (12,) result to target_q before publishing.
        if (self._stage_d is not None
                and self._phase in self._stage_d_phases):
            target_q = self._apply_stage_d_residual(
                target_q=target_q, actual=actual,
                roll=roll, pitch=pitch,
            )

        # Cache last command and send
        self._last_target_q = target_q
        self._send_cmd(target_q)

    def _apply_stage_d_residual(
        self,
        target_q: list,
        actual: list,
        roll: float,
        pitch: float,
    ) -> list:
        """Add the Stage D (33→12) residual to target_q during active phases.

        Returns a new list with the per-joint correction applied. If the
        target cache cannot be refreshed and was never seeded, returns the
        input unchanged — the residual is skipped rather than poisoned with
        NaN, since Stage D was trained on continuous values only.
        """
        # Refresh target_pos_base from the grounding thread when one is
        # configured. Keep the previous value if the call returns None.
        if self._grounding_getter is not None:
            try:
                fresh = self._grounding_getter()
            except Exception as e:
                logger.warning(f"grounding_getter raised: {e}")
                fresh = None
            if fresh is not None:
                fresh_arr = np.asarray(fresh, dtype=np.float32)
                if fresh_arr.shape == (3,) and not np.isnan(fresh_arr).any():
                    self._stage_d_target_cache = fresh_arr.copy()

        target = self._stage_d_target_cache
        if target is None:
            return target_q

        # Pull the additional motor / IMU fields needed for the state vector.
        # Reading under the same lock as the main read keeps these consistent
        # with the actual / roll / pitch already in hand.
        with self._state_lock:
            ms = self._low_state.motor_state
            dq        = [ms[i].dq      for i in range(12)]
            tau_est   = [ms[i].tau_est for i in range(12)]
            gyro      = list(self._low_state.imu_state.gyroscope[0:3])
            accel     = list(self._low_state.imu_state.accelerometer[0:3])

        fr_foot_fk = self._estimate_fr_foot(actual)
        foot_to_target_err = (target.astype(np.float32) - fr_foot_fk).astype(np.float32)

        state = np.empty(33, dtype=np.float32)
        state[0:3]   = np.asarray(actual[0:3], dtype=np.float32)
        state[3:6]   = np.asarray(dq[0:3],    dtype=np.float32)
        state[6:9]   = foot_to_target_err
        state[9:12]  = np.asarray(target_q[0:3], dtype=np.float32)
        state[12]    = float(self._phase_progress)
        state[13]    = float(roll)
        state[14]    = float(pitch)
        state[15:27] = np.asarray(tau_est, dtype=np.float32)
        state[27:30] = np.asarray(gyro,    dtype=np.float32)
        state[30:33] = np.asarray(accel,   dtype=np.float32)

        try:
            residual = self._stage_d.predict(state)
        except Exception as e:
            logger.warning(f"Stage D inference failed: {e} — residual skipped")
            return target_q

        if self._stage_d_residual_mask is not None:
            residual = residual * self._stage_d_residual_mask

        out = list(target_q)
        for i in range(12):
            out[i] = float(target_q[i]) + float(residual[i])
        return out

    def _send_cmd(self, target_q: list) -> None:
        for i in range(12):
            self._low_cmd.motor_cmd[i].q   = target_q[i]
            self._low_cmd.motor_cmd[i].dq  = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0

            if i in (FR_HIP, FR_THIGH, FR_CALF) and self._phase in (
                "lift", "extend", "hold",
                # FIX: keep KP_FR during retract — previously jumped to
                # KP_STABLE=100 here, causing a sudden torque spike on the
                # mid-air extended FR leg that destabilized the support legs.
                "retract_curl", "retract_rotate", "retract_extend"
            ):
                # Stage D v2: compliance mode softens the FR leg during
                # extend + hold so the operator can physically guide the paw
                # onto the button. lift stays at KP_FR so the leg gets to the
                # approach pose reliably, and retract stays at KP_FR so the
                # gain-restore transition out of hold → retract_curl starts
                # from _retract_start = list(actual) — the same mechanism that
                # eliminated the v2.0 retract torque spike.
                if self._compliance_active and self._phase in ("extend", "hold"):
                    self._low_cmd.motor_cmd[i].kp = KP_FR_COMPLIANT
                    self._low_cmd.motor_cmd[i].kd = KD_FR_COMPLIANT
                else:
                    self._low_cmd.motor_cmd[i].kp = KP_FR
                    self._low_cmd.motor_cmd[i].kd = KD_FR
            elif self._phase == "lower_to_sit":
                self._low_cmd.motor_cmd[i].kp = KP_SOFT
                self._low_cmd.motor_cmd[i].kd = KD_SOFT
            else:
                self._low_cmd.motor_cmd[i].kp = KP_STABLE
                self._low_cmd.motor_cmd[i].kd = KD_STABLE

        self._maybe_apply_gravity_ff()

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)

    def _maybe_apply_gravity_ff(self) -> None:
        """Patch motor_cmd[i].tau with dynamic static gravity-comp torques.

        Called from ``_send_cmd`` (and the WholeBody subclass override) after
        the per-motor gain loop has set ``tau=0``. No-op when
        ``self._gravity_ff_enabled`` is False or the current phase is not in
        ``self._gravity_ff_phases``. Reads the latest joint positions from
        ``rt/lowstate`` under the existing state lock so FF tracks the actual
        pose rather than the commanded pose (which would diverge under sag).
        """
        if not self._gravity_ff_enabled:
            return
        if self._phase not in self._gravity_ff_phases:
            return
        if self._low_state is None:
            return
        with self._state_lock:
            q_actual = [self._low_state.motor_state[i].q for i in range(12)]
        # FR is airborne for the full lift→retract_extend window; the default
        # gravity_ff_phases set is exactly that window so this stays True
        # there. If a caller adds a quadripod phase to gravity_ff_phases
        # they almost certainly want fr_in_air=False; we only have phase
        # information here so use the phase-derived flag.
        fr_in_air = self._phase in (
            "lift", "extend", "hold",
            "retract_curl", "retract_rotate", "retract_extend",
        )
        tau_ff = compute_gravity_ff_torques(
            q_actual,
            fr_in_air=fr_in_air,
            body_mass_kg=self._gravity_ff_body_mass,
        )
        for i in range(12):
            self._low_cmd.motor_cmd[i].tau = float(tau_ff[i])

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

        logger.info("Low-level active.")

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
        """Simplified FK for FR foot position in robot base frame."""
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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Heuristic contact phase controller — wall or ground press"
    )
    parser.add_argument("--interface",   type=str,   default=DEFAULT_INTERFACE)
    parser.add_argument("--target-x",   type=float,  default=0.27,
                        help="FR foot x + 7.6cm forward offset for ground press")
    parser.add_argument("--target-y",   type=float,  default=-0.10)
    parser.add_argument("--target-z",   type=float,  default=0.038,
                        help="Button top height in base frame (~0.038m for 1.5in button)")
    parser.add_argument("--mode",        type=str,   default="ground",
                        choices=["wall", "ground"],
                        help="Press mode: 'ground' (default) or 'wall'")
    parser.add_argument("--foot-force",  action="store_true",
                        help="Enable foot-force contact detection stub "
                             "(disabled by default — unreliable on this hardware)")
    parser.add_argument("--gravity-ff",  action="store_true",
                        help="Apply dynamic gravity-comp FF on FL/RR/RL "
                             "during lift/extend/hold/retract_*.")
    parser.add_argument("--ff-body-mass", type=float, default=GO2_NOMINAL_MASS_KG,
                        help="Body mass (kg) used to scale gravity FF torques. "
                             f"Default {GO2_NOMINAL_MASS_KG}; raise if rear "
                             f"still sags, lower if FF overshoots.")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Heuristic Contact — Gate-Based Phase Controller")
    print("="*60)
    print(f"Interface:  {args.interface}")
    print(f"Mode:       {args.mode}")
    print(f"Target:     ({args.target_x}, {args.target_y}, {args.target_z})")
    print(f"Foot force: {'enabled (stub)' if args.foot_force else 'disabled'}")
    print(f"Gravity FF: {'ENABLED (mass={:.1f} kg)'.format(args.ff_body_mass) if args.gravity_ff else 'disabled'}")
    print(f"Audio:      disabled (CLI does not wire a detector; pass via script)")
    print("\nPhases: sit_to_stand → weight_shift → lift → extend → "
          "hold → retract → weight_unshift → settle → lower_to_sit → done")
    print("Each phase waits for gate confirmation before advancing.")
    if args.mode == "ground":
        print(f"\nGround press offsets (tune with print_pose.py):")
        print(f"  FR_LIFT_OFFSET_GROUND  = {FR_LIFT_OFFSET_GROUND.tolist()}")
        print(f"  FR_PRESS_OFFSET_GROUND = {FR_PRESS_OFFSET_GROUND.tolist()}")
    else:
        print(f"\nWall press offsets:")
        print(f"  FR_LIFT_OFFSET_WALL    = {FR_LIFT_OFFSET_WALL.tolist()}")
        print(f"  FR_EXTEND_OFFSET_WALL  = {FR_EXTEND_OFFSET_WALL.tolist()}")
    print("\nWARNING: Robot will sit down, stand in low-level, lift FR leg.")
    input("Press Enter to continue (Ctrl+C to abort)...\n")

    controller = HeuristicContact(
        args.interface,
        gravity_ff_enabled=args.gravity_ff,
        gravity_ff_body_mass=args.ff_body_mass,
    )
    print("✓ Connected")

    if args.dry_run:
        print("Dry-run — skipping execution.")
    else:
        target = np.array([args.target_x, args.target_y, args.target_z])
        # CLI invocation leaves audio_detector=None (default).
        # Data collection scripts pass a real AudioLiveDetector instance.
        result = controller.execute(
            target_offset_xyz=target,
            press_mode=args.mode,
            use_foot_force=args.foot_force,
        )
        print(f"\nResult:")
        print(f"  success:           {result.success}")
        print(f"  contact_step:      {result.contact_step}")
        print(f"  contact_method:    {result.contact_method}")   # NEW (v2.1)
        print(f"  timeout:           {result.timeout}")
        print(f"  press_mode:        {result.press_mode}")
        print(f"  duration:          {result.duration_s:.2f}s")
        print(f"  traj_shape:        {result.joint_trajectory.shape}")
        # (v2.0 spec): print captured phase transitions for quick inspection
        print(f"  phase_transitions: {list(result.phase_transitions.keys())}")
        if "hold_end" in result.phase_transitions:
            fr_press = result.phase_transitions["hold_end"][0:3]
            print(f"    FR @ hold_end: hip={fr_press[0]:+.3f} "
                  f"thigh={fr_press[1]:+.3f} calf={fr_press[2]:+.3f}")