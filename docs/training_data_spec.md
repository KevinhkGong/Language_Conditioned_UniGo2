# Training Data Specification — Stage C & Stage D

**Project:** Language-Conditioned Whole-Body Contact Manipulation for Quadruped Robots
**Version:** 2.1
**Date:** April 21, 2026
**Purpose:** Define exactly what training data needs to be collected and in what format, so that data collection scripts (Priority 3) and network training code (Priority 4) can be implemented without ambiguity.

**Changes from v2.0:** Added USB-microphone-based audio contact verification. The button's built-in sound module (speaks its color when pressed, e.g., "RED") is used as the authoritative physical-press signal. New per-episode metadata fields: `success_fk`, `success_audio_live`, `contact_method`, `color_detected`, `color_expected`, `success_target`, `audio_transcript`. New HDF5 dataset: raw `audio` waveform. New Python modules: `AudioRecorder`, `AudioLiveDetector`, `ColorDetector`. New pip deps: `sounddevice`, `faster-whisper`, `scipy`. Modified `heuristic_contact.py`: optional `audio_detector` parameter on `execute()`; `ContactResult.contact_method` field.

**Changes from v1.0 (carried over):** Added Jacobian-PID expert for corrective data collection. Added `HeuristicContactCorrective` subclass. Added K_p tuning procedure. Added Jacobian-PID as a 5th evaluation baseline.

---

## 1. Scope

This document specifies training data for two learned components:

- **Stage C — Contact Planner MLP:** Predicts standoff position and FR leg waypoints from 3D target position.
- **Stage D — Reactive Closed-Loop Controller:** Residual policy that corrects FR leg joint commands per-step during contact based on live foot-to-target error.

Both stages are trained on **real-robot data only** — no synthetic pretraining. This decision was made because:

1. The existing simulation environment (`sim/press_env.py`) is configured for ground-press with 4 fixed buttons in a 2×2 grid. The real robot task is **wall press** with a single button. The geometries are incompatible.
2. The simulation heuristic uses kinematic base teleportation, not true locomotion. Standoff "learning" on this data would only recover a hardcoded offset — no useful signal.
3. Standoff prediction is a geometric regression problem over a small input space. It does not need large-scale pretraining.

The 500 synthetic episodes collected in Summary 2 remain on disk as historical artifacts but are not used for training.

---

## 2. Design Decisions (Locked)

| Decision | Choice |
|---|---|
| Stage C output dimensions | 12 — 3 standoff + 9 FR leg waypoints |
| Stage C input dimensions | 5 — target_pos (3) + interaction one-hot (2) |
| Stage C data source | Real robot only |
| Stage D output dimensions | 3 — residual delta from Stage C waypoint |
| Stage D input dimensions | 15 — see Section 5 |
| Stage D control frequency | 500 Hz |
| Stage D training approach | Imitation from Jacobian-PID expert with deliberate perturbations |
| Collection-time expert | `HeuristicContactCorrective` — subclass adding Jacobian-PID correction to lift/extend/hold phases |
| Grounding during contact | Re-ground every frame in background thread (~5 Hz) |
| Foot forces | Not recorded (unreliable on Go2X hardware) |
| Perturbation method | Algorithmic via `move_to_position` + measure actual via re-grounding |
| Stage D recording architecture | External observer thread reading `heuristic._low_state` under `_state_lock` |
| File format | HDF5 per-episode |
| RGB storage | Subset only, for verification — not for training |
| Total target episode count | 50 (to start; revisit after initial collection session) |
| Audio contact verification | USB microphone (laptop-connected), placed within 50cm of button |
| Live audio detection | Energy-threshold RMS, ~50-100 Hz polling, single-bool output |
| Post-hoc color identification | `faster-whisper` tiny model on GPU, run after `execute()` returns |

---

## 3. Critical Architectural Addition — Jacobian-PID Expert

### 3.1 Why This Is Needed

The baseline heuristic (`heuristic_contact.py`) commands a **fixed** FR joint trajectory regardless of robot position relative to the button. Under perturbed standoffs, it commands the same joints and misses the button. This means the delta between commanded joints and the nominal waypoint is always zero — there is no training signal for Stage D.

To generate meaningful training data under perturbation, the collection expert must actively correct the FR leg trajectory based on live foot-to-target error.

### 3.2 What the Expert Does

At each 500 Hz control step during lift/extend/hold phases:

```
1. Read latest target_pos_base from grounding thread
2. Read current FR joints from robot state
3. Compute foot_pos = _estimate_fr_foot(fr_joints)   (existing FK)
4. error = target_pos_base - foot_pos                (3-vector, base frame)
5. J = fr_jacobian(fr_joints)                        (3x3 analytic)
6. delta = pinv(J) @ (K_p * error)                   (3-vector, joint space)
7. corrected_fr = heuristic_nominal_fr + delta
8. Clip to joint limits; send via _send_cmd
```

Retract phases (retract_curl, retract_rotate, retract_extend) are **not** corrected — they use pure heuristic logic to avoid complications during leg withdrawal.

### 3.3 Dual Role of the Expert

| Role | Usage |
|---|---|
| **Data collection** | Jacobian-PID generates successful demonstrations under perturbation. Per-step `delta` provides training labels for Stage D. |
| **Evaluation baseline** | Jacobian-PID is reported as its own method variant alongside Baselines 1-3 and Stage D. |

This gives an honest research comparison: Stage D is evaluated against a strong classical controller, not just against open-loop baselines.

### 3.4 Updated Method Variants Table

| Variant | Approach | Contact Phase | Re-grounding |
|---|---|---|---|
| Baseline 1 | Heuristic dead-reckoning | Scripted fixed sequence | None |
| Baseline 2 | Learned Stage C | Scripted fixed sequence | None |
| Baseline 3 | Learned Stage C | Scripted fixed sequence | Approach only |
| **Baseline 4** | **Learned Stage C** | **Jacobian-PID** | **Approach + Contact** |
| Core Method | Learned Stage C | Learned Stage D | Approach + Contact |

The research question becomes: *when does learned Stage D outperform Jacobian-PID?* Likely answers: state-dependent effects (body lean via IMU), learned smoothness, FK error compensation, transfer to push task.

---

## 4. Stage C Data Specification

### 4.1 Input (5 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `target_pos_base` | (3,) | float32 | Button position in robot base frame, measured via grounding at moment of contact phase start |
| `interaction_onehot` | (2,) | float32 | `[1, 0]` for press, `[0, 1]` for push. v1 trains press-only but the slot is reserved |

All positions are in **robot base frame at the moment of standoff** (right before low-level contact control begins). Convention:
- +x = forward of robot
- +y = left of robot
- +z = above base link

### 4.2 Output — Approach Head (3 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `standoff_offset` | (3,) | float32 | `(dx, dy, dyaw)` — ideal base pose relative to button |

`(dx, dy)` in meters, `dyaw` in radians.

### 4.3 Output — Contact Head (9 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `lift_waypoint_fr` | (3,) | float32 | FR `[hip, thigh, calf]` joint angles at end of lift phase |
| `extend_waypoint_fr` | (3,) | float32 | FR `[hip, thigh, calf]` joint angles at end of extend phase |
| `press_waypoint_fr` | (3,) | float32 | FR `[hip, thigh, calf]` joint angles at end of hold phase (just before retract) |

All angles in radians. Joint ordering follows the real robot convention from `heuristic_contact.py`:
- Index 0: FR_hip
- Index 1: FR_thigh
- Index 2: FR_calf

### 4.4 Label Source

Labels come from **achieved joint positions** (read from `rt/lowstate`), not commanded positions. This is important — under `KP_FR=40` the FR leg rarely reaches commanded targets exactly, and what you want to learn is "what pose actually made contact," not "what pose did we command."

Contact head labels come from the `phase_transitions` dict added to `ContactResult` (see Section 7).

Approach head labels come from the robot's actual pose at standoff. Specifically: the `target_pos_base_at_standoff` field captures where the button was in base frame at standoff, and the standoff offset is the negation of that in x/y (plus any yaw).

**Training-time filtering:** For Stage C training, filter episodes by `success_target` (the audio-verified success metric — see Section 5.7). Episodes that "succeeded" only by FK-proximity but didn't physically press the button provide unreliable labels.

### 4.5 Expected Volume

| Purpose | Episodes | Notes |
|---|---|---|
| Approach head — varied button placement | 15 | Heights × distances × angles |
| Contact head | Same episodes reused | Every successful contact gives one datapoint |

Both heads share the same episode pool. 15 episodes with varied button placement gives 15 approach-head points AND 15 contact-head points.

---

## 5. Stage D Data Specification

### 5.1 Input (15 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `fr_joint_pos` | (3,) | float32 | Actual FR joint positions `[hip, thigh, calf]` |
| `fr_joint_vel` | (3,) | float32 | Actual FR joint velocities |
| `foot_to_target_error` | (3,) | float32 | `target_pos_base - FK(fr_joint_pos)` in base frame |
| `nominal_waypoint_fr` | (3,) | float32 | Current heuristic FR command (pre-correction) |
| `phase_progress` | (1,) | float32 | `[0, 1]` fractional progress within current phase |
| `imu_rp` | (2,) | float32 | IMU roll, pitch (yaw omitted — not relevant for contact correction) |

### 5.2 Output (3 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `delta_fr` | (3,) | float32 | Residual correction `[Δhip, Δthigh, Δcalf]` added to `nominal_waypoint_fr` |

Final command sent to PD controller:
```
commanded_fr = nominal_waypoint_fr + delta_fr
```

### 5.3 Training Label Derivation

For each 500 Hz step during lift/extend/hold:

```
delta_label[t] = jacobian_pid_delta[t]
```

Where `jacobian_pid_delta[t]` is the correction computed by the Jacobian-PID expert at that step. This is recorded directly from the `HeuristicContactCorrective` run and stored as a per-step field.

Under no perturbation, `delta_label[t] ≈ 0` because `error ≈ 0`. Under perturbation, `delta_label[t]` has meaningful magnitude.

The policy learns: *given state + error, produce the same delta the Jacobian-PID expert did.*

**Training-time filtering:** For Stage D training, filter episodes by `success_target` (audio-verified success). Stage D should learn only from episodes where the expert's corrections produced a physically successful press.

### 5.4 Recording Window

Recording is active only during phases where Stage D will eventually operate:

```
recording_active = heuristic._phase in {"lift", "extend", "hold"}
```

Lift/extend/hold are the manipulation phases where correction matters. Retract phases use pure heuristic logic and are not recorded. Approximately 3000-4000 steps per episode at 500 Hz.

### 5.5 Per-Step Recording Schema

Stored as HDF5 dataset `per_step`:

| Field | Shape (T,) | Dtype | Description |
|---|---|---|---|
| `timestamp` | (T,) | float64 | Seconds from episode start |
| `joint_pos_actual` | (T, 12) | float32 | All 12 joints from `rt/lowstate` |
| `joint_vel_actual` | (T, 12) | float32 | All 12 joint velocities |
| `joint_pos_cmd` | (T, 12) | float32 | Final commanded joint positions sent to PD (post-correction) |
| `fr_foot_pos_fk` | (T, 3) | float32 | FR foot position via FK, base frame |
| `target_pos_base` | (T, 3) | float32 | Latest from grounding thread (cached between updates) |
| `foot_to_target_error` | (T, 3) | float32 | `target_pos_base - fr_foot_pos_fk` |
| `imu_rpy` | (T, 3) | float32 | Roll, pitch, yaw |
| `phase_label` | (T,) | int8 | 0=lift, 1=extend, 2=hold |
| `phase_progress` | (T,) | float32 | `[0, 1]` within current phase |
| `nominal_waypoint_fr` | (T, 3) | float32 | Heuristic FR command before Jacobian-PID correction |
| `jacobian_pid_delta` | (T, 3) | float32 | Correction added by Jacobian-PID expert — this is the Stage D label |

T ≈ 3000-4000 per episode. Audio is **not** included in per-step — it lives in a separate root-level dataset (see 5.8).

### 5.6 Grounding Thread Output Schema

Stored as HDF5 dataset `grounding`:

| Field | Shape (G,) | Dtype | Description |
|---|---|---|---|
| `timestamp` | (G,) | float64 | Seconds from episode start |
| `position_base` | (G, 3) | float32 | Button position from grounding (NaN if None) |
| `confidence` | (G,) | float32 | GroundingDINO confidence |
| `depth_m` | (G,) | float32 | Corrected depth at centroid |
| `position_base_valid` | (G,) | bool | False when grounding returned None |

G ≈ 75 per episode at 5 Hz over 15 seconds.

### 5.7 Episode Metadata

Stored as HDF5 group attributes on the root:

| Field | Type | Description |
|---|---|---|
| `episode_id` | str | `YYYYMMDD_HHMMSS` format |
| `target_pos_base_at_standoff` | (3,) float32 | Button position at t=0 of contact phase |
| `target_pos_world_initial` | (3,) float32 | Button position in world frame (world = robot base frame at first grounding) |
| `perturbation_commanded` | (2,) float32 | `(dx, dy)` commanded perturbation to ideal standoff |
| `k_p` | float32 | Jacobian-PID gain used for this episode |
| `success_fk` | bool | FK-proximity based: foot FK estimate within `contact_proximity_m` of target |
| `success_audio_live` | bool | Live energy detector fired during hold phase |
| `audio_detection_time_s` | float | Seconds from episode start when audio was detected; NaN if not detected |
| `contact_method` | str | Which signal fired first: `"fk"`, `"audio"`, `"foot_force"`, or `"none"` |
| `color_detected` | str | Post-hoc Whisper transcription result: `"red"`, `"blue"`, etc., or `""` |
| `color_expected` | str | Expected color from task config (for wall-press v1, always `"red"`) |
| `success_target` | bool | `color_detected == color_expected` — **primary evaluation metric** |
| `audio_transcript` | str | Full Whisper transcript (for debugging and failure analysis) |
| `audio_sample_rate` | int | Hz; matches the `audio` dataset below |
| `contact_step` | int | Step at first contact (-1 if none) |
| `duration_s` | float | Total contact phase duration |
| `failure_mode` | str | Empty if success; else "missed_button", "fell", "timeout" |
| `phase_transitions` | group | Nested group, keys = phase names, values = 12-dim joint vectors |
| `button_height_m` | float | Measured button height above floor (ground truth) |
| `notes` | str | Free-form human notes |

**Three-tier success semantics:**

- `success_fk` — controller's own belief: FK said the foot got close enough. Approximate.
- `success_audio_live` — physical press confirmation: audio energy spiked during hold. The button's sound module fired, so the button was actually depressed.
- `success_target` — correctness verification: the detected color word matches the expected color. Required for multi-button tasks; `success_target == success_audio_live` for single-button v1.

`success_target` is the primary metric reported in evaluation. `success_fk` and `success_audio_live` are secondary diagnostics useful for understanding failure modes.

### 5.8 Audio Dataset

Stored as root-level HDF5 dataset:

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `audio` | `(N,)` | float32 | Raw mono audio waveform from USB mic, full episode duration |

Where `N = duration_s * audio_sample_rate`. At 16 kHz and ~15s episodes, that's ~240k samples = ~960KB uncompressed. With gzip compression on float audio (not highly compressible), expect ~700-800KB per episode. Total audio storage for 50 episodes: ~40MB.

The audio is used only during post-hoc analysis (re-running `ColorDetector` if needed) and for debugging — it does not feed into the Stage C or Stage D networks.

---

## 6. Directory Structure

```
~/Robotics/Language_Conditioned_UniGo2/
├── data/
│   ├── real/
│   │   └── stage_d/
│   │       ├── tuning/                       # K_p tuning runs
│   │       │   ├── episode_kp030_20260422_140000.h5
│   │       │   └── ...
│   │       ├── episode_20260422_143000.h5    # main collection
│   │       ├── episode_20260422_143300.h5
│   │       └── rgb_subset/
│   │           └── episode_20260422_143000/
│   │               ├── 0000_t0.00.jpg
│   │               └── ...
│   └── synthetic/                            # legacy, not used
└── docs/
    └── training_data_spec.md                 # this file
```

A "Stage D episode with no perturbation" naturally doubles as a Stage C episode — the same HDF5 file can serve both pipelines. Training scripts for Stage C extract just the metadata fields they need.

Audio is embedded in each episode's HDF5 file (root-level `audio` dataset). No separate audio subdirectory needed.

---

## 7. Required Code Changes

### 7.1 `src/planner/heuristic_contact.py` — v2.0 and v2.1 additions

Three sets of additive changes. All are already applied to the file delivered with the v2.1 spec — do not re-edit.

**Change 1 (v2.0): `phase_transitions` dict**

In `HeuristicContact.__init__`, alongside the existing internal state storage:

```python
self._phase_transitions = {}  # Dict[str, np.ndarray] — 12-dim actual joint vectors
```

In `_control_loop`, at each gate-passed phase advance (immediately before `self._phase = "next"`), capture the actual joint state. All 9 transition points are captured:

- `"sit_to_stand_end"` — at sit_to_stand → weight_shift transition
- `"weight_shift_end"` — at weight_shift → lift transition
- `"lift_end"` — at lift → extend transition
- `"extend_end"` — at extend → hold transition
- `"hold_end"` — at hold → retract_curl transition
- `"retract_curl_end"` — at retract_curl → retract_rotate transition
- `"retract_rotate_end"` — at retract_rotate → retract_extend transition
- `"retract_extend_end"` — at retract_extend → weight_unshift transition
- `"settle_end"` — at settle → lower_to_sit transition

The 3 mandatory keys for Stage C training are `lift_end`, `extend_end`, `hold_end`.

**Change 2 (v2.0): `current_waypoint_fr` attribute**

In `HeuristicContact.__init__`:
```python
self.current_waypoint_fr = np.zeros(3, dtype=np.float32)
```

In `_control_loop`, immediately before `self._send_cmd(target_q)`:
```python
self.current_waypoint_fr = np.array(target_q[0:3], dtype=np.float32)
```

This captures the **current commanded FR pose** at every 500 Hz step, including during interpolation within a phase. The recorder thread reads this to log `nominal_waypoint_fr` in per-step data.

**Change 3 (v2.1): `audio_detector` parameter and `contact_method` tracking**

`HeuristicContact.execute()` now accepts an optional `audio_detector`:

```python
def execute(
    self,
    target_offset_xyz:   np.ndarray,
    press_mode:          Literal["wall", "ground"] = "ground",
    contact_proximity_m: float = CONTACT_PROXIMITY_M,
    use_foot_force:      bool  = False,
    audio_detector:      Optional[object] = None,
) -> ContactResult:
```

**Contract for `audio_detector`:**
- Duck-typed: only `.sound_detected: bool` attribute is read. Any object exposing that attribute works.
- Caller is responsible for lifecycle: call `audio_detector.start_episode()` (or equivalent reset) before `execute()`. The heuristic does not manage the detector thread.
- `None` (default) preserves v2.0 behavior exactly.

**Behavior during hold phase:** An audio `sound_detected=True` now triggers `self._contact_step` assignment the same way FK proximity does. Whichever signal fires first wins. Priority when multiple fire on the same step: `audio > fk > foot_force` (audio is prioritized because it's physical proof of button depression).

**Updated `ContactResult`:**

```python
@dataclass
class ContactResult:
    success:            bool
    contact_step:       int
    timeout:            bool
    press_mode:         str
    target_offset_xyz:  np.ndarray
    joint_trajectory:   np.ndarray
    fr_foot_trajectory: np.ndarray
    duration_s:         float
    phase_transitions:  dict
    contact_method:     str   # NEW (v2.1): "fk", "audio", "foot_force", "none"
```

Values of `contact_method`:
- `"fk"` — FK-proximity threshold fired first (geometric estimate)
- `"audio"` — audio energy spike fired first (physical press)
- `"foot_force"` — foot-force spike fired first (disabled by default; currently unreliable)
- `"none"` — hold phase timed out at `CONTACT_MAX_STEPS` with no contact signal

### 7.2 `src/planner/heuristic_contact_corrective.py` — NEW Subclass

The Jacobian-PID expert. Overrides `_control_loop` to apply the correction after computing the heuristic's nominal command but before sending. **No v2.1 changes required** — the subclass inherits the parent's `audio_detector` parameter automatically via `execute()`.

```python
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
```

**Notes:**
- Sending two commands per 500 Hz cycle (parent + override) is acceptable. DDS motor PD uses the most recent command.
- `grounding_getter` is injected as a callable so the subclass does not need to know about grounding thread internals.
- `last_delta_fr` is exposed for `StageDRecorder` to log as `jacobian_pid_delta`.
- **To pass audio_detector:** the collection script passes it via `execute()`, not via subclass `__init__`. Example in Section 10.

---

## 8. K_p Tuning Procedure

### 8.1 Why Tune

K_p is the proportional gain converting meter-error to joint-space delta. Too low → foot doesn't reach button within phase duration. Too high → oscillation, support leg destabilization, potential hardware damage.

Empirical tuning is required because the optimal K_p depends on FK accuracy, PD gains (KP_FR=40), phase durations, and perturbation magnitudes.

### 8.2 Tuning Session Protocol

**Duration:** ~45-60 minutes before main collection session.

**Setup:**
1. Wall button mounted at 0.25 m height, 0.75 m from robot start mark.
2. Robot at ideal standoff (`PRESS_OFFSET_X = 0.203 m, PRESS_OFFSET_Y = 0.140 m`).
3. Grounding thread running, publishing `target_pos_base`.
4. Audio recorder and detector running with calibrated baseline.
5. Data collection script ready but not running.

**Test procedure:**
For each K_p ∈ {0.3, 0.5, 1.0}, run 2 episodes:

1. **Episode A (no perturbation):**
   - Robot at ideal standoff
   - Run `HeuristicContactCorrective(k_p=K)`
   - Expected: correction delta near zero throughout, successful press, audio fires

2. **Episode B (+3cm perturbation):**
   - Before starting: `go2.move_to_position(ideal_x + 0.03, ideal_y, 0)`
   - Wait for settle (2s)
   - Run `HeuristicContactCorrective(k_p=K)`
   - Expected: correction delta grows during lift, peaks during extend, decays during hold as foot closes on button. Audio fires when button depresses.

3. **Review** the correction log after each run:
   - Plot `|error|` vs time and `|delta|` vs time
   - Check for smoothness, monotonic error decrease, stability
   - Confirm `success_audio_live == True` in both Episodes A and B

**Tuning outputs per K_p:**

| K_p | A: fk success | A: audio success | B: fk success | B: audio success | B final error (cm) | Oscillation? | Support leg disturbance? |
|---|---|---|---|---|---|---|---|
| 0.3 | ? | ? | ? | ? | ? | no | no |
| 0.5 | ? | ? | ? | ? | ? | no | no |
| 1.0 | ? | ? | ? | ? | ? | ? | ? |

### 8.3 K_p Selection Criteria

Pick the K_p that satisfies all of:

1. Episode A succeeds (both `success_fk` and `success_audio_live`) without spurious motion (correction near zero)
2. Episode B succeeds (both `success_fk` and `success_audio_live`) — foot reaches AND depresses the button
3. No visible oscillation in commanded FR joint trajectory
4. No visible support leg disturbance during lift/extend
5. Final `|error|` during hold < 2 cm
6. `success_audio_live == True` in both Episode A and Episode B (audio confirms physical press, not just FK proximity)

**If `success_fk == True` but `success_audio_live == False`:** the foot got close enough geometrically but didn't depress the button. Either increase K_p, or increase `FR_EXTEND_OFFSET_WALL[2]` (calf component) to press deeper. This is a symptom worth fixing before full collection.

Recommended starting point: **K_p = 0.5**. Increase to 1.0 only if 0.5 cannot reach the button under +3cm perturbation. Decrease to 0.3 if 0.5 shows oscillation.

### 8.4 Safety Rules

- If the robot shows violent motion or instability at any K_p, immediately abort via Ctrl+C and reduce K_p by half for the next run.
- Never tune at K_p > 1.0 without first checking behavior at 0.5 and 1.0.
- The incident that damaged the Ethernet port occurred at `KP_FR_LIFT=80`. Raising the **PD gain** is different from raising K_p, but the principle is the same: changes to leg-response parameters on extended legs are dangerous.
- Do tuning on the floor (not on a table) so any fall does not cause additional damage.

### 8.5 Logging Tuning Runs

Save tuning episodes to `data/real/stage_d/tuning/` with filenames encoding K_p:

```
episode_kp030_20260422_140000.h5
episode_kp050_20260422_140300.h5
episode_kp100_20260422_140600.h5
```

These are not used for training but should be kept for reproducibility and for a tuning-analysis plot in the final report.

---

## 9. New Files to Create

### 9.1 `src/planner/heuristic_contact_corrective.py`

The Jacobian-PID subclass. Full implementation shown in Section 7.2.

### 9.2 `src/data/grounding_thread.py`

Background thread running `VisualGrounder.ground()` at target frequency.

```python
"""
src/data/grounding_thread.py

Background thread that runs visual grounding at a configurable rate and
exposes the latest result to the 500 Hz control and recorder threads
via a thread-safe shared variable.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundingSample:
    timestamp:  float
    position_base: Optional[np.ndarray]
    confidence: float
    depth_m:    float
    valid:      bool


class GroundingThread:
    """
    Runs grounder.ground() in a background thread at rate_hz.

    Usage:
        gt = GroundingThread(camera, grounder, prompt="red button", rate_hz=5)
        gt.start()
        ...
        latest = gt.get_latest_position()  # thread-safe
        ...
        gt.stop()
        log = gt.get_log()
    """

    def __init__(self, camera, grounder, prompt: str = "red button",
                 rate_hz: float = 5.0):
        self.camera = camera
        self.grounder = grounder
        self.prompt = prompt
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz

        self._lock = threading.Lock()
        self._latest_position: Optional[np.ndarray] = None
        self._latest_sample: Optional[GroundingSample] = None
        self._log: list = []
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: float = 0.0

    def start(self):
        self._t_start = time.monotonic()
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop, name="grounding", daemon=True)
        self._thread.start()
        logger.info(f"GroundingThread started at {self.rate_hz} Hz")

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info(f"GroundingThread stopped. {len(self._log)} samples logged.")

    def _loop(self):
        while not self._stop_flag.is_set():
            t0 = time.monotonic()
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(self.period_s)
                    continue
                result = self.grounder.ground(frame, prompt=self.prompt)
            except Exception as e:
                logger.warning(f"Grounding call failed: {e}")
                time.sleep(self.period_s)
                continue

            if result is None:
                sample = GroundingSample(
                    timestamp=time.monotonic() - self._t_start,
                    position_base=None,
                    confidence=0.0,
                    depth_m=0.0,
                    valid=False,
                )
            else:
                pos = result.position_base
                sample = GroundingSample(
                    timestamp=time.monotonic() - self._t_start,
                    position_base=pos.copy() if pos is not None else None,
                    confidence=float(result.confidence),
                    depth_m=float(result.depth_m) if result.depth_m is not None else 0.0,
                    valid=pos is not None,
                )

            with self._lock:
                self._latest_sample = sample
                if sample.valid:
                    self._latest_position = sample.position_base
                self._log.append(sample)

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.period_s - elapsed)
            time.sleep(sleep_t)

    def get_latest_position(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_position is None else self._latest_position.copy()

    def get_log(self) -> list:
        with self._lock:
            return list(self._log)
```

### 9.3 `src/data/recorder.py`

500 Hz observer thread that reads heuristic state and writes HDF5.

```python
"""
src/data/recorder.py

External observer thread that records per-step Stage D training data at
500 Hz during the contact phase. Reads state from the HeuristicContactCorrective
instance (under _state_lock) and from the grounding thread.
"""

import logging
import threading
import time
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


PHASE_TO_LABEL = {
    "lift":   0,
    "extend": 1,
    "hold":   2,
}
PHASE_DURATION_S = {
    "lift":   1.2,
    "extend": 1.2,
    "hold":   3.0,
}
ACTIVE_PHASES = ("lift", "extend", "hold")


class StageDRecorder:
    """
    Observer thread polling at 500 Hz. Appends per-step fields to in-memory
    arrays during active phases. Writes to HDF5 at save().
    """

    def __init__(self, heuristic, grounding_thread, ctrl_dt: float = 0.002):
        self.heuristic = heuristic
        self.grounding_thread = grounding_thread
        self.ctrl_dt = ctrl_dt

        self._buffer_lock = threading.Lock()
        self._recording = False
        self._t_start: float = 0.0
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._reset_buffers()
        self._metadata: dict = {}

    def _reset_buffers(self):
        self.timestamp: list = []
        self.joint_pos_actual: list = []
        self.joint_vel_actual: list = []
        self.joint_pos_cmd: list = []
        self.fr_foot_pos_fk: list = []
        self.target_pos_base: list = []
        self.foot_to_target_error: list = []
        self.imu_rpy: list = []
        self.phase_label: list = []
        self.phase_progress: list = []
        self.nominal_waypoint_fr: list = []
        self.jacobian_pid_delta: list = []

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop, name="stage_d_recorder", daemon=True)
        self._thread.start()
        logger.info("StageDRecorder observer thread started")

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def begin_episode(self, metadata: dict):
        with self._buffer_lock:
            self._reset_buffers()
            self._metadata = dict(metadata)
            self._t_start = time.monotonic()
            self._recording = True

    def end_episode(self, result):
        with self._buffer_lock:
            self._recording = False
            self._metadata.update({
                "contact_step": int(result.contact_step),
                "duration_s": float(result.duration_s),
                "press_mode": result.press_mode,
                "target_offset_xyz": np.asarray(result.target_offset_xyz, dtype=np.float32),
                "contact_method": str(result.contact_method),
                "phase_transitions": {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in result.phase_transitions.items()
                },
            })

    def save(self, hdf5_path: str,
             audio: Optional[np.ndarray] = None,
             audio_sample_rate: int = 16000,
             metadata_override: Optional[dict] = None):
        """
        Args:
            hdf5_path: output file path
            audio: raw audio waveform from AudioRecorder (optional)
            audio_sample_rate: Hz for the audio dataset (required if audio is given)
            metadata_override: extra metadata fields to merge (e.g., audio-derived
                success fields from the collection script — success_fk,
                success_audio_live, audio_detection_time_s, color_detected, etc.)
        """
        with self._buffer_lock:
            if len(self.timestamp) == 0:
                logger.warning(f"No samples recorded; skipping {hdf5_path}")
                return

            # Merge extra metadata from caller (audio success fields etc)
            final_metadata = dict(self._metadata)
            if metadata_override:
                final_metadata.update(metadata_override)

            with h5py.File(hdf5_path, "w") as f:
                grp = f.create_group("per_step")
                for name, data, dtype in [
                    ("timestamp", self.timestamp, np.float64),
                    ("joint_pos_actual", self.joint_pos_actual, np.float32),
                    ("joint_vel_actual", self.joint_vel_actual, np.float32),
                    ("joint_pos_cmd", self.joint_pos_cmd, np.float32),
                    ("fr_foot_pos_fk", self.fr_foot_pos_fk, np.float32),
                    ("target_pos_base", self.target_pos_base, np.float32),
                    ("foot_to_target_error", self.foot_to_target_error, np.float32),
                    ("imu_rpy", self.imu_rpy, np.float32),
                    ("phase_label", self.phase_label, np.int8),
                    ("phase_progress", self.phase_progress, np.float32),
                    ("nominal_waypoint_fr", self.nominal_waypoint_fr, np.float32),
                    ("jacobian_pid_delta", self.jacobian_pid_delta, np.float32),
                ]:
                    grp.create_dataset(name, data=np.array(data, dtype=dtype),
                                       compression="gzip", compression_opts=4)

                g_log = self.grounding_thread.get_log()
                gg = f.create_group("grounding")
                if g_log:
                    gg.create_dataset("timestamp",
                        data=np.array([s.timestamp for s in g_log], dtype=np.float64))
                    gg.create_dataset("position_base",
                        data=np.array([
                            s.position_base if s.position_base is not None
                            else np.full(3, np.nan, dtype=np.float32)
                            for s in g_log
                        ], dtype=np.float32))
                    gg.create_dataset("confidence",
                        data=np.array([s.confidence for s in g_log], dtype=np.float32))
                    gg.create_dataset("depth_m",
                        data=np.array([s.depth_m for s in g_log], dtype=np.float32))
                    gg.create_dataset("position_base_valid",
                        data=np.array([s.valid for s in g_log], dtype=bool))

                # NEW (v2.1): raw audio waveform
                if audio is not None and len(audio) > 0:
                    f.create_dataset("audio", data=audio.astype(np.float32),
                                     compression="gzip", compression_opts=4)
                    f.attrs["audio_sample_rate"] = audio_sample_rate

                for k, v in final_metadata.items():
                    if k == "phase_transitions":
                        pt_grp = f.create_group("phase_transitions")
                        for pname, pval in v.items():
                            pt_grp.create_dataset(pname, data=pval)
                    elif isinstance(v, (np.ndarray, list, tuple)):
                        f.attrs[k] = np.asarray(v)
                    else:
                        f.attrs[k] = v

            logger.info(f"Wrote episode HDF5: {hdf5_path}  "
                        f"({len(self.timestamp)} steps, {len(g_log)} groundings, "
                        f"audio={'yes' if audio is not None else 'no'})")

    def _loop(self):
        while not self._stop_flag.is_set():
            t0 = time.monotonic()

            if self._recording:
                phase = self.heuristic._phase
                if phase in ACTIVE_PHASES:
                    self._sample_one_step(phase)

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.ctrl_dt - elapsed)
            time.sleep(sleep_t)

    def _sample_one_step(self, phase: str):
        with self.heuristic._state_lock:
            if self.heuristic._low_state is None:
                return
            q  = [self.heuristic._low_state.motor_state[i].q  for i in range(12)]
            dq = [self.heuristic._low_state.motor_state[i].dq for i in range(12)]
            rpy = list(self.heuristic._low_state.imu_state.rpy[0:3])

        cmd = list(self.heuristic._last_target_q) if self.heuristic._last_target_q else [0.0]*12
        waypoint = self.heuristic.current_waypoint_fr.copy()
        delta    = self.heuristic.last_delta_fr.copy() \
            if hasattr(self.heuristic, "last_delta_fr") \
            else np.zeros(3, dtype=np.float32)

        foot = self.heuristic._estimate_fr_foot(q)

        target = self.grounding_thread.get_latest_position()
        if target is None:
            target_arr = np.full(3, np.nan, dtype=np.float32)
            error_arr  = np.full(3, np.nan, dtype=np.float32)
        else:
            target_arr = target.astype(np.float32)
            error_arr  = (target_arr - foot).astype(np.float32)

        step = self.heuristic._phase_step
        phase_dur_steps = PHASE_DURATION_S[phase] / self.ctrl_dt
        progress = min(1.0, step / phase_dur_steps)

        with self._buffer_lock:
            self.timestamp.append(time.monotonic() - self._t_start)
            self.joint_pos_actual.append(np.array(q,   dtype=np.float32))
            self.joint_vel_actual.append(np.array(dq,  dtype=np.float32))
            self.joint_pos_cmd.append(np.array(cmd, dtype=np.float32))
            self.fr_foot_pos_fk.append(foot)
            self.target_pos_base.append(target_arr)
            self.foot_to_target_error.append(error_arr)
            self.imu_rpy.append(np.array(rpy, dtype=np.float32))
            self.phase_label.append(PHASE_TO_LABEL[phase])
            self.phase_progress.append(np.float32(progress))
            self.nominal_waypoint_fr.append(waypoint.astype(np.float32))
            self.jacobian_pid_delta.append(delta.astype(np.float32))
```

### 9.4 `scripts/collect_stage_d.py`

Main collection orchestration script. Supports three modes: `--tune-kp`, `--mode stage-c`, `--mode stage-d`. See Section 10 for detailed CLI usage and per-episode procedure.

### 9.5 `scripts/validate_episode.py`

Single-episode HDF5 sanity check. Implements the checklist from Section 11.1.

### 9.6 `scripts/validate_dataset.py`

Dataset-level checks across all episodes. Implements the checklist from Section 11.2.

### 9.7 `scripts/measure_standoff.py`

Interactive utility to help position the robot at a known standoff during tuning and Stage C collection. Prints live `target_pos_base` from grounding so the user can manually position the robot at a measured offset.

### 9.8 `src/data/audio_recorder.py` (NEW v2.1)

Background thread that records mono 16 kHz audio from the default USB mic for the full episode duration.

```python
"""
src/data/audio_recorder.py

Records full-episode audio from USB microphone via sounddevice.
Stored in-memory as a numpy buffer; returned via end_episode() at end.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Full-episode audio recorder. Mono, configurable sample rate.

    Usage:
        rec = AudioRecorder(sample_rate=16000, device_index=None)
        rec.start_stream()          # opens mic, starts background capture
        ...
        rec.start_episode()         # resets buffer
        result = heuristic.execute(...)
        audio = rec.end_episode()   # returns np.ndarray (N,) float32
        ...
        rec.stop_stream()
    """

    def __init__(self, sample_rate: int = 16000,
                 device_index: Optional[int] = None,
                 block_duration_ms: int = 50):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.block_size = int(sample_rate * block_duration_ms / 1000)

        self._stream: Optional[sd.InputStream] = None
        self._buffer_lock = threading.Lock()
        self._recording = False
        self._buffer: list = []
        self._t_start: float = 0.0

    def start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"AudioRecorder stream opened "
                    f"({self.sample_rate} Hz, device={self.device_index})")

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("AudioRecorder stream closed")

    def start_episode(self):
        with self._buffer_lock:
            self._buffer = []
            self._recording = True
            self._t_start = time.monotonic()

    def end_episode(self) -> np.ndarray:
        with self._buffer_lock:
            self._recording = False
            if not self._buffer:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._buffer).astype(np.float32)

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"sounddevice status: {status}")
        if not self._recording:
            return
        with self._buffer_lock:
            self._buffer.append(indata[:, 0].copy())
```

### 9.9 `src/data/audio_live_detector.py` (NEW v2.1)

Background detection thread that monitors mic energy in real time. Exposes `sound_detected: bool` for the heuristic's control loop.

```python
"""
src/data/audio_live_detector.py

Real-time energy-threshold audio detector. When RMS energy exceeds
baseline + threshold_db for at least min_duration_ms, sets
sound_detected = True. Single-writer, multi-reader; no lock needed for
reads of the bool (atomic in CPython).
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioLiveDetector:
    """
    Args:
        sample_rate: Hz
        device_index: mic device index (None = default)
        threshold_db: how much above baseline RMS to trigger detection
        min_duration_ms: minimum sustained high-energy duration (spike reject)
        baseline_duration_s: silence window to calibrate baseline at start
    """

    def __init__(self, sample_rate: int = 16000,
                 device_index: Optional[int] = None,
                 threshold_db: float = 20.0,
                 min_duration_ms: int = 50,
                 baseline_duration_s: float = 2.0,
                 block_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.threshold_db = threshold_db
        self.min_duration_ms = min_duration_ms
        self.baseline_duration_s = baseline_duration_s
        self.block_size = int(sample_rate * block_duration_ms / 1000)
        self.block_duration_ms = block_duration_ms

        # Public state
        self.sound_detected: bool = False
        self.detection_time_s: Optional[float] = None

        self._stream: Optional[sd.InputStream] = None
        self._baseline_rms: Optional[float] = None
        self._baseline_samples: list = []
        self._baseline_done = False
        self._high_energy_ms: float = 0.0
        self._episode_t_start: float = 0.0
        self._calibrating = False

    def start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("AudioLiveDetector stream opened")

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def calibrate_baseline(self, duration_s: Optional[float] = None):
        """Block for duration_s of silence to measure ambient noise floor."""
        duration = duration_s or self.baseline_duration_s
        logger.info(f"Calibrating baseline for {duration:.1f}s — keep quiet...")
        self._baseline_samples = []
        self._baseline_done = False
        self._calibrating = True
        time.sleep(duration)
        self._calibrating = False

        if not self._baseline_samples:
            logger.warning("No samples captured during calibration")
            self._baseline_rms = 1e-4
        else:
            all_samples = np.concatenate(self._baseline_samples)
            self._baseline_rms = float(np.sqrt(np.mean(all_samples ** 2)))
            self._baseline_rms = max(self._baseline_rms, 1e-6)
        self._baseline_done = True
        logger.info(f"Baseline RMS={self._baseline_rms:.6f}  "
                    f"threshold_above_baseline={self.threshold_db}dB")

    def start_episode(self):
        """Reset detection state at the start of an episode."""
        self.sound_detected = False
        self.detection_time_s = None
        self._high_energy_ms = 0.0
        self._episode_t_start = time.monotonic()

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"sounddevice status: {status}")

        samples = indata[:, 0]

        if self._calibrating:
            self._baseline_samples.append(samples.copy())
            return

        if not self._baseline_done or self._baseline_rms is None:
            return

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1e-8:
            db_above = -120.0
        else:
            db_above = 20.0 * np.log10(rms / self._baseline_rms)

        if db_above > self.threshold_db:
            self._high_energy_ms += self.block_duration_ms
            if (self._high_energy_ms >= self.min_duration_ms
                    and not self.sound_detected):
                self.sound_detected = True
                self.detection_time_s = time.monotonic() - self._episode_t_start
                logger.info(f"Sound detected: rms={rms:.6f} "
                            f"({db_above:.1f}dB above baseline) at "
                            f"t={self.detection_time_s:.3f}s")
        else:
            self._high_energy_ms = 0.0
```

### 9.10 `src/data/color_detector.py` (NEW v2.1)

Post-hoc Whisper-based color word identification. Runs on GPU after `execute()` returns.

```python
"""
src/data/color_detector.py

Post-hoc color word identification from audio waveform via faster-whisper.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

VALID_COLORS = ["red", "blue", "green", "yellow"]


class ColorDetector:
    """
    Lazy-loads faster-whisper tiny model on first use.

    Usage:
        det = ColorDetector()
        det.load()
        result = det.detect(audio_waveform, sample_rate=16000)
        # result = {
        #     "color_detected": "red" or None,
        #     "transcript": "red",
        #     "success_audio_transcript": True,
        # }
    """

    def __init__(self, model_size: str = "tiny", device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self):
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper {self.model_size} on {self.device}")
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        if self._model is None:
            self.load()

        # Whisper requires 16kHz mono float32
        if sample_rate != 16000:
            from scipy import signal
            n_target = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, n_target).astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return {
                "color_detected": None,
                "transcript": "",
                "success_audio_transcript": False,
            }

        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1,
        )
        transcript = " ".join(s.text for s in segments).lower().strip()

        detected = None
        for color in VALID_COLORS:
            if color in transcript:
                detected = color
                break

        return {
            "color_detected": detected,
            "transcript": transcript,
            "success_audio_transcript": detected is not None,
        }
```

---

## 10. Step-by-Step Collection Procedures

### 10.1 Pre-Session Setup (Every Fresh Terminal)

```bash
# Ethernet IP assignment (does not persist across reboots)
sudo ip addr flush dev enx98fc84e68f1a
sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
sudo ip link set enx98fc84e68f1a up
ping -c 3 192.168.123.161

# Environment and repo
conda activate env_go2
cd ~/Robotics/Language_Conditioned_UniGo2

# Install v2.1 audio dependencies (one-time per env)
pip install sounddevice faster-whisper scipy

# Verify USB mic device index
python -c "import sounddevice as sd; print(sd.query_devices())"
# Note the integer index of your USB mic — pass to AudioRecorder/AudioLiveDetector
```

Confirm:
- Robot standing in Sport Mode (wait ~60s after power on)
- Ethernet cable taped securely
- Wall button mounted; floor clear of obstructions
- Floor marks for starting positions
- USB microphone plugged in, within 50cm of button, pointed toward it

### 10.2 Phase 0 — K_p Tuning (First Session Only, ~45-60 min)

**Step 1:** Mount button at 0.25 m height, 0.75 m ahead of robot start mark.

**Step 2:** Run tuning script:
```bash
python scripts/collect_stage_d.py --tune-kp --k-p-values 0.3 0.5 1.0
```

For each K_p, the script will:
1. Calibrate audio baseline (2s of silence — keep quiet)
2. Instruct you to position robot at ideal standoff (Episode A)
3. Wait for Enter confirmation
4. Run `HeuristicContactCorrective(k_p=K)` with grounding thread + audio detector
5. Save HDF5 to `data/real/stage_d/tuning/`
6. Print summary: correction magnitude max, final error, `success_fk`, `success_audio_live`, `success_target`, duration
7. Instruct you to position robot at ideal + 3cm (Episode B)
8. Repeat

**Step 3:** After all 6 runs (3 K_p × 2 episodes), review summary table printed by script. Apply selection criteria from Section 8.3.

**Step 4:** Record chosen K_p. Use this value for all subsequent collection.

**Time: 45-60 minutes.**

### 10.2.5 Pilot Audio Validation (NEW v2.1, ~10 min)

Before proceeding to Phase 1 collection, run 5 pilot episodes at no-perturbation and check:

| Check | Target |
|---|---|
| `success_fk` rate | ≥ 4/5 |
| `success_audio_live` rate | ≥ 4/5 |
| `success_fk` and `success_audio_live` agreement | All 5 episodes |
| `color_detected == "red"` when `success_audio_live` is True | All detected cases |
| Ambient noise floor (from calibration logs) | RMS baseline stable, no spikes during silence |

**If `success_audio_live` rate is low (< 60%):**
1. Manually press button by hand — does `AudioLiveDetector.sound_detected` fire? If no, mic/threshold problem.
2. Have robot press, inspect camera footage — did foot visibly depress button? If no, increase `FR_EXTEND_OFFSET_WALL[2]` (calf component) by 0.1 rad and retest.
3. Check USB mic position — should be within 50cm of button, pointed toward it.

**If audio and FK agree in all 5 pilots:** proceed to Phase 1 confidently.

**If audio disagrees with FK:** audio is authoritative, FK is approximate. Proceed to Phase 1 but use `success_target` (audio-based) as the primary metric in training and evaluation.

### 10.3 Phase 1 — Stage C Collection (Varied Button Placements, ~1-1.5 hr)

15 episodes across 5 heights × 3 distances.

**Heights:** 0.15 m, 0.20 m, 0.25 m, 0.30 m, 0.35 m
**Distances:** 0.60 m, 0.75 m, 0.90 m

For each of the 15 combinations:

**Step 1:** Mount/move button to target height; measure with tape. Record floor distance.

**Step 2:** Position robot at floor mark matching the target distance.

**Step 3:** Run collection:
```bash
python scripts/collect_stage_d.py \
    --mode stage-c \
    --button-height 0.25 \
    --distance 0.75 \
    --angle 0 \
    --perturbation none \
    --k-p 0.5
```

The script:
1. `ChannelFactoryInitialize(0, 'enx98fc84e68f1a')` — once
2. Instantiate `Go2Interface`, `Go2Camera`, `VisualGrounder`, `HeuristicContactCorrective`, `AudioRecorder`, `AudioLiveDetector`, `ColorDetector`
3. Preload Whisper model (`color_detector.load()`)
4. Open mic streams (`audio_recorder.start_stream()`, `audio_detector.start_stream()`)
5. Calibrate audio baseline (2s of silence)
6. Start `GroundingThread`, `StageDRecorder`
7. `go2.stand()` — wait for settle
8. Initial grounding → `target_pos_base_initial`
9. If confidence < 0.5 or depth invalid, abort with message
10. `standoff = target_pos_base_initial + (-PRESS_OFFSET_X, -PRESS_OFFSET_Y, 0)`
11. `go2.move_to_position(standoff[0], standoff[1], 0, speed=0.3)`
12. `go2.stop()`; wait 2s for settle
13. Re-ground → `target_pos_base_at_standoff`
14. `audio_recorder.start_episode()`
15. `audio_detector.start_episode()`
16. `recorder.begin_episode(metadata)`
17. `heuristic.execute(target_offset=target_pos_base_at_standoff, press_mode='wall', audio_detector=audio_detector)`
18. `recorder.end_episode(result)`
19. `audio_waveform = audio_recorder.end_episode()`
20. `color_result = color_detector.detect(audio_waveform, sample_rate=16000)`
21. Build metadata override with success fields:
    ```python
    metadata_override = {
        'success_fk':             result.success,
        'success_audio_live':     audio_detector.sound_detected,
        'audio_detection_time_s': audio_detector.detection_time_s
                                    if audio_detector.detection_time_s is not None
                                    else float('nan'),
        'contact_method':         result.contact_method,
        'color_detected':         color_result['color_detected'] or "",
        'color_expected':         "red",
        'success_target':         (color_result['color_detected'] == "red"),
        'audio_transcript':       color_result['transcript'],
    }
    ```
22. `recorder.save(hdf5_path, audio=audio_waveform, audio_sample_rate=16000, metadata_override=metadata_override)`
23. Stop grounding thread
24. Prompt user: keep or discard episode

**Step 4:** Immediately validate:
```bash
python scripts/validate_episode.py data/real/stage_d/episode_YYYYMMDD_HHMMSS.h5
```

**Step 5:** Reposition for next configuration.

**Time: ~5 min per episode × 15 = 75 min.**

### 10.4 Phase 2 — Stage D No-Perturbation Collection (10 episodes, ~50 min)

Fixed button position: height 0.25 m, distance 0.75 m, angle 0°.

```bash
python scripts/collect_stage_d.py \
    --mode stage-d \
    --button-height 0.25 \
    --distance 0.75 \
    --angle 0 \
    --perturbation none \
    --num-episodes 10 \
    --k-p 0.5
```

Per-episode procedure is identical to Phase 1 (audio wiring included). The script loops 10 times, prompting user to reposition robot between episodes.

**Time: ~5 min per episode × 10 = 50 min.**

### 10.5 Phase 3 — Stage D Small Perturbation Collection (15 episodes, ~75 min)

```bash
python scripts/collect_stage_d.py \
    --mode stage-d \
    --button-height 0.25 \
    --distance 0.75 \
    --angle 0 \
    --perturbation random-small \
    --num-episodes 15 \
    --k-p 0.5
```

Perturbation: `dx, dy ∈ uniform(-0.02, 0.02)`. Script samples a new random perturbation per episode. `move_to_position` commanded offset is ideal + perturbation. The `perturbation_commanded` metadata field records the sampled value; `target_pos_base_at_standoff` records what was actually observed.

**Time: ~5 min per episode × 15 = 75 min.**

### 10.6 Phase 4 — Stage D Large Perturbation Collection (10 episodes, ~50 min)

```bash
python scripts/collect_stage_d.py \
    --mode stage-d \
    --button-height 0.25 \
    --distance 0.75 \
    --angle 0 \
    --perturbation random-large \
    --num-episodes 10 \
    --k-p 0.5
```

Perturbation: `dx, dy ∈ uniform(-0.05, 0.05)`. Expect higher failure rate (~60%). Failed episodes are still saved and included in training data — they teach the policy reach limits.

**Time: ~5 min per episode × 10 = 50 min.**

### 10.7 Post-Session Validation

```bash
python scripts/validate_dataset.py data/real/stage_d/
python scripts/summarize_dataset.py data/real/stage_d/ --output dataset_summary.md
```

Expected dataset characteristics:

| Phase | Episodes | `success_fk` | `success_audio_live` | `success_target` |
|---|---|---|---|---|
| K_p tuning (ignored) | 6 | — | — | — |
| Stage C varied | 15 | ≥ 80% | ≥ 70% | ≥ 70% |
| Stage D no-perturb | 10 | ≥ 85% | ≥ 80% | ≥ 80% |
| Stage D small perturb | 15 | ≥ 70% | ≥ 65% | ≥ 65% |
| Stage D large perturb | 10 | ≥ 50% | ≥ 45% | ≥ 45% |
| **Total for training** | **50** | ≥ 70% | ≥ 65% | **≥ 65% (primary)** |

Note: `success_target ≤ success_audio_live` always (can't be correct color if no audio). `success_audio_live` is expected to be slightly lower than `success_fk` because a close foot-miss triggers FK proximity but doesn't depress the button.

### 10.8 Session Time Budget

| Phase | Time | Cumulative |
|---|---|---|
| Setup | 15 min | 0:15 |
| K_p tuning | 60 min | 1:15 |
| Pilot audio validation | 10 min | 1:25 |
| Stage C (15 eps) | 75 min | 2:40 |
| Stage D no-perturb (10 eps) | 50 min | 3:30 |
| Stage D small perturb (15 eps) | 75 min | 4:45 |
| Stage D large perturb (10 eps) | 50 min | 5:35 |
| Validation | 15 min | 5:50 |
| **Total** | **~5.5-6 hours** | |

Realistic budget including breaks and unexpected issues: **7-8 hours**. Split across 2 sessions if needed.

**Split plan:** Session 1 = K_p tuning + pilot + Stage C (~2.5 hr). Session 2 = all Stage D collection (~3.5 hr).

---

## 11. Data Validation Checklist

### 11.1 Per-Episode Integrity

Run `scripts/validate_episode.py path/to/episode.h5`. Checks:

- [ ] HDF5 file opens without errors
- [ ] Root attributes contain all fields from Section 5.7 (including v2.1 audio fields: `success_fk`, `success_audio_live`, `audio_detection_time_s`, `contact_method`, `color_detected`, `color_expected`, `success_target`, `audio_transcript`, `audio_sample_rate`)
- [ ] `per_step` group has all 12 fields from Section 5.5 with consistent T
- [ ] `grounding` group has all 5 fields from Section 5.6 with consistent G
- [ ] `phase_transitions` group contains at minimum `lift_end`, `extend_end`, `hold_end`
- [ ] `audio` dataset present, dtype float32, length matches `duration_s * audio_sample_rate` (±10%)
- [ ] `audio_sample_rate` attribute present on root
- [ ] `contact_method` is one of `{"fk", "audio", "foot_force", "none"}`
- [ ] If `success_audio_live` is True, `audio_detection_time_s` is not NaN and lies within episode duration
- [ ] No NaN in `joint_pos_actual`, `joint_vel_actual`, `joint_pos_cmd`, `imu_rpy`
- [ ] NaN only allowed in `target_pos_base` / `foot_to_target_error` during grounding gaps
- [ ] `jacobian_pid_delta` not identically zero for perturbed episodes
- [ ] Episode duration reasonable: 8 s < duration < 40 s

### 11.2 Dataset-Level

Run `scripts/validate_dataset.py data/real/stage_d/`. Checks:

- [ ] 50 total training episodes (excluding K_p tuning)
- [ ] `success_target` rate ≥ 60% overall (audio-verified success, primary metric)
- [ ] `success_fk` rate ≥ 70% overall
- [ ] `success_fk` and `success_target` agreement ≥ 80% (large disagreement suggests calibration issue)
- [ ] `contact_method == "audio"` rate ≥ 60% (audio should usually fire before FK)
- [ ] `target_pos_base_at_standoff` histogram spans intended ranges
- [ ] Phase duration distributions reasonable
- [ ] `foot_to_target_error` magnitudes: ≤ 3 cm (no-perturb), up to 5-7 cm (large-perturb)
- [ ] `jacobian_pid_delta` magnitude grows with perturbation magnitude

### 11.3 Coordinate Frame Spot-Check

- [ ] `target_pos_base_at_standoff[0]` ≈ 0.203 m for no-perturbation episodes
- [ ] `target_pos_base_at_standoff[1]` ≈ 0.140 m for no-perturbation episodes
- [ ] `target_pos_base_at_standoff[2]` matches button height relative to base link

---

## 12. Summary

After this data collection session, the expected outputs are:

- ~50 HDF5 episode files in `data/real/stage_d/` containing everything needed for both Stage C and Stage D training, plus audio-verified success labels
- Modified `heuristic_contact.py` (v2.1) with `phase_transitions`, `current_waypoint_fr`, and `audio_detector` integration
- New file `src/planner/heuristic_contact_corrective.py` with Jacobian-PID subclass
- New files `src/data/grounding_thread.py`, `src/data/recorder.py`
- New files (v2.1) `src/data/audio_recorder.py`, `src/data/audio_live_detector.py`, `src/data/color_detector.py`
- Collection and validation scripts in `scripts/`
- Tuned K_p value documented for use in evaluation

With this data in hand, Priority 4 can implement:

- Stage C MLP (5 → 12 regression) trained on episode metadata, filtered by `success_target`
- Stage D residual policy (15 → 3) trained on per-step recordings, filtered by `success_target`
- Evaluation comparing all 5 method variants (Baselines 1-4 + Core Method), with `success_target` as the primary metric

*Training Data Specification v2.1 — Language-Conditioned Whole-Body Contact Manipulation | April 21, 2026*
