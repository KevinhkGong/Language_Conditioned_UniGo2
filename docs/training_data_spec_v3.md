# Training Data Specification — Stage C & Stage D

**Project:** Language-Conditioned Whole-Body Contact Manipulation for Quadruped Robots
**Version:** 3.0
**Date:** April 26, 2026
**Status:** Final — describes the data collection protocol that was actually executed and the dataset that was assembled

---

## Document History

- **v1.0** (April 19, 2026): Original Jacobian-PID expert collection plan
- **v2.0** (April 21, 2026): Added Jacobian-PID corrective expert, K_p tuning procedure
- **v2.1** (April 21, 2026): Added USB-microphone audio contact verification with `faster-whisper` post-hoc color detection
- **v3.0** (April 26, 2026): **Major revision.** Pivoted from Jacobian-PID expert to hand-guided kinesthetic demonstration. Documents the actual executed protocol and the assembled dataset. Original Jacobian-PID design preserved as Appendix A.

**Reason for pivot:** During implementation testing, the Jacobian-PID expert with closed-loop visual servoing introduced a closed-loop control variable (the Jacobian gain `K_p`) that was difficult to tune safely on hardware (one tuning attempt at `KP_FR_LIFT=80` caused a forward flip damaging the Ethernet port housing). Hand-guided kinesthetic demonstration was substituted because it preserves the data structure (per-step achieved deltas as labels) without requiring expert-controller tuning, while also providing a more direct demonstration of operator intent. Audio contact verification via USB microphone was retained from v2.1.

---

## 1. Scope

This document specifies training data for two learned components:

- **Stage C — Contact Planner (two MLPs):**
  - Standoff head: predicts `(dx, dy, dyaw)` standoff offset from 3D target position
  - Waypoint head: predicts FR leg waypoints (lift, extend, press) from 3D target position
- **Stage D — Reactive Closed-Loop Residual Policy:** predicts whole-body 12-dim joint corrections during contact based on live state including foot-to-target error.

Both stages are trained on **real-robot data only** — no synthetic pretraining. The 500 synthetic episodes from Summary 2 remain on disk as historical artifacts but are not used.

---

## 2. Final Design Decisions

### 2.1 Architecture (locked)

| Decision | Choice |
|---|---|
| Stage C — Standoff MLP | 5 → 64 → 64 → 64 → 3, ReLU |
| Stage C — Waypoint MLP | 5 → 64 → 64 → 64 → 9, ReLU |
| Stage D output dimensions | **12** — full whole-body residual (lift, extend, hold phases only) |
| Stage D architecture | 33 → 128 → 128 → 128 → 12, ReLU, no output activation |
| Stage D state input | 33 dims (see Section 5.1) |
| Stage D control frequency | 500 Hz |
| Stage D loss | Per-joint weighted MSE: `((pred-target)**2 * w).sum(dim=1).mean()` |
| Stage D optimizer | AdamW, weight_decay=1e-4, no LR scheduling, no early stopping |
| State normalization | Z-score per-dim, fit on train split only, serialized in checkpoint bundle |
| Training data source | Real robot only |
| File format | HDF5 per-episode |

### 2.2 Joint Weight Strategy

The Stage D loss uses per-joint weights that depend on data composition:

| Training data | Joint weights | Use case |
|---|---|---|
| v2 only | `[5,5,5,0,0,0,0,0,0,0,0,0]` | Baseline FR-only model |
| v2 + v3 combined | `[5,5,5,1,1,1,1,1,1,1,1,1]` | Whole-body model — primary deployment candidate |

The 5x weighting on FR reflects the fact that FR is the primary manipulator. The 1x weighting on support legs in the combined regime gives meaningful gradient on rear-leg outputs from v3 demonstrations without overwhelming FR learning.

### 2.3 Demonstration Paradigm (locked)

- **Imitation learning**, not reinforcement learning
- **Hand-guided kinesthetic teaching**, not Jacobian-PID expert
- **Two collection regimes:**
  - **v2: FR-only guided** — Person 1 hand on FR paw, support legs at full stiffness (KP=100). Compliance only on FR (KP=15 during extend+hold).
  - **v3: Whole-body guided** — Person 1 on FR paw, Person 2 on rear haunches/lower back. Both FR (KP=15) and support legs (KP=55) softened during extend+hold.

### 2.4 Contact Verification (locked)

- **Audio contact verification** via USB microphone, energy-threshold live detection during hold phase
- **Post-hoc color identification** via `faster-whisper` tiny model on GPU
- **Three-tier success semantics:**
  - `success_fk` — FK-proximity (controller's belief)
  - `success_audio_live` — physical press detected by audio energy spike
  - `success_target` — color word in transcript matches expected color (primary metric)

This system from v2.1 was retained and used for all v2 and v3 collection.

---

## 3. Hand-Guided Demonstration Protocol

### 3.1 Two Subclasses

**`HeuristicContactGuided`** (v2 collection):
- Inherits from `HeuristicContact`
- Overrides `_send_cmd` to apply `KP_FR_COMPLIANT=15`, `KD_FR_COMPLIANT=1.5` to FR motors during extend+hold
- All other phases unchanged from base class
- `collection_mode = "hand_guided"`
- `gain_schedule = "fr_only_compliant"`

**`HeuristicContactWholeBody`** (v3 collection):
- Inherits from `HeuristicContactGuided`
- Further overrides `_send_cmd` to apply `KP_SUPPORT_SOFT=55`, `KD_SUPPORT_SOFT=5.5` to support legs (FL, RL, RR) during extend+hold
- Lift, retract, and lower_to_sit phases use parent's gain schedule (full stiffness)
- `collection_mode = "wholebody_guided"`
- `gain_schedule = "wholebody_v2_kpsupport55"`

### 3.2 Gain Schedule Rationale

| Phase | FR motors | Support legs (FL/RL/RR) |
|---|---|---|
| sit_to_stand | KP=40, KD=3 | KP=100, KD=6 |
| weight_shift | KP=40, KD=3 | KP=100, KD=6 |
| lift | KP=40, KD=3 | KP=100, KD=6 |
| **extend (v2)** | **KP=15, KD=1.5** | KP=100, KD=6 |
| **hold (v2)** | **KP=15, KD=1.5** | KP=100, KD=6 |
| **extend (v3)** | **KP=15, KD=1.5** | **KP=55, KD=5.5** |
| **hold (v3)** | **KP=15, KD=1.5** | **KP=55, KD=5.5** |
| retract_curl, retract_rotate, retract_extend | KP=40, KD=3 | KP=100, KD=6 |
| weight_unshift, settle | KP=40, KD=3 | KP=100, KD=6 |
| lower_to_sit | KP=25, KD=1.5 | KP=25, KD=1.5 |

### 3.3 Calibration of `KP_SUPPORT_SOFT`

A first attempt with `KP_SUPPORT_SOFT=45` produced observable rear sag during hold phase even with no operator pressure (~1cm rear drop over 1-2 seconds). This was diagnosed as soft gains being insufficient to support the body against gravity once the FR leg was extended forward (CoM shifted).

The constant was raised to `55` and the gain schedule string updated to `"wholebody_v2_kpsupport55"`. Five episodes had been collected at the original `KP=45` (`gain_schedule="wholebody_v1_kpsupport45"`) and were excluded from primary training data due to the sag artifact and Person 2's compensating over-pressure.

The five excluded episodes are preserved in `data/real/stage_d_v3_first/` for historical reference but are not used in training. Their file metadata distinguishes them from primary v3 data via `gain_schedule`.

### 3.4 Two-Person Coordination Protocol

For v3 collection, both operators coordinate verbally during each episode:

| Verbal cue | Trigger | Action |
|---|---|---|
| "Compliance" | `[GUIDED] Compliance active` prints | Both place hands at respective contact points |
| "Extending" | Begin of extend phase | Person 2 begins gentle body adjustment |
| "Easy" | Either operator senses resistance | Both ease off |
| "Holding" | Body reaches desired position | Both stabilize |
| "Audio" | `[GUIDED] Contact detected` prints | Prepare to release |
| "*** RELEASE NOW ***" | Watcher cue prints | Both release hands simultaneously |

Synchronized release is critical: gain restoration to KP=100 happens within the same control cycle as the phase transition from hold to retract_curl. Late release while body is displaced creates a torque jolt as support legs return to commanded position.

### 3.5 Three Demonstration Modes

Three demonstration techniques emerged during the v3 collection sessions and are present in the dataset:

**Mode S — Stabilize-Only (NAV_EXTRA_FORWARD_M = 0.15)**
Person 2 prevents natural rear sag by applying gentle counter-pressure on rear haunches. No deliberate body translation. Body posture remains approximately at commanded weight-shifted pose throughout extend+hold. Person 1 demonstrates full FR reach as primary actor.

**Mode T — Translate-Mode (NAV_EXTRA_FORWARD_M = 0.10)**
Standoff is moved 5cm farther back (button is just outside FR's natural reach). Person 2 actively translates the body forward by 1-2cm to close the reach gap. Person 1 demonstrates the remaining FR extension. Both contributions are needed for successful contact.

**Mode S- — Stabilize+Backward (NAV_EXTRA_FORWARD_M = 0.15, drift-induced)**
When grounding accuracy drifted during a session and the robot was navigating too close to the button, operators allowed natural sag and added small -x translation to prevent FR overshoot. This emerged as adaptive behavior rather than planned protocol.

These modes are distinguishable in the data via the `nav_extra_forward_m` metadata field and through analysis of the per-step rear-leg `achieved_delta_q`.

---

## 4. Stage C Data Specification

### 4.1 Input (5 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `target_pos_base` | (3,) | float32 | Button position in robot base frame at standoff |
| `interaction_onehot` | (2,) | float32 | `[1, 0]` for press, `[0, 1]` for push (push not collected) |

### 4.2 Standoff Head Output (3 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `standoff_offset` | (3,) | float32 | `(dx, dy, dyaw)` ideal base pose offset from current to target standoff |

Label source: `standoff_dxdy_commanded` metadata field, which records the navigation offset the controller actually executed at standoff time. Filter at training time by `success_target=True` and presence of this field.

### 4.3 Waypoint Head Output (9 dims)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `lift_waypoint_fr` | (3,) | float32 | FR `[hip, thigh, calf]` joint angles at end of lift phase |
| `extend_waypoint_fr` | (3,) | float32 | FR joints at end of extend phase |
| `press_waypoint_fr` | (3,) | float32 | FR joints at end of hold phase |

Label source: `phase_transitions` HDF5 group, populated at each gate-passed phase advance with actual achieved joint positions (not commanded). Required keys: `lift_end`, `extend_end`, `hold_end`.

---

## 5. Stage D Data Specification

### 5.1 State Input (33 dims)

| Slice | Field | Description |
|---|---|---|
| [0:3] | `fr_joint_pos` | FR joints `[hip, thigh, calf]` actual |
| [3:6] | `fr_joint_vel` | FR joint velocities |
| [6:9] | `foot_to_target_error` | `target_pos_base - FK(fr_joint_pos)` |
| [9:12] | `nominal_waypoint_fr` | Current heuristic FR command (pre-correction) |
| [12:13] | `phase_progress` | Fractional progress within current phase |
| [13:15] | `imu_rp` | Roll, pitch (yaw omitted) |
| [15:27] | `joint_tau_est` | Estimated torques on all 12 joints |
| [27:30] | `imu_gyro` | Angular velocity |
| [30:33] | `imu_accel` | Linear acceleration |

### 5.2 Output (12 dims)

`achieved_delta_q[t] = joint_pos_actual[t+1] - joint_pos_cmd[t]` for all 12 joints.

The 12-dim output design makes the architecture compatible with both v2-only training (rear-leg weights = 0) and v2+v3 training (rear-leg weights = 1), without architecture changes between regimes.

### 5.3 Recording Window

Recording is active only during phases where Stage D operates:

```python
recording_active = heuristic._phase in {"lift", "extend", "hold"}
```

Approximately 3000-4000 steps per episode at 500 Hz (~7-8 seconds total).

### 5.4 Per-Step Recording Schema

HDF5 group `per_step`:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `timestamp` | (T,) | float64 | Seconds from episode start |
| `joint_pos_actual` | (T, 12) | float32 | All 12 joints from `rt/lowstate` |
| `joint_vel_actual` | (T, 12) | float32 | All 12 velocities |
| `joint_pos_cmd` | (T, 12) | float32 | Final commanded joint positions |
| `joint_kp_used` | (T, 12) | float32 | Active KP per motor (v3 only) |
| `joint_tau_est` | (T, 12) | float32 | Estimated torques |
| `fr_foot_pos_fk` | (T, 3) | float32 | FR foot via FK, base frame |
| `target_pos_base` | (T, 3) | float32 | Latest grounding (cached between updates) |
| `foot_to_target_error` | (T, 3) | float32 | `target_pos_base - fr_foot_pos_fk` |
| `imu_rpy` | (T, 3) | float32 | Roll, pitch, yaw |
| `imu_gyro` | (T, 3) | float32 | Angular velocity |
| `imu_accel` | (T, 3) | float32 | Linear acceleration |
| `phase_label` | (T,) | int8 | 0=lift, 1=extend, 2=hold |
| `phase_progress` | (T,) | float32 | `[0, 1]` within current phase |
| `nominal_waypoint_fr` | (T, 3) | float32 | Current FR command before any correction |

`achieved_delta_q` is computed at load time from `joint_pos_actual` and `joint_pos_cmd`, not stored directly.

### 5.5 Grounding Thread Output

HDF5 group `grounding`:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `timestamp` | (G,) | float64 | Seconds from episode start |
| `position_base` | (G, 3) | float32 | Button position from grounding (NaN if invalid) |
| `confidence` | (G,) | float32 | GroundingDINO confidence |
| `depth_m` | (G,) | float32 | Corrected depth |
| `position_base_valid` | (G,) | bool | False when grounding returned None |

G ≈ 75 per episode at 5 Hz over 15 seconds.

### 5.6 Episode Metadata

HDF5 root attributes:

| Field | Type | Description |
|---|---|---|
| `episode_id` | str | `YYYYMMDD_HHMMSS` |
| `collection_mode` | str | `"hand_guided"` (v2) or `"wholebody_guided"` (v3) |
| `gain_schedule` | str | `"fr_only_compliant"` (v2), `"wholebody_v1_kpsupport45"` (v3 first batch, excluded), or `"wholebody_v2_kpsupport55"` (v3 primary) |
| `target_pos_base_at_standoff` | (3,) float32 | Button position at start of contact phase |
| `target_pos_world_initial` | (3,) float32 | Initial grounding position |
| `standoff_dxdy_commanded` | (2,) float32 | Navigation offset commanded at standoff |
| `press_offset_x` | float32 | Constant from `heuristic_contact.py` (0.593m) |
| `press_offset_y` | float32 | Constant from `heuristic_contact.py` (-0.047m) |
| `nav_extra_forward_m` | float32 | Per-session standoff bias (0.10, 0.15, or 0.18) |
| `y_sit_stand_drift_comp` | float32 | Lateral drift compensation |
| `success_fk` | bool | FK-proximity success |
| `success_audio_live` | bool | Audio energy spike during hold |
| `audio_detection_time_s` | float | Time of audio detection (NaN if not detected) |
| `contact_method` | str | First-firing signal: `"fk"`, `"audio"`, `"foot_force"`, `"none"` |
| `color_detected` | str | Whisper transcription result |
| `color_expected` | str | Expected color from task config (always `"red"`) |
| `success_target` | bool | `color_detected == color_expected` — **primary metric** |
| `audio_transcript` | str | Full Whisper transcript |
| `audio_sample_rate` | int | Hz (typically 16000) |
| `contact_step` | int | Step at first contact (-1 if none) |
| `duration_s` | float | Total contact phase duration |
| `failure_mode` | str | Empty if success; else descriptive string |
| `phase_transitions` | group | Nested 12-dim joint vectors at each phase boundary |
| `button_height_m` | float | Measured button height above floor |
| `notes` | str | Free-form lab notebook entry |

### 5.7 Audio Dataset

Root-level HDF5 dataset:

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `audio` | (N,) | float32 | Raw mono waveform from USB mic, full episode duration |

At 16 kHz and ~15s episodes, ~240k samples ≈ 700-800KB compressed per episode. Total audio storage for primary dataset (~35 episodes): ~30MB.

---

## 6. Dataset Inventory

### 6.1 Final Training Dataset

| Directory | Episodes | Mode | Used for training? | Notes |
|---|---|---|---|---|
| `data/real/stage_d_v2/` | 18 (15 successful) | FR-only guided | **Yes** | Primary v2 data |
| `data/real/stage_d_v3/` | 20 (20 successful) | Whole-body guided, KP=55 | **Yes** | Primary v3 data |
| `data/real/stage_d/` | ~40 (variable, ~25% success) | Pre-protocol exploration | No | Collected before contact protocol stabilized |
| `data/real/stage_d_v3_first/` | 5 | Whole-body guided, KP=45 | No | Excluded due to gain calibration (rear sag, over-pressure) |

**Primary training dataset:** 35 successful episodes (15 v2 + 20 v3).

### 6.2 v3 Episode Distribution

The 20 primary v3 episodes were collected in a single session on April 25, 2026, at `gain_schedule="wholebody_v2_kpsupport55"`. Distribution by mode:

| Mode | Count | NAV_EXTRA_FORWARD_M | Demonstration |
|---|---|---|---|
| S (stabilize-only) | ~10 | 0.15 | Person 2 prevents sag, no translation |
| S- (stabilize + small backward) | ~5 | 0.15 | Adaptive — drift caused robot too close, allowed sag + slight -x |
| T (translate-mode) | 5 | 0.10 | Person 2 actively translates body forward 1-2cm |

The exact mode per episode is recorded in the lab notebook. The `nav_extra_forward_m` metadata field distinguishes T from S/S-. The S vs S- distinction is in lab notebook only.

### 6.3 Standoff Coverage

Across the 35 primary episodes, the `target_pos_base_at_standoff` distribution:

- x range: 0.49–0.62 m (13cm spread)
- y range: -0.10 to -0.26 m (16cm spread)
- z range: 0.15–0.22 m (button heights tested)

This range emerged from a combination of intentional button-position variation, grounding accuracy drift over sessions, and Person 2's adaptive corrections. The variation is treated as a feature: the model sees diverse state-action pairs that match deployment-regime variation.

---

## 7. Final Validation Results

Stage D was trained in three configurations to validate the design. Full results in eval JSON files at `models/stage_d_*/eval.json`.

### 7.1 Three Configurations

| Test | Data | Joint Weights | Output Directory |
|---|---|---|---|
| A | v2 only | `[5,5,5,0,0,0,0,0,0,0,0,0]` | `models/stage_d_v2only/` |
| B | v2 + v3 combined | `[5,5,5,1,1,1,1,1,1,1,1,1]` | `models/stage_d_combined/` |
| C | v2 + v3 combined | `[5,5,5,0,0,0,0,0,0,0,0,0]` | `models/stage_d_combined_fronly/` |

### 7.2 Headline Per-Joint MSE

| Joint | A: v2 + FR-only | C: combined + FR-only | B: combined + full |
|---|---|---|---|
| FR_hip | 0.00004 | 0.00002 | 0.00001 |
| FR_thigh | 0.00011 | 0.00012 | 0.00010 |
| FR_calf | 0.00025 | 0.00002 | 0.00003 |
| FL_hip | 0.01094 | 0.00644 | **0.00002** |
| RL_thigh | 0.00942 | 0.00964 | **0.00005** |
| RR_calf | 0.02245 | 0.02352 | **0.00007** |
| Overall | 0.00630 | 0.00634 | **0.00005** |

Test B's overall MSE is 126x lower than Test A. Rear-leg MSE dropped 100-300x.

### 7.3 Distribution Match (Test B)

For all 12 joints, predicted distribution width matches true distribution width (e.g., FR_hip pred_std=0.0342 vs true_std=0.0353; RR_calf pred_std=0.0260 vs true_std=0.0243). This indicates state-dependent learning rather than averaged or noisy outputs.

### 7.4 Conclusions

1. **Whole-body residual policy works.** Test B demonstrates that a 12-dim policy trained on hand-guided demonstrations produces coordinated whole-body corrections matching the demonstrated distribution.
2. **v3 demonstrations contain learnable signal.** Two-person hand-guided collection at softened support gains produced data the model uses effectively.
3. **No tradeoff between FR and rear-leg learning.** Adding rear-leg supervision did not degrade FR predictions.
4. **35 episodes is sufficient for first results.** Additional episodes would broaden state distribution coverage, particularly for large foot-to-target geometries that all three models handle poorly.

### 7.5 Deployment Recommendation

The combined model in `models/stage_d_combined/` is the candidate for deployment. Apply the full 12-dim residual at inference. The v2-only model in `models/stage_d_v2only/` serves as the FR-only baseline for evaluation comparison; at inference it must slice output to `[0:3]` to avoid pushing rear legs with trunk-leakage noise.

---

## 8. Method Variants for Evaluation

The final method variants reported in evaluation:

| Variant | Approach | Contact Phase | Re-grounding |
|---|---|---|---|
| Baseline 1 | Heuristic dead-reckoning | Scripted fixed sequence | None |
| Baseline 2 | Learned Stage C | Scripted fixed sequence | None |
| Baseline 3 | Learned Stage C | Scripted fixed sequence | Approach only |
| **Core Method** | **Learned Stage C** | **Stage D (combined model)** | **Approach + Contact** |

The originally planned "Baseline 4: Jacobian-PID reactive" was not implemented (see Appendix A).

---

## Appendix A — Original Jacobian-PID Design (Not Used)

This appendix preserves the original v2.1 design for reference. None of these components were implemented or used in the final dataset.

### A.1 Jacobian-PID Expert Concept

The original plan called for a `HeuristicContactCorrective` subclass that would override `_control_loop` to apply visual-servoing correction to the FR leg during lift, extend, and hold phases:

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

This was intended both as a data-collection expert (the per-step `delta` would be the Stage D label) and as an evaluation baseline.

### A.2 K_p Tuning Procedure (Not Executed)

The original plan called for a structured K_p tuning session before primary collection: tests at K_p ∈ {0.3, 0.5, 1.0} with no perturbation and with +3cm perturbation, evaluating correction smoothness, support-leg stability, and audio success.

### A.3 Algorithmic Perturbations (Not Executed)

The original plan called for deliberate standoff perturbation via `go2.move_to_position(ideal_x + dx, ideal_y + dy, 0)` before each Jacobian-PID episode, with `dx, dy ∈ uniform(-0.05, 0.05)` for random-large perturbations. This was intended to give the Jacobian-PID expert non-trivial corrections to make.

### A.4 Reason for Pivot

During implementation testing, raising `KP_FR_LIFT` from the safe 40 to an experimental 80 (intended to support Jacobian-PID corrections during lift) caused the robot to execute a violent forward flip, damaging the Ethernet port housing. Subsequent analysis identified this as a fundamental risk of the Jacobian-PID approach: the expert's correction magnitude depends on K_p and tuning failures could be catastrophic at hardware scale.

Hand-guided kinesthetic teaching was substituted because:
- Operator force is bounded by human strength (no runaway saturation)
- Operator can release immediately on observing instability
- The data structure (per-step achieved-delta as label) is identical
- Demonstration intent is more directly expressed (operator decides when and how to correct, model learns to imitate that decision)

The pivot preserved all surrounding infrastructure (audio detection, HDF5 schema, recorder thread, grounding thread, validators) — only the expert's control loop changed.

---

## Appendix B — File Inventory

### B.1 Source Files (Implemented and Used)

```
src/planner/
├── heuristic_contact.py              base controller, gate-based phase machine
├── heuristic_contact_guided.py       v2 collection — FR softening
└── heuristic_contact_wholebody.py    v3 collection — FR + support softening

src/data/
├── recorder.py                       per-step + audio HDF5 writer
├── grounding_thread.py               background visual grounding at 5 Hz
├── audio_recorder.py                 USB mic full-episode capture
├── audio_live_detector.py            energy-threshold live detection
└── color_detector.py                 post-hoc Whisper transcription

src/models/
├── stage_c.py                        StandoffMLP, WaypointMLP
└── stage_d.py                        StageDPolicy, StageDBundle

scripts/
├── collect_guided.py                 v2 collection orchestrator
├── collect_wholebody.py              v3 collection orchestrator
├── train_stage_c.py                  trains both Stage C heads
├── eval_stage_c.py                   per-sample Stage C eval
├── train_stage_d.py                  trains 12-dim residual policy
├── eval_stage_d.py                   per-joint × per-phase eval
├── validate_episode_v2.py            v2 episode HDF5 validator
└── validate_episode_v3.py            v3 episode HDF5 validator (extends v2)
```

### B.2 Source Files (Designed but Not Implemented)

```
src/planner/
└── heuristic_contact_corrective.py   Jacobian-PID expert — see Appendix A

scripts/
└── (no Jacobian-PID-specific orchestrator was written)
```

### B.3 Data Layout

```
~/Robotics/Language_Conditioned_UniGo2/
├── data/
│   ├── real/
│   │   ├── stage_d/                  pre-protocol exploration (excluded)
│   │   ├── stage_d_v2/               primary v2 data — 18 files
│   │   ├── stage_d_v3/               primary v3 data — 20 files
│   │   └── stage_d_v3_first/         excluded KP=45 batch — 5 files
│   └── synthetic/                    legacy, not used
└── models/
    ├── stage_c/                      combined Stage C (v2 + v3)
    ├── stage_d_v2only/               FR-only baseline
    ├── stage_d_combined/             primary deployment model
    └── stage_d_combined_fronly/      diagnostic comparison
```

---

*Training Data Specification v3.0 — Language-Conditioned Whole-Body Contact Manipulation | April 26, 2026*
