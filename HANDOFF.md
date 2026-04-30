# Language-Conditioned UniGo2 — Project Handoff

**Last updated:** 2026-04-30
**Status:** Demo-ready. Friday 2026-05-01 evaluation runs imminent.
**Audience:** future maintainers, thesis writeup, or a fresh Claude session continuing this work.

This document is intended to be self-contained. A reader who has never seen this repo should be able to: replay the demo, understand every design choice, and reproduce or extend the system. Where I am uncertain about a value, I mark it `TODO: verify` or `(uncertain)`.

---

## Table of Contents

1. System architecture
2. Hardware setup
3. Critical constants
4. Deployment commands (the ones that work)
5. CLI flag reference
6. Data collection regime history
7. Trained model checkpoints
8. Failure modes characterized
9. Engineering decisions and rationale
10. Known issues / TODOs
11. Repo layout
12. Operating procedures
13. Internals (controller hierarchy, data structures, schemas)
14. Demo plan (Friday 2026-05-01)
15. Future work directions

---

## 1. System Architecture

### 1.1 Pipeline overview

```
voice prompt ("press the red button")
    │
    ▼
Stage A — Whisper ASR (faster-whisper)
    │  free-text → object + color extraction
    ▼
Stage B — Visual grounding (RGBD)
    │  GroundingDINO (open-vocab detection) → SAM2 mask → Depth Anything V2
    │  → camera intrinsics deprojection → target_pos_base ∈ ℝ³
    │  - One-shot at episode start (initial detection)
    │  - Approach re-grounding after navigation
    │  - GroundingThread (5 Hz) running through hold phase for live target updates
    ▼
Stage C — Planner (two MLP heads, 5→64×3→{3, 9})
    │  Input: target_pos_base (3) + interaction_onehot (2)
    │  • Standoff head (3): (dx, dy, dyaw) for body navigation
    │  • Waypoint head (9): FR [hip, thigh, calf] at lift / extend / press
    ▼
HeuristicContactWholeBody (500 Hz state machine)
    │  sit_to_stand → weight_shift → lift → extend → hold
    │  → retract_curl → retract_rotate → retract_extend
    │  → weight_unshift → settle → lower_to_sit
    │
    │  Stage D residual added during {lift, extend, hold}
    │  Gravity-FF on FL/RR/RL during airborne FR phases
    │  Audio + FK (+ foot-force, disabled) parallel contact detection
    ▼
Low-level joint commands at 500 Hz: (q*, kp, kd, τ_ff)
```

### 1.2 Stage I/O and architectures

| Stage | Net | Input dim | Output dim | Train data | Loss |
|---|---|---|---|---|---|
| **A** | Whisper (frozen) | audio | text | n/a | n/a |
| **B** | GroundingDINO + SAM2 + DepthAnything V2 (frozen) | RGBD | (3,) target_pos_base | n/a | n/a |
| **C — standoff** | 5→64→64→64→3, ReLU | 5 | 3 | success-filtered demo episodes | MSE |
| **C — waypoints** | 5→64→64→64→9, ReLU | 5 | 9 | success-filtered demo episodes | MSE |
| **D — single-step** | 33→128→128→128→12, ReLU | 33 | 12 | per-step from demo episodes (~50k samples / regime) | weighted MSE |
| **D — chunked** | 33→128→256→256→K·12, ReLU | 33 | K·12 reshaped (K, 12) | same as single-step | weighted MSE per chunk position |

The dual-head Stage C is one bundle checkpoint (`stage_c.pt`); the single-step and chunked Stage D are separate checkpoints with the same input dim but different output layers.

### 1.3 Heuristic state machine phases

Phases live in `src/planner/heuristic_contact.py` (1774 lines, ground truth for everything below).

| Phase | min steps (`STEPS_*`) | Time @ 500 Hz | Gate condition | Gravity-FF? | Stage D residual? |
|---|---|---|---|---|---|
| `sit_to_stand` | 1000 | 2.0 s | all 12 joints within 0.2 rad of `STAND_POS` | – | – |
| `weight_shift` | 800 | 1.6 s | hips 3, 9, 6 within `GATE_THRESHOLD`=0.15 rad of `WEIGHT_SHIFT_POS` | – | – |
| `lift` | 600 | 1.2 s | `|FR_thigh − fr_l[1]| < 0.15 rad` | ✓ FL/RR/RL | ✓ |
| `extend` | 600 | 1.2 s | `|FR_thigh − fr_e[1]| < 0.15 rad` | ✓ FL/RR/RL | ✓ |
| `hold` | up to `CONTACT_MAX_STEPS=6000` (12 s) | depends on contact | contact detected → hold for STEPS_HOLD=500 (1 s); otherwise hard cap | ✓ FL/RR/RL | ✓ |
| `retract_curl` | 400 | 0.8 s | `|FR_calf − target| < 0.15` | ✓ FL/RR/RL | – |
| `retract_rotate` | 600 | 1.2 s | `|FR_thigh − target| < 0.15` | ✓ FL/RR/RL | – |
| `retract_extend` | 400 | 0.8 s | `|FR_calf − target| < 0.15` | ✓ FL/RR/RL | – |
| `weight_unshift` | 800 | 1.6 s | all hips within threshold of `STAND_POS` | – | – |
| `settle` | 500 | 1.0 s | trivial wait | – | – |
| `lower_to_sit` | 1500 | 3.0 s | all 12 joints within threshold of `SIT_POS` | – | – |

Forced advance: every gate has a `GATE_TIMEOUT=500` step (1 s) buffer beyond `min_steps`; if the gate hasn't passed by then, the controller logs a warning and advances anyway.

### 1.4 Closed-loop perception integration

Three places target_pos_base updates flow into the controller:

1. **One-shot detection** at episode start (before navigation). Sets the initial standoff target.
2. **Approach re-grounding** after the body has moved to standoff. Refreshes target_pos_base so Stage C has a fresh observation when predicting FR waypoints.
3. **`GroundingThread`** (5 Hz) running through the active phases (lift / extend / hold). Updates a thread-safe `latest_position`. Two consumers:
   - **Stage D state input** — `foot_to_target_error[t] = target_pos_base[t] − fr_foot_pos_fk[t]` is recomputed every control step from the latest grounding; if grounding is stale, the controller falls back to the last valid value (cached in `_stage_d_target_cache` to keep Stage D's input never-NaN).
   - **`_maybe_track_moving_target()`** — opt-in via `--track-target-during-hold`. Two modes:
     - `stage_c`: re-predict FR waypoints from the live target via `StageCInference.predict_waypoints()`. Bounded by training-data spatial sensitivity.
     - `jacobian`: analytical inverse Jacobian of FR foot. Maps Cartesian Δtarget to FR joint Δq. Per-update clamp ±0.10 rad to reject spikes. With `--track-target-relift-threshold > 0`, a large lateral shift in `hold` (before contact latches) rewinds `phase = "lift"` for a clean re-press.

---

## 2. Hardware Setup

### 2.1 Robot

- **Unitree Go2** (quadruped, 12 actuated DOF + IMU + foot-force sensors).
- **SDK:** `unitree_sdk2py` (DDS-based). All low-level control flows through `src/robot/go2_interface.py`.
- **Body mass:** `GO2_NOMINAL_MASS_KG = 15.0` kg (used for gravity FF scaling). Empirical, not measured precisely.

### 2.2 Ethernet / network

- **Interface name:** `enx98fc84e68f1a` (USB-C-to-Ethernet adapter; this name is specific to this PC's MAC). Hard-coded as `DEFAULT_INTERFACE` in scripts.
- **Robot IP:** `192.168.123.161/24`
- **Host static IP:** `192.168.123.99/24`

**Pre-session bring-up sequence** (from `robot_connection`):

```bash
# 1. Assign static IP to the Ethernet interface
sudo ip addr flush dev enx98fc84e68f1a
sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
sudo ip link set enx98fc84e68f1a up

# 2. Verify robot is reachable
ping -c 3 192.168.123.161

# 3. If ping fails — scan subnet to find robot IP
sudo nmap -sn 192.168.123.0/24

# 4. Activate environment
conda activate env_go2
```

**Hardware caveat:** the Ethernet port housing on the Go2 was damaged in an earlier incident (a forward-flip during `KP_FR_LIFT=80` Jacobian-PID tuning) and is currently held by tape. Connection is stable but treat as a known fragility.

### 2.3 Camera

- **Source:** Go2 onboard front-facing camera (1920 × 1080).
- **API:** `unitree_sdk2py.go2.video.video_client.VideoClient`. Wrapped in `Go2Camera` (`src/perception/go2_camera.py`).
- **Two intrinsics profiles** in `src/perception/grounding.py`:
  - `calib_2026_04` — calibrated `plumb_bob` model, K and D from a checkerboard session in April 2026 (recommended for all current models trained with `intrinsics_version_filter="calib_2026_04"`).
  - `urdf_legacy` — pinhole approximation derived from URDF (HFOV=120°, no distortion). Used by older v3-era models if loaded with `--intrinsics urdf_legacy`.

```
calib_2026_04:
  K = [[1310.778,    0.000, 1018.711],
       [   0.000, 1320.251,  637.377],
       [   0.000,    0.000,    1.000]]
  D = [-0.415971, 0.158898, -0.015395, -0.008031, 0.000]   # plumb_bob

urdf_legacy:
  fx = fy = (1920/2) / tan(60°) ≈ 554.3
  cx = 960, cy = 540
  D = zeros(5)
```

- **Camera extrinsics** (camera frame → robot base, from URDF, in `grounding.py`):
  - Translation: forward `+0.327` m, lateral `0.000` m, height `+0.043` m above base link.
  - Rotation: `R_cam_to_base = [[0,0,1],[-1,0,0],[0,-1,0]]` (camera Z forward, X right, Y down → base X forward, Y left, Z up).

### 2.4 Microphone

- USB microphone, device index passed via `--mic-index` (current sessions used **11**, an earlier session reportedly used 5 — verify with `python -c "import sounddevice as sd; print(sd.query_devices())"`).
- **Capture rate:** 48 kHz; resampled to 16 kHz for Whisper.
- **Calibration:** 2 s of silent baseline at session start; threshold = baseline + 20 dB (configurable via `--audio-threshold-db`).
- **Detection:** 50 ms sustained above-threshold RMS, debounced. See §13.4 for algorithm details.
- **Live signal:** `AudioLiveDetector.sound_detected` (bool), polled at 500 Hz inside `hold` phase as the **primary contact signal**.
- **Recording:** simultaneous `AudioRecorder` saves the full mic stream to `data/eval/audio/<trial_id>.wav` for Whisper transcription post-trial.

### 2.5 Battery considerations

The Go2's battery droop becomes load-bearing during long sessions, especially with the rear-leg-loaded press posture. Practical effects:

- After ~30–45 min of active use, FR press depth degrades visibly (actuators near voltage-limited torque ceiling).
- Recommend swapping battery between evaluation blocks.
- We currently do **not** log battery voltage in the CSV. A simple `gh` issue: add `battery_v` from `low_state` to the CSV per trial — recommended before the final eval run.

### 2.6 Conda environments

- `env_go2` — primary deployment environment. Python 3.11, Torch 2.7.0 + CUDA 12.8, RTX 5080 GPU.
- `env_isaaclab` — separate, unused for this project's deployment work (Isaac Sim install).

Setup: `pip install -r requirements.txt`. PyTorch must be installed *first* with CUDA 12.8 wheels:

```bash
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
```

GroundingDINO weights are at `~/Robotics/weights/groundingdino_swint_ogc.pth`; Depth Anything V2 weights at `~/Robotics/weights/depth_anything_v2_metric_hypersim_vitl.pth`.

---

## 3. Critical Constants

All values verbatim from source (extracted via direct code inspection on 2026-04-30). Joint indices follow Unitree Go2 SDK ordering.

### 3.1 Joint ordering (Unitree Go2 SDK)

```
0  = FR_hip      1  = FR_thigh     2  = FR_calf
3  = FL_hip      4  = FL_thigh     5  = FL_calf
6  = RR_hip      7  = RR_thigh     8  = RR_calf
9  = RL_hip      10 = RL_thigh     11 = RL_calf
```

Confirmed in `heuristic_contact.py`: `FR_HIP=0, FR_THIGH=1, FR_CALF=2`. **Isaac Lab joint ordering** (which is `(FL, FR, RL, RR)` per group) is **not** used in this codebase — all references are SDK-native.

### 3.2 Control timing

```python
CTRL_DT = 0.002                # 500 Hz
STEPS_SIT_TO_STAND      = 1000 # 2.0 s
STEPS_WEIGHT_SHIFT      = 800  # 1.6 s
STEPS_LIFT              = 600  # 1.2 s
STEPS_EXTEND            = 600  # 1.2 s
STEPS_HOLD              = 500  # 1.0 s post-contact
STEPS_RETRACT_CURL      = 400  # 0.8 s
STEPS_RETRACT_ROTATE    = 600  # 1.2 s
STEPS_RETRACT_EXTEND    = 400  # 0.8 s
STEPS_WEIGHT_SHIFT_BACK = 800  # 1.6 s
STEPS_SETTLE            = 500  # 1.0 s
STEPS_LOWER_TO_SIT      = 1500 # 3.0 s
GATE_TIMEOUT            = 500  # +1.0 s before forced advance
GATE_THRESHOLD          = 0.15 # rad
CONTACT_PROXIMITY_M     = 0.06 # FK contact threshold
CONTACT_MAX_STEPS       = 6000 # hard cap on hold (12 s)
```

### 3.3 PD gains (across configurations)

```python
KP_STABLE = 100.0            KD_STABLE = 6.0       # default support
KP_FR     = 40.0             KD_FR     = 3.0       # FR (non-compliant)
KP_FR_COMPLIANT = 15.0       KD_FR_COMPLIANT = 1.5 # FR during extend+hold (guided)
KP_SUPPORT_SOFT = 55.0       KD_SUPPORT_SOFT = 5.5 # support during extend+hold (wholebody)
KP_SOFT   = 25.0             KD_SOFT   = 3.0       # used at lower_to_sit
```

Note: `KP_SUPPORT_SOFT` defaults to `55.0` in `wholebody.py`, matching the v3 collection regime. **For collection** (`scripts/collect_wholebody.py`) the default was lowered to `35.0` after gravity FF was added (see §6, §9).

### 3.4 Reference poses

```python
STAND_POS = [
    -0.021, +0.667, -1.359,   # FR  (hip, thigh, calf)
    +0.025, +0.666, -1.368,   # FL
    -0.078, +0.661, -1.348,   # RR
    +0.072, +0.660, -1.347,   # RL
]

SIT_POS = [
    -0.086, +1.230, -2.769,   # FR
    +0.108, +1.230, -2.776,   # FL
    -0.423, +1.243, -2.763,   # RR
    +0.418, +1.247, -2.776,   # RL
]

# Standard 3-leg tripod (FR airborne)
WEIGHT_SHIFT_POS = STAND_POS, with these deltas:
    FL_hip   -= 0.30   (idx 3)
    RL_hip   -= 0.25   (idx 9)
    RR_hip   += 0.15   (idx 6)
    FL_thigh -= 0.10   (idx 4)
    RL_thigh -= 0.10   (idx 10)

# Alternative "handshake-style" tripod (used by HeuristicContactWholeBody
# weight_shift_pos_override path; not the deployment default)
WEIGHT_SHIFT_POS_HANDSHAKE:
    FL_hip   = -0.225      FL_thigh = +0.667     FL_calf  = -1.048
    RR_hip   = -0.281      RR_thigh = +0.965     RR_calf  = -1.674
    RL_hip   = -0.262      RL_thigh = +1.079     RL_calf  = -1.948
```

### 3.5 FR leg fallback offsets (used when Stage C waypoints unavailable)

```python
# Wall press
FR_LIFT_OFFSET_WALL   = [+0.04, -2.0, +0.10]   # ∆ from STAND_POS for FR
FR_EXTEND_OFFSET_WALL = [+0.04, -2.0, +0.52]

# Ground press
FR_LIFT_OFFSET_GROUND  = [+0.04, -0.8, +0.1]
FR_PRESS_OFFSET_GROUND = [+0.04, -1.0, +0.8]
FR_RETRACT_CURL_CALF_OFFSET = -0.5
```

These are baseline_1's hardcoded fallback. Stage C overrides them in `baseline_2` and `core_method`.

### 3.6 Standoff geometry — **values differ between scripts**

This is a known inconsistency, see §10.1.

```python
# scripts/run_methods.py (deployment)
PRESS_OFFSET_X      = 0.203
PRESS_OFFSET_Y      = 0.140
NAV_EXTRA_FORWARD_M = 0.15
# Formula: dx = target_x − PRESS_OFFSET_X − NAV_EXTRA_FORWARD_M
#          (subtracts NAV_EXTRA, value 0.15)

# scripts/collect_wholebody.py (collection)
PRESS_OFFSET_X      = 0.593
PRESS_OFFSET_Y      = 0.0
NAV_EXTRA_FORWARD_M = 0.20
# Formula: standoff_dx = target_x − PRESS_OFFSET_X + NAV_EXTRA_FORWARD_M + perturbation
#          (adds NAV_EXTRA, value 0.20)
```

History (from `log` file, project notes, Summary 6):

| When | PRESS_OFFSET_X | PRESS_OFFSET_Y | NAV_EXTRA_FORWARD_M | Notes |
|---|---|---|---|---|
| Session 20260424 | 0.593 | -0.047 | 0.10 | First successful episode (only one without v2.1 metadata) |
| v3 collection (Apr 26-29) | 0.593 | 0.0 | 0.10–0.20 | Per-mode: stabilize=0.15, translate=0.10, drift-adapted=0.15 |
| Deployment (current) | 0.203 | 0.140 | 0.15 | run_methods.py — values look like a refactor; net dx is ~equivalent in practice but the press_offset semantics differ |

**Both scripts produce a similar net forward dx in practice** (collect: target − 0.393; run: target − 0.353), but the constants are not interpretable in a single consistent frame. **Verify before any future re-tune.**

### 3.7 Camera calibration

(See §2.3 for K and D matrices.) `CAMERA_INTRINSICS_VERSION_CALIB = "calib_2026_04"`.

### 3.8 Gravity FF

```python
GO2_NOMINAL_MASS_KG = 15.0
GO2_GRAVITY = 9.81
SUPPORT_THIGH_FF_BASELINE_NM = 3.0   # empirical thigh constant

GRAVITY_FF_DEFAULT_PHASES = (
    "lift", "extend", "hold",
    "retract_curl", "retract_rotate", "retract_extend",
)

# Within _per_leg_gravity_ff():
L_CALF = 0.213   # m

# Static handshake-FF capture (NOT used dynamically — preserved as reference)
# from src/planner/heuristic_contact_wholebody.py
SUPPORT_GRAVITY_FF = {
    3: (-0.20, +3.08, +9.70),  # FL  (hip, thigh, calf) in N·m
    6: (-0.92, +2.89, +6.89),  # RR
    9: (+2.60, +3.49, +8.51),  # RL
}
```

Live deployment uses the **dynamic** gravity FF (analytical calf via `J^T·F_grav` plus empirical thigh constant), not the static `SUPPORT_GRAVITY_FF` table. The static table is a debug reference captured during a one-time Sport Mode handshake-capture run; it is preserved in code but never called.

### 3.9 Foot-force contact (currently disabled)

```python
FOOT_FORCE_CONTACT_THRESHOLD = 10.0   # raw units — TO BE CALIBRATED
FR_FOOT_FORCE_IDX = 0   # index into low_state.foot_force for FR
```

The foot-force sensor on the Go2 was unreliable during initial testing; threshold not calibrated. To re-enable: pass `use_foot_force=True` to `execute()` and calibrate threshold against a baseline measured with FR airborne.

---

## 4. Deployment Commands (the ones that work)

### 4.1 Best-validated demo (single-step v5)

This is the configuration that produced the most consistent successes during the 2026-04-29 test session.

```bash
python scripts/run_methods.py \
  --variant core_method \
  --stage-c-ckpt-dir models/stage_c_v5 \
  --stage-d-ckpt models/stage_d_v5/stage_d.pt \
  --stage-d-device cpu \
  --gravity-ff \
  --no-compliance \
  --residual-scale 0.5 \
  --mic-index 11
```

Why each flag:
- `--stage-d-device cpu` — frees the GPU for `GroundingThread` (GroundingDINO + SAM2 + Depth Anything are GPU-bound; contention with Stage D causes 500 Hz control jitter).
- `--gravity-ff` — dynamic FF on FL/RR/RL during airborne FR phases (the fix for static rear sag, see §8.1).
- `--no-compliance` — FR uses KP_FR=40 (not KP_FR_COMPLIANT=15) during press; matches deployment of a non-guided press.
- `--residual-scale 0.5` — half-magnitude residual; full scale was visibly destabilising rear legs under battery droop.

### 4.2 Chunked variant (when v6 chunked is validated)

```bash
python scripts/run_methods.py \
  --variant core_method \
  --stage-c-ckpt-dir models/stage_c_v5 \
  --stage-d-ckpt models/stage_d_v5/stage_d.pt \
  --use-chunked \
  --chunked-checkpoint models/stage_d_v6_chunked/stage_d.pt \
  --chunk-size 25 \
  --ensemble-decay 0.1 \
  --fallback-checkpoint models/stage_d_v5/stage_d.pt \
  --stage-d-device cpu \
  --gravity-ff \
  --no-compliance \
  --residual-scale 0.5 \
  --mic-index 11
```

Critical: **`--chunk-size` must equal the trained checkpoint's `chunk_size`** (currently 25 for `stage_d_v6_chunked`). Mismatch silently triggers fallback to single-step v5 — see §8.7.

### 4.3 Baseline_1 (heuristic standoff + hardcoded waypoints)

```bash
python scripts/run_methods.py \
  --variant baseline_1 \
  --gravity-ff \
  --mic-index 11
```

No Stage C or Stage D loaded. Standoff via `compute_heuristic_standoff()` (PRESS_OFFSET_X − NAV_EXTRA_FORWARD). FR waypoints from `FR_LIFT_OFFSET_WALL` and `FR_EXTEND_OFFSET_WALL`.

### 4.4 Baseline_2 (Stage C standoff + Stage C waypoints, no residual)

```bash
python scripts/run_methods.py \
  --variant baseline_2 \
  --stage-c-ckpt-dir models/stage_c_v5 \
  --gravity-ff \
  --mic-index 11
```

Tests whether learned planning alone is sufficient; isolates Stage D's contribution against the otherwise-identical pipeline.

### 4.5 Moving-target demo (Jacobian tracking with re-lift)

```bash
python scripts/run_methods.py \
  --variant core_method \
  --stage-c-ckpt-dir models/stage_c_v5 \
  --stage-d-ckpt models/stage_d_v5/stage_d.pt \
  --stage-d-device cpu \
  --gravity-ff \
  --no-compliance \
  --residual-scale 0.5 \
  --track-target-during-hold \
  --track-target-mode jacobian \
  --track-target-relift-threshold 0.03 \
  --mic-index 11
```

Tracking and re-lift fire only with `core_method` (need contact regrounding). For the demo: have a confederate slide the button laterally during the press attempt; the foot will pull back up and re-press at the new spot when lateral move > 3 cm.

---

## 5. CLI Flag Reference

### 5.1 `scripts/run_methods.py`

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--variant` | str | (required) | `baseline_1` \| `baseline_2` \| `core_method` |
| `--prompt` | str | `"press the red button"` | |
| `--interface` | str | `enx98fc84e68f1a` | |
| `--stage-c-ckpt-dir` | str | `models/stage_c` | dir containing `stage_c.pt` |
| `--stage-d-ckpt` | str | `models/stage_d_combined/stage_d.pt` | single-step path |
| `--output-dir` | str | `data/eval` | |
| `--mic-index` | int | None | mandatory for live audio detection in practice |
| `--audio-threshold-db` | float | `20.0` | dB above baseline RMS |
| `--stage-c-fallback` | flag | False | force heuristic standoff |
| `--waypoints-fallback` | flag | False | force hardcoded waypoints |
| `--notes` | str | `""` | written to `operator_notes` CSV column |
| `--whisper-model-path` | str | None | override faster-whisper model path |
| `--residual-fr-only` | flag | False | zero Stage D residual on rear legs |
| `--rear-kp` | float | None | override `KP_SUPPORT_SOFT` for core_method |
| `--rear-kd` | float | None | override `KD_SUPPORT_SOFT` for core_method |
| `--gravity-ff` | flag | False | enable dynamic gravity FF on FL/RR/RL |
| `--no-compliance` | flag | False | disable FR softening on core_method |
| `--residual-scale` | float | `1.0` | scalar applied to Stage D residual |
| `--no-contact-regrounding` | flag | False | suppress 5 Hz GroundingThread |
| `--stage-d-device` | str | `cuda` | choices: `cuda`, `cpu` |
| `--use-chunked` | flag | False | use chunked Stage D |
| `--chunked-checkpoint` | str | None | required if `--use-chunked` |
| `--chunk-size` | int | `25` | **must match checkpoint** |
| `--ensemble-decay` | float | `0.1` | exponential decay for chunked ensemble |
| `--fallback-checkpoint` | str | None | single-step fallback if chunked fails |
| `--track-target-during-hold` | flag | False | enable closed-loop target tracking |
| `--track-target-mode` | str | `stage_c` | choices: `stage_c`, `jacobian` |
| `--track-target-relift-threshold` | float | `0.0` | re-lift on lateral shift > X m (jacobian only) |
| `--no-save-grounding-frames` | flag | False | disable frame capture for demos |
| `--intrinsics` | str | `calib_2026_04` | choices: `calib_2026_04`, `urdf_legacy` |

### 5.2 `scripts/collect_wholebody.py`

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--interface` / `--network-interface` | str | `enx98fc84e68f1a` | |
| `--mic-index` | int | (required) | |
| `--audio-threshold-db` | float | `20.0` | |
| `--button-height` | float | `0.25` | for metadata only (m) |
| `--distance` | float | `0.75` | from robot start to button (m) |
| `--angle` | float | `0.0` | yaw (rad) |
| `--num-episodes` | int | `5` | |
| `--perturbation` | str | `none` | only `none` is implemented |
| `--cardboard-test` | flag | False | first-time-on-new-gain-schedule mode (caps at 2 eps, loud warning) |
| `--rear-kp` | float | `35.0` | `kp_support_soft` for collection — lower than 55 because gravity FF carries the static torque |
| `--rear-kd` | float | `3.5` | `kd_support_soft` |
| `--gravity-ff` | flag | True | default ON for collection |
| `--ff-body-mass` | float | `15.0` | |
| `--verbose` | flag | False | |

### 5.3 `scripts/train_stage_d.py`

| Flag | Default | Notes |
|---|---|---|
| `--data-dirs` | `data/real/stage_d_v2` | comma-separated |
| `--format-filter` | None | `v2` \| `v3` |
| `--epochs` | `50` | |
| `--batch-size` | `256` | |
| `--lr` | `1e-3` | |
| `--weight-decay` | `1e-4` | |
| `--joint-weights` | `5,5,5,0,0,0,0,0,0,0,0,0` | 12 comma-separated floats |
| `--val-fraction` | `0.2` | |
| `--seed` | `42` | |
| `--out-dir` | `models/stage_d` | |
| `--device` | `auto` | |
| `--phases` | `0,1,2` | comma-separated phase ids (0=lift, 1=extend, 2=hold) |
| `--no-save` | False | |
| `--gain-schedule-filter` | None | exact-string match against root attr `gain_schedule` |
| `--intrinsics-version-filter` | None | exact-string match against `camera_intrinsics_version` |
| `--chunk-size` | `1` | `1` = single-step (legacy); `K>1` trains the chunked policy |

### 5.4 `scripts/train_stage_c.py`

Same shape as `train_stage_d.py` but no `joint-weights`, no `phases`, no `chunk-size`. Defaults: `epochs=200`, `batch-size=8`, `data-dirs=data/real/stage_d_v2,data/real/stage_d_v3`. Stage C trains on the same HDF5s as Stage D — only `phase_transitions` and `standoff_dxdy_commanded` are used.

### 5.5 `scripts/eval_stage_d.py` and `scripts/eval_stage_d_chunked.py`

Same as train scripts minus optimizer flags. Chunked version adds `--chunk-size` (override the checkpoint's value if needed; mismatched values raise a hard error).

### 5.6 `scripts/recompute_failure_modes.py`

| Flag | Default |
|---|---|
| `--data-dir` | `data/eval` |
| `--dry-run` | False |
| `--force-success TRIAL_ID` | (repeatable) — manually override a row to success and append a note |

### 5.7 `scripts/verify_calibration.py`

| Flag | Default |
|---|---|
| `--positions` | `3` |
| `--frames` | `10` |
| `--warmup` | `5` |
| `--prompt` | `"red button"` |
| `--output-dir` | `data/calibration_check/<UTC timestamp>` |
| `--no-save-frames` | False |

### 5.8 Filter mechanisms (all train/eval scripts)

`--gain-schedule-filter` and `--intrinsics-version-filter` filter HDF5 episodes by their root-level string attributes. These let you train a model on a homogeneous slice (e.g., only `wholebody_v2_kpsupport35` + `calib_2026_04`) without renaming files. Episodes lacking the attribute (older v2 collection) are excluded when a filter is set.

---

## 6. Data Collection Regime History

The training data set under `data/real/stage_d_v3/` is a **mixed** collection — episodes were recorded under several gain schedules and intrinsics versions, distinguished by HDF5 root attributes. Filtering at train time is the mechanism for clean regime-specific models.

| Regime | Episodes | KP_FR | KP_SUPPORT during press | Gravity FF? | Intrinsics | gain_schedule attr | Notes |
|---|---|---|---|---|---|---|---|
| **v2 (FR-only)** | ~15 | 15 (compliant) | 100 | – | unknown / urdf_legacy | `fr_only_compliant` | Hand-guided FR only; rear at full stiffness |
| **v3 first attempt** | 5 | 15 | 45 | – | urdf_legacy | `wholebody_v1_kpsupport45` | Rear sag observed, excluded from training, preserved at `data/real/stage_d_v3_first/` (TODO: verify path still exists) |
| **v3 main** | ~30 | 15 | 55 | – | urdf_legacy | `wholebody_v2_kpsupport55` | Demo modes: stabilize-only (NAV=0.15), translate-mode (NAV=0.10), stabilize+backward (NAV=0.15, drift-adapted) |
| **v4 / FF-stable** | (uncertain count) | 15 | 35 | ✓ | calib_2026_04 | (likely) `wholebody_v2_kpsupport35` | First collection where dynamic gravity FF carried the rear torque — KP_SUPPORT lowered because PD no longer needs to fight gravity; intrinsics calibrated for the first time |
| **v5** | 24 added 2026-04-29 | 15 | 35 | ✓ | calib_2026_04 | `wholebody_v2_kpsupport35` | Same regime as v4 plus position diversity. Three blocks at button heights 0.43 m, 0.55 m, 0.50 m floor (TODO: verify; from `log` file). Recorded with the final HDF5 schema |

**Note:** Counts above are best-guess from `data/real/stage_d_v3/`'s contents (43 files total, episode ID timestamps spanning 2026-04-28 to 2026-04-29) and the project notes file. Models trained with `gain-schedule-filter=wholebody_v2_kpsupport35` + `intrinsics-version-filter=calib_2026_04` see the v4 + v5 union.

The **v6_chunked** model is a chunked-architecture retrain on the same v4+v5 union (same filters, K=25). All current Stage C/D checkpoints (`stage_c_v4`, `stage_c_v5`, `stage_d_v4*`, `stage_d_v5*`, `stage_d_v6_chunked`) train on `data/real/stage_d_v3` with these filters.

A separate `data/real/stage_d_v5/` directory exists but is empty as of 2026-04-30 (TODO: confirm intended use — possibly a future home for v5-only episodes).

---

## 7. Trained Model Checkpoints

Under `models/`:

| Checkpoint | Type | Joint weights | Filters | Status / role |
|---|---|---|---|---|
| `stage_c_v4` | Stage C (5→{3,9}) | n/a | wholebody_v2_kpsupport35 + calib_2026_04 | Earlier Stage C baseline |
| `stage_c_v5` | Stage C | n/a | same | **Current deployment Stage C** |
| `stage_d_v4` | Stage D single-step | `[5,5,5,1,1,1,1,1,1,1,1,1]` | same | Original v4 weights |
| `stage_d_v4_3_1` | Stage D single-step | `[3,3,3,1,1,1,1,1,1,1,1,1]` | same | Softer FR weighting (better rear behavior) |
| `stage_d_v4_uniform` | Stage D single-step | `[1,1,1,1,1,1,1,1,1,1,1,1]` | same | Ablation — equal weights |
| `stage_d_v5` | Stage D single-step | `[3,3,3,1,1,1,1,1,1,1,1,1]` | same | **Current deployment Stage D** (single-step) |
| `stage_d_v5_51` | Stage D single-step | `[5,5,5,1,1,1,1,1,1,1,1,1]` | same | v5 retrained with 5/1 weights |
| `stage_d_v5_uni` | Stage D single-step | `[1,1,1,1,1,1,1,1,1,1,1,1]` | same | v5 uniform-weight ablation |
| `stage_d_v6_chunked` | Stage D **chunked**, K=25 | `[3,3,3,1,1,1,1,1,1,1,1,1]` | same | **Current chunked candidate** (validated offline; not deployment-validated as of 2026-04-30) |

All are 50 epochs, batch 256, lr=1e-3, weight decay=1e-4, val_fraction=0.2 seed=42, on `data/real/stage_d_v3`.

### 7.1 Recommended deployment

For the Friday demo: `stage_c_v5` + `stage_d_v5` (single-step). `stage_d_v6_chunked` is a fallback if single-step shows control jitter that chunking would smooth — but verify with a smoke trial first; it has not been deployment-validated.

### 7.2 Headline offline metrics

`models/stage_d_v6_chunked/eval.json` reports per-joint val MSE (16906 val samples, 9 held-out episodes, K=25):

| Joint | val MSE |
|---|---|
| FR_hip   | 1.52e-05 |
| FR_thigh | 2.17e-04 |
| FR_calf  | 2.09e-04 |
| FL_hip   | 4.06e-05 |
| FL_thigh | 9.20e-05 |
| FL_calf  | 2.61e-05 |
| RL_hip   | 5.48e-05 |
| RL_thigh | 5.40e-05 |
| RL_calf  | 2.39e-05 |
| RR_hip   | 6.11e-05 |
| RR_thigh | 6.38e-05 |
| RR_calf  | 2.00e-05 |

These are very small values — well-fit on the train distribution but with the spatial-sensitivity caveat for Stage C noted in §8.5.

---

## 8. Failure Modes Characterized

### 8.1 PD-without-FF static sag (irreducible by gain alone)

With KP_SUPPORT=55, KD_SUPPORT=5.5, the support legs alone cannot carry the body's rear weight statically once FR is extended forward. Steady-state error: `e_ss = τ_required / KP`. For a 15 kg robot with FR airborne and CoM shifted, required calf torque is ≳ 9 N·m per leg, giving `e_ss ≳ 0.16 rad` — visible as ~1 cm of rear drop over 1–2 s. Fix: dynamic gravity FF supplies the torque; PD only provides damping and tracks operator-induced motion. This is **why** `KP_SUPPORT_SOFT` could be lowered from 55 to 35 in collection.

### 8.2 Sport Mode handshake FF capture (preserved as reference)

In an early debug run, we captured per-leg motor torques during a Sport Mode handshake (which holds the body stable while one leg is airborne) by averaging `motor_state[i].tau_est` over a few seconds at known body posture. The result is the static `SUPPORT_GRAVITY_FF` table in `wholebody.py`. This table is **not** used at runtime — the analytical+empirical dynamic FF is — but the table is preserved as a reference to compare future analytical FF formulations against.

### 8.3 GPU contention between perception and Stage D

`GroundingThread` runs GroundingDINO + SAM2 + Depth Anything on CUDA at 5 Hz. Each detection takes ~150–250 ms wall time. When Stage D inference also runs on CUDA, the per-step `predict()` call competes for the same GPU and intermittently blocks past the 2 ms control-loop budget — visible as "one-damp" shakes at phase transitions, and as Stage D residual being applied stale. **Fix:** `--stage-d-device cpu`. CPU forward pass on the (33→128×3→12) MLP is ~1–2 ms — still within budget, plus the GroundingThread now has the GPU to itself.

### 8.4 Camera FOV bias at high button heights

Worst Stage D val errors clustered at `foot_to_target ≈ [0.74, -0.13, 0.015]` (far forward, slightly off-axis). This corresponds to the tail of the training distribution — high buttons close to the FOV edge after standoff. Stage D has not seen those geometries enough to predict precise residuals; it tends to under-press. Mitigations: collect at varied button heights (Priority 1 in Summary 6, partly addressed in v5), or use a wider-FOV camera / camera tilt.

### 8.5 Stage C v5 spatial-sensitivity collapse

Empirically, Stage C v5's waypoint head exhibits near-zero spatial sensitivity: shifting `target_pos_base` by 5 cm produces only ~0.17° change in predicted FR pose. Training data was concentrated near a single button position, so the model effectively memorised one set of waypoints. **Implication:** `--track-target-mode stage_c` for moving-target tracking is a no-op in practice on v5; use `--track-target-mode jacobian` (analytical) instead. **Fix for the future:** collect at more diverse target positions and retrain.

### 8.6 Battery droop affecting deployment

The Go2's battery sags ~10–15% in voltage over a multi-hour session. With KP_FR=40 and full Stage D residual, the actuator hits its voltage-limited torque ceiling and fails to reach the press depth (visible as "FR foot stops 1–2 cm above the button"). Mitigations: `--residual-scale 0.5`, `--no-compliance` (stiffer FR), swap battery between blocks, log battery in CSV (recommended TODO).

### 8.7 Silent fallback with mismatched chunk_size

`StageDChunkedInference` catches load failures and silently delegates to a fallback single-step model when `--fallback-checkpoint` is set. If the user passes `--chunk-size 10` against a checkpoint with K=25, the load raises `ValueError` and the runner falls back without aborting. The CSV row says `core_method` but the trial actually ran on the single-step model. Always check the WARNING line:

```
Chunked checkpoint at <path> failed to load (<reason>); falling back to single-step <fallback>.
```

Recommended hardening: have `run_methods.py` log `is_fallback` after construction and refuse to start if a chunked variant unexpectedly fell back.

### 8.8 Demonstrator stochasticity

Hand-guided demos vary by ±cm on FR placement and ±deg on body posture between operators. Stage D's per-step residuals therefore include demonstrator noise as well as state-dependent corrections. With single-step inference the model can amplify this noise into 500 Hz jitter; chunked + temporal ensembling smooths it. (This is the main motivation for the chunked variant.)

### 8.9 Multi-session perception drift

Across the v3 collection sessions (~2 hours each), `standoff_x` drifted from ~0.55 m to ~0.50 m at the same physical button position. Likely cause: thermal or lighting shift in Depth Anything V2's prediction. Operators adapted by adjusting `NAV_EXTRA_FORWARD_M` — this contributed to the mode mixture in the v3 dataset (S, T, S- modes; see Summary 6 §3.5).

### 8.10 NAV_EXTRA_FORWARD_M sign mismatch

`run_methods.py` and `collect_wholebody.py` apply NAV_EXTRA_FORWARD_M with **opposite signs** in their standoff formulas (see §3.6). The net dx is nearly equivalent in practice, but the constants are not interpretable in a single consistent frame. Verify before any future re-tune. (Listed as TODO in §10.)

---

## 9. Engineering Decisions and Rationale

### 9.1 Dynamic gravity FF over static table

The static `SUPPORT_GRAVITY_FF` table was captured at a specific body posture (Sport Mode handshake stand). At any other posture (e.g., weight-shifted tripod with FR forward-extended), the per-leg gravity load differs. Dynamic FF computes the calf torque analytically from the current joint configuration via `J^T·F_grav`, plus an empirical thigh constant — adapts across postures without retraining or recapturing. Used for FL/RR/RL during airborne FR phases (FR is excluded because its torque is set by the press goal, not gravity).

### 9.2 Stage D on CPU

Stage D is a small MLP (33→128×3→12, ~50 k params). CPU forward is ~1–2 ms — fits within the 2 ms control budget. Running on GPU contends with the perception thread (see §8.3), and the absolute speedup is negligible at this model size. **Decision: default to CPU when GroundingThread is active.**

### 9.3 Joint weights `[3,3,3,1,...]` (softer than original `[5,5,5,1,...]`)

The original Stage D training spec called for `[5,5,5,0,...]` (FR-only) or `[5,5,5,1,...]` (full body with strong FR emphasis). Empirically, the 5x weighting on FR caused the rear-leg residuals to under-correct; lowering to 3x balanced the loss enough that rear-leg predictions tracked their distribution better (cf. `stage_d_v5_51` vs `stage_d_v5` offline metrics). Current default: `[3,3,3,1,1,1,1,1,1,1,1,1]`.

### 9.4 `--residual-scale 0.5` as the chunked-ensemble default

Even with temporal ensembling, the chunked model occasionally produces 1–2× larger residuals than the demonstrator did at any single step (when several future-step predictions all align high). Halving the residual at deployment empirically eliminated rear-leg shake without compromising FR press depth. The Stage D model does not need to be retrained; this is a deployment-time scaler.

### 9.5 `--no-compliance` for deployment

Compliance (KP_FR=15) was a *training-time* affordance for hand-guided collection — it let the operator move the FR paw without fighting the motor. In deployment there is no operator on the paw, so KP_FR=15 just produces a soft, sluggish press that misses the target depth under battery droop. `--no-compliance` reverts FR to KP_FR=40 during press, which is what the Stage D residual was actually trained against (since training data was recorded with the *commanded* targets at compliant gains, the achieved deltas are post-compliance — but the *deployment* gain difference matters when scaling residuals).

### 9.6 Why not always run with contact regrounding?

The 5 Hz GroundingThread runs heavy perception models. It contends for GPU with Stage D, and if grounding goes stale (false negative or low confidence), the cached `target_pos_base` may go stale too. For diagnostic isolation we keep `--no-contact-regrounding` available so we can characterise the system at fixed standoff. In normal operation (especially for moving-target demos), regrounding is on.

### 9.7 Yaw alignment at standoff — why it matters and why it's not implemented

The standoff offset Stage C predicts is `(dx, dy, dyaw)` — but the controller currently only commands `dx, dy`. Yaw alignment matters because the FR foot's reach is sensitive to body yaw (a 5° yaw error displaces the foot by ~3 cm at full reach). We don't implement it because Stage C v5's `dyaw` predictions are noisy (training data did not include systematic yaw variation), and the robot's odometry from `move_to_position` accumulates yaw error over a session anyway. Future work: collect at varied body yaws and add a `dyaw` correction step.

### 9.8 Stage D didn't switch to ACT-style fully

We considered porting Stage D to a full ACT-style architecture (encoder + transformer + CVAE for action chunks). The chunked MLP was chosen as a step-toward variant that reuses the existing data pipeline. ACT would likely improve performance but requires substantial training-loop rewrites and didn't fit the demo timeline.

---

## 10. Known Issues / TODOs

1. **NAV_EXTRA_FORWARD_M sign mismatch** between `run_methods.py` (`-`) and `collect_wholebody.py` (`+`), with also-different `PRESS_OFFSET_X` (0.203 vs 0.593) and `PRESS_OFFSET_Y` (0.140 vs 0.0). Reconcile before any re-tune.
2. **`CONTACT_MAX_STEPS=6000` (12 s) may be too long** in casual deployment but **too short** for chunked Stage D under battery droop, where press descent can be slow. Worth instrumenting per-trial press time.
3. **Stage D was not trained on intentional press-depth motion.** Training data is hand-bound, so the residual learned is mostly stabilization, not active descent. Future work: collect with intentional press-depth variation.
4. **`stage_d_v6_chunked` not validated at deployment** as of 2026-04-30. Run a smoke trial before relying on it for the demo.
5. **Static `SUPPORT_GRAVITY_FF` table is dead code** (preserved as historical reference). Decide whether to delete or document.
6. **Foot-force sensor threshold** uncalibrated. Disabled by default. Re-enable would require a new calibration session.
7. **`data/real/stage_d_v5/`** is an empty directory. Either populate or delete.
8. **Battery voltage not logged** per trial in CSV. Recommended addition before final eval.
9. **`run_methods.py` does not log `is_fallback`** for chunked Stage D. Silent fallback can masquerade as a chunked-model trial. (See §8.7.)
10. **Sign reconciliation:** within `run_methods.py`, `compute_heuristic_standoff` uses `- NAV_EXTRA_FORWARD_M`, which makes it **closer** to the button than the bare press-offset standoff. Verify this is intended; collection script subtracts press-offset and *adds* nav-extra (going farther forward). The semantics are not matched.

---

## 11. Repo Layout

```
Language_Conditioned_UniGo2/
├── HANDOFF.md                  ← this file
├── README.md
├── environment.yaml            ← refers to env_isaaclab; deployment uses env_go2
├── requirements.txt
├── setup.py
├── robot_connection            ← Ethernet bring-up cheatsheet (read me for setup)
├── log                         ← scratch notebook of per-session collection notes
├── preliminary_train_log.md    ← Apr 26 training results dump
├── camera_test.jpg             ← single test capture, debug artifact
├── docs/
│   ├── training_data_spec.md           ← v2.1 (superseded)
│   └── training_data_spec_v3.md        ← current spec
├── Implementation_summaries/
│   ├── Summary1.md … Summary6.md       ← session-by-session writeups
│   └── Summary4.docx
├── configs/                    ← (uncertain content — not currently used by runtime)
├── sim/                        ← Isaac Lab sim wiring (not used for deployment)
├── tests/                      ← unit tests (smoke)
├── src/
│   ├── audio/                  ← (TODO: verify — likely audio modules)
│   ├── control/                ← (uncertain)
│   ├── data/
│   │   ├── audio_live_detector.py      ← live RMS detection
│   │   ├── audio_recorder.py           ← stream→WAV
│   │   ├── color_detector.py           ← faster-whisper post-hoc
│   │   ├── dataset.py                  ← StageD samples + STATE_DIM
│   │   ├── grounding_thread.py         ← 5 Hz background grounding
│   │   └── recorder.py                 ← StageDRecorder writes HDF5 episodes
│   ├── language/               ← (Stage A — TODO: verify content)
│   ├── models/
│   │   ├── stage_c.py                  ← dual-head MLP
│   │   └── stage_d.py                  ← single-step + chunked policy + bundle (de)serialisation
│   ├── perception/
│   │   └── grounding.py                ← VisualGrounder + INTRINSICS_PROFILES
│   ├── planner/
│   │   ├── heuristic_contact.py        ← base state machine (1774 lines)
│   │   ├── heuristic_contact_corrective.py  ← Jacobian-PID variant (deprecated)
│   │   ├── heuristic_contact_guided.py      ← FR softening for hand demo
│   │   ├── heuristic_contact_wholebody.py   ← +support softening +gravity FF wiring
│   │   └── heuristic_policy.py
│   ├── policy/
│   │   ├── stage_c_runtime.py          ← StageCInference (predict_standoff/waypoints)
│   │   └── stage_d_runtime.py          ← StageDInference + StageDChunkedInference
│   └── robot/                  ← Go2Interface wrapper around unitree_sdk2py
├── scripts/
│   ├── run_methods.py                  ← THE deployment runner
│   ├── collect_wholebody.py            ← demo episode recorder
│   ├── train_stage_c.py
│   ├── train_stage_d.py
│   ├── eval_stage_c.py
│   ├── eval_stage_d.py
│   ├── eval_stage_d_chunked.py
│   ├── recompute_failure_modes.py      ← retro CSV failure-mode fix
│   ├── verify_calibration.py           ← intrinsics regression check
│   ├── inspect_v4_calibration.py
│   ├── calibrate_wall_standoff.py
│   ├── capture_grounding_debug.py
│   ├── debug_regrounding.py
│   ├── debug_whisper.py
│   ├── measure_standoff.py
│   ├── print_pose.py
│   ├── smoke_audio_modules.py
│   ├── smoke_test_sim.py
│   ├── summarize_dataset.py
│   ├── test_*.py                       ← assorted unit-test entry points
│   └── validate_episode*.py            ← HDF5 schema validators
├── models/
│   ├── stage_c_v4/  stage_c_v5/
│   ├── stage_d_v4/  stage_d_v4_3_1/  stage_d_v4_uniform/
│   ├── stage_d_v5/  stage_d_v5_51/   stage_d_v5_uni/
│   ├── stage_d_v6_chunked/
│   └── old_models/                     ← archive
└── data/
    ├── real/
    │   ├── stage_d_v3/                 ← 43 training episodes (v4+v5 union)
    │   ├── stage_d_v5/                 ← (currently empty — TODO: verify intended use)
    │   └── old_data/
    ├── eval/
    │   ├── eval_<timestamp>.csv        ← one CSV per launch of run_methods.py (one or many trials)
    │   ├── audio/<trial_id>.wav        ← per-trial audio
    │   └── grounding/<trial_id>_*.jpg  ← per-trial frame captures
    ├── calibration/                    ← intrinsics calibration artifacts
    ├── calibration_check/              ← verify_calibration.py outputs (timestamped folders)
    ├── debug/
    ├── measurements/
    └── synthetic/                      ← Isaac Sim outputs (unused at deploy)
```

---

## 12. Operating Procedures

### 12.1 Pre-session bring-up

```bash
# 1. Bring up Ethernet to the robot
sudo ip addr flush dev enx98fc84e68f1a
sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
sudo ip link set enx98fc84e68f1a up
ping -c 3 192.168.123.161

# 2. Activate environment
conda activate env_go2

# 3. Verify mic index
python -c "import sounddevice as sd; print(sd.query_devices())"
# Note the index of the USB mic; pass via --mic-index
```

### 12.2 Verify camera calibration before a session

```bash
python scripts/verify_calibration.py \
  --positions 3 --frames 10 \
  --prompt "red button" \
  --output-dir data/calibration_check/$(date -u +%Y%m%dT%H%M%SZ)
```

This drives the robot to 3 fixed standoff positions, grounds the button at each, and dumps annotated frames for visual inspection. A successful run shows the projected centroid landing on the button in all annotated frames.

### 12.3 Cardboard sanity test

When changing the gain regime (e.g., updating `kp_support_soft`), use `--cardboard-test`:

```bash
python scripts/collect_wholebody.py \
  --cardboard-test --num-episodes 2 \
  --mic-index 11 --rear-kp 35 --rear-kd 3.5 --gravity-ff
```

This caps the run at 2 episodes and prints a loud warning to remind operators to test on a soft surface (a piece of cardboard) before trusting the new gain set on the real button. Useful for catching rear-sag or overshoot before committing to a session.

### 12.4 Inspection after a collection block

```bash
python scripts/inspect_v4_calibration.py    # checks calibration drift
python scripts/summarize_dataset.py         # episode counts, gain-schedule mix
python scripts/validate_episode.py <h5>     # schema check
```

### 12.5 Filtering training data by regime

To train only on the FF-stable v4+v5 union:

```bash
python scripts/train_stage_d.py \
  --data-dirs data/real/stage_d_v3 \
  --gain-schedule-filter wholebody_v2_kpsupport35 \
  --intrinsics-version-filter calib_2026_04 \
  --joint-weights 3,3,3,1,1,1,1,1,1,1,1,1 \
  --out-dir models/stage_d_<your_label>
```

Same recipe for `train_stage_c.py` and the eval scripts.

### 12.6 Retroactive CSV failure-mode fix

If old eval CSVs contain rows mislabelled `failure_mode=miss` despite `success_audio_live=True ∧ color=red`, the classifier was buggy. Run:

```bash
python scripts/recompute_failure_modes.py [--dry-run]
```

Optionally force-success specific known-good trials whose detectors silently failed:

```bash
python scripts/recompute_failure_modes.py \
  --force-success 20260429_222106_core_method \
  --force-success 20260428_174015_baseline_2
```

---

## 13. Internals

### 13.1 Controller hierarchy

```
HeuristicContact   (base, src/planner/heuristic_contact.py)
├── HeuristicContactCorrective  (Jacobian-PID corrective; DEPRECATED)
└── HeuristicContactGuided      (FR softening for hand demo)
    └── HeuristicContactWholeBody  (+ support softening + gravity FF wiring)
```

#### Base `HeuristicContact`

Carries the full state machine, gain-application loop, gravity-FF, target tracking, and Stage D residual application. Key state:

- `self._compliance_active: bool` — flipped True by guided variants. Gates the FR gain swap to compliant values during extend+hold inside `_send_cmd()`.
- `self._gravity_ff_enabled: bool` and `self._gravity_ff_phases: Set[str]`.
- `self._stage_d` — inference handle (single-step or chunked).
- `self._stage_d_residual_mask`, `self._stage_d_residual_scale`.
- `self._fr_waypoints: Dict[str, np.ndarray]` (`lift`, `extend`, `press`).
- `self._phase_transitions: dict` — `{lift_end, extend_end, hold_end}` written at gate-passed transitions, used by Stage C training labels and demo logging.
- `self._stage_d_target_cache: np.ndarray` — last valid target_pos_base (never NaN).
- `self._track_target_*` family — closed-loop tracking state (mode, threshold, last_tracked).

The control loop (`_control_loop`, runs at 500 Hz):
1. Increment phase_step.
2. `_maybe_track_moving_target()` (no-op unless tracking enabled).
3. Read `actual` joints, IMU, foot force under `_state_lock`.
4. Phase logic computes `target_q`.
5. If active phase and Stage D enabled, `_apply_stage_d_residual(target_q, ...)` adds the residual.
6. `_send_cmd(target_q)` — fills `motor_cmd` with KP/KD per gain-schedule, applies gravity FF via `_maybe_apply_gravity_ff()`, publishes.

#### `HeuristicContactGuided`

Overrides nothing in the control loop; just sets `self._compliance_active = True` and spawns a `_phase_watcher_thread` that prints operator-facing messages (`[GUIDED] Compliance active`, `[GUIDED] Contact detected`). Class attribute `collection_mode = "hand_guided"`.

#### `HeuristicContactWholeBody`

Inherits from `HeuristicContactGuided`. Adds `kp_support_soft` and `kd_support_soft` ctor params. Overrides `_send_cmd()` to extend the soft-gain branch to support legs (FL, RL, RR) during extend+hold. Sets `self.gain_schedule = f"wholebody_v2_kpsupport{int(kp_support_soft)}"` for HDF5 metadata. `_maybe_apply_gravity_ff()` is implemented in the base class but only fires when `gravity_ff_enabled=True`.

#### `HeuristicContactCorrective`

The original Jacobian-PID corrective variant. Extends `HeuristicContact` (not Guided). Overrides `_control_loop` to compute and apply a Jacobian-PID FR delta during corrective phases, with proportional gain `k_p`. **Deprecated** in favour of Stage D residuals — preserved in the repo for reference but not part of the deployment path.

### 13.2 GroundingThread

`src/data/grounding_thread.py`. Daemon thread polling `camera.get_frame()` at `rate_hz` (default 5 Hz) and calling `grounder.ground(frame, prompt)`. Emits `GroundingSample(timestamp, position_base, confidence, depth_m, valid)`. Stores latest **valid** position in `self._latest_position` (thread-safe). All samples (valid or not) appended to `self._log`. Public:

- `start()` / `stop()` — lifecycle.
- `get_latest_position() -> Optional[np.ndarray]` — last good (3,) base-frame position, or None if grounding has been failing.
- `get_log() -> list[GroundingSample]` — full history for HDF5 dump.

### 13.3 StageDInference vs StageDChunkedInference

Both expose `predict(state_33: np.ndarray) -> np.ndarray (12,)`.

**`StageDInference`** — stateless. Pre-allocated `(1, 33)` GPU tensor reused across calls. One forward pass per step, ~1 ms on CPU.

**`StageDChunkedInference`** — stateful.
- Constructor signature: `(checkpoint_path, chunk_size=25, decay_rate=0.1, device="cpu", fallback_to_single_step=False, fallback_checkpoint_path=None)`.
- Maintains `self._chunk_buffer: deque(maxlen=chunk_size)` of `(predicted_at_step, chunk: (K, 12))`. NaN chunks are refused (not inserted), so a single bad prediction does not poison the ensemble.
- Per `predict()` call: predicts a fresh chunk, inserts (if finite), then returns a weighted average over alive past chunks: `Σ w_i · chunk_i[offset_i] / Σ w_i`, where `w_i = exp(-decay_rate · offset_i)` and `offset_i = step_counter - predicted_at_i`.
- On first step (no alive chunks), returns zero residual rather than NaN.
- **`reset()`** clears the buffer and `_step_counter`. Called automatically by `_apply_stage_d_residual` on phase change (cf. `heuristic_contact.py:1200`).
- **Fallback** — if `load_stage_d_chunked_bundle()` raises and `fallback_to_single_step=True`, the instance silently delegates all `predict()` calls to a wrapped `StageDInference`. `is_fallback` property exposes this for diagnostics. **Caveat:** the runner does not log this state at startup beyond the WARNING at construction time (see §8.7, §10.9).

### 13.4 AudioLiveDetector

`src/data/audio_live_detector.py`. Streams 48 kHz mic via `sounddevice.InputStream`, blocks of `block_duration_ms=20` ms.

Algorithm in the per-block callback:
```python
rms = sqrt(mean(samples**2))
db_above = 20 * log10(rms / baseline_rms)
if db_above > threshold_db:        # default 20 dB
    high_energy_ms += block_duration_ms
    if high_energy_ms >= min_duration_ms:    # default 50 ms
        sound_detected = True
        detection_time_s = time.time() - episode_start_time
else:
    high_energy_ms = 0
```

`baseline_rms` is computed from a 2 s silence calibration (`baseline_duration_s=2.0`) at session start. `sound_detected` is **never reset** automatically — call `start_episode()` between trials to reset (cleared in our codebase by `start_episode()` returning, plus a defensive clear by `_rewind_hold_to_lift()` for the re-lift path).

### 13.5 AudioRecorder + ColorDetector

`AudioRecorder` runs a parallel `sounddevice.InputStream` (or shares one via `aux_callback`) and accumulates the mono float32 stream during a trial. `end_episode()` returns the concatenated waveform.

`ColorDetector` wraps faster-whisper. `detect(audio, sample_rate)` returns `{"color_detected": <str or None>, "transcript": str, "success_audio_transcript": bool}`. Transcribe call uses `language="en"`, `beam_size=5`, `temperature=0.0`, no VAD, with an `initial_prompt` listing valid colors. First color word from `VALID_COLORS = ["red", "blue", "green", "yellow", "orange"]` found in the transcript wins.

### 13.6 CSV schema (one row per trial, written by `run_methods.py`)

```
trial_id, variant, prompt,
target_x, target_y, target_z,
standoff_x, standoff_y, standoff_z,
standoff_dx_cmd, standoff_dy_cmd,
standoff_source, fr_waypoints_source,
duration_s,
contact_method, success_fk, success_audio_live, audio_detection_time_s,
color_detected, color_expected, success_target,
failure_mode, operator_kept, operator_notes,
audio_path, stage_c_checkpoint, stage_d_checkpoint, git_sha,
stage_d_residual_mode, rear_kp_used, rear_kd_used, tau_scheme,
compliance_mode, stage_d_residual_scale, contact_regrounding,
stage_d_device, camera_intrinsics_version
```

Truth metric for thesis: **`success_target = success_audio_live AND (color_detected == "red")`**. This combines a physical contact signal (the button's onboard sound module fired) with a language match (Whisper picked up "red" in the recording — the button's audio output speaks the color name). Raw `color_detected="red"` over-counts because lab speech can contaminate; `success_fk` undercounts because silent presses fail FK proximity.

### 13.7 HDF5 episode schema

`src/data/recorder.py`. One file per episode under `data/real/stage_d_v3/episode_wholebody_<timestamp>.h5`.

**Root attributes:**
- `success_target: bool`, `episode_id: str`, `duration_s: float`
- `target_pos_base_at_standoff: (3,) float32`, `standoff_dxdy_commanded: (3,) float32`
- `press_offset_x, press_offset_y, nav_extra_forward_m: float32`
- `y_sit_stand_drift_comp: float`
- `contact_step: int`, `contact_method: str`, `press_mode: str` ("wall" or "ground")
- `target_offset_xyz: (3,) float32`
- `collection_mode: str` ("hand_guided" or "wholebody_guided")
- `gain_schedule: str` (v3 only — e.g., "wholebody_v2_kpsupport55")
- `camera_K: (3, 3) float64`, `camera_D: (5,) float64`, `camera_intrinsics_version: str`
- `audio_sample_rate: int` (if `/audio` dataset present)

**Group `/per_step` (T-length per-step arrays):**
- `timestamp: (T,) float64`
- `joint_pos_actual, joint_vel_actual, joint_pos_cmd: (T, 12) float32`
- `joint_kp_used: (T, 12) float32` (v3 only)
- `joint_tau_est: (T, 12) float32`
- `fr_foot_pos_fk: (T, 3) float32`
- `target_pos_base: (T, 3) float32` (NaN where grounding lapsed)
- `foot_to_target_error: (T, 3) float32` (target − fk; NaN where target was NaN)
- `imu_rpy: (T, 3) float32`, `imu_gyro: (T, 3) float32`, `imu_accel: (T, 3) float32`
- `phase_label: (T,) int8` (0=lift, 1=extend, 2=hold)
- `phase_progress: (T,) float32` ∈ [0, 1]
- `nominal_waypoint_fr: (T, 3) float32`
- `jacobian_pid_delta: (T, 3) float32` (zero for non-corrective variants)
- `achieved_delta_fr: (T, 3) float32` (training label, last row duplicates t-1)

`achieved_delta_q[t] = joint_pos_actual[t+1] - joint_pos_cmd[t]` (12-dim) is **computed on load**, not stored.

**Group `/phase_transitions`:**
- `lift_end, extend_end, hold_end: (12,) float32` — actual joint pose at gate-passed transitions. **Required** for Stage D training.

**Group `/grounding` (optional):**
- `timestamp, position_base, confidence, depth_m, position_base_valid` — full 5 Hz log.

**Dataset `/audio`** (optional): mono float32 waveform.

### 13.8 Stage D state vector (33 dims)

Built per step in `dataset.py:_build_state_33d`:

| Slice | Field | Source |
|---|---|---|
| `[0:3]` | `fr_joint_pos` | `joint_pos_actual[t, 0:3]` |
| `[3:6]` | `fr_joint_vel` | `joint_vel_actual[t, 0:3]` |
| `[6:9]` | `foot_to_target_error` | `target_pos_base[t] − fr_foot_pos_fk[t]` |
| `[9:12]` | `nominal_waypoint_fr` | current heuristic FR target |
| `[12]` | `phase_progress` | `[0, 1]` within current phase |
| `[13:15]` | `imu_rp` | roll, pitch (yaw dropped) |
| `[15:27]` | `joint_tau_est` | all 12 motor torques |
| `[27:30]` | `imu_gyro` | body angular velocity |
| `[30:33]` | `imu_accel` | body linear acceleration |

Z-score normalised at training time; mean/std bundled in the checkpoint. The runtime wrapper applies the same normalisation before forward.

### 13.9 Hardware idiosyncrasies

- **Sport Mode SDK quirks.** The `unitree_sdk2py` Sport Mode is request-response (`StandUp`, `StandDown`, `BalanceStand`, `ReleaseMode`). Our flow does `StandUp → BalanceStand` for navigation, then `StandDown → ReleaseMode` to drop into low-level control. Calling Sport Mode APIs while low-level is active will silently fail or clobber the gain schedule — order matters.
- **`SelectMode timed out` warning** at retract is benign — Sport Mode wasn't ready to take over after our low-level session; the recovery path uses the controller and cleanup proceeds.
- **Foot-force sensor unreliability.** Not calibrated. Default disabled.
- **Ethernet incident.** A previous Jacobian-PID tuning attempt at `KP_FR_LIFT=80` caused a forward flip; the Ethernet port housing is taped. Avoid abrupt FR torques in lift phase. Now mitigated by gain caps and gravity FF.
- **Brake / whine regimes.** The Go2 actuators emit a high-pitched whine when commanded position is held for long durations against gravity (especially at compliance gains). This is a hardware behaviour, not a control issue. Reducing `STEPS_HOLD` is the only mitigation if it bothers operators during demos.

### 13.10 Workarounds for SDK quirks

- The control loop uses `last_target_q` to resend the previous command if `low_state` is momentarily stale (subscriber GIL-blocked). Without this, a 2 ms torque dropout at 500 Hz produces visible "one damp" shakes at phase transitions.
- `Go2Interface.move_to_position` short-circuits if `|dx, dy| < tolerance` (the "Already within position tolerance" log line) to avoid sending zero-displacement Sport Mode commands that would reject.
- `SetSitDown` is wrapped to first call `SwitchMode(0)` to drop out of low-level.

---

## 14. Demo Plan (Friday 2026-05-01)

### 14.1 Primary configuration

```bash
python scripts/run_methods.py \
  --variant core_method \
  --stage-c-ckpt-dir models/stage_c_v5 \
  --stage-d-ckpt models/stage_d_v5/stage_d.pt \
  --stage-d-device cpu \
  --gravity-ff \
  --no-compliance \
  --residual-scale 0.5 \
  --mic-index 11
```

Stable, validated in the 2026-04-29 session. Most consistent press successes against the known v5-trained button position.

### 14.2 Why this configuration

- v5 single-step is the most validated Stage D checkpoint. v6 chunked is offline-strong but not deployment-validated as of 2026-04-30; relying on it would be a risk.
- `--no-compliance` is the right gain regime for unattended press (no operator on the paw).
- `--residual-scale 0.5` survives battery droop better than full magnitude.
- `--stage-d-device cpu` keeps GroundingThread responsive.
- `--gravity-ff` is non-negotiable — without it, support legs sag during press.

### 14.3 Optional demo extension: moving target

Add `--track-target-during-hold --track-target-mode jacobian --track-target-relift-threshold 0.03` and have a confederate slide the button laterally during the press attempt. Demonstrates contact-time regrounding closing the loop.

### 14.4 Fallback options

| Symptom during demo | Fallback |
|---|---|
| Stage D destabilising rear legs (visible rocking) | Add `--residual-fr-only` (zero rear residual) |
| FR not reaching button (battery droop) | Add `--rear-kp 35` for lighter rear, swap battery |
| GroundingThread starves control loop (intermittent jitter) | Add `--no-contact-regrounding` (loses moving-target capability but stabilises) |
| Stage C waypoints look off | Add `--waypoints-fallback` (reverts to hardcoded `FR_LIFT_OFFSET_WALL` etc.) |
| Stage C standoff dy looks off | Add `--stage-c-fallback` (uses heuristic standoff) |
| Whole pipeline misbehaving | Drop to `--variant baseline_2` to test if Stage C alone reaches the button |
| Catastrophic | Drop to `--variant baseline_1` (heuristic + hardcoded everything) |

### 14.5 Trial cadence

- Aim for 5–10 trials per variant for a statistical comparison.
- ~3 minutes per trial including operator notes.
- Plan ~30 min per variant per battery, with 1 battery swap for a 4-variant evaluation.
- Verify mic baseline calibration silently at the start of each variant block (re-launch script).

---

## 15. Future Work Directions

### 15.1 Immediate (could improve current results)

1. **Yaw alignment at standoff.** Stage C predicts `dyaw` but it's not commanded. Adding it would reduce a known reach error.
2. **Wider-FOV camera or camera tilt.** High buttons fall near the FOV edge after standoff and degrade Depth Anything's accuracy. A second camera or a brief downward tilt during approach would address this.
3. **Multi-session collection** to get true session-diversity in Stage C / Stage D training data. The current dataset is 2 sessions; cross-session generalisation is not characterised.
4. **Battery voltage logging in CSV.** Trivial change. Would let you slice success rate by battery state and confirm/refute the droop hypothesis.

### 15.2 Architectural

5. **ACT-style chunking with CVAE.** The current chunked Stage D is an MLP that outputs flat `K · 12` values. ACT's transformer + CVAE encoder would condition on a learned action latent, smoothing demonstrator stochasticity more effectively. Significant rewrite.
6. **Foot-force sensor recalibration** for foot-force contact detection. Currently disabled because uncalibrated. Calibrating against an FR-airborne baseline + an FR-grounded baseline would re-enable a third independent contact signal.
7. **Dynamic FF with full-body inertial model** (instead of per-leg analytical). Would handle dynamic motions (e.g., during weight_unshift) where the current static-posture assumption breaks down.

### 15.3 Data collection

8. **Press-depth variation in collection.** Current demos are hand-bound; the demonstrator controls press depth implicitly. Explicitly varying press depth during collection would give Stage D direct training signal for active descent.
9. **Position diversity in Stage C training.** v5 is concentrated near a single button position. A position-diverse retrain would unlock real Stage C-based moving-target tracking (currently only Jacobian works).
10. **Push interaction.** Original proposal included push as a second task; only press is collected. The dataset and architecture support a push interaction (one-hot encoded as `[0, 1]`), but no episodes have been collected.

### 15.4 Reproducibility

11. **Reconcile NAV_EXTRA_FORWARD_M sign and PRESS_OFFSET semantics** between collect and run scripts (§10.1). Add a unit test that asserts the two scripts produce the same standoff target for the same target_pos_base, given matched flag values.
12. **Decommission dead code paths.** `HeuristicContactCorrective`, the static `SUPPORT_GRAVITY_FF` table, and the empty `data/real/stage_d_v5/` directory should each be either documented as historical-reference or deleted.

---

## Appendix A: Quick-reference flag combinations

| What you want to do | Add these flags |
|---|---|
| Run the headline core_method | `--variant core_method --stage-d-device cpu --gravity-ff --no-compliance --residual-scale 0.5 --mic-index 11` |
| Compare against Stage C alone | `--variant baseline_2 --gravity-ff --mic-index 11` |
| Compare against fully-scripted | `--variant baseline_1 --gravity-ff --mic-index 11` |
| Use chunked Stage D | `--use-chunked --chunked-checkpoint <path> --chunk-size 25 --ensemble-decay 0.1 --fallback-checkpoint <single-step path>` |
| Only learned residual on FR | `--residual-fr-only` |
| Half-magnitude residual | `--residual-scale 0.5` |
| Stiffer rear during press | `--rear-kp 75 --rear-kd 7.5` |
| Disable contact regrounding | `--no-contact-regrounding` |
| Run Jacobian moving-target tracking | `--track-target-during-hold --track-target-mode jacobian` |
| Re-lift on lateral target jump | `--track-target-relift-threshold 0.03` |
| Force urdf_legacy intrinsics for old models | `--intrinsics urdf_legacy` |

---

## Appendix B: Pre-trial checklist

1. Robot battery > 30 V (TODO: confirm threshold) — swap if lower.
2. Ethernet bring-up (`§12.1`).
3. Mic index verified.
4. `git status` clean (commits the trial against a known git_sha; written to CSV).
5. `data/eval/` writable.
6. Camera calibration verified within the session if drift suspected (`§12.2`).
7. Workspace clear: no obstacles within 1 m radius of the standoff position.
8. Two operators present for any guided trial; one for `run_methods.py`.

---

*End of handoff document.*
*Generated 2026-04-30. Reference inputs: `src/planner/heuristic_contact.py`, `src/planner/heuristic_contact_wholebody.py`, `src/perception/grounding.py`, `src/data/{audio_live_detector,audio_recorder,color_detector,dataset,grounding_thread,recorder}.py`, `src/models/{stage_c,stage_d}.py`, `src/policy/{stage_c_runtime,stage_d_runtime}.py`, `scripts/{run_methods,collect_wholebody,train_stage_c,train_stage_d,eval_stage_c,eval_stage_d,eval_stage_d_chunked,recompute_failure_modes,verify_calibration}.py`, `models/*/config.json`, `docs/training_data_spec_v3.md`, `Implementation_summaries/Summary6.md`, `log`, `robot_connection`.*
