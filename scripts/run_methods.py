"""
scripts/run_methods.py

Friday demo orchestrator — one trial per invocation across three method
variants under a controlled evaluation protocol.

Variant table:

    baseline_1   Heuristic standoff,  hardcoded waypoints,        HeuristicContact
    baseline_2   Stage C standoff,    Stage C waypoints,          HeuristicContact
                 + 2-shot approach re-grounding
    core_method  Stage C standoff,    Stage C waypoints,          HeuristicContactWholeBody
                 + 2-shot approach re-grounding,
                 + 5 Hz contact-time re-grounding feeding Stage D residual

CLI:

    python scripts/run_methods.py \\
        --variant {baseline_1,baseline_2,core_method} \\
        --prompt "press the red button" \\
        --interface enx98fc84e68f1a \\
        --stage-c-ckpt-dir models/stage_c_v5 \\
        --stage-d-ckpt models/stage_d_v5/stage_d.pt \\
        --output-dir data/eval \\
        --mic-index 6 \\
        [--stage-c-fallback] [--waypoints-fallback] [--notes "..."]

Outputs:

    data/eval/eval_<invocation_timestamp>.csv  per-invocation CSV (one row).
    data/eval/audio/<trial_id>.wav             raw mic capture (always written).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

# Third-party
from scipy.io import wavfile

# Project
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.robot.go2_interface import Go2Interface
import cv2

from src.perception import grounding as _grounding_module
from src.perception.grounding import (
    Go2Camera,
    VisualGrounder,
    _annotate as _grounding_annotate,
)
from src.planner.heuristic_contact import HeuristicContact
from src.planner.heuristic_contact_wholebody import HeuristicContactWholeBody
from src.data.audio_recorder import AudioRecorder
from src.data.audio_live_detector import AudioLiveDetector
from src.data.color_detector import ColorDetector
from src.data.grounding_thread import GroundingThread

from src.policy.stage_c_runtime import StageCInference
from src.policy.stage_d_runtime import StageDChunkedInference, StageDInference

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_INTERFACE = "enx98fc84e68f1a"
DEFAULT_PROMPT    = "press the red button"
DEFAULT_OUTPUT    = "data/eval"

# Wall-press standoff geometry collect wholebody
PRESS_OFFSET_X      = 0.593
PRESS_OFFSET_Y      = 0
NAV_EXTRA_FORWARD_M = 0.15

# Audio.
AUDIO_SAMPLE_RATE   = 16000
AUDIO_CAPTURE_RATE  = 48000
AUDIO_BASELINE_S    = 2.0

# Grounding.
MIN_GROUNDING_CONF  = 0.5
GROUND_RETRY_MAX    = 3
GROUND_RETRY_SLEEP  = 0.5
CONTACT_GROUNDING_HZ = 5.0

EXPECTED_COLOR = "red"


# ──────────────────────────────────────────────
# Variant config
# ──────────────────────────────────────────────

VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline_1": {
        "use_stage_c_standoff":    False,
        "use_stage_c_waypoints":   False,
        "controller_class":        HeuristicContact,
        "stage_d_ckpt":            None,
        "approach_regrounding":    False,
        "contact_regrounding":     False,
    },
    "baseline_2": {
        "use_stage_c_standoff":    True,
        "use_stage_c_waypoints":   True,
        "controller_class":        HeuristicContact,
        "stage_d_ckpt":            None,
        "approach_regrounding":    True,
        "contact_regrounding":     False,
    },
    "core_method": {
        "use_stage_c_standoff":    True,
        "use_stage_c_waypoints":   True,
        "controller_class":        HeuristicContactWholeBody,
        "stage_d_ckpt":            "models/stage_d_combined/stage_d.pt",
        "approach_regrounding":    True,
        "contact_regrounding":     True,
    },
}


# ──────────────────────────────────────────────
# CSV row schema
# ──────────────────────────────────────────────

CSV_FIELDS = [
    "trial_id", "variant", "prompt",
    "target_x", "target_y", "target_z",
    "standoff_x", "standoff_y", "standoff_z",
    "standoff_dx_cmd", "standoff_dy_cmd",
    "standoff_source",
    "fr_waypoints_source",
    "duration_s",
    "contact_method",
    "success_fk",
    "success_audio_live",
    "audio_detection_time_s",
    "color_detected",
    "color_expected",
    "success_target",
    "failure_mode",
    "operator_kept",
    "operator_notes",
    "audio_path",
    "stage_c_checkpoint",
    "stage_d_checkpoint",
    "git_sha",
    "stage_d_residual_mode",
    "rear_kp_used",
    "rear_kd_used",
    "tau_scheme",
    "compliance_mode",
    "stage_d_residual_scale",
    "contact_regrounding",
    "stage_d_device",
    "camera_intrinsics_version",
]


@dataclass
class TrialOutcome:
    """Single CSV row payload. Defaults are valid even for crashed trials."""
    trial_id: str = ""
    variant: str = ""
    prompt: str = ""
    target_x: float = float("nan")
    target_y: float = float("nan")
    target_z: float = float("nan")
    standoff_x: float = float("nan")
    standoff_y: float = float("nan")
    standoff_z: float = float("nan")
    standoff_dx_cmd: float = float("nan")
    standoff_dy_cmd: float = float("nan")
    standoff_source: str = ""
    fr_waypoints_source: str = ""
    duration_s: float = float("nan")
    contact_method: str = "none"
    success_fk: bool = False
    success_audio_live: bool = False
    audio_detection_time_s: float = float("nan")
    color_detected: str = ""
    color_expected: str = EXPECTED_COLOR
    success_target: bool = False
    failure_mode: str = "unknown"
    operator_kept: bool = False
    operator_notes: str = ""
    audio_path: str = ""
    stage_c_checkpoint: str = ""
    stage_d_checkpoint: str = ""
    git_sha: str = ""
    stage_d_residual_mode: str = "n/a"
    rear_kp_used: str = "n/a"
    rear_kd_used: str = "n/a"
    tau_scheme: str = "n/a"
    compliance_mode: str = "n/a"
    stage_d_residual_scale: str = "n/a"
    contact_regrounding: str = "n/a"
    stage_d_device: str = "n/a"
    camera_intrinsics_version: str = ""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_heuristic_standoff(target_pos_base: np.ndarray) -> np.ndarray:
    """Wall-press heuristic standoff from training_data_spec / Summary 4.

    Returns an (3,) ``[dx, dy, dyaw]`` offset suitable for
    ``Go2Interface.move_to_position``.
    """
    dx = float(target_pos_base[0]) - PRESS_OFFSET_X 
    dy = float(target_pos_base[1]) - PRESS_OFFSET_Y
    return np.array([dx, dy, 0.0], dtype=np.float32)


def _save_grounding_artifacts(
    frame: np.ndarray,
    result,
    prompt: str,
    save_stem: Path,
) -> None:
    """Write a raw + annotated JPG pair next to each successful grounding.

    ``save_stem`` is a path *prefix* (no extension); the function appends
    ``_raw.jpg`` and ``_annotated.jpg``. Demo-friendly visuals: the
    annotated frame carries the SAM2 mask overlay, GroundingDINO bbox,
    centroid dot, and a metadata strip (prompt, confidence, depth, base
    coords) — same renderer the grounding CLI uses.
    """
    try:
        save_stem.parent.mkdir(parents=True, exist_ok=True)
        raw_path = save_stem.with_name(save_stem.name + "_raw.jpg")
        cv2.imwrite(str(raw_path), frame)
        if result is not None:
            annotated = _grounding_annotate(frame, result, prompt)
            ann_path = save_stem.with_name(save_stem.name + "_annotated.jpg")
            cv2.imwrite(str(ann_path), annotated)
    except Exception as e:
        logger.warning(f"Failed to save grounding frame {save_stem}: {e}")


def _try_ground(
    camera: Go2Camera,
    grounder: VisualGrounder,
    prompt: str,
    *,
    retries: int = GROUND_RETRY_MAX,
    min_conf: float = MIN_GROUNDING_CONF,
    save_stem: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Take up to ``retries`` frames, return first valid ``position_base``.

    When ``save_stem`` is provided, write the raw + annotated frames of
    the *successful* grounding to ``<stem>_raw.jpg`` and
    ``<stem>_annotated.jpg``. Failed attempts are not saved.
    """
    for attempt in range(1, retries + 1):
        frame = camera.get_frame()
        if frame is None:
            logger.warning(f"camera.get_frame() returned None (attempt {attempt})")
            time.sleep(GROUND_RETRY_SLEEP)
            continue
        result = grounder.ground(frame, prompt=prompt)
        if (result is None
                or result.position_base is None
                or result.confidence is None
                or result.confidence < min_conf):
            logger.warning(
                f"Grounding attempt {attempt}/{retries} failed "
                f"(conf={getattr(result, 'confidence', None)})"
            )
            time.sleep(GROUND_RETRY_SLEEP)
            continue
        if save_stem is not None:
            _save_grounding_artifacts(frame, result, prompt, save_stem)
        return np.asarray(result.position_base, dtype=np.float32).copy()
    return None


def _classify_failure(
    *,
    success_fk: bool,
    audio_detected: bool,
    color_detected: Optional[str],
    timeout_phase: Optional[str],
) -> str:
    """Auto-classify a trial's failure mode.

    A trial is "successful" (failure_mode == "") if any contact signal
    fired (FK proximity OR live audio) AND Whisper transcribed the
    expected color word. Earlier versions only counted FK as "real
    success", which mis-labelled audio-driven presses as ``miss``; that
    bug is fixed here.
    """
    contact_succ = bool(success_fk or audio_detected)
    color_match = (color_detected == EXPECTED_COLOR)
    color_other = bool(color_detected) and not color_match  # heard a non-red color word

    if contact_succ and color_match:
        return ""
    if audio_detected and color_other:
        return "wrong_color"
    if timeout_phase is not None:
        return "timeout"
    return "miss"


def _operator_prompt(auto_label: str) -> Tuple[bool, Optional[str], str]:
    """Return (kept, mode_override, notes)."""
    print(f"\nOutcome auto-classified as: {auto_label or 'success'}")
    print("Operator decision? [k=keep, d=discard, f=fall, "
          "a=approach_failure, q=abort]")
    try:
        key = input("> ").strip().lower()
    except EOFError:
        key = ""
    if   key == "k":
        kept, override = True,  None
    elif key == "d":
        kept, override = False, None
    elif key == "f":
        kept, override = True,  "fall"
    elif key == "a":
        kept, override = True,  "approach_failure"
    elif key == "q":
        kept, override = False, "operator_abort"
    else:
        logger.warning(f"Unrecognized operator input {key!r} — treating as discard.")
        kept, override = False, None

    try:
        notes = input("Notes (one line, empty OK): ").strip()
    except EOFError:
        notes = ""
    return kept, override, notes


def _save_audio(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio is None or audio.size == 0:
        logger.warning(f"Audio buffer empty — writing zero-length WAV at {path}")
        audio = np.zeros(0, dtype=np.float32)
    wavfile.write(str(path), sample_rate, audio.astype(np.float32))
    logger.info(f"Audio → {path}")


def _append_csv_row(csv_path: Path, outcome: TrialOutcome) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    row = asdict(outcome)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    logger.info(f"CSV row appended → {csv_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Friday demo: run one trial of a method variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", required=True, choices=list(VARIANTS.keys()),
                   help="Method variant to run.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Language prompt for visual grounding.")
    p.add_argument("--interface", default=DEFAULT_INTERFACE,
                   help="DDS Ethernet interface name.")
    p.add_argument("--stage-c-ckpt-dir", default="models/stage_c_v5",
                   help="Directory containing stage_c.pt.")
    p.add_argument("--stage-d-ckpt", default="models/stage_d_v5/stage_d.pt",
                   help="Path to Stage D bundle checkpoint.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                   help="Where to write CSV + audio.")
    p.add_argument("--mic-index", type=int, default=None,
                   help="USB mic device index.")
    p.add_argument("--audio-threshold-db", type=float, default=20.0,
                   help="AudioLiveDetector threshold above baseline RMS (dB).")
    p.add_argument("--stage-c-fallback", action="store_true",
                   help="Force heuristic standoff even when variant uses Stage C.")
    p.add_argument("--waypoints-fallback", action="store_true",
                   help="Force hardcoded waypoints even when variant uses Stage C.")
    p.add_argument("--notes", default="",
                   help="Free-form notes appended to the CSV row.")
    p.add_argument("--whisper-model-path", default=None,
                   help="Optional override for the faster-whisper model path.")
    p.add_argument(
        "--residual-fr-only",
        action="store_true",
        help="Zero Stage D residual on rear legs (indices 3:12). "
             "Diagnostic flag. Only meaningful for core_method.",
    )
    p.add_argument(
        "--rear-kp",
        type=float,
        default=None,
        help="Override KP_SUPPORT_SOFT for soft-gain support legs in core_method "
             "(default: 55.0). Only meaningful for core_method; warning emitted "
             "and value ignored for baseline_1 and baseline_2.",
    )
    p.add_argument(
        "--rear-kd",
        type=float,
        default=None,
        help="Override KD_SUPPORT_SOFT for soft-gain support legs in core_method "
             "(default: 5.5). Only meaningful for core_method; warning emitted "
             "and value ignored for baseline_1 and baseline_2.",
    )
    p.add_argument(
        "--gravity-ff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply dynamic gravity-compensation feedforward torque to support "
             "legs (FL, RR, RL) during the weight-bearing tripod phases "
             "(lift, extend, hold, retract_*). On by default (paper config); "
             "pass --no-gravity-ff to disable. Implemented in the base "
             "HeuristicContact class so it's available for every variant.",
    )
    p.add_argument(
        "--compliance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compliance mode on core_method (FR uses soft KP_FR_COMPLIANT "
             "and support legs use soft KP_SUPPORT_SOFT during extend+hold). "
             "OFF by default (paper config: --no-compliance was used so FR uses "
             "KP_FR=40 and support legs use KP_STABLE=100 — matches baseline_2's "
             "gain regime). Pass --compliance to re-enable. Stage C waypoints, "
             "Stage D residual, and contact regrounding remain active regardless. "
             "No effect on baseline_1/baseline_2 (compliance is structurally off).",
    )
    p.add_argument(
        "--residual-scale",
        type=float,
        default=0.5,
        help="Multiply Stage D residual by this scalar before applying. "
             "Default 0.5 (paper config — full-magnitude residual was found to "
             "be destabilizing on hardware). Set to 1.0 to apply the unscaled "
             "residual. Only meaningful for core_method; ignored for "
             "baseline_1 and baseline_2.",
    )
    p.add_argument(
        "--no-contact-regrounding",
        action="store_true",
        help="Suppress the 5 Hz GroundingThread for core_method. Stage D's "
             "foot_to_target_error input falls back to the cached standoff "
             "target for the contact phase. Used for diagnostic — the "
             "GroundingThread runs GPU-bound perception (GroundingDINO + "
             "SAM2 + Depth Anything) that contends with Stage D's per-step "
             "predict() on the same GPU and can starve the 500 Hz control "
             "loop. Only meaningful for core_method; no-op otherwise.",
    )
    p.add_argument(
        "--stage-d-device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device for Stage D inference. Default 'cpu' (paper config — frees "
             "the GPU for the contact-time GroundingThread, which runs "
             "GroundingDINO + SAM2 + Depth Anything; CPU forward pass is "
             "~1-2 ms, well within the 2 ms 500-Hz control budget). Pass "
             "'cuda' to keep Stage D on the GPU. Only meaningful for variants "
             "that load Stage D (core_method); no-op for baseline_1 and baseline_2.",
    )
    p.add_argument(
        "--use-chunked",
        action="store_true",
        help="Use the chunked Stage D model with temporal ensembling instead "
             "of the single-step MLP. Requires --chunked-checkpoint pointing "
             "at a chunked-trained Stage D checkpoint. Only meaningful for "
             "core_method.",
    )
    p.add_argument(
        "--chunked-checkpoint",
        type=str,
        default=None,
        help="Path to a Stage D chunked checkpoint (produced by "
             "train_stage_d.py --chunk-size > 1). Required if --use-chunked "
             "is set.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Chunk size K for the chunked Stage D model. Must match the "
             "checkpoint's chunk_size. Default 25 (50 ms at 500 Hz).",
    )
    p.add_argument(
        "--ensemble-decay",
        type=float,
        default=0.1,
        help="Exponential decay rate m used to weight past chunks: "
             "w_i = exp(-m * i) where i is steps since the chunk was emitted. "
             "Larger m → faster forgetting. Default 0.1 (weights halve every "
             "~7 steps).",
    )
    p.add_argument(
        "--fallback-checkpoint",
        type=str,
        default=None,
        help="Optional path to a single-step Stage D checkpoint. If set and "
             "--use-chunked fails to load the chunked checkpoint, the runner "
             "transparently falls back to the single-step inference. If "
             "unset, a chunked-load failure is fatal.",
    )
    p.add_argument(
        "--track-target-during-hold",
        action="store_true",
        help="Closed-loop FR target tracking. When set, the controller "
             "updates the FR target in place during lift/extend/hold from "
             "the latest grounded button position. Mode is selected by "
             "--track-target-mode. Requires contact regrounding (i.e. "
             "core_method); 'stage_c' mode also requires --stage-c-ckpt-dir. "
             "Useful for moving-target demos where the button is perturbed "
             "mid-contact.",
    )
    p.add_argument(
        "--track-target-mode",
        choices=["stage_c", "jacobian"],
        default="stage_c",
        help="Tracking mode for --track-target-during-hold. 'stage_c' "
             "re-predicts FR waypoints via Stage C's waypoint head (limited "
             "by the trained model's spatial sensitivity). 'jacobian' maps "
             "the Cartesian target delta to FR joint-space deltas via the "
             "analytical inverse Jacobian — geometric chase, no model "
             "dependency. Default 'stage_c'.",
    )
    p.add_argument(
        "--track-target-relift-threshold",
        type=float,
        default=0.0,
        help="Lateral-move threshold (metres) above which a Jacobian "
             "tracking update during 'hold' rewinds the phase machine back "
             "to 'lift', producing a clean re-lift / re-extend / re-press "
             "at the new target instead of dragging the foot along the "
             "surface. Only fires before contact has been detected. "
             "0.0 (default) disables. Typical: 0.03 (3 cm).",
    )
    p.add_argument(
        "--no-save-grounding-frames",
        action="store_true",
        help="Disable saving raw + annotated JPG frames for each successful "
             "grounding stage (initial / approach / standoff). When unset "
             "(default), frames are written to "
             "<output-dir>/grounding/<trial_id>_<stage>_{raw,annotated}.jpg "
             "for use in demo visualisation.",
    )
    p.add_argument(
        "--intrinsics",
        type=str,
        choices=sorted(_grounding_module.INTRINSICS_PROFILES.keys()),
        default=_grounding_module.CAMERA_INTRINSICS_VERSION_CALIB,
        help="Camera intrinsics profile applied to all grounding calls "
             "(initial detection, approach re-grounding, contact-time "
             "GroundingThread). Default 'calib_2026_04' (plumb_bob "
             "calibration from April 2026). Use 'urdf_legacy' to run "
             "v3-era Stage C/D models that were trained against URDF-"
             "derived intrinsics with no distortion correction; otherwise "
             "their target_pos_base / foot_to_target_error inputs will "
             "be subtly OOD relative to training.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
# Trial body
# ──────────────────────────────────────────────

def run_trial(
    args: argparse.Namespace,
    csv_path: Path,
    audio_dir: Path,
    grounding_dir: Optional[Path] = None,
) -> TrialOutcome:
    variant_cfg = VARIANTS[args.variant]
    trial_id = f"{_now_stamp()}_{args.variant}"
    audio_path = audio_dir / f"{trial_id}.wav"
    save_grounding = (grounding_dir is not None
                      and not args.no_save_grounding_frames)
    g_stem = (
        (grounding_dir / trial_id) if save_grounding else None
    )

    outcome = TrialOutcome(
        trial_id=trial_id,
        variant=args.variant,
        prompt=args.prompt,
        operator_notes=args.notes,
        audio_path=str(audio_path.relative_to(Path.cwd())) if audio_path.is_absolute()
                   else str(audio_path),
        git_sha=_git_short_sha(),
        camera_intrinsics_version=str(
            _grounding_module.CAMERA_INTRINSICS_VERSION),
    )

    # Resources that need an explicit teardown if we get past creation.
    audio_recorder: Optional[AudioRecorder] = None
    audio_detector: Optional[AudioLiveDetector] = None
    grounding_thread: Optional[GroundingThread] = None
    audio_waveform: np.ndarray = np.zeros(0, dtype=np.float32)
    contact_result = None

    # Mutable holders so the except branch can still write the CSV row.
    try:
        # ── 1. DDS init ─────────────────────────────────────────────────
        logger.info(f"ChannelFactoryInitialize on {args.interface}")
        ChannelFactoryInitialize(0, args.interface)

        # ── 2. Robot + perception + audio ───────────────────────────────
        logger.info("Constructing Go2Interface")
        go2 = Go2Interface(network_interface=args.interface,
                           already_initialized=True)

        logger.info("Constructing Go2Camera")
        camera = Go2Camera(already_initialized=True)

        logger.info("Constructing VisualGrounder")
        grounder = VisualGrounder()

        logger.info(f"Constructing AudioLiveDetector (mic={args.mic_index})")
        audio_detector = AudioLiveDetector(
            sample_rate=AUDIO_SAMPLE_RATE,
            device_index=args.mic_index,
            threshold_db=args.audio_threshold_db,
            min_duration_ms=50,
            baseline_duration_s=AUDIO_BASELINE_S,
            capture_sample_rate=AUDIO_CAPTURE_RATE,
        )
        logger.info(f"Constructing AudioRecorder (mic={args.mic_index})")
        audio_recorder = AudioRecorder(
            sample_rate=AUDIO_SAMPLE_RATE,
            device_index=args.mic_index,
            capture_sample_rate=AUDIO_CAPTURE_RATE,
            aux_callback=audio_detector._callback,
        )
        logger.info("Constructing ColorDetector (faster-whisper)")
        color_detector = ColorDetector(
            model_size="small", device="cuda", compute_type="float16",
            model_path=(os.path.expanduser(args.whisper_model_path)
                        if args.whisper_model_path else None),
        )
        color_detector.load()

        audio_recorder.start_stream()
        logger.info("Calibrating audio baseline (keep quiet)…")
        audio_detector.calibrate_baseline(AUDIO_BASELINE_S)

        # ── 3. Stage C / Stage D models ─────────────────────────────────
        stage_c: Optional[StageCInference] = None
        if variant_cfg["use_stage_c_standoff"] or variant_cfg["use_stage_c_waypoints"]:
            if not args.stage_c_fallback or not args.waypoints_fallback:
                stage_c_ckpt = Path(args.stage_c_ckpt_dir) / "stage_c.pt"
                logger.info(f"Loading StageCInference from {stage_c_ckpt}")
                stage_c = StageCInference(
                    standoff_ckpt_path=stage_c_ckpt,
                    waypoint_ckpt_path=stage_c_ckpt,
                )
                outcome.stage_c_checkpoint = str(stage_c_ckpt)

        stage_d: Optional[Any] = None
        if variant_cfg["stage_d_ckpt"] is not None:
            if args.use_chunked:
                if args.chunked_checkpoint is None:
                    raise ValueError(
                        "--use-chunked requires --chunked-checkpoint.")
                chunked_ckpt = Path(args.chunked_checkpoint)
                logger.info(
                    f"Loading StageDChunkedInference from {chunked_ckpt} "
                    f"(chunk_size={args.chunk_size}, "
                    f"decay={args.ensemble_decay}) "
                    f"on device={args.stage_d_device}")
                stage_d = StageDChunkedInference(
                    checkpoint_path=chunked_ckpt,
                    chunk_size=args.chunk_size,
                    decay_rate=args.ensemble_decay,
                    device=args.stage_d_device,
                    fallback_to_single_step=(
                        args.fallback_checkpoint is not None),
                    fallback_checkpoint_path=args.fallback_checkpoint,
                )
                outcome.stage_d_checkpoint = str(chunked_ckpt)
            else:
                stage_d_ckpt = Path(args.stage_d_ckpt)
                logger.info(
                    f"Loading StageDInference from {stage_d_ckpt} "
                    f"on device={args.stage_d_device}")
                stage_d = StageDInference(
                    ckpt_path=stage_d_ckpt, device=args.stage_d_device)
                outcome.stage_d_checkpoint = str(stage_d_ckpt)
            outcome.stage_d_device = args.stage_d_device
        else:
            if args.stage_d_device != "cuda":
                # Flag set but variant doesn't load Stage D — surface it.
                logger.warning(
                    "--stage-d-device ignored: variant %s does not use Stage D",
                    args.variant,
                )
            if args.use_chunked:
                logger.warning(
                    "--use-chunked ignored: variant %s does not use Stage D",
                    args.variant,
                )

        # ── 4. Stand ────────────────────────────────────────────────────
        logger.info("Standing up")
        go2.stand()
        time.sleep(1.0)

        # ── 5. Initial grounding (with retry) ───────────────────────────
        logger.info("Initial grounding")
        target_initial = _try_ground(
            camera, grounder, args.prompt,
            save_stem=(Path(f"{g_stem}_initial") if g_stem else None),
        )
        if target_initial is None:
            logger.error("Initial grounding failed after retries — aborting trial")
            outcome.failure_mode = "no_detection"
            outcome.operator_kept = False
            return outcome
        outcome.target_x = float(target_initial[0])
        outcome.target_y = float(target_initial[1])
        outcome.target_z = float(target_initial[2])
        logger.info(f"target_initial = {target_initial.tolist()}")

        # ── 6. Decide standoff source ───────────────────────────────────
        use_stage_c_standoff = (
            variant_cfg["use_stage_c_standoff"]
            and not args.stage_c_fallback
            and stage_c is not None
        )
        if use_stage_c_standoff:
            standoff_offset_1 = stage_c.predict_standoff(
                target_initial, interaction="press")
            outcome.standoff_source = "stage_c"
        else:
            standoff_offset_1 = compute_heuristic_standoff(target_initial)
            if args.variant == "baseline_1":
                outcome.standoff_source = "heuristic"
            else:
                outcome.standoff_source = "stage_c_fallback_heuristic"
        outcome.standoff_dx_cmd = float(standoff_offset_1[0])
        outcome.standoff_dy_cmd = float(standoff_offset_1[1])

        # ── 7. Navigate (1- or 2-shot) ──────────────────────────────────
        logger.info(
            f"move_to_position dx={standoff_offset_1[0]:+.3f} "
            f"dy={standoff_offset_1[1]:+.3f}"
        )
        go2.move_to_position(float(standoff_offset_1[0]),
                             float(standoff_offset_1[1]),
                             0.0, speed=0.3)
        go2.stop()
        time.sleep(1.0)

        if variant_cfg["approach_regrounding"]:
            logger.info("Approach re-grounding")
            target_mid = _try_ground(
                camera, grounder, args.prompt,
                save_stem=(Path(f"{g_stem}_approach") if g_stem else None),
            )
            if target_mid is not None:
                if use_stage_c_standoff:
                    standoff_offset_2 = stage_c.predict_standoff(
                        target_mid, interaction="press")
                else:
                    standoff_offset_2 = compute_heuristic_standoff(target_mid)
                correction_dx = float(standoff_offset_2[0]) - float(standoff_offset_1[0])
                correction_dy = float(standoff_offset_2[1]) - float(standoff_offset_1[1])
                logger.info(
                    f"Correction dx={correction_dx:+.3f} dy={correction_dy:+.3f}"
                )
                if abs(correction_dx) > 0.02 or abs(correction_dy) > 0.02:
                    go2.move_to_position(correction_dx, correction_dy,
                                         0.0, speed=0.2)
                    go2.stop()
                    time.sleep(2.0)
            else:
                logger.warning("Approach re-grounding failed — keeping shot 1")

        # ── 8. Final grounding for contact target ───────────────────────
        target_at_standoff = _try_ground(
            camera, grounder, args.prompt,
            save_stem=(Path(f"{g_stem}_standoff") if g_stem else None),
        )
        if target_at_standoff is None:
            logger.warning(
                "Final grounding failed — falling back to initial target.")
            target_at_standoff = target_initial.copy()
        outcome.standoff_x = float(target_at_standoff[0])
        outcome.standoff_y = float(target_at_standoff[1])
        outcome.standoff_z = float(target_at_standoff[2])

        # ── 9. Stage C waypoints (or hardcoded fallback) ────────────────
        use_stage_c_waypoints = (
            variant_cfg["use_stage_c_waypoints"]
            and not args.waypoints_fallback
            and stage_c is not None
        )
        if use_stage_c_waypoints:
            fr_waypoints = stage_c.predict_waypoints(
                target_at_standoff, interaction="press")
            outcome.fr_waypoints_source = "stage_c"
            logger.info(
                f"Stage C waypoints: lift={fr_waypoints['lift'].tolist()} "
                f"extend={fr_waypoints['extend'].tolist()} "
                f"press={fr_waypoints['press'].tolist()}"
            )
        else:
            fr_waypoints = None
            outcome.fr_waypoints_source = (
                "hardcoded" if args.variant == "baseline_1"
                else "stage_c_fallback_hardcoded"
            )

        # ── 10. Optional contact-time grounding thread (5 Hz) ───────────
        # The CLI flag --no-contact-regrounding suppresses the thread even
        # when the variant config asks for it. Used to diagnose GPU contention
        # between the 5 Hz perception pipeline and Stage D's per-step predict().
        grounding_getter = None
        contact_regrounding_active = (
            variant_cfg["contact_regrounding"] and not args.no_contact_regrounding
        )
        if contact_regrounding_active:
            logger.info(
                f"Starting GroundingThread @ {CONTACT_GROUNDING_HZ} Hz")
            grounding_thread = GroundingThread(
                camera=camera, grounder=grounder, prompt=args.prompt,
                rate_hz=CONTACT_GROUNDING_HZ,
            )
            grounding_thread.start()
            grounding_getter = grounding_thread.get_latest_position
            outcome.contact_regrounding = "on"
        elif (variant_cfg["contact_regrounding"]
              and args.no_contact_regrounding):
            logger.info(
                "Contact-time GroundingThread SUPPRESSED (--no-contact-regrounding). "
                "Stage D state will use cached standoff target during the "
                "contact phase.")
            outcome.contact_regrounding = "suppressed"
        elif args.no_contact_regrounding:
            logger.warning(
                "--no-contact-regrounding ignored: variant %s does not start "
                "a contact-time GroundingThread.",
                args.variant,
            )

        # ── 11. Build contact controller ────────────────────────────────
        controller_class: Type = variant_cfg["controller_class"]
        residual_mask: Optional[np.ndarray] = None
        if args.residual_fr_only:
            if variant_cfg["stage_d_ckpt"] is None:
                logger.warning(
                    "--residual-fr-only is a no-op for variant %r "
                    "(no Stage D residual to mask).",
                    args.variant,
                )
                outcome.stage_d_residual_mode = "n/a"
            else:
                residual_mask = np.zeros(12, dtype=np.float32)
                residual_mask[0:3] = 1.0  # FR only
                outcome.stage_d_residual_mode = "fr_only"
                logger.info(
                    "Stage D residual mask active: FR-only "
                    "(indices [3:12] zeroed before sum into target_q)."
                )
        else:
            outcome.stage_d_residual_mode = (
                "full" if variant_cfg["stage_d_ckpt"] is not None else "n/a"
            )
        logger.info(f"Constructing {controller_class.__name__}")
        controller_kwargs: Dict[str, Any] = dict(
            network_interface=args.interface,
            already_initialized=True,
            fr_waypoints=fr_waypoints,
            stage_d_inference=stage_d,
            grounding_getter=grounding_getter,
            stage_d_residual_mask=residual_mask,
        )

        # Dynamic gravity-comp FF is implemented in HeuristicContact (the base
        # class), so it works for every variant. On by default (paper config);
        # disable with --no-gravity-ff.
        controller_kwargs["gravity_ff_enabled"] = bool(args.gravity_ff)
        if args.gravity_ff:
            logger.info(
                "Dynamic gravity-comp FF active: analytical calf + empirical "
                "thigh applied to FL/RR/RL during lift/extend/hold/retract_*."
            )
        else:
            logger.info("Dynamic gravity-comp FF disabled (--no-gravity-ff).")

        # Closed-loop target tracking — opt-in moving-target demo behaviour.
        # Both modes require contact-time regrounding for live target_pos_base
        # updates. 'stage_c' mode additionally requires Stage C (for live
        # waypoint re-prediction); 'jacobian' mode uses an analytical inverse
        # Jacobian and has no model dependency.
        if args.track_target_during_hold:
            stage_c_ok = (stage_c is not None and use_stage_c_waypoints)
            need_stage_c = (args.track_target_mode == "stage_c")
            can_track = (
                grounding_getter is not None
                and (stage_c_ok if need_stage_c else True)
            )
            if can_track:
                if stage_c_ok:
                    controller_kwargs["stage_c_for_tracking"] = stage_c
                controller_kwargs["track_target_during_hold"] = True
                controller_kwargs["track_target_mode"] = args.track_target_mode
                if args.track_target_mode == "jacobian":
                    controller_kwargs["track_target_relift_threshold"] = (
                        float(args.track_target_relift_threshold))
                relift_msg = (
                    f", relift_threshold={args.track_target_relift_threshold:.3f} m"
                    if (args.track_target_mode == "jacobian"
                        and args.track_target_relift_threshold > 0)
                    else ""
                )
                logger.info(
                    "Closed-loop target tracking active: mode=%s%s, "
                    "FR target updated from grounded button position on "
                    "every active-phase control step (5 mm move threshold).",
                    args.track_target_mode, relift_msg,
                )
            else:
                missing = []
                if need_stage_c and not stage_c_ok:
                    missing.append("Stage C waypoints")
                if grounding_getter is None:
                    missing.append("contact regrounding")
                logger.warning(
                    "--track-target-during-hold (mode=%s) ignored: missing "
                    "%s for variant %s.",
                    args.track_target_mode,
                    " + ".join(missing), args.variant,
                )

        if controller_class is HeuristicContactWholeBody:
            if args.rear_kp is not None:
                controller_kwargs["kp_support_soft"] = float(args.rear_kp)
            if args.rear_kd is not None:
                controller_kwargs["kd_support_soft"] = float(args.rear_kd)
            if args.rear_kp is not None or args.rear_kd is not None:
                logger.info(
                    "Soft-support gain overrides active: "
                    "rear_kp=%s rear_kd=%s",
                    args.rear_kp, args.rear_kd,
                )
            controller_kwargs["stage_d_residual_scale"] = float(args.residual_scale)
            logger.info(
                "Stage D residual scale: residual multiplied by %.3f before "
                "adding to target_q (paper default 0.5).",
                args.residual_scale,
            )
        else:
            if args.rear_kp is not None or args.rear_kd is not None:
                logger.warning(
                    "--rear-kp / --rear-kd ignored: variant %s does not use soft gains",
                    args.variant,
                )
            if args.compliance:
                logger.warning(
                    "--compliance is a no-op for variant %s (compliance is "
                    "already off in HeuristicContact base class).",
                    args.variant,
                )
            if args.residual_scale != 0.5:
                logger.warning(
                    "--residual-scale ignored: variant %s does not use Stage D",
                    args.variant,
                )

        controller = controller_class(**controller_kwargs)

        # ── Compliance mode on the WholeBody controller ─────────────────
        # OFF by default (paper config). When compliance is on, FR uses
        # KP_FR_COMPLIANT=15 and support legs use self._kp_support_soft during
        # extend+hold. When off, both branches fall through to the stiff
        # regime (FR=KP_FR=40, support=KP_STABLE=100 — same as baseline_2),
        # while keeping Stage D residual + grounding thread + Stage C
        # waypoints intact for FR-only correction studies.
        if (not args.compliance
                and controller_class is HeuristicContactWholeBody
                and hasattr(controller, "_compliance_active")):
            controller._compliance_active = False
            # Honest metadata: the soft-support gains aren't being applied
            # any more, so don't keep the wholebody_v2_kpsupport... tag.
            if hasattr(controller, "gain_schedule"):
                controller.gain_schedule = "wholebody_compliance_off"
            logger.info(
                "Compliance mode disabled — FR=KP_FR (40), "
                "support=KP_STABLE (100) during extend+hold."
            )

        # Record what the controller actually applied — defaults survive a
        # silently-rejected typo in the flag.
        if hasattr(controller, "_compliance_active"):
            outcome.compliance_mode = (
                "on" if bool(controller._compliance_active) else "off"
            )
        if hasattr(controller, "_stage_d_residual_scale"):
            outcome.stage_d_residual_scale = (
                f"{float(controller._stage_d_residual_scale):.3f}"
                if variant_cfg["stage_d_ckpt"] is not None else "n/a"
            )
        if hasattr(controller, "_gravity_ff_enabled"):
            outcome.tau_scheme = (
                "ff_dynamic" if bool(controller._gravity_ff_enabled) else "ff_off"
            )
        if hasattr(controller, "_kp_support_soft"):
            outcome.rear_kp_used = f"{float(controller._kp_support_soft):.3f}"
        if hasattr(controller, "_kd_support_soft"):
            outcome.rear_kd_used = f"{float(controller._kd_support_soft):.3f}"

        # ── 12. Episode begin: arm audio ────────────────────────────────
        audio_recorder.start_episode()
        audio_detector.start_episode()

        # ── 13. Execute contact ─────────────────────────────────────────
        logger.info("controller.execute (press_mode='wall')")
        contact_result = controller.execute(
            target_offset_xyz=target_at_standoff.astype(np.float64),
            press_mode="wall",
            audio_detector=audio_detector,
        )
        outcome.duration_s = float(contact_result.duration_s)
        outcome.contact_method = str(contact_result.contact_method)
        outcome.success_fk = bool(contact_result.success
                                  and contact_result.contact_method == "fk")

        # ── 14. Stop grounding thread (if running) ──────────────────────
        if grounding_thread is not None:
            grounding_thread.stop()
            grounding_thread = None

        # ── 15. Audio / color analysis ──────────────────────────────────
        audio_waveform = audio_recorder.end_episode()
        outcome.success_audio_live = bool(audio_detector.sound_detected)
        if audio_detector.detection_time_s is not None:
            outcome.audio_detection_time_s = float(
                audio_detector.detection_time_s)

        color_result = color_detector.detect(
            audio_waveform, sample_rate=AUDIO_SAMPLE_RATE)
        outcome.color_detected = color_result.get("color_detected") or ""
        outcome.success_target = (outcome.color_detected == EXPECTED_COLOR)

        # ── 16. Auto failure_mode ───────────────────────────────────────
        timeout_phase = None
        if getattr(contact_result, "timeout", False):
            timeout_phase = "execute"
        outcome.failure_mode = _classify_failure(
            success_fk=outcome.success_fk,
            audio_detected=outcome.success_audio_live,
            color_detected=outcome.color_detected or None,
            timeout_phase=timeout_phase,
        )

        # ── 17. Sit ─────────────────────────────────────────────────────
        try:
            go2.sit()
        except Exception as e:
            logger.warning(f"sit() failed: {e}")
        time.sleep(0.5)

        # ── 18. Operator prompt ─────────────────────────────────────────
        kept, override, op_notes = _operator_prompt(outcome.failure_mode)
        outcome.operator_kept = kept
        if override is not None:
            outcome.failure_mode = override
        # Combine CLI notes (if any) with operator-typed notes.
        combined = " | ".join(s for s in (args.notes, op_notes) if s)
        outcome.operator_notes = combined

        return outcome

    except Exception:
        logger.error("run_trial raised — recording crashed trial outcome")
        logger.error(traceback.format_exc())
        outcome.failure_mode = "exception"
        outcome.operator_kept = False
        return outcome

    finally:
        # Best-effort flush regardless of how we got here.
        if grounding_thread is not None:
            try:
                grounding_thread.stop()
            except Exception as e:
                logger.warning(f"grounding_thread.stop() failed: {e}")

        if audio_recorder is not None:
            if audio_waveform.size == 0:
                try:
                    audio_waveform = audio_recorder.end_episode()
                except Exception as e:
                    logger.warning(f"audio_recorder.end_episode() failed: {e}")
            try:
                _save_audio(audio_waveform, audio_path, AUDIO_SAMPLE_RATE)
            except Exception as e:
                logger.warning(f"audio save failed: {e}")
            try:
                audio_recorder.stop_stream()
            except Exception as e:
                logger.warning(f"audio_recorder.stop_stream() failed: {e}")
        if audio_detector is not None:
            try:
                audio_detector.stop_stream()
            except Exception as e:
                logger.warning(f"audio_detector.stop_stream() failed: {e}")

        try:
            _append_csv_row(csv_path, outcome)
        except Exception as e:
            logger.error(f"CSV write failed: {e}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Apply the chosen intrinsics profile BEFORE VisualGrounder is
    # constructed inside run_trial. The grounder's _unproject reads
    # CAMERA_K / CAMERA_D from src.perception.grounding at call time,
    # so this swap takes effect for every subsequent grounding call.
    _grounding_module.use_intrinsics(args.intrinsics)
    logger.info(
        f"Camera intrinsics profile: {_grounding_module.CAMERA_INTRINSICS_VERSION}")

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    grounding_dir = output_dir / "grounding"
    invocation_stamp = _now_stamp()
    csv_path = output_dir / f"eval_{invocation_stamp}.csv"

    logger.info("=" * 72)
    logger.info(f"Friday demo trial — variant={args.variant}")
    logger.info(f"Prompt:           {args.prompt!r}")
    logger.info(f"Interface:        {args.interface}")
    logger.info(f"CSV out:          {csv_path}")
    logger.info(f"Audio dir:        {audio_dir}")
    logger.info(f"stage_c_ckpt_dir: {args.stage_c_ckpt_dir}")
    logger.info(f"stage_d_ckpt:     {args.stage_d_ckpt}")
    logger.info(f"intrinsics:       {_grounding_module.CAMERA_INTRINSICS_VERSION}")
    logger.info(f"--stage-c-fallback={args.stage_c_fallback}  "
                f"--waypoints-fallback={args.waypoints_fallback}")
    logger.info("=" * 72)

    outcome = run_trial(
        args, csv_path=csv_path, audio_dir=audio_dir,
        grounding_dir=grounding_dir,
    )

    print("\nTrial summary:")
    print(f"  trial_id            = {outcome.trial_id}")
    print(f"  variant             = {outcome.variant}")
    print(f"  standoff_source     = {outcome.standoff_source}")
    print(f"  fr_waypoints_source = {outcome.fr_waypoints_source}")
    print(f"  contact_method      = {outcome.contact_method}")
    print(f"  success_audio_live  = {outcome.success_audio_live}")
    print(f"  color_detected      = {outcome.color_detected!r}")
    print(f"  success_target      = {outcome.success_target}")
    print(f"  failure_mode        = {outcome.failure_mode!r}")
    print(f"  operator_kept       = {outcome.operator_kept}")
    print(f"  audio               = {outcome.audio_path}")
    print(f"  CSV                 = {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
