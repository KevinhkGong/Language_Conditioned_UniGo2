#!/usr/bin/env python
"""
scripts/collect_stage_d.py

Main orchestration script for Stage C / Stage D training data collection.

Three modes:
  --tune-kp                 K_p tuning session (spec Section 8.2, 10.2).
                            Runs 2 episodes per K_p value (no-perturbation + +3cm),
                            prints summary table, does NOT auto-select K_p.

  --mode stage-c            Stage C varied-placement collection (spec Section 10.3).
                            Single episode at a specified button placement.

  --mode stage-d            Stage D repeated collection (spec Sections 10.4–10.6).
                            Multiple episodes with optional per-episode random
                            perturbation.

Every mode shares a single per-episode procedure that wires together:
  - Go2Interface (Sport Mode navigation)
  - Go2Camera + VisualGrounder (perception)
  - HeuristicContactCorrective (Jacobian-PID-augmented contact)
  - GroundingThread (background re-grounding at 5 Hz)
  - StageDRecorder (500 Hz observer thread writing HDF5)
  - AudioRecorder + AudioLiveDetector + ColorDetector (v2.1 audio verification)

ChannelFactoryInitialize is called EXACTLY ONCE at program startup. All
downstream classes are instantiated with already_initialized=True.

Usage:
  # K_p tuning (run once per robot/session)
  python scripts/collect_stage_d.py --tune-kp \\
      --k-p-values 0.3 0.5 1.0 --mic-index 4

  # Pilot audio validation (5 no-perturb episodes)
  python scripts/collect_stage_d.py --mode stage-d \\
      --button-height 0.25 --distance 0.75 --angle 0 \\
      --perturbation none --num-episodes 5 --k-p 0.5 --pilot-check --mic-index 4

  # Stage C varied placement (1 episode at a time)
  python scripts/collect_stage_d.py --mode stage-c \\
      --button-height 0.25 --distance 0.75 --angle 0 \\
      --perturbation none --k-p 0.5 --mic-index 4

  # Stage D main collection (10 episodes, random small perturbation)
  python scripts/collect_stage_d.py --mode stage-d \\
      --button-height 0.25 --distance 0.75 --angle 0 \\
      --perturbation random-small --num-episodes 15 --k-p 0.5 --mic-index 4

Spec: training_data_spec.md v2.1, Section 10
"""

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
# DDS init MUST happen before any imports that subscribe or publish to DDS.
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.robot.go2_interface import Go2Interface
from src.perception.grounding import Go2Camera, VisualGrounder
from src.planner.heuristic_contact import (
    DEFAULT_INTERFACE,
    CONTACT_PROXIMITY_M,
)
from src.planner.heuristic_contact_corrective import HeuristicContactCorrective
from src.data.grounding_thread import GroundingThread
from src.data.recorder import StageDRecorder

# Audio modules (implemented separately by teammate per spec Sections 9.8-9.10)
from src.data.audio_recorder import AudioRecorder
from src.data.audio_live_detector import AudioLiveDetector
from src.data.color_detector import ColorDetector


logger = logging.getLogger("collect_stage_d")


# ──────────────────────────────────────────────
# Constants (from Summary 4, Summary 5, spec Section 9)
# ──────────────────────────────────────────────

PROMPT              = "red button"
PRESS_OFFSET_X      = 0.203         # m — forward of FR foot at standing
PRESS_OFFSET_Y      = 0.140         # m — left of FR foot at standing
MIN_GROUNDING_CONF  = 0.5
AUDIO_SAMPLE_RATE   = 16000         # Hz
AUDIO_BASELINE_S    = 2.0           # s of silence for AudioLiveDetector calibration
SETTLE_AFTER_NAV_S  = 2.0           # s to wait after go2.stop() before re-ground
GROUNDING_RATE_HZ   = 5.0
COLOR_EXPECTED      = "red"         # single-button v1

OUTPUT_ROOT = Path("data/real/stage_d")
TUNING_SUBDIR = "tuning"


# ──────────────────────────────────────────────
# Episode metadata + results
# ──────────────────────────────────────────────

@dataclass
class EpisodeConfig:
    """Configuration for a single episode."""
    episode_id: str
    k_p: float
    button_height_m: float
    distance_m: float
    angle_rad: float
    perturbation_cmd: np.ndarray   # (2,) dx, dy in metres
    output_path: Path
    notes: str = ""


@dataclass
class EpisodeSummary:
    """Compact per-episode summary for printing / tuning table."""
    episode_id: str
    k_p: float
    perturbation_norm_cm: float
    duration_s: float
    contact_method: str
    success_fk: bool
    success_audio_live: bool
    success_target: bool
    final_error_cm: float       # last non-nan foot_to_target_error magnitude
    max_delta_rad: float        # peak |jacobian_pid_delta| during episode
    color_detected: str


# ──────────────────────────────────────────────
# Shared runtime context — held for entire process
# ──────────────────────────────────────────────

@dataclass
class Runtime:
    go2: Go2Interface
    camera: Go2Camera
    grounder: VisualGrounder
    heuristic: HeuristicContactCorrective
    audio_recorder: AudioRecorder
    audio_detector: AudioLiveDetector
    color_detector: ColorDetector
    network_interface: str


# Global handle for signal handler. Set in main() after Runtime is built.
_runtime_handle: Optional[Runtime] = None
_interrupt_count = 0


# ──────────────────────────────────────────────
# Signal handling — graceful Ctrl+C
# ──────────────────────────────────────────────

def _sigint_handler(signum, frame):
    """
    Graceful shutdown. First Ctrl+C: attempt clean stop. Second Ctrl+C:
    hard exit. Best-effort — do not raise from inside the handler.
    """
    global _interrupt_count
    _interrupt_count += 1

    if _interrupt_count >= 2:
        logger.error("Second Ctrl+C — hard exit.")
        os._exit(1)

    logger.warning("Ctrl+C received — attempting graceful shutdown.")
    rt = _runtime_handle
    if rt is None:
        logger.warning("No runtime to clean up; exiting.")
        sys.exit(130)

    # Best-effort cleanup. Swallow all errors so we always reach sys.exit.
    try:
        if hasattr(rt, "heuristic") and rt.heuristic is not None:
            rt.heuristic._stop_flag.set()
    except Exception as e:
        logger.warning(f"heuristic stop_flag failed: {e}")

    try:
        rt.audio_recorder.stop_stream()
    except Exception as e:
        logger.warning(f"audio_recorder stop failed: {e}")

    try:
        rt.audio_detector.stop_stream()
    except Exception as e:
        logger.warning(f"audio_detector stop failed: {e}")

    try:
        rt.go2.sit()
    except Exception as e:
        logger.warning(f"go2.sit() failed: {e}")

    logger.warning("Cleanup attempted; exiting.")
    sys.exit(130)


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def _make_episode_id() -> str:
    """Unique per-episode ID with second-level timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_output_path(episode_id: str, subdir: Optional[str] = None,
                      kp_tag: Optional[str] = None) -> Path:
    """Construct the HDF5 output path for an episode."""
    base = OUTPUT_ROOT if subdir is None else OUTPUT_ROOT / subdir
    base.mkdir(parents=True, exist_ok=True)
    if kp_tag:
        fname = f"episode_kp{kp_tag}_{episode_id}.h5"
    else:
        fname = f"episode_{episode_id}.h5"
    return base / fname


def _sample_perturbation(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Per-episode perturbation sample. Returns (dx, dy) in metres."""
    if kind == "none":
        return np.zeros(2, dtype=np.float32)
    if kind == "random-small":
        return rng.uniform(-0.02, 0.02, size=2).astype(np.float32)
    if kind == "random-large":
        return rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    raise ValueError(f"Unknown perturbation kind: {kind}")


def _prompt_yes_no(msg: str, default: str = "y") -> bool:
    """Prompt user for y/n; default on empty input."""
    suffix = " [Y/n] " if default.lower() == "y" else " [y/N] "
    while True:
        ans = input(msg + suffix).strip().lower()
        if not ans:
            ans = default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _compute_episode_summary(
    cfg: EpisodeConfig,
    result,
    recorder: StageDRecorder,
    audio_detector: AudioLiveDetector,
    color_result: dict,
) -> EpisodeSummary:
    """
    Build an EpisodeSummary from the in-memory recorder buffers + result.
    Must be called BEFORE recorder.save() if we want to read per-step fields;
    safer to call after begin/end_episode set metadata but before buffers reset.
    In our flow we call it immediately after end_episode and before save — the
    recorder's save() does not clear buffers.
    """
    # final_error_cm — last finite-magnitude foot_to_target_error from recorder
    final_err_cm = float("nan")
    max_delta_rad = 0.0
    with recorder._buffer_lock:
        if recorder.foot_to_target_error:
            for err in reversed(recorder.foot_to_target_error):
                if err is not None and np.all(np.isfinite(err)):
                    final_err_cm = float(np.linalg.norm(err) * 100.0)
                    break
        if recorder.jacobian_pid_delta:
            mags = [
                float(np.linalg.norm(d)) for d in recorder.jacobian_pid_delta
                if d is not None and np.all(np.isfinite(d))
            ]
            if mags:
                max_delta_rad = max(mags)

    detected = color_result.get("color_detected") or ""
    return EpisodeSummary(
        episode_id=cfg.episode_id,
        k_p=cfg.k_p,
        perturbation_norm_cm=float(np.linalg.norm(cfg.perturbation_cmd) * 100.0),
        duration_s=float(result.duration_s),
        contact_method=str(result.contact_method),
        success_fk=bool(result.success),
        success_audio_live=bool(getattr(audio_detector, "sound_detected", False)),
        success_target=bool(detected == COLOR_EXPECTED),
        final_error_cm=final_err_cm,
        max_delta_rad=max_delta_rad,
        color_detected=detected,
    )


def _print_episode_summary(s: EpisodeSummary) -> None:
    print()
    print(f"  ── Episode {s.episode_id}  (k_p={s.k_p})")
    print(f"     perturbation:        {s.perturbation_norm_cm:.1f} cm")
    print(f"     duration:            {s.duration_s:.2f} s")
    print(f"     contact_method:      {s.contact_method}")
    print(f"     success_fk:          {s.success_fk}")
    print(f"     success_audio_live:  {s.success_audio_live}")
    print(f"     success_target:      {s.success_target}  (detected='{s.color_detected}')")
    print(f"     final |err|:         {s.final_error_cm:.2f} cm")
    print(f"     max |delta|:         {s.max_delta_rad:.4f} rad")
    print()


# ──────────────────────────────────────────────
# Runtime construction / teardown
# ──────────────────────────────────────────────

def _build_runtime(args) -> Runtime:
    """
    Called once after ChannelFactoryInitialize. Creates all long-lived objects:
    robot interface, camera, grounder, heuristic, audio streams, Whisper.
    """
    logger.info("Constructing Go2Interface…")
    go2 = Go2Interface(network_interface=args.interface, already_initialized=True)

    logger.info("Constructing Go2Camera…")
    camera = Go2Camera(already_initialized=True)

    logger.info("Constructing VisualGrounder (this loads SAM2 + GroundingDINO + DAV2)…")
    grounder = VisualGrounder()   # loads all three models onto GPU (~10-15s)

    logger.info(f"Constructing HeuristicContactCorrective (k_p={args.k_p})…")
    heuristic = HeuristicContactCorrective(
        network_interface=args.interface,
        already_initialized=True,
        k_p=args.k_p,
        grounding_getter=None,    # wired per-episode once GroundingThread exists
        record_correction=True,
    )

    logger.info(f"Constructing AudioRecorder (mic_index={args.mic_index})…")
    audio_recorder = AudioRecorder(
        sample_rate=AUDIO_SAMPLE_RATE,
        device_index=args.mic_index,
    )

    logger.info(f"Constructing AudioLiveDetector (mic_index={args.mic_index})…")
    audio_detector = AudioLiveDetector(
        sample_rate=AUDIO_SAMPLE_RATE,
        device_index=args.mic_index,
        threshold_db=args.audio_threshold_db,
        min_duration_ms=50,
        baseline_duration_s=AUDIO_BASELINE_S,
    )

    logger.info("Constructing ColorDetector (Whisper tiny)…")
    color_detector = ColorDetector(model_size="tiny", device="cuda", compute_type="float16")

    logger.info("Pre-loading Whisper model…")
    color_detector.load()

    logger.info("Opening audio streams…")
    audio_recorder.start_stream()
    audio_detector.start_stream()

    logger.info("Calibrating AudioLiveDetector baseline (keep quiet)…")
    audio_detector.calibrate_baseline(AUDIO_BASELINE_S)

    return Runtime(
        go2=go2,
        camera=camera,
        grounder=grounder,
        heuristic=heuristic,
        audio_recorder=audio_recorder,
        audio_detector=audio_detector,
        color_detector=color_detector,
        network_interface=args.interface,
    )


def _teardown_runtime(rt: Runtime) -> None:
    """Best-effort cleanup at normal shutdown. Swallow errors."""
    logger.info("Shutting down…")
    for name, fn in [
        ("audio_recorder.stop_stream",  rt.audio_recorder.stop_stream),
        ("audio_detector.stop_stream",  rt.audio_detector.stop_stream),
    ]:
        try:
            fn()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")


# ──────────────────────────────────────────────
# Core per-episode procedure (spec Section 10.3 steps 1-24)
# ──────────────────────────────────────────────

def run_one_episode(
    rt: Runtime,
    cfg: EpisodeConfig,
    *,
    skip_nav: bool = False,
) -> Optional[EpisodeSummary]:
    """
    Execute one full episode end-to-end.

    Args:
        rt:       shared Runtime (already constructed)
        cfg:      per-episode config (k_p, button placement, perturbation, output path)
        skip_nav: if True, do NOT call go2.move_to_position. Used in --tune-kp mode
                  where the user manually positions the robot for Episode A and
                  we only apply the +3cm perturbation for Episode B.

    Returns:
        EpisodeSummary on success, None on abort (e.g., grounding failure).
    """
    logger.info("─" * 72)
    logger.info(f"Starting episode {cfg.episode_id}  "
                f"k_p={cfg.k_p}  pert={cfg.perturbation_cmd.tolist()}")
    logger.info("─" * 72)

    go2 = rt.go2
    camera = rt.camera
    grounder = rt.grounder
    heuristic = rt.heuristic
    audio_recorder = rt.audio_recorder
    audio_detector = rt.audio_detector
    color_detector = rt.color_detector

    # Update heuristic k_p in place (persisted across episodes in the same run)
    heuristic.k_p = cfg.k_p

    # ── Step 1: stand ───────────────────────────────────────────────────────
    logger.info("stand()…")
    go2.stand()
    time.sleep(1.0)   # post-stand settle

    # ── Step 2: one-shot initial grounding ──────────────────────────────────
    logger.info("Initial grounding…")
    frame = camera.get_frame()
    if frame is None:
        logger.error("Camera returned None — aborting episode.")
        return None
    initial = grounder.ground(frame, prompt=PROMPT)
    if (initial is None
            or initial.confidence < MIN_GROUNDING_CONF
            or initial.position_base is None):
        conf = getattr(initial, "confidence", None)
        logger.error(
            f"Initial grounding failed "
            f"(result={initial}, confidence={conf}, min={MIN_GROUNDING_CONF}) "
            f"— aborting episode."
        )
        return None
    target_pos_base_initial = initial.position_base.copy()
    logger.info(
        f"Initial target_pos_base = "
        f"[{target_pos_base_initial[0]:+.3f}, "
        f"{target_pos_base_initial[1]:+.3f}, "
        f"{target_pos_base_initial[2]:+.3f}]  "
        f"conf={initial.confidence:.2f}"
    )

    # ── Step 3-5: navigate to standoff (possibly skipped in tune-kp mode A) ──
    if not skip_nav:
        standoff_dx = (
            float(target_pos_base_initial[0])
            - PRESS_OFFSET_X
            + float(cfg.perturbation_cmd[0])
        )
        standoff_dy = (
            float(target_pos_base_initial[1])
            - PRESS_OFFSET_Y
            + float(cfg.perturbation_cmd[1])
        )
        logger.info(
            f"Navigate move_to_position(dx={standoff_dx:+.3f}, dy={standoff_dy:+.3f}, 0)…"
        )
        go2.move_to_position(standoff_dx, standoff_dy, 0.0, speed=0.3)
        go2.stop()
    else:
        # In tune-kp Episode B, apply perturbation only (user positioned for A)
        if float(np.linalg.norm(cfg.perturbation_cmd)) > 1e-4:
            dx = float(cfg.perturbation_cmd[0])
            dy = float(cfg.perturbation_cmd[1])
            logger.info(
                f"Applying tuning perturbation move_to_position(dx={dx:+.3f}, "
                f"dy={dy:+.3f}, 0)…"
            )
            go2.move_to_position(dx, dy, 0.0, speed=0.3)
            go2.stop()

    # ── Step 6: settle before re-grounding ──────────────────────────────────
    time.sleep(SETTLE_AFTER_NAV_S)

    # ── Step 7: re-ground to capture target at standoff ─────────────────────
    logger.info("Re-grounding at standoff…")
    frame = camera.get_frame()
    if frame is None:
        logger.error("Camera returned None during re-ground — aborting.")
        return None
    standoff_result = grounder.ground(frame, prompt=PROMPT)
    if (standoff_result is None
            or standoff_result.confidence < MIN_GROUNDING_CONF
            or standoff_result.position_base is None):
        logger.error("Re-grounding at standoff failed — aborting.")
        return None
    target_pos_base_at_standoff = standoff_result.position_base.copy()
    logger.info(
        f"Standoff target_pos_base = "
        f"[{target_pos_base_at_standoff[0]:+.3f}, "
        f"{target_pos_base_at_standoff[1]:+.3f}, "
        f"{target_pos_base_at_standoff[2]:+.3f}]  "
        f"conf={standoff_result.confidence:.2f}"
    )

    # ── Step 8-9: spin up background threads ────────────────────────────────
    grounding_thread = GroundingThread(
        camera=camera, grounder=grounder,
        prompt=PROMPT, rate_hz=GROUNDING_RATE_HZ,
    )
    # Plumb the thread into the heuristic's corrective loop
    heuristic.grounding_getter = grounding_thread.get_latest_position

    recorder = StageDRecorder(heuristic=heuristic, grounding_thread=grounding_thread)

    grounding_thread.start()
    recorder.start()

    # ── Step 10-11: audio lifecycle — start episode ─────────────────────────
    audio_recorder.start_episode()
    audio_detector.start_episode()

    # ── Step 12: begin_episode with initial metadata ────────────────────────
    # Final success fields are filled via metadata_override at save() time.
    initial_metadata = {
        "episode_id": cfg.episode_id,
        "target_pos_base_at_standoff": target_pos_base_at_standoff.astype(np.float32),
        "target_pos_world_initial":    target_pos_base_initial.astype(np.float32),
        "perturbation_commanded":      cfg.perturbation_cmd.astype(np.float32),
        "k_p":                         float(cfg.k_p),
        "button_height_m":             float(cfg.button_height_m),
        "color_expected":              COLOR_EXPECTED,
        "notes":                       cfg.notes,
    }
    recorder.begin_episode(initial_metadata)

    # ── Step 13: execute contact ────────────────────────────────────────────
    logger.info("Calling heuristic.execute() — audio detection active…")
    try:
        result = heuristic.execute(
            target_offset_xyz=target_pos_base_at_standoff,
            press_mode="wall",
            contact_proximity_m=CONTACT_PROXIMITY_M,
            use_foot_force=False,
            audio_detector=audio_detector,
        )
    except Exception as e:
        logger.exception(f"heuristic.execute raised: {e}")
        recorder.stop()
        grounding_thread.stop()
        # drain/discard audio so mic streams are ready for next episode
        try:
            audio_recorder.end_episode()
        except Exception:
            pass
        return None

    # ── Step 14: end_episode (captures result metadata) ─────────────────────
    recorder.end_episode(result)

    # ── Step 15: flush audio waveform ───────────────────────────────────────
    audio_waveform = audio_recorder.end_episode()

    # ── Step 16: transcribe colour (post-hoc, on GPU) ───────────────────────
    logger.info("Running Whisper color detection…")
    color_result = color_detector.detect(audio_waveform, sample_rate=AUDIO_SAMPLE_RATE)
    logger.info(f"Color detector: transcript='{color_result['transcript']}' "
                f"detected='{color_result['color_detected']}'")

    # ── Step 17: build metadata_override ────────────────────────────────────
    metadata_override = {
        "success_fk":              bool(result.success),
        "success_audio_live":      bool(getattr(audio_detector, "sound_detected", False)),
        "audio_detection_time_s":  (
            float(audio_detector.detection_time_s)
            if getattr(audio_detector, "detection_time_s", None) is not None
            else float("nan")
        ),
        "contact_method":          str(result.contact_method),
        "color_detected":          color_result.get("color_detected") or "",
        "color_expected":          COLOR_EXPECTED,
        "success_target":          (color_result.get("color_detected") == COLOR_EXPECTED),
        "audio_transcript":        color_result.get("transcript", ""),
    }

    # ── Build summary BEFORE save() so we can optionally discard ────────────
    summary = _compute_episode_summary(
        cfg=cfg, result=result, recorder=recorder,
        audio_detector=audio_detector, color_result=color_result,
    )

    # ── Step 18: save HDF5 ──────────────────────────────────────────────────
    recorder.save(
        str(cfg.output_path),
        audio=audio_waveform,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        metadata_override=metadata_override,
    )

    # ── Step 19: stop background threads ────────────────────────────────────
    recorder.stop()
    grounding_thread.stop()
    heuristic.grounding_getter = None    # detach from dead thread

    # ── Step 24: print summary ──────────────────────────────────────────────
    _print_episode_summary(summary)
    return summary


# ──────────────────────────────────────────────
# Mode A: K_p tuning (spec Section 8.2, 10.2)
# ──────────────────────────────────────────────

def run_tune_kp(rt: Runtime, args) -> None:
    """
    For each K_p in args.k_p_values, run two episodes:
      A: no perturbation   (user positions robot manually)
      B: +3cm forward perturbation via move_to_position
    """
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " K_p TUNING MODE — safety reminder".ljust(70)             + "║")
    print("║" + "".ljust(70) + "║")
    print("║" + " Previous gain experiment caused a forward flip + cracked".ljust(70)  + "║")
    print("║" + " Ethernet port. If the robot moves violently, Ctrl+C NOW".ljust(70)   + "║")
    print("║" + " and HALVE the K_p value.".ljust(70)                                   + "║")
    print("║" + " Do tuning on the floor, not a table.".ljust(70)                       + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    summaries: list[tuple[float, str, Optional[EpisodeSummary]]] = []

    for k_p in args.k_p_values:
        for label, perturbation in [
            ("A", np.zeros(2, dtype=np.float32)),
            ("B", np.array([0.03, 0.0], dtype=np.float32)),
        ]:
            print()
            print("=" * 72)
            print(f"Tuning run  k_p={k_p}  Episode {label}  "
                  f"perturbation={perturbation.tolist()}")
            print("=" * 72)
            print(f"Position the robot at IDEAL STANDOFF  "
                  f"({PRESS_OFFSET_X*100:.1f} cm behind button, "
                  f"{PRESS_OFFSET_Y*100:.1f} cm right of button center).")
            print(f"(Use scripts/measure_standoff.py in another terminal if helpful.)")
            input("Press Enter when robot is positioned and ready...")

            episode_id = _make_episode_id()
            kp_tag = f"{int(round(k_p * 100)):03d}"
            output_path = _make_output_path(
                episode_id, subdir=TUNING_SUBDIR, kp_tag=kp_tag)
            cfg = EpisodeConfig(
                episode_id=episode_id,
                k_p=k_p,
                button_height_m=args.button_height,
                distance_m=args.distance,
                angle_rad=0.0,
                perturbation_cmd=perturbation,
                output_path=output_path,
                notes=f"tune-kp Episode {label}",
            )
            s = run_one_episode(rt, cfg, skip_nav=True)
            summaries.append((k_p, label, s))

    # ── Final tuning summary table ──────────────────────────────────────────
    print()
    print("╔" + "═" * 80 + "╗")
    print("║  K_p TUNING RESULTS — pick the gain per spec Section 8.3".ljust(81) + "║")
    print("╚" + "═" * 80 + "╝")
    print()
    # Group by k_p
    by_kp: dict[float, dict[str, Optional[EpisodeSummary]]] = {}
    for k_p, label, s in summaries:
        by_kp.setdefault(k_p, {})[label] = s

    header = (f"{'K_p':>6}  {'A:fk':>5}  {'A:audio':>7}  {'B:fk':>5}  "
              f"{'B:audio':>7}  {'B:err_cm':>8}  {'B:max_d':>7}")
    print(header)
    print("-" * len(header))
    for k_p in sorted(by_kp.keys()):
        a = by_kp[k_p].get("A")
        b = by_kp[k_p].get("B")
        a_fk    = "OK"  if a and a.success_fk         else ("--" if a is None else "no")
        a_aud   = "OK"  if a and a.success_audio_live else ("--" if a is None else "no")
        b_fk    = "OK"  if b and b.success_fk         else ("--" if b is None else "no")
        b_aud   = "OK"  if b and b.success_audio_live else ("--" if b is None else "no")
        b_err   = f"{b.final_error_cm:.1f}"  if b else "--"
        b_delta = f"{b.max_delta_rad:.3f}"   if b else "--"
        print(f"{k_p:>6.2f}  {a_fk:>5}  {a_aud:>7}  {b_fk:>5}  "
              f"{b_aud:>7}  {b_err:>8}  {b_delta:>7}")
    print()
    print("Acceptance criteria (spec Section 8.3):")
    print("  1. Episode A: both success_fk AND success_audio_live must be OK")
    print("  2. Episode B: both success_fk AND success_audio_live must be OK")
    print("  3. No oscillation / support leg disturbance (human judgement)")
    print("  4. Final |err| during hold < 2 cm")
    print()
    print("Recommended starting point: K_p = 0.5 if it satisfies all of the above.")
    print()


# ──────────────────────────────────────────────
# Mode B: Stage C — single varied-placement episode
# ──────────────────────────────────────────────

def run_stage_c(rt: Runtime, args) -> None:
    """
    Single-episode Stage C collection at a specified placement.
    """
    episode_id = _make_episode_id()
    rng = np.random.default_rng()
    perturbation = _sample_perturbation(args.perturbation, rng)
    output_path = _make_output_path(episode_id, subdir=None)
    cfg = EpisodeConfig(
        episode_id=episode_id,
        k_p=args.k_p,
        button_height_m=args.button_height,
        distance_m=args.distance,
        angle_rad=args.angle,
        perturbation_cmd=perturbation,
        output_path=output_path,
        notes=f"stage-c height={args.button_height}m dist={args.distance}m "
              f"angle={args.angle}rad pert={args.perturbation}",
    )

    s = run_one_episode(rt, cfg)
    if s is None:
        logger.warning("Episode aborted — no HDF5 saved.")
        return

    if _prompt_yes_no("Keep this episode?", default="y"):
        logger.info(f"Kept: {output_path}")
    else:
        try:
            output_path.unlink()
            logger.info(f"Discarded {output_path}")
        except Exception as e:
            logger.warning(f"Failed to unlink {output_path}: {e}")


# ──────────────────────────────────────────────
# Mode C: Stage D — multi-episode loop with perturbation
# ──────────────────────────────────────────────

def run_stage_d(rt: Runtime, args) -> None:
    """
    Multi-episode collection. Each episode samples a perturbation per
    args.perturbation scheme. Pilot-check mode adds extra guidance after 5.
    """
    rng = np.random.default_rng(args.seed)
    kept: list[EpisodeSummary] = []
    aborted = 0

    for i in range(args.num_episodes):
        print()
        print("#" * 72)
        print(f"# STAGE-D EPISODE {i+1}/{args.num_episodes}")
        print("#" * 72)
        if i > 0:
            # Prompt between episodes so the user can reset the scene.
            print("Between-episode reset:")
            print("  - Ensure button is visible and upright.")
            print("  - Move robot back toward the approach starting mark if drifted.")
            print("  - Ensure Ethernet tape is still secure.")
            input("Press Enter when ready for the next episode...")

        episode_id = _make_episode_id()
        perturbation = _sample_perturbation(args.perturbation, rng)
        output_path = _make_output_path(episode_id, subdir=None)

        cfg = EpisodeConfig(
            episode_id=episode_id,
            k_p=args.k_p,
            button_height_m=args.button_height,
            distance_m=args.distance,
            angle_rad=args.angle,
            perturbation_cmd=perturbation,
            output_path=output_path,
            notes=f"stage-d ep{i+1}/{args.num_episodes} pert={args.perturbation}",
        )

        s = run_one_episode(rt, cfg)
        if s is None:
            aborted += 1
            continue

        # In Stage D we keep all episodes (including failures) unless obvious garbage.
        # Spec Section 10.6: failures teach the policy reach limits.
        kept.append(s)

    # ── Post-session summary ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"STAGE-D SESSION COMPLETE — {len(kept)}/{args.num_episodes} kept, "
          f"{aborted} aborted")
    print("=" * 72)
    if kept:
        fk_rate    = sum(s.success_fk         for s in kept) / len(kept)
        audio_rate = sum(s.success_audio_live for s in kept) / len(kept)
        tgt_rate   = sum(s.success_target     for s in kept) / len(kept)
        audio_fired = sum(s.contact_method == "audio" for s in kept) / len(kept)
        print(f"  success_fk rate:         {fk_rate*100:.1f}%")
        print(f"  success_audio_live rate: {audio_rate*100:.1f}%")
        print(f"  success_target rate:     {tgt_rate*100:.1f}%")
        print(f"  contact_method == audio: {audio_fired*100:.1f}%")

    # ── Pilot-check guidance (spec Section 10.2.5) ──────────────────────────
    if args.pilot_check:
        print()
        print("╔" + "═" * 70 + "╗")
        print("║  PILOT VALIDATION CHECKLIST (spec Section 10.2.5)".ljust(71)           + "║")
        print("╠" + "═" * 70 + "╣")
        print("║  success_fk rate           ≥ 4/5 ?".ljust(71)                          + "║")
        print("║  success_audio_live rate   ≥ 4/5 ?".ljust(71)                          + "║")
        print("║  success_fk == success_audio_live in all 5 ?".ljust(71)                + "║")
        print("║  color_detected == 'red' whenever success_audio_live ?".ljust(71)      + "║")
        print("║  Ambient noise floor stable (no spikes during silence) ?".ljust(71)    + "║")
        print("╠" + "═" * 70 + "╣")
        print("║  If success_audio_live < 60%:".ljust(71)                                 + "║")
        print("║    1. Manually press button — does sound_detected fire?".ljust(71)       + "║")
        print("║    2. Inspect camera: does foot visibly depress button?".ljust(71)       + "║")
        print("║       If no, raise FR_EXTEND_OFFSET_WALL[2] by 0.1 rad".ljust(71)        + "║")
        print("║    3. Check mic placement (<50cm, pointed at button).".ljust(71)         + "║")
        print("╚" + "═" * 70 + "╝")
        print()


# ──────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage C / Stage D data collection orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interface", type=str, default=DEFAULT_INTERFACE,
                   help="Ethernet interface name for DDS.")
    p.add_argument("--mic-index", type=int, required=True,
                   help="USB microphone device index (from sd.query_devices()).")
    p.add_argument("--audio-threshold-db", type=float, default=20.0,
                   help="AudioLiveDetector threshold above baseline RMS (dB).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")

    # Mode selection (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune-kp", action="store_true",
                      help="Run K_p tuning session (2 episodes per K_p).")
    mode.add_argument("--mode", type=str, choices=["stage-c", "stage-d"],
                      help="Main collection mode.")

    # Shared placement args
    p.add_argument("--button-height", type=float, default=0.25,
                   help="Button height above floor (m).")
    p.add_argument("--distance", type=float, default=0.75,
                   help="Approach distance from robot start to button (m).")
    p.add_argument("--angle", type=float, default=0.0,
                   help="Approach yaw angle (rad).")

    # Tune-kp mode
    p.add_argument("--k-p-values", type=float, nargs="+",
                   default=[0.3, 0.5, 1.0],
                   help="K_p values to sweep in --tune-kp mode.")

    # Stage-c/d mode
    p.add_argument("--perturbation", type=str,
                   choices=["none", "random-small", "random-large"],
                   default="none",
                   help="Per-episode perturbation scheme.")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Number of episodes (stage-d only).")
    p.add_argument("--k-p", type=float, default=0.5,
                   help="K_p gain for Jacobian-PID corrective controller.")
    p.add_argument("--pilot-check", action="store_true",
                   help="After session, print pilot-validation checklist.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for perturbation sampling.")

    return p


def main() -> int:
    global _runtime_handle

    args = _build_arg_parser().parse_args()

    # Validate arg combinations
    if not args.tune_kp and args.mode is None:
        logger.error("Must pass either --tune-kp or --mode stage-{c,d}")
        return 2
    if args.mode == "stage-c" and args.num_episodes != 1:
        logger.warning("stage-c ignores --num-episodes; running exactly 1 episode.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── DDS init ONCE, before any class that uses it ───────────────────────
    logger.info(f"ChannelFactoryInitialize on {args.interface}…")
    ChannelFactoryInitialize(0, args.interface)

    # Build runtime (robot + camera + grounder + heuristic + audio)
    rt = _build_runtime(args)
    _runtime_handle = rt
    signal.signal(signal.SIGINT, _sigint_handler)

    t0 = time.monotonic()
    try:
        if args.tune_kp:
            run_tune_kp(rt, args)
        elif args.mode == "stage-c":
            run_stage_c(rt, args)
        elif args.mode == "stage-d":
            run_stage_d(rt, args)
    finally:
        elapsed = time.monotonic() - t0
        logger.info(f"Session elapsed: {elapsed/60:.1f} min")
        _teardown_runtime(rt)

    return 0


if __name__ == "__main__":
    sys.exit(main())