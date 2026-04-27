#!/usr/bin/env python
"""
scripts/collect_guided.py

Stage D v2 hand-guided demonstration collection.

Kinesthetic variant of scripts/collect_stage_d.py. The operator physically
guides the FR paw onto the button during the extend + hold phases, while
gains are softened to KP_FR_COMPLIANT / KD_FR_COMPLIANT. The Jacobian-PID
corrective controller is NOT used.

Output: data/real/stage_d_v2/episode_guided_YYYYMMDD_HHMMSS.h5
Schema: v2.1 superset — four additive per-step fields
        (joint_tau_est, imu_gyro, imu_accel, achieved_delta_fr)
        plus root attr collection_mode="hand_guided".

Pre-session assumptions (same as collect_stage_d.py):
    - env_go2 conda env active
    - Ethernet interface up, robot powered on, Sport Mode available
    - USB mic plugged in and index known (use sd.query_devices())
    - Button placed against the wall within reach of the standoff

Usage:
    python scripts/collect_guided.py \\
        --button-height 0.25 --distance 0.75 --angle 0 \\
        --num-episodes 10 --mic-index 4

Spec: Stage D v2 compliance-guided demonstration, April 2026.
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

# DDS init MUST happen before any imports that subscribe or publish to DDS.
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.robot.go2_interface import Go2Interface
from src.perception.grounding import Go2Camera, VisualGrounder
from src.planner.heuristic_contact import (
    DEFAULT_INTERFACE,
    CONTACT_PROXIMITY_M,
)
from src.planner.heuristic_contact_guided import HeuristicContactGuided
from src.data.grounding_thread import GroundingThread
from src.data.recorder import StageDRecorder

from src.data.audio_recorder import AudioRecorder
from src.data.audio_live_detector import AudioLiveDetector
from src.data.color_detector import ColorDetector


logger = logging.getLogger("collect_guided")


# ──────────────────────────────────────────────
# Constants (mirrored from collect_stage_d.py)
# ──────────────────────────────────────────────

PROMPT              = "red button"
PRESS_OFFSET_X      = 0.593
PRESS_OFFSET_Y      = -0.047
NAV_EXTRA_FORWARD_M = 0.22
MIN_GROUNDING_CONF  = 0.5
AUDIO_SAMPLE_RATE   = 16000
AUDIO_CAPTURE_RATE  = 48000
AUDIO_BASELINE_S    = 2.0
SETTLE_AFTER_NAV_S  = 2.0
GROUNDING_RATE_HZ   = 5.0
COLOR_EXPECTED      = "red"
Y_SIT_STAND_DRIFT_COMP = 0.0

OUTPUT_ROOT = Path("data/real/stage_d_v2")


# ──────────────────────────────────────────────
# Episode config + summary
# ──────────────────────────────────────────────

@dataclass
class EpisodeConfig:
    episode_id:       str
    button_height_m:  float
    distance_m:       float
    angle_rad:        float
    perturbation_cmd: np.ndarray   # kept for metadata parity with v2.1 files
    output_path:      Path
    notes:            str = ""


@dataclass
class EpisodeSummary:
    episode_id:          str
    duration_s:          float
    contact_method:      str
    success_fk:          bool
    success_audio_live:  bool
    success_target:      bool
    color_detected:      str


@dataclass
class Runtime:
    go2:              Go2Interface
    camera:           Go2Camera
    grounder:         VisualGrounder
    heuristic:        HeuristicContactGuided
    audio_recorder:   AudioRecorder
    audio_detector:   AudioLiveDetector
    color_detector:   ColorDetector
    network_interface: str


_runtime_handle: Optional[Runtime] = None
_interrupt_count = 0


def _sigint_handler(signum, frame):
    global _interrupt_count
    _interrupt_count += 1
    if _interrupt_count >= 2:
        logger.error("Second Ctrl+C — hard exit.")
        os._exit(1)

    logger.warning("Ctrl+C — attempting graceful shutdown.")
    rt = _runtime_handle
    if rt is None:
        sys.exit(130)
    try:
        rt.heuristic._stop_flag.set()
    except Exception as e:
        logger.warning(f"heuristic stop_flag failed: {e}")
    for name, fn in [
        ("audio_recorder.stop_stream", rt.audio_recorder.stop_stream),
        ("audio_detector.stop_stream", rt.audio_detector.stop_stream),
        ("go2.sit",                    rt.go2.sit),
    ]:
        try:
            fn()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
    sys.exit(130)


def _make_episode_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_output_path(episode_id: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT / f"episode_guided_{episode_id}.h5"


def _prompt_yes_no(msg: str, default: str = "y") -> bool:
    suffix = " [Y/n] " if default.lower() == "y" else " [y/N] "
    while True:
        ans = input(msg + suffix).strip().lower()
        if not ans:
            ans = default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _print_safety_banner() -> None:
    """
    Printed once before the first episode. The compliance mode puts the
    operator's hand near a moving leg — the safety posture matters.
    """
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " STAGE D v2 — HAND-GUIDED DEMONSTRATION".ljust(70)              + "║")
    print("║" + "".ljust(70)                                                     + "║")
    print("║" + " FR leg gains drop to (KP=15, KD=1.5) during extend + hold.".ljust(70)  + "║")
    print("║" + " Other three legs stay rigid at KP=100. Do NOT push on them.".ljust(70) + "║")
    print("║" + "".ljust(70)                                                     + "║")
    print("║" + " Checklist before first episode:".ljust(70)                       + "║")
    print("║" + "   1. Gloves on.".ljust(70)                                       + "║")
    print("║" + "   2. Kill switch within reach (operator holds; helper ready).".ljust(70) + "║")
    print("║" + "   3. Mic placement confirmed <50cm, pointed at button.".ljust(70) + "║")
    print("║" + "   4. Quiet room during AudioLiveDetector baseline calibration.".ljust(70) + "║")
    print("║" + "   5. Support-leg joints visually rigid at standoff.".ljust(70)   + "║")
    print("║" + "   6. Button sound module working (test once manually).".ljust(70) + "║")
    print("║" + "".ljust(70)                                                     + "║")
    print("║" + " Compliance window = extend + hold only. The gain RESTORE on".ljust(70)   + "║")
    print("║" + " exit from hold will happen from the leg's ACHIEVED position,".ljust(70)  + "║")
    print("║" + " so the leg will not jerk — but keep hand clear once the".ljust(70)       + "║")
    print("║" + " '[GUIDED] Contact detected' message prints.".ljust(70)                     + "║")
    print("╚" + "═" * 70 + "╝")
    print()


def _print_phase_expectations() -> None:
    print("Expected terminal output per episode:")
    print("  [INFO] sit_to_stand → weight_shift → lift  (normal gains)")
    print("  [GUIDED] Compliance active — guide FR paw to button. Audio will end hold.")
    print("  [INFO] hold ... (operator presses paw onto button; audio fires)")
    print("  [GUIDED] Contact detected via audio at step <N>")
    print("  [INFO] retract_curl → retract_rotate → retract_extend → weight_unshift → settle → sit")
    print()


# ──────────────────────────────────────────────
# Runtime construction
# ──────────────────────────────────────────────

def _build_runtime(args) -> Runtime:
    logger.info("Constructing Go2Interface…")
    go2 = Go2Interface(network_interface=args.interface, already_initialized=True)

    logger.info("Constructing Go2Camera…")
    camera = Go2Camera(already_initialized=True)

    logger.info("Constructing VisualGrounder (SAM2 + GroundingDINO + DAV2)…")
    grounder = VisualGrounder()

    logger.info("Constructing HeuristicContactGuided…")
    heuristic = HeuristicContactGuided(
        network_interface=args.interface,
        already_initialized=True,
    )

    logger.info(f"Constructing AudioLiveDetector (device={args.mic_index}, "
                f"{AUDIO_CAPTURE_RATE}→{AUDIO_SAMPLE_RATE} Hz)…")
    audio_detector = AudioLiveDetector(
        sample_rate=AUDIO_SAMPLE_RATE,
        device_index=args.mic_index,
        threshold_db=args.audio_threshold_db,
        min_duration_ms=50,
        baseline_duration_s=AUDIO_BASELINE_S,
        capture_sample_rate=AUDIO_CAPTURE_RATE,
    )

    logger.info(f"Constructing AudioRecorder (device={args.mic_index})…")
    audio_recorder = AudioRecorder(
        sample_rate=AUDIO_SAMPLE_RATE,
        device_index=args.mic_index,
        capture_sample_rate=AUDIO_CAPTURE_RATE,
        aux_callback=audio_detector._callback,
    )

    logger.info("Constructing ColorDetector (Whisper small)…")
    color_detector = ColorDetector(
        model_size="small", device="cuda", compute_type="float16",
        model_path=os.path.expanduser("~/Robotics/weights/faster-whisper-small"),
    )
    color_detector.load()

    logger.info("Opening audio streams…")
    audio_recorder.start_stream()
    logger.info("Calibrating AudioLiveDetector baseline (keep quiet)…")
    audio_detector.calibrate_baseline(AUDIO_BASELINE_S)

    return Runtime(
        go2=go2, camera=camera, grounder=grounder, heuristic=heuristic,
        audio_recorder=audio_recorder, audio_detector=audio_detector,
        color_detector=color_detector, network_interface=args.interface,
    )


def _teardown_runtime(rt: Runtime) -> None:
    logger.info("Shutting down…")
    for name, fn in [
        ("audio_recorder.stop_stream", rt.audio_recorder.stop_stream),
        ("audio_detector.stop_stream", rt.audio_detector.stop_stream),
    ]:
        try:
            fn()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")


# ──────────────────────────────────────────────
# Per-episode procedure
# ──────────────────────────────────────────────

def run_one_episode(rt: Runtime, cfg: EpisodeConfig) -> Optional[EpisodeSummary]:
    logger.info("─" * 72)
    logger.info(f"Guided episode {cfg.episode_id}")
    logger.info("─" * 72)

    go2 = rt.go2
    camera = rt.camera
    grounder = rt.grounder
    heuristic = rt.heuristic
    audio_recorder = rt.audio_recorder
    audio_detector = rt.audio_detector
    color_detector = rt.color_detector

    # Stand + initial grounding
    logger.info("stand()…")
    go2.stand()
    time.sleep(1.0)

    frame = camera.get_frame()
    if frame is None:
        logger.error("Camera returned None — aborting episode.")
        return None
    initial = grounder.ground(frame, prompt=PROMPT)
    if (initial is None
            or initial.confidence < MIN_GROUNDING_CONF
            or initial.position_base is None):
        logger.error(f"Initial grounding failed (conf={getattr(initial,'confidence',None)}) — aborting.")
        return None
    target_pos_base_initial = initial.position_base.copy()
    logger.info(f"Initial target_pos_base = {target_pos_base_initial.tolist()}  "
                f"conf={initial.confidence:.2f}")

    # Navigate to standoff (no perturbation by default for guided mode)
    standoff_dx = (float(target_pos_base_initial[0])
                   - PRESS_OFFSET_X
                   + NAV_EXTRA_FORWARD_M
                   + float(cfg.perturbation_cmd[0]))
    standoff_dy = (float(target_pos_base_initial[1])
                   - PRESS_OFFSET_Y
                   - Y_SIT_STAND_DRIFT_COMP
                   + float(cfg.perturbation_cmd[1]))
    standoff_commanded = np.array([standoff_dx, standoff_dy, 0.0], dtype=np.float32)
    logger.info(f"move_to_position(dx={standoff_dx:+.3f}, dy={standoff_dy:+.3f}, 0)…")
    go2.move_to_position(standoff_dx, standoff_dy, 0.0, speed=0.3)
    go2.stop()
    go2.balance_stand()
    time.sleep(SETTLE_AFTER_NAV_S)

    # Re-ground at standoff
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
    logger.info(f"Standoff target_pos_base = {target_pos_base_at_standoff.tolist()}  "
                f"conf={standoff_result.confidence:.2f}")

    # Background threads — grounding_thread still runs so recorder can log
    # target_pos_base per-step, even though heuristic.grounding_getter is unwired.
    grounding_thread = GroundingThread(
        camera=camera, grounder=grounder,
        prompt=PROMPT, rate_hz=GROUNDING_RATE_HZ,
    )
    recorder = StageDRecorder(heuristic=heuristic, grounding_thread=grounding_thread)
    grounding_thread.start()
    recorder.start()

    audio_recorder.start_episode()
    audio_detector.start_episode()

    initial_metadata = {
        "episode_id":                  cfg.episode_id,
        "target_pos_base_at_standoff": target_pos_base_at_standoff.astype(np.float32),
        "target_pos_world_initial":    target_pos_base_initial.astype(np.float32),
        "perturbation_commanded":      cfg.perturbation_cmd.astype(np.float32),
        # k_p kept in metadata for schema parity with v2.1; not meaningful here.
        "k_p":                         0.0,
        "button_height_m":             float(cfg.button_height_m),
        "color_expected":              COLOR_EXPECTED,
        "notes":                       cfg.notes,
        # NEW: self-document calibration constants and what we commanded.
        # Enables post-hoc drift analysis across sessions even if the
        # offset constants are re-tuned between collections.
        "standoff_dxdy_commanded":     standoff_commanded,
        "press_offset_x":              np.float32(PRESS_OFFSET_X),
        "press_offset_y":              np.float32(PRESS_OFFSET_Y),
        "nav_extra_forward_m":         np.float32(NAV_EXTRA_FORWARD_M),
        "y_sit_stand_drift_comp":      np.float32(Y_SIT_STAND_DRIFT_COMP),
    }
    recorder.begin_episode(initial_metadata)
    print("\n" + "="*60)
    print("Ready to begin contact phase.")
    print("  - Position yourself on the FR side of the robot")
    print("  - One hand ready to guide the FR paw")
    print("  - Other hand on Ctrl+C")
    print("  - You'll see '[GUIDED] Compliance active' when to grab the leg")
    print("="*60)
    input("Press Enter to begin (Ctrl+C to abort)...\n")
    logger.info("heuristic.execute() — wall press, compliance mode on…")
    try:
        result = heuristic.execute(
            target_offset_xyz=target_pos_base_at_standoff,
            press_mode="wall",
            contact_proximity_m=CONTACT_PROXIMITY_M,
            use_foot_force=False,
            audio_detector=audio_detector,
            target_refresh_fn=None,
        )
    except Exception as e:
        logger.exception(f"heuristic.execute raised: {e}")
        recorder.stop()
        grounding_thread.stop()
        try:
            audio_recorder.end_episode()
        except Exception:
            pass
        return None

    recorder.end_episode(result)
    audio_waveform = audio_recorder.end_episode()

    logger.info("Running Whisper color detection…")
    color_result = color_detector.detect(audio_waveform, sample_rate=AUDIO_SAMPLE_RATE)
    logger.info(f"Color detector: transcript='{color_result['transcript']}' "
                f"detected='{color_result['color_detected']}'")

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
        "success_target":          (
            bool(getattr(audio_detector, "sound_detected", False))
            and color_result.get("color_detected") == COLOR_EXPECTED
        ),
        "audio_transcript":        color_result.get("transcript", ""),
        "collection_mode":         "hand_guided",
    }

    # Summary BEFORE save() so the user can still decide to discard the file.
    summary = EpisodeSummary(
        episode_id=cfg.episode_id,
        duration_s=float(result.duration_s),
        contact_method=str(result.contact_method),
        success_fk=bool(result.success),
        success_audio_live=bool(getattr(audio_detector, "sound_detected", False)),
        success_target=bool(metadata_override["success_target"]),
        color_detected=metadata_override["color_detected"],
    )

    print()
    print(f"  ── Guided episode {summary.episode_id}")
    print(f"     duration:            {summary.duration_s:.2f} s")
    print(f"     contact_method:      {summary.contact_method}")
    print(f"     success_fk:          {summary.success_fk}")
    print(f"     success_audio_live:  {summary.success_audio_live}")
    print(f"     success_target:      {summary.success_target}  "
          f"(detected='{summary.color_detected}')")
    print()

    keep = _prompt_yes_no("Save this episode?", default="y")
    if keep:
        recorder.save(
            str(cfg.output_path),
            audio=audio_waveform,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            metadata_override=metadata_override,
            use_v2_schema=True,
        )
        logger.info(f"Saved {cfg.output_path}")
    else:
        logger.info("Episode discarded — no HDF5 written.")

    recorder.stop()
    grounding_thread.stop()
    return summary


# ──────────────────────────────────────────────
# Session driver
# ──────────────────────────────────────────────

def run_session(rt: Runtime, args) -> None:
    kept: list[EpisodeSummary] = []
    aborted = 0

    _print_safety_banner()
    _print_phase_expectations()
    input("Press Enter to start first episode (Ctrl+C to abort)...")

    for i in range(args.num_episodes):
        print()
        print("#" * 72)
        print(f"# GUIDED EPISODE {i+1}/{args.num_episodes}")
        print("#" * 72)
        if i > 0:
            print("Between-episode reset:")
            print("  - Button visible and upright.")
            print("  - Robot back near approach mark if drifted.")
            print("  - Gloves still on; kill switch still in reach.")
            input("Press Enter when ready for the next episode...")

        episode_id = _make_episode_id()
        output_path = _make_output_path(episode_id)
        cfg = EpisodeConfig(
            episode_id=episode_id,
            button_height_m=args.button_height,
            distance_m=args.distance,
            angle_rad=args.angle,
            perturbation_cmd=np.zeros(2, dtype=np.float32),
            output_path=output_path,
            notes=f"guided ep{i+1}/{args.num_episodes}",
        )
        s = run_one_episode(rt, cfg)
        if s is None:
            aborted += 1
            continue
        kept.append(s)

    print()
    print("=" * 72)
    print(f"GUIDED SESSION — {len(kept)}/{args.num_episodes} ran, {aborted} aborted")
    print("=" * 72)
    if kept:
        audio_rate = sum(s.success_audio_live for s in kept) / len(kept)
        tgt_rate   = sum(s.success_target     for s in kept) / len(kept)
        print(f"  success_audio_live rate: {audio_rate*100:.1f}%")
        print(f"  success_target rate:     {tgt_rate*100:.1f}%")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage D v2 hand-guided demonstration collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interface",         type=str, default=DEFAULT_INTERFACE,
                   help="Ethernet interface for DDS.")
    # --network-interface kept as a longer alias so existing wrapper scripts work.
    p.add_argument("--network-interface", dest="interface", type=str,
                   help="Alias for --interface.")
    p.add_argument("--mic-index",         type=int, required=True,
                   help="USB mic device index.")
    p.add_argument("--audio-threshold-db", type=float, default=20.0,
                   help="AudioLiveDetector threshold above baseline RMS (dB).")
    p.add_argument("--button-height",     type=float, default=0.25,
                   help="Button height above floor (m).")
    p.add_argument("--distance",          type=float, default=0.75,
                   help="Approach distance from robot start to button (m).")
    p.add_argument("--angle",             type=float, default=0.0,
                   help="Approach yaw angle (rad).")
    p.add_argument("--num-episodes",      type=int, default=10)
    p.add_argument("--perturbation",      type=str, choices=["none"], default="none",
                   help="Only 'none' is supported in guided mode.")
    p.add_argument("--verbose",           action="store_true")
    return p


def main() -> int:
    global _runtime_handle
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"ChannelFactoryInitialize on {args.interface}…")
    ChannelFactoryInitialize(0, args.interface)

    rt = _build_runtime(args)
    _runtime_handle = rt
    signal.signal(signal.SIGINT, _sigint_handler)

    t0 = time.monotonic()
    try:
        run_session(rt, args)
    finally:
        elapsed = time.monotonic() - t0
        logger.info(f"Session elapsed: {elapsed/60:.1f} min")
        _teardown_runtime(rt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
