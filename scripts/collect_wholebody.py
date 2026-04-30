#!/usr/bin/env python
"""
scripts/collect_wholebody.py

Stage D v3 whole-body kinesthetic demonstration collection.

Mirrors scripts/collect_guided.py but uses HeuristicContactWholeBody, which
softens BOTH the FR leg (KP=15) AND the support legs (KP=45) during the
extend + hold phases. The relaxed support gains let a second operator
demonstrate body posture adjustments — rear hip lower, sagittal weight
shift, lateral CoM shift — while the first operator continues to guide
the FR paw onto the button.

Two-person protocol
───────────────────

Person 1 — FR demonstrator
    - One hand on the FR paw at all times during extend + hold.
    - Light steering only: lateral correction (left/right), small terminal
      forward push to close the last few millimeters.
    - The forward motion of the FR paw should come primarily from Person 2's
      body adjustment, not Person 1's pull. Person 1 finishes the press,
      they don't drive it.

Person 2 — body demonstrator
    - Hands on the rear haunches or the pelvis. Fingertips and wrist only —
      not forearm or shoulder. Soft touch.
    - Demonstrate ONE body adjustment per episode for the cleanest signal:
      rear hip lower, sagittal shift forward, or lateral CoM shift toward
      the FR side. Vary across episodes, not within an episode.

Soft, no-resistance demonstration
─────────────────────────────────

The "soft, no-resistance demonstration produces good training data because
the leg's achieved position is determined by the controller commanding +
soft PD gain + gentle external force, in clean equilibrium. Hard
demonstration saturates the motor's torque limit and contaminates the
achieved-vs-commanded relationship, producing nonlinear, episode-specific
noise that the model can't learn from."

If either person feels resistance against their guidance, ease off — do
not push through. The robot is telling you the controller is fighting
back, which means the demonstration is no longer informative.

Synchronized release
────────────────────

When the audio cue fires (button is pressed), BOTH operators let go at
exactly the same moment. The script prints a release prompt at hold-end.
A staggered release leaves one leg's PD loop still resisting an external
force at the moment of gain restore, which is the single most likely way
to produce a hardware-jolt event during this protocol.

Output: data/real/stage_d_v3/episode_wholebody_YYYYMMDD_HHMMSS.h5
Schema: v2.1 superset + v2 fields + v3 additive fields
        (per_step/joint_kp_used, root attr gain_schedule="wholebody_v1_kpsupport45",
         root attr collection_mode="wholebody_guided").

Pre-session assumptions:
    - env_go2 conda env active
    - Ethernet interface up, robot powered on, Sport Mode available
    - USB mic plugged in and index known (use sd.query_devices())
    - Button placed against the wall within reach of the standoff
    - Two trained operators present, gloves on both
    - Two kill switches accessible (operator + helper)

First-time use of a new gain schedule:
    Pass --cardboard-test to lock the run to 2 episodes and print the
    first-time-on-new-gains warning. Always use this flag before regular
    collection on a new robot, new battery, new floor surface, etc.

Spec: Stage D v3 whole-body demonstration, April 2026.
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
    GO2_NOMINAL_MASS_KG,
)
from src.planner.heuristic_contact_wholebody import HeuristicContactWholeBody
from src.data.grounding_thread import GroundingThread
from src.data.recorder import StageDRecorder

from src.data.audio_recorder import AudioRecorder
from src.data.audio_live_detector import AudioLiveDetector
from src.data.color_detector import ColorDetector


logger = logging.getLogger("collect_wholebody")


# ──────────────────────────────────────────────
# Constants (mirrored from collect_guided.py)
# ──────────────────────────────────────────────

PROMPT              = "red button"
PRESS_OFFSET_X      = 0.593
PRESS_OFFSET_Y      = 0
NAV_EXTRA_FORWARD_M = 0.20
MIN_GROUNDING_CONF  = 0.5
AUDIO_SAMPLE_RATE   = 16000
AUDIO_CAPTURE_RATE  = 48000
AUDIO_BASELINE_S    = 2.0
SETTLE_AFTER_NAV_S  = 2.0
GROUNDING_RATE_HZ   = 5.0
COLOR_EXPECTED      = "red"
Y_SIT_STAND_DRIFT_COMP = 0.0

OUTPUT_ROOT = Path("data/real/stage_d_v3")


# ──────────────────────────────────────────────
# Episode config + summary
# ──────────────────────────────────────────────

@dataclass
class EpisodeConfig:
    episode_id:       str
    button_height_m:  float
    distance_m:       float
    angle_rad:        float
    perturbation_cmd: np.ndarray
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
    heuristic:        HeuristicContactWholeBody
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
    return OUTPUT_ROOT / f"episode_wholebody_{episode_id}.h5"


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


# ──────────────────────────────────────────────
# Banners
# ──────────────────────────────────────────────

def _print_safety_banner(cardboard_test: bool) -> None:
    """One-time banner before the first whole-body episode."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " STAGE D v3 — WHOLE-BODY HAND-GUIDED DEMONSTRATION".ljust(70) + "║")
    print("║" + "".ljust(70)                                                    + "║")
    print("║" + " Two-person protocol:".ljust(70)                                  + "║")
    print("║" + "   Person 1: hand on FR paw (light steering, terminal push).".ljust(70) + "║")
    print("║" + "   Person 2: hands on rear haunches or pelvis (gentle).".ljust(70) + "║")
    print("║" + "".ljust(70)                                                    + "║")
    print("║" + " Gain schedule: wholebody_v1_kpsupport45".ljust(70)              + "║")
    print("║" + "   FR leg:        KP=15 KD=1.5  (compliant, extend+hold)".ljust(70) + "║")
    print("║" + "   Support legs:  KP=45 KD=4.5  (compliant, extend+hold)".ljust(70) + "║")
    print("║" + "   All else:      KP=100 KD=6   (rigid)".ljust(70)               + "║")
    print("║" + "".ljust(70)                                                    + "║")
    print("║" + " The four key principles:".ljust(70)                              + "║")
    print("║" + "   1. Person 1 STEERS, doesn't drive — body brings FR forward.".ljust(70) + "║")
    print("║" + "   2. Person 2 uses fingertips/wrist only, not forearm/shoulder.".ljust(70) + "║")
    print("║" + "   3. If you feel resistance, EASE OFF — don't push through.".ljust(70) + "║")
    print("║" + "   4. SYNCHRONIZED RELEASE on audio cue — both let go at once.".ljust(70) + "║")
    print("║" + "".ljust(70)                                                    + "║")
    print("║" + " Checklist before first episode:".ljust(70)                       + "║")
    print("║" + "   1. Gloves on BOTH operators.".ljust(70)                        + "║")
    print("║" + "   2. Two kill switches accessible (one each).".ljust(70)         + "║")
    print("║" + "   3. Mic placement <50cm from button, pointed at it.".ljust(70)  + "║")
    print("║" + "   4. Quiet room during AudioLiveDetector baseline.".ljust(70)    + "║")
    print("║" + "   5. Confirm visually that support legs are RIGID at standoff.".ljust(70) + "║")
    print("║" + "   6. Button sound module working (test once manually).".ljust(70) + "║")
    print("║" + "".ljust(70)                                                    + "║")
    print("║" + " Compliance window = extend + hold only. Gain restore happens".ljust(70) + "║")
    print("║" + " from achieved positions, so legs don't jerk — but do NOT keep".ljust(70) + "║")
    print("║" + " hands on rear haunches once '[GUIDED] Contact detected' prints.".ljust(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()
    if cardboard_test:
        print("=" * 60)
        print("CARDBOARD TEST MODE — first session with new gain schedule.")
        print("Limited to 2 episodes. Verify safety before regular collection.")
        print("=" * 60)
        print()
    else:
        # First-time-on-new-gains advisory. The KP_SUPPORT_SOFT=45 value
        # has not been hardware-validated as of this commit; flag it
        # unless the operator explicitly acknowledged via --cardboard-test.
        print("⚠  Gain schedule wholebody_v1_kpsupport45 is NEW.")
        print("⚠  If this is the first session on these gains, consider")
        print("⚠  rerunning with --cardboard-test (limits to 2 episodes).")
        print()


def _print_phase_expectations() -> None:
    print("Expected terminal output per episode:")
    print("  [INFO] sit_to_stand → weight_shift → lift  (rigid gains, lift FR clears)")
    print("  [WHOLEBODY] Both FR and support legs soften during extend+hold.")
    print("  [GUIDED] Compliance active — guide FR paw to button. Audio will end hold.")
    print("  [INFO] hold ... (Person 1 finishes press, Person 2 demonstrates posture)")
    print("  [GUIDED] Contact detected via audio at step <N>")
    print("  *** RELEASE NOW — both people let go simultaneously ***")
    print("  [INFO] retract_curl → retract_rotate → retract_extend → weight_unshift → settle → sit")
    print()


def _print_episode_readiness(i: int, total: int) -> None:
    print()
    print("─" * 65)
    print(f"   Episode {i+1}/{total} — two-person readiness check")
    print("─" * 65)
    print("   Person 1 (FR demonstrator):  hand on FR paw, ready?")
    print("   Person 2 (body demonstrator): hands on rear haunches, ready?")
    print("─" * 65)
    input("Press Enter when both confirm (Ctrl+C to abort)...")


# ──────────────────────────────────────────────
# Release-cue watcher
# ──────────────────────────────────────────────

class _ReleaseCueWatcher:
    """
    Polls heuristic._phase and prints the synchronized-release banner the
    moment the controller leaves hold (i.e., enters retract_curl). The
    parent's [GUIDED] Contact detected message also fires at this transition;
    the release cue follows immediately.
    """
    def __init__(self, heuristic):
        self._heuristic = heuristic
        self._stop = None  # type: Optional[__import__("threading").Event]
        self._thread = None

    def start(self):
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="wholebody_release_cue", daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self):
        last = None
        announced = False
        while not self._stop.is_set():
            phase = self._heuristic._phase
            if (last == "hold" and phase == "retract_curl"
                    and not announced):
                print("*** RELEASE NOW — both people let go simultaneously ***")
                print("*** RELEASE NOW — both people let go simultaneously ***")
                print("*** RELEASE NOW — both people let go simultaneously ***")
                print("*** RELEASE NOW — both people let go simultaneously ***")
                print("*** RELEASE NOW — both people let go simultaneously ***")
                print("*** RELEASE NOW — both people let go simultaneously ***")
                announced = True
            last = phase
            time.sleep(0.01)


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

    logger.info(
        "Constructing HeuristicContactWholeBody (kp_support_soft=%.2f, "
        "kd_support_soft=%.2f, gravity_ff_enabled=%s, ff_body_mass=%.2f kg)…",
        args.rear_kp, args.rear_kd, args.gravity_ff, args.ff_body_mass,
    )
    heuristic = HeuristicContactWholeBody(
        network_interface=args.interface,
        already_initialized=True,
        kp_support_soft=args.rear_kp,
        kd_support_soft=args.rear_kd,
        gravity_ff_enabled=args.gravity_ff,
        gravity_ff_body_mass=args.ff_body_mass,
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
    logger.info(f"Whole-body episode {cfg.episode_id}")
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

    # Navigate to standoff
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

    # Background threads
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
        "k_p":                         0.0,
        "button_height_m":             float(cfg.button_height_m),
        "color_expected":              COLOR_EXPECTED,
        "notes":                       cfg.notes,
        "standoff_dxdy_commanded":     standoff_commanded,
        "press_offset_x":              np.float32(PRESS_OFFSET_X),
        "press_offset_y":              np.float32(PRESS_OFFSET_Y),
        "nav_extra_forward_m":         np.float32(NAV_EXTRA_FORWARD_M),
        "y_sit_stand_drift_comp":      np.float32(Y_SIT_STAND_DRIFT_COMP),
        # Collection regime — recorded so the trainer can filter v3 episodes
        # collected with gravity FF (clean body dynamics, only FR-side human
        # contribution) from earlier human-stabilized episodes. Support-leg
        # gains are also recorded since collection lowers them from the
        # historical 55/5.5 to 35/3.5 once FF supplies static torque.
        "gravity_ff_enabled":          bool(getattr(rt.heuristic, "_gravity_ff_enabled", False)),
        "gravity_ff_body_mass_kg":     np.float32(getattr(rt.heuristic, "_gravity_ff_body_mass", 0.0)),
        "kp_support_soft":             np.float32(getattr(rt.heuristic, "_kp_support_soft", 0.0)),
        "kd_support_soft":             np.float32(getattr(rt.heuristic, "_kd_support_soft", 0.0)),
    }
    recorder.begin_episode(initial_metadata)

    print("\n" + "=" * 60)
    print("Ready to begin contact phase.")
    print("  - Person 1 on FR side, hand on FR paw.")
    print("  - Person 2 behind robot, hands on rear haunches.")
    print("  - One operator hand on Ctrl+C; helper on second kill switch.")
    print("  - You'll see '[WHOLEBODY] ...' and '[GUIDED] Compliance active'")
    print("    when to begin guidance.")
    print("=" * 60)
    input("Press Enter to begin (Ctrl+C to abort)...\n")

    release_watcher = _ReleaseCueWatcher(heuristic)
    release_watcher.start()
    logger.info("heuristic.execute() — wall press, whole-body compliance on…")
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
        release_watcher.stop()
        return None
    finally:
        release_watcher.stop()

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
        "collection_mode":         "wholebody_guided",
    }

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
    print(f"  ── Whole-body episode {summary.episode_id}")
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
            use_v2_schema=True,   # v3 is a superset of v2 schema
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

    _print_safety_banner(cardboard_test=args.cardboard_test)
    _print_phase_expectations()
    input("Press Enter to start first episode (Ctrl+C to abort)...")

    for i in range(args.num_episodes):
        print()
        print("#" * 72)
        print(f"# WHOLE-BODY EPISODE {i+1}/{args.num_episodes}")
        print("#" * 72)
        if i > 0:
            print("Between-episode reset:")
            print("  - Button visible and upright.")
            print("  - Robot back near approach mark if drifted.")
            print("  - Gloves still on; both kill switches still in reach.")
        _print_episode_readiness(i, args.num_episodes)

        episode_id = _make_episode_id()
        output_path = _make_output_path(episode_id)
        cfg = EpisodeConfig(
            episode_id=episode_id,
            button_height_m=args.button_height,
            distance_m=args.distance,
            angle_rad=args.angle,
            perturbation_cmd=np.zeros(2, dtype=np.float32),
            output_path=output_path,
            notes=f"wholebody ep{i+1}/{args.num_episodes}",
        )
        s = run_one_episode(rt, cfg)
        if s is None:
            aborted += 1
            continue
        kept.append(s)

    print()
    print("=" * 72)
    print(f"WHOLE-BODY SESSION — {len(kept)}/{args.num_episodes} ran, {aborted} aborted")
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
        description="Stage D v3 whole-body hand-guided demonstration collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interface",         type=str, default=DEFAULT_INTERFACE,
                   help="Ethernet interface for DDS.")
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
    p.add_argument("--num-episodes",      type=int, default=5,
                   help="Whole-body sessions are slower; default is small.")
    p.add_argument("--perturbation",      type=str, choices=["none"], default="none",
                   help="Only 'none' is supported.")
    p.add_argument("--cardboard-test",    action="store_true",
                   help="First-time-on-new-gain-schedule mode. Caps run at "
                        "2 episodes and prints an extra-loud warning.")
    p.add_argument("--rear-kp",           type=float, default=35.0,
                   help="kp_support_soft for FL/RR/RL during extend+hold. "
                        "Default 35 (collection-only — lower than the legacy "
                        "55 because gravity FF now supplies the static "
                        "torque, leaving PD to handle damping and intentional "
                        "operator translations). Run a pre-collection sanity "
                        "test with Person 2's hands off before committing to "
                        "a session; if rear sags or the body resists Person "
                        "2's translations, retune.")
    p.add_argument("--rear-kd",           type=float, default=3.5,
                   help="kd_support_soft for FL/RR/RL during extend+hold. "
                        "Default 3.5 (preserves the 0.1 KD/KP ratio of the "
                        "legacy 55/5.5 regime).")
    p.add_argument("--gravity-ff",        action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply dynamic gravity-comp FF on FL/RR/RL during "
                        "lift/extend/hold/retract_*. Default True for "
                        "collection — body holds posture without Person 2 "
                        "supporting the rear, so achieved_delta_q reflects "
                        "only FR-side demonstration. Pass --no-gravity-ff to "
                        "fall back to the legacy human-stabilized regime.")
    p.add_argument("--ff-body-mass",      type=float,
                   default=GO2_NOMINAL_MASS_KG,
                   help="Body mass (kg) used to scale gravity FF torques. "
                        f"Default {GO2_NOMINAL_MASS_KG}; raise if rear still "
                        f"sags during pre-collection sanity check, lower if "
                        f"FF overshoots.")
    p.add_argument("--verbose",           action="store_true")
    return p


def main() -> int:
    global _runtime_handle
    args = _build_arg_parser().parse_args()

    if args.cardboard_test:
        args.num_episodes = min(args.num_episodes, 2)

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
