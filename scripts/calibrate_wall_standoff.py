#!/usr/bin/env python
"""
scripts/calibrate_wall_standoff.py

Wall-press standoff + offset calibration tool.

Two things are uncalibrated for wall press:
  1. Where the robot should physically stand (standoff position).
  2. What FR_LIFT_OFFSET_WALL and FR_EXTEND_OFFSET_WALL should be so that
     the wall-press motion actually reaches the button from that standoff.

This script tunes BOTH iteratively and interactively:

  - Live offsets: FR_LIFT_OFFSET_WALL and FR_EXTEND_OFFSET_WALL are held
    as mutable runtime state. Between attempts, the user can adjust any
    component (hip/thigh/calf) of either offset via a guided prompt.
  - Per-attempt flow: user manually positions the robot, script takes a
    grounding reading, runs ONE wall press with the current runtime
    offsets, user visually confirms whether the FR paw reached the button,
    outcome is logged to CSV.
  - Persistence: offsets can be saved to/loaded from JSON between sessions.

Requires the heuristic_contact.py patch that adds runtime offset kwargs
to execute().  See heuristic_contact_PATCH_NOTES.md.

Usage:
    python scripts/calibrate_wall_standoff.py --interface enx98fc84e68f1a

Output:
    data/calibration/wall_standoff_<timestamp>.csv       — attempt log
    data/calibration/offsets_<timestamp>.json            — saved offsets
"""

import argparse
import csv
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.robot.go2_interface import Go2Interface
from src.perception.grounding import Go2Camera, VisualGrounder
from src.planner.heuristic_contact import (
    HeuristicContact,
    DEFAULT_INTERFACE,
)


logger = logging.getLogger("calibrate_wall_standoff")


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PROMPT = "red button"
MIN_CONF = 0.7                   # confidence gate before attempting a press
CONTACT_PROXIMITY_LOOSE = 0.10   # intentionally loose — human judges ground truth
OUTPUT_DIR = Path("data/calibration")

# Path A conservative starting offsets — lift paw modestly, reach forward
# with a medium extension. Designed to reliably under-reach on the first
# attempt so the failure mode is "misses button" not "slams wall".
#
# Tune upward from here:
#   - More negative lift thigh → paw lifts higher up (under body)
#   - More negative extend thigh → paw reaches further forward
#   - More positive extend calf → calf straightens, extending paw's x-reach
DEFAULT_FR_LIFT_OFFSET_WALL   = np.array([+0.04, -0.4, -0.3], dtype=np.float64)
DEFAULT_FR_EXTEND_OFFSET_WALL = np.array([+0.04, -1.2, +0.2], dtype=np.float64)


# ──────────────────────────────────────────────
# Runtime offset state
# ──────────────────────────────────────────────

@dataclass
class WallOffsets:
    """Mutable runtime state for wall-press FR leg offsets."""
    lift:   np.ndarray  # [hip, thigh, calf]  — rad
    extend: np.ndarray  # [hip, thigh, calf]  — rad

    @classmethod
    def defaults(cls) -> "WallOffsets":
        return cls(
            lift=DEFAULT_FR_LIFT_OFFSET_WALL.copy(),
            extend=DEFAULT_FR_EXTEND_OFFSET_WALL.copy(),
        )

    def as_dict(self) -> dict:
        return {
            "fr_lift_offset_wall":   self.lift.tolist(),
            "fr_extend_offset_wall": self.extend.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WallOffsets":
        return cls(
            lift=np.asarray(d["fr_lift_offset_wall"],   dtype=np.float64),
            extend=np.asarray(d["fr_extend_offset_wall"], dtype=np.float64),
        )

    def display(self) -> str:
        h, t, c = self.lift
        lift_s = f"[{h:+.3f}, {t:+.3f}, {c:+.3f}]"
        h, t, c = self.extend
        ext_s  = f"[{h:+.3f}, {t:+.3f}, {c:+.3f}]"
        return (
            f"  FR_LIFT_OFFSET_WALL    = {lift_s}\n"
            f"  FR_EXTEND_OFFSET_WALL  = {ext_s}"
        )


# ──────────────────────────────────────────────
# Per-attempt log record
# ──────────────────────────────────────────────

@dataclass
class CalibAttempt:
    timestamp:           str
    target_x:            float
    target_y:            float
    target_z:            float
    confidence:          float
    depth_m:             float
    # Offsets used for THIS attempt (captured per-attempt so we can later
    # find exactly which offset values produced successful presses)
    lift_hip:            float
    lift_thigh:          float
    lift_calf:           float
    extend_hip:          float
    extend_thigh:        float
    extend_calf:         float
    fk_success:          bool
    fk_contact_step:     int
    contact_method:      str
    duration_s:          float
    human_success:       str   # "y" / "p" / "n"
    notes:               str


# ──────────────────────────────────────────────
# Signal handling
# ──────────────────────────────────────────────

_runtime_ref: dict = {}


def _sigint_handler(signum, frame):
    print("\n\n── Ctrl+C — cleaning up ──")
    go2 = _runtime_ref.get("go2")
    heuristic = _runtime_ref.get("heuristic")
    if heuristic is not None:
        try:
            heuristic._stop_flag.set()
        except Exception:
            pass
    if go2 is not None:
        try:
            go2.sit()
        except Exception:
            pass
    print("Cleanup attempted. Exiting.")
    sys.exit(130)


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def _prompt_choice(prompt: str, choices: list[str],
                   default: Optional[str] = None) -> str:
    choice_str = "/".join(c.upper() if c == default else c for c in choices)
    while True:
        suffix = f" [{choice_str}] " if default else f" [{'/'.join(choices)}] "
        ans = input(prompt + suffix).strip().lower()
        if not ans and default:
            return default
        if ans in choices:
            return ans


def _prompt_float(prompt: str, default: Optional[float] = None) -> Optional[float]:
    """Prompt for a float. Returns default on empty input."""
    suffix = f" [{default}] " if default is not None else " "
    while True:
        ans = input(prompt + suffix).strip()
        if not ans:
            return default
        try:
            return float(ans)
        except ValueError:
            print(f"  Invalid number: {ans!r}. Try again or Ctrl+C to abort.")


# ──────────────────────────────────────────────
# Offset editing UI
# ──────────────────────────────────────────────

def edit_offsets(offsets: WallOffsets) -> None:
    """
    Three-prompt edit: which offset (lift/extend), which component
    (hip/thigh/calf), new value. Mutates the passed WallOffsets in place.
    """
    print()
    print("── Edit offsets ──")
    print(offsets.display())
    print()

    # Step 1: which offset
    which = _prompt_choice(
        "Which offset to adjust?  (l=lift, e=extend, c=cancel)",
        ["l", "e", "c"], default="c",
    )
    if which == "c":
        return

    arr = offsets.lift if which == "l" else offsets.extend
    which_name = "LIFT" if which == "l" else "EXTEND"

    # Step 2: which component
    comp = _prompt_choice(
        f"Which component of {which_name}?  (h=hip, t=thigh, c=calf, x=cancel)",
        ["h", "t", "c", "x"], default="x",
    )
    if comp == "x":
        return

    idx = {"h": 0, "t": 1, "c": 2}[comp]
    comp_name = {"h": "hip", "t": "thigh", "c": "calf"}[comp]

    # Step 3: new value
    current = float(arr[idx])
    print(f"  Current {which_name} {comp_name} offset: {current:+.3f} rad")
    new_val = _prompt_float(f"  New value?", default=current)
    if new_val is None:
        return

    # Sanity clamp to plausible range for joint offsets
    if abs(new_val) > 3.0:
        print(f"  Value {new_val} looks too large (abs > 3.0 rad). "
              f"Cancelling edit — re-enter with a smaller magnitude.")
        return

    # Warn on large single-step changes — precedent of KP_FR_LIFT=80 incident
    delta = abs(new_val - current)
    if delta > 0.2:
        print(f"  WARNING: change of {delta:.3f} rad is large. "
              f"Previous gain experiment cracked the Ethernet port.")
        confirm = _prompt_choice("  Apply anyway?", ["y", "n"], default="n")
        if confirm != "y":
            print("  Edit cancelled.")
            return

    arr[idx] = new_val
    print()
    print("── Updated offsets ──")
    print(offsets.display())
    print()


# ──────────────────────────────────────────────
# Persistence: save/load offsets JSON
# ──────────────────────────────────────────────

def save_offsets(offsets: WallOffsets, path: Path,
                 note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note":      note,
        **offsets.as_dict(),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved offsets to {path}")


def find_most_recent_offsets_file(dir_path: Path) -> Optional[Path]:
    """Return the newest offsets_*.json in dir_path, or None if none exist."""
    if not dir_path.is_dir():
        return None
    candidates = sorted(
        dir_path.glob("offsets_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_offsets(path: Path) -> WallOffsets:
    with path.open() as f:
        d = json.load(f)
    return WallOffsets.from_dict(d)


def prompt_resume_or_reset(dir_path: Path) -> WallOffsets:
    """Startup prompt: resume from last saved offsets, or reset to defaults."""
    most_recent = find_most_recent_offsets_file(dir_path)

    if most_recent is None:
        print("No previous offsets file found — starting from defaults.")
        return WallOffsets.defaults()

    mtime = datetime.fromtimestamp(most_recent.stat().st_mtime)
    mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")

    try:
        saved = load_offsets(most_recent)
    except Exception as e:
        print(f"Failed to load {most_recent}: {e}")
        print("Starting from defaults.")
        return WallOffsets.defaults()

    print()
    print(f"Found previous offsets: {most_recent.name}")
    print(f"  Last modified: {mtime_str}")
    print("  Values:")
    print(saved.display())
    print()
    choice = _prompt_choice(
        "Resume with these offsets or reset to defaults?  (r=resume, d=default)",
        ["r", "d"], default="r",
    )
    if choice == "r":
        print("Resuming with saved offsets.")
        return saved
    print("Starting from defaults.")
    return WallOffsets.defaults()


# ──────────────────────────────────────────────
# Grounding
# ──────────────────────────────────────────────

def _take_stable_grounding(camera, grounder, min_conf=MIN_CONF,
                           n_samples=5):
    samples = []
    for _ in range(n_samples):
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.3)
            continue
        r = grounder.ground(frame, prompt=PROMPT)
        if r is None or r.position_base is None or r.confidence < min_conf:
            time.sleep(0.3)
            continue
        samples.append((r.position_base, r.confidence, r.depth_m))
        time.sleep(0.2)

    if len(samples) < 3:
        return None, None, None

    positions = np.stack([s[0] for s in samples])
    confidences = np.array([s[1] for s in samples])
    depths = np.array([s[2] if s[2] is not None else np.nan for s in samples])

    std_x = float(np.std(positions[:, 0]))
    std_y = float(np.std(positions[:, 1]))
    std_z = float(np.std(positions[:, 2]))
    if std_x > 0.02 or std_y > 0.02 or std_z > 0.02:
        logger.warning(
            f"Unstable grounding: std=(x={std_x*100:.1f}cm, "
            f"y={std_y*100:.1f}cm, z={std_z*100:.1f}cm)"
        )
        return None, None, None

    median_pos = np.median(positions, axis=0)
    median_conf = float(np.median(confidences))
    median_depth = (
        float(np.nanmedian(depths))
        if not np.all(np.isnan(depths))
        else float("nan")
    )
    return median_pos, median_conf, median_depth


# ──────────────────────────────────────────────
# Per-attempt procedure
# ──────────────────────────────────────────────

def run_one_attempt(
    attempt_idx: int,
    go2: Go2Interface,
    camera: Go2Camera,
    grounder: VisualGrounder,
    heuristic: HeuristicContact,
    offsets: WallOffsets,
) -> Optional[CalibAttempt]:
    print()
    print("=" * 72)
    print(f"Attempt {attempt_idx}")
    print("=" * 72)
    print("Offsets for this attempt:")
    print(offsets.display())
    print()

    # Step 1: manual positioning
    print("Step 1: Place the robot on the floor facing the button.")
    print("  - Robot should be SITTING before you move it.")
    print("  - Pick up and place, facing the button.")
    input("Press Enter once robot is positioned and hands-off...")

    # Step 2: stand and take grounding reading
    print()
    print("Step 2: Standing up briefly to take a grounding reading…")
    try:
        go2.stand()
        time.sleep(1.5)
    except Exception as e:
        print(f"go2.stand() failed: {e}")
        return None

    print("Taking 5 grounding samples (median filtering)…")
    position, conf, depth = _take_stable_grounding(camera, grounder)
    if position is None:
        print("  ✗ Could not get stable grounding.")
        print("    Reposition so button is more centered / closer, then retry.")
        try:
            go2.sit()
        except Exception:
            pass
        return None

    print(f"  ✓ Grounding:  target = ({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})")
    print(f"             conf = {conf:.3f}  depth = {depth:.3f}m")

    # Step 3: confirm proceed
    print()
    print("Step 3: About to run WALL PRESS.")
    print(f"  FR_LIFT_OFFSET_WALL    = {offsets.lift.tolist()}")
    print(f"  FR_EXTEND_OFFSET_WALL  = {offsets.extend.tolist()}")
    print("  Watch the FR paw during extend — does it reach the button?")
    choice = _prompt_choice(
        "Proceed?  (y=proceed, r=redo grounding, n=skip & sit)",
        ["y", "r", "n"], default="y",
    )
    if choice != "y":
        try:
            go2.sit()
        except Exception:
            pass
        return None

    # Step 4: execute wall press with current runtime offsets
    print()
    print("Step 4: Running wall press with current offsets. Watch carefully.")
    t0 = time.time()
    try:
        result = heuristic.execute(
            target_offset_xyz=position,
            press_mode="wall",
            contact_proximity_m=CONTACT_PROXIMITY_LOOSE,
            use_foot_force=False,
            audio_detector=None,
            # NEW: pass runtime overrides directly into execute()
            fr_lift_offset_wall=offsets.lift,
            fr_extend_offset_wall=offsets.extend,
        )
    except TypeError as e:
        # Most likely cause: heuristic_contact.py patch not applied.
        print()
        print("  ✗ heuristic.execute() rejected the runtime offset kwargs.")
        print("    Most likely the heuristic_contact.py patch is not applied.")
        print("    See heuristic_contact_PATCH_NOTES.md.")
        print(f"    Original error: {e}")
        try:
            go2.sit()
        except Exception:
            pass
        sys.exit(2)
    except Exception as e:
        print(f"  ✗ heuristic.execute() raised: {e}")
        try:
            go2.sit()
        except Exception:
            pass
        return None
    duration = time.time() - t0

    print()
    print(f"  Result:")
    print(f"    fk_success:      {result.success}     (ignore — FK unreliable for wall)")
    print(f"    contact_method:  {result.contact_method}")
    print(f"    contact_step:    {result.contact_step}")
    print(f"    duration:        {duration:.1f}s")

    # Step 5: human assessment
    print()
    print("Step 5: What did you SEE during extend/hold?")
    print("  y = paw reached the button (clear contact)")
    print("  p = partial (close miss or grazed)")
    print("  n = miss (paw didn't reach button or went past it)")
    human = _prompt_choice("Human visual assessment:", ["y", "p", "n"], default="n")

    notes = input("Notes (what to adjust next — e.g. 'paw 5cm short', 'over-reached 3cm'): ").strip()

    return CalibAttempt(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        target_x=float(position[0]),
        target_y=float(position[1]),
        target_z=float(position[2]),
        confidence=float(conf),
        depth_m=float(depth),
        lift_hip=float(offsets.lift[0]),
        lift_thigh=float(offsets.lift[1]),
        lift_calf=float(offsets.lift[2]),
        extend_hip=float(offsets.extend[0]),
        extend_thigh=float(offsets.extend[1]),
        extend_calf=float(offsets.extend[2]),
        fk_success=bool(result.success),
        fk_contact_step=int(result.contact_step),
        contact_method=str(result.contact_method),
        duration_s=float(duration),
        human_success=human,
        notes=notes,
    )


# ──────────────────────────────────────────────
# CSV + summary
# ──────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp",
    "target_x", "target_y", "target_z",
    "confidence", "depth_m",
    "lift_hip", "lift_thigh", "lift_calf",
    "extend_hip", "extend_thigh", "extend_calf",
    "fk_success", "fk_contact_step", "contact_method",
    "duration_s",
    "human_success", "notes",
]


def write_csv(attempts: list[CalibAttempt], path: Path) -> None:
    if not attempts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for a in attempts:
            d = asdict(a)
            for k, v in list(d.items()):
                if isinstance(v, float):
                    d[k] = f"{v:.4f}"
            writer.writerow(d)


def print_summary(attempts: list[CalibAttempt]) -> None:
    if not attempts:
        print("\nNo attempts logged.\n")
        return

    print()
    print("=" * 100)
    print(f"  CALIBRATION SUMMARY — {len(attempts)} attempts")
    print("=" * 100)
    print(f"  {'#':>3}  {'target_xy':>18}  {'lift_thigh':>10} {'lift_calf':>9}  "
          f"{'ext_thigh':>9} {'ext_calf':>9}  {'hum':>3}  notes")
    print("-" * 100)
    for i, a in enumerate(attempts, 1):
        tgt_str = f"({a.target_x:+.2f},{a.target_y:+.2f})"
        print(
            f"  {i:>3}  {tgt_str:>18}  "
            f"{a.lift_thigh:>+10.3f} {a.lift_calf:>+9.3f}  "
            f"{a.extend_thigh:>+9.3f} {a.extend_calf:>+9.3f}  "
            f"{a.human_success:>3}  {a.notes[:36]}"
        )
    print()

    successes = [a for a in attempts if a.human_success == "y"]
    if successes:
        xs = np.array([a.target_x for a in successes])
        ys = np.array([a.target_y for a in successes])
        zs = np.array([a.target_z for a in successes])
        lift_thighs = np.array([a.lift_thigh for a in successes])
        ext_thighs  = np.array([a.extend_thigh for a in successes])
        ext_calves  = np.array([a.extend_calf for a in successes])
        print(f"  {len(successes)} SUCCESSFUL attempts — convergence:")
        print(f"    target (robot's view of button):")
        print(f"      median x = {np.median(xs):+.3f}  "
              f"(range {xs.min():+.3f} to {xs.max():+.3f})")
        print(f"      median y = {np.median(ys):+.3f}")
        print(f"      median z = {np.median(zs):+.3f}")
        print(f"    offsets (what reached the button):")
        print(f"      lift  thigh ≈ {np.median(lift_thighs):+.3f}")
        print(f"      ext   thigh ≈ {np.median(ext_thighs):+.3f}")
        print(f"      ext   calf  ≈ {np.median(ext_calves):+.3f}")
        print()
        print(f"  → Use 'v' to save current offsets, then update")
        print(f"    heuristic_contact.py defaults before data collection.")
    else:
        print("  No successful attempts yet. Keep iterating.")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate wall-press offsets + standoff.")
    ap.add_argument("--interface", type=str, default=DEFAULT_INTERFACE)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for name in ("root", "sam2", "groundingdino"):
        logging.getLogger(name).setLevel(logging.WARNING)

    print()
    print("╔" + "═" * 70 + "╗")
    print("║  WALL OFFSETS + STANDOFF CALIBRATION".ljust(71)                    + "║")
    print("║".ljust(71)                                                         + "║")
    print("║  FR_LIFT_OFFSET_WALL and FR_EXTEND_OFFSET_WALL are held as live".ljust(71)+ "║")
    print("║  runtime state — edit between attempts without restarting.".ljust(71) + "║")
    print("║".ljust(71)                                                         + "║")
    print("║  SAFETY:".ljust(71)                                                 + "║")
    print("║   - Start on the floor, not a table".ljust(71)                     + "║")
    print("║   - Start CONSERVATIVE — under-reach first, tune up".ljust(71)     + "║")
    print("║   - If paw looks like it's going to hit the wall, Ctrl+C".ljust(71) + "║")
    print("║   - Never raise a thigh-offset by more than 0.2 rad at a time".ljust(71)+ "║")
    print("║     (precedent: KP_FR_LIFT=80 incident cracked Ethernet port)".ljust(71)+ "║")
    print("╚" + "═" * 70 + "╝")
    print()

    # Resume or reset offsets
    offsets = prompt_resume_or_reset(OUTPUT_DIR)

    # DDS init (once)
    logger.info(f"ChannelFactoryInitialize on {args.interface}…")
    ChannelFactoryInitialize(0, args.interface)

    # Build objects
    logger.info("Constructing Go2Interface…")
    go2 = Go2Interface(network_interface=args.interface, already_initialized=True)
    logger.info("Constructing Go2Camera…")
    camera = Go2Camera(already_initialized=True)
    logger.info("Constructing VisualGrounder (loading models, ~10–15s)…")
    grounder = VisualGrounder()
    logger.info("Constructing HeuristicContact (base — NOT corrective)…")
    heuristic = HeuristicContact(
        network_interface=args.interface,
        already_initialized=True,
    )

    _runtime_ref["go2"] = go2
    _runtime_ref["heuristic"] = heuristic
    signal.signal(signal.SIGINT, _sigint_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"wall_standoff_{timestamp}.csv"
    offsets_path = OUTPUT_DIR / f"offsets_{timestamp}.json"

    attempts: list[CalibAttempt] = []

    try:
        attempt_idx = 1
        while True:
            print()
            if attempts:
                n_ok = sum(1 for a in attempts if a.human_success == "y")
                n_p  = sum(1 for a in attempts if a.human_success == "p")
                print(f"── So far: {len(attempts)} attempts  "
                      f"({n_ok} success, {n_p} partial)")
            print()
            print("── Current offsets ──")
            print(offsets.display())
            print()
            choice = _prompt_choice(
                f"Next action?  (a=attempt {attempt_idx}, e=edit offsets, "
                f"v=save offsets, s=summary, q=quit)",
                ["a", "e", "v", "s", "q"], default="a",
            )

            if choice == "q":
                break
            if choice == "e":
                edit_offsets(offsets)
                continue
            if choice == "v":
                save_offsets(offsets, offsets_path,
                             note=f"after {len(attempts)} attempts")
                continue
            if choice == "s":
                print_summary(attempts)
                continue

            # choice == "a" — run an attempt
            result = run_one_attempt(
                attempt_idx, go2, camera, grounder, heuristic, offsets,
            )
            if result is not None:
                attempts.append(result)
                write_csv(attempts, csv_path)   # incremental save
                print(f"  Logged attempt {attempt_idx}: "
                      f"human={result.human_success}")
                # Auto-save offsets after a successful attempt so we don't
                # lose the values that worked.
                if result.human_success == "y":
                    save_offsets(offsets, offsets_path,
                                 note=f"after attempt {attempt_idx} "
                                      f"(human=y)")
                attempt_idx += 1
    finally:
        print_summary(attempts)
        write_csv(attempts, csv_path)
        save_offsets(offsets, offsets_path,
                     note=f"final after {len(attempts)} attempts")
        try:
            go2.sit()
        except Exception as e:
            logger.warning(f"Final sit failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())