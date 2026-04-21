#!/usr/bin/env python
"""
scripts/measure_standoff.py

Interactive live-grounding display. Loops at ~5 Hz:
  - Grab frame from Go2Camera
  - Run GroundingDINO + SAM2 + depth prediction
  - Print target_pos_base and computed standoff = pos - (PRESS_OFFSET_X, PRESS_OFFSET_Y, 0)

Use this while manually positioning the robot for:
  - K_p tuning Episode A (ideal standoff)
  - Stage C collection at specified button placements

When the standoff prints approximately (0, 0, z_button), the robot is at
the ideal standoff.

Usage:
    python scripts/measure_standoff.py
    python scripts/measure_standoff.py --interface enx98fc84e68f1a --prompt "red button"

Spec: training_data_spec.md v2.1, Section 9.7
"""

import argparse
import logging
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.perception.grounding import Go2Camera, VisualGrounder
from src.planner.heuristic_contact import DEFAULT_INTERFACE


PRESS_OFFSET_X = 0.203   # m — forward of FR foot at standing
PRESS_OFFSET_Y = 0.140   # m — left of FR foot at standing

logger = logging.getLogger("measure_standoff")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live standoff-measurement tool.")
    ap.add_argument("--interface", type=str, default=DEFAULT_INTERFACE)
    ap.add_argument("--prompt", type=str, default="red button")
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="Minimum grounding confidence to print a reading.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"ChannelFactoryInitialize on {args.interface}…")
    ChannelFactoryInitialize(0, args.interface)

    logger.info("Constructing Go2Camera…")
    camera = Go2Camera(already_initialized=True)

    logger.info("Constructing VisualGrounder (loads models, ~10–15s)…")
    grounder = VisualGrounder()

    period = 1.0 / args.rate_hz

    print()
    print("=" * 72)
    print(f"Live standoff measurement — prompt: '{args.prompt}'")
    print("=" * 72)
    print()
    print(f"Ideal standoff: x≈{PRESS_OFFSET_X:+.3f}  y≈{PRESS_OFFSET_Y:+.3f}")
    print(f"(position robot so (standoff_x, standoff_y) ≈ (0, 0))")
    print()
    print(f"{'time':>8}  {'conf':>5}  "
          f"{'target_x':>9} {'target_y':>9} {'target_z':>9}  "
          f"{'off_x':>7} {'off_y':>7}  depth(m)")
    print("-" * 80)

    t0 = time.monotonic()
    try:
        while True:
            t1 = time.monotonic()
            frame = camera.get_frame()
            if frame is None:
                print(f"{time.monotonic()-t0:>8.1f}  (no frame)")
                time.sleep(period)
                continue

            result = grounder.ground(frame, prompt=args.prompt)
            if result is None:
                print(f"{time.monotonic()-t0:>8.1f}  (no detection)")
            elif result.position_base is None:
                print(f"{time.monotonic()-t0:>8.1f}  conf={result.confidence:.2f}  "
                      f"(depth too close / unreliable)")
            elif result.confidence < args.min_conf:
                print(f"{time.monotonic()-t0:>8.1f}  conf={result.confidence:.2f}  "
                      f"(below min_conf {args.min_conf})")
            else:
                pos = result.position_base
                off_x = float(pos[0]) - PRESS_OFFSET_X
                off_y = float(pos[1]) - PRESS_OFFSET_Y
                depth = float(result.depth_m) if result.depth_m is not None else float("nan")
                print(
                    f"{time.monotonic()-t0:>8.1f}  {result.confidence:>5.2f}  "
                    f"{pos[0]:>+9.3f} {pos[1]:>+9.3f} {pos[2]:>+9.3f}  "
                    f"{off_x:>+7.3f} {off_y:>+7.3f}  {depth:.3f}"
                )

            elapsed = time.monotonic() - t1
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())