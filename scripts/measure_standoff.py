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


HEADER_LINE = (
    f"{'time':>8}  {'conf':>5}  "
    f"{'target_x':>9} {'target_y':>9} {'target_z':>9}  "
    f"{'off_x':>7} {'off_y':>7}  depth(m)"
)
SEPARATOR = "-" * 80


def _print_header() -> None:
    print()
    print(SEPARATOR)
    print(HEADER_LINE)
    print(SEPARATOR)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live standoff-measurement tool.")
    ap.add_argument("--interface", type=str, default=DEFAULT_INTERFACE)
    ap.add_argument("--prompt", type=str, default="red button")
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="Minimum grounding confidence to print a reading.")
    ap.add_argument("--header-every", type=int, default=10,
                    help="Re-print the column header every N rows "
                         "(0 to disable).")
    ap.add_argument("--show-library-logs", action="store_true",
                    help="Show internal GroundingDINO/SAM2 log messages "
                         "(suppressed by default for cleaner output).")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG logging on this script's logger.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress noisy library loggers unless --show-library-logs is set.
    # Default to quiet so the data table stays readable. These are the
    # loggers SAM2 / GroundingDINO emit on each call.
    if not args.show_library_logs:
        for name in ("root", "sam2", "groundingdino"):
            logging.getLogger(name).setLevel(logging.WARNING)

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
    print(f"Ideal standoff: target_x≈{PRESS_OFFSET_X:+.3f}  "
          f"target_y≈{PRESS_OFFSET_Y:+.3f}")
    print(f"(position robot so (off_x, off_y) ≈ (0, 0))")
    _print_header()

    t0 = time.monotonic()
    row_count = 0

    try:
        while True:
            t1 = time.monotonic()
            frame = camera.get_frame()
            if frame is None:
                print(f"{time.monotonic()-t0:>8.1f}  (no frame)")
                row_count += 1
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
            row_count += 1

            # Reprint header every N rows so it stays on-screen in the terminal
            if args.header_every > 0 and row_count % args.header_every == 0:
                _print_header()

            elapsed = time.monotonic() - t1
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())