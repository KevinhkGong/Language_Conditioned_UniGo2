"""Minimal hardware test wrapper for the ground-button heuristic controller.

Examples:
    python test_heuristic_contact.py --phase weight_shift
    python test_heuristic_contact.py --phase prepress
    python test_heuristic_contact.py --phase full
"""

from __future__ import annotations

import argparse
import logging
import time
import numpy as np

from src.planner.heuristic_contact import HeuristicContact

PHASE_PRESETS = {
    "stand": "weight_shift",
    "weight_shift": "weight_shift",
    "lift": "lift",
    "prepress": "move_to_prepress",
    "press": "descend_press",
    "full": None,
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase-by-phase test wrapper for heuristic_contact")
    parser.add_argument("--interface", type=str, default="enx98fc84e68f1a")
    parser.add_argument("--phase", type=str, choices=list(PHASE_PRESETS.keys()), default="weight_shift")
    parser.add_argument("--target-x", type=float, default=0.19)
    parser.add_argument("--target-y", type=float, default=-0.11)
    parser.add_argument("--target-z", type=float, default=-0.295)
    parser.add_argument("--contact-proximity", type=float, default=0.05)
    args = parser.parse_args()

    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)
    stop_after = PHASE_PRESETS[args.phase]

    print("\n" + "=" * 72)
    print("test_heuristic_contact.py")
    print("=" * 72)
    print(f"phase preset:       {args.phase}")
    print(f"stop_after_phase:   {stop_after}")
    print(f"button center/body: ({args.target_x:+.3f}, {args.target_y:+.3f}, {args.target_z:+.3f})")
    input("Press Enter to continue (Ctrl+C to abort)...\n")

    controller = HeuristicContact(network_interface=args.interface)
    result = controller.execute(
        target_offset_xyz=target,
        contact_proximity_m=args.contact_proximity,
        stop_after_phase=stop_after,
    )

    print("\nTest result:")
    print(f"  success:         {result.success}")
    print(f"  timeout:         {result.timeout}")
    print(f"  final_phase:     {result.final_phase}")
    print(f"  timeout_reason:  {result.timeout_reason}")
    print(f"  press_detected:  {result.press_detected}")
    print(f"  duration:        {result.duration_s:.2f}s")
    if result.debug_trace:
        last = result.debug_trace[-1]
        print("  last_debug:")
        print(f"    phase/step:    {last['phase']} / {last['phase_step']}")
        print(f"    foot_act:      ({last['foot'][0]:+.3f}, {last['foot'][1]:+.3f}, {last['foot'][2]:+.3f})")
        if last.get("cmd_foot") is not None:
            print(f"    foot_cmd:      ({last['cmd_foot'][0]:+.3f}, {last['cmd_foot'][1]:+.3f}, {last['cmd_foot'][2]:+.3f})")
        if last.get("cmd_foot_target") is not None:
            print(f"    foot_tgt:      ({last['cmd_foot_target'][0]:+.3f}, {last['cmd_foot_target'][1]:+.3f}, {last['cmd_foot_target'][2]:+.3f})")
        print(f"    gate:          {last['gate']}")
    if result.safe_hold_active:
        print("\nSafe hold is active. The controller is still publishing the hold pose.")
        print("Use the physical controller / manual recovery first, then press Ctrl+C here.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nExiting while safe hold was active.")


if __name__ == "__main__":
    main()
