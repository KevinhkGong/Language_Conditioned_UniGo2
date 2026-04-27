"""
src/planner/heuristic_contact_guided.py

Compliance-guided subclass of HeuristicContact for Stage D v2 kinesthetic
demonstration collection.

The only control-logic difference vs. the base class is that FR leg gains
drop to KP_FR_COMPLIANT / KD_FR_COMPLIANT during the extend and hold phases.
That softening is implemented entirely in the base class _send_cmd and keyed
off self._compliance_active — which this subclass flips to True at __init__.

No control loop override. The parent still computes nominal FR waypoints;
with soft gains the human operator physically pushes the paw through them
to make contact with the button. Audio detection (wired via the standard
execute(audio_detector=...) path) ends the hold phase the same way FK
proximity does for the Jacobian-PID expert.

Operator flow per episode:
    1. Robot sits → stands → weight_shift → lift   (normal gains, leg rises)
    2. extend begins  → gains drop, operator guides paw toward button
    3. hold            → operator presses paw onto button
    4. audio spike or 3s timeout  → hold ends, gains snap back to KP_FR
    5. retract_curl    → _retract_start = list(actual), so the gain restore
                         does NOT jerk the leg (same mechanism as v2.0 fix)
    6. retract / unshift / settle / sit — identical to base class

Spec: Stage D v2 compliance-guided demonstration, April 2026.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from src.planner.heuristic_contact import HeuristicContact

logger = logging.getLogger(__name__)


class HeuristicContactGuided(HeuristicContact):

    # Class-level tag read by StageDRecorder to stamp collection_mode into
    # episode metadata. Kept as a class attribute so callers can inspect
    # without having to instantiate (useful for dry tests).
    collection_mode: str = "hand_guided"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flip compliance on. _send_cmd keys off this flag plus self._phase
        # to softline the FR gains during extend + hold only.
        self._compliance_active = True
        self._phase_watcher_stop: Optional[threading.Event] = None
        self._phase_watcher_thread: Optional[threading.Thread] = None
        logger.info("HeuristicContactGuided initialised — compliance ACTIVE for extend+hold.")

    def execute(self, *args, **kwargs):
        """
        Wrap the base execute() to run a small phase-watcher thread that
        prints a user-facing message when gains go soft (entering extend)
        and when contact is registered (leaving hold). The watcher polls
        self._phase — no _control_loop override required.
        """
        self._phase_watcher_stop = threading.Event()
        self._phase_watcher_thread = threading.Thread(
            target=self._phase_watcher_loop,
            name="guided_phase_watcher",
            daemon=True,
        )
        self._phase_watcher_thread.start()
        try:
            result = super().execute(*args, **kwargs)
        finally:
            self._phase_watcher_stop.set()
            if self._phase_watcher_thread is not None:
                self._phase_watcher_thread.join(timeout=1.0)
        return result

    def _phase_watcher_loop(self):
        last_phase = None
        announced_extend = False
        announced_hold_end = False
        while not self._phase_watcher_stop.is_set():
            phase = self._phase
            if phase != last_phase:
                if phase == "extend" and not announced_extend:
                    print(
                        "[GUIDED] Compliance active — guide FR paw to button. "
                        "Audio will end hold."
                    )
                    announced_extend = True
                if (last_phase == "hold"
                        and phase == "retract_curl"
                        and not announced_hold_end):
                    if self._contact_step >= 0:
                        print(
                            f"[GUIDED] Contact detected via {self._contact_method} "
                            f"at step {self._contact_step}"
                        )
                    else:
                        print(
                            "[GUIDED] Hold timed out — no contact detected"
                        )
                    announced_hold_end = True
                last_phase = phase
            time.sleep(0.01)


# ──────────────────────────────────────────────
# CLI smoke — quick sanity check that the subclass constructs and sets flags
# ──────────────────────────────────────────────
# Run with: python -m src.planner.heuristic_contact_guided
# This opens real DDS; for a pure unit check without DDS, see
# scripts/test_guided_mock.py.

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="HeuristicContactGuided — construction sanity check (opens DDS).",
    )
    parser.add_argument("--interface", type=str, default=None)
    args = parser.parse_args()

    if args.interface is None:
        from src.planner.heuristic_contact import DEFAULT_INTERFACE
        args.interface = DEFAULT_INTERFACE

    print("Constructing HeuristicContactGuided...")
    controller = HeuristicContactGuided(network_interface=args.interface)
    print(f"  collection_mode     = {controller.collection_mode}")
    print(f"  _compliance_active  = {controller._compliance_active}")
    assert controller.collection_mode == "hand_guided"
    assert controller._compliance_active is True
    print("OK")
