"""
src/data/grounding_thread.py

Background thread that runs visual grounding at a configurable rate and
exposes the latest result to the 500 Hz control and recorder threads
via a thread-safe shared variable.

Spec: training_data_spec.md v2.1, Section 9.2
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundingSample:
    timestamp:  float
    position_base: Optional[np.ndarray]
    confidence: float
    depth_m:    float
    valid:      bool


class GroundingThread:
    """
    Runs grounder.ground() in a background thread at rate_hz.

    Usage:
        gt = GroundingThread(camera, grounder, prompt="red button", rate_hz=5)
        gt.start()
        ...
        latest = gt.get_latest_position()  # thread-safe
        ...
        gt.stop()
        log = gt.get_log()
    """

    def __init__(self, camera, grounder, prompt: str = "red button",
                 rate_hz: float = 5.0):
        self.camera = camera
        self.grounder = grounder
        self.prompt = prompt
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz

        self._lock = threading.Lock()
        self._latest_position: Optional[np.ndarray] = None
        self._latest_sample: Optional[GroundingSample] = None
        self._log: list = []
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: float = 0.0

    def start(self):
        self._t_start = time.monotonic()
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop, name="grounding", daemon=True)
        self._thread.start()
        logger.info(f"GroundingThread started at {self.rate_hz} Hz")

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info(f"GroundingThread stopped. {len(self._log)} samples logged.")

    def _loop(self):
        while not self._stop_flag.is_set():
            t0 = time.monotonic()
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(self.period_s)
                    continue
                result = self.grounder.ground(frame, prompt=self.prompt)
            except Exception as e:
                logger.warning(f"Grounding call failed: {e}")
                time.sleep(self.period_s)
                continue

            if result is None:
                sample = GroundingSample(
                    timestamp=time.monotonic() - self._t_start,
                    position_base=None,
                    confidence=0.0,
                    depth_m=0.0,
                    valid=False,
                )
            else:
                pos = result.position_base
                sample = GroundingSample(
                    timestamp=time.monotonic() - self._t_start,
                    position_base=pos.copy() if pos is not None else None,
                    confidence=float(result.confidence),
                    depth_m=float(result.depth_m) if result.depth_m is not None else 0.0,
                    valid=pos is not None,
                )

            with self._lock:
                self._latest_sample = sample
                if sample.valid:
                    self._latest_position = sample.position_base
                self._log.append(sample)

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.period_s - elapsed)
            time.sleep(sleep_t)

    def get_latest_position(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_position is None else self._latest_position.copy()

    def get_log(self) -> list:
        with self._lock:
            return list(self._log)