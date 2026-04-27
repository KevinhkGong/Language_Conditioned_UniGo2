"""
src/data/audio_live_detector.py

Real-time energy-threshold audio detector. When RMS energy exceeds
baseline + threshold_db for at least min_duration_ms, sets
sound_detected = True. Single-writer, multi-reader; no lock needed for
reads of the bool (atomic in CPython).
"""

import logging
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioLiveDetector:
    """
    Args:
        sample_rate: Hz
        device_index: mic device index (None = default)
        threshold_db: how much above baseline RMS to trigger detection
        min_duration_ms: minimum sustained high-energy duration (spike reject)
        baseline_duration_s: silence window to calibrate baseline at start
    """

    def __init__(self, sample_rate: int = 16000,
                 device_index: Optional[int] = None,
                 threshold_db: float = 20.0,
                 min_duration_ms: int = 50,
                 baseline_duration_s: float = 2.0,
                 block_duration_ms: int = 20,
                 capture_sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.threshold_db = threshold_db
        self.min_duration_ms = min_duration_ms
        self.baseline_duration_s = baseline_duration_s
        # Use capture_sample_rate for block sizing when PipeWire forces a higher rate.
        # RMS detection is energy-based so resampling is not needed here.
        _capture_rate = capture_sample_rate if capture_sample_rate else sample_rate
        self._capture_rate = _capture_rate
        self.block_size = int(_capture_rate * block_duration_ms / 1000)
        self.block_duration_ms = block_duration_ms

        # Public state
        self.sound_detected: bool = False
        self.detection_time_s: Optional[float] = None

        self._stream: Optional[sd.InputStream] = None
        self._baseline_rms: Optional[float] = None
        self._baseline_samples: list = []
        self._baseline_done = False
        self._high_energy_ms: float = 0.0
        self._episode_t_start: float = 0.0
        self._calibrating = False

    def start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self._capture_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"AudioLiveDetector stream opened ({self._capture_rate} Hz, device={self.device_index})")

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def calibrate_baseline(self, duration_s: Optional[float] = None):
        """Block for duration_s of silence to measure ambient noise floor."""
        duration = duration_s or self.baseline_duration_s
        logger.info(f"Calibrating baseline for {duration:.1f}s — keep quiet...")
        self._baseline_samples = []
        self._baseline_done = False
        self._calibrating = True
        time.sleep(duration)
        self._calibrating = False

        if not self._baseline_samples:
            logger.warning("No samples captured during calibration")
            self._baseline_rms = 1e-4
        else:
            all_samples = np.concatenate(self._baseline_samples)
            self._baseline_rms = float(np.sqrt(np.mean(all_samples ** 2)))
            self._baseline_rms = max(self._baseline_rms, 1e-6)
        self._baseline_done = True
        logger.info(f"Baseline RMS={self._baseline_rms:.6f}  "
                    f"threshold_above_baseline={self.threshold_db}dB")

    def start_episode(self):
        """Reset detection state at the start of an episode."""
        self.sound_detected = False
        self.detection_time_s = None
        self._high_energy_ms = 0.0
        self._episode_t_start = time.monotonic()

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"sounddevice status: {status}")

        samples = indata[:, 0]

        if self._calibrating:
            self._baseline_samples.append(samples.copy())
            return

        if not self._baseline_done or self._baseline_rms is None:
            return

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1e-8:
            db_above = -120.0
        else:
            db_above = 20.0 * np.log10(rms / self._baseline_rms)

        if db_above > self.threshold_db:
            self._high_energy_ms += self.block_duration_ms
            if (self._high_energy_ms >= self.min_duration_ms
                    and not self.sound_detected):
                self.sound_detected = True
                self.detection_time_s = time.monotonic() - self._episode_t_start
                logger.info(f"Sound detected: rms={rms:.6f} "
                            f"({db_above:.1f}dB above baseline) at "
                            f"t={self.detection_time_s:.3f}s")
        else:
            self._high_energy_ms = 0.0
