"""
src/data/audio_recorder.py

Records full-episode audio from USB microphone via sounddevice.
Stored in-memory as a numpy buffer; returned via end_episode() at end.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Full-episode audio recorder. Mono, configurable sample rate.

    Usage:
        rec = AudioRecorder(sample_rate=16000, device_index=None)
        rec.start_stream()          # opens mic, starts background capture
        ...
        rec.start_episode()         # resets buffer
        result = heuristic.execute(...)
        audio = rec.end_episode()   # returns np.ndarray (N,) float32
        ...
        rec.stop_stream()
    """

    def __init__(self, sample_rate: int = 16000,
                 device_index: Optional[int] = None,
                 block_duration_ms: int = 50):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.block_size = int(sample_rate * block_duration_ms / 1000)

        self._stream: Optional[sd.InputStream] = None
        self._buffer_lock = threading.Lock()
        self._recording = False
        self._buffer: list = []
        self._t_start: float = 0.0

    def start_stream(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size,
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"AudioRecorder stream opened "
                    f"({self.sample_rate} Hz, device={self.device_index})")

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("AudioRecorder stream closed")

    def start_episode(self):
        with self._buffer_lock:
            self._buffer = []
            self._recording = True
            self._t_start = time.monotonic()

    def end_episode(self) -> np.ndarray:
        with self._buffer_lock:
            self._recording = False
            if not self._buffer:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._buffer).astype(np.float32)

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"sounddevice status: {status}")
        if not self._recording:
            return
        with self._buffer_lock:
            self._buffer.append(indata[:, 0].copy())
