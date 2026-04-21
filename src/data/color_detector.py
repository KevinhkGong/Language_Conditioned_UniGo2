"""
src/data/color_detector.py

Post-hoc color word identification from audio waveform via faster-whisper.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

VALID_COLORS = ["red", "blue", "green", "yellow", "orange"]


class ColorDetector:
    """
    Lazy-loads faster-whisper tiny model on first use.

    Usage:
        det = ColorDetector()
        det.load()
        result = det.detect(audio_waveform, sample_rate=16000)
        # result = {
        #     "color_detected": "red" or None,
        #     "transcript": "red",
        #     "success_audio_transcript": True,
        # }
    """

    def __init__(self, model_size: str = "small", device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self):
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper {self.model_size} on {self.device}")
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        if self._model is None:
            self.load()

        # Whisper requires 16kHz mono float32
        if sample_rate != 16000:
            from scipy import signal
            n_target = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, n_target).astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return {
                "color_detected": None,
                "transcript": "",
                "success_audio_transcript": False,
            }

        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1,
        )
        transcript = " ".join(s.text for s in segments).lower().strip()
        normalized = re.sub(r"[^a-z]+", " ", transcript).strip()
        tokens = normalized.split()

        detected = None
        for color in VALID_COLORS:
            if color in tokens or color in transcript:
                detected = color
                break

        return {
            "color_detected": detected,
            "transcript": transcript,
            "success_audio_transcript": detected is not None,
        }
