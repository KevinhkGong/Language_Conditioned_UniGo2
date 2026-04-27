"""
src/data/color_detector.py

Post-hoc color word identification from audio waveform via faster-whisper.
"""

import logging
import os
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
                 compute_type: str = "float16", model_path: Optional[str] = None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_path = model_path
        self._model = None

    def load(self):
        from faster_whisper import WhisperModel
        model_id = os.path.expanduser(self.model_path) if self.model_path else self.model_size
        logger.info(f"Loading faster-whisper from {model_id!r} on {self.device}")
        self._model = WhisperModel(model_id, device=self.device,
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

        # Normalize amplitude if the recording is quiet.
        # Short button-press announcements often peak around 0.2-0.3,
        # which Whisper's default VAD filters out as non-speech.
        peak = float(np.abs(audio).max())
        if 0 < peak < 0.5:
            audio = audio / peak * 0.9

        segments, _ = self._model.transcribe(
            audio, language="en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.05,
            initial_prompt="The button announces its color: red, blue, green, or yellow.",
        )
        transcript = " ".join(s.text for s in segments).lower().strip()

        detected = None
        for color in VALID_COLORS:
            if color in transcript:
                detected = color
                break

        return {
            "color_detected": detected,
            "transcript": transcript,
            "success_audio_transcript": detected is not None,
        }
