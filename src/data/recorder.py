"""
src/data/recorder.py

External observer thread that records per-step Stage D training data at
500 Hz during the contact phase. Reads state from the HeuristicContactCorrective
instance (under _state_lock) and from the grounding thread.

Spec: training_data_spec.md v2.1, Section 9.3
"""

import logging
import threading
import time
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


PHASE_TO_LABEL = {
    "lift":   0,
    "extend": 1,
    "hold":   2,
}
PHASE_DURATION_S = {
    "lift":   1.2,
    "extend": 1.2,
    "hold":   3.0,
}
ACTIVE_PHASES = ("lift", "extend", "hold")


class StageDRecorder:
    """
    Observer thread polling at 500 Hz. Appends per-step fields to in-memory
    arrays during active phases. Writes to HDF5 at save().
    """

    def __init__(self, heuristic, grounding_thread, ctrl_dt: float = 0.002):
        self.heuristic = heuristic
        self.grounding_thread = grounding_thread
        self.ctrl_dt = ctrl_dt

        self._buffer_lock = threading.Lock()
        self._recording = False
        self._t_start: float = 0.0
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._reset_buffers()
        self._metadata: dict = {}

    def _reset_buffers(self):
        self.timestamp: list = []
        self.joint_pos_actual: list = []
        self.joint_vel_actual: list = []
        self.joint_pos_cmd: list = []
        self.fr_foot_pos_fk: list = []
        self.target_pos_base: list = []
        self.foot_to_target_error: list = []
        self.imu_rpy: list = []
        self.phase_label: list = []
        self.phase_progress: list = []
        self.nominal_waypoint_fr: list = []
        self.jacobian_pid_delta: list = []

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop, name="stage_d_recorder", daemon=True)
        self._thread.start()
        logger.info("StageDRecorder observer thread started")

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def begin_episode(self, metadata: dict):
        with self._buffer_lock:
            self._reset_buffers()
            self._metadata = dict(metadata)
            self._t_start = time.monotonic()
            self._recording = True

    def end_episode(self, result):
        with self._buffer_lock:
            self._recording = False
            self._metadata.update({
                "contact_step": int(result.contact_step),
                "duration_s": float(result.duration_s),
                "press_mode": result.press_mode,
                "target_offset_xyz": np.asarray(result.target_offset_xyz, dtype=np.float32),
                "contact_method": str(result.contact_method),
                "phase_transitions": {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in result.phase_transitions.items()
                },
            })

    def save(self, hdf5_path: str,
             audio: Optional[np.ndarray] = None,
             audio_sample_rate: int = 16000,
             metadata_override: Optional[dict] = None):
        """
        Args:
            hdf5_path: output file path
            audio: raw audio waveform from AudioRecorder (optional)
            audio_sample_rate: Hz for the audio dataset (required if audio is given)
            metadata_override: extra metadata fields to merge (e.g., audio-derived
                success fields from the collection script — success_fk,
                success_audio_live, audio_detection_time_s, color_detected, etc.)
        """
        with self._buffer_lock:
            if len(self.timestamp) == 0:
                logger.warning(f"No samples recorded; skipping {hdf5_path}")
                return

            # Merge extra metadata from caller (audio success fields etc)
            final_metadata = dict(self._metadata)
            if metadata_override:
                final_metadata.update(metadata_override)

            with h5py.File(hdf5_path, "w") as f:
                grp = f.create_group("per_step")
                for name, data, dtype in [
                    ("timestamp", self.timestamp, np.float64),
                    ("joint_pos_actual", self.joint_pos_actual, np.float32),
                    ("joint_vel_actual", self.joint_vel_actual, np.float32),
                    ("joint_pos_cmd", self.joint_pos_cmd, np.float32),
                    ("fr_foot_pos_fk", self.fr_foot_pos_fk, np.float32),
                    ("target_pos_base", self.target_pos_base, np.float32),
                    ("foot_to_target_error", self.foot_to_target_error, np.float32),
                    ("imu_rpy", self.imu_rpy, np.float32),
                    ("phase_label", self.phase_label, np.int8),
                    ("phase_progress", self.phase_progress, np.float32),
                    ("nominal_waypoint_fr", self.nominal_waypoint_fr, np.float32),
                    ("jacobian_pid_delta", self.jacobian_pid_delta, np.float32),
                ]:
                    grp.create_dataset(name, data=np.array(data, dtype=dtype),
                                       compression="gzip", compression_opts=4)

                g_log = self.grounding_thread.get_log()
                gg = f.create_group("grounding")
                if g_log:
                    gg.create_dataset("timestamp",
                        data=np.array([s.timestamp for s in g_log], dtype=np.float64))
                    gg.create_dataset("position_base",
                        data=np.array([
                            s.position_base if s.position_base is not None
                            else np.full(3, np.nan, dtype=np.float32)
                            for s in g_log
                        ], dtype=np.float32))
                    gg.create_dataset("confidence",
                        data=np.array([s.confidence for s in g_log], dtype=np.float32))
                    gg.create_dataset("depth_m",
                        data=np.array([s.depth_m for s in g_log], dtype=np.float32))
                    gg.create_dataset("position_base_valid",
                        data=np.array([s.valid for s in g_log], dtype=bool))

                # NEW (v2.1): raw audio waveform
                if audio is not None and len(audio) > 0:
                    f.create_dataset("audio", data=audio.astype(np.float32),
                                     compression="gzip", compression_opts=4)
                    f.attrs["audio_sample_rate"] = audio_sample_rate

                for k, v in final_metadata.items():
                    if k == "phase_transitions":
                        pt_grp = f.create_group("phase_transitions")
                        for pname, pval in v.items():
                            pt_grp.create_dataset(pname, data=pval)
                    elif isinstance(v, (np.ndarray, list, tuple)):
                        f.attrs[k] = np.asarray(v)
                    else:
                        f.attrs[k] = v

            logger.info(f"Wrote episode HDF5: {hdf5_path}  "
                        f"({len(self.timestamp)} steps, {len(g_log)} groundings, "
                        f"audio={'yes' if audio is not None else 'no'})")

    def _loop(self):
        while not self._stop_flag.is_set():
            t0 = time.monotonic()

            if self._recording:
                phase = self.heuristic._phase
                if phase in ACTIVE_PHASES:
                    self._sample_one_step(phase)

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, self.ctrl_dt - elapsed)
            time.sleep(sleep_t)

    def _sample_one_step(self, phase: str):
        with self.heuristic._state_lock:
            if self.heuristic._low_state is None:
                return
            q  = [self.heuristic._low_state.motor_state[i].q  for i in range(12)]
            dq = [self.heuristic._low_state.motor_state[i].dq for i in range(12)]
            rpy = list(self.heuristic._low_state.imu_state.rpy[0:3])

        cmd = list(self.heuristic._last_target_q) if self.heuristic._last_target_q else [0.0]*12
        waypoint = self.heuristic.current_waypoint_fr.copy()
        delta    = self.heuristic.last_delta_fr.copy() \
            if hasattr(self.heuristic, "last_delta_fr") \
            else np.zeros(3, dtype=np.float32)

        foot = self.heuristic._estimate_fr_foot(q)

        target = self.grounding_thread.get_latest_position()
        if target is None:
            target_arr = np.full(3, np.nan, dtype=np.float32)
            error_arr  = np.full(3, np.nan, dtype=np.float32)
        else:
            target_arr = target.astype(np.float32)
            error_arr  = (target_arr - foot).astype(np.float32)

        step = self.heuristic._phase_step
        phase_dur_steps = PHASE_DURATION_S[phase] / self.ctrl_dt
        progress = min(1.0, step / phase_dur_steps)

        with self._buffer_lock:
            self.timestamp.append(time.monotonic() - self._t_start)
            self.joint_pos_actual.append(np.array(q,   dtype=np.float32))
            self.joint_vel_actual.append(np.array(dq,  dtype=np.float32))
            self.joint_pos_cmd.append(np.array(cmd, dtype=np.float32))
            self.fr_foot_pos_fk.append(foot)
            self.target_pos_base.append(target_arr)
            self.foot_to_target_error.append(error_arr)
            self.imu_rpy.append(np.array(rpy, dtype=np.float32))
            self.phase_label.append(PHASE_TO_LABEL[phase])
            self.phase_progress.append(np.float32(progress))
            self.nominal_waypoint_fr.append(waypoint.astype(np.float32))
            self.jacobian_pid_delta.append(delta.astype(np.float32))