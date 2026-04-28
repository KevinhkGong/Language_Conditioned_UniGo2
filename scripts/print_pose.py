"""
scripts/print_pose_extended.py

Real-time robot pose display with emphasis on IMU orientation (roll, pitch, yaw)
and stability metrics. Useful for:
  - Confirming robot is level before starting an episode
  - Comparing IMU state across multiple placements
  - Detecting yaw misalignment that causes navigation drift
  - Recording baseline IMU state for training data augmentation

Usage:
    python scripts/print_pose_extended.py
    python scripts/print_pose_extended.py --log   # append CSV log to data/pose_log.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

# ANSI colors
RESET = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
CLEAR = "\033[2J\033[H"

# IMU thresholds for "level" indicator
LEVEL_ROLL_THRESHOLD_RAD = 0.05   # ~2.9°
LEVEL_PITCH_THRESHOLD_RAD = 0.05  # ~2.9°
LEVEL_YAW_DRIFT_THRESHOLD = 0.02  # rad/s, detect yaw drift


# FR joint index mapping (per-leg order, real robot convention)
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]


def rad2deg(r):
    return r * 180.0 / math.pi


def level_indicator(value_rad, threshold_rad):
    """Return (status_text, color) based on abs(value) vs threshold."""
    av = abs(value_rad)
    if av < threshold_rad * 0.5:
        return "LEVEL", GREEN
    elif av < threshold_rad:
        return "near", YELLOW
    else:
        return "OFF", RED


def format_bar(value, min_v=-0.5, max_v=+0.5, width=40):
    """Render a horizontal bar centered at zero."""
    if value is None or not math.isfinite(value):
        return "  (nan)  "
    # Clamp
    v = max(min_v, min(max_v, value))
    # Map to column
    zero_col = width // 2
    half_width = width // 2
    span = max(abs(min_v), abs(max_v))
    col = zero_col + int((v / span) * half_width)
    col = max(0, min(width - 1, col))
    bar = ["·"] * width
    bar[zero_col] = "│"  # center mark
    if col < zero_col:
        for i in range(col, zero_col):
            bar[i] = "◄"
    elif col > zero_col:
        for i in range(zero_col + 1, col + 1):
            bar[i] = "►"
    return "".join(bar)


class PoseDisplay:
    def __init__(self, log_path=None):
        self.low_state = None
        self.low_cmd = None  # latest LowCmd_ from rt/lowcmd (Sport Mode or our own publisher)
        self.low_cmd_age_s = None
        self._low_cmd_t = None
        self.history = deque(maxlen=20)  # last 20 samples for smoothing
        self.start_time = time.monotonic()
        self.log_path = log_path
        self.log_file = None
        self.log_writer = None

        if log_path:
            exists = os.path.exists(log_path)
            self.log_file = open(log_path, 'a', newline='')
            self.log_writer = csv.writer(self.log_file)
            if not exists:
                self.log_writer.writerow([
                    'timestamp', 'elapsed_s',
                    'roll_rad', 'pitch_rad', 'yaw_rad',
                    'roll_deg', 'pitch_deg', 'yaw_deg',
                    'gyro_x', 'gyro_y', 'gyro_z',
                    'acc_x', 'acc_y', 'acc_z',
                    'temp',
                    *JOINT_NAMES,
                    *[f"{n}_kp" for n in JOINT_NAMES],
                    *[f"{n}_kd" for n in JOINT_NAMES],
                    *[f"{n}_q_cmd" for n in JOINT_NAMES],
                    *[f"{n}_tau_ff" for n in JOINT_NAMES],
                    *[f"{n}_tau_est" for n in JOINT_NAMES],
                    'foot_force_FR', 'foot_force_FL',
                    'foot_force_RR', 'foot_force_RL',
                ])

    def lowstate_callback(self, msg):
        self.low_state = msg

    def lowcmd_callback(self, msg):
        self.low_cmd = msg
        self._low_cmd_t = time.monotonic()

    def close(self):
        if self.log_file:
            self.log_file.close()

    def render(self):
        if self.low_state is None:
            print("Waiting for /rt/lowstate…")
            return

        imu = self.low_state.imu_state
        rpy = imu.rpy  # [roll, pitch, yaw] rad
        gyro = imu.gyroscope  # [wx, wy, wz] rad/s
        acc = imu.accelerometer  # [ax, ay, az] m/s^2
        temp = imu.temperature

        elapsed = time.monotonic() - self.start_time

        # Record history
        self.history.append({
            'roll': rpy[0],
            'pitch': rpy[1],
            'yaw': rpy[2],
            'gyro_z': gyro[2],
            't': elapsed,
        })

        # Clear screen, home cursor
        print(CLEAR, end='')

        print(f"{BOLD}╔══════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{BOLD}║ Go2 Pose Display — Extended (t={elapsed:7.1f}s)              ║{RESET}")
        print(f"{BOLD}╚══════════════════════════════════════════════════════════════╝{RESET}")
        print()

        # ─── IMU Orientation ───────────────────────────────────────────
        print(f"{BOLD}── IMU Orientation ─────────────────────────────────────────────{RESET}")

        roll_status, roll_c = level_indicator(rpy[0], LEVEL_ROLL_THRESHOLD_RAD)
        pitch_status, pitch_c = level_indicator(rpy[1], LEVEL_PITCH_THRESHOLD_RAD)

        print(f"  Roll  : {roll_c}{rad2deg(rpy[0]):+7.2f}°  ({rpy[0]:+.4f} rad)  [{roll_status}]{RESET}")
        print(f"          {format_bar(rpy[0], -0.3, +0.3, 50)}")
        print(f"          Left tilt ←    level    → Right tilt")
        print()
        print(f"  Pitch : {pitch_c}{rad2deg(rpy[1]):+7.2f}°  ({rpy[1]:+.4f} rad)  [{pitch_status}]{RESET}")
        print(f"          {format_bar(rpy[1], -0.3, +0.3, 50)}")
        print(f"          Nose down ←    level    → Nose up")
        print()
        print(f"  Yaw   : {CYAN}{rad2deg(rpy[2]):+7.2f}°  ({rpy[2]:+.4f} rad){RESET}")
        print(f"          {format_bar(rpy[2], -math.pi, +math.pi, 50)}")
        print(f"          (absolute — relative to robot boot orientation)")
        print()

        # ─── Angular Velocity (Gyro) ───────────────────────────────────
        print(f"{BOLD}── Angular Velocity ────────────────────────────────────────────{RESET}")
        yaw_rate_deg = rad2deg(gyro[2])
        is_yaw_drifting = abs(gyro[2]) > LEVEL_YAW_DRIFT_THRESHOLD
        drift_color = RED if is_yaw_drifting else GREEN
        drift_label = "DRIFTING" if is_yaw_drifting else "stable"
        print(f"  Yaw rate  : {drift_color}{yaw_rate_deg:+7.2f}°/s  [{drift_label}]{RESET}")
        print(f"  Pitch rate: {rad2deg(gyro[1]):+7.2f}°/s")
        print(f"  Roll rate : {rad2deg(gyro[0]):+7.2f}°/s")
        print()

        # ─── Acceleration ──────────────────────────────────────────────
        print(f"{BOLD}── Acceleration ────────────────────────────────────────────────{RESET}")
        # z should be ~9.8 m/s² when upright
        z_ok = 9.0 < acc[2] < 10.6
        z_col = GREEN if z_ok else YELLOW
        print(f"  acc z : {z_col}{acc[2]:+7.2f} m/s²  (expect ~9.81 when upright){RESET}")
        print(f"  acc x : {acc[0]:+7.2f} m/s²")
        print(f"  acc y : {acc[1]:+7.2f} m/s²")
        print()

        # ─── Stability Over Last 20 Samples ───────────────────────────
        if len(self.history) >= 5:
            rolls = [h['roll'] for h in self.history]
            pitches = [h['pitch'] for h in self.history]
            yaws = [h['yaw'] for h in self.history]

            roll_std = np.std(rolls)
            pitch_std = np.std(pitches)
            yaw_std = np.std(yaws)

            print(f"{BOLD}── Stability (last {len(self.history)} samples) ─────────────────────{RESET}")
            print(f"  Roll  std : {rad2deg(roll_std):+5.2f}°   (low = stable)")
            print(f"  Pitch std : {rad2deg(pitch_std):+5.2f}°")
            print(f"  Yaw   std : {rad2deg(yaw_std):+5.2f}°")
            print()

        # ─── Joint Positions ──────────────────────────────────────────
        print(f"{BOLD}── Joint Positions (rad) ───────────────────────────────────────{RESET}")
        for i in range(0, 12, 3):
            leg = JOINT_NAMES[i].split('_')[0]
            hip = self.low_state.motor_state[i].q
            thigh = self.low_state.motor_state[i + 1].q
            calf = self.low_state.motor_state[i + 2].q
            print(f"  {leg:3s}: hip={hip:+.3f}  thigh={thigh:+.3f}  calf={calf:+.3f}")
        print()

        # ─── Joint Gains (kp / kd) ────────────────────────────────────
        # Sourced from rt/lowcmd. Sport Mode (preprogrammed motions like
        # the handshake action) publishes its own low-level commands; our
        # own low-level controller also publishes here. If no message has
        # arrived yet, this section says so rather than printing zeros.
        print(f"{BOLD}── Joint Gains kp / kd (from rt/lowcmd) ────────────────────────{RESET}")
        if self.low_cmd is None:
            print(f"  {YELLOW}(no LowCmd_ received yet — robot may be idle in high-level "
                  f"or Sport Mode is not publishing){RESET}")
        else:
            age = (time.monotonic() - self._low_cmd_t) if self._low_cmd_t else None
            age_color = GREEN if (age is not None and age < 0.25) else YELLOW
            age_text = f"{age*1000:.0f} ms" if age is not None else "n/a"
            print(f"  last cmd age: {age_color}{age_text}{RESET}")
            for i in range(0, 12, 3):
                leg = JOINT_NAMES[i].split('_')[0]
                kp_h, kd_h = self.low_cmd.motor_cmd[i].kp,     self.low_cmd.motor_cmd[i].kd
                kp_t, kd_t = self.low_cmd.motor_cmd[i+1].kp,   self.low_cmd.motor_cmd[i+1].kd
                kp_c, kd_c = self.low_cmd.motor_cmd[i+2].kp,   self.low_cmd.motor_cmd[i+2].kd
                print(f"  {leg:3s}:  hip kp={kp_h:6.2f} kd={kd_h:5.2f}   "
                      f"thigh kp={kp_t:6.2f} kd={kd_t:5.2f}   "
                      f"calf kp={kp_c:6.2f} kd={kd_c:5.2f}")
        print()

        # ─── Feedforward Torques (cmd) vs Estimated Torques (state) ───
        # tau on rt/lowcmd is the feedforward term Sport Mode adds on top
        # of its low-PD tracking.  tau_est on rt/lowstate is the actual
        # motor torque inferred from current.  Comparing them shows how
        # much of the "hold" load is FF vs PD: when FF dominates, |tau|
        # on the cmd side is large and kp/kd are small.
        print(f"{BOLD}── Joint Torques (N·m) ─────────────────────────────────────────{RESET}")
        if self.low_cmd is None:
            print(f"  {YELLOW}(no LowCmd_ — cmd tau unavailable; showing tau_est only){RESET}")
            for i in range(0, 12, 3):
                leg = JOINT_NAMES[i].split('_')[0]
                te_h = self.low_state.motor_state[i].tau_est
                te_t = self.low_state.motor_state[i+1].tau_est
                te_c = self.low_state.motor_state[i+2].tau_est
                print(f"  {leg:3s}:  est  hip={te_h:+7.2f}  thigh={te_t:+7.2f}  calf={te_c:+7.2f}")
        else:
            print(f"  {'leg':<5}{'cmd_hip':>9} {'est_hip':>9}   "
                  f"{'cmd_thi':>9} {'est_thi':>9}   "
                  f"{'cmd_clf':>9} {'est_clf':>9}")
            for i in range(0, 12, 3):
                leg = JOINT_NAMES[i].split('_')[0]
                tc_h = self.low_cmd.motor_cmd[i].tau
                tc_t = self.low_cmd.motor_cmd[i+1].tau
                tc_c = self.low_cmd.motor_cmd[i+2].tau
                te_h = self.low_state.motor_state[i].tau_est
                te_t = self.low_state.motor_state[i+1].tau_est
                te_c = self.low_state.motor_state[i+2].tau_est
                print(f"  {leg:<5}{tc_h:+9.2f} {te_h:+9.2f}   "
                      f"{tc_t:+9.2f} {te_t:+9.2f}   "
                      f"{tc_c:+9.2f} {te_c:+9.2f}")
        print()

        # ─── Foot Forces ──────────────────────────────────────────────
        print(f"{BOLD}── Foot Forces (raw) ───────────────────────────────────────────{RESET}")
        # foot order: FR=0, FL=1, RR=2, RL=3 on Go2
        ff = [self.low_state.foot_force[i] for i in range(4)]
        print(f"  FR: {ff[0]:4d}   FL: {ff[1]:4d}   RR: {ff[2]:4d}   RL: {ff[3]:4d}")
        total = sum(ff)
        asymmetry = (ff[0] + ff[2]) - (ff[1] + ff[3])  # right - left
        print(f"  total={total:5d}  |  right-left asym: {asymmetry:+5d}")
        print()

        # ─── Overall Assessment ───────────────────────────────────────
        print(f"{BOLD}── Assessment ──────────────────────────────────────────────────{RESET}")
        issues = []
        if abs(rpy[0]) > LEVEL_ROLL_THRESHOLD_RAD:
            issues.append(f"Roll is {rad2deg(rpy[0]):+.1f}° — robot tilted left/right")
        if abs(rpy[1]) > LEVEL_PITCH_THRESHOLD_RAD:
            issues.append(f"Pitch is {rad2deg(rpy[1]):+.1f}° — nose up/down")
        if abs(gyro[2]) > LEVEL_YAW_DRIFT_THRESHOLD:
            issues.append(f"Yaw is drifting at {rad2deg(gyro[2]):+.2f}°/s")
        if len(self.history) >= 10:
            if np.std([h['roll'] for h in self.history]) > 0.01:
                issues.append("Roll is oscillating")

        if not issues:
            print(f"  {GREEN}✓ Robot is level and stable. Ready to place.{RESET}")
        else:
            for issue in issues:
                print(f"  {YELLOW}⚠ {issue}{RESET}")

        print()
        print(f"{CYAN}  Press Ctrl+C to exit{RESET}")

        # Log if enabled
        if self.log_writer:
            now = datetime.now().isoformat()
            row = [
                now, f"{elapsed:.3f}",
                f"{rpy[0]:.6f}", f"{rpy[1]:.6f}", f"{rpy[2]:.6f}",
                f"{rad2deg(rpy[0]):.3f}", f"{rad2deg(rpy[1]):.3f}", f"{rad2deg(rpy[2]):.3f}",
                f"{gyro[0]:.4f}", f"{gyro[1]:.4f}", f"{gyro[2]:.4f}",
                f"{acc[0]:.3f}", f"{acc[1]:.3f}", f"{acc[2]:.3f}",
                f"{temp:.1f}",
            ]
            for i in range(12):
                row.append(f"{self.low_state.motor_state[i].q:.4f}")
            # KP, KD, q_cmd, tau_ff from rt/lowcmd (empty cells when no
            # LowCmd_ yet); tau_est always from rt/lowstate.
            if self.low_cmd is not None:
                for i in range(12):
                    row.append(f"{self.low_cmd.motor_cmd[i].kp:.4f}")
                for i in range(12):
                    row.append(f"{self.low_cmd.motor_cmd[i].kd:.4f}")
                for i in range(12):
                    row.append(f"{self.low_cmd.motor_cmd[i].q:.4f}")
                for i in range(12):
                    row.append(f"{self.low_cmd.motor_cmd[i].tau:.4f}")
            else:
                row.extend([""] * 48)
            for i in range(12):
                row.append(f"{self.low_state.motor_state[i].tau_est:.4f}")
            for i in range(4):
                row.append(f"{self.low_state.foot_force[i]}")
            self.log_writer.writerow(row)
            self.log_file.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', action='store_true',
                    help='Append CSV log to data/pose_log.csv')
    ap.add_argument('--rate', type=float, default=10.0,
                    help='Display refresh rate in Hz (default 10)')
    args = ap.parse_args()

    log_path = None
    if args.log:
        log_path = 'data/pose_log.csv'
        os.makedirs('data', exist_ok=True)

    print("Initializing DDS…")
    ChannelFactoryInitialize(0, 'enx98fc84e68f1a')

    display = PoseDisplay(log_path=log_path)

    sub = ChannelSubscriber('rt/lowstate', LowState_)
    sub.Init(display.lowstate_callback, 10)

    # Also listen on rt/lowcmd to read the kp/kd that the active controller
    # (Sport Mode, handshake, or our own low-level publisher) is sending.
    sub_cmd = ChannelSubscriber('rt/lowcmd', LowCmd_)
    sub_cmd.Init(display.lowcmd_callback, 10)

    period = 1.0 / args.rate
    print(f"Displaying pose at {args.rate}Hz. Press Ctrl+C to exit.")
    if log_path:
        print(f"Logging to: {log_path}")

    time.sleep(1)  # let first lowstate arrive

    try:
        while True:
            display.render()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        display.close()


if __name__ == "__main__":
    main()