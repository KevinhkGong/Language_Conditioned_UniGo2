"""
scripts/print_pose.py

Real-time joint state monitor for Unitree Go2.

Subscribes to rt/lowstate and continuously prints all 12 motor states
(position q, velocity dq, torque tau) with a refreshing terminal display.

Usage:
    python scripts/print_pose.py
    python scripts/print_pose.py --rate 50    # 50 Hz refresh (default 10)

Press Ctrl+C to stop.

Joint ordering (Go2):
    0=FR_hip   1=FR_thigh  2=FR_calf
    3=FL_hip   4=FL_thigh  5=FL_calf
    6=RR_hip   7=RR_thigh  8=RR_calf
    9=RL_hip  10=RL_thigh 11=RL_calf
"""

import time
import threading
import argparse
import sys
import os

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

DEFAULT_INTERFACE = "enx98fc84e68f1a"

JOINT_NAMES = [
    "FR_hip  ", "FR_thigh", "FR_calf ",
    "FL_hip  ", "FL_thigh", "FL_calf ",
    "RR_hip  ", "RR_thigh", "RR_calf ",
    "RL_hip  ", "RL_thigh", "RL_calf ",
]


class PoseMonitor:

    def __init__(self, interface: str):
        self._low_state = None
        self._lock = threading.Lock()
        self._count = 0

        ChannelFactoryInitialize(0, interface)

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._state_cb, 10)

    def _state_cb(self, msg: LowState_):
        with self._lock:
            self._low_state = msg
            self._count += 1

    def get_state(self):
        with self._lock:
            return self._low_state, self._count

    def print_loop(self, rate_hz: float):
        dt = 1.0 / rate_hz

        # Wait for first message
        print("Waiting for LowState...", end="", flush=True)
        while True:
            state, _ = self.get_state()
            if state is not None:
                break
            time.sleep(0.1)
        print(" connected!\n")

        try:
            while True:
                state, count = self.get_state()
                if state is None:
                    time.sleep(dt)
                    continue

                # Move cursor to top and overwrite
                sys.stdout.write("\033[H\033[J")

                imu = state.imu_state
                lines = []
                lines.append("═" * 72)
                lines.append("  Go2 Real-Time Joint State Monitor")
                lines.append("═" * 72)
                lines.append(f"  Messages received: {count:>8d}     "
                             f"IMU rpy: ({imu.rpy[0]:+6.3f}, "
                             f"{imu.rpy[1]:+6.3f}, {imu.rpy[2]:+6.3f})")
                lines.append("─" * 72)
                lines.append(f"  {'Joint':<10s}  {'q (rad)':>10s}  "
                             f"{'dq (rad/s)':>12s}  {'tau (Nm)':>10s}")
                lines.append("─" * 72)

                for i in range(12):
                    m = state.motor_state[i]
                    lines.append(f"  {JOINT_NAMES[i]:<10s}  {m.q:>+10.4f}  "
                                 f"{m.dq:>+12.4f}  {m.tau_est:>+10.4f}")

                    # Blank line between leg groups
                    if i in (2, 5, 8):
                        lines.append("")

                lines.append("─" * 72)

                # Foot force estimates
                lines.append(f"  Foot forces:  "
                             f"FR={state.foot_force[0]:>5d}  "
                             f"FL={state.foot_force[1]:>5d}  "
                             f"RR={state.foot_force[2]:>5d}  "
                             f"RL={state.foot_force[3]:>5d}")

                lines.append("═" * 72)
                lines.append("  Ctrl+C to stop")

                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Go2 joint state monitor"
    )
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Display refresh rate in Hz (default: 10)")
    args = parser.parse_args()

    monitor = PoseMonitor(DEFAULT_INTERFACE)
    monitor.print_loop(args.rate)