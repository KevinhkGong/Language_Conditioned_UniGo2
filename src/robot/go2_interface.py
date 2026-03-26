"""
src/robot/go2_interface.py

Go2 Sport Mode interface for real-robot deployment.
Wraps unitree_sdk2_python's SportClient into a clean API
for the language-conditioned contact manipulation pipeline.

Usage:
    from src.robot.go2_interface import Go2Interface

    robot = Go2Interface("enx98fc84e68f1a")
    robot.stand()
    robot.move_to_position(dx=0.8, dy=0.0)
    robot.stop()

Requirements:
    - conda activate unigo2  (or env_go2)
    - unitree_sdk2_python installed
    - Go2 connected via Ethernet on 192.168.123.x subnet
    - sudo ip addr add 192.168.123.99/24 dev <interface>
"""

import time
import math
import logging

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_NETWORK_INTERFACE = "enx98fc84e68f1a"

# Sport Mode velocity limits (m/s and rad/s)
MAX_VX    = 0.5   # forward/backward
MAX_VY    = 0.3   # lateral
MAX_VYAW  = 0.5   # rotational

# Control loop frequency for move_to_position
CONTROL_HZ = 10   # 10 Hz position control loop

# Approach tolerances
POS_TOLERANCE_M   = 0.05   # 5cm — close enough in x/y
YAW_TOLERANCE_RAD = 0.05   # ~3 degrees


class Go2Interface:
    """
    High-level interface to the Unitree Go2 via Sport Mode SDK.

    Provides:
        - stand / stop
        - raw velocity move
        - dead-reckoning move to relative position

    NOT included here (future work):
        - Low-level joint control (for leg-lift contact phase)
        - State subscription (IMU, joint pos)
        - Mode switching to low-level

    Notes on dead-reckoning:
        move_to_position uses open-loop time integration.
        The Go2 does not expose odometry over the SDK in Sport Mode,
        so position is estimated from commanded velocity × time.
        Errors accumulate — this is sufficient for short approach
        distances (~1m) but should be upgraded to closed-loop
        once SAM2 re-grounding is integrated (Phase 4).
    """

    def __init__(self, network_interface: str = DEFAULT_NETWORK_INTERFACE):
        """
        Initialise the Sport Mode client and connect to the Go2.

        Args:
            network_interface: Ethernet interface name (e.g. enx98fc84e68f1a).
                               Run `ip link show` to find yours.
        """
        self.network_interface = network_interface
        self._connected = False

        logger.info(f"Initialising Go2Interface on interface: {network_interface}")

        # Initialise the DDS channel factory — must be called once before
        # any client is created.
        ChannelFactoryInitialize(0, network_interface)

        # Create and initialise the Sport Mode client
        self._client = SportClient()
        self._client.SetTimeout(10.0)
        self._client.Init()

        self._connected = True
        logger.info("Go2Interface connected successfully.")

    # ──────────────────────────────────────────────
    # Basic motion commands
    # ──────────────────────────────────────────────

    def stand(self, settle_time: float = 2.0) -> None:
        """
        Command the robot to stand up from a lying/sitting position.
        Blocks until settle_time has elapsed.

        Args:
            settle_time: Seconds to wait after issuing the command.
        """
        self._check_connected()
        logger.info("Standing up...")
        ret = self._client.StandUp()
        self._check_return(ret, "StandUp")
        time.sleep(settle_time)

        logger.info("Switching to balance stand (walking-ready)...")
        ret = self._client.BalanceStand()
        self._check_return(ret, "BalanceStand")
        time.sleep(1.0)
        logger.info("Stand complete — ready for Move commands.")

    def balance_stand(self, settle_time: float = 1.0) -> None:
        """
        Switch to balance-stand mode (body held level, feet stationary).
        Use this before issuing Move commands.

        Args:
            settle_time: Seconds to wait after issuing the command.
        """
        self._check_connected()
        ret = self._client.BalanceStand()
        self._check_return(ret, "BalanceStand")
        time.sleep(settle_time)

    def stop(self) -> None:
        """
        Immediately stop all movement. Robot holds current position.
        """
        self._check_connected()
        ret = self._client.StopMove()
        self._check_return(ret, "StopMove")
        logger.info("Robot stopped.")

    def sit(self, settle_time: float = 2.0) -> None:
        """
        Command the robot to sit down safely.

        Args:
            settle_time: Seconds to wait after issuing the command.
        """
        self._check_connected()
        logger.info("Sitting down...")
        ret = self._client.StandDown()
        self._check_return(ret, "StandDown")
        time.sleep(settle_time)

    # ──────────────────────────────────────────────
    # Velocity control
    # ──────────────────────────────────────────────

    def move(self, vx: float, vy: float, vyaw: float, duration: float) -> None:
        """
        Send a continuous velocity command for a fixed duration.

        Args:
            vx:       Forward velocity (m/s). Positive = forward.
            vy:       Lateral velocity (m/s). Positive = left.
            vyaw:     Yaw rate (rad/s). Positive = counter-clockwise.
            duration: How long to apply this command (seconds).

        Safety: velocities are clipped to hardware limits.
        """
        self._check_connected()

        # Clip to safe limits
        vx   = max(-MAX_VX,   min(MAX_VX,   vx))
        vy   = max(-MAX_VY,   min(MAX_VY,   vy))
        vyaw = max(-MAX_VYAW, min(MAX_VYAW, vyaw))

        logger.debug(f"move(vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}, "
                     f"duration={duration:.2f}s)")

        start = time.time()
        dt = 1.0 / CONTROL_HZ

        while time.time() - start < duration:
            ret = self._client.Move(vx, vy, vyaw)
            self._check_return(ret, "Move", raise_on_error=False)
            time.sleep(dt)

        self.stop()

    # ──────────────────────────────────────────────
    # Position control (dead-reckoning)
    # ──────────────────────────────────────────────

    def move_to_position(
        self,
        dx: float,
        dy: float,
        dyaw: float = 0.0,
        speed: float = 0.3,
    ) -> None:
        """
        Walk to a relative position offset using dead-reckoning.

        This is the primary method called by the Contact Planner output.
        The planner outputs a goal base offset (dx, dy, dyaw) and this
        method handles the velocity control loop to reach it.

        Dead-reckoning strategy:
            1. Rotate to face the target direction (yaw correction first)
            2. Walk forward/lateral while correcting yaw
            3. Stop when estimated position is within POS_TOLERANCE_M

        Limitation: Open-loop — position error accumulates over distance.
        Replace inner loop with SAM2 re-grounding in Phase 4.

        Args:
            dx:    Forward displacement (m). Positive = forward.
            dy:    Lateral displacement (m). Positive = left.
            dyaw:  Yaw rotation (rad). Positive = counter-clockwise.
            speed: Walking speed (m/s). Default 0.3 m/s (conservative).
        """
        self._check_connected()

        logger.info(f"move_to_position: dx={dx:.3f}m, dy={dy:.3f}m, "
                    f"dyaw={dyaw:.3f}rad at speed={speed:.2f}m/s")

        dt = 1.0 / CONTROL_HZ

        # ── Phase 1: Yaw correction (rotate in place first) ──────────
        if abs(dyaw) > YAW_TOLERANCE_RAD:
            logger.debug(f"Phase 1: yaw correction {math.degrees(dyaw):.1f}°")
            yaw_sign     = 1.0 if dyaw > 0 else -1.0
            yaw_speed    = min(MAX_VYAW, abs(dyaw) / 1.0)   # complete in ~1s
            yaw_duration = abs(dyaw) / yaw_speed
            self.move(0.0, 0.0, yaw_sign * yaw_speed, yaw_duration)
            time.sleep(0.3)   # brief settle

        # ── Phase 2: Translate to target ─────────────────────────────
        distance   = math.sqrt(dx**2 + dy**2)
        if distance < POS_TOLERANCE_M:
            logger.info("Already within position tolerance — no translation needed.")
            return

        # Decompose into forward and lateral velocity components
        # (robot frame: x=forward, y=left)
        ratio_x  = dx / distance
        ratio_y  = dy / distance
        vx_cmd   = ratio_x * speed
        vy_cmd   = ratio_y * speed

        # Estimated time to cover distance at given speed
        duration = distance / speed

        logger.debug(f"Phase 2: translate {distance:.3f}m "
                     f"(vx={vx_cmd:.3f}, vy={vy_cmd:.3f}) "
                     f"for {duration:.2f}s")

        self.move(vx_cmd, vy_cmd, 0.0, duration)

        logger.info("move_to_position complete.")

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _check_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "Go2Interface not connected. "
                "Did ChannelFactory.Init() succeed?"
            )

    @staticmethod
    def _check_return(
        ret: int,
        method_name: str,
        raise_on_error: bool = True
    ) -> None:
        """
        Check the return code from a SportClient call.
        All SDK methods return 0 on success, non-zero on failure.
        """
        if ret != 0:
            msg = f"SportClient.{method_name}() returned error code {ret}"
            if raise_on_error:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)


# ──────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Minimal smoke test — stand up, walk forward 0.5m, stop, sit.

    Usage:
        python src/robot/go2_interface.py
        python src/robot/go2_interface.py --interface enx98fc84e68f1a

    WARNING: Robot will move. Ensure 2m of clear space in front.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Go2Interface smoke test")
    parser.add_argument(
        "--interface",
        type=str,
        default=DEFAULT_NETWORK_INTERFACE,
        help="Ethernet interface name (default: enx98fc84e68f1a)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect only — do not send any motion commands"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Go2Interface Smoke Test")
    print("="*60)
    print(f"Interface: {args.interface}")
    print("WARNING: Robot will move. Ensure 2m clear space ahead.")
    input("Press Enter to continue (Ctrl+C to abort)...\n")

    robot = Go2Interface(args.interface)
    print("✓ Connected to Go2")

    if args.dry_run:
        print("Dry-run mode — skipping motion commands.")
    else:
        print("Standing up...")
        robot.stand(settle_time=2.0)
        print("✓ Standing")

        print("Walking forward 0.5m...")
        robot.move_to_position(dx=0.5, dy=-1)
        print("✓ Move complete")

        time.sleep(1.0)

        print("Sitting down...")
        robot.sit(settle_time=2.0)
        print("✓ Sitting")

    print("\nSmoke test complete.")