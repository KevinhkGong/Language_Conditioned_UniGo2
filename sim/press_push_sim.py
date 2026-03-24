"""
sim/press_push_sim.py — CLI Entry Point for Press Task Simulation

Handles AppLauncher startup, then imports PressEnv from press_env.py.
The environment definition (PressEnvCfg, PressEnv) lives in press_env.py
so it can be imported by other scripts (smoke tests, training, eval)
without triggering a second AppLauncher.

Usage:
    # Launch with GUI
    python sim/press_push_sim.py --num_envs 4

    # Headless for data generation
    python sim/press_push_sim.py --num_envs 64 --headless
"""

from __future__ import annotations

import argparse
import torch

# =============================================================================
# Section 1: App Launch (MUST happen before any isaaclab imports)
# =============================================================================

parser = argparse.ArgumentParser(description="Go2 Press Task Simulation")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments")
parser.add_argument("--config",   type=str, default="configs/default.yaml",
                    help="Path to config file")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse — nothing from isaaclab can be imported before this
app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# =============================================================================
# Section 2: Imports (safe after AppLauncher)
# =============================================================================

# Environment lives in press_env.py — import it here after app is live
from sim.press_env import PressEnv, PressEnvCfg   # noqa: E402


# =============================================================================
# Section 3: Main loop
# =============================================================================

def main():
    """Launch the press environment and step with random actions."""
    env_cfg = PressEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = PressEnv(
        cfg=env_cfg,
        render_mode=None if args_cli.headless else "human",
    )

    obs, info = env.reset()
    print(f"[INFO] Environment created with {env.num_envs} envs")
    print(f"[INFO] Observation shape: {obs['policy'].shape}")
    print(f"[INFO] Action space:      {env.action_space}")

    count = 0
    while simulation_app.is_running():
        actions = 2.0 * torch.rand(
            env.num_envs, env_cfg.action_space, device=env.device
        ) - 1.0

        obs, reward, terminated, truncated, info = env.step(actions)

        if count % 100 == 0:
            print(
                f"[Step {count:5d}]  "
                f"mean_reward={reward.mean().item():.3f}  "
                f"terminated={terminated.sum().item():.0f}  "
                f"truncated={truncated.sum().item():.0f}"
            )

        count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()