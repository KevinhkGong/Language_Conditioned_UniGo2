"""
scripts/collect_data.py — Synthetic Training Data Collection

Runs the heuristic baseline policy in PressEnv and saves (obs, action,
reward, done, info) trajectories for training the contact planner (Stage C).

Each episode is saved as a separate entry. Successful episodes (robot
reached the correct button) are flagged so the planner can train on
positive examples only if desired.

Output format: NumPy .npz files, one per collection run, saved to
data/synthetic/. Each file contains arrays of shape:
    obs:      (N, 44)   — full observation vectors
    actions:  (N, 12)   — heuristic joint offsets
    rewards:  (N,)      — per-step rewards
    dones:    (N,)      — bool, True when episode ended
    success:  (N,)      — bool, True when episode ended with correct press
    phase:    (N,)      — int, 0=approach 1=press (for debugging)
    target_id:(N,)      — int, which button (0-3) was the target

Where N = num_envs × num_episodes × steps_per_episode (approximately).

Usage (from repo root, inside env_isaaclab):
    python scripts/collect_data.py --num_envs 4 --num_episodes 50
    python scripts/collect_data.py --num_envs 64 --num_episodes 200 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# =============================================================================
# Section 1: App Launch (MUST happen before any isaaclab imports)
# =============================================================================

parser = argparse.ArgumentParser(description="Collect training data for Go2 press task")
parser.add_argument("--num_envs",     type=int,   default=4,
                    help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int,   default=50,
                    help="Number of episodes to collect per environment")
parser.add_argument("--max_steps",    type=int,   default=300,
                    help="Max steps per episode (env truncates at 300)")
parser.add_argument("--output_dir",   type=str,   default="data/synthetic",
                    help="Directory to save collected data")
parser.add_argument("--tag",          type=str,   default="heuristic",
                    help="Tag appended to output filename")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Section 2: Imports (safe after AppLauncher)
# =============================================================================

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.press_env import PressEnv, PressEnvCfg
from src.planner.heuristic_policy import HeuristicPolicy


# =============================================================================
# Section 3: Data Collection
# =============================================================================

def collect(
    env:         PressEnv,
    policy:      HeuristicPolicy,
    num_episodes: int,
    max_steps:   int,
) -> dict[str, np.ndarray]:
    """Run the heuristic policy and collect trajectories.

    Args:
        env:          Initialized PressEnv
        policy:       HeuristicPolicy instance
        num_episodes: Number of complete episodes to collect per env
        max_steps:    Safety cap on steps per episode

    Returns:
        Dictionary of stacked numpy arrays ready to save.
    """
    num_envs = env.num_envs

    # Accumulators — lists of per-step tensors, moved to CPU at end
    all_obs      = []
    all_actions  = []
    all_rewards  = []
    all_dones    = []
    all_success  = []
    all_phase    = []
    all_target_id = []

    # Track how many complete episodes each env has finished
    ep_counts  = torch.zeros(num_envs, dtype=torch.long)
    ep_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    obs, _ = env.reset()
    obs_tensor = obs["policy"]  # (num_envs, 44)

    total_steps  = 0
    episodes_done = 0
    target_episodes = num_episodes * num_envs

    print(f"\n[collect] Starting data collection")
    print(f"[collect] Target: {num_episodes} episodes × {num_envs} envs "
          f"= {target_episodes} total episodes")
    print(f"[collect] Max steps/episode: {max_steps}\n")

    t_start = time.time()

    while episodes_done < target_episodes:
        # --- Policy forward pass ---
        actions = policy(obs_tensor)           # (num_envs, 12)
        phase   = policy.get_phase(obs_tensor) # (num_envs,)

        # --- Step environment ---
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        # --- Record step data (move to CPU immediately to free GPU memory) ---
        all_obs.append(obs_tensor.cpu())
        all_actions.append(actions.cpu())
        all_rewards.append(reward.cpu())
        all_dones.append(done.cpu())
        all_phase.append(phase.cpu())
        all_target_id.append(env.target_button_idx.cpu())

        # Success = terminated (not truncated) — means robot reached target
        # (terminated includes falls too, but falls have large negative reward,
        #  so we further filter by reward sign below when saving)
        step_success = terminated & ~truncated & (reward > 0)
        all_success.append(step_success.cpu())

        # --- Track episode completions ---
        newly_done = done.cpu()
        episodes_done += newly_done.sum().item()

        # --- Advance observation ---
        obs_tensor = next_obs["policy"]
        total_steps += 1

        # --- Progress logging ---
        if total_steps % 100 == 0:
            elapsed = time.time() - t_start
            rate = episodes_done / elapsed if elapsed > 0 else 0
            print(
                f"[collect] step={total_steps:5d}  "
                f"episodes={episodes_done:4d}/{target_episodes}  "
                f"rate={rate:.1f} ep/s"
            )

        # Safety cap: avoid infinite loop if env never terminates
        if total_steps > max_steps * num_episodes * 2:
            print(f"[collect] WARNING: hit step cap, stopping early.")
            break

    elapsed = time.time() - t_start
    print(f"\n[collect] Done. {total_steps} steps in {elapsed:.1f}s "
          f"({total_steps/elapsed:.0f} steps/s)")

    # --- Stack and convert to numpy ---
    obs_arr       = torch.cat(all_obs,       dim=0).numpy()   # (N, 44)
    actions_arr   = torch.cat(all_actions,   dim=0).numpy()   # (N, 12)
    rewards_arr   = torch.cat(all_rewards,   dim=0).numpy()   # (N,)
    dones_arr     = torch.cat(all_dones,     dim=0).numpy()   # (N,)
    success_arr   = torch.cat(all_success,   dim=0).numpy()   # (N,)
    phase_arr     = torch.cat(all_phase,     dim=0).numpy()   # (N,)
    target_id_arr = torch.cat(all_target_id, dim=0).numpy()   # (N,)

    # --- Print collection stats ---
    n_total    = len(obs_arr)
    n_success  = success_arr.sum()
    n_episodes = dones_arr.sum()
    print(f"\n[collect] Dataset stats:")
    print(f"          Total steps    : {n_total:,}")
    print(f"          Total episodes : {int(n_episodes):,}")
    print(f"          Successes      : {int(n_success):,}  "
          f"({100*n_success/max(n_episodes,1):.1f}% of episodes)")
    print(f"          Approach steps : {(phase_arr == 0).sum():,}")
    print(f"          Press steps    : {(phase_arr == 1).sum():,}")
    print(f"          Button targets : {np.bincount(target_id_arr, minlength=4)}")

    return {
        "obs":       obs_arr,
        "actions":   actions_arr,
        "rewards":   rewards_arr,
        "dones":     dones_arr,
        "success":   success_arr,
        "phase":     phase_arr,
        "target_id": target_id_arr,
    }


# =============================================================================
# Section 4: Save
# =============================================================================

def save(data: dict[str, np.ndarray], output_dir: str, tag: str) -> str:
    """Save collected data as a .npz file.

    Args:
        data:       Dictionary of numpy arrays
        output_dir: Directory to write into
        tag:        Filename tag (e.g. "heuristic")

    Returns:
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename  = f"{tag}_{timestamp}.npz"
    filepath  = os.path.join(output_dir, filename)

    np.savez_compressed(filepath, **data)
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"\n[save] Saved {size_mb:.1f} MB → {filepath}")
    return filepath


# =============================================================================
# Section 5: Main
# =============================================================================

def main():
    # --- Build environment ---
    env_cfg = PressEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = PressEnv(
        cfg=env_cfg,
        render_mode=None if args_cli.headless else "human",
    )

    print(f"[main] Environment ready: {env.num_envs} envs on {env.device}")

    # --- Build heuristic policy ---
    # env is passed at construction time so the policy can slide the base
    policy = HeuristicPolicy(env=env)

    # --- Collect ---
    data = collect(
        env=env,
        policy=policy,
        num_episodes=args_cli.num_episodes,
        max_steps=args_cli.max_steps,
    )

    # --- Save ---
    save(data, output_dir=args_cli.output_dir, tag=args_cli.tag)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()