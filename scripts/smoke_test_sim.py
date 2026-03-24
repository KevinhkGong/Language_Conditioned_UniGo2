"""
scripts/smoke_test_sim.py — Smoke Test for PressEnv

Validates that the press_push_sim environment initializes correctly and
that all tensor shapes, observation slices, rewards, and reset behavior
are consistent with the spec in sim/press_env.py.

Usage (from repo root, inside env_isaaclab conda env):
    python scripts/smoke_test_sim.py --num_envs 4
    python scripts/smoke_test_sim.py --num_envs 4 --headless

IMPORTANT: Follows the mandatory AppLauncher-first pattern. AppLauncher
must run before any isaaclab module is imported. PressEnv is imported from
sim/press_env.py (no AppLauncher there), not from press_push_sim.py.
"""

from __future__ import annotations

import argparse
import sys
import os

# =============================================================================
# Section 1: App Launch (MUST happen before any isaaclab imports)
# =============================================================================

parser = argparse.ArgumentParser(description="Smoke test for Go2 Press Environment")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments")
parser.add_argument("--steps",    type=int, default=50,
                    help="Number of steps in the multi-step loop check")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Section 2: Imports (safe after AppLauncher)
# =============================================================================

import torch

# Ensure repo root is on sys.path so `sim.press_env` resolves correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from press_env.py — no AppLauncher there, safe to import
from sim.press_env import PressEnv, PressEnvCfg   # noqa: E402

# =============================================================================
# Section 3: Helpers
# =============================================================================

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

_results: list[tuple[str, str]] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    _results.append((label, "PASS" if condition else "FAIL"))
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def obs_summary(obs_tensor: torch.Tensor):
    """Print a per-slice breakdown of the observation vector for env 0."""
    slices = {
        "joint_pos   [0:12] ": obs_tensor[0,  0:12],
        "joint_vel  [12:24] ": obs_tensor[0, 12:24],
        "base_pos   [24:27] ": obs_tensor[0, 24:27],
        "base_quat  [27:31] ": obs_tensor[0, 27:31],
        "base_linvel[31:34] ": obs_tensor[0, 31:34],
        "base_angvel[34:37] ": obs_tensor[0, 34:37],
        "target_pos [37:40] ": obs_tensor[0, 37:40],
        "target_1hot[40:44] ": obs_tensor[0, 40:44],
    }
    print(f"\n  {INFO}  Obs slice values (env 0):")
    for name, vals in slices.items():
        vals_str = "  ".join(f"{v:+.3f}" for v in vals.tolist())
        print(f"         {name}  {vals_str}")


# =============================================================================
# Section 4: Smoke Tests
# =============================================================================

def run_smoke_tests():
    num_envs  = args_cli.num_envs
    num_steps = args_cli.steps

    # ------------------------------------------------------------------
    # CHECK 1: Environment construction
    # ------------------------------------------------------------------
    section("CHECK 1 — Environment Construction")

    try:
        env_cfg = PressEnvCfg()
        env_cfg.scene.num_envs = num_envs
        render_mode = None if args_cli.headless else "human"
        env = PressEnv(cfg=env_cfg, render_mode=render_mode)
        check("PressEnv constructed without exception", True)
    except Exception as e:
        check("PressEnv constructed without exception", False, str(e))
        print(f"\n  {FAIL}  Cannot continue — environment failed to construct.")
        return

    check("env.num_envs matches requested",
          env.num_envs == num_envs,
          f"got {env.num_envs}, expected {num_envs}")

    check("env.device is a string",
          isinstance(env.device, str),
          f"device={env.device}")

    # ------------------------------------------------------------------
    # CHECK 2: Reset
    # ------------------------------------------------------------------
    section("CHECK 2 — Reset")

    try:
        obs, info = env.reset()
        check("env.reset() returned without exception", True)
    except Exception as e:
        check("env.reset() returned without exception", False, str(e))
        return

    check("obs is a dict",
          isinstance(obs, dict),
          f"type={type(obs)}")

    check("obs contains 'policy' key",
          "policy" in obs,
          f"keys={list(obs.keys())}")

    obs_tensor = obs["policy"]

    check("obs['policy'] is a torch.Tensor",
          isinstance(obs_tensor, torch.Tensor))

    check("obs['policy'].shape == (num_envs, 44)",
          obs_tensor.shape == (num_envs, 44),
          f"got {tuple(obs_tensor.shape)}")

    check("obs['policy'] contains no NaN",
          not obs_tensor.isnan().any().item())

    check("obs['policy'] contains no Inf",
          not obs_tensor.isinf().any().item())

    obs_summary(obs_tensor)

    # ------------------------------------------------------------------
    # CHECK 3: Observation slice sanity
    # ------------------------------------------------------------------
    section("CHECK 3 — Observation Slice Sanity")

    joint_pos   = obs_tensor[:,  0:12]
    joint_vel   = obs_tensor[:, 12:24]
    base_pos    = obs_tensor[:, 24:27]
    base_quat   = obs_tensor[:, 27:31]
    target_pos  = obs_tensor[:, 37:40]
    target_1hot = obs_tensor[:, 40:44]

    check("joint_pos values within [-4, 4] rad",
          (joint_pos.abs() < 4.0).all().item(),
          f"max_abs={joint_pos.abs().max().item():.3f}")

    check("joint_vel near zero after reset (|vel| < 1.0 rad/s)",
          (joint_vel.abs() < 1.0).all().item(),
          f"max_abs={joint_vel.abs().max().item():.3f}")

    base_z = base_pos[:, 2]
    check("base_z in [0.2, 0.6] m after reset (standing height)",
          ((base_z > 0.2) & (base_z < 0.6)).all().item(),
          f"mean={base_z.mean().item():.3f}")

    quat_norms = torch.norm(base_quat, dim=-1)
    check("base_quat is unit-length (|q| ≈ 1.0, tol=1e-3)",
          ((quat_norms - 1.0).abs() < 1e-3).all().item(),
          f"norms={[f'{v:.4f}' for v in quat_norms.tolist()]}")

    # Target must be in front of the robot in each env's local frame.
    # In world frame: target.x > base.x (panel is panel_distance=0.8m ahead).
    # Checking absolute x > 0 is wrong because env_origins can have negative x.
    base_pos_cur = obs_tensor[:, 24:27]
    target_ahead = (target_pos[:, 0] > base_pos_cur[:, 0]).all().item()
    target_offset = (target_pos[:, 0] - base_pos_cur[:, 0])
    check("target_pos x > base_pos x (panel is in front of robot)",
          target_ahead,
          f"offsets={[f'{v:.3f}' for v in target_offset.tolist()]}")

    onehot_sum = target_1hot.sum(dim=-1)
    check("target one-hot sums to 1.0 per env",
          ((onehot_sum - 1.0).abs() < 1e-4).all().item(),
          f"sums={onehot_sum.tolist()}")

    check("target one-hot values are only 0 or 1",
          ((target_1hot == 0) | (target_1hot == 1)).all().item())

    # ------------------------------------------------------------------
    # CHECK 4: Single step with zero actions
    # ------------------------------------------------------------------
    section("CHECK 4 — Single Step (zero actions)")

    zero_actions = torch.zeros(num_envs, env_cfg.action_space, device=env.device)

    try:
        obs2, reward, terminated, truncated, info2 = env.step(zero_actions)
        check("env.step() returned without exception", True)
    except Exception as e:
        check("env.step() returned without exception", False, str(e))
        return

    check("obs2['policy'].shape == (num_envs, 44)",
          obs2["policy"].shape == (num_envs, 44),
          f"got {tuple(obs2['policy'].shape)}")

    check("reward.shape == (num_envs,)",
          reward.shape == (num_envs,),
          f"got {tuple(reward.shape)}")

    check("terminated.shape == (num_envs,)",
          terminated.shape == (num_envs,),
          f"got {tuple(terminated.shape)}")

    check("truncated.shape == (num_envs,)",
          truncated.shape == (num_envs,),
          f"got {tuple(truncated.shape)}")

    # Robot is ~0.8m from panel at reset → distance penalty ≈ -0.8 → reward < 0
    check("reward is negative at step 0 (distance penalty dominates)",
          (reward < 0).all().item(),
          f"rewards={[f'{v:.3f}' for v in reward.tolist()]}")

    check("terminated all False at step 0",
          (~terminated).all().item(),
          f"terminated={terminated.tolist()}")

    check("truncated all False at step 0",
          (~truncated).all().item(),
          f"truncated={truncated.tolist()}")

    print(f"\n  {INFO}  Step 0 rewards: {[f'{v:.3f}' for v in reward.tolist()]}")

    # ------------------------------------------------------------------
    # CHECK 5: Multi-step loop with random actions
    # ------------------------------------------------------------------
    section(f"CHECK 5 — Multi-Step Loop ({num_steps} steps, random actions)")

    obs3, _ = env.reset()
    total_terminated = 0
    total_truncated  = 0
    reward_history   = []
    nan_detected     = False

    for step in range(num_steps):
        rand_actions = 2.0 * torch.rand(
            num_envs, env_cfg.action_space, device=env.device
        ) - 1.0

        obs3, reward3, term3, trunc3, _ = env.step(rand_actions)

        total_terminated += term3.sum().item()
        total_truncated  += trunc3.sum().item()
        reward_history.append(reward3.mean().item())

        if obs3["policy"].isnan().any():
            nan_detected = True
            print(f"  {WARN}  NaN in obs at step {step}")
            break

    check("No NaN in obs during multi-step loop", not nan_detected)

    mean_r = sum(reward_history) / len(reward_history)
    print(f"\n  {INFO}  Reward over {num_steps} steps — "
          f"mean={mean_r:.3f}  "
          f"min={min(reward_history):.3f}  "
          f"max={max(reward_history):.3f}")
    print(f"  {INFO}  Total terminations : {total_terminated}")
    print(f"  {INFO}  Total truncations  : {total_truncated}")

    check("Mean reward is finite", abs(mean_r) < 1e6, f"mean={mean_r:.3f}")

    # ------------------------------------------------------------------
    # CHECK 6: Forced full reset after episode loop
    # ------------------------------------------------------------------
    section("CHECK 6 — Forced Full Reset")

    try:
        obs4, _ = env.reset()
        check("env.reset() after episode loop succeeded", True)
    except Exception as e:
        check("env.reset() after episode loop succeeded", False, str(e))
        return

    check("obs shape still (num_envs, 44) after reset",
          obs4["policy"].shape == (num_envs, 44),
          f"got {tuple(obs4['policy'].shape)}")

    check("obs contains no NaN after reset",
          not obs4["policy"].isnan().any().item())

    # ------------------------------------------------------------------
    # CHECK 7: Target button randomization
    # ------------------------------------------------------------------
    section("CHECK 7 — Target Button Randomization")

    seen_targets: set[int] = set()
    for _ in range(20):
        env.reset()
        seen_targets.update(env.target_button_idx.tolist())

    check("All 4 button IDs seen across 20 resets",
          seen_targets == {0, 1, 2, 3},
          f"seen={sorted(seen_targets)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")

    passed = sum(1 for _, r in _results if r == "PASS")
    failed = sum(1 for _, r in _results if r == "FAIL")
    total  = len(_results)

    print(f"\n  Total checks : {total}")
    print(f"  {PASS}  Passed : {passed}")

    if failed > 0:
        print(f"  {FAIL}  Failed : {failed}")
        print(f"\n  Failed checks:")
        for label, result in _results:
            if result == "FAIL":
                print(f"    - {label}")
    else:
        print(f"\n  All checks passed — environment looks healthy.\n")

    env.close()


# =============================================================================
# Section 5: Entry Point
# =============================================================================

if __name__ == "__main__":
    run_smoke_tests()
    simulation_app.close()