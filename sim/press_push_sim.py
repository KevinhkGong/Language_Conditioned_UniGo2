"""
Simulation Environment — Press Task (Isaac Lab)

A Unitree Go2 quadruped must approach a panel and press the correct colored
button among distractors. Built on Isaac Lab's DirectRLEnv for gym compatibility
and GPU-parallelized simulation.

Usage:
    # Launch with GUI (requires Isaac Lab installed)
    python sim/press_push_sim.py --num_envs 4

    # Headless for data generation
    python sim/press_push_sim.py --num_envs 64 --headless

Note: Isaac Lab requires the AppLauncher to be called BEFORE any isaaclab imports.
      This is why the import structure below looks unusual — it's mandatory.
"""

from __future__ import annotations

import argparse
import torch
import yaml

# =============================================================================
# Section 1: App Launch (MUST happen before any isaaclab imports)
# =============================================================================

parser = argparse.ArgumentParser(description="Go2 Press Task Simulation")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

# Isaac Lab adds its own CLI args (--headless, --device, etc.)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Omniverse application — nothing from isaaclab can be imported before this
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Section 2: Imports (safe to import isaaclab modules AFTER app launch)
# =============================================================================

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# Pre-defined Go2 articulation configuration from Isaac Lab assets
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


# =============================================================================
# Section 3: Environment Configuration
# =============================================================================

@configclass
class PressEnvCfg(DirectRLEnvCfg):
    """Configuration for the Go2 button-pressing environment.

    This defines everything about the simulation: physics, robot, scene layout,
    task parameters, and observation/action spaces.
    """

    # --- Environment timing ---
    decimation = 4                  # sim steps per env step (control at 30 Hz)
    episode_length_s = 10.0         # max seconds per episode

    # --- Action space ---
    # 12 joints on Go2: 3 per leg (hip, thigh, calf) × 4 legs
    action_space = 12

    # --- Observation space ---
    # Robot state (12 joint pos + 12 joint vel + 3 base pos + 4 base quat + 6 base vel)
    #   = 37 dims
    # Target button position (3 dims) + target button ID one-hot (4 dims)
    #   = 7 dims
    # Total = 44
    observation_space = 44
    state_space = 0

    # --- Simulation physics ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                # physics timestep
        render_interval=decimation,
        disable_contact_processing=False,  # we need contact info for press detection
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- Terrain ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # --- Robot ---
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- Task parameters ---
    # Panel: a flat rigid body that holds the buttons
    panel_distance = 0.8            # meters in front of robot
    panel_height = 0.15             # height of button centers above ground
    panel_size = (0.3, 0.3, 0.02)  # width, height, depth (meters)

    # Buttons: small cylinders on the panel surface
    num_buttons = 4                 # total buttons (1 target + 3 distractors)
    button_radius = 0.02           # meters
    button_depth = 0.015           # how far buttons protrude from panel

    # Contact detection
    contact_force_threshold = 1.0  # Newtons — minimum force to count as a "press"

    # Reward weights
    reward_distance_weight = -1.0   # penalize distance to target
    reward_contact_correct = 10.0   # reward for pressing correct button
    reward_contact_wrong = -5.0     # penalty for pressing wrong button
    reward_action_penalty = -0.01   # penalize large actions (smooth motion)


# =============================================================================
# Section 4: Environment Implementation
# =============================================================================

class PressEnv(DirectRLEnv):
    """Go2 button-pressing environment.

    The robot starts facing a panel with colored buttons. It must use its body
    and legs to press the designated target button. This environment generates
    training data for the contact planner (Stage C of the pipeline).

    Observations:
        - Joint positions (12)
        - Joint velocities (12)
        - Base position (3)
        - Base orientation quaternion (4)
        - Base linear + angular velocity (6)
        - Target button position in world frame (3)
        - Target button one-hot encoding (4)
        Total: 44

    Actions:
        - Joint position targets for all 12 joints

    Rewards:
        - Distance to target button (negative, encourages approach)
        - Correct contact bonus
        - Wrong contact penalty
        - Action smoothness penalty
    """

    cfg: PressEnvCfg

    def __init__(self, cfg: PressEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache joint indices for the Go2's 12 actuated joints
        self.joint_idx, _ = self.robot.find_joints(".*")

        # Buffers for tracking which button is the target per environment
        # Shape: (num_envs,) — integer index into [0, num_buttons)
        self.target_button_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Button positions in local env frame: (num_envs, num_buttons, 3)
        self.button_positions = torch.zeros(
            self.num_envs, self.cfg.num_buttons, 3, device=self.device
        )

        # Pre-compute button layout on the panel (2×2 grid)
        self._init_button_layout()

    def _init_button_layout(self):
        """Compute the 2×2 grid positions for buttons on the panel.

        Layout (viewed from robot's perspective):
            [0] [1]     (top row)
            [2] [3]     (bottom row)
        """
        spacing = self.cfg.panel_size[0] / 3  # spacing between buttons
        offsets = torch.tensor([
            [-spacing / 2,  spacing / 2],   # button 0: top-left
            [ spacing / 2,  spacing / 2],   # button 1: top-right
            [-spacing / 2, -spacing / 2],   # button 2: bottom-left
            [ spacing / 2, -spacing / 2],   # button 3: bottom-right
        ], device=self.device)

        for env_id in range(self.num_envs):
            for btn_id in range(self.cfg.num_buttons):
                self.button_positions[env_id, btn_id, 0] = self.cfg.panel_distance  # x: forward
                self.button_positions[env_id, btn_id, 1] = offsets[btn_id, 0]       # y: lateral
                self.button_positions[env_id, btn_id, 2] = (
                    self.cfg.panel_height + offsets[btn_id, 1]
                )  # z: vertical

    # ---- Scene Setup ----

    def _setup_scene(self):
        """Spawn all assets into the simulation scene.

        This method is called once during environment initialization. It creates:
        1. The Go2 robot (from UNITREE_GO2_CFG)
        2. A ground plane
        3. The button panel with colored buttons
        4. Scene lighting
        """
        # Spawn the Go2 robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Spawn the panel as a static rigid body (thin box)
        panel_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Panel",
            spawn=sim_utils.CuboidCfg(
                size=self.cfg.panel_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # has collision geometry but doesn't respond to froces - fixed in place
                    kinematic_enabled=True,  # panel doesn't move
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5),  # grey panel
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(self.cfg.panel_distance, 0.0, self.cfg.panel_height),
            ),
        )
        self.panel = RigidObject(panel_cfg)

        # Spawn buttons as small colored cylinders on the panel
        # Colors: red, blue, green, yellow
        button_colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.0, 1.0),  # blue
            (0.0, 1.0, 0.0),  # green
            (1.0, 1.0, 0.0),  # yellow
        ]

        self.buttons = []
        spacing = self.cfg.panel_size[0] / 3
        offsets = [
            (-spacing / 2,  spacing / 2),
            ( spacing / 2,  spacing / 2),
            (-spacing / 2, -spacing / 2),
            ( spacing / 2, -spacing / 2),
        ]

        for i in range(self.cfg.num_buttons):
            btn_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Button_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.button_radius,
                    height=self.cfg.button_depth,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # buttons are fixed
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=button_colors[i],
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(
                        self.cfg.panel_distance - self.cfg.panel_size[2] / 2 - self.cfg.button_depth / 2,
                        offsets[i][0],
                        self.cfg.panel_height + offsets[i][1],
                    ),
                    rot=(0.707, 0.0, 0.707, 0.0),  # rotate cylinder to face robot
                ),
            )
            self.buttons.append(RigidObject(btn_cfg))

        # Clone environments and register assets
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["panel"] = self.panel
        for i, btn in enumerate(self.buttons):
            self.scene.rigid_objects[f"button_{i}"] = btn

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---- Actions ----

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process raw actions before applying to simulation.

        Actions are joint position targets scaled around the default standing pose.
        We clamp them to safe joint limits.

        Args:
            actions: (num_envs, 12) raw action tensor from policy
        """
        # Scale actions and add to default joint positions
        self.actions = actions.clone()
        default_pos = self.robot.data.default_joint_pos
        self.joint_targets = default_pos + actions * 0.25  # scale factor for safety

    def _apply_action(self):
        """Write the computed joint targets into the simulation."""
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.joint_idx)

    # ---- Observations ----

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for the policy.

        Returns:
            Dictionary with "policy" key containing the observation tensor.

        Observation vector (44 dims):
            [0:12]   Joint positions
            [12:24]  Joint velocities
            [24:27]  Base position (x, y, z)
            [27:31]  Base orientation (quaternion wxyz)
            [31:37]  Base velocity (linear xyz + angular xyz)
            [37:40]  Target button position (x, y, z) in world frame
            [40:44]  Target button one-hot encoding
        """
        # Robot proprioception
        joint_pos = self.robot.data.joint_pos                    # (num_envs, 12)
        joint_vel = self.robot.data.joint_vel                    # (num_envs, 12)
        base_pos = self.robot.data.root_pos_w                   # (num_envs, 3)
        base_quat = self.robot.data.root_quat_w                 # (num_envs, 4)
        base_lin_vel = self.robot.data.root_lin_vel_w            # (num_envs, 3)
        base_ang_vel = self.robot.data.root_ang_vel_w            # (num_envs, 3)

        # Target info
        # Gather the position of each env's target button
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        for env_id in range(self.num_envs):
            target_pos[env_id] = self.button_positions[env_id, self.target_button_idx[env_id]]

        # Add scene origin offset to get world-frame position
        target_pos_world = target_pos + self.scene.env_origins

        # One-hot encoding of target button
        target_onehot = torch.zeros(
            self.num_envs, self.cfg.num_buttons, device=self.device
        )
        target_onehot.scatter_(1, self.target_button_idx.unsqueeze(1), 1.0)

        obs = torch.cat([
            joint_pos,           # 12
            joint_vel,           # 12
            base_pos,            # 3
            base_quat,           # 4
            base_lin_vel,        # 3
            base_ang_vel,        # 3
            target_pos_world,    # 3
            target_onehot,       # 4
        ], dim=-1)              # total: 44

        return {"policy": obs}

    # ---- Rewards ----

    def _get_rewards(self) -> torch.Tensor:
        """Compute per-environment reward.

        Reward components:
            1. Distance penalty: negative distance from robot base to target button
            2. Contact reward: large bonus for pressing correct button
            3. Wrong contact penalty: negative reward for pressing wrong button
            4. Action penalty: penalizes large joint position changes
        """
        # Current robot base position
        base_pos = self.robot.data.root_pos_w  # (num_envs, 3)

        # Target button world position
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        for env_id in range(self.num_envs):
            target_pos[env_id] = self.button_positions[env_id, self.target_button_idx[env_id]]
        target_pos_world = target_pos + self.scene.env_origins

        # 1. Distance reward: encourage approaching the target
        dist = torch.norm(base_pos - target_pos_world, dim=-1)
        reward_dist = self.cfg.reward_distance_weight * dist

        # 2. Contact detection (simplified: check if robot is very close to a button)
        # In a full implementation, we'd use Isaac Lab's ContactSensor
        reward_contact = torch.zeros(self.num_envs, device=self.device)
        contact_threshold = self.cfg.button_radius * 3  # proximity threshold

        for btn_id in range(self.cfg.num_buttons):
            btn_pos = self.button_positions[:, btn_id] + self.scene.env_origins
            btn_dist = torch.norm(base_pos - btn_pos, dim=-1)
            in_contact = btn_dist < contact_threshold

            # Correct button
            is_target = (self.target_button_idx == btn_id)
            reward_contact += (
                in_contact & is_target
            ).float() * self.cfg.reward_contact_correct

            # Wrong button
            reward_contact += (
                in_contact & ~is_target
            ).float() * self.cfg.reward_contact_wrong

        # 3. Action penalty
        reward_action = self.cfg.reward_action_penalty * torch.sum(
            self.actions ** 2, dim=-1
        )

        total_reward = reward_dist + reward_contact + reward_action
        return total_reward

    # ---- Termination / Truncation ----

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes should end.

        Returns:
            terminated: (num_envs,) — True if robot successfully pressed correct button
                        or fell over (z height too low)
            truncated:  (num_envs,) — True if episode time limit reached
        """
        # Terminated: robot fell over (base height below threshold)
        base_height = self.robot.data.root_pos_w[:, 2]
        fell_over = base_height < 0.15  # Go2 standing height is ~0.35m

        # Terminated: correct button pressed (proximity check)
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        for env_id in range(self.num_envs):
            target_pos[env_id] = self.button_positions[env_id, self.target_button_idx[env_id]]
        target_pos_world = target_pos + self.scene.env_origins
        base_pos = self.robot.data.root_pos_w
        dist_to_target = torch.norm(base_pos - target_pos_world, dim=-1)
        success = dist_to_target < (self.cfg.button_radius * 3)

        terminated = fell_over | success

        # Truncated: time limit
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    # ---- Resets ----

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset specific environments.

        Randomizes:
            - Robot starting pose (small perturbation from default)
            - Which button is the target (uniform random from 0..num_buttons-1)

        Args:
            env_ids: indices of environments to reset
        """
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Reset robot to default standing pose with small random perturbation
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Small random position offset (±5cm lateral, ±2cm forward)
        default_root_state[:, 0] += sample_uniform(-0.02, 0.02, (num_resets,), device=self.device)
        default_root_state[:, 1] += sample_uniform(-0.05, 0.05, (num_resets,), device=self.device)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joint positions to default
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        # Randomize target button for each reset environment
        self.target_button_idx[env_ids] = torch.randint(
            0, self.cfg.num_buttons, (num_resets,), device=self.device
        )


# =============================================================================
# Section 5: Gym Registration
# =============================================================================

import gymnasium as gym

gym.register(
    id="Isaac-Go2-Press-Direct-v0",
    entry_point=f"{__name__}:PressEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PressEnvCfg,
    },
)


# =============================================================================
# Section 6: CLI — Run with random actions for testing
# =============================================================================

def main():
    """Launch the press environment and step with random actions."""
    # Override num_envs from CLI
    env_cfg = PressEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create the environment
    env = PressEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else "human")

    # Reset once before stepping
    obs, info = env.reset()
    print(f"[INFO] Environment created with {env.num_envs} envs")
    print(f"[INFO] Observation shape: {obs['policy'].shape}")
    print(f"[INFO] Action space: {env.action_space}")

    # Step loop with random actions
    count = 0
    while simulation_app.is_running():
        # Sample random joint position targets
        actions = 2.0 * torch.rand(env.num_envs, env_cfg.action_space, device=env.device) - 1.0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(actions)

        if count % 100 == 0:
            print(
                f"[Step {count:5d}] "
                f"mean_reward={reward.mean().item():.3f}  "
                f"terminated={terminated.sum().item():.0f}  "
                f"truncated={truncated.sum().item():.0f}"
            )

        count += 1

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()