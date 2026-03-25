"""
src/planner/heuristic_policy.py — Rule-Based Heuristic Baseline

A two-phase open-loop controller for the Go2 press task.
Uses privileged ground-truth target position from the observation vector
(obs[37:40]) — no visual perception required.
This is Baseline 1 from the project proposal.

Phase 1 — Slide (approach):
    Directly moves the robot base toward the target button each step via
    write_root_pose_to_sim. Joint actions are zero (robot holds standing pose).
    This is not physically realistic locomotion — it is a data collection
    shortcut that gets the robot to the panel deterministically every episode.
    NOTE: Replace with a pre-trained locomotion policy for real-robot deployment.

Phase 2 — Press:
    Once the robot base is within approach_threshold of the target button,
    base sliding stops. The front-right (FR) leg raises and extends toward
    the button using joint position offsets. The other 3 legs hold the
    default standing pose to maintain balance.

Joint ordering (verified from UNITREE_GO2_CFG):
    [0]  FL_hip_joint      [4]  FL_thigh_joint    [8]  FL_calf_joint
    [1]  FR_hip_joint      [5]  FR_thigh_joint     [9]  FR_calf_joint
    [2]  RL_hip_joint      [6]  RL_thigh_joint    [10]  RL_calf_joint
    [3]  RR_hip_joint      [7]  RR_thigh_joint    [11]  RR_calf_joint

Obs vector layout (44 dims):
    [0:12]   joint positions
    [12:24]  joint velocities
    [24:27]  base position (x, y, z) in world frame
    [27:31]  base quaternion (w, x, y, z)
    [31:34]  base linear velocity
    [34:37]  base angular velocity
    [37:40]  target button position (x, y, z) in world frame
    [40:44]  target button one-hot encoding
"""

from __future__ import annotations
import torch


# =============================================================================
# Joint index constants (verified from UNITREE_GO2_CFG)
# =============================================================================

FL_HIP,   FR_HIP,   RL_HIP,   RR_HIP   = 0, 1, 2, 3
FL_THIGH, FR_THIGH, RL_THIGH, RR_THIGH = 4, 5, 6, 7
FL_CALF,  FR_CALF,  RL_CALF,  RR_CALF  = 8, 9, 10, 11

NUM_JOINTS = 12


# =============================================================================
# Heuristic Policy
# =============================================================================

class HeuristicPolicy:
    """Two-phase rule-based controller for the Go2 button-pressing task.

    Phase 1 slides the robot base directly toward the target (bypasses
    locomotion). Phase 2 uses FR leg joint offsets to press the button.

    Args:
        env:                 PressEnv instance — needed for base sliding
        approach_threshold:  distance (m) at which robot switches to press phase
        slide_speed:         meters per step to move base forward in Phase 1
        lateral_gain:        how aggressively to correct lateral offset
    """

    def __init__(
        self,
        env,
        approach_threshold: float = 0.20,
        slide_speed:        float = 0.02,
        lateral_gain:       float = 0.8,
    ):
        self.env                = env
        self.approach_threshold = approach_threshold
        self.slide_speed        = slide_speed
        self.lateral_gain       = lateral_gain
        self.device             = str(env.device)



        # Cache FR foot body index
        fr_foot_indices, _ = self.env.robot.find_bodies("FR_foot")
        self._fr_foot_idx = fr_foot_indices[0]

        # FR foot offset from base measured at reset (before physics settles).
        # foot_world = base_world + _foot_offset
        # To place foot at button: base_target = button_pos - _foot_offset
        # Gives base_z = button_z + 0.300 = 0.340m (slight crouch, not floating)
        self._foot_offset = torch.tensor(
            [+0.178, -0.173, -0.300], device=self.device
        )

        # Cache default joint positions for Phase 1 stabilisation
        self._default_joint_pos: torch.Tensor | None = None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actions and slide base for a batch of observations.

        Args:
            obs: (num_envs, 44) observation tensor from PressEnv

        Returns:
            actions: (num_envs, 12) joint position offsets.
                     Phase 1 envs return all zeros (base is moved directly).
                     Phase 2 envs return FR leg press offsets.
        """
        num_envs = obs.shape[0]

        # --- Extract positions ---
        base_pos   = obs[:, 24:27]   # (num_envs, 3) world frame
        base_quat  = obs[:, 27:31]   # (num_envs, 4) quaternion wxyz
        target_pos = obs[:, 37:40]   # (num_envs, 3) world frame

        # Compute how far the base needs to move to place foot at button.
        # base_target = button_pos - foot_offset
        # delta = base_target - base_pos = (button_pos - foot_offset) - base_pos
        base_target_x = target_pos[:, 0] - self._foot_offset[0]
        base_target_y = target_pos[:, 1] - self._foot_offset[1]
        delta_x = base_target_x - base_pos[:, 0]   # how far base needs to go forward
        delta_y = base_target_y - base_pos[:, 1]   # how far base needs to go laterally

        # --- Phase flag ---
        # True  = within press range AND not badly overshot
        # False = still approaching (or overshot past -0.05m — treat as failed)
        # Fix C: guard against delta_x going negative (robot slid past panel).
        # If delta_x < -0.05 the robot has overshot badly; we fall through to
        # approaching=True so it stops sliding and the episode times out cleanly.
        in_press_phase = (delta_x < self.approach_threshold) & (delta_x > -0.05)
        approaching    = ~in_press_phase

        # --- Joint actions (default: all zeros = hold standing pose) ---
        actions = torch.zeros(num_envs, NUM_JOINTS, device=self.device)

        # Cache default joint positions on first call (robot data is ready by then)
        if self._default_joint_pos is None:
            self._default_joint_pos = self.env.robot.data.default_joint_pos.clone()

        # =================================================================
        # Phase 1 — Slide base toward target + stabilise joints
        # =================================================================
        # Directly write new root pose for approaching envs each step.
        # Forward step is clamped to slide_speed so we don't overshoot.
        # Lateral step steers toward button y proportionally.
        #
        # FIX: We also write default joint positions back every step during
        # Phase 1. Without this, PhysX simulates the legs under gravity as
        # the base moves, causing the robot to collapse. Writing joint state
        # each step keeps the legs locked at the standing pose so the body
        # stays upright while sliding.

        if approaching.any():
            approaching_ids = torch.where(approaching)[0]

            # Current pose for approaching envs only
            current_pos  = base_pos[approaching].clone()    # (k, 3)
            current_quat = base_quat[approaching].clone()   # (k, 4)

            # Fix A: step forward but stop exactly at approach_threshold.
            # Subtracting the threshold means we stop when delta_x == threshold,
            # never overshooting into negative delta_x (past the panel).
            fwd_step = torch.clamp(
                delta_x[approaching] - self.approach_threshold,
                min=0.0, max=self.slide_speed,
            )
            current_pos[:, 0] += fwd_step

            # Lateral: proportional correction toward button y
            # Lateral correction gets its own cap (2x slide_speed) so the
            # robot can correct y-offset faster than it moves forward in x.
            lat_step = torch.clamp(
                self.lateral_gain * delta_y[approaching],
                min=-self.slide_speed * 2,
                max= self.slide_speed * 2,
            )
            current_pos[:, 1] += lat_step

            # Write base pose — Isaac Lab format: [pos(3), quat(4)] = 7 dims
            root_pose = torch.cat([current_pos, current_quat], dim=-1)  # (k, 7)
            self.env.robot.write_root_pose_to_sim(
                root_pose, env_ids=approaching_ids
            )

            # Fix B: Hold joints at default pose via actions rather than
            # write_joint_state_to_sim. Writing joint state directly fights
            # PhysX every step and causes oscillation/collapse. Instead we
            # command the PD controller to hold default positions by returning
            # zero actions (which _pre_physics_step maps to default_pos + 0).
            # This lets the sim's own actuator stiffness resist gravity cleanly.
            # (actions already initialised to zeros above — no extra code needed)

        # =================================================================
        # Phase 2 — Final approach: slide base until foot contacts button
        # =================================================================
        # Buttons are now at foot height (z≈0.027–0.053m), so the FR foot
        # reaches them naturally as the base slides forward. No leg raise
        # needed. We continue sliding slowly (half speed) to close the
        # final gap and trigger the proximity contact detection.

        pressing = in_press_phase

        if pressing.any():
            pressing_ids = torch.where(pressing)[0]

            # Slide base gradually toward the target position (same as Phase 1).
            # base_target = button_pos - foot_offset (x and y only, keep z).
            # Moving gradually avoids the destabilizing lurch from a full teleport.
            btn_pos  = target_pos[pressing]                             # (k, 3)
            base_tgt = btn_pos - self._foot_offset.unsqueeze(0)        # (k, 3)

            current_pos  = base_pos[pressing].clone()
            current_quat = base_quat[pressing].clone()

            # Step toward base_target in x and y, clamped to slide_speed
            dx = base_tgt[:, 0] - current_pos[:, 0]
            dy = base_tgt[:, 1] - current_pos[:, 1]
            current_pos[:, 0] += torch.clamp(dx, min=-self.slide_speed, max=self.slide_speed)
            current_pos[:, 1] += torch.clamp(dy, min=-self.slide_speed * 2, max=self.slide_speed * 2)
            # z unchanged — keep current crouched height

            root_pose = torch.cat([current_pos, current_quat], dim=-1)
            self.env.robot.write_root_pose_to_sim(root_pose, env_ids=pressing_ids)

        return actions

    def get_phase(self, obs: torch.Tensor) -> torch.Tensor:
        """Return phase label per env: 0 = approach, 1 = press.

        Used for labelling collected trajectories.

        Args:
            obs: (num_envs, 44)

        Returns:
            phase: (num_envs,) int tensor, values in {0, 1}
        """
        base_pos   = obs[:, 24:27]
        target_pos = obs[:, 37:40]
        delta_x    = target_pos[:, 0] - base_pos[:, 0]
        return (
            (delta_x < self.approach_threshold) & (delta_x > -0.05)
        ).long()