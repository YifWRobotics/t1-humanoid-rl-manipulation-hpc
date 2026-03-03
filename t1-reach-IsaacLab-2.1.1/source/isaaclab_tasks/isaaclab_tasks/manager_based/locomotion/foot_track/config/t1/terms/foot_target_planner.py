#!/usr/bin/env python3
import torch


class FootTargetPlanner:
    def __init__(
        self,
        debug: bool = False,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
        dt: float = 0.02,
        base_left: tuple[float, float, float] = (0.0, 0.1, 0.0),
        base_right: tuple[float, float, float] = (0.0, -0.1, 0.0),
        countdown_takes_s: float = 1.0,
        cmd_ranges: object | None = None,
    ):
        self.debug = debug
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dt = dt
        
        # Handle cmd_ranges - can be a Ranges object or dict with vel_x, vel_y, vel_yaw
        if cmd_ranges is None:
            # Default ranges if not provided
            self.cmd_vel_x_range = (-0.3, 0.3)
            self.cmd_vel_y_range = (-0.3, 0.3)
            self.cmd_vel_yaw_range = (-0.3, 0.3)
        else:
            # Extract ranges from Ranges object
            self.cmd_vel_x_range = cmd_ranges.lin_vel_x
            self.cmd_vel_y_range = cmd_ranges.lin_vel_y
            self.cmd_vel_yaw_range = cmd_ranges.ang_vel_z
        if self.debug:
            assert self.batch_size == 1, "When debug is True, batch_size must be 1."

        # hip offsets in local frame (COM = 0)
        base_left_t = torch.tensor(base_left, dtype=torch.float32, device=self.device)
        base_right_t = torch.tensor(base_right, dtype=torch.float32, device=self.device)
        self.left_hip_offset = base_left_t.expand(self.batch_size, -1).clone()
        self.right_hip_offset = base_right_t.expand(self.batch_size, -1).clone()
        self.nominal_right_to_left = base_left_t - base_right_t
        self.nominal_left_to_right = base_right_t - base_left_t

        # current command: [vx, vy, wz] per batch
        self.cmd = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)

        # constants as tensors
        self.countdown_takes_s = countdown_takes_s
        
        # Map to left/right
        self.left_foot_target_pos = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)
        self.right_foot_target_pos = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)
        self.left_foot_target_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.right_foot_target_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        
        self.cmd_stance = torch.randint(0, 2, (self.batch_size, 1), dtype=torch.bool, device=self.device)
        left_is_swing = self.cmd_stance
        self.cmd_stance_to_swing_pos = torch.where(
            left_is_swing,
            self.nominal_right_to_left.expand(self.batch_size, -1),
            self.nominal_left_to_right.expand(self.batch_size, -1)
        )
        # self.cmd_stance_to_swing_pos = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)
        self.cmd_stance_to_swing_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.cmd_countdown = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=self.device)
        
        
    # def _compute_targets_from_stance(self, env_ids: torch.Tensor, which_foot_is_swing: str):
    #     """
    #     Compute BOTH swing and stance foot targets from current stance.
    #     - Computes relative pose from stance to swing: 0.5*(v_xy + wyaw×hip_normal)*T
    #     - Adds hip offsets for spacing
    #     - Includes yaw targets: swing_yaw = wyaw*T, stance_yaw = -wyaw*T

    #     Args:
    #         env_ids: Long tensor of environment indices to update
    #         which_foot_is_swing: "left" or "right"

    #     Returns:
    #         swing_target (N,3), stance_target (N,3), swing_yaw (N,), stance_yaw (N,)
    #     """
    #     if env_ids.numel() == 0:
    #         # Return empty tensors with correct shapes when nothing to do
    #         empty_pos = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
    #         empty_yaw = torch.zeros(0, dtype=torch.float32, device=self.device)
    #         return empty_pos, empty_pos, empty_yaw, empty_yaw

    #     vx = self.cmd[env_ids, 0]
    #     vy = self.cmd[env_ids, 1]
    #     wz = self.cmd[env_ids, 2]

    #     if which_foot_is_swing == "left":
    #         swing_hip = self.left_hip_offset[env_ids]
    #         stance_hip = self.right_hip_offset[env_ids]
    #     else:
    #         swing_hip = self.right_hip_offset[env_ids]
    #         stance_hip = self.left_hip_offset[env_ids]

    #     # hip_normal perpendicular to swing hip offset in xy plane
    #     hip_normal = torch.stack([
    #         -swing_hip[:, 1],
    #         swing_hip[:, 0],
    #     ], dim=1)

    #     # Components
    #     xy_component = torch.stack([vx, vy], dim=1)
    #     yaw_component = wz.unsqueeze(1) * hip_normal

    #     # Relative step (Raibert) with 0.5 scaling
    #     step_xy = 0.5 * (xy_component + yaw_component) * self.single_swing_period

    #     # Swing target = step + swing hip offset
    #     swing_target_xy = step_xy + swing_hip[:, :2]
    #     # Stance target = -step + stance hip offset (rotational symmetry)
    #     stance_target_xy = -step_xy + stance_hip[:, :2]

    #     zeros_n = torch.zeros(step_xy.shape[0], dtype=torch.float32, device=self.device)

    #     swing_target = torch.cat([swing_target_xy, zeros_n.unsqueeze(-1)], dim=1)
    #     stance_target = torch.cat([stance_target_xy, zeros_n.unsqueeze(-1)], dim=1)

    #     swing_yaw = 0.5 * wz * self.single_swing_period
    #     stance_yaw = 0.5 * -wz * self.single_swing_period

    #     return swing_target, stance_target, swing_yaw, stance_yaw

    def update_cmd_countdown(self) -> torch.Tensor:
        """Return countdown timer (0 during wait, 1->0 during step), finish within self.single_swing_period."""
        self.cmd_countdown = torch.clamp(self.cmd_countdown - self.dt / self.countdown_takes_s, min=0.0, max=1.0)
        return self.cmd_countdown
    
    def _v_to_step_targets(self, env_ids: torch.Tensor, v_b: torch.Tensor):
        """
        Compute step targets using Zhaoyuan's method.
        """
        assert env_ids.shape[0] == v_b.shape[0], "env_ids and v_b must have the same number of environments"
        SWING_DURATION = 0.5

        left_is_swing = self.cmd_stance[env_ids]
        # N is env_ids.shape[0]
        vx = v_b[:, 0] # (N,)
        vy = v_b[:, 1]
        wz = v_b[:, 2]
        
        shift_from_nominal = torch.stack([
            vx * SWING_DURATION, 
            vy * SWING_DURATION,
            torch.zeros(env_ids.shape[0], dtype=torch.float32, device=self.device),
        ], dim=1) # (N, 3)
        
        nominal_stance_to_swing_pos = torch.where(
            left_is_swing,
            self.nominal_right_to_left.expand(env_ids.shape[0], -1),
            self.nominal_left_to_right.expand(env_ids.shape[0], -1)
        ) # (N, 3)
        
        self.cmd_stance_to_swing_pos[env_ids] = nominal_stance_to_swing_pos + shift_from_nominal
        self.cmd_stance_to_swing_yaw[env_ids] = wz * SWING_DURATION  # (N,)
        
        return self.cmd_stance_to_swing_pos, self.cmd_stance_to_swing_yaw
