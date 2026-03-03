#!/usr/bin/env python3
# import pygame
import torch
import math
from collections import deque
from collections.abc import Sequence

# -----------------------------
# constants / params
# -----------------------------
WIDTH, HEIGHT = 800, 800
FPS = 60

SCALE = 600.0  # meters -> pixels (0.5 m -> 300 px) tweak to taste
CENTER = (WIDTH // 2, HEIGHT // 2)

SINGLE_SWING_PERIOD = 0.5
DELAY_TIMESTEPS = 40
CMD_STEP = 0.1
CMD_MAX = 0.3
HISTORY_LEN = 120
DT = 1.0 / FPS  # time step for phase clock

BG_COLOR = (25, 25, 25)
COM_COLOR = (255, 255, 255)
LEFT_COLOR = (255, 80, 80)
RIGHT_COLOR = (80, 120, 255)
LEFT_FUT_COLOR = (255, 180, 180)
RIGHT_FUT_COLOR = (180, 200, 255)
GRID_COLOR = (50, 50, 50)

FONT_COLOR = (230, 230, 230)


def clamp(x, lo, hi):
    # Torch-friendly clamp for tensors or scalars
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, lo, hi)
    return max(lo, min(hi, x))


def local_to_screen(xy):
    """
    xy: torch.tensor([x, y]) in meters, return (sx, sy) in pixels.
    Coordinate system: +x forward (up on screen), +y left (left on screen)
    """
    if isinstance(xy, torch.Tensor):
        x = float(xy[0].item())
        y = float(xy[1].item())
    else:
        x, y = xy
    # +x forward -> -y screen (up), +y left -> -x screen (left)
    sx = CENTER[0] - y * SCALE  # y left -> left on screen
    sy = CENTER[1] - x * SCALE  # x forward -> up on screen
    return int(sx), int(sy)


class FootTargetPlanner:
    def __init__(
        self,
        debug: bool = False,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
        dt: float = 0.02,
        base_left: tuple[float, float, float] = (0.0, 0.12, 0.0),
        base_right: tuple[float, float, float] = (0.0, -0.12, 0.0),
        single_swing_period: float = 0.5,
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

        # current command: [vx, vy, wz] per batch
        self.cmd = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)

        # histories (2D) for debug-only visualization
        self.left_hist = deque(maxlen=HISTORY_LEN)
        self.right_hist = deque(maxlen=HISTORY_LEN)

        # constants as tensors
        self.single_swing_period = torch.tensor(single_swing_period, dtype=torch.float32, device=self.device)
        
        # Phase clock management (following foot_trajectory.py)
        self.phase_time = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.prev_clock = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.curr_clock = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        # Delay counter for keeping clock at 0 for initial timesteps
        self.delay_counter = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.delay_timesteps = DELAY_TIMESTEPS  # Number of timesteps to stay at 0 before updating
        self.threshold = 1.e-3
        
        # Foot positions: start and target for each foot (for stitching)
        # Initialize at hip offsets (natural standing position)
        self.left_foot_start_pos = self.left_hip_offset.clone()
        self.right_foot_start_pos = self.right_hip_offset.clone()
        
        # Compute initial targets from stance with left as swing
        swing_t, stance_t, swing_yaw, stance_yaw = self._compute_targets_from_stance(
            torch.arange(self.batch_size, device=self.device, dtype=torch.long), which_foot_is_swing="left"
        )
        # Map to left/right
        self.left_foot_target_pos = swing_t
        self.right_foot_target_pos = stance_t
        self.left_foot_target_yaw = swing_yaw
        self.right_foot_target_yaw = stance_yaw
        
        # Swing to stance relative values (position in stance frame, relative yaw)
        # Initialize after computing initial targets
        self.swing_to_stance_pos = torch.zeros(self.batch_size, 3, dtype=torch.float32, device=self.device)
        self.swing_to_stance_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        
        # Compute initial swing to stance values
        self._compute_stance_to_swing_target()

    def _compute_targets_from_stance(self, env_ids: torch.Tensor, which_foot_is_swing: str):
        """
        Compute BOTH swing and stance foot targets from current stance.
        - Computes relative pose from stance to swing: 0.5*(v_xy + wyaw×hip_normal)*T
        - Adds hip offsets for spacing
        - Includes yaw targets: swing_yaw = wyaw*T, stance_yaw = -wyaw*T

        Args:
            env_ids: Long tensor of environment indices to update
            which_foot_is_swing: "left" or "right"

        Returns:
            swing_target (N,3), stance_target (N,3), swing_yaw (N,), stance_yaw (N,)
        """
        if env_ids.numel() == 0:
            # Return empty tensors with correct shapes when nothing to do
            empty_pos = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
            empty_yaw = torch.zeros(0, dtype=torch.float32, device=self.device)
            return empty_pos, empty_pos, empty_yaw, empty_yaw

        vx = self.cmd[env_ids, 0]
        vy = self.cmd[env_ids, 1]
        wz = self.cmd[env_ids, 2]

        T = self.single_swing_period

        if which_foot_is_swing == "left":
            swing_hip = self.left_hip_offset[env_ids]
            stance_hip = self.right_hip_offset[env_ids]
        else:
            swing_hip = self.right_hip_offset[env_ids]
            stance_hip = self.left_hip_offset[env_ids]

        # hip_normal perpendicular to swing hip offset in xy plane
        hip_normal = torch.stack([
            -swing_hip[:, 1],
            swing_hip[:, 0],
        ], dim=1)

        # Components
        xy_component = torch.stack([vx, vy], dim=1)
        yaw_component = wz.unsqueeze(1) * hip_normal

        # Relative step (Raibert) with 0.5 scaling
        step_xy = 0.5 * (xy_component + yaw_component) * T

        # Swing target = step + swing hip offset
        swing_target_xy = step_xy + swing_hip[:, :2]
        # Stance target = -step + stance hip offset (rotational symmetry)
        stance_target_xy = -step_xy + stance_hip[:, :2]

        zeros_n = torch.zeros(step_xy.shape[0], dtype=torch.float32, device=self.device)

        swing_target = torch.cat([swing_target_xy, zeros_n.unsqueeze(-1)], dim=1)
        stance_target = torch.cat([stance_target_xy, zeros_n.unsqueeze(-1)], dim=1)

        swing_yaw = 0.5 * wz * T
        stance_yaw = 0.5 * -wz * T

        return swing_target, stance_target, swing_yaw, stance_yaw

    def is_in_delay(self):
        """Return boolean tensor indicating which envs are in delay period."""
        in_delay = (torch.abs(torch.sin(self.curr_clock * 2 * math.pi)) <= self.threshold) & (self.delay_counter < self.delay_timesteps)
        return in_delay

    def _update_clock(self):
        """Update phase clock, mimicking foot_trajectory.py clock management.

        The clock stays at 1.0 for the first delay_timesteps timesteps before updating.
        When clock is at 1.0, countdown will be 0.
        """
        self.prev_clock = self.curr_clock.clone()

        in_delay = (torch.abs(torch.sin(self.prev_clock * 2 * math.pi)) <= self.threshold) & (self.delay_counter < self.delay_timesteps)
        self.delay_counter = torch.where(in_delay, self.delay_counter + 1, 0)
        can_update = ~in_delay
        self.phase_time = torch.where(can_update, self.phase_time + self.dt, self.phase_time)

        # Normalize to [0, 1) for a full gait cycle (2 * SINGLE_SWING_PERIOD)
        full_cycle = 2.0 * SINGLE_SWING_PERIOD
        self.curr_clock = torch.where(can_update, (self.phase_time % full_cycle) / full_cycle, self.curr_clock)

    def _update_foot_targets(self):
        """
        Detect swing transitions and update foot targets.
        Left foot swings when sin(clock * 2π) > 0 (right is stance)
        Right foot swings when sin(clock * 2π) <= 0 (left is stance)

        New logic: Compute swing foot target, then stance foot is symmetric.
        """
        prev_sin = torch.sin(self.prev_clock * 2 * math.pi)
        curr_sin = torch.sin(self.curr_clock * 2 * math.pi)
        
        # Left swing begins: transition from <= 0 to > 0
        left_swing_begin = torch.logical_and(prev_sin <= 0, curr_sin > 0)
        # Right swing begins: transition from > 0 to <= 0
        right_swing_begin = torch.logical_and(prev_sin > 0, curr_sin <= 0)

        if left_swing_begin.any():
            env_ids = torch.where(left_swing_begin)[0]
            # Set start position as the OLD target (stitching - where we're lifting from)
            self.left_foot_start_pos[env_ids] = self.left_foot_target_pos[env_ids].clone()
            
            # Compute BOTH targets from stance (right stance, left swing)
            swing_t, stance_t, swing_yaw, stance_yaw = self._compute_targets_from_stance(env_ids, which_foot_is_swing="left")
            self.left_foot_target_pos[env_ids] = swing_t
            self.right_foot_target_pos[env_ids] = stance_t
            self.left_foot_target_yaw[env_ids] = swing_yaw
            self.right_foot_target_yaw[env_ids] = stance_yaw
        
        if right_swing_begin.any():
            env_ids = torch.where(right_swing_begin)[0]
            # Set start position as the OLD target (stitching - where we're lifting from)
            self.right_foot_start_pos[env_ids] = self.right_foot_target_pos[env_ids].clone()
            
            # Compute BOTH targets from stance (left stance, right swing)
            swing_t, stance_t, swing_yaw, stance_yaw = self._compute_targets_from_stance(env_ids, which_foot_is_swing="right")
            self.right_foot_target_pos[env_ids] = swing_t
            self.left_foot_target_pos[env_ids] = stance_t
            self.right_foot_target_yaw[env_ids] = swing_yaw
            self.left_foot_target_yaw[env_ids] = stance_yaw
    
    def _compute_future_foot_targets(self):
        """
        Compute future foot targets (one phase ahead with alternated swing/stance).
        
        Returns:
            left_future, right_future: Future target positions for visualization
        """
        # Shift phase by pi to look one step ahead
        future_sin = torch.sin(self.curr_clock * 2 * math.pi + math.pi)

        # Future swing/stance determination
        left_future_swing = future_sin > 0
        right_future_swing = future_sin <= 0

        left_future = torch.zeros_like(self.left_foot_target_pos)
        right_future = torch.zeros_like(self.right_foot_target_pos)
        left_future_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        right_future_yaw = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        # Future: left will swing
        if left_future_swing.any():
            env_ids = torch.where(left_future_swing)[0]
            swing_t, stance_t, swing_yaw, stance_yaw = self._compute_targets_from_stance(env_ids, which_foot_is_swing="left")
            left_future[env_ids] = swing_t
            right_future[env_ids] = stance_t
            left_future_yaw[env_ids] = swing_yaw
            right_future_yaw[env_ids] = stance_yaw

        # Future: right will swing
        if right_future_swing.any():
            env_ids = torch.where(right_future_swing)[0]
            swing_t, stance_t, swing_yaw, stance_yaw = self._compute_targets_from_stance(env_ids, which_foot_is_swing="right")
            right_future[env_ids] = swing_t
            left_future[env_ids] = stance_t
            right_future_yaw[env_ids] = swing_yaw
            left_future_yaw[env_ids] = stance_yaw

        return left_future, right_future, left_future_yaw, right_future_yaw
    
    def _get_current_foot_positions(self):
        """
        Return the current desired foot positions.
        During stance: foot stays at target position.
        During swing: foot is at target position (simplified - could interpolate).
        """
        # For now, just return the target positions
        # (In a full implementation, you'd interpolate during swing phase)
        return self.left_foot_target_pos, self.right_foot_target_pos

    def _compute_stance_to_swing_target(self):
        """
        Compute swing target relative to stance target, accounting for rotation.
        Returns:
            swing_rel_pos (N, 3): Swing target position relative to stance (xyz in stance frame)
            swing_rel_yaw (N,): Swing target yaw relative to stance yaw
        """
        curr_sin = torch.sin(self.curr_clock * 2 * math.pi)
        left_is_swing = curr_sin > 0  # (N,) boolean tensor
        
        # Select swing and stance positions/yaws based on phase
        # swing_mask: True where left is swing, False where right is swing
        swing_abs_pos = torch.where(
            left_is_swing.unsqueeze(-1),
            self.left_foot_target_pos,
            self.right_foot_target_pos
        )  # (N, 3)
        stance_abs_pos = torch.where(
            left_is_swing.unsqueeze(-1),
            self.right_foot_target_pos,
            self.left_foot_target_pos
        )  # (N, 3)
        
        swing_abs_yaw = torch.where(
            left_is_swing,
            self.left_foot_target_yaw,
            self.right_foot_target_yaw
        )  # (N,)
        stance_abs_yaw = torch.where(
            left_is_swing,
            self.right_foot_target_yaw,
            self.left_foot_target_yaw
        )  # (N,)
        
        # Compute relative position in stance frame
        # 1. Get displacement in COM frame
        displacement_xy = swing_abs_pos[:, :2] - stance_abs_pos[:, :2]  # (N, 2)
        displacement_z = swing_abs_pos[:, 2] - stance_abs_pos[:, 2]  # (N,)
        
        # 2. Rotate xy by negative stance yaw to get relative position in stance frame
        # Rotation matrix R(-θ) = [[cos(θ), sin(θ)], [-sin(θ), cos(θ)]]
        cos_stance = torch.cos(-stance_abs_yaw)
        sin_stance = torch.sin(-stance_abs_yaw)
        
        # Apply rotation: R @ displacement_xy
        swing_rel_x = cos_stance * displacement_xy[:, 0] - sin_stance * displacement_xy[:, 1]
        swing_rel_y = sin_stance * displacement_xy[:, 0] + cos_stance * displacement_xy[:, 1]
        # Z doesn't change with yaw rotation (yaw is around z-axis)
        swing_rel_z = displacement_z
        
        swing_rel_pos = torch.stack([swing_rel_x, swing_rel_y, swing_rel_z], dim=1)  # (N, 3)
        
        # 3. Relative yaw
        swing_rel_yaw = swing_abs_yaw - stance_abs_yaw  # (N,)
        
        # Update member variables
        self.swing_to_stance_pos = swing_rel_pos
        self.swing_to_stance_yaw = swing_rel_yaw
        
        return swing_rel_pos, swing_rel_yaw#, swing_abs_pos, stance_abs_pos