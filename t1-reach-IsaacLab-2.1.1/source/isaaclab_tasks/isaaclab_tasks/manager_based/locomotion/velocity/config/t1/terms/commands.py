from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg

import numpy as np
import torch
import h5py
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from pathlib import Path

if TYPE_CHECKING:
    from .commands_cfg import UniformVelocityCommandPlsCfg, PhaseCommandCfg, FKCommandCfg


class UniformVelocityCommandPls(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandPlsCfg

    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandPlsCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading` parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_rotonly_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        temp = r.uniform_(0.0, 1.0)
        self.is_standing_env[env_ids] = temp <= self.cfg.rel_standing_envs
        self.is_rotonly_env[env_ids] = temp <= self.cfg.rel_standing_envs + self.cfg.rel_rotonly_envs

        self.env.command_manager.get_term("phase")._resample_command(env_ids)  # force resample phase command

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        rotonly_env_ids = self.is_rotonly_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        self.vel_command_b[rotonly_env_ids, :2] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # note: parent only deals with callbacks, not marker visibility
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_linvel_viz = VisualizationMarkers(self.cfg.goal_linvel_viz_cfg)
                self.goal_angvel_viz = VisualizationMarkers(self.cfg.goal_angvel_viz_cfg)
                self.curr_linvel_viz = VisualizationMarkers(self.cfg.current_linvel_viz_cfg)
                self.curr_angvel_viz = VisualizationMarkers(self.cfg.current_angvel_viz_cfg)
            self.goal_linvel_viz.set_visibility(True)
            self.goal_angvel_viz.set_visibility(True)
            self.curr_linvel_viz.set_visibility(True)
            self.curr_angvel_viz.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_linvel_viz.set_visibility(False)
                self.goal_angvel_viz.set_visibility(False)
                self.curr_linvel_viz.set_visibility(False)
                self.curr_angvel_viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(self.robot.data.root_ang_vel_b[:, 2])
        self.goal_linvel_viz.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_viz.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_viz.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_viz.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_linvel_viz.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        # obtain default scale of the marker
        default_scale = self.goal_angvel_viz.cfg.markers["arrow"].scale
        # arrow-scale based on angular velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        # arrow direction: point up (+z) for positive angvel (ccw), down (-z) for negative angvel (cw)
        # rotate around y-axis: +90 degrees for +z, -90 degrees for -z
        pitch_angle = torch.where(z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2))
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        # convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class PhaseCommand(CommandTerm):
    """
    A phase command that repeats from [0, 1].
    """

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.step_dt = env.step_dt
        self.phase_dt = torch.ones_like(self.phase, device=self.device) / torch.empty(self.num_envs, device=self.device).uniform_(
            cfg.phase_frequency_hz_range[0], cfg.phase_frequency_hz_range[1]
        )
        self.phase_increment_per_step = torch.full_like(self.phase, self.step_dt) / self.phase_dt
        self.phase_max = torch.ones_like(self.phase)

    def set_phase(self, phase: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Set phase externally."""
        self.phase[env_ids] = phase

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, self.phase_max)  # regulate to [0, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample phase freq and dt
        self.phase_dt = torch.ones_like(self.phase, device=self.device) / torch.empty(self.num_envs, device=self.device).uniform_(
            self.cfg.phase_frequency_hz_range[0], self.cfg.phase_frequency_hz_range[1]
        )
        self.phase_increment_per_step = torch.full_like(self.phase, self.step_dt) / self.phase_dt

    def _update_metrics(self):
        pass  # no metrics

    @property
    def command(self) -> torch.Tensor:
        return self.phase


class FKCommand(CommandTerm):
    """Command generator that provides hand pose targets from FK (Forward Kinematics) data.
    
    This command loads hand pose data from npz files containing a matrix of shape (N, 7, 2),
    where N is the number of hand poses, 7 represents position (3) + quaternion (4), and
    2 represents left hand (index 0) and right hand (index 1).
    
    The command samples two indices independently for left and right hands, similar to
    GMRReferenceCommand. It supports:
    - Default pose tracking (rel_default)
    - Randomly sampled hand poses (rel_random)
    - Independent sampling for left and right hands
    
    The command output is a dictionary containing:
        - 'left_hand_pos': (num_envs, 3) - Left hand position reference
        - 'left_hand_quat': (num_envs, 4) - Left hand quaternion reference (xyzw)
        - 'right_hand_pos': (num_envs, 3) - Right hand position reference
        - 'right_hand_quat': (num_envs, 4) - Right hand quaternion reference (xyzw)
    """
    
    cfg: FKCommandCfg
    
    def __init__(self, cfg: FKCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the FK command generator.
        
        Args:
            cfg: Configuration for the FK command.
            env: The environment instance.
            
        Raises:
            ValueError: If the fk_data_path doesn't exist or contains invalid data.
            FileNotFoundError: If the npz file is not found.
        """
        super().__init__(cfg, env)
        self.cfg = cfg
        self._env = env
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # Default hand poses (prep3 keyframe)
        self.default_left_hand_pos = torch.tensor(
            [0.217952, +0.274974, 0.073840], dtype=torch.float32, device=self.device
        )
        self.default_left_hand_quat = torch.tensor(
            [+0.689605, -0.721816, -0.056384, +0.015724], dtype=torch.float32, device=self.device
        )
        self.default_right_hand_pos = torch.tensor(
            [0.217952, -0.274974, 0.073840], dtype=torch.float32, device=self.device
        )
        self.default_right_hand_quat = torch.tensor(
            [-0.689605, -0.721816, +0.056384, +0.015724], dtype=torch.float32, device=self.device
        )

        self._load_hand_pose_data()

        self.dp_enabled = self.cfg.rel_DProllout > 0.0
        if self.dp_enabled:
            self._load_dp_rollout_data()

        self._initialize_buffers()
        self._initialize_visualization()
    
    def _load_hand_pose_data(self):
        """Load hand pose data from npz file.
        
        The npz file should contain a 'poses' key with a (N, 7, 2) array where:
        - N: number of hand poses
        - 7: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        - 2: [left_hand (index 0), right_hand (index 1)]
        
        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If npz file has invalid format or shape.
        """
        data_path = Path(self.cfg.fk_data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"FK data path does not exist: {data_path}")
        
        if not data_path.is_file() or data_path.suffix != ".npz":
            raise ValueError(f"FK data path must be a .npz file: {data_path}")
        
        data = np.load(data_path, allow_pickle=True)

        if 'poses' not in data:
            available_keys = list(data.files)
            raise ValueError(
                f"NPZ file does not contain 'poses' key. Available keys: {available_keys}. "
                f"Expected a 'poses' key with shape (N, 7, 2)."
            )

        hand_poses = data['poses']

        if len(hand_poses.shape) != 3:
            raise ValueError(
                f"Invalid hand pose data shape: expected 3D array (N, 7, 2), got {len(hand_poses.shape)}D array with shape {hand_poses.shape}."
            )
        
        if hand_poses.shape[1] != 7:
            raise ValueError(
                f"Invalid hand pose data shape: expected (N, 7, 2), got {hand_poses.shape}. "
                f"The second dimension should be 7 [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w], "
                f"but got {hand_poses.shape[1]}."
            )
        
        if hand_poses.shape[2] != 2:
            raise ValueError(
                f"Invalid hand pose data shape: expected (N, 7, 2), got {hand_poses.shape}. "
                f"The third dimension should be 2 [left_hand (index 0), right_hand (index 1)], "
                f"but got {hand_poses.shape[2]}."
            )
        
        self.hand_poses = torch.from_numpy(hand_poses).float().to(self.device)  # (N, 7, 2)
        self.num_poses = hand_poses.shape[0]
        omni.log.info(f"Loaded {self.num_poses} hand poses from {data_path.name}")

    def _load_dp_rollout_data(self):
        """Load DP rollout trajectory data from HDF5 file.

        The HDF5 file should have structure:
            data/demo_X/teleop/Command/hand/left_pose_(Trunk): (N, 7)
            data/demo_X/teleop/Command/hand/right_pose_(Trunk): (N, 7)

        Where each pose is [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].

        The data is at 10Hz, and the policy runs at 50Hz, so we update the command index
        every 5 policy timesteps (step_advance = 5).

        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If HDF5 file has invalid structure.
        """
        data_path = Path(self.cfg.dp_data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"DP rollout data path does not exist: {data_path}")

        if not data_path.is_file() or data_path.suffix not in [".hdf5", ".h5"]:
            raise ValueError(f"DP rollout data path must be a .hdf5 or .h5 file: {data_path}")

        with h5py.File(data_path, "r") as f:
            data_group = f["data"]
            demo_keys = sorted(list(data_group.keys()))

            all_left_poses = []
            all_right_poses = []
            traj_starts = []
            traj_ends = []
            current_idx = 0

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                hand_cmd = demo["teleop"]["Command"]["hand"]

                left_pose = hand_cmd["left_pose_(Trunk)"][:]  # (N, 7)
                right_pose = hand_cmd["right_pose_(Trunk)"][:]  # (N, 7)

                traj_len = left_pose.shape[0]
                traj_starts.append(current_idx)
                traj_ends.append(current_idx + traj_len - 1)
                current_idx += traj_len

                all_left_poses.append(left_pose)
                all_right_poses.append(right_pose)

            all_left_poses = np.concatenate(all_left_poses, axis=0)  # (total_frames, 7)
            all_right_poses = np.concatenate(all_right_poses, axis=0)  # (total_frames, 7)

        self.dp_left_poses = torch.from_numpy(all_left_poses).float().to(self.device)  # (total_frames, 7)
        self.dp_right_poses = torch.from_numpy(all_right_poses).float().to(self.device)  # (total_frames, 7)
        self.dp_traj_starts = torch.tensor(traj_starts, dtype=torch.long, device=self.device)  # (num_trajs,)
        self.dp_traj_ends = torch.tensor(traj_ends, dtype=torch.long, device=self.device)  # (num_trajs,)
        self.dp_num_trajs = len(demo_keys)
        self.dp_total_frames = all_left_poses.shape[0]

        # Data is at 10Hz, policy at 50Hz, so update every 5 steps
        self.dp_step_advance = 5

        omni.log.info(f"Loaded {self.dp_num_trajs} DP rollout trajectories ({self.dp_total_frames} frames) from {data_path.name}")

    def _initialize_buffers(self):
        """Initialize buffers for hand pose tracking."""
        ref_body_cfg = SceneEntityCfg(self.cfg.asset_name, body_names=[self.cfg.reference_body_name])
        ref_body_cfg.resolve(self._env.scene)
        self.reference_body_idx = ref_body_cfg.body_ids[0]

        self.left_hand_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.right_hand_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.is_default = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_fk = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_random = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_dp_rollout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # DP rollout trajectory tracking
        if self.dp_enabled:
            self.dp_traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.dp_global_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.dp_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Arm selection flags (which arm(s) to update this resampling period)
        self.update_left_arm = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.update_right_arm = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Hand pose buffers, initialized with defaults
        self.left_hand_pos = self.default_left_hand_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.left_hand_quat = self.default_left_hand_quat.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.right_hand_pos = self.default_right_hand_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.right_hand_quat = self.default_right_hand_quat.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Random pose buffers (extended workspace beyond FK)
        self.random_left_hand_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.random_left_hand_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.random_right_hand_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.random_right_hand_quat = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def command(self) -> dict[str, torch.Tensor]:
        """The hand pose command as a dictionary of tensors.
        
        Returns:
            Dictionary containing:
                - 'left_hand_pos': (num_envs, 3) - Left hand position reference
                - 'left_hand_quat': (num_envs, 4) - Left hand quaternion reference (xyzw)
                - 'right_hand_pos': (num_envs, 3) - Right hand position reference
                - 'right_hand_quat': (num_envs, 4) - Right hand quaternion reference (xyzw)
        """
        return {
            "left_hand_pos": self.left_hand_pos,
            "left_hand_quat": self.left_hand_quat,
            "right_hand_pos": self.right_hand_pos,
            "right_hand_quat": self.right_hand_quat,
        }
    
    @property
    def default_envs(self) -> torch.Tensor:
        """Boolean tensor indicating which environments are using default poses."""
        return self.is_default
    
    @property
    def random_envs(self) -> torch.Tensor:
        """Boolean tensor indicating which environments are using random poses."""
        return self.is_random

    @property
    def dp_rollout_envs(self) -> torch.Tensor:
        """Boolean tensor indicating which environments are following DP rollout trajectories."""
        return self.is_dp_rollout

    def _update_metrics(self):
        """Update metrics based on current state."""
        pass  # no metrics
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample hand pose indices for specified environments.

        At each resampling (every 2-4 seconds), randomly chooses which arm(s) to update:
        - Left arm only (33% probability)
        - Right arm only (33% probability)
        - Both arms (33% probability)

        The selected arm(s) will be updated during this resampling period, while non-selected
        arm(s) maintain their previous target. This introduces randomness and eliminates
        coupling between the arms.

        For DP rollout environments (rel_DProllout > 0), both arms always follow trajectory data.

        Args:
            env_ids: Environment indices to resample.
        """
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        n = len(env_ids)

        # Same pose index for both hands to maintain correlation
        pose_idx = torch.randint(0, self.num_poses, (n,), device=self.device)
        self.left_hand_idx[env_ids] = pose_idx
        self.right_hand_idx[env_ids] = pose_idx

        # Randomly select which arm(s) to update: 0=left, 1=right, 2=both
        arm_selection = torch.randint(0, 3, (n,), device=self.device)
        self.update_left_arm[env_ids] = (arm_selection == 0) | (arm_selection == 2)
        self.update_right_arm[env_ids] = (arm_selection == 1) | (arm_selection == 2)

        indicator = torch.empty(n, device=self.device).uniform_(0, 1)
        threshold_default = self.cfg.rel_default
        threshold_fk = threshold_default + self.cfg.rel_fk
        threshold_random = threshold_fk + self.cfg.rel_random

        self.is_default[env_ids] = indicator < threshold_default
        self.is_fk[env_ids] = (indicator >= threshold_default) & (indicator < threshold_fk)
        self.is_random[env_ids] = (indicator >= threshold_fk) & (indicator < threshold_random)
        self.is_dp_rollout[env_ids] = (indicator >= threshold_random) & self.dp_enabled

        dp_mask = self.is_dp_rollout[env_ids]
        if dp_mask.any():
            dp_env_ids = env_ids[dp_mask]
            self.update_left_arm[dp_env_ids] = True
            self.update_right_arm[dp_env_ids] = True
            self._resample_dp_trajectories(dp_env_ids)

        random_env_mask = self.is_random[env_ids]
        if random_env_mask.any():
            random_env_ids_local = torch.where(random_env_mask)[0]
            random_env_ids_global = env_ids[random_env_ids_local]
            self._sample_random_poses(random_env_ids_global)

        self._update_hand_poses(env_ids)

    def _resample_dp_trajectories(self, env_ids: torch.Tensor):
        """Resample DP rollout trajectories for specified environments.

        Randomly selects a trajectory for each environment and starts from the beginning.
        The temporal sequence matters for task-oriented trajectories (e.g., box picking).

        Args:
            env_ids: Environment indices to resample trajectories for.
        """
        n = len(env_ids)

        traj_indices = torch.randint(0, self.dp_num_trajs, (n,), device=self.device)
        self.dp_traj_idx[env_ids] = traj_indices

        # Start from beginning (temporal sequence matters)
        traj_starts = self.dp_traj_starts[traj_indices]  # (n,)
        self.dp_global_idx[env_ids] = traj_starts
        self.dp_step_counter[env_ids] = 0
    
    def _sample_random_poses(self, env_ids: torch.Tensor):
        """Sample purely random hand poses in extended workspace.

        These poses extend beyond FK-feasible workspace to train policy robustness.

        Args:
            env_ids: Environment indices to sample random poses for.
        """
        import isaaclab.utils.math as math_utils

        n = len(env_ids)
        ranges = self.cfg.random_pose_ranges

        self.random_left_hand_pos[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*ranges.left_x)
        self.random_left_hand_pos[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*ranges.left_y)
        self.random_left_hand_pos[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*ranges.left_z)

        self.random_right_hand_pos[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*ranges.right_x)
        self.random_right_hand_pos[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*ranges.right_y)
        self.random_right_hand_pos[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*ranges.right_z)

        # Random euler angles -> delta quat applied to default orientation
        left_roll = torch.empty(n, device=self.device).uniform_(*ranges.left_roll)
        left_pitch = torch.empty(n, device=self.device).uniform_(*ranges.left_pitch)
        left_yaw = torch.empty(n, device=self.device).uniform_(*ranges.left_yaw)
        left_quat_delta = math_utils.quat_from_euler_xyz(left_roll, left_pitch, left_yaw)
        self.random_left_hand_quat[env_ids] = math_utils.quat_mul(
            self.default_left_hand_quat.unsqueeze(0).expand(n, -1),
            left_quat_delta
        )

        right_roll = torch.empty(n, device=self.device).uniform_(*ranges.right_roll)
        right_pitch = torch.empty(n, device=self.device).uniform_(*ranges.right_pitch)
        right_yaw = torch.empty(n, device=self.device).uniform_(*ranges.right_yaw)
        right_quat_delta = math_utils.quat_from_euler_xyz(right_roll, right_pitch, right_yaw)
        self.random_right_hand_quat[env_ids] = math_utils.quat_mul(
            self.default_right_hand_quat.unsqueeze(0).expand(n, -1),
            right_quat_delta
        )

    def _update_command(self):
        """Update the hand pose command.

        For DP rollout environments, this advances the trajectory index every dp_step_advance
        (5) policy steps, since the DP data is at 10Hz and policy runs at 50Hz.
        """
        if self.dp_enabled and self.is_dp_rollout.any():
            self._advance_dp_trajectories()
        self._update_hand_poses(slice(None))

    def _advance_dp_trajectories(self):
        """Advance DP trajectory indices for environments following DP rollouts.

        Called every policy step. Increments the step counter and advances the frame
        index every dp_step_advance (5) steps to match 10Hz data rate with 50Hz policy.
        """
        dp_envs = self.is_dp_rollout.nonzero(as_tuple=True)[0]
        if len(dp_envs) == 0:
            return

        self.dp_step_counter[dp_envs] += 1
        advance_mask = self.dp_step_counter[dp_envs] >= self.dp_step_advance
        if not advance_mask.any():
            return

        advance_envs = dp_envs[advance_mask]
        traj_indices = self.dp_traj_idx[advance_envs]
        traj_ends = self.dp_traj_ends[traj_indices]

        new_global_idx = self.dp_global_idx[advance_envs] + 1
        new_global_idx = torch.clamp(new_global_idx, max=traj_ends)
        self.dp_global_idx[advance_envs] = new_global_idx

        self.dp_step_counter[advance_envs] = 0
    
    def _update_hand_poses(self, env_ids: Sequence[int] | slice):
        """Update hand poses using sampled indices.

        Only updates the hand poses for arms that have their update flags set to True.
        This allows independent control of left/right arms during each resampling period.

        Args:
            env_ids: Environment indices to update.
        """
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        left_hand_data = self.hand_poses[self.left_hand_idx[env_ids], :, 0]  # (len(env_ids), 7)
        right_hand_data = self.hand_poses[self.right_hand_idx[env_ids], :, 1]  # (len(env_ids), 7)

        fk_left_hand_pos = left_hand_data[:, :3]
        fk_left_hand_quat = left_hand_data[:, 3:7]
        fk_right_hand_pos = right_hand_data[:, :3]
        fk_right_hand_quat = right_hand_data[:, 3:7]

        # Priority: FK -> override random -> override DP -> override default
        left_hand_pos_new = fk_left_hand_pos
        left_hand_quat_new = fk_left_hand_quat
        right_hand_pos_new = fk_right_hand_pos
        right_hand_quat_new = fk_right_hand_quat

        # Override with random poses for random environments
        left_hand_pos_new = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.random_left_hand_pos[env_ids],
            left_hand_pos_new,
        )
        left_hand_quat_new = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.random_left_hand_quat[env_ids],
            left_hand_quat_new,
        )
        right_hand_pos_new = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.random_right_hand_pos[env_ids],
            right_hand_pos_new,
        )
        right_hand_quat_new = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.random_right_hand_quat[env_ids],
            right_hand_quat_new,
        )

        # Override with DP rollout poses
        if self.dp_enabled:
            dp_mask = self.is_dp_rollout[env_ids]
            if dp_mask.any():
                dp_global_indices = self.dp_global_idx[env_ids]
                dp_left_data = self.dp_left_poses[dp_global_indices]  # (len(env_ids), 7)
                dp_right_data = self.dp_right_poses[dp_global_indices]  # (len(env_ids), 7)

                dp_left_pos = dp_left_data[:, :3]
                dp_left_quat = dp_left_data[:, 3:7]
                dp_right_pos = dp_right_data[:, :3]
                dp_right_quat = dp_right_data[:, 3:7]

                left_hand_pos_new = torch.where(
                    dp_mask.unsqueeze(-1),
                    dp_left_pos,
                    left_hand_pos_new,
                )
                left_hand_quat_new = torch.where(
                    dp_mask.unsqueeze(-1),
                    dp_left_quat,
                    left_hand_quat_new,
                )
                right_hand_pos_new = torch.where(
                    dp_mask.unsqueeze(-1),
                    dp_right_pos,
                    right_hand_pos_new,
                )
                right_hand_quat_new = torch.where(
                    dp_mask.unsqueeze(-1),
                    dp_right_quat,
                    right_hand_quat_new,
                )

        # Override with default poses (highest priority)
        left_hand_pos_new = torch.where(
            self.is_default[env_ids].unsqueeze(-1),
            self.default_left_hand_pos.unsqueeze(0),
            left_hand_pos_new,
        )
        left_hand_quat_new = torch.where(
            self.is_default[env_ids].unsqueeze(-1),
            self.default_left_hand_quat.unsqueeze(0),
            left_hand_quat_new,
        )
        right_hand_pos_new = torch.where(
            self.is_default[env_ids].unsqueeze(-1),
            self.default_right_hand_pos.unsqueeze(0),
            right_hand_pos_new,
        )
        right_hand_quat_new = torch.where(
            self.is_default[env_ids].unsqueeze(-1),
            self.default_right_hand_quat.unsqueeze(0),
            right_hand_quat_new,
        )

        # Write back only for selected arms
        self.left_hand_pos[env_ids] = torch.where(
            self.update_left_arm[env_ids].unsqueeze(-1),
            left_hand_pos_new,
            self.left_hand_pos[env_ids],
        )
        self.left_hand_quat[env_ids] = torch.where(
            self.update_left_arm[env_ids].unsqueeze(-1),
            left_hand_quat_new,
            self.left_hand_quat[env_ids],
        )
        self.right_hand_pos[env_ids] = torch.where(
            self.update_right_arm[env_ids].unsqueeze(-1),
            right_hand_pos_new,
            self.right_hand_pos[env_ids],
        )
        self.right_hand_quat[env_ids] = torch.where(
            self.update_right_arm[env_ids].unsqueeze(-1),
            right_hand_quat_new,
            self.right_hand_quat[env_ids],
        )

        if self.cfg.enable_workspace_limits:
            self._clamp_to_workspace(env_ids)
        if self.cfg.enable_orientation_limits:
            self._clamp_orientation(env_ids)

    def _clamp_to_workspace(self, env_ids: torch.Tensor):
        """Clamp hand positions to workspace limits.

        Only clamps FK-sampled and default poses, NOT random poses.
        Random poses use the larger random_pose_ranges and should not be constrained.

        Args:
            env_ids: Environment indices to clamp.
        """
        limits = self.cfg.workspace_limits

        # Skip random pose envs (they use larger random_pose_ranges)
        non_random_mask = ~self.is_random[env_ids]
        non_random_env_ids = env_ids[non_random_mask]

        if len(non_random_env_ids) == 0:
            return

        left_update_mask = self.update_left_arm[non_random_env_ids]
        if left_update_mask.any():
            left_env_ids = non_random_env_ids[left_update_mask]
            self.left_hand_pos[left_env_ids, 0] = torch.clamp(
                self.left_hand_pos[left_env_ids, 0], limits.left_x[0], limits.left_x[1]
            )
            self.left_hand_pos[left_env_ids, 1] = torch.clamp(
                self.left_hand_pos[left_env_ids, 1], limits.left_y[0], limits.left_y[1]
            )
            self.left_hand_pos[left_env_ids, 2] = torch.clamp(
                self.left_hand_pos[left_env_ids, 2], limits.left_z[0], limits.left_z[1]
            )

        right_update_mask = self.update_right_arm[non_random_env_ids]
        if right_update_mask.any():
            right_env_ids = non_random_env_ids[right_update_mask]

            self.right_hand_pos[right_env_ids, 0] = torch.clamp(
                self.right_hand_pos[right_env_ids, 0], limits.right_x[0], limits.right_x[1]
            )
            self.right_hand_pos[right_env_ids, 1] = torch.clamp(
                self.right_hand_pos[right_env_ids, 1], limits.right_y[0], limits.right_y[1]
            )
            self.right_hand_pos[right_env_ids, 2] = torch.clamp(
                self.right_hand_pos[right_env_ids, 2], limits.right_z[0], limits.right_z[1]
            )

    def _clamp_orientation(self, env_ids: torch.Tensor):
        """Clamp hand orientations to be within max deviation from default.

        Only clamps the arms that are selected for update during this resampling period.
        If orientation exceeds limit, it is reset to default orientation.

        Args:
            env_ids: Environment indices to clamp.
        """
        import isaaclab.utils.math as math_utils

        max_dev_rad = self.cfg.max_orientation_deviation_deg * torch.pi / 180.0

        left_update_mask = self.update_left_arm[env_ids]
        if left_update_mask.any():
            left_env_ids = env_ids[left_update_mask]
            num_left = len(left_env_ids)
            default_left_expanded = self.default_left_hand_quat.unsqueeze(0).expand(num_left, -1)
            left_quat_error = math_utils.quat_mul(
                math_utils.quat_conjugate(default_left_expanded),
                self.left_hand_quat[left_env_ids]
            )
            left_angle = 2.0 * torch.acos(torch.clamp(torch.abs(left_quat_error[:, 0]), -1.0, 1.0))
            left_exceed = left_angle > max_dev_rad
            if left_exceed.any():
                exceed_ids = left_env_ids[left_exceed]
                self.left_hand_quat[exceed_ids] = self.default_left_hand_quat.unsqueeze(0).expand(len(exceed_ids), -1)

        right_update_mask = self.update_right_arm[env_ids]
        if right_update_mask.any():
            right_env_ids = env_ids[right_update_mask]
            num_right = len(right_env_ids)
            default_right_expanded = self.default_right_hand_quat.unsqueeze(0).expand(num_right, -1)
            right_quat_error = math_utils.quat_mul(
                math_utils.quat_conjugate(default_right_expanded),
                self.right_hand_quat[right_env_ids]
            )
            right_angle = 2.0 * torch.acos(torch.clamp(torch.abs(right_quat_error[:, 0]), -1.0, 1.0))
            right_exceed = right_angle > max_dev_rad
            if right_exceed.any():
                exceed_ids = right_env_ids[right_exceed]
                self.right_hand_quat[exceed_ids] = self.default_right_hand_quat.unsqueeze(0).expand(len(exceed_ids), -1)
    
    def _initialize_visualization(self):
        self.robot_frame_indices = []
        if self.cfg.viz_robot_frames:
            asset_cfg = SceneEntityCfg(self.cfg.asset_name, body_names=self.cfg.viz_robot_frames)
            asset_cfg.resolve(self._env.scene)
            self.robot_frame_indices = asset_cfg.body_ids
        
        if self.cfg.debug_vis:
            fk_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FKHandFrames")
            fk_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.fk_frame_markers = VisualizationMarkers(fk_marker_cfg)

            if len(self.robot_frame_indices) > 0:
                robot_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RobotFrames")
                robot_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
                self.robot_frame_markers = VisualizationMarkers(robot_marker_cfg)

            fk_viz_cfg = self.cfg.fk_viz_cfg.replace()
            fk_viz_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # green: FK envs
            self.fk_viz = VisualizationMarkers(fk_viz_cfg)

            random_viz_cfg = self.cfg.random_viz_cfg.replace()
            random_viz_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)  # red: random envs
            self.random_viz = VisualizationMarkers(random_viz_cfg)

            dp_rollout_viz_cfg = self.cfg.dp_rollout_viz_cfg.replace()
            dp_rollout_viz_cfg.markers["sphere"].visual_material.diffuse_color = (0.6, 0.0, 0.8)  # purple: DP rollout envs
            self.dp_rollout_viz = VisualizationMarkers(dp_rollout_viz_cfg)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for FK hand poses and robot frames."""
        if debug_vis:
            if hasattr(self, "fk_frame_markers"):
                self.fk_frame_markers.set_visibility(True)
            if hasattr(self, "robot_frame_markers"):
                self.robot_frame_markers.set_visibility(True)
            if hasattr(self, "fk_viz"):
                self.fk_viz.set_visibility(True)
            if hasattr(self, "random_viz"):
                self.random_viz.set_visibility(True)
            if hasattr(self, "dp_rollout_viz"):
                self.dp_rollout_viz.set_visibility(True)
        else:
            if hasattr(self, "fk_frame_markers"):
                self.fk_frame_markers.set_visibility(False)
            if hasattr(self, "robot_frame_markers"):
                self.robot_frame_markers.set_visibility(False)
            if hasattr(self, "fk_viz"):
                self.fk_viz.set_visibility(False)
            if hasattr(self, "random_viz"):
                self.random_viz.set_visibility(False)
            if hasattr(self, "dp_rollout_viz"):
                self.dp_rollout_viz.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if hasattr(self, "fk_frame_markers"):
            ref_pos_w = self.robot.data.body_pos_w[:, self.reference_body_idx, :]
            ref_rot_w = self.robot.data.body_quat_w[:, self.reference_body_idx, :]
            left_hand_pos_w, left_hand_quat_w = math_utils.combine_frame_transforms(
                ref_pos_w, ref_rot_w, self.left_hand_pos, self.left_hand_quat
            )
            right_hand_pos_w, right_hand_quat_w = math_utils.combine_frame_transforms(
                ref_pos_w, ref_rot_w, self.right_hand_pos, self.right_hand_quat
            )
            fk_pos_w = torch.cat([left_hand_pos_w, right_hand_pos_w], dim=0)
            fk_quat_w = torch.cat([left_hand_quat_w, right_hand_quat_w], dim=0)
            self.fk_frame_markers.visualize(fk_pos_w, fk_quat_w)

        if hasattr(self, "robot_frame_markers") and len(self.robot_frame_indices) > 0:
            robot_positions = self.robot.data.body_pos_w[:, self.robot_frame_indices]
            robot_orientations = self.robot.data.body_quat_w[:, self.robot_frame_indices]
            self.robot_frame_markers.visualize(robot_positions.reshape(-1, 3), robot_orientations.reshape(-1, 4))

        _hide = torch.tensor([0.0, 0.0, -100.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        if hasattr(self, "fk_viz"):
            robot_pos_w = self.robot.data.root_pos_w.clone()
            robot_pos_w[:, 2] += 0.6
            self.fk_viz.visualize(
                torch.where(self.is_fk.unsqueeze(-1), robot_pos_w, _hide),
                self.robot.data.root_quat_w,
            )

        if hasattr(self, "random_viz"):
            robot_pos_w = self.robot.data.root_pos_w.clone()
            robot_pos_w[:, 2] += 0.6
            self.random_viz.visualize(
                torch.where(self.is_random.unsqueeze(-1), robot_pos_w, _hide),
                self.robot.data.root_quat_w,
            )

        if hasattr(self, "dp_rollout_viz"):
            robot_pos_w = self.robot.data.root_pos_w.clone()
            robot_pos_w[:, 2] += 0.6
            self.dp_rollout_viz.visualize(
                torch.where(self.is_dp_rollout.unsqueeze(-1), robot_pos_w, _hide),
                self.robot.data.root_quat_w,
            )


