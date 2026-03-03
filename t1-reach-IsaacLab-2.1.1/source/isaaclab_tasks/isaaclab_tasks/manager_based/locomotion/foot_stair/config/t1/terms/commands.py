from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTermCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from pathlib import Path
from isaaclab_tasks.manager_based.manipulation.reach.config.t1.mdp.observation import pose_7d_to_9d
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_euler_xyz,
    quat_unique,
    quat_apply_inverse,
    euler_xyz_from_quat,
)
import h5py
from .foot_target_planner import FootTargetPlanner

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import FootPosCommandCfg, FootTrackCommandCfg, VelocityToFootTrackCommandCfg



class FootPosCommand(CommandTerm):
    cfg: FootPosCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: FootPosCommandCfg, env: ManagerBasedEnv):
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
        path = Path(self.cfg.path_dataset)
        if not path.is_file() or path.suffix not in [".npz", ".h5"]:
            raise FileNotFoundError(f"Invalid dataset! Should be a .npz or .h5 file. The one set in config is {self.cfg.path_dataset}.")

        # obtain the robot asset
        # -- robot and bodies
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_ids = self.robot.find_bodies(cfg.body_names)[0]
        self.joint_ids = self.robot.find_joints(
            [
                # "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
            preserve_order=True,
        )[0]

        # lUse h5 dataset
        with h5py.File(self.cfg.path_dataset, "r") as dataset:
            self.starting_steps = torch.from_numpy(dataset["starting_steps"][:]).long().to(self.device)  # (n,)
            self.ending_steps = torch.from_numpy(dataset["ending_steps"][:]).long().to(self.device)  # (n,)
            # Get average dt from the dts array
            dts_array = dataset["dts"][:]
            self.traj_dt = torch.tensor(float(np.mean(dts_array[dts_array > 0])), dtype=torch.float32, device=self.device)

            self.data_foot_pos = torch.from_numpy(dataset["foot_pos_cmd"][:]).float().to(self.device)
            self.data_foot_indicator = torch.from_numpy(dataset["foot_indicator_cmd"][:]).float().to(self.device)
            self.data_left_foot = torch.from_numpy(dataset["left_foot_positions"][:]).float().to(self.device)  # (k, 7)
            self.data_right_foot = torch.from_numpy(dataset["right_foot_positions"][:]).float().to(self.device)  # (k, 7)
            self.data_countdown = torch.from_numpy(dataset["countdown_cmd"][:]).float().to(self.device)  # (k, 1)
            self.data_qref = torch.from_numpy(dataset["qs"][:]).float().to(self.device)  # (k, dim_qref)
            self.data_dqref = torch.from_numpy(dataset["dqs"][:]).float().to(self.device)  # (k, dim_qref)

        # compute trajectory properties
        self.traj_num = len(self.starting_steps)
        self.traj_lengths = self.ending_steps - self.starting_steps + 1  # inclusive of ending step

        # step advance based on dt ratio
        self.step_advance = max(1, int(self.traj_dt / self._env.step_dt))  # at least need to step 1 step for advance
        self.step_counter = torch.zeros(self.num_envs, device=self.device)

        # create buffers to store the command
        self.foot_pos_command = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BLF = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BRF = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BLF_random = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BRF_random = torch.zeros(self.num_envs, 7, device=self.device)
        self.countdown = torch.zeros(self.num_envs, 1, device=self.device)
        self.foot_indicator = torch.zeros(self.num_envs, 2, device=self.device)
        self.qref_indices = torch.tensor(self.cfg.qref_indices, device=self.device)
        self.qref = torch.zeros(self.num_envs, len(self.cfg.qref_indices), device=self.device)
        self.dqref = torch.zeros_like(self.qref)
        self.is_random = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # trajectory tracking
        self.traj_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.traj_step_offset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # offset within trajectory
        self.global_step_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # index into data arrays

        # -- metrics
        self.metrics["error_EE_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_EE_ori"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "FootPosCommand:\n"
        msg += f"\tNum Trajs: {self.traj_num}\n"
        msg += f"\tMax Traj Length: {self.traj_lengths.max().item()}\n"
        msg += f"\tDt: {self.traj_dt}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.concatenate([self.xyzwxyz_BLF, self.xyzwxyz_BRF, self.qref], dim=1)

    @property
    def left_foot_pose_command(self) -> torch.Tensor:
        return self.xyzwxyz_BLF

    @property
    def right_foot_pose_command(self) -> torch.Tensor:
        return self.xyzwxyz_BRF

    @property
    def joint_command(self) -> torch.Tensor:
        return self.qref

    @property
    def joint_vel_command(self) -> torch.Tensor:
        return self.dqref

    @property
    def countdown_command(self) -> torch.Tensor:
        return self.countdown

    @property
    def foot_indicator_command(self) -> torch.Tensor:
        return self.foot_indicator

    @property
    def random_envs(self) -> torch.Tensor:
        return self.is_random

    @property
    def left_foot_position_current(self) -> torch.Tensor:
        """Returns left foot pose (position + quaternion) in world frame."""
        pos = self.robot.data.body_pos_w[:, self.body_ids[0]]
        quat = self.robot.data.body_quat_w[:, self.body_ids[0]]
        return torch.cat([pos, quat], dim=-1)  # (num_envs, 7)

    @property
    def right_foot_position_current(self) -> torch.Tensor:
        """Returns right foot pose (position + quaternion) in world frame."""
        pos = self.robot.data.body_pos_w[:, self.body_ids[1]]
        quat = self.robot.data.body_quat_w[:, self.body_ids[1]]
        return torch.cat([pos, quat], dim=-1)  # (num_envs, 7)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """
        Compute tracking errors for end-effectors.
        """
        self._env: ManagerBasedRLEnv
        max_command_step = self._env.max_episode_length

        # get current poses
        pos_curr = self.robot.data.body_pos_w[:, self.body_ids]  # (num_envs, 2, 3)
        quat_curr = self.robot.data.body_quat_w[:, self.body_ids]  # (num_envs, 2, 4)

        # get target poses
        pos_target_lf, quat_target_lf = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.xyzwxyz_BLF[:, :3],
            self.xyzwxyz_BLF[:, 3:],
        )
        pos_target_rf, quat_target_rf = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.xyzwxyz_BRF[:, :3],
            self.xyzwxyz_BRF[:, 3:],
        )

        # compute errors for each hand separately
        pos_error_lf, rot_error_lf = math_utils.compute_pose_error(
            pos_target_lf,
            quat_target_lf,
            pos_curr[:, 0, :],
            quat_curr[:, 0, :],
        )

        pos_error_rf, rot_error_rf = math_utils.compute_pose_error(
            pos_target_rf,
            quat_target_rf,
            pos_curr[:, 1, :],
            quat_curr[:, 1, :],
        )

        # accumulate errors
        self.metrics["error_EE_pos"] = (torch.norm(pos_error_lf, dim=-1) + torch.norm(pos_error_rf, dim=-1)) / max_command_step
        self.metrics["error_EE_ori"] = (torch.norm(rot_error_lf, dim=-1) + torch.norm(rot_error_rf, dim=-1)) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)

        # randomly sample trajectories
        traj_indices = torch.randint(0, self.traj_num, (n,), device=self.device)
        self.traj_idx[env_ids] = traj_indices

        # compute starting offset within each trajectory based on resample ratio
        ratios = torch.empty(n, device=self.device).uniform_(self.cfg.resample_ratio_range[0], self.cfg.resample_ratio_range[1])

        # get trajectory lengths for selected trajectories
        selected_traj_lengths = self.traj_lengths[traj_indices]

        # compute offset within trajectory (0 to traj_length-1)
        traj_offsets = (ratios * selected_traj_lengths.float()).long()
        traj_offsets = torch.clamp(traj_offsets, torch.zeros_like(traj_offsets), selected_traj_lengths - 1)

        # store offset and compute global index
        self.traj_step_offset[env_ids] = traj_offsets
        self.global_step_idx[env_ids] = self.starting_steps[traj_indices] + traj_offsets

        # reset step counter
        self.step_counter[env_ids] = 0

        # determine random env
        r = torch.empty(n, device=self.device)
        self.is_random[env_ids] = r.uniform_(0, 1) < self.cfg.rel_random

        # sample random pose
        # -- translation
        self.xyzwxyz_BLF_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_lf.pos_x)
        self.xyzwxyz_BLF_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_lf.pos_y)
        self.xyzwxyz_BLF_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_lf.pos_z)
        self.xyzwxyz_BRF_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_rf.pos_x)
        self.xyzwxyz_BRF_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_rf.pos_y)
        self.xyzwxyz_BRF_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_rf.pos_z)
        # -- orientation
        euler_angles_lf = torch.zeros_like(self.xyzwxyz_BLF_random[env_ids, :3])
        euler_angles_lf[:, 0].uniform_(*self.cfg.ranges_lf.roll)
        euler_angles_lf[:, 1].uniform_(*self.cfg.ranges_lf.pitch)
        euler_angles_lf[:, 2].uniform_(*self.cfg.ranges_lf.yaw)
        quat_lf = quat_from_euler_xyz(euler_angles_lf[:, 0], euler_angles_lf[:, 1], euler_angles_lf[:, 2])
        quat_lf = math_utils.quat_unique(quat_lf)  # Ensure w >= 0
        self.xyzwxyz_BLF_random[env_ids, 3:] = quat_lf
        euler_angles_rf = torch.zeros_like(self.xyzwxyz_BRF_random[env_ids, :3])
        euler_angles_rf[:, 0].uniform_(*self.cfg.ranges_rf.roll)
        euler_angles_rf[:, 1].uniform_(*self.cfg.ranges_rf.pitch)
        euler_angles_rf[:, 2].uniform_(*self.cfg.ranges_rf.yaw)
        quat_rf = quat_from_euler_xyz(euler_angles_rf[:, 0], euler_angles_rf[:, 1], euler_angles_rf[:, 2])
        quat_rf = math_utils.quat_unique(quat_rf)  # Ensure w >= 0
        self.xyzwxyz_BRF_random[env_ids, 3:] = quat_rf

        # on resample, also reset the joint pos to the staring qref
        self.xyzwxyz_BLF[env_ids] = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.xyzwxyz_BLF_random[env_ids],
            self.data_left_foot[self.global_step_idx[env_ids]],
        )
        self.xyzwxyz_BRF[env_ids] = torch.where(
            self.is_random[env_ids].unsqueeze(-1), self.xyzwxyz_BRF_random[env_ids], self.data_right_foot[self.global_step_idx[env_ids]]
        )
        self.qref[env_ids] = self.data_qref[self.global_step_idx[env_ids]][:, self.qref_indices]
        self.dqref[env_ids] = self.data_dqref[self.global_step_idx[env_ids]][:, self.qref_indices]
        self.foot_pos_command[env_ids] = self.data_foot_pos[self.global_step_idx[env_ids]]

    def _update_command(self):
        # advance steps
        self.step_counter += 1
        advance_envs = self.step_counter >= self.step_advance

        if torch.any(advance_envs):
            # increment offset for environments that need to advance
            self.traj_step_offset[advance_envs] += 1

            # clamp to trajectory limits per environment
            max_offsets = self.traj_lengths[self.traj_idx] - 1
            self.traj_step_offset = torch.clamp(self.traj_step_offset, torch.zeros_like(self.traj_step_offset), max_offsets)

            # update global indices
            self.global_step_idx = self.starting_steps[self.traj_idx] + self.traj_step_offset

            # reset step counter for advanced envs
            self.step_counter[advance_envs] = 0

        # update commands using global indices
        # for random envs, keep the command constant
        self.xyzwxyz_BLF = torch.where(self.is_random.unsqueeze(-1), self.xyzwxyz_BLF, self.data_left_foot[self.global_step_idx])
        self.xyzwxyz_BRF = torch.where(self.is_random.unsqueeze(-1), self.xyzwxyz_BRF, self.data_right_foot[self.global_step_idx])
        self.qref = self.data_qref[self.global_step_idx, :][:, self.qref_indices]
        self.dqref = self.data_dqref[self.global_step_idx, :][:, self.qref_indices]
        self.foot_indicator = self.data_foot_indicator[self.traj_idx, :]
        self.countdown = self.data_countdown[self.global_step_idx, :]
        self.foot_pos_command = self.data_foot_pos[self.global_step_idx, :]

        # TEST: directly set to qref, only to see if the command / qref makes sense
        # print("THIS SHOULD BE OFF IF YOU ARE NOT TESTING!!!")
        # self.robot.set_joint_position_target(self.qref, self.joint_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "lf_target_viz"):
                self.lf_target_viz = VisualizationMarkers(self.cfg.EE_target_viz_cfg)
                self.lf_current_viz = VisualizationMarkers(self.cfg.EE_curr_viz_cfg)
                self.rf_target_viz = VisualizationMarkers(self.cfg.EE_target_viz_cfg)
                self.rf_current_viz = VisualizationMarkers(self.cfg.EE_curr_viz_cfg)
                self.random_viz = VisualizationMarkers(self.cfg.random_viz_cfg)
            self.lf_target_viz.set_visibility(True)
            self.lf_current_viz.set_visibility(True)
            self.rf_target_viz.set_visibility(True)
            self.rf_current_viz.set_visibility(True)
            self.random_viz.set_visibility(True)
        else:
            if hasattr(self, "lf_target_viz"):
                self.lf_target_viz.set_visibility(False)
                self.lf_current_viz.set_visibility(False)
                self.rf_target_viz.set_visibility(False)
                self.rf_current_viz.set_visibility(False)
                self.random_viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "lf_target_viz"):
            # Get robot base pose
            robot_pos_w = self.robot.data.root_pos_w  # (num_envs, 3)
            robot_quat_w = self.robot.data.root_quat_w  # (num_envs, 4)

            # Get offseted commands
            lf_pose_target = self.left_foot_pose_command
            rf_pose_target = self.right_foot_pose_command

            # Convert target poses from body frame to world frame
            # For left hand (xyzwxyz format: position first, then quaternion)
            lf_pos_w = math_utils.quat_apply(robot_quat_w, lf_pose_target[:, :3]) + robot_pos_w
            lf_quat_w = math_utils.quat_mul(robot_quat_w, lf_pose_target[:, 3:7])

            # For right hand
            rf_pos_w = math_utils.quat_apply(robot_quat_w, rf_pose_target[:, :3]) + robot_pos_w
            rf_quat_w = math_utils.quat_mul(robot_quat_w, rf_pose_target[:, 3:7])

            self.lf_target_viz.visualize(lf_pos_w, lf_quat_w)
            self.rf_target_viz.visualize(rf_pos_w, rf_quat_w)

            # Visualize left hand target/current
            self.lf_current_viz.visualize(self.robot.data.body_pos_w[:, self.body_ids[0]], self.robot.data.body_quat_w[:, self.body_ids[0]])

            # Visualize right hand target/current
            self.rf_current_viz.visualize(self.robot.data.body_pos_w[:, self.body_ids[1]], self.robot.data.body_quat_w[:, self.body_ids[1]])

            # Visualize random
            random_marker_pos = robot_pos_w.clone()
            random_marker_pos[:, 2] = 1.2


class FootTrackCommand(CommandTerm):
    """Command generator that provides reference foot tracking data from NPZ files.

    This command loads footstep tracking data from NPZ files and provides reference trajectories
    for foot tracking tasks. The data includes joint positions/velocities, foot transforms,
    and footstep commands.

    The NPZ file should contain:
    - q: (n, nq) - joint positions with base
    - qd: (n, nv) - joint velocities
    - T_blf: (n, 4) - body frame to left foot frame transform (x, y, z, yaw)
    - T_brf: (n, 4) - body frame to right foot frame transform (x, y, z, yaw)
    - T_stsw: (n, 4) - stance foot to swing foot transform (x, y, z, yaw)
    - p_wcom: (n, 3) - CoM position in world frame
    - T_wbase: (n, 7) - base transform in world frame (x, y, z, qw, qx, qy, qz)
    - v_b: (n, 6) - base velocity in base frame (linear xyz, angular xyz)
    - cmd_footstep: (n, 4) - [x, y, z, yaw] in stance foot frame
    - cmd_stance: (n, 1) - 0=left stance, 1=right stance
    - cmd_countdown: (n, 1) - countdown timer: 0 during wait, 0->1->0 during step
    - traj: (k,) - starting indices of each trajectory
    - traj_dt: float - time step between frames

    The command output provides:
        - q: (num_envs, nq) - joint positions
        - qd: (num_envs, nv) - joint velocities
        - T_blf: (num_envs, 4) - body to left foot transform
        - T_brf: (num_envs, 4) - body to right foot transform
        - T_stsw: (num_envs, 4) - stance to swing foot transform
        - p_wcom: (num_envs, 3) - CoM position in world frame
        - T_wbase: (num_envs, 7) - base transform in world frame
        - v_b: (num_envs, 6) - base velocity in base frame
        - cmd_footstep: (num_envs, 4) - footstep command
        - cmd_stance: (num_envs, 1) - stance foot indicator
        - cmd_countdown: (num_envs, 1) - countdown timer
    """

    cfg: FootTrackCommandCfg

    def __init__(self, cfg: FootTrackCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the foot track command generator.

        Args:
            cfg: Configuration for the foot track command.
            env: The environment instance.

        Raises:
            ValueError: If the foot_track_data_path doesn't exist or contains invalid data.
            FileNotFoundError: If no valid .npz file is found.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Store config and env
        self.cfg = cfg
        self._env = env

        # Load foot tracking data from NPZ file
        self._load_foot_track_data()
        self._initialize_buffers()
        self._initialize_visualization()

        # Load important indexes
        # Resolve foot body indices if configured
        self.left_foot_idx = None
        self.right_foot_idx = None

        if hasattr(self.cfg, "left_foot_name") and self.cfg.left_foot_name:
            asset_cfg = SceneEntityCfg(self.cfg.asset_name, body_names=[self.cfg.left_foot_name])
            asset_cfg.resolve(self._env.scene)
            self.left_foot_idx = asset_cfg.body_ids[0]

        if hasattr(self.cfg, "right_foot_name") and self.cfg.right_foot_name:
            asset_cfg = SceneEntityCfg(self.cfg.asset_name, body_names=[self.cfg.right_foot_name])
            asset_cfg.resolve(self._env.scene)
            self.right_foot_idx = asset_cfg.body_ids[0]

        # Log information
        import omni.log

        omni.log.info(f"FootTrackCommand initialized with {self.traj_num} trajectories")
        omni.log.info(f"  Joint dimensions - q: {self.num_q}, qd: {self.num_qd}")
        omni.log.info(f"  Total frames loaded: {self.data_q.shape[0]}")

    def _load_foot_track_data(self):
        """Load foot tracking data from NPZ file.

        Raises:
            FileNotFoundError: If path doesn't exist or is not a .npz file.
            ValueError: If NPZ file has invalid format.
        """
        import omni.log

        data_path = Path(self.cfg.foot_track_data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Foot track data path does not exist: {data_path}")

        if not data_path.is_file() or data_path.suffix != ".npz":
            raise ValueError(f"Invalid path (must be .npz file): {data_path}")

        omni.log.info(f"Loading foot tracking data from {data_path}")

        # Load NPZ file
        data = np.load(data_path, allow_pickle=True)

        required_keys = [
            "q",
            "qd",
            "T_blf",
            "T_brf",
            "T_stsw",
            "p_wcom",
            "T_wbase",
            "v_b",
            "cmd_footstep",
            "cmd_stance",
            "cmd_countdown",
            "traj",
            "traj_dt",
        ]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"NPZ file missing keys: {missing_keys}")

        # Extract trajectory metadata
        traj_indices = data["traj"]  # Starting indices of each trajectory
        self.traj_num = len(traj_indices)
        self.traj_dt = float(data["traj_dt"])

        # Compute trajectory start and end indices
        starting_steps = []
        ending_steps = []

        for i in range(len(traj_indices)):
            start_idx = int(traj_indices[i])
            if i < len(traj_indices) - 1:
                end_idx = int(traj_indices[i + 1]) - 1
            else:
                end_idx = data["q"].shape[0] - 1

            starting_steps.append(start_idx)
            ending_steps.append(end_idx)

        self.starting_steps = torch.tensor(starting_steps, dtype=torch.long, device=self.device)
        self.ending_steps = torch.tensor(ending_steps, dtype=torch.long, device=self.device)
        self.traj_lengths = self.ending_steps - self.starting_steps + 1  # inclusive

        # Infer dimensions
        self.num_q = data["q"].shape[1]
        self.num_qd = data["qd"].shape[1]
        num_frames = data["q"].shape[0]

        # Validate shapes
        expected_shapes = {
            "q": (num_frames, self.num_q),
            "qd": (num_frames, self.num_qd),
            "T_blf": (num_frames, 4),
            "T_brf": (num_frames, 4),
            "T_stsw": (num_frames, 4),
            "p_wcom": (num_frames, 3),
            "T_wbase": (num_frames, 7),
            "v_b": (num_frames, 6),
            "cmd_footstep": (num_frames, 4),
            "cmd_stance": (num_frames, 1),
            "cmd_countdown": (num_frames, 1),
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = data[key].shape
            if actual_shape != expected_shape:
                raise ValueError(f"Invalid shape for '{key}': expected {expected_shape}, got {actual_shape}")

        # Load all data tensors and move to device
        self.data_q = torch.from_numpy(data["q"]).float().to(self.device)
        self.data_qd = torch.from_numpy(data["qd"]).float().to(self.device)
        self.data_T_blf = torch.from_numpy(data["T_blf"]).float().to(self.device)
        self.data_T_brf = torch.from_numpy(data["T_brf"]).float().to(self.device)
        self.data_T_stsw = torch.from_numpy(data["T_stsw"]).float().to(self.device)
        self.data_p_wcom = torch.from_numpy(data["p_wcom"]).float().to(self.device)
        self.data_T_wbase = torch.from_numpy(data["T_wbase"]).float().to(self.device)
        self.data_v_b = torch.from_numpy(data["v_b"]).float().to(self.device)
        self.data_cmd_footstep = torch.from_numpy(data["cmd_footstep"]).float().to(self.device)
        self.data_cmd_stance = torch.from_numpy(data["cmd_stance"]).float().to(self.device)
        self.data_cmd_countdown = torch.from_numpy(data["cmd_countdown"]).float().to(self.device)

        omni.log.info("Successfully loaded foot tracking data")
        omni.log.info(f"Total trajectories: {self.traj_num}")
        omni.log.info(f"Total frames: {num_frames}")

    def _initialize_buffers(self):
        """Initialize buffers for foot tracking."""
        # Trajectory tracking
        self.traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_step_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.global_step_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Step counter for frame advancement
        self.step_advance = max(1, int(self.traj_dt / self._env.step_dt))
        self.step_counter = torch.zeros(self.num_envs, device=self.device)

        # Reference state buffers (command outputs)
        self.q_ref = torch.zeros(self.num_envs, self.num_q, device=self.device)
        self.qd_ref = torch.zeros(self.num_envs, self.num_qd, device=self.device)
        self.T_blf_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_brf_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_stsw_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.p_wcom_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self.T_wbase_ref = torch.zeros(self.num_envs, 7, device=self.device)
        self.v_b_ref = torch.zeros(self.num_envs, 6, device=self.device)
        self.cmd_footstep_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.cmd_stance_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.cmd_countdown_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_cmd_countdown_ref = torch.zeros(self.num_envs, 1, device=self.device)

        self.T_wsw_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_wst_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.stance_id_aux = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.swing_id_aux = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Store left and right foot world frame pose when stepping starts
        self.left_foot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_foot_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.right_foot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_foot_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # Random pose buffers
        self.T_blf_random = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_brf_random = torch.zeros(self.num_envs, 4, device=self.device)

        # Metrics
        self.metrics["foot_track_time"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return string representation of the command generator."""
        msg = "FootTrackCommand:\n"
        msg += f"\tNumber of trajectories: {self.traj_num}\n"
        msg += f"\tJoint dimensions - q: {self.num_q}, qd: {self.num_qd}\n"
        msg += f"\tMax trajectory length: {self.traj_lengths.max().item()}\n"
        msg += f"\tDt: {self.traj_dt}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> dict[str, torch.Tensor]:
        """The foot tracking command as a dictionary of tensors.

        Returns:
            Dictionary containing:
                - 'q': (num_envs, nq) - Joint positions
                - 'qd': (num_envs, nv) - Joint velocities
                - 'T_blf': (num_envs, 4) - Body to left foot transform
                - 'T_brf': (num_envs, 4) - Body to right foot transform
                - 'T_stsw': (num_envs, 4) - Stance to swing transform
                - 'cmd_footstep': (num_envs, 4) - Footstep command
                - 'cmd_stance': (num_envs, 1) - Stance indicator
                - 'cmd_countdown': (num_envs, 1) - Countdown timer
                - 'p_wcom': (num_envs, 3) - CoM position
                - 'T_wbase': (num_envs, 7) - Base transform
                - 'v_b': (num_envs, 6) - Base velocity
        """
        return {
            "q": self.q_ref,
            "qd": self.qd_ref,
            "T_blf": self.T_blf_ref,
            "T_brf": self.T_brf_ref,
            "T_stsw": self.T_stsw_ref,
            "cmd_footstep": self.cmd_footstep_ref,
            "cmd_stance": self.cmd_stance_ref,
            "cmd_countdown": self.cmd_countdown_ref,
            "p_wcom": self.p_wcom_ref,
            "T_wbase": self.T_wbase_ref,
            "v_b": self.v_b_ref,
        }

    @property
    def ref_q(self) -> torch.Tensor:
        """Joint positions reference."""
        return self.q_ref

    @property
    def ref_qd(self) -> torch.Tensor:
        """Joint velocities reference."""
        return self.qd_ref

    @property
    def ref_p_wcom(self) -> torch.Tensor:
        """CoM position in world frame."""
        return self.p_wcom_ref

    @property
    def ref_v_b(self) -> torch.Tensor:
        """Base velocity in base frame (linear xyz, angular xyz)."""
        return self.v_b_ref

    @property
    def ref_T_wbase(self) -> torch.Tensor:
        """Base transform in world frame (x, y, z, qw, qx, qy, qz)."""
        return self.T_wbase_ref

    @property
    def ref_T_blf(self) -> torch.Tensor:
        """Body to left foot transform (x, y, z, yaw)."""
        return self.T_blf_ref

    @property
    def ref_T_brf(self) -> torch.Tensor:
        """Body to right foot transform (x, y, z, yaw)."""
        return self.T_brf_ref

    @property
    def ref_T_stsw(self) -> torch.Tensor:
        """Stance to swing foot transform (x, y, z, yaw)."""
        return self.T_stsw_ref

    @property
    def ref_T_wlf(self) -> torch.Tensor:
        """Left foot transform in world frame (x, y, z, qw, qx, qy, qz)."""
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        lf_pos_b = self.T_blf_ref[:, :3]
        lf_yaw = self.T_blf_ref[:, 3]
        lf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(lf_yaw), torch.zeros_like(lf_yaw), lf_yaw)
        lf_pos_w, lf_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, lf_pos_b, lf_quat_b)
        return torch.cat([lf_pos_w, lf_quat_w], dim=-1)

    @property
    def ref_T_wrf(self) -> torch.Tensor:
        """Right foot transform in world frame (x, y, z, qw, qx, qy, qz)."""
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        rf_pos_b = self.T_brf_ref[:, :3]
        rf_yaw = self.T_brf_ref[:, 3]
        rf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(rf_yaw), torch.zeros_like(rf_yaw), rf_yaw)
        rf_pos_w, rf_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, rf_pos_b, rf_quat_b)
        return torch.cat([rf_pos_w, rf_quat_w], dim=-1)

    @property
    def cmd_footstep(self) -> torch.Tensor:
        """Footstep command [x, y, z, yaw]."""
        return self.cmd_footstep_ref

    @property
    def cmd_foot_indicator(self) -> torch.Tensor:
        """Stance foot indicator (0=left, 1=right)."""
        return self.cmd_stance_ref

    @property
    def cmd_swing_foot_indicator(self) -> torch.Tensor:
        """Swing foot indicator (1=left, 0=right). Opposite of stance foot."""
        return 1.0 - self.cmd_stance_ref

    @property
    def cmd_countdown(self) -> torch.Tensor:
        """Countdown timer (0 during wait, 0->1->0 during step)."""
        return self.cmd_countdown_ref

    @property
    def prev_cmd_countdown(self):
        return self.prev_cmd_countdown_ref

    @property
    def T_wst(self):
        return self.T_wst_des

    @property
    def T_wsw(self):
        return self.T_wsw_des

    @property
    def stance_id(self):
        return self.stance_id_aux

    @property
    def swing_id(self):
        return self.swing_id_aux

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics based on current state."""
        # Track average trajectory step offset
        self.metrics["foot_track_time"] += self.traj_step_offset.float()

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def resample_command(self, env_ids: Sequence[int]):
        """Resample trajectory and starting frame for specified environments.

        Args:
            env_ids: Environment indices to resample.
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Randomly sample trajectories
        traj_indices = torch.randint(0, self.traj_num, (n,), device=self.device)
        self.traj_idx[env_ids] = traj_indices

        # Compute starting offset within each trajectory based on resample ratio
        ratios = torch.empty(n, device=self.device).uniform_(self.cfg.resample_ratio_range[0], self.cfg.resample_ratio_range[1])
        selected_traj_lengths = self.traj_lengths[traj_indices]
        traj_offsets = (ratios * selected_traj_lengths.float()).long()
        traj_offsets = torch.clamp(traj_offsets, torch.zeros_like(traj_offsets), selected_traj_lengths - 1)

        # Store offset and compute global index
        self.traj_step_offset[env_ids] = traj_offsets
        self.global_step_idx[env_ids] = self.starting_steps[traj_indices] + traj_offsets
        self.step_counter[env_ids] = 0

        # Sample random foot transforms (x, y, z, yaw)
        # if self.cfg.rel_random > 0.0 and hasattr(self.cfg, 'ranges_lf') and hasattr(self.cfg, 'ranges_rf'):
        #     # Left foot random transform
        #     self.T_blf_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_lf.pos_x)
        #     self.T_blf_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_lf.pos_y)
        #     self.T_blf_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_lf.pos_z)
        #     self.T_blf_random[env_ids, 3] = r.uniform_(*self.cfg.ranges_lf.yaw)

        #     # Right foot random transform
        #     self.T_brf_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_rf.pos_x)
        #     self.T_brf_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_rf.pos_y)
        #     self.T_brf_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_rf.pos_z)
        #     self.T_brf_random[env_ids, 3] = r.uniform_(*self.cfg.ranges_rf.yaw)

        # Update reference states for resampled environments
        self._update_reference_states(env_ids)
        self._update_stance_swing(env_ids, force_update=True)

    def _update_command(self):
        """Update the reference motion command by advancing frames."""
        # Advance steps
        self.step_counter += 1
        advance_envs = self.step_counter >= self.step_advance

        if torch.any(advance_envs):
            # Increment offset for environments that need to advance
            self.traj_step_offset[advance_envs] += 1

            # Clamp to trajectory limits per environment
            max_offsets = self.traj_lengths[self.traj_idx] - 1
            self.traj_step_offset = torch.clamp(self.traj_step_offset, torch.zeros_like(self.traj_step_offset), max_offsets)

            # Update global indices
            self.global_step_idx = self.starting_steps[self.traj_idx] + self.traj_step_offset

            # Reset step counter for advanced envs
            self.step_counter[advance_envs] = 0
            self._update_stance_swing(torch.arange(self.num_envs, device=self._env.device)[advance_envs], force_update=False)

        # Update reference states using global indices
        self._update_reference_states(slice(None))

    def _update_reference_states(self, env_ids: Sequence[int] | slice):
        """Update reference states using integer indexing (no interpolation).

        Args:
            env_ids: Environment indices to update.
        """
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Get trajectory data for all environments
        traj_q = self.data_q[self.global_step_idx[env_ids]]
        traj_qd = self.data_qd[self.global_step_idx[env_ids]]
        traj_T_blf = self.data_T_blf[self.global_step_idx[env_ids]]
        traj_T_brf = self.data_T_brf[self.global_step_idx[env_ids]]
        traj_T_stsw = self.data_T_stsw[self.global_step_idx[env_ids]]
        traj_p_wcom = self.data_p_wcom[self.global_step_idx[env_ids]]
        traj_T_wbase = self.data_T_wbase[self.global_step_idx[env_ids]]
        traj_v_b = self.data_v_b[self.global_step_idx[env_ids]]
        traj_cmd_footstep = self.data_cmd_footstep[self.global_step_idx[env_ids]]
        traj_cmd_stance = self.data_cmd_stance[self.global_step_idx[env_ids]]
        traj_cmd_countdown = self.data_cmd_countdown[self.global_step_idx[env_ids]]

        # Joint states are NOT randomized
        self.q_ref[env_ids] = traj_q
        self.qd_ref[env_ids] = traj_qd

        # CoM, base transforms, and base velocity are NOT randomized
        self.p_wcom_ref[env_ids] = traj_p_wcom
        self.T_wbase_ref[env_ids] = traj_T_wbase
        self.v_b_ref[env_ids] = traj_v_b
        self.T_blf_ref[env_ids] = traj_T_blf
        self.T_brf_ref[env_ids] = traj_T_brf
        self.T_stsw_ref[env_ids] = traj_T_stsw

        # Stance-to-swing, footstep commands, and countdown
        self.cmd_footstep_ref[env_ids] = traj_cmd_footstep
        self.cmd_stance_ref[env_ids] = traj_cmd_stance
        self.prev_cmd_countdown_ref[env_ids] = self.cmd_countdown_ref[env_ids]
        self.cmd_countdown_ref[env_ids] = traj_cmd_countdown

    def _update_stance_swing(self, env_ids, force_update=False):
        # Stance and swing ids
        indicator = self.cmd_foot_indicator == 0.0
        self.stance_id_aux = torch.where(indicator, self.left_foot_idx, self.right_foot_idx).reshape(-1).long()
        self.swing_id_aux = torch.where(indicator, self.right_foot_idx, self.left_foot_idx).reshape(-1).long()

        # World references - update T_wst_des and T_wsw_des when countdown changes (stepping starts)
        countdown_diff = torch.abs(self.prev_cmd_countdown_ref[env_ids] - self.cmd_countdown_ref[env_ids]).squeeze(-1)
        start_stepping_mask = countdown_diff > 0.7  # large abs change in countdown indicate start of step...
        if force_update:
            start_stepping_mask = torch.ones_like(start_stepping_mask, dtype=torch.bool)  # or we override update on start
        env_ids_start_stepping = env_ids[start_stepping_mask]

        if len(env_ids_start_stepping) > 0:
            # Get stance foot position and orientation in world frame (x, y, z, qw, qx, qy, qz)
            stance_pos_w = self.robot.data.body_pos_w[env_ids_start_stepping, self.stance_id_aux[env_ids_start_stepping].long()]  # (n, 3)
            stance_quat_w = self.robot.data.body_quat_w[env_ids_start_stepping, self.stance_id_aux[env_ids_start_stepping].long()]  # (n, 4)
            _, _, stance_yaw = euler_xyz_from_quat(stance_quat_w)  # (n,)

            # T_wst_des is (x, y, z, yaw) in world frame
            self.T_wst_des[env_ids_start_stepping, :3] = stance_pos_w
            self.T_wst_des[env_ids_start_stepping, 2] = 0.0  # TEMP: set z to 0
            self.T_wst_des[env_ids_start_stepping, 3] = stance_yaw

            # cmd_footstep_ref is [x, y, z, yaw] in stance foot frame
            cmd_x = self.cmd_footstep_ref[env_ids_start_stepping, 0]
            cmd_y = self.cmd_footstep_ref[env_ids_start_stepping, 1]
            cmd_z = self.cmd_footstep_ref[env_ids_start_stepping, 2]
            cmd_yaw = self.cmd_footstep_ref[env_ids_start_stepping, 3]  # yaw

            # Transform footstep command from stance foot frame to world frame
            cos_yaw = torch.cos(stance_yaw)
            sin_yaw = torch.sin(stance_yaw)
            swing_x_w = stance_pos_w[:, 0] + cmd_x * cos_yaw - cmd_y * sin_yaw
            swing_y_w = stance_pos_w[:, 1] + cmd_x * sin_yaw + cmd_y * cos_yaw
            # swing_z_w = stance_pos_w[:, 2] + cmd_z
            swing_yaw_w = stance_yaw + cmd_yaw

            # T_wsw_des is (x, y, z, yaw) in world frame
            self.T_wsw_des[env_ids_start_stepping, 0] = swing_x_w
            self.T_wsw_des[env_ids_start_stepping, 1] = swing_y_w
            # self.T_wsw_des[env_ids_start_stepping, 2] = swing_z_w
            self.T_wsw_des[env_ids_start_stepping, 2] = 0.0
            self.T_wsw_des[env_ids_start_stepping, 3] = swing_yaw_w

            is_left_stance = self.cmd_foot_indicator[env_ids_start_stepping].squeeze(-1) == 0.0
            left_stance_envs = env_ids_start_stepping[is_left_stance]
            right_stance_envs = env_ids_start_stepping[~is_left_stance]

            # Store positions (x, y, z)
            if len(left_stance_envs) > 0:
                self.left_foot_pos_w[left_stance_envs] = self.T_wst_des[left_stance_envs, :3]
                self.right_foot_pos_w[left_stance_envs] = self.T_wsw_des[left_stance_envs, :3]

                # Store quaternions from yaw
                left_yaw = self.T_wst_des[left_stance_envs, 3]
                right_yaw = self.T_wsw_des[left_stance_envs, 3]
                self.left_foot_quat_w[left_stance_envs, 0] = torch.cos(left_yaw / 2)  # w
                self.left_foot_quat_w[left_stance_envs, 1] = 0.0  # x
                self.left_foot_quat_w[left_stance_envs, 2] = 0.0  # y
                self.left_foot_quat_w[left_stance_envs, 3] = torch.sin(left_yaw / 2)  # z
                self.right_foot_quat_w[left_stance_envs, 0] = torch.cos(right_yaw / 2)  # w
                self.right_foot_quat_w[left_stance_envs, 1] = 0.0  # x
                self.right_foot_quat_w[left_stance_envs, 2] = 0.0  # y
                self.right_foot_quat_w[left_stance_envs, 3] = torch.sin(right_yaw / 2)  # z

            if len(right_stance_envs) > 0:
                self.right_foot_pos_w[right_stance_envs] = self.T_wst_des[right_stance_envs, :3]
                self.left_foot_pos_w[right_stance_envs] = self.T_wsw_des[right_stance_envs, :3]

                # Store quaternions from yaw
                right_yaw = self.T_wst_des[right_stance_envs, 3]
                left_yaw = self.T_wsw_des[right_stance_envs, 3]
                self.right_foot_quat_w[right_stance_envs, 0] = torch.cos(right_yaw / 2)  # w
                self.right_foot_quat_w[right_stance_envs, 1] = 0.0  # x
                self.right_foot_quat_w[right_stance_envs, 2] = 0.0  # y
                self.right_foot_quat_w[right_stance_envs, 3] = torch.sin(right_yaw / 2)  # z
                self.left_foot_quat_w[right_stance_envs, 0] = torch.cos(left_yaw / 2)  # w
                self.left_foot_quat_w[right_stance_envs, 1] = 0.0  # x
                self.left_foot_quat_w[right_stance_envs, 2] = 0.0  # y
                self.left_foot_quat_w[right_stance_envs, 3] = torch.sin(left_yaw / 2)  # z

    def _initialize_visualization(self):
        """Initialize visualization markers for foot tracking."""
        import omni.log

        # Create markers if debug visualization is enabled
        if self.cfg.debug_vis and hasattr(self.cfg, "foot_target_viz_cfg"):
            omni.log.info("Initializing foot tracking visualization...")

            # Target foot markers
            lf_target_cfg = self.cfg.foot_target_viz_cfg.replace(prim_path="/Visuals/LeftFootTarget")
            self.lf_target_markers = VisualizationMarkers(lf_target_cfg)

            rf_target_cfg = self.cfg.foot_target_viz_cfg.replace(prim_path="/Visuals/RightFootTarget")
            self.rf_target_markers = VisualizationMarkers(rf_target_cfg)

            # Footstep command marker (visualizes target footstep in world frame)
            footstep_cmd_cfg = self.cfg.foot_target_viz_cfg.replace(prim_path="/Visuals/FootstepCommand")
            self.footstep_cmd_markers = VisualizationMarkers(footstep_cmd_cfg)
            footstep_cmd_w_cfg = self.cfg.foot_target_viz_cfg.replace(prim_path="/Visuals/FootstepCommandWorld")
            self.footsetp_w_cmd_markers = VisualizationMarkers(footstep_cmd_w_cfg)

            # Foot command markers (boxes) - visualize left and right foot command positions
            if hasattr(self.cfg, "foot_cmd_viz_cfg"):
                lf_cmd_cfg = self.cfg.foot_cmd_viz_cfg.replace(prim_path="/Visuals/LeftFootCmd")
                self.lf_cmd_markers = VisualizationMarkers(lf_cmd_cfg)
                rf_cmd_cfg = self.cfg.foot_cmd_viz_cfg.replace(prim_path="/Visuals/RightFootCmd")
                self.rf_cmd_markers = VisualizationMarkers(rf_cmd_cfg)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for foot tracking."""
        if debug_vis:
            if hasattr(self, "lf_target_markers"):
                self.lf_target_markers.set_visibility(True)
            if hasattr(self, "rf_target_markers"):
                self.rf_target_markers.set_visibility(True)
            if hasattr(self, "footstep_cmd_markers"):
                self.footstep_cmd_markers.set_visibility(True)
            if hasattr(self, "footsetp_w_cmd_markers"):
                self.footsetp_w_cmd_markers.set_visibility(True)
            if hasattr(self, "lf_cmd_markers"):
                self.lf_cmd_markers.set_visibility(True)
            if hasattr(self, "rf_cmd_markers"):
                self.rf_cmd_markers.set_visibility(True)
        else:
            if hasattr(self, "lf_target_markers"):
                self.lf_target_markers.set_visibility(False)
            if hasattr(self, "rf_target_markers"):
                self.rf_target_markers.set_visibility(False)
            if hasattr(self, "footstep_cmd_markers"):
                self.footstep_cmd_markers.set_visibility(False)
            if hasattr(self, "footsetp_w_cmd_markers"):
                self.footsetp_w_cmd_markers.set_visibility(True)
            if hasattr(self, "lf_cmd_markers"):
                self.lf_cmd_markers.set_visibility(False)
            if hasattr(self, "rf_cmd_markers"):
                self.rf_cmd_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization of foot targets."""
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        # Visualize foot target positions
        if hasattr(self, "lf_target_markers") and hasattr(self, "rf_target_markers"):
            # Get current robot base pose in world frame
            robot_pos_w = self.robot.data.root_pos_w  # (num_envs, 3)
            robot_quat_w = self.robot.data.root_quat_w  # (num_envs, 4)

            # Convert T_blf and T_brf (x, y, z, yaw) to world frame
            # Extract position and yaw from transforms
            lf_pos_b = self.T_blf_ref[:, :3]  # (num_envs, 3)
            lf_yaw = self.T_blf_ref[:, 3]  # (num_envs,)

            rf_pos_b = self.T_brf_ref[:, :3]  # (num_envs, 3)
            rf_yaw = self.T_brf_ref[:, 3]  # (num_envs,)

            # Convert yaw to quaternion (rotation around z-axis)
            lf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(lf_yaw), torch.zeros_like(lf_yaw), lf_yaw)

            rf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(rf_yaw), torch.zeros_like(rf_yaw), rf_yaw)

            # Transform from body frame to world frame using CURRENT robot base pose
            # This makes the visualization follow the robot's actual position in each environment
            lf_pos_w, lf_quat_w = math_utils.combine_frame_transforms(robot_pos_w, robot_quat_w, lf_pos_b, lf_quat_b)

            rf_pos_w, rf_quat_w = math_utils.combine_frame_transforms(robot_pos_w, robot_quat_w, rf_pos_b, rf_quat_b)

        #     # Visualize targets
        #     self.lf_target_markers.visualize(lf_pos_w, lf_quat_w)
        #     self.rf_target_markers.visualize(rf_pos_w, rf_quat_w)

        # Visualize footstep command in world frame
        if hasattr(self, "footstep_cmd_markers") and self.left_foot_idx is not None and self.right_foot_idx is not None:
            # Get cmd_footstep: [x, y, sin(yaw), cos(yaw)] in stance foot frame
            cmd_x = self.cmd_footstep_ref[:, 0]  # (num_envs,)
            cmd_y = self.cmd_footstep_ref[:, 1]  # (num_envs,)
            cmd_yaw = self.cmd_footstep_ref[:, 3]  # (num_envs,)

            # Build position and quaternion in stance foot frame
            cmd_pos_stance = torch.stack([cmd_x, cmd_y, torch.zeros_like(cmd_x)], dim=-1)  # (num_envs, 3)
            cmd_quat_stance = math_utils.quat_from_euler_xyz(torch.zeros_like(cmd_yaw), torch.zeros_like(cmd_yaw), cmd_yaw)  # (num_envs, 4)

            # Get stance foot indicator (0=left stance, 1=right stance)
            stance_indicator = self.cmd_stance_ref.squeeze(-1)  # (num_envs,)
            is_left_stance = (stance_indicator < 0.5).unsqueeze(-1)  # (num_envs, 1)
            stance_pos_w = torch.where(is_left_stance, lf_pos_w, rf_pos_w)  # (num_envs, 3)
            stance_quat_w = torch.where(is_left_stance, lf_quat_w, rf_quat_w)  # (num_envs, 4)

            # Transform cmd_footstep from stance foot frame to world frame
            cmd_pos_w, cmd_quat_w = math_utils.combine_frame_transforms(stance_pos_w, stance_quat_w, cmd_pos_stance, cmd_quat_stance)

            # Visualize the footstep command
            self.footstep_cmd_markers.visualize(cmd_pos_w, cmd_quat_w)
            # Visualize left and right foot command positions in world frame

        if hasattr(self, "lf_cmd_markers") and hasattr(self, "rf_cmd_markers"):
            # Use the stored command positions and quaternions
            self.lf_cmd_markers.visualize(self.left_foot_pos_w, self.left_foot_quat_w)
            self.rf_cmd_markers.visualize(self.right_foot_pos_w, self.right_foot_quat_w)


class VelocityToFootTrackCommand(FootTrackCommand):
    cfg: VelocityToFootTrackCommandCfg

    def __init__(self, cfg: FootTrackCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the foot track command generator.

        Args:
            cfg: Configuration for the foot track command.
            env: The environment instance.

        Raises:
            ValueError: If the foot_track_data_path doesn't exist or contains invalid data.
            FileNotFoundError: If no valid .npz file is found.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Store config and env
        self.cfg = cfg
        self._env = env
        # Load foot tracking data from NPZ file
        self.target_planner = FootTargetPlanner(debug=False, 
                                                device=self.device, 
                                                batch_size=self.num_envs,
                                                dt=env.step_dt,
                                                base_left=self.cfg.base_left, 
                                                base_right=self.cfg.base_right, 
                                                single_swing_period=self.cfg.single_swing_period, 
                                                cmd_ranges=self.cfg.ranges)

        # -- command: x vel, y vel, yaw vel, heading
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        # self.heading_target = torch.zeros(self.num_envs, device=self.device)
        # -- metrics
        self._initialize_buffers()
        self._initialize_visualization()

        # Log information
        # import omni.log

    def _initialize_buffers(self):
        """Initialize buffers for foot tracking."""
        # Trajectory tracking
        self.traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_step_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.global_step_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Step counter for frame advancement
        # self.step_advance = max(1, int(self.traj_dt / self._env.step_dt))
        self.step_counter = torch.zeros(self.num_envs, device=self.device)

        # Reference state buffers (command outputs)
        self.q_ref = torch.zeros(self.num_envs, self.num_q, device=self.device)
        self.qd_ref = torch.zeros(self.num_envs, self.num_qd, device=self.device)
        # self.T_blf_ref = torch.zeros(self.num_envs, 4, device=self.device)
        # self.T_brf_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_stsw_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.p_wcom_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self.T_wbase_ref = torch.zeros(self.num_envs, 7, device=self.device)
        self.v_b_ref = torch.zeros(self.num_envs, 6, device=self.device)
        self.cmd_footstep_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self.cmd_stance_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.cmd_countdown_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_cmd_countdown_ref = torch.zeros(self.num_envs, 1, device=self.device)

        self.T_wsw_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.T_wst_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.stance_id_aux = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.swing_id_aux = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # # Store left and right foot world frame pose when stepping starts
        self.left_foot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_foot_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.right_foot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_foot_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # Metrics
        self.metrics["foot_track_time"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return string representation of the command generator."""
        msg = "VelocityToFootTrackCommand:\n"
        # msg += f"\tNumber of trajectories: {self.traj_num}\n"
        # msg += f"\tJoint dimensions - q: {self.num_q}, qd: {self.num_qd}\n"
        # msg += f"\tMax trajectory length: {self.traj_lengths.max().item()}\n"
        msg += f"\tDt: {self.traj_dt}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        # if self.cfg.heading_command:
        #     self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        #     # update heading envs
        #     self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # # update standing envs
        # self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.target_planner.cmd[env_ids] = self.vel_command_b[env_ids]

    def resample_command(self, env_ids: Sequence[int]):
        traj_q = self.data_q[0]
        traj_qd = self.data_qd[0]

        self.q_ref[env_ids] = traj_q
        self.qd_ref[env_ids] = traj_qd

    """
    Properties
    """

    @property
    def command(self) -> dict[str, torch.Tensor]:
        """The foot tracking command as a dictionary of tensors.

        Returns:
            Dictionary containing:
                - 'q': (num_envs, nq) - Joint positions
                - 'qd': (num_envs, nv) - Joint velocities
                - 'T_blf': (num_envs, 4) - Body to left foot transform
                - 'T_brf': (num_envs, 4) - Body to right foot transform
                - 'T_stsw': (num_envs, 4) - Stance to swing transform
                - 'cmd_footstep': (num_envs, 4) - Footstep command
                - 'cmd_stance': (num_envs, 1) - Stance indicator
                - 'cmd_countdown': (num_envs, 1) - Countdown timer
                - 'p_wcom': (num_envs, 3) - CoM position
                - 'T_wbase': (num_envs, 7) - Base transform
                - 'v_b': (num_envs, 6) - Base velocity
        """
        return {
            'q': self.q_ref,
            'qd': self.qd_ref,
            'T_blf': self.T_blf_ref,
            'T_brf': self.T_brf_ref,
            'T_stsw': self.T_stsw_ref,
            'cmd_footstep': self.cmd_footstep_ref,
            'cmd_stance': self.cmd_stance_ref,
            'cmd_countdown': self.cmd_countdown_ref,
            'p_wcom': self.p_wcom_ref,
            'T_wbase': self.T_wbase_ref,
            'v_b': self.v_b_ref
        }

    @property
    def ref_T_stsw(self) -> torch.Tensor:
        """Stance to swing foot transform (x, y, z, yaw)."""
        # Combine position (N, 3) and yaw (N,) into (N, 4)
        # print("ref_T_stsw called: ", self.target_planner.swing_to_stance_pos, self.target_planner.swing_to_stance_yaw)
        return torch.cat([
            self.target_planner.swing_to_stance_pos,
            self.target_planner.swing_to_stance_yaw.unsqueeze(-1)
        ], dim=-1)

    @property
    def T_brf_ref(self) -> torch.Tensor:
        """Body to right foot transform (x, y, z, yaw).
        
        Computes the relative transform from root to right foot target,
        accounting for root position and orientation.
        """
        # Get right foot target from planner (in local/COM frame)
        rf_target_pos = self.target_planner.right_foot_target_pos.clone()  # (N, 3)
        rf_target_yaw = self.target_planner.right_foot_target_yaw.clone()  # (N,)
        
        # Get root position and orientation (in world frame)
        root_pos_w = self.robot.data.root_pos_w  # (N, 3)
        root_quat_w = self.robot.data.root_quat_w  # (N, 4)
        
        # Extract root yaw from quaternion
        _, _, root_yaw = euler_xyz_from_quat(root_quat_w)  # (N,)

        # Rotate rf_target_pos[:, :2] 2d vectors with root_yaw
        # cos_theta = torch.cos(-root_yaw)
        # sin_theta = torch.sin(-root_yaw)
        # rot_mat = torch.stack([torch.stack([cos_theta, -sin_theta], dim=-1),
        #                        torch.stack([sin_theta,  cos_theta], dim=-1)], dim=-2)  # (N, 2, 2)
        # rf_target_pos[:, :2] = torch.bmm(rot_mat, rf_target_pos[:, :2].unsqueeze(-1)).squeeze(-1)
        
        # rf_target_pos[:, :2] = rf_target_pos[:, :2] + root_pos_w[:, :2] #+ self._env.scene.env_origins[:, :2]
        rf_target_pos[:, 2] = 0.01 + 0.0630 - root_pos_w[:, 2]

        # Rotate foot target position by inverse root quaternion to get it in root frame
        # Since foot target is in COM frame (same as root), we just rotate it
        # rf_pos_in_root = quat_apply_inverse(root_quat_w, rf_target_pos)  # (N, 3)
        
        # Compute relative yaw (foot yaw relative to root yaw)
        rf_yaw_rel = rf_target_yaw #- root_yaw  # (N,)
        
        # Combine position and yaw into (x, y, z, yaw) format
        return torch.cat([rf_target_pos, rf_yaw_rel.unsqueeze(-1)], dim=-1)  # (N, 4)

    @property
    def T_blf_ref(self) -> torch.Tensor:
        """Body to left foot transform (x, y, z, yaw).
        
        Computes the relative transform from root to left foot target,
        accounting for root position and orientation.
        """
        # Get left foot target from planner (in local/COM frame)
        lf_target_pos = self.target_planner.left_foot_target_pos.clone()  # (N, 3)
        lf_target_yaw = self.target_planner.left_foot_target_yaw.clone()  # (N,)
        
        # Get root position and orientation (in world frame)
        root_pos_w = self.robot.data.root_pos_w  # (N, 3)
        root_quat_w = self.robot.data.root_quat_w  # (N, 4)
        
        # Extract root yaw from quaternion
        _, _, root_yaw = euler_xyz_from_quat(root_quat_w)  # (N,)

        # Rotate lf_target_pos[:, :2] 2d vectors with root_yaw
        # cos_theta = torch.cos(-root_yaw)
        # sin_theta = torch.sin(-root_yaw)
        # rot_mat = torch.stack([torch.stack([cos_theta, -sin_theta], dim=-1),
        #                        torch.stack([sin_theta,  cos_theta], dim=-1)], dim=-2)  # (N, 2, 2)
        # lf_target_pos[:, :2] = torch.bmm(rot_mat, lf_target_pos[:, :2].unsqueeze(-1)).squeeze(-1)

        # lf_target_pos[:, :2] = lf_target_pos[:, :2] + root_pos_w[:, :2] #+ self._env.scene.env_origins[:, :2]
        lf_target_pos[:, 2] = 0.01 + 0.0630 - root_pos_w[:, 2]
        # Rotate foot target position by inverse root quaternion to get it in root frame
        # Since foot target is in COM frame (same as root), we just rotate it
        # lf_pos_in_root = quat_apply_inverse(root_quat_w, lf_target_pos)  # (N, 3)
        
        # Compute relative yaw (foot yaw relative to root yaw)
        lf_yaw_rel = lf_target_yaw #- root_yaw  # (N,)
        
        # Combine position and yaw into (x, y, z, yaw) format
        return torch.cat([lf_target_pos, lf_yaw_rel.unsqueeze(-1)], dim=-1)  # (N, 4)

    @property
    def ref_T_wlf(self) -> torch.Tensor:
        """Left foot transform in world frame (x, y, z, qw, qx, qy, qz)."""
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        lf_pos_b = self.T_blf_ref[:, :3]
        lf_yaw = self.T_blf_ref[:, 3]
        lf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(lf_yaw),torch.zeros_like(lf_yaw),lf_yaw)
        lf_pos_w, lf_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, lf_pos_b, lf_quat_b)
        return torch.cat([lf_pos_w, lf_quat_w], dim=-1)

    @property
    def ref_T_wrf(self) -> torch.Tensor:
        """Right foot transform in world frame (x, y, z, qw, qx, qy, qz)."""
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        rf_pos_b = self.T_brf_ref[:, :3]
        rf_yaw = self.T_brf_ref[:, 3]
        rf_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(rf_yaw), torch.zeros_like(rf_yaw), rf_yaw)
        rf_pos_w, rf_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, rf_pos_b, rf_quat_b)
        return torch.cat([rf_pos_w, rf_quat_w], dim=-1)

    @property
    def ref_T_wbase(self) -> torch.Tensor:
        """Base transform in world frame (x, y, z, qw, qx, qy, qz)."""
        # Return fixed transform: [0, 0, 0.68, 1, 0, 0, 0] for all envs
        fixed_transform = torch.zeros(self.num_envs, 7, device=self.device)
        fixed_transform[:, 2] = 0.68  # z position
        fixed_transform[:, 3] = 1.0  # qw
        return fixed_transform
    @property
    def cmd_footstep(self) -> torch.Tensor:
        """Footstep command [x, y, sin(yaw), cos(yaw)]."""
        # print("cmd_footstep called: ", self.cmd_footstep_ref)
        return torch.cat([
            self.target_planner.swing_to_stance_pos,
            self.target_planner.swing_to_stance_yaw.unsqueeze(-1)
        ], dim=-1)

    @property
    def cmd_foot_indicator(self) -> torch.Tensor:
        """Stance foot indicator (0=left, 1=right)."""
        # print("cmd_foot_indicator called: ", self.cmd_stance_ref)
        return self.cmd_stance_ref

    @property
    def cmd_swing_foot_indicator(self) -> torch.Tensor:
        """Swing foot indicator (1=left, 0=right). Opposite of stance foot."""
        # print("cmd_swing_foot_indicator called: ", (1.0 - self.cmd_stance_ref))
        return 1.0 - self.cmd_stance_ref

    @property
    def cmd_countdown(self) -> torch.Tensor:
        """Countdown timer (0 during wait, 0->1->0 during step)."""
        # During delay period (when in_delay is True), countdown should be 0
        in_delay = self.target_planner.is_in_delay()
        countdown = (1 - ((self.target_planner.curr_clock * 2) % 1)).unsqueeze(-1)
        # Set countdown to 0 when in delay
        countdown = torch.where(in_delay.unsqueeze(-1), torch.zeros_like(countdown), countdown)
        return countdown

    def __str__(self) -> str:
        return "VelocityToFootTrackCommand"

    def _update_command(self):
        self.target_planner._update_clock()
        self.target_planner._update_foot_targets()
        
        # Update stance foot indicator based on phase clock
        # sin <= 0: left is stance (right swings)
        # sin > 0: right is stance (left swings)
        curr_sin = torch.sin(self.target_planner.curr_clock * 2 * math.pi)
        # 0 = left stance, 1 = right stance
        self.cmd_stance_ref = (curr_sin > 0).float().unsqueeze(-1)  # (N, 1)
        self.cmd_footstep_ref = self.target_planner._compute_stance_to_swing_target()

    def _debug_vis_callback(self, event):
        """Callback for debug visualization of foot targets."""
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return
        if hasattr(self, "lf_cmd_markers") and hasattr(self, "rf_cmd_markers"):
            # Use the stored command positions and quaternions
            self.lf_cmd_markers.visualize(self.ref_T_wlf[:,:3], self.ref_T_wlf[:,3:])
            self.rf_cmd_markers.visualize(self.ref_T_wrf[:,:3], self.ref_T_wrf[:,3:])
