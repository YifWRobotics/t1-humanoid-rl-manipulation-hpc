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
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import SE3IKCommandCfg


class SE3IKCommand(CommandTerm):
    cfg: SE3IKCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: SE3IKCommandCfg, env: ManagerBasedEnv):
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
        if not path.is_file() or path.suffix != ".npz":
            raise FileNotFoundError(
                f"Invalid dataset! Should be a .npz file. The one set in config is {self.cfg.path_dataset}."
            )

        # obtain the robot asset
        # -- robot and bodies
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_ids = self.robot.find_bodies(cfg.body_names)[0]
        self.joint_ids = self.robot.find_joints([
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
        ], preserve_order=True)[0]

        # load the dataset with new format
        dataset = np.load(self.cfg.path_dataset)

        # trajectory metadata
        self.starting_steps = torch.from_numpy(dataset['starting_steps']).long().to(self.device)  # (n,)
        self.ending_steps = torch.from_numpy(dataset['ending_steps']).long().to(self.device)  # (n,)
        self.traj_dt = dataset['dt'].item()  # float
        print(f"[__init__] starting_steps shape: {self.starting_steps.shape}")
        print(f"[__init__] ending_steps shape: {self.ending_steps.shape}")
        print(f"[__init__] traj_dt: {self.traj_dt}")

        # trajectory data (shared across all trajectories)
        self.data_xyzwxyz_BLH = torch.from_numpy(dataset['xyzwxyz_BLH']).float().to(self.device)  # (k, 7)
        self.data_xyzwxyz_BRH = torch.from_numpy(dataset['xyzwxyz_BRH']).float().to(self.device)  # (k, 7)
        self.data_qref = torch.from_numpy(dataset['qref']).float().to(self.device)  # (k, dim_qref)
        print(f"[__init__] data_xyzwxyz_BLH shape: {self.data_xyzwxyz_BLH.shape}")
        print(f"[__init__] data_xyzwxyz_BRH shape: {self.data_xyzwxyz_BRH.shape}")
        print(f"[__init__] data_qref shape: {self.data_qref.shape}")

        # compute trajectory properties
        self.traj_num = len(self.starting_steps)
        self.traj_lengths = self.ending_steps - self.starting_steps + 1  # inclusive of ending step
        print(f"[__init__] traj_num: {self.traj_num}")
        print(f"[__init__] traj_lengths shape: {self.traj_lengths.shape}")

        # step advance based on dt ratio
        self.step_advance = max(1, int(self.traj_dt / self._env.step_dt))  # at least need to step 1 step for advance
        self.step_counter = torch.zeros(self.num_envs, device=self.device)
        print(f"[__init__] step_advance: {self.step_advance}")
        print(f"[__init__] step_counter shape: {self.step_counter.shape}")
        print(f"[__init__] num_envs: {self.num_envs}")

        # create buffers to store the command
        # -- command: xyzwxyz_BLH, xyzwxyz_BRH, q_upper
        self.xyzwxyz_BLH = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BRH = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BLH_random = torch.zeros(self.num_envs, 7, device=self.device)
        self.xyzwxyz_BRH_random = torch.zeros(self.num_envs, 7, device=self.device)
        self.qref_indices = torch.tensor(self.cfg.qref_indices, device=self.device)
        self.qref = torch.zeros(self.num_envs, len(self.cfg.qref_indices), device=self.device)
        self.is_random = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        print(f"[__init__] xyzwxyz_BLH shape: {self.xyzwxyz_BLH.shape}")
        print(f"[__init__] xyzwxyz_BRH shape: {self.xyzwxyz_BRH.shape}")
        print(f"[__init__] qref_indices shape: {self.qref_indices.shape}")
        print(f"[__init__] qref shape: {self.qref.shape}")

        # trajectory tracking
        self.traj_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.traj_step_offset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # offset within trajectory
        self.global_step_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # index into data arrays
        print(f"[__init__] traj_idx shape: {self.traj_idx.shape}")
        print(f"[__init__] traj_step_offset shape: {self.traj_step_offset.shape}")
        print(f"[__init__] global_step_idx shape: {self.global_step_idx.shape}")

        # -- metrics
        self.metrics["error_EE_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_EE_ori"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "SE3IKCommand:\n"
        msg += f"\tNum Trajs: {self.traj_num}\n"
        msg += f"\tMax Traj Length: {self.traj_lengths.max().item()}\n"
        msg += f"\tDt: {self.traj_dt}\n"
        return msg

    """
    Properties
    """
    @property
    def command(self) -> torch.Tensor:
        """The desired SE3 and joint commands. Shape is (num_envs, 7+7+dim_qref)."""
        return torch.concatenate([self.xyzwxyz_BLH, self.xyzwxyz_BRH, self.qref], dim=1)

    @property
    def left_hand_pose_command(self) -> torch.Tensor:
        return pose_7d_to_9d(self.xyzwxyz_BLH)
        # return self.xyzwxyz_BLH

    @property
    def right_hand_pose_command(self) -> torch.Tensor:
        return pose_7d_to_9d(self.xyzwxyz_BRH)
        # return self.xyzwxyz_BRH

    @property
    def joint_reference(self) -> torch.Tensor:
        return self.qref

    @property
    def random_envs(self) -> torch.Tensor:
        return self.is_random

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
        pos_target_lh, quat_target_lh = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.xyzwxyz_BLH[:, :3],
            self.xyzwxyz_BLH[:, 3:],
        )
        pos_target_rh, quat_target_rh = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.xyzwxyz_BRH[:, :3],
            self.xyzwxyz_BRH[:, 3:],
        )

        # compute errors for each hand separately
        pos_error_lh, rot_error_lh = math_utils.compute_pose_error(
            pos_target_lh,  # left hand target
            quat_target_lh,
            pos_curr[:, 0, :],    # left hand current
            quat_curr[:, 0, :],
        )

        pos_error_rh, rot_error_rh = math_utils.compute_pose_error(
            pos_target_rh,  # right hand target
            quat_target_rh,
            pos_curr[:, 1, :],    # right hand current
            quat_curr[:, 1, :],
        )

        # accumulate errors
        self.metrics["error_EE_pos"] = (torch.norm(pos_error_lh, dim=-1) + torch.norm(pos_error_rh, dim=-1)) / max_command_step
        self.metrics["error_EE_ori"] = (torch.norm(rot_error_lh, dim=-1) + torch.norm(rot_error_rh, dim=-1)) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Sample the SE3IK command.

        This function samples a trajectory of SE3 command and qref for needed environments.
        """
        n = len(env_ids)
        print(f"[_resample_command] env_ids length: {n}")
        print(f"[_resample_command] env_ids: {env_ids if n < 10 else f'{env_ids[:10]}...(showing first 10)'}")

        # randomly sample trajectories
        traj_indices = torch.randint(0, self.traj_num, (n,), device=self.device)
        self.traj_idx[env_ids] = traj_indices
        print(f"[_resample_command] traj_indices shape: {traj_indices.shape}")

        # compute starting offset within each trajectory based on resample ratio
        ratios = torch.empty(n, device=self.device).uniform_(
            self.cfg.resample_ratio_range[0], self.cfg.resample_ratio_range[1]
        )
        print(f"[_resample_command] ratios shape: {ratios.shape}")

        # get trajectory lengths for selected trajectories
        selected_traj_lengths = self.traj_lengths[traj_indices]
        print(f"[_resample_command] selected_traj_lengths shape: {selected_traj_lengths.shape}")

        # compute offset within trajectory (0 to traj_length-1)
        traj_offsets = (ratios * selected_traj_lengths.float()).long()
        traj_offsets = torch.clamp(traj_offsets, torch.zeros_like(traj_offsets), selected_traj_lengths - 1)
        print(f"[_resample_command] traj_offsets shape: {traj_offsets.shape}")

        # store offset and compute global index
        self.traj_step_offset[env_ids] = traj_offsets
        self.global_step_idx[env_ids] = self.starting_steps[traj_indices] + traj_offsets
        print(f"[_resample_command] global_step_idx[env_ids] shape: {self.global_step_idx[env_ids].shape}")

        # reset step counter
        self.step_counter[env_ids] = 0

        # determine random env
        r = torch.empty(n, device=self.device)
        self.is_random[env_ids] = r.uniform_(0, 1) < self.cfg.rel_random
        print(f"[_resample_command] is_random[env_ids] shape: {self.is_random[env_ids].shape}")

        # sample random pose
        # -- translation
        self.xyzwxyz_BLH_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_lh.pos_x)
        self.xyzwxyz_BLH_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_lh.pos_y)
        self.xyzwxyz_BLH_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_lh.pos_z)
        self.xyzwxyz_BRH_random[env_ids, 0] = r.uniform_(*self.cfg.ranges_rh.pos_x)
        self.xyzwxyz_BRH_random[env_ids, 1] = r.uniform_(*self.cfg.ranges_rh.pos_y)
        self.xyzwxyz_BRH_random[env_ids, 2] = r.uniform_(*self.cfg.ranges_rh.pos_z)
        print(f"[_resample_command] xyzwxyz_BLH_random[env_ids] shape: {self.xyzwxyz_BLH_random[env_ids].shape}")
        print(f"[_resample_command] xyzwxyz_BRH_random[env_ids] shape: {self.xyzwxyz_BRH_random[env_ids].shape}")
        # -- orientation
        euler_angles_lh = torch.zeros_like(self.xyzwxyz_BLH_random[env_ids, :3])
        euler_angles_lh[:, 0].uniform_(*self.cfg.ranges_lh.roll)
        euler_angles_lh[:, 1].uniform_(*self.cfg.ranges_lh.pitch)
        euler_angles_lh[:, 2].uniform_(*self.cfg.ranges_lh.yaw)
        print(f"[_resample_command] euler_angles_lh shape: {euler_angles_lh.shape}")
        quat_lh = quat_from_euler_xyz(euler_angles_lh[:, 0], euler_angles_lh[:, 1], euler_angles_lh[:, 2])
        quat_lh = math_utils.quat_unique(quat_lh)  # Ensure w >= 0
        print(f"[_resample_command] quat_lh shape: {quat_lh.shape}")
        self.xyzwxyz_BLH_random[env_ids, 3:] = quat_lh
        euler_angles_rh = torch.zeros_like(self.xyzwxyz_BRH_random[env_ids, :3])
        # FIXED: Mirror roll and pitch for right hand (RH base frame has Y,Z flipped relative to LH)
        euler_angles_rh[:, 0].uniform_(*self.cfg.ranges_rh.roll)
        euler_angles_rh[:, 0] = -euler_angles_rh[:, 0]  # Negate roll
        euler_angles_rh[:, 1].uniform_(*self.cfg.ranges_rh.pitch)
        euler_angles_rh[:, 1] = -euler_angles_rh[:, 1]  # Negate pitch
        euler_angles_rh[:, 2].uniform_(*self.cfg.ranges_rh.yaw)
        print(f"[_resample_command] euler_angles_rh shape: {euler_angles_rh.shape}")
        quat_rh = quat_from_euler_xyz(euler_angles_rh[:, 0], euler_angles_rh[:, 1], euler_angles_rh[:, 2])
        quat_rh = math_utils.quat_unique(quat_rh)  # Ensure w >= 0
        print(f"[_resample_command] quat_rh shape: {quat_rh.shape}")
        self.xyzwxyz_BRH_random[env_ids, 3:] = quat_rh

        # on resample, also reset the joint pos to the staring qref
        self.xyzwxyz_BLH[env_ids] = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.xyzwxyz_BLH_random[env_ids],
            self.data_xyzwxyz_BLH[self.global_step_idx[env_ids]],
        )
        self.xyzwxyz_BRH[env_ids] = torch.where(
            self.is_random[env_ids].unsqueeze(-1),
            self.xyzwxyz_BRH_random[env_ids],
            self.data_xyzwxyz_BRH[self.global_step_idx[env_ids]]
        )
        print(f"[_resample_command] xyzwxyz_BLH[env_ids] final shape: {self.xyzwxyz_BLH[env_ids].shape}")
        print(f"[_resample_command] xyzwxyz_BRH[env_ids] final shape: {self.xyzwxyz_BRH[env_ids].shape}")
        self.qref[env_ids] = self.data_qref[self.global_step_idx, :][:, self.qref_indices][env_ids]
        print(f"[_resample_command] qref[env_ids] final shape: {self.qref[env_ids].shape}")

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
        self.xyzwxyz_BLH = torch.where(
            self.is_random.unsqueeze(-1),
            self.xyzwxyz_BLH,
            self.data_xyzwxyz_BLH[self.global_step_idx]
        )
        self.xyzwxyz_BRH = torch.where(
            self.is_random.unsqueeze(-1),
            self.xyzwxyz_BRH,
            self.data_xyzwxyz_BRH[self.global_step_idx]
        )
        self.qref = self.data_qref[self.global_step_idx, :][:, self.qref_indices]

        # TEST: directly set to qref, only to see if the command / qref makes sense
        # print("THIS SHOULD BE OFF IF YOU ARE NOT TESTING!!!")
        # self.robot.set_joint_position_target(self.qref, self.joint_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "lh_target_viz"):
                self.lh_target_viz = VisualizationMarkers(self.cfg.EE_target_viz_cfg)
                self.lh_current_viz = VisualizationMarkers(self.cfg.EE_curr_viz_cfg)
                self.rh_target_viz = VisualizationMarkers(self.cfg.EE_target_viz_cfg)
                self.rh_current_viz = VisualizationMarkers(self.cfg.EE_curr_viz_cfg)
                self.random_viz = VisualizationMarkers(self.cfg.random_viz_cfg)
            self.lh_target_viz.set_visibility(True)
            self.lh_current_viz.set_visibility(True)
            self.rh_target_viz.set_visibility(True)
            self.rh_current_viz.set_visibility(True)
            self.random_viz.set_visibility(True)
        else:
            if hasattr(self, "lh_target_viz"):
                self.lh_target_viz.set_visibility(False)
                self.lh_current_viz.set_visibility(False)
                self.rh_target_viz.set_visibility(False)
                self.rh_current_viz.set_visibility(False)
                self.random_viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "lh_target_viz"):
            # Get robot base pose
            robot_pos_w = self.robot.data.root_pos_w  # (num_envs, 3)
            robot_quat_w = self.robot.data.root_quat_w  # (num_envs, 4)

            # Get offseted commands
            lh_pose_target = self.left_hand_pose_command
            rh_pose_target = self.right_hand_pose_command

            # Convert target poses from body frame to world frame
            # For left hand (xyzwxyz format: position first, then quaternion)
            lh_pos_w = math_utils.quat_apply(robot_quat_w, lh_pose_target[:, :3]) + robot_pos_w
            lh_quat_w = math_utils.quat_mul(robot_quat_w, lh_pose_target[:, 3:7])

            # For right hand
            rh_pos_w = math_utils.quat_apply(robot_quat_w, rh_pose_target[:, :3]) + robot_pos_w
            rh_quat_w = math_utils.quat_mul(robot_quat_w, rh_pose_target[:, 3:7])

            self.lh_target_viz.visualize(lh_pos_w, lh_quat_w)
            self.rh_target_viz.visualize(rh_pos_w, rh_quat_w)

            # Visualize left hand target/current
            self.lh_current_viz.visualize(
                self.robot.data.body_pos_w[:, self.body_ids[0]],
                self.robot.data.body_quat_w[:, self.body_ids[0]]
            )

            # Visualize right hand target/current
            self.rh_current_viz.visualize(
                self.robot.data.body_pos_w[:, self.body_ids[1]],
                self.robot.data.body_quat_w[:, self.body_ids[1]]
            )

            # Visualize random
            random_marker_pos = torch.tensor(robot_pos_w, device=self.device)
            random_marker_pos[:, 2] = 1.2

            # Only show markers for random environments
            random_marker_pos = torch.where(
                self.is_random.unsqueeze(-1),
                random_marker_pos,
                torch.tensor([0.0, 0.0, -100.0], device=self.device)  # hide for non-random
            )
            self.random_viz.visualize(random_marker_pos, robot_quat_w)
