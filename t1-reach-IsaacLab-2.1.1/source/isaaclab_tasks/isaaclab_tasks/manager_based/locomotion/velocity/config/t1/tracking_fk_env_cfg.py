# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import torch
import os

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from .terms import commands_cfg as t1_commands_cfg
from .terms import commands as t1_commands
import isaaclab_tasks.manager_based.manipulation.reach.config.t1.terms.curriculum as T1_reach_curriculum
from isaaclab_assets.robots.booster import T1_REACH_CFG  # isort: skip
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

LH_IDX = 0
RH_IDX = 1


@configclass
class T1ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


def get_hand_pose_error(env: ManagerBasedRLEnv, idx, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get hand pose error in base frame from FK command."""
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    if idx == LH_IDX:
        hand_pos_cmd = fk_cmd.command["left_hand_pos"]  # (num_envs, 3)
        hand_quat_cmd = fk_cmd.command["left_hand_quat"]  # (num_envs, 4)
    else:
        hand_pos_cmd = fk_cmd.command["right_hand_pos"]  # (num_envs, 3)
        hand_quat_cmd = fk_cmd.command["right_hand_quat"]  # (num_envs, 4)

    hand_pos_b, hand_quat_b = get_hand_pose_b(env, asset_cfg)
    p_MeasureCmd_b = hand_pos_cmd - hand_pos_b
    quat_error = math_utils.quat_mul(math_utils.quat_conjugate(hand_quat_b), hand_quat_cmd)
    return torch.cat([p_MeasureCmd_b, math_utils.matrix_from_quat(quat_error).flatten(start_dim=-2)], dim=-1)


def get_left_hand_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left hand pose command from FK data in base frame.

    Returns the position and rotation (as 9D rotation matrix) of the left hand
    in base frame.

    Returns:
        torch.Tensor: Hand pose in base frame with shape (num_envs, 12) where:
            - [:, 0:3] is position (x, y, z)
            - [:, 3:12] is rotation matrix (full 3x3 matrix flattened: 9 values)
    """

    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    hand_pos = fk_cmd.command["left_hand_pos"]  # (num_envs, 3)
    hand_quat = fk_cmd.command["left_hand_quat"]  # (num_envs, 4)
    # Convert quaternion to rotation matrix and flatten to 9D representation
    hand_rot_mat = math_utils.matrix_from_quat(hand_quat)  # (num_envs, 3, 3)
    hand_rot = hand_rot_mat.flatten(start_dim=-2)  # (num_envs, 9)
    return torch.cat([hand_pos, hand_rot], dim=-1)  # (num_envs, 12)


def get_right_hand_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right hand pose command from FK data in base frame.

    Returns the position and rotation (as 9D rotation matrix) of the right hand
    in base frame.

    Returns:
        torch.Tensor: Hand pose in base frame with shape (num_envs, 12) where:
            - [:, 0:3] is position (x, y, z)
            - [:, 3:12] is rotation matrix (full 3x3 matrix flattened: 9 values)
    """
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    hand_pos = fk_cmd.command["right_hand_pos"]  # (num_envs, 3)
    hand_quat = fk_cmd.command["right_hand_quat"]  # (num_envs, 4)
    # Convert quaternion to rotation matrix and flatten to 9D representation
    hand_rot_mat = math_utils.matrix_from_quat(hand_quat)  # (num_envs, 3, 3)
    hand_rot = hand_rot_mat.flatten(start_dim=-2)  # (num_envs, 9)
    return torch.cat([hand_pos, hand_rot], dim=-1)  # (num_envs, 12)


def get_hand_pose_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get hand pose in base frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Get hand positions and quaternions in world frame
    hand_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze(1)  # (num_envs, 3)
    hand_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze(1)  # (num_envs, 4)

    # Get root position and quaternion
    root_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    root_quat_w = asset.data.root_quat_w  # (num_envs, 4)

    # Transform to base frame
    hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, hand_pos_w, hand_quat_w)

    return hand_pos_b, hand_quat_b  # (num_envs, 3), (num_envs, 4)


def get_hands_pose_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get hand poses (position + rotation) in base frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names (e.g., left and right hands).

    Returns:
        Hand poses in base frame (num_envs, num_bodies * 12) where each hand has:
            - 3 values for position (x, y, z)
            - 9 values for rotation matrix (full 3x3 matrix flattened)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get hand positions and quaternions in world frame
    hand_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    hand_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)
    num_bodies = hand_pos_w.shape[1]

    # Get root position and quaternion
    root_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    root_quat_w = asset.data.root_quat_w  # (num_envs, 4)

    # Expand root transforms to match number of bodies
    root_pos_w_expanded = root_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)  # (num_envs, num_bodies, 3)
    root_quat_w_expanded = root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)  # (num_envs, num_bodies, 4)

    # Flatten for batch processing
    root_pos_w_flat = root_pos_w_expanded.reshape(-1, 3)  # (num_envs * num_bodies, 3)
    root_quat_w_flat = root_quat_w_expanded.reshape(-1, 4)  # (num_envs * num_bodies, 4)
    hand_pos_w_flat = hand_pos_w.reshape(-1, 3)  # (num_envs * num_bodies, 3)
    hand_quat_w_flat = hand_quat_w.reshape(-1, 4)  # (num_envs * num_bodies, 4)

    # Transform to base frame
    hand_pos_b_flat, hand_quat_b_flat = math_utils.subtract_frame_transforms(root_pos_w_flat, root_quat_w_flat, hand_pos_w_flat, hand_quat_w_flat)

    # Reshape back to (num_envs, num_bodies, 3/4)
    hand_pos_b = hand_pos_b_flat.reshape(env.num_envs, num_bodies, 3)  # (num_envs, num_bodies, 3)
    hand_quat_b = hand_quat_b_flat.reshape(env.num_envs, num_bodies, 4)  # (num_envs, num_bodies, 4)

    # Convert quaternions to rotation matrices and flatten to 9D representation
    hand_rot_mat_b = math_utils.matrix_from_quat(hand_quat_b)  # (num_envs, num_bodies, 3, 3)
    hand_rot_9d_b = hand_rot_mat_b.flatten(start_dim=-2)  # (num_envs, num_bodies, 9)

    # Concatenate position and rotation for each hand
    hand_pose_b = torch.cat([hand_pos_b, hand_rot_9d_b], dim=-1)  # (num_envs, num_bodies, 12)

    return hand_pose_b.reshape(env.num_envs, -1)  # Flatten to (num_envs, num_bodies*12)


def get_hands_acc_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get hand linear accelerations in world frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names.

    Returns:
        Hand accelerations in world frame (num_envs, num_bodies * 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    hand_acc_w = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # (num_envs, num_bodies, 3)

    return hand_acc_w.reshape(env.num_envs, -1)  # Flatten to (num_envs, num_bodies*3)


def fk_hand_position_tracking_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, hand_index: int, std: float) -> torch.Tensor:
    """Reward for tracking FK reference hand positions in base frame using exponential kernel.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names.
        hand_index: Index of the hand (LH_IDX=0 or RH_IDX=1).
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get FK reference hand position (in base frame)
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    if hand_index == LH_IDX:
        hand_pos_base_ref = fk_cmd.command["left_hand_pos"]  # (num_envs, 3)
    else:
        hand_pos_base_ref = fk_cmd.command["right_hand_pos"]  # (num_envs, 3)

    # Get current hand position in world frame
    hand_pos_world_current = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (num_envs, 3)

    # Transform current hand position from world to base frame
    # FK data is generated relative to Trunk
    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_pos_w = asset.data.body_pos_w[:, trunk_body_id]  # (num_envs, 3)
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]  # (num_envs, 4)
    hand_pos_base_current, _ = math_utils.subtract_frame_transforms(trunk_pos_w, trunk_quat_w, hand_pos_world_current)

    # Compute position error in base frame
    pos_error = torch.norm(hand_pos_base_current - hand_pos_base_ref, dim=1)

    # Return exponential reward
    return torch.exp(-(pos_error**2) / std**2)


def fk_hand_orientation_tracking_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, hand_index: int) -> torch.Tensor:
    """Penalty for orientation error from FK reference hand orientations in base frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names.
        hand_index: Index of the hand (LH_IDX=0 or RH_IDX=1).

    Returns:
        Penalty tensor of shape (num_envs,) - returns error magnitude as penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get FK reference hand orientation
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    if hand_index == LH_IDX:
        hand_quat_base_ref = fk_cmd.command["left_hand_quat"]  # (num_envs, 4) - xyzw format
    else:
        hand_quat_base_ref = fk_cmd.command["right_hand_quat"]  # (num_envs, 4) - xyzw format

    # Get current hand orientation in world frame
    hand_quat_world_current = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)

    # Transform current hand orientation from world to base frame
    # q_TrunkHand = q_WTrunk^T * q_WorldHand
    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]  # (num_envs, 4)
    trunk_quat_inv = math_utils.quat_conjugate(trunk_quat_w)
    hand_quat_base_current = math_utils.quat_mul(trunk_quat_inv, hand_quat_world_current)
    quat_error = math_utils.quat_error_magnitude(hand_quat_base_current, hand_quat_base_ref)
    return quat_error


def fk_hand_orientation_tracking_reward_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, hand_index: int, std: float) -> torch.Tensor:
    """Reward for tracking FK reference hand orientations in base frame using exponential kernel.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names.
        hand_index: Index of the hand (LH_IDX=0 or RH_IDX=1).
        std: Standard deviation for exponential reward.

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get FK reference hand orientation
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")
    if hand_index == LH_IDX:
        hand_quat_base_ref = fk_cmd.command["left_hand_quat"]  # (num_envs, 4) - xyzw format
    elif hand_index == RH_IDX:
        hand_quat_base_ref = fk_cmd.command["right_hand_quat"]  # (num_envs, 4) - xyzw format


    hand_quat_world_current = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)
    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]  # (num_envs, 4)
    trunk_quat_inv = math_utils.quat_conjugate(trunk_quat_w)
    hand_quat_base_current = math_utils.quat_mul(trunk_quat_inv, hand_quat_world_current)
    quat_error = math_utils.quat_error_magnitude(hand_quat_base_current, hand_quat_base_ref)
    reward = torch.exp(-quat_error**2 / std**2)

    return reward


def joint_posture(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint deviations from default pose when command is small (stand still cost).

    Returns the sum of absolute joint deviations, scaled by whether the command norm is below threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    default_qpos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    qpos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    err_jnt = qpos - default_qpos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt), dim=1)
    return (err_jnt_sqsum).reshape(-1)  


def dp_rollout_position_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward for tracking DP rollout hand positions using exponential kernel.

    Returns exponential reward based on position error across both hands for DP rollout environments.
    For non-DP environments, returns 0.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names (should include both hands).
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,) ranging from 0 to 1.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")

    if not hasattr(fk_cmd, 'is_dp_rollout'):
        return torch.zeros(env.num_envs, device=env.device)

    is_dp = fk_cmd.is_dp_rollout  # (num_envs,)

    left_pos_ref = fk_cmd.command["left_hand_pos"]  # (num_envs, 3)
    right_pos_ref = fk_cmd.command["right_hand_pos"]  # (num_envs, 3)


    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_pos_w = asset.data.body_pos_w[:, trunk_body_id]
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]
    left_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    left_pos_b, _ = math_utils.subtract_frame_transforms(trunk_pos_w, trunk_quat_w, left_pos_w)
    left_err = torch.norm(left_pos_b - left_pos_ref, dim=1)
    right_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]
    right_pos_b, _ = math_utils.subtract_frame_transforms(trunk_pos_w, trunk_quat_w, right_pos_w)
    right_err = torch.norm(right_pos_b - right_pos_ref, dim=1)

    avg_err = (left_err + right_err) / 2.0

    reward = torch.exp(-(avg_err**2) / std**2)
    return torch.where(is_dp, reward, torch.zeros_like(reward))


def dp_rollout_orientation_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward for tracking DP rollout hand orientations using exponential kernel.

    Returns exponential reward based on orientation error across both hands for DP rollout environments.
    For non-DP environments, returns 0.

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names (should include both hands).
        std: Standard deviation for the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,) ranging from 0 to 1.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")

    if not hasattr(fk_cmd, 'is_dp_rollout'):
        return torch.zeros(env.num_envs, device=env.device)

    is_dp = fk_cmd.is_dp_rollout  # (num_envs,)

    left_quat_ref = fk_cmd.command["left_hand_quat"]  # (num_envs, 4)
    right_quat_ref = fk_cmd.command["right_hand_quat"]  # (num_envs, 4)

    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]
    trunk_quat_inv = math_utils.quat_conjugate(trunk_quat_w)
    left_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    left_quat_b = math_utils.quat_mul(trunk_quat_inv, left_quat_w)
    left_err = math_utils.quat_error_magnitude(left_quat_b, left_quat_ref)
    right_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]
    right_quat_b = math_utils.quat_mul(trunk_quat_inv, right_quat_w)
    right_err = math_utils.quat_error_magnitude(right_quat_b, right_quat_ref)

    avg_err = (left_err + right_err) / 2.0
    reward = torch.exp(-(avg_err**2) / std**2)
    return torch.where(is_dp, reward, torch.zeros_like(reward))


def dp_rollout_tracking_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, pos_std: float, ori_std: float) -> torch.Tensor:
    """Reward for tracking DP rollout trajectories (only active for DP rollout environments).

    Combines position and orientation tracking into a single exponential reward.
    For non-DP environments, returns 0 (neutral).

    Args:
        env: The environment.
        asset_cfg: Configuration for the robot asset with body names (should include both hands).
        pos_std: Standard deviation for position tracking (meters).
        ori_std: Standard deviation for orientation tracking (radians).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    fk_cmd: t1_commands.FKCommand = env.command_manager.get_term("both_hand_pose")

    if not hasattr(fk_cmd, 'is_dp_rollout'):
        return torch.zeros(env.num_envs, device=env.device)

    is_dp = fk_cmd.is_dp_rollout  # (num_envs,)

    left_pos_ref = fk_cmd.command["left_hand_pos"]
    right_pos_ref = fk_cmd.command["right_hand_pos"]
    left_quat_ref = fk_cmd.command["left_hand_quat"]
    right_quat_ref = fk_cmd.command["right_hand_quat"]

    trunk_body_id = asset.find_bodies("Trunk")[0][0]
    trunk_pos_w = asset.data.body_pos_w[:, trunk_body_id]
    trunk_quat_w = asset.data.body_quat_w[:, trunk_body_id]
    trunk_quat_inv = math_utils.quat_conjugate(trunk_quat_w)


    left_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    left_pos_b, _ = math_utils.subtract_frame_transforms(trunk_pos_w, trunk_quat_w, left_pos_w)
    left_pos_err = torch.norm(left_pos_b - left_pos_ref, dim=1)
    left_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    left_quat_b = math_utils.quat_mul(trunk_quat_inv, left_quat_w)
    left_ori_err = math_utils.quat_error_magnitude(left_quat_b, left_quat_ref)

    right_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]
    right_pos_b, _ = math_utils.subtract_frame_transforms(trunk_pos_w, trunk_quat_w, right_pos_w)
    right_pos_err = torch.norm(right_pos_b - right_pos_ref, dim=1)
    right_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]
    right_quat_b = math_utils.quat_mul(trunk_quat_inv, right_quat_w)
    right_ori_err = math_utils.quat_error_magnitude(right_quat_b, right_quat_ref)

    pos_reward = torch.exp(-((left_pos_err + right_pos_err) / 2.0) ** 2 / pos_std ** 2)
    ori_reward = torch.exp(-((left_ori_err + right_ori_err) / 2.0) ** 2 / ori_std ** 2)
    reward = torch.sqrt(pos_reward * ori_reward)

    return torch.where(is_dp, reward, torch.zeros_like(reward))


@configclass
class T1ReachObservationsFK:
    """Observations for FK tracking."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        left_hand_pose_command = ObsTerm(func=get_left_hand_pose_command)
        right_hand_pose_command = ObsTerm(func=get_right_hand_pose_command)
        hands_pose_b = ObsTerm(func=get_hands_pose_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee", "right_hand_ee"])})
        left_hand_error = ObsTerm(func=get_hand_pose_error, params={"idx": LH_IDX, "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"])})
        right_hand_error = ObsTerm(func=get_hand_pose_error, params={"idx": RH_IDX, "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"])})

        actions = ObsTerm(
            func=mdp.last_action,
            params={
                "action_name": "arm_actions"
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 30

    policy: PolicyCfg = PolicyCfg()


@configclass
class T1ReachCommandsFK:
    """Commands for FK tracking."""

    both_hand_pose = t1_commands_cfg.FKCommandCfg(
        asset_name="robot",
        fk_data_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "hand_pose_commands.npz"),
        dp_data_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "20260119_1925_80traj_AmelieDPRollout_10Hz.hdf5"),  # Check it is 10Hz
        rel_default=0.1,
        # Update rel_fk and rel_DP_rollout both to 0.35 for Joint Optimization stage
        rel_fk=0.7,
        rel_DProllout=0.0,
        rel_random=0.2,
        debug_vis=True,
        resampling_time_range=(3.0, 5.0),
        viz_robot_frames=["left_hand_ee", "right_hand_ee"],
        reference_body_name="Trunk",
        # Safety limits for FK-sampled poses
        enable_workspace_limits=True,
        workspace_limits=t1_commands_cfg.FKCommandCfg.WorkspaceLimits(
            left_x=(0.05, 0.5),
            left_y=(-0.2, 0.55),
            left_z=(-0.15, 0.6),
            right_x=(0.05, 0.5),
            right_y=(-0.55, 0.2),
            right_z=(-0.15, 0.6),
        ),
        enable_orientation_limits=True,
        max_orientation_deviation_deg=20,
        # Extended workspace for purely random poses (larger than FK limits)
        random_pose_ranges=t1_commands_cfg.FKCommandCfg.RandomPoseRanges(
            left_x=(0, 0.7),
            left_y=(-0.4, 0.7),
            left_z=(-0.2, 0.7),
            right_x=(0, 0.7),
            right_y=(-0.7, 0.4),
            right_z=(-0.2, 0.7),
        ),
    )


@configclass
class T1ReachTerminationsFK:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    arm_collision = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Waist",
                    "AL.*",  
                    "AR.*",  
                    "H.*",
                    "Trunk"
                ]
            ),
            "threshold": 0.5,  
        },
    )


@configclass
class T1ReachEventsFK:
    """Configuration for FK tracking environment events."""

    # Randomize end-effector mass (0-2kg payload), higher masses can harm training
    add_end_effector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link", "right_hand_link"]),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            ),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.7, 1.3),  
            "damping_distribution_params": (0.7, 1.3),    
        },
    )

    # Randomize joint friction
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            ),
            "friction_distribution_params": (0.01, 0.3),  # 0.01-0.3 N·m static friction
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Reset with higher noise for robustness
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.15, 0.15),  # 0.15 rad (±8.6°)
            "velocity_range": (0.2, 0.2),     # 0.2 rad/s
        },
    )

    # Exposure to small disturbances
    left_hand_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),  # Tunable
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link"]),
            "force_range": (-2.0, 2.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    right_hand_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),  # Tunable
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_link"]),
            "force_range": (-2.0, 2.0),
            "torque_range": (-0.5, 0.5),
        },
    )



@configclass
class T1ReachActionsFK:
    """Joint position actions for arm control."""
    arm_actions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        ],
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class T1ReachRewardsFK:
    """Reward terms for the MDP - FK tracking with end-effector pose rewards."""

    posture = RewTerm(
        func=joint_posture,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
                preserve_order=True,
            )
        },
    )

    # End-effector position and orienttion tracking rewards
    left_hand_position_tracking = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]),
            "hand_index": LH_IDX,
            "std": 1.0,
        },
    )

    left_hand_position_tracking_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]),
            "hand_index": LH_IDX,
            "std": 0.4,
        },
    )

    left_hand_position_tracking_very_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]),
            "hand_index": LH_IDX,
            "std": 0.2,
        },
    )

    left_hand_position_tracking_very_very_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=4.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]),
            "hand_index": LH_IDX,
            "std": 0.08,
        },
    )

    right_hand_position_tracking = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]),
            "hand_index": RH_IDX,
            "std": 1.0,
        },
    )

    right_hand_position_tracking_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]),
            "hand_index": RH_IDX,
            "std": 0.4,
        },
    )

    right_hand_position_tracking_very_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]),
            "hand_index": RH_IDX,
            "std": 0.2,
        },
    )

    right_hand_position_tracking_very_very_fine = RewTerm(
        func=fk_hand_position_tracking_reward,
        weight=4.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]),
            "hand_index": RH_IDX,
            "std": 0.08,
        },
    )

    left_hand_orientation_tracking = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]), "hand_index": LH_IDX, "std": 1.2},
    )

    left_hand_orientation_tracking_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]), "hand_index": LH_IDX, "std": 0.8},
    )

    left_hand_orientation_tracking_very_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]), "hand_index": LH_IDX, "std": 0.5},
    )

    right_hand_orientation_tracking = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]), "hand_index": RH_IDX, "std": 1.2},
    )

    right_hand_orientation_tracking_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]), "hand_index": RH_IDX, "std": 0.8},
    )

    right_hand_orientation_tracking_very_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]), "hand_index": RH_IDX, "std": 0.5},
    )

    left_hand_orientation_tracking_very_very_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee"]), "hand_index": LH_IDX, "std": 0.35},
    )

    right_hand_orientation_tracking_very_very_fine = RewTerm(
        func=fk_hand_orientation_tracking_reward_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_hand_ee"]), "hand_index": RH_IDX, "std": 0.35},
    )

    # Smoothness penalties - penalize excess motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            )
        },
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-3e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            )
        },
    )

    # End-effector penalty - penalize jerky hand movements
    ee_accel = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-1.4e-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee", "right_hand_ee"]),
        },
    )

    # Safety penalties
    torque_limit = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            )
        },
    )

    joint_pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
            )
        },
    )

    # DP Rollout Tracking Rewards - active for DP rollout environments only
    dp_position_tracking = RewTerm(
        func=dp_rollout_position_tracking,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee", "right_hand_ee"]),
            "std": 0.03,  # 3cm
        },
    )

    dp_orientation_tracking = RewTerm(
        func=dp_rollout_orientation_tracking,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee", "right_hand_ee"]),
            "std": 0.1,  # ~5.7 degrees
        },
    )

    # DP Rollout Tracking Reward - active reward for DP environments
    dp_tracking_reward = RewTerm(
        func=dp_rollout_tracking_reward,
        weight=2.0,  # Only affects DP rollout envs (returns 0 for others)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_ee", "right_hand_ee"]),
            "pos_std": 0.03,  # 3cm
            "ori_std": 0.1,  # ~5.7 degrees
        },
    )


@configclass
class T1ReachCurriculumFK:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight,
        params={
            "term_name": "action_rate",
            "start_step": 12000 * 24,
            "end_step": 20000 * 24,
        },
    )

    joint_vel = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight,
        params={
            "term_name": "joint_vel",
            "start_step": 12000 * 24,
            "end_step": 20000 * 24,
        },
    )

    joint_acc = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight,
        params={
            "term_name": "joint_acc",
            "start_step": 12000 * 24,
            "end_step": 20000 * 24,
        },
    )

    posture = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight,
        params={
            "term_name": "posture",
            "start_step": 1000 * 24,
            "end_step": 5000 * 24,
        },
    )


@configclass
class T1ReachEnvCfgFK(ReachEnvCfg):
    scene: T1ReachSceneCfg = T1ReachSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: T1ReachObservationsFK = T1ReachObservationsFK()
    commands: T1ReachCommandsFK = T1ReachCommandsFK()
    terminations: T1ReachTerminationsFK = T1ReachTerminationsFK()
    curriculum: T1ReachCurriculumFK = T1ReachCurriculumFK()
    rewards: T1ReachRewardsFK = T1ReachRewardsFK()
    events: T1ReachEventsFK = T1ReachEventsFK()
    actions: T1ReachActionsFK = T1ReachActionsFK()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Simulation settings
        self.sim.dt = 0.01
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # Robot configuration
        self.scene.robot = T1_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # PhysX settings for contact handling
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 60000

        # Curriculum weight parameters
        self.curriculum.action_rate.params["start_weight"] = -0.01
        self.curriculum.action_rate.params["end_weight"] = -0.1
        self.curriculum.joint_vel.params["start_weight"] = -0.001
        self.curriculum.joint_vel.params["end_weight"] = -0.002
        self.curriculum.joint_acc.params["start_weight"] = -1e-7
        self.curriculum.joint_acc.params["end_weight"] = -3e-6
        self.curriculum.posture.params["start_weight"] = -0.2
        self.curriculum.posture.params["end_weight"] = -0.2


@configclass
class T1ReachEnvCfgFK_PLAY(T1ReachEnvCfgFK):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 1.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.curriculum = None
        self.events.add_end_effector_mass = None
        self.events.scale_actuator_gains = None
        # Update FK command settings for play mode (Update rel_fk and rel_DP_rollout to 0.35 for Joint Optimization stage)
        self.commands.both_hand_pose.rel_default = 0.1
        self.commands.both_hand_pose.rel_fk = 0.7
        self.commands.both_hand_pose.rel_random = 0.2
        self.commands.both_hand_pose.rel_DP_rollout = 0.0
        self.commands.both_hand_pose.resampling_time_range = (3.0, 5.0)
