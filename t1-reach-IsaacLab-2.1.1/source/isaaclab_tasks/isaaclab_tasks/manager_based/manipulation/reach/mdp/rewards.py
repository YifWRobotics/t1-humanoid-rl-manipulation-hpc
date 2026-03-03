# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def linear_velocity_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the linear velocity error using exp.

    The function computes the linear velocity error between the desired linear velocity (from the command) and the
    current linear velocity of the asset's body (in world frame). The linear velocity error is computed as the exp of the
    L2-norm of the difference between the desired and current linear velocities.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_lin_vel_b = command[:, 7:10]
    des_lin_vel_w = quat_apply(asset.data.root_quat_w, des_lin_vel_b)
    curr_lin_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_lin_vel_w - des_lin_vel_w, dim=1)
    return torch.exp(-distance**2 / std**2)


def angular_velocity_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the angular velocity error using exp.

    The function computes the angular velocity error between the desired angular velocity (from the command) and the
    current angular velocity of the asset's body (in world frame). The angular velocity error is computed as the exp of the
    L2-norm of the difference between the desired and current angular velocities.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_ang_vel_b = command[:, 10:13]
    des_ang_vel_w = quat_apply(asset.data.root_quat_w, des_ang_vel_b)
    curr_ang_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], 3:]  # type: ignore
    distance = torch.norm(curr_ang_vel_w - des_ang_vel_w, dim=1)
    return torch.exp(-distance**2 / std**2)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the position error using exp.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). 
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.exp(-distance**2 / std**2)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the orientation error using exp.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the exp of the
    shortest path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    ref_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    # ref_error = torch.square(offset_body_vel_w[:,indices] - body_vel[:,indices])
    exp_ref_error = torch.exp(-ref_error / std**2)
    # rews = torch.mean(exp_ref_error, dim=[-1, -2])
    rews = exp_ref_error
    if torch.isnan(rews).any():
        print("nan in reference_traj")
    return rews


def reach_target(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, pos_thresh: float, ori_thresh: float) -> torch.Tensor:
    """
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ori_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    if pos_error < pos_thresh and ori_error < ori_thresh:
        reward = 1.0 / pos_error + 1.0 / ori_error
        return reward
    else:
        return torch.tensor(0.0, device=env.device, dtype=torch.float32)


def end_effector_lin_acc_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward the zero end-effector acceleration using exp.

    The function computes the end-effector acceleration and applies an exponential kernel to it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_acc_w = asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_acc_w, dim=1)
    return torch.exp(-std * distance**2)


def end_effector_lin_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward the zero end-effector acceleration using exp.

    The function computes the end-effector acceleration and applies an exponential kernel to it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_acc_w = asset.data.body_lin_acc_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_acc_w, dim=1)
    return distance**2


def end_effector_ang_acc_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward the zero end-effector acceleration using exp.

    The function computes the end-effector acceleration and applies an exponential kernel to it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_acc_w = asset.data.body_ang_acc_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_acc_w, dim=1)
    return torch.exp(-std * distance**2)


def end_effector_ang_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward the zero end-effector acceleration using exp.

    The function computes the end-effector acceleration and applies an exponential kernel to it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_acc_w = asset.data.body_ang_acc_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_acc_w, dim=1)
    return distance**2
