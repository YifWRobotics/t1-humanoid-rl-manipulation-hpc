from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple
import math

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_mul, quat_inv, quat_error_magnitude
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_regularization(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the reference"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    qref = getattr(env.command_manager.get_term(command_name), attr_name)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - qref
    return torch.sum(torch.abs(angle), dim=1)


def joint_tracking_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    term = env.command_manager.get_term(command_name)
    qref = getattr(term, attr_name)
    angle = torch.norm(asset.data.joint_pos[:, env.command_manager.get_term(command_name).joint_ids] - qref, dim=1)
    conditioned_angle = angle * ~getattr(term, "random_envs")
    return torch.exp(-conditioned_angle**2 / std**2)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power on the articulation using l2 power.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their power contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def position_command_error_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the position error using exp.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). 
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = getattr(env.command_manager.get_term(command_name), attr_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.exp(-distance**2 / std**2)


def orientation_command_error_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the orientation error using exp.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the exp of the
    shortest path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = getattr(env.command_manager.get_term(command_name), attr_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    ref_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    # ref_error = torch.square(offset_body_vel_w[:,indices] - body_vel[:,indices])
    exp_ref_error = torch.exp(-ref_error**2 / std**2)
    # rews = torch.mean(exp_ref_error, dim=[-1, -2])
    rews = exp_ref_error
    if torch.isnan(rews).any():
        print("nan in reference_traj")
    return rews


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = getattr(env.command_manager.get_term(command_name), attr_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    ref_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return ref_error
