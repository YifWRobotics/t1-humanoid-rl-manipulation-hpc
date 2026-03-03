from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple
import math

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.utils.math import (
    quat_apply_inverse,
    yaw_quat,
    quat_mul,
    quat_inv,
    quat_error_magnitude,
    combine_frame_transforms,
    subtract_frame_transforms,
    quat_apply,
    euler_xyz_from_quat,
)
from ..terms.commands import FootTrackCommand
from ..terms.commands_cfg import FootTrackCommandCfg
from .utils import get_joint_ref_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_regularization(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_posture(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint deviations from default pose when command is small (stand still cost).

    Returns the sum of absolute joint deviations, scaled by whether the command norm is below threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err_jnt = asset.data.joint_pos - asset.data.default_joint_pos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt[:, asset_cfg.joint_ids]), dim=1)
    return (err_jnt_sqsum).reshape(-1)  # only have this cost when stand


def correct_joint_posture(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    joint_indices, _ = get_joint_ref_indices()
    joint_indices = joint_indices.to(command.command["q"].device)
    err_jnt = asset.data.joint_pos[:, asset_cfg.joint_ids] - command.command["q"][:, joint_indices[asset_cfg.joint_ids]]
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt), dim=1)
    is_standing = torch.logical_or(
        command.command["cmd_countdown"] < 0.001, command.command["cmd_countdown"] > 0.4
    ).reshape(-1)
    correct_posture = err_jnt_sqsum.float()
    return correct_posture * is_standing


def joint_pos_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    term: FootTrackCommand = env.command_manager.get_term(command_name)
    joint_indices, _ = get_joint_ref_indices()
    joint_indices = joint_indices.to(term.command["q"].device)
    error = torch.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - term.command["q"][:, joint_indices[asset_cfg.joint_ids]], dim=1
    )
    return torch.exp(-(error**2) / std**2)


def joint_pos_tracking_sum_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    term: FootTrackCommand = env.command_manager.get_term(command_name)
    joint_indices, _ = get_joint_ref_indices()
    joint_indices = joint_indices.to(term.command["q"].device)
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - term.command["q"][:, joint_indices[asset_cfg.joint_ids]]
    )  # B x J
    return torch.sum(torch.exp(-(error**2) / std**2), dim=-1)  # B


def joint_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    term: FootTrackCommand = env.command_manager.get_term(command_name)
    _, joint_indices_qd = get_joint_ref_indices()
    joint_indices_qd = joint_indices_qd.to(term.command["qd"].device)
    error = torch.norm(
        asset.data.joint_vel[:, asset_cfg.joint_ids] - term.command["qd"][:, joint_indices_qd[asset_cfg.joint_ids]],
        dim=1,
    )
    return torch.exp(-(error**2) / std**2)


def joint_vel_tracking_sum_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    term: FootTrackCommand = env.command_manager.get_term(command_name)
    _, joint_indices_qd = get_joint_ref_indices()
    joint_indices_qd = joint_indices_qd.to(term.command["qd"].device)
    error = asset.data.joint_vel[:, asset_cfg.joint_ids] - term.command["qd"][:, joint_indices_qd[asset_cfg.joint_ids]]
    return torch.sum(torch.exp(-(error**2) / std**2), dim=-1)


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
    return torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def position_command_error_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
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
    return torch.exp(-(distance**2) / std**2)


def orientation_command_error_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
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
    exp_ref_error = torch.exp(-ref_error / std**2)
    # rews = torch.mean(exp_ref_error, dim=[-1, -2])
    rews = exp_ref_error
    if torch.isnan(rews).any():
        print("nan in reference_traj")
    return rews


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = getattr(env.command_manager.get_term(command_name), attr_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    ref_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return ref_error


def ref_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute position error between current body position and reference position.
    Reference format: (n, (x, y, z, yaw)) in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    ref = command_term.command[attr_name]

    # get current body position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (n, 3)
    curr_pos_b, _ = subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, curr_pos_w)
    # transform reference position from base frame to world frame
    ref_pos_b = ref[:, :3]  # (n, 3) - (x, y, z)
    ref_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, ref_pos_b)
    # compute exp track
    return torch.exp(-(torch.norm(curr_pos_w - ref_pos_w, dim=1) ** 2) / std**2)


def ref_pos_xy_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute position error between current body position and reference position.
    Reference format: (n, (x, y, z, yaw)) in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)

    # get current body position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (n, 3)
    # transform reference position from base frame to world frame
    ref_pos_b = ref[:, :3]  # (n, 3) - (x, y, z)
    ref_pos_w = quat_apply(asset.data.root_quat_w, ref_pos_b) + asset.data.root_pos_w

    # compute exp track
    return torch.exp(-(torch.norm(curr_pos_w[:, :2] - ref_pos_w[:, :2], dim=1) ** 2) / std**2)


def ref_yaw_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute yaw error between current body orientation and reference yaw.
    Reference format: (n, (x, y, z, yaw)) in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    ref = command_term.command[attr_name]

    # get current body orientation
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (n, 4)
    _, curr_quat_b = subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, None, curr_quat_w)
    _, _, curr_yaw_b = euler_xyz_from_quat(curr_quat_b)  # (n, 3) Euler angles

    # extract reference yaw (4th column)
    ref_yaw_BFoot = ref[:, 3]  # (n,)

    # compute exp yaw tracking
    return torch.exp(-(torch.abs(curr_yaw_b - ref_yaw_BFoot) ** 2) / std**2)

def ref_base_yaw_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    _, _, ref_yaw = euler_xyz_from_quat(command_term.command[attr_name][:, 3:7])
    curr_quat_w = asset.data.root_quat_w
    _, _, curr_yaw_w = euler_xyz_from_quat(curr_quat_w)
    yaw_error = torch.abs(curr_yaw_w - ref_yaw)
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

    return torch.exp(-(torch.abs(yaw_error) ** 2) / std**2)

def ref_linvel_exp(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    ref = command_term.command[attr_name][:, :2]
    curr_linvel = asset.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(curr_linvel - ref), dim=-1) / std**2)


def ref_left_to_right_foot_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Regulate the measured left_to_right foot vector to match the reference left_to_right foot vector.

    Computes the vector from left foot to right foot (in base frame) for both measured and reference,
    then rewards matching using an exponential kernel.

    Args:
        env: The environment.
        command_name: Name of the command term containing T_blf and T_brf.
        std: Standard deviation for exponential kernel.
        asset_cfg: Scene entity config for the robot, must contain left_foot_link and right_foot_link.

    Returns:
        Exponential reward for left_to_right foot vector tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    # Get reference transforms (T_blf and T_brf are in base frame: x, y, z, yaw)
    T_blf_ref = command_term.command["T_blf"]  # (num_envs, 4)
    T_brf_ref = command_term.command["T_brf"]  # (num_envs, 4)

    # Extract reference foot positions in base frame
    left_foot_pos_b_ref = T_blf_ref[:, :3]  # (num_envs, 3)
    right_foot_pos_b_ref = T_brf_ref[:, :3]  # (num_envs, 3)

    # Compute reference left_to_right vector (right - left)
    ref_left_to_right = right_foot_pos_b_ref - left_foot_pos_b_ref  # (num_envs, 3)

    # Get measured foot positions in world frame
    left_foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (num_envs, 3)
    right_foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]  # (num_envs, 3)

    # Transform measured foot positions to base frame
    left_foot_pos_b_meas, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, left_foot_pos_w
    )  # (num_envs, 3)
    right_foot_pos_b_meas, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, right_foot_pos_w
    )  # (num_envs, 3)

    # Compute measured left_to_right vector (right - left)
    meas_left_to_right = right_foot_pos_b_meas - left_foot_pos_b_meas  # (num_envs, 3)

    # Compute error and exponential reward
    error = torch.norm(meas_left_to_right - ref_left_to_right, dim=-1)  # (num_envs,)
    return torch.exp(-(error**2) / std**2)


def ref_left_to_right_foot_yaw_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Regulate the measured left_to_right foot yaw angle to match the reference left_to_right foot yaw angle.

    Computes the yaw angle difference between right and left feet (in base frame) for both measured and reference,
    then rewards matching using an exponential kernel.

    Args:
        env: The environment.
        command_name: Name of the command term containing T_blf and T_brf.
        std: Standard deviation for exponential kernel.
        asset_cfg: Scene entity config for the robot, must contain left_foot_link and right_foot_link.

    Returns:
        Exponential reward for left_to_right foot yaw angle tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    # Get reference transforms (T_blf and T_brf are in base frame: x, y, z, yaw)
    T_blf_ref = command_term.command["T_blf"]  # (num_envs, 4)
    T_brf_ref = command_term.command["T_brf"]  # (num_envs, 4)

    # Extract reference foot yaw angles in base frame
    left_foot_yaw_b_ref = T_blf_ref[:, 3]  # (num_envs,)
    right_foot_yaw_b_ref = T_brf_ref[:, 3]  # (num_envs,)

    # Compute reference left_to_right yaw angle (right - left)
    ref_left_to_right_yaw = right_foot_yaw_b_ref - left_foot_yaw_b_ref  # (num_envs,)

    # Get measured foot orientations in world frame
    left_foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)
    right_foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]  # (num_envs, 4)

    # Transform measured foot orientations to base frame
    _, left_foot_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, None, left_foot_quat_w
    )  # (num_envs, 4)
    _, right_foot_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, None, right_foot_quat_w
    )  # (num_envs, 4)

    # Extract yaw angles from quaternions
    _, _, left_foot_yaw_b_meas = euler_xyz_from_quat(left_foot_quat_b)  # (num_envs,)
    _, _, right_foot_yaw_b_meas = euler_xyz_from_quat(right_foot_quat_b)  # (num_envs,)

    # Compute measured left_to_right yaw angle (right - left)
    meas_left_to_right_yaw = right_foot_yaw_b_meas - left_foot_yaw_b_meas  # (num_envs,)

    # Compute error and exponential reward
    error = torch.abs(meas_left_to_right_yaw - ref_left_to_right_yaw)  # (num_envs,)
    return torch.exp(-(error**2) / std**2)


def ref_pos_exp_w(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute position error between current body position and reference position in world frame.
    Reference format: (n, (x, y, z, qw, qx, qy, qz)) in world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)

    # get current body position in world frame
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (n, 3)

    # extract reference position (first 3 columns)
    ref_pos_w = ref[:, :3]  # (n, 3)

    # compute exp position tracking in world frame
    return torch.exp(-(torch.norm(curr_pos_w - ref_pos_w, dim=1) ** 2) / std**2)


def ref_yaw_exp_w(
    env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Compute yaw error between current body orientation and reference yaw in world frame.
    Reference format: (n, (x, y, z, qw, qx, qy, qz)) in world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)

    # get current body orientation in world frame
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (n, 4)
    # extract yaw from current quaternion
    curr_yaw = torch.atan2(
        2 * (curr_quat_w[:, 0] * curr_quat_w[:, 3] + curr_quat_w[:, 1] * curr_quat_w[:, 2]),
        1 - 2 * (curr_quat_w[:, 2] ** 2 + curr_quat_w[:, 3] ** 2),
    )

    # extract reference quaternion (last 4 columns)
    ref_quat_w = ref[:, 3:7]  # (n, 4)
    # extract yaw from reference quaternion
    ref_yaw = torch.atan2(
        2 * (ref_quat_w[:, 0] * ref_quat_w[:, 3] + ref_quat_w[:, 1] * ref_quat_w[:, 2]),
        1 - 2 * (ref_quat_w[:, 2] ** 2 + ref_quat_w[:, 3] ** 2),
    )

    # compute exp yaw tracking in world frame
    return torch.exp(-(torch.abs(curr_yaw - ref_yaw) ** 2) / std**2)


def track_pos_xy_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Track foot linear XY to commanded footstep XY, under stance foot frame.
    NOTE: asset_cfg must be left foot first, right foot second.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)

    # get positions using advanced indexing
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    pos_WSt = asset.data.body_pos_w[env_ids, command.stance_id]  # stance foot pos, shape (n, 3)
    pos_WSw = asset.data.body_pos_w[env_ids, command.swing_id]  # swing foot pos, shape (n, 3)
    quat_WSt = asset.data.body_quat_w[env_ids, command.stance_id]  # stance foot quat, shape (n, 4)

    # transform swing position to stance frame
    pos_StSw = quat_apply_inverse(quat_WSt, pos_WSw - pos_WSt)

    # compute XY error
    xy_error = torch.norm(pos_StSw[:, :2] - command.command["cmd_footstep"][:, :2], dim=1)

    return torch.exp(-(xy_error**2) / std**2)


def track_scaled_pos_xy_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Track foot linear XY to commanded footstep XY, under stance foot frame.
    NOTE: asset_cfg must be left foot first, right foot second.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    new_command = command.command["cmd_footstep"].clone()
    linvel_x = asset.data.root_lin_vel_b[:, 0]
    linvel_y = asset.data.root_lin_vel_b[:, 1]
    # TODO: better function for y. If push towards the stance foot, might need to change the stance foot
    new_command_x = torch.clamp(
        torch.where(
            torch.abs(linvel_x) >= threshold,
            threshold / 2 * torch.sign(linvel_x),
            command.command["cmd_footstep"][:, 0],
        ),
        min=-0.2,
        max=0.2,
    )
    new_command_y = torch.where(
        torch.abs(linvel_y) >= threshold, threshold / 2 * torch.sign(linvel_y), command.command["cmd_footstep"][:, 1]
    )
    new_command[:, 0] = new_command_x
    new_command[:, 1] = new_command_y
    # get positions using advanced indexing
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    pos_WSt = asset.data.body_pos_w[env_ids, command.stance_id]  # stance foot pos, shape (n, 3)
    pos_WSw = asset.data.body_pos_w[env_ids, command.swing_id]  # swing foot pos, shape (n, 3)
    quat_WSt = asset.data.body_quat_w[env_ids, command.stance_id]  # stance foot quat, shape (n, 4)

    # transform swing position to stance frame
    pos_StSw = quat_apply_inverse(quat_WSt, pos_WSw - pos_WSt)

    # compute XY error
    xy_error = torch.norm(pos_StSw[:, :2] - new_command[:, :2], dim=1)
    # print(new_command, linvel_x, linvel_y)
    return torch.exp(-(xy_error**2) / std**2)


def track_pos_xy_w_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    pos_WSt = asset.data.body_pos_w[env_ids, command.stance_id]  # stance foot pos, shape (n, 3)
    pos_WSw = asset.data.body_pos_w[env_ids, command.swing_id]  # swing foot pos, shape (n, 3)
    xy_errorSt = torch.norm(pos_WSt[:, :2] - command.T_wst[:, :2], dim=1)
    xy_errorSw = torch.norm(pos_WSw[:, :2] - command.T_wsw[:, :2], dim=1)
    return torch.exp(-(xy_errorSt**2) / std**2) + torch.exp(-(xy_errorSw**2) / std**2)


def track_single_foot_pos_xy_w_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    foot_id: int,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # shape (n, 3)
    is_stance = (command.command["cmd_stance"].squeeze(-1).long() == foot_id).float()
    ref_pos_w = torch.where(
        is_stance.unsqueeze(-1).bool(),
        command.T_wst,  # Use stance target if this foot is stance
        command.T_wsw,  # Use swing target if this foot is swing
    )

    # Compute XY error
    xy_error = torch.norm(foot_pos_w[:, :2] - ref_pos_w[:, :2], dim=1)

    return torch.exp(-(xy_error**2) / std**2)


def track_yaw_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Track foot orientation to commanded footstep yaw, under stance foot frame.
    NOTE: asset_cfg must be left foot first, right foot second.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)

    # get quaternions using advanced indexing
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    quat_WSt = asset.data.body_quat_w[env_ids, command.stance_id]  # stance foot quat, shape (n, 4)
    quat_WSw = asset.data.body_quat_w[env_ids, command.swing_id]  # swing foot quat, shape (n, 4)

    quat_stsw = quat_mul(quat_inv(quat_WSt), quat_WSw)
    _, _, yaw_stsw = euler_xyz_from_quat(quat_stsw)
    return torch.exp(-((yaw_stsw - command.command["cmd_footstep"][:, 3]) ** 2) / std**2)


def track_yaw_w_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg):
    """Track foot yaw orientations to reference yaw values in world frame.
    Tracks both stance and swing feet yaw against their world frame reference values.
    NOTE: asset_cfg must be left foot first, right foot second.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)

    # Get stance and swing foot quaternions in world frame
    quat_WSt = asset.data.body_quat_w[env_ids, command.stance_id]  # stance foot quat, shape (n, 4)
    quat_WSw = asset.data.body_quat_w[env_ids, command.swing_id]  # swing foot quat, shape (n, 4)

    # Extract yaw from current orientations
    _, _, yaw_WSt = euler_xyz_from_quat(quat_WSt)
    _, _, yaw_WSw = euler_xyz_from_quat(quat_WSw)

    # Extract reference yaw directly from T_wst and T_wsw (format: x, y, z, yaw)
    ref_yaw_st = command.T_wst[:, 3]  # (n,)
    ref_yaw_sw = command.T_wsw[:, 3]  # (n,)

    # Compute yaw errors for both feet
    yaw_errorSt = torch.abs(yaw_WSt - ref_yaw_st)
    yaw_errorSw = torch.abs(yaw_WSw - ref_yaw_sw)

    return torch.exp(-(yaw_errorSt**2) / std**2) + torch.exp(-(yaw_errorSw**2) / std**2)


def track_single_foot_yaw_w_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    foot_id: int,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    _, _, foot_yaw = euler_xyz_from_quat(foot_quat_w)
    is_stance = (command.command["cmd_stance"].squeeze(-1).long() == foot_id).float()
    ref_yaw = torch.where(
        is_stance.bool(),
        command.T_wst[:, 3],  # Use stance target yaw if this foot is stance
        command.T_wsw[:, 3],  # Use swing target yaw if this foot is swing
    )
    yaw_error = torch.abs(foot_yaw - ref_yaw)

    return torch.exp(-(yaw_error**2) / std**2)


def flat_link_orientation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector of the selected link.
    """
    # Get the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute projected gravity for the specific body
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)
    gravity_dir = asset.data.GRAVITY_VEC_W  # (3,)
    projected_gravity_b = quat_apply_inverse(body_quat, gravity_dir)  # (num_envs, 3)
    return torch.exp(-torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) / std**2)


def trunk_yaw_align_with_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward base (trunk) yaw being close to the average of the two feet yaws.

    The provided :attr:`asset_cfg.body_ids` must correspond to
    ``["Trunk", "left_foot_link", "right_foot_link"]`` in this order.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # body_quat_w: (num_envs, 3, 4) for [Trunk, left_foot_link, right_foot_link]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    trunk_quat = body_quat_w[:, 0]
    left_quat = body_quat_w[:, 1]
    right_quat = body_quat_w[:, 2]

    # Extract yaw angles
    _, _, trunk_yaw = euler_xyz_from_quat(trunk_quat)
    _, _, left_yaw = euler_xyz_from_quat(left_quat)
    _, _, right_yaw = euler_xyz_from_quat(right_quat)

    avg_feet_yaw = 0.5 * (left_yaw + right_yaw)

    # Angle difference wrapped to [-pi, pi]
    yaw_diff = trunk_yaw - avg_feet_yaw
    yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

    return torch.exp(-(yaw_diff**2) / std**2)


def track_joint_posture_exp(env, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize joint deviations from default pose when command is small (stand still cost).

    Returns the sum of absolute joint deviations, scaled by whether the command norm is below threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err_jnt = asset.data.joint_pos - asset.data.default_joint_pos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-(err_jnt_sqsum**2) / std**2)


def track_linvel_b_l2(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "foot_pos_cmd",
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    cmd: FootTrackCommand = env.command_manager.get_term(command_name)
    lin_vel_error = torch.sum(torch.square(cmd.ref_v_b[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def correct_foot_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 0.1
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    # foot indicator: 0=left stance, 1=right stance
    stance_id = torch.where(command.command["cmd_stance"].squeeze(-1).long() == 0, 0, 1)
    swing_id = torch.where(command.command["cmd_stance"].squeeze(-1).long() == 0, 1, 0)

    batch_indices = torch.arange(contacts.shape[0], device=contacts.device)
    swing_in_air = ~contacts[batch_indices, swing_id]  # shape: (num_envs,)
    swing_on_ground = contacts[batch_indices, swing_id]  # shape: (num_envs,)
    stance_on_ground = contacts[batch_indices, stance_id]
    is_stepping = torch.logical_and(
        command.command["cmd_countdown"] >= 0.001, command.command["cmd_countdown"] <= 0.4
    ).reshape(-1)
    is_standing = torch.logical_or(
        command.command["cmd_countdown"] < 0.001, command.command["cmd_countdown"] > 0.4
    ).reshape(-1)
    correct_contact_during_stepping = (
        torch.logical_and(is_stepping, swing_in_air).float() + torch.logical_and(is_stepping, stance_on_ground).float()
    )
    correct_contact_during_standing = (
        torch.logical_and(is_standing, swing_on_ground).float()
        + torch.logical_and(is_standing, stance_on_ground).float()
    )

    # print(f"swing_in_air: {swing_in_air}\nstance_on_ground: {stance_on_ground}\nis_stepping: {is_stepping}\nis_standing: {is_standing}")
    # print(f"correct_stepping: {correct_contact_during_stepping}")
    # print(f"correct_standing: {correct_contact_during_standing}")

    # all correct: 1.0; half: 0.5: all wrong: 0.0
    return (correct_contact_during_stepping + correct_contact_during_standing) * 0.5


def correct_single_foot_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_sensor_cfg: SceneEntityCfg,
    foot_id: int,
) -> torch.Tensor:
    """Reward correct single foot contact during stepping and standing.

    During stepping (countdown > 0): reward when the single foot is in the air
    During standing (countdown == 0): reward when the single foot is on the ground

    Returns 1.0 for correct contact, 0.0 for incorrect contact.
    """
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 10.0
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    # Detect if the foot is stance, if stance, id = 1 else 0
    is_stance = torch.where(command.command["cmd_stance"].squeeze(-1).long() == foot_id, 1, 0)

    batch_indices = torch.arange(contacts.shape[0], device=contacts.device)
    incorrect_during_standing = ~contacts[batch_indices, 0]
    incorrect_during_stepping = contacts[batch_indices, 0] ^ is_stance
    should_stepping = torch.logical_and(
        command.command["cmd_countdown"] >= 0.001, command.command["cmd_countdown"] <= 0.4
    ).reshape(-1)
    should_standing = torch.logical_or(
        command.command["cmd_countdown"] < 0.001, command.command["cmd_countdown"] > 0.4
    ).reshape(-1)
    incorrect_contact_during_stepping = torch.logical_and(should_stepping, incorrect_during_stepping).float()
    incorrect_contact_during_standing = torch.logical_and(should_standing, incorrect_during_standing).float()
    # print("\nfoot id:", foot_id)
    # print("contacts:", contacts[:, 0])
    # print("is_stance:", is_stance)
    # print("should_stepping:", should_stepping)
    # print("should_standing:", should_standing)
    # print("incorrect_stepping:", incorrect_contact_during_stepping)
    # print("incorrect_standing:", incorrect_contact_during_standing)
    return incorrect_contact_during_stepping * 2.0 + incorrect_contact_during_standing


def feet_rotate(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalize feet rotate-in-place.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]

    angvel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2:3]
    reward = torch.sum(angvel.norm(dim=-1) * contacts, dim=1)
    return reward


def stance_height_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Penalize both foot heights from desired height during stance phase.
    During stance, both feet should be nearly zero (on the ground).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    both_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].reshape(
        -1, 2
    )  # (num_envs, 2) # get height of two feet
    is_left_stance = command.command["cmd_stance"].squeeze(-1).long() == 0
    stance_height = torch.where(is_left_stance, both_height[:, 0], both_height[:, 1])
    return torch.abs(stance_height)  # (num_envs,)


def stance_height_penalty_stair(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Penalize stance foot height deviation from initial neutral height for stair climbing.

    Uses lf_neutral and rf_neutral from command to determine the initial foot heights.
    During stance, the stance foot should maintain its neutral height on the stair.
    """
    from ..terms.commands_cfg import FootTrackMotionTypeCommandCfg

    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    both_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].reshape(-1, 2)

    lf_neutral_height = command.lf_neutral[:, 2]
    rf_neutral_height = command.rf_neutral[:, 2]
    is_left_stance = command.command["cmd_stance"].squeeze(-1).long() == 0
    stance_height = torch.where(is_left_stance, both_height[:, 0], both_height[:, 1])
    stance_neutral_height = torch.where(is_left_stance, lf_neutral_height, rf_neutral_height)
    return torch.abs(stance_height - stance_neutral_height)


def swing_foot_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    swing_scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    both_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].reshape(
        -1, 2
    )  # (num_envs, 2) # get height of two feet
    is_left_stance = command.command["cmd_stance"].squeeze(-1).long() == 0
    # stance_height = torch.where(is_left_stance, both_height[:, 0], both_height[:, 1])
    swing_height = torch.where(is_left_stance, both_height[:, 1], both_height[:, 0])
    # env_ids = torch.arange(asset.num_instances, device=asset.device)
    # swing_height = asset.data.body_pos_w[env_ids, command.swing_id, 2]
    # height_error = torch.abs(swing_height - target_height)
    # height_reward = torch.exp(-height_error / std)
    is_swing = torch.logical_and(command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5)
    encourage_swing_and_penalize_stance = torch.where(  # 1 for swing, -1 for swing foot during double support phase.
        is_swing.squeeze(-1), swing_scale * torch.ones_like(swing_height), -torch.ones_like(swing_height)
    )
    return swing_height * is_swing.squeeze(-1)


def correct_foot_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stance: bool,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    foot_pos = (
        asset.data.body_pos_w[env_ids, command.stance_id, 2]
        if stance
        else asset.data.body_pos_w[env_ids, command.swing_id, 2]
    )
    return torch.abs(foot_pos) * torch.logical_and(
        command.command["cmd_countdown"].reshape(-1) > 0.0, command.command["cmd_countdown"].reshape(-1) <= 0.5
    )


def feet_air_time(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Time spent (in s) in the air since the last detach.
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # Time spent (in s) in contact since the last contact.
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, current_air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for double support
    reward *= torch.logical_and(
        command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5
    ).reshape(-1)
    return reward


def feet_air_time_at_contact(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds at the moment of contact.

    This function rewards the agent for taking steps up to a specified threshold at the moment
    when the foot makes contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (num_envs, num_feet) True if foot just made contact.
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Time spent (in s) in the air before the last contact.
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)
    reward = torch.sum(first_contact * (last_air_time - threshold), dim=1)
    return reward


def feet_airtime_in_swing(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, foot_id: int
) -> torch.Tensor:
    """Reward long air time for the correct swing foot during swing phase.

    This function rewards the agent for keeping the swing foot in the air for a longer duration
    during the swing phase, up to a specified threshold. Only the correct swing foot is rewarded.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids[0]]
    is_swing = torch.where(command.command["cmd_stance"].squeeze(-1).long() == foot_id, 0, 1).float()
    reward = torch.square(torch.clamp(air_time, max=threshold)) * is_swing
    reward *= (
        torch.logical_and(command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5)
        .reshape(-1)
        .float()
    )
    return reward


def feet_contacttime_in_stance(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, foot_id: int
) -> torch.Tensor:
    """Reward long contact time for the correct stance foot during standing phase.

    This function rewards the agent for keeping the stance foot in contact with the ground
    for a longer duration during the standing phase (countdown == 0), up to a specified threshold.
    Only the correct stance foot is rewarded.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids[0]]
    is_stance = torch.where(command.command["cmd_stance"].squeeze(-1).long() == foot_id, 1, 0).float()
    reward = torch.square(torch.clamp(contact_time, max=threshold)) * is_stance
    reward *= (
        torch.logical_and(command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5)
        .reshape(-1)
        .float()
    )
    return reward


def feet_airtime_penalty(
    env, command_name: str, min_air_time: float, std: float, sensor_cfg: SceneEntityCfg, foot_id: int
) -> torch.Tensor:
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids[0]]
    is_swing = torch.where(command.command["cmd_stance"].squeeze(-1).long() == foot_id, 0, 1).float()
    # Penalize when air time is below threshold using exponential kernel
    error = torch.clamp(min_air_time - air_time, min=0.0)
    penalty = (1.0 - torch.exp(-(error**2) / std**2)) * is_swing
    penalty *= torch.logical_and(
        command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5
    ).reshape(-1)
    return penalty


def swing_foot_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    foot_id: int,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    is_swing_foot = torch.where(command.command["cmd_stance"].squeeze(-1).long() == foot_id, 0, 1).float()
    height_error = torch.clamp(foot_height - target_height, min=0.0)
    reward = torch.exp(-((height_error) ** 2) / std**2)
    is_swinging = torch.logical_and(
        command.command["cmd_countdown"] > 0.0, command.command["cmd_countdown"] <= 0.5
    ).reshape(-1)
    reward *= is_swinging.float() * is_swing_foot
    return reward


def base_height_range(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        terrain_offset = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        adjusted_min_height = min_height + terrain_offset
        adjusted_max_height = max_height + terrain_offset
    else:
        adjusted_min_height = min_height
        adjusted_max_height = max_height
    current_height = asset.data.root_pos_w[:, 2]
    below_min = torch.clamp(adjusted_min_height - current_height, min=0.0)
    above_max = torch.clamp(current_height - adjusted_max_height, min=0.0)

    # Return L2 norm penalty
    return torch.square(below_min) + torch.square(above_max)


def base_height_range_from_foot(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
) -> torch.Tensor:
    """Penalize base height relative to the lowest foot position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    lowest_foot_height = torch.min(foot_heights, dim=1)[0]
    base_height = asset.data.root_pos_w[:, 2]
    relative_height = base_height - lowest_foot_height
    below_min = torch.clamp(min_height - relative_height, min=0.0)
    above_max = torch.clamp(relative_height - max_height, min=0.0)
    # print(below_min,above_max)
    return torch.square(below_min) + torch.square(above_max)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using exp kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    squared_penalty = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-squared_penalty / std**2)


def base_height_exp_from_foot(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
) -> torch.Tensor:
    """Reward base height relative to the lowest foot using exp kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    lowest_foot_height = torch.min(foot_heights, dim=1)[0]
    base_height = asset.data.root_pos_w[:, 2]
    relative_height = base_height - lowest_foot_height

    squared_penalty = torch.square(relative_height - target_height)
    return torch.exp(-squared_penalty / std**2)


def base_z_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    attr_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base z position tracking to reference base z using exponential kernel.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    ref_base_transform = command_term.command[attr_name]  # Format: [x, y, z, qw, qx, qy, qz]
    ref_z = ref_base_transform[:, 2]  # Extract z coordinate
    current_z = asset.data.root_pos_w[:, 2]
    z_error = torch.abs(current_z - ref_z)
    return torch.exp(-(z_error ** 2) / std ** 2)


def action_countdown_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    is_standing = torch.logical_or(
        command.command["cmd_countdown"] < 0.001, command.command["cmd_countdown"] > 0.5
    ).reshape(-1)
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * is_standing


def action_smooth(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "prev_action_rate"):
        env._prev_action = torch.zeros_like(env.action_manager.action)
    cur_action = env.action_manager.action - env.action_manager.prev_action
    smoothness = cur_action - env._prev_action
    env._prev_action = cur_action.clone()
    return torch.sum(torch.square(smoothness), dim=1)


def base_in_middle_during_standing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward center of mass being in the middle of the two feet during standing phase.
    Requires both feet to be in contact.

    Uses the actual COM position instead of body frame position for accurate balance tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)

    # Get foot positions
    left_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    right_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]

    # Get actual center of mass position (root_com_pose_w returns [pos, quat])
    com_pos = asset.data.root_com_pose_w[:, :3]  # (num_envs, 3) - extract position from [pos, quat]
    # print(com_pos)
    # Compute mid-point between feet
    foot_mid_pos = 0.5 * (left_foot_pos + right_foot_pos)

    # Compute XY error (ignore Z for standing balance)
    com_error = torch.norm(com_pos[:, :2] - foot_mid_pos[:, :2], dim=1)
    com_reward = torch.exp(-(com_error**2) / std**2)

    # Check if both feet are in contact
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 0.1
    both_feet_in_contact = torch.logical_and(contacts[:, 0], contacts[:, 1]).float()

    is_standing = torch.logical_or(
        command.command["cmd_countdown"] < 0.001, command.command["cmd_countdown"] > 0.5
    ).squeeze(-1)
    return com_reward * is_standing * both_feet_in_contact


def feet_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute feet velocity
    feet_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    feet_acc = feet_acc.norm(dim=-1).mean(dim=-1)
    # compute the reward
    return feet_acc


def contact_force_smoothness_l2_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Applies a penalty to the difference between the current and previous contact forces
    """
    # Verify we received exactly two bodies (the feet)
    # Get the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(contact_sensor.data.net_forces_w_history, torch.Tensor)

    # Indices (num envs, history, bodies, axes)
    forces = contact_sensor.data.net_forces_w_history[:, :2, sensor_cfg.body_ids, 2]
    # Indices (num_envs, history, bodies)
    deltas = forces[:, 0, :] - forces[:, 1, :]

    penalty = -torch.norm(deltas, dim=1)
    return penalty


def track_body_yaw_to_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track body yaw to commanded footstep yaw using exponential kernel.

    Args:
        env: The environment.
        command_name: Name of the command term.
        std: Standard deviation for exponential kernel.
        asset_cfg: Scene entity config for the body to track.

    Returns:
        Exponential reward for yaw tracking.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get commanded yaw from footstep (format: x, y, z, yaw)
    cmd_yaw = command.command["cmd_footstep"][:, 3]  # (num_envs,)

    # Get current body yaw
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)
    _, _, body_yaw = euler_xyz_from_quat(body_quat_w)

    # Compute yaw error
    yaw_error = torch.abs(body_yaw - cmd_yaw)

    return torch.exp(-(yaw_error**2) / std**2)


def track_joint_yaw_to_cmd_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track hip yaw joints to commanded footstep yaw using exponential kernel.

    This function rewards when the average of hip yaw joints matches the commanded yaw.
    Useful for tracking Left_Hip_Yaw and Right_Hip_Yaw joints.

    Args:
        env: The environment.
        command_name: Name of the command term.
        std: Standard deviation for exponential kernel.
        asset_cfg: Scene entity config for the joints to track (e.g., Left_Hip_Yaw, Right_Hip_Yaw).

    Returns:
        Exponential reward for joint yaw tracking.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get commanded yaw from footstep (format: x, y, z, yaw)
    cmd_yaw = command.command["cmd_footstep"][:, 3]  # (num_envs,)

    # Get current hip yaw joint positions
    joint_yaw = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (num_envs, num_joints)

    # Average the joint yaws (for left and right hip yaw)
    avg_joint_yaw = torch.mean(joint_yaw, dim=1)  # (num_envs,)

    # Compute yaw error
    yaw_error = torch.abs(avg_joint_yaw - cmd_yaw)

    return torch.exp(-(yaw_error**2) / std**2)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 2 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def foot_link_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize foot link velocities using L2 norm to encourage smoother, more controlled foot movements.

    This reward minimizes the linear velocities of the left and right foot links.
    NOTE: asset_cfg should contain both left and right foot links.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get linear velocities of foot links in world frame
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, num_feet, 3)
    # Compute L2 norm across all velocity components
    return torch.sum(torch.square(foot_velocities), dim=(1, 2))


def joint_performance_asymmetry_penalty(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Penalize asymmetry in joint tracking performance between left and right legs.

    Computes: abs(||ref_qpos_left - current_left|| - ||ref_qpos_right - current_right||)
    This encourages both legs to perform equally well in tracking their reference positions.

    Args:
        env: The environment.
        left_cfg: Scene entity config for left leg joints.
        right_cfg: Scene entity config for right leg joints.
        command_name: Name of the command term containing reference joint positions.

    Returns:
        Absolute difference in tracking error between left and right legs.
    """
    asset: Articulation = env.scene[left_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)

    # Get joint index mapping
    joint_indices, _ = get_joint_ref_indices()
    joint_indices = joint_indices.to(command.command["q"].device)

    # Get current joint positions for left and right legs
    current_left = asset.data.joint_pos[:, left_cfg.joint_ids]  # (num_envs, num_left_joints)
    current_right = asset.data.joint_pos[:, right_cfg.joint_ids]  # (num_envs, num_right_joints)

    # Get reference joint positions for left and right legs
    ref_left = command.command["q"][:, joint_indices[left_cfg.joint_ids]]  # (num_envs, num_left_joints)
    ref_right = command.command["q"][:, joint_indices[right_cfg.joint_ids]]  # (num_envs, num_right_joints)

    # Compute L2 tracking error for each leg
    error_left = torch.norm(current_left - ref_left, dim=1)  # (num_envs,)
    error_right = torch.norm(current_right - ref_right, dim=1)  # (num_envs,)

    # Return absolute difference in performance
    return torch.abs(error_left - error_right)


def feet_air_time_if_base_moves(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # (num_envs, num_feet) True if foot just made contact.
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    assert contact_sensor.data.last_air_time is not None

    # Time spent (in s) in the air before the last contact.
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    base_vel = env.scene["robot"].data.root_vel_w  # no reward for zero base vel
    reward *= torch.norm(base_vel[:, :2], dim=1) > 0.1

    return reward


# def cost_of_transport(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     ee_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """Compute the cost of transport (energy per distance) during stepping.

# Only accumulates energy and distance when countdown > 0 (during active stepping).
# Distance is tracked by swing foot displacement.
# CoT = Total Energy / Total Distance

# Returns:
#     Cost of transport value. Returns 0 when not enough distance traveled yet.

# NOTE: asset_cfg must be left foot first, right foot second.
# """
# asset: Articulation = env.scene[asset_cfg.name]
# ee: Articulation = env.scene[ee_cfg.name]
# command: FootTrackCommand = env.command_manager.get_term(command_name)

# foot_indicator = command.command['cmd_stance'].squeeze(-1).long()
# swing_id = torch.where(foot_indicator == 0, ee_cfg.body_ids[1], ee_cfg.body_ids[0])
# env_ids = torch.arange(asset.num_instances, device=asset.device)
# swing_pos_w = ee.data.body_pos_w[env_ids, swing_id]

# if not hasattr(env, '_cot_total_energy'):
#     env._cot_total_energy = torch.zeros(env.num_envs, device=env.device)
#     env._cot_total_distance = torch.zeros(env.num_envs, device=env.device)
#     env._cot_prev_swing_pos = swing_pos_w.clone()

# is_stepping = (command.command['cmd_countdown'] > 0).squeeze(-1)
# power = torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
# swing_displacement = torch.norm(swing_pos_w[:, :2] - env._cot_prev_swing_pos[:, :2], dim=1)

# energy = power * env.step_dt
# env._cot_total_energy[is_stepping] += energy[is_stepping]
# env._cot_total_distance[is_stepping] += swing_displacement[is_stepping]
# env._cot_prev_swing_pos = swing_pos_w.clone()

# just_finished = (command.command['cmd_countdown'] == 0).squeeze(-1)
# env._cot_total_energy[just_finished] = 0.0
# env._cot_total_distance[just_finished] = 0.0

# cot = env._cot_total_energy / torch.clamp(env._cot_total_distance, min=1e-3)
# print(cot)
# return cot
