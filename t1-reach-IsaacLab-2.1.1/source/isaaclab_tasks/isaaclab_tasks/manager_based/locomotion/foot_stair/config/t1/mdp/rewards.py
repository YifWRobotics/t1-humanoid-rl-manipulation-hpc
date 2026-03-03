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
    joint_indices = joint_indices.to(command.ref_q.device)
    err_jnt = asset.data.joint_pos[:, asset_cfg.joint_ids] - command.ref_q[:, joint_indices[asset_cfg.joint_ids]]
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt), dim=1)
    is_standing = (command.cmd_countdown == 0.0).reshape(-1)
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
    joint_indices = joint_indices.to(term.ref_q.device)
    error = torch.norm(asset.data.joint_pos[:, asset_cfg.joint_ids] - term.ref_q[:, joint_indices[asset_cfg.joint_ids]], dim=1)
    return torch.exp(-(error**2) / std**2)


def joint_vel_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities that deviate from the reference."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    term: FootTrackCommand = env.command_manager.get_term(command_name)
    _, joint_indices_qd = get_joint_ref_indices()
    joint_indices_qd = joint_indices_qd.to(term.ref_qd.device)
    error = torch.norm(asset.data.joint_vel[:, asset_cfg.joint_ids] - term.ref_qd[:, joint_indices_qd[asset_cfg.joint_ids]], dim=1)
    return torch.exp(-(error**2) / std**2)


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
    return torch.exp(-(distance**2) / std**2)


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
    exp_ref_error = torch.exp(-ref_error / std**2)
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


def ref_pos_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    return torch.exp(-(torch.norm(curr_pos_w - ref_pos_w, dim=1) ** 2) / std**2)

def ref_pos_xy_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    return torch.exp(-(torch.norm(curr_pos_w[:,:2] - ref_pos_w[:,:2], dim=1) ** 2) / std**2)


def ref_yaw_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute yaw error between current body orientation and reference yaw.
    Reference format: (n, (x, y, z, yaw)) in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)

    # get current body orientation
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (n, 4)
    # extract yaw from current quaternion
    # curr_yaw = torch.atan2(2 * (curr_quat_w[:, 0] * curr_quat_w[:, 3] + curr_quat_w[:, 1] * curr_quat_w[:, 2]),
    #                        1 - 2 * (curr_quat_w[:, 2]**2 + curr_quat_w[:, 3]**2))
    _, _, curr_yaw = euler_xyz_from_quat(curr_quat_w)  # (n, 3) Euler angles

    # extract reference yaw (4th column)
    ref_yaw = ref[:, 3]  # (n,)

    # compute exp yaw tracking
    return torch.exp(-(torch.abs(curr_yaw - ref_yaw) ** 2) / std**2)


def ref_linvel_exp(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)[:, :3]
    curr_linvel = asset.data.root_lin_vel_b
    return torch.exp(-torch.norm(curr_linvel - ref, dim=1) / std)


def ref_pos_exp_w(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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


def ref_yaw_exp_w(env: ManagerBasedRLEnv, command_name: str, attr_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute yaw error between current body orientation and reference yaw in world frame.
    Reference format: (n, (x, y, z, qw, qx, qy, qz)) in world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ref = getattr(env.command_manager.get_term(command_name), attr_name)

    # get current body orientation in world frame
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (n, 4)
    # extract yaw from current quaternion
    curr_yaw = torch.atan2(
        2 * (curr_quat_w[:, 0] * curr_quat_w[:, 3] + curr_quat_w[:, 1] * curr_quat_w[:, 2]), 1 - 2 * (curr_quat_w[:, 2] ** 2 + curr_quat_w[:, 3] ** 2)
    )

    # extract reference quaternion (last 4 columns)
    ref_quat_w = ref[:, 3:7]  # (n, 4)
    # extract yaw from reference quaternion
    ref_yaw = torch.atan2(
        2 * (ref_quat_w[:, 0] * ref_quat_w[:, 3] + ref_quat_w[:, 1] * ref_quat_w[:, 2]), 1 - 2 * (ref_quat_w[:, 2] ** 2 + ref_quat_w[:, 3] ** 2)
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
    xy_error = torch.norm(pos_StSw[:, :2] - command.cmd_footstep[:, :2], dim=1)

    return torch.exp(-(xy_error**2) / std**2)


def track_pos_xy_w_exp(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    pos_WSt = asset.data.body_pos_w[env_ids, command.stance_id]  # stance foot pos, shape (n, 3)
    pos_WSw = asset.data.body_pos_w[env_ids, command.swing_id]  # swing foot pos, shape (n, 3)
    xy_errorSt = torch.norm(pos_WSt[:, :2] - command.T_wst[:, :2], dim=1)
    xy_errorSw = torch.norm(pos_WSw[:, :2] - command.T_wsw[:, :2], dim=1)
    return torch.exp(-(xy_errorSt**2 + xy_errorSw**2) / std**2)


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
    return torch.exp(-((yaw_stsw - command.cmd_footstep[:, 3]) ** 2) / std**2)


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
    stance_id = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == 0, 0, 1)
    swing_id = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == 0, 1, 0)

    batch_indices = torch.arange(contacts.shape[0], device=contacts.device)
    swing_in_air = ~contacts[batch_indices, swing_id]  # shape: (num_envs,)
    swing_on_ground = contacts[batch_indices, swing_id]  # shape: (num_envs,)
    stance_on_ground = contacts[batch_indices, stance_id]
    is_stepping = (command.cmd_countdown > 0.0).reshape(-1)
    is_standing = (command.cmd_countdown == 0.0).reshape(-1)
    correct_contact_during_stepping = (is_stepping & swing_in_air).float() + (is_stepping & stance_on_ground).float()
    correct_contact_during_standing = (is_standing & swing_on_ground).float() + (is_standing & stance_on_ground).float()

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
    contacts = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 0.1
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    # Detect if the foot is stance, if stance, id = 1 else 0
    is_stance = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == foot_id, 1, 0)

    batch_indices = torch.arange(contacts.shape[0], device=contacts.device)
    incorrect_during_standing = ~contacts[batch_indices, 0]
    incorrect_during_stepping = contacts[batch_indices, 0] ^ is_stance
    is_stepping = (command.cmd_countdown > 0.0).reshape(-1)
    is_standing = (command.cmd_countdown == 0.0).reshape(-1)
    incorrect_contact_during_stepping = (is_stepping & incorrect_during_stepping).float()
    incorrect_contact_during_standing = (is_standing & incorrect_during_standing).float()
    # print("foot id:", foot_id)
    # print("contacts:", contacts[:, 0])
    # print("is_stance:", is_stance)
    # print("is_stepping:", is_stepping)
    # print("is_standing:", is_standing)
    # print("correct_stepping:", correct_contact_during_stepping)
    # print("correct_standing:", correct_contact_during_standing)
    return incorrect_contact_during_stepping + incorrect_contact_during_standing


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
    stance_id = command.cmd_foot_indicator.squeeze(-1).long()
    height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].reshape(-1, 2)  # (num_envs, 2) # get height of two feet
    # Use batch indexing to select stance foot height
    batch_indices = torch.arange(height.shape[0], device=height.device)
    return torch.abs(height[batch_indices, stance_id])  # (num_envs,)


def swing_foot_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    swing_height = asset.data.body_pos_w[env_ids, command.swing_id, 2]
    height_error = torch.abs(swing_height - target_height)
    height_reward = torch.exp(-height_error / std)
    is_swing = (command.cmd_countdown > 0.0).squeeze(-1)
    return height_reward * is_swing


def correct_foot_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stance: bool,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    countdown = command.cmd_countdown.reshape(-1)
    foot_pos = asset.data.body_pos_w[env_ids, command.stance_id, 2] if stance else asset.data.body_pos_w[env_ids, command.swing_id, 2]
    return torch.abs(foot_pos) * (countdown != 0.0)


def feet_air_time(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for double support
    reward *= (command.cmd_countdown.reshape(-1)) != 0.0
    return reward

def feet_airtime_in_swing(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, foot_id: int) -> torch.Tensor:
    """Reward long air time for the correct swing foot during swing phase.

    This function rewards the agent for keeping the swing foot in the air for a longer duration
    during the swing phase, up to a specified threshold. Only the correct swing foot is rewarded.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids[0]]
    is_swing = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == foot_id, 0, 1).float()
    reward = torch.square(torch.clamp(air_time, max=threshold)) * is_swing
    reward *= (command.cmd_countdown.reshape(-1) != 0.0).float()
    return reward

def feet_contacttime_in_stance(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, foot_id:int) -> torch.Tensor:
    """Reward long contact time for the correct stance foot during standing phase.

    This function rewards the agent for keeping the stance foot in contact with the ground
    for a longer duration during the standing phase (countdown == 0), up to a specified threshold.
    Only the correct stance foot is rewarded.
    """
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids[0]]
    is_stance = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == foot_id, 1, 0).float()
    reward = torch.square(torch.clamp(contact_time, max=threshold)) * is_stance
    reward *= (command.cmd_countdown.reshape(-1) != 0.0).float()
    return reward

def feet_airtime_penalty(env, command_name: str, min_air_time: float, std: float, sensor_cfg: SceneEntityCfg, foot_id:int) -> torch.Tensor:
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids[0]]
    is_swing = torch.where(command.cmd_foot_indicator.squeeze(-1).long() == foot_id, 0, 1).float()
    # Penalize when air time is below threshold using exponential kernel
    error = torch.clamp(min_air_time - air_time, min=0.0)
    penalty = (1.0 - torch.exp(-(error**2) / std**2)) * is_swing
    penalty *= (command.cmd_countdown.reshape(-1)) > 0.0
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
    env_ids = torch.arange(asset.num_instances, device=asset.device)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    is_swing_foot = (command.cmd_swing_foot_indicator.squeeze(-1) == (1 - foot_id)).float()

    height_error = torch.clamp(foot_height - target_height, min=0.0)
    reward = torch.exp(-(height_error)**2 / std)
    is_swinging = (command.ccmd_countdown.reshape(-1)) > 0.0
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


def action_countdown_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    is_standing = (command.cmd_countdown == 0.0).reshape(-1)
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * is_standing

def base_in_middle_during_standing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    base_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward center of mass being in the middle of the two feet during standing phase.
    Requires both feet to be in contact.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command: FootTrackCommand = env.command_manager.get_term(command_name)
    left_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    right_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[1]]
    com: Articulation = env.scene[base_cfg.name]
    com_pos = com.data.body_pos_w[:, base_cfg.body_ids[0]]
    foot_mid_pos = 0.5 * (left_foot_pos + right_foot_pos)
    com_error = torch.norm(com_pos[:, :2] - foot_mid_pos[:, :2], dim=1)
    com_reward = torch.exp(-(com_error**2) / std**2)

    # Check if both feet are in contact
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 0.1
    both_feet_in_contact = (contacts[:, 0] & contacts[:, 1]).float()

    is_standing = (command.cmd_countdown == 0.0).squeeze(-1)
    return com_reward * is_standing * both_feet_in_contact


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 4 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


# def cost_of_transport(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     ee_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """Compute the cost of transport (energy per distance) during stepping.

#     Only accumulates energy and distance when countdown > 0 (during active stepping).
#     Distance is tracked by swing foot displacement.
#     CoT = Total Energy / Total Distance

#     Returns:
#         Cost of transport value. Returns 0 when not enough distance traveled yet.

#     NOTE: asset_cfg must be left foot first, right foot second.
#     """
#     asset: Articulation = env.scene[asset_cfg.name]
#     ee: Articulation = env.scene[ee_cfg.name]
#     command: FootTrackCommand = env.command_manager.get_term(command_name)

#     foot_indicator = command.cmd_foot_indicator.squeeze(-1).long()
#     swing_id = torch.where(foot_indicator == 0, ee_cfg.body_ids[1], ee_cfg.body_ids[0])
#     env_ids = torch.arange(asset.num_instances, device=asset.device)
#     swing_pos_w = ee.data.body_pos_w[env_ids, swing_id]

#     if not hasattr(env, '_cot_total_energy'):
#         env._cot_total_energy = torch.zeros(env.num_envs, device=env.device)
#         env._cot_total_distance = torch.zeros(env.num_envs, device=env.device)
#         env._cot_prev_swing_pos = swing_pos_w.clone()

#     is_stepping = (command.cmd_countdown > 0).squeeze(-1)
#     power = torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
#     swing_displacement = torch.norm(swing_pos_w[:, :2] - env._cot_prev_swing_pos[:, :2], dim=1)

#     energy = power * env.step_dt
#     env._cot_total_energy[is_stepping] += energy[is_stepping]
#     env._cot_total_distance[is_stepping] += swing_displacement[is_stepping]
#     env._cot_prev_swing_pos = swing_pos_w.clone()

#     just_finished = (command.cmd_countdown == 0).squeeze(-1)
#     env._cot_total_energy[just_finished] = 0.0
#     env._cot_total_distance[just_finished] = 0.0

#     cot = env._cot_total_energy / torch.clamp(env._cot_total_distance, min=1e-3)
#     return cot
