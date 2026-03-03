from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def pose_7d_to_9d(pose_7d: torch.Tensor) -> torch.Tensor:
    """Convert 7D pose (position + quaternion) to 9D pose (position + 6D rotation).

    Args:
        pose_7d: Input poses of shape (n, 7) with format [x, y, z, qw, qx, qy, qz]

    Returns:
        Output poses of shape (n, 9) with format [x, y, z, r1, r2, r3, r4, r5, r6]
        where r1-r6 are the first two columns of the rotation matrix flattened
    """
    # extract position and quaternion
    position = pose_7d[..., :3]  # (n, 3)
    quaternion = pose_7d[..., 3:]  # (n, 4) - [qw, qx, qy, qz]

    # convert quaternion to rotation matrix
    rot_matrix = math_utils.matrix_from_quat(quaternion)  # (n, 3, 3)
    # take first two columns and flatten to 6D
    rot_6d = rot_matrix[..., :2].reshape(pose_7d.shape[0], 6)  # (n, 6)
    # concatenate position and 6D rotation
    pose_9d = torch.cat([position, rot_6d], dim=-1)  # (n, 9)

    return pose_9d

def pose_7d_to_12d(pose_7d: torch.Tensor) -> torch.Tensor:
    """Convert 7D pose (position + quaternion) to 12D pose (position + 12D rotation).

    Args:
        pose_7d: Input poses of shape (n, 7) with format [x, y, z, qw, qx, qy, qz]

    Returns:
        Output poses of shape (n, 12) with format [x, y, z, r1, r2, r3, r4, r5, r6, r7, r8, r9]
        where r1-r6 are the first two columns of the rotation matrix flattened
    """
    # extract position and quaternion
    position = pose_7d[..., :3]  # (n, 3)
    quaternion = pose_7d[..., 3:]  # (n, 4) - [qw, qx, qy, qz]

    # convert quaternion to rotation matrix
    rot_matrix = math_utils.matrix_from_quat(quaternion)  # (n, 3, 3)
    # take first two columns and flatten to 6D
    rot_9d = rot_matrix.reshape(pose_7d.shape[0], 9)  # (n, 9)
    # concatenate position and 6D rotation
    pose_12d = torch.cat([position, rot_9d], dim=-1)  # (n, 12)

    return pose_12d

def body_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get body pose in base frame.

    Returns the body poses relative to the robot base frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        Body poses in base frame [num_envs, 9*num_bodies] with format [x, y, z, r1-r6] per body.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get body poses in world frame
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_bodies, 4)

    # Get base pose in world frame
    base_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    base_quat_w = asset.data.root_quat_w  # (num_envs, 4)

    # Transform body poses to base frame
    base_quat_w_inv = math_utils.quat_inv(base_quat_w)  # (num_envs, 4)

    # Transform positions: subtract base position, then rotate to base frame
    body_pos_b = body_pos_w - base_pos_w.unsqueeze(1)  # (num_envs, num_bodies, 3)
    body_pos_b = math_utils.quat_apply(base_quat_w_inv.unsqueeze(1), body_pos_b)  # (num_envs, num_bodies, 3)

    # Transform orientations: apply inverse base rotation
    body_quat_b = math_utils.quat_mul(base_quat_w_inv.unsqueeze(1), body_quat_w)  # (num_envs, num_bodies, 4)

    # Combine into 7D pose
    pose_7d = torch.cat([body_pos_b, body_quat_b], dim=-1)  # (num_envs, num_bodies, 7)

    # Convert to 9D and flatten
    return pose_7d_to_9d(pose_7d.reshape(env.num_envs, -1))

def body_pose_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose_7d = asset.data.body_pose_w[:, asset_cfg.body_ids]
    return pose_7d_to_9d(pose_7d.reshape(env.num_envs, -1))

def body_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_vel_w[:, asset_cfg.body_ids]


def body_acc_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_acc_w[:, asset_cfg.body_ids]


def joint_vel_rel_w_zero(env: ManagerBasedEnv, zero_idxs: list[int] = [], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    res = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    if len(zero_idxs) != 0:
        res[:, zero_idxs] = 0.0

    return res
