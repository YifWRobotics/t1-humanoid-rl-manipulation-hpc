
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


def body_pose_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    attr_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:asset_cfg.body_ids will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7*num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz]. Output is
            stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    command = getattr(env.command_manager.get_term(command_name), attr_name)
    pose = asset.data.body_state_w[:, asset_cfg.body_ids, :7]

    pose[..., :3] = pose[..., :3] - command[..., :3].unsqueeze(1)
    pose[..., 3:] = math_utils.quat_mul(math_utils.quat_inv(command[..., 3:7].unsqueeze(1)), pose[..., 3:])
    return pose_7d_to_9d(pose.reshape(env.num_envs, -1))
    # return pose.reshape(env.num_envs, -1)


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
