from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from ..terms.commands import FootTrackCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_to_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot lower body joints to the reference state from the command term."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: FootTrackCommand = env.command_manager.get_term(command_name)
    root_states = asset.data.default_root_state[env_ids].clone()
    command_term.resample_command(env_ids)
    # Get joint references (last 12 joints)
    joint_pos_ref = command_term.command['q'][env_ids, -12:]
    joint_vel_ref = command_term.command['qd'][env_ids, -12:]
    # print(joint_pos_ref)

    T_wbase_ref = command_term.command['T_wbase'][env_ids]

    robot_base_pos = T_wbase_ref[:, 0:3] + env.scene.env_origins[env_ids]
    robot_base_quat = T_wbase_ref[:, 3:7]

    asset.write_joint_position_to_sim(
        joint_pos_ref,
        asset_cfg.joint_ids,
        env_ids=env_ids,
    )
    asset.write_joint_velocity_to_sim(
        joint_vel_ref,
        asset_cfg.joint_ids,
        env_ids=env_ids,
    )
    asset.write_root_pose_to_sim(
        torch.cat([robot_base_pos, robot_base_quat], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(command_term.v_b_ref[env_ids, :], env_ids=env_ids)
    asset.set_joint_position_target(
        joint_pos_ref,
        asset_cfg.joint_ids,
        env_ids=env_ids,
    )
