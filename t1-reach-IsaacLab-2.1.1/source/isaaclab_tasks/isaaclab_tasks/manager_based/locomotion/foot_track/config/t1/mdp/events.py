from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from ..terms.commands import FootTrackCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_to_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_offset_range: tuple[float, float] = (0.0, 0.0),
    heading_randomization: bool = True,
):
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: FootTrackCommand = env.command_manager.get_term(command_name)
    command_term.resample_command(env_ids)
    # Get joint references (last 12 joints)
    joint_pos_ref = command_term.command['q'][env_ids, -12:]
    # Add random offset of 0.1 rad to each joint position reference
    joint_pos_ref = joint_pos_ref + torch.empty_like(joint_pos_ref, device=env.device).uniform_(*pos_offset_range)
    joint_vel_ref = command_term.command['qd'][env_ids, -12:]

    T_wbase_ref = command_term.command['T_wbase'][env_ids]

    robot_base_pos = T_wbase_ref[:, 0:3] + env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, 0.15], device=env.device)
    robot_base_quat = T_wbase_ref[:, 3:7]

    if heading_randomization:
        # Randomize the robot base heading uniformly over [-pi, pi]
        heading = torch.empty(len(env_ids), device=env.device).uniform_(-torch.pi, torch.pi)
        heading_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(len(env_ids), 3)
        heading_quat = math_utils.quat_from_angle_axis(heading, heading_axis)
        robot_base_quat = math_utils.quat_mul(heading_quat, robot_base_quat)
    else:
        # Use identity quaternion (no rotation) when heading randomization is disabled
        heading_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(len(env_ids), 4)

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
    base_lin_vel = command_term.v_b_ref[env_ids, :3]
    base_ang_vel = command_term.v_b_ref[env_ids, 3:]
    base_lin_vel = math_utils.quat_apply(heading_quat, base_lin_vel)
    base_ang_vel = math_utils.quat_apply(heading_quat, base_ang_vel)
    rotated_base_vel = torch.cat([base_lin_vel, base_ang_vel], dim=-1)
    asset.write_root_velocity_to_sim(rotated_base_vel, env_ids=env_ids)
    asset.set_joint_position_target(
        joint_pos_ref,
        asset_cfg.joint_ids,
        env_ids=env_ids,
    )

def reset_boxes_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stair_box: SceneEntityCfg = SceneEntityCfg("stair_box"),
    left_box: SceneEntityCfg = SceneEntityCfg("left_box"),
    right_box: SceneEntityCfg = SceneEntityCfg("right_box"),
    base_height: float = 0.10,
):
    """Reset box positions using neutral foot positions and swing end from command data.

    Places boxes at:
    - left_box: lf_neutral (left foot neutral position)
    - right_box: rf_neutral (right foot neutral position)
    - stair_box: swing_end (swing end position)

    Box heights are scaled based on the z-coordinate of the neutral positions.
    """
    from ..terms.commands import FootTrackMotionTypeCommand

    stair_cfg: RigidObject = env.scene[stair_box.name]
    left_cfg: RigidObject = env.scene[left_box.name]
    right_cfg: RigidObject = env.scene[right_box.name]
    command_term: FootTrackMotionTypeCommand = env.command_manager.get_term(command_name)

    # Get neutral positions from command (these are per-trajectory values)
    lf_neutral = command_term.lf_neutral[env_ids]
    rf_neutral = command_term.rf_neutral[env_ids]
    swing_end = command_term.swing_end[env_ids]

    # Default orientation (no rotation)
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    # Set left box position - center at half the neutral height
    left_positions = env.scene.env_origins[env_ids].clone()
    left_positions[:, 0] += lf_neutral[:, 0]
    left_positions[:, 2] += lf_neutral[:, 2] - 0.005
    left_orientations = default_quat.expand(len(env_ids), 4)
    left_transforms = torch.cat([left_positions, left_orientations], dim=-1)
    left_cfg.write_root_pose_to_sim(left_transforms, env_ids=env_ids)

    # Set right box position - center at half the neutral height
    right_positions = env.scene.env_origins[env_ids].clone()
    right_positions[:, 0] += rf_neutral[:, 0]
    right_positions[:, 1] += rf_neutral[:, 1]
    right_positions[:, 2] += rf_neutral[:, 2] - 0.005
    right_orientations = default_quat.expand(len(env_ids), 4)
    right_transforms = torch.cat([right_positions, right_orientations], dim=-1)
    right_cfg.write_root_pose_to_sim(right_transforms, env_ids=env_ids)

    # Set stair box position - center at half the swing_end height
    stair_positions = env.scene.env_origins[env_ids].clone()
    stair_positions[:, 0] += swing_end[:, 0]
    stair_positions[:, 1] += swing_end[:, 1]
    stair_positions[:, 2] += swing_end[:, 2] - 0.005
    stair_orientations = default_quat.expand(len(env_ids), 4)
    stair_transforms = torch.cat([stair_positions, stair_orientations], dim=-1)
    stair_cfg.write_root_pose_to_sim(stair_transforms, env_ids=env_ids)
    # print(left_positions[:, 2], right_positions[:, 2], stair_positions[:, 2])
    # Reset velocities to zero for kinematic bodies (prevents drift)
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    left_cfg.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    right_cfg.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    stair_cfg.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
def reset_to_reference_stair(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: FootTrackCommand = env.command_manager.get_term(command_name)
    command_term.resample_command(env_ids)
    # Get joint references (last 12 joints)
    joint_pos_ref = command_term.command['q'][env_ids, -12:]
    joint_vel_ref = command_term.command['qd'][env_ids, -12:]
    T_wbase_ref = command_term.command['T_wbase'][env_ids]
    robot_base_pos = T_wbase_ref[:, 0:3] + env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, 0.02], device=env.device)
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
    # Reset root velocity
    base_lin_vel = command_term.v_b_ref[env_ids, :3]
    base_ang_vel = command_term.v_b_ref[env_ids, 3:]
    base_vel = torch.cat([base_lin_vel, base_ang_vel], dim=-1)
    asset.write_root_velocity_to_sim(base_vel, env_ids=env_ids)
    # Set joint position targets for PD controllers
    asset.set_joint_position_target(
        joint_pos_ref,
        asset_cfg.joint_ids,
        env_ids=env_ids,
    )
    asset.update(dt=0.0)
    # print(asset.data.root_com_vel_w)