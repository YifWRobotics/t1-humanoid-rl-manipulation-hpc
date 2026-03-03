from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple
import math

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_mul, quat_inv, quat_error_magnitude
from .T1_observations import should_stand, should_walk, conditioned_get_phase
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_feet_contact_number(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, swing_period: float, pos_rw: float, neg_rw: float
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    phase = conditioned_get_phase(env)

    # compute expected contact pattern based on phase
    swing_phase_1 = 0.25  # when left foot should be in swing
    swing_phase_2 = 0.75  # when right foot should be in swing
    swing_tolerance = 0.5 * swing_period

    # check if in swing phases
    in_swing_1 = torch.abs(phase - swing_phase_1) <= swing_tolerance
    in_swing_2 = torch.abs(phase - swing_phase_2) <= swing_tolerance

    # expected contact pattern: [left_foot, right_foot]
    expected_contacts = torch.ones_like(contacts)  # default [true, true]
    expected_contacts[:, 0] = torch.where(in_swing_1, False, True)
    expected_contacts[:, 1] = torch.where(in_swing_2, False, True)

    reward = torch.where(contacts == expected_contacts, pos_rw, neg_rw)
    return torch.mean(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = 0.0626  # recorded from the default position
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)

    foot_z_target_error = torch.square(
        (
            asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
            - (target_height + offset).repeat(1, 2)
        ).clip(max=0.0)
    )

    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_velocity_tanh * foot_z_target_error
    return torch.exp(-torch.sum(reward, dim=1) / std)


@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return (a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0)


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    phase = conditioned_get_phase(env)

    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1

    if (torch.sum(contacts == stance_mask) > torch.sum(contacts == mask_2)):
        swing_mask = 1 - stance_mask
    else:
        swing_mask = 1 - mask_2

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod)  # + offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.abs(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)

    return reward


def feet_swing(
    env: ManagerBasedRLEnv,
    swing_period: float,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    phase = conditioned_get_phase(env)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        > 1.0
    )

    left_swing = torch.abs(phase - 0.25) < 0.5 * swing_period
    right_swing = torch.abs(phase - 0.75) < 0.5 * swing_period
    reward = ((left_swing & ~contacts[:, 0]).float() + (right_swing & ~contacts[:, 1]).float()) * should_walk(env)

    return reward


def quadratic_ref(phi, offset: float, swing_period_ratio: float, foot_max_height: float):
    rsq = swing_period_ratio ** 2
    a = (-4 * foot_max_height) / rsq
    b = 8 * foot_max_height * offset / rsq
    c = (foot_max_height * rsq - 4 * foot_max_height * offset * offset) / rsq
    ref = a * phi * phi + b * phi + c
    ref[torch.abs(phi - offset) > 0.5 * swing_period_ratio] = 0.0  # foot on ground if not in swing period
    return ref


def stepfunc_ref(phi, offset: float, swing_period_ratio: float, foot_max_height: float):
    ref = torch.ones_like(phi) * foot_max_height
    ref[torch.abs(phi - offset) > 0.5 * swing_period_ratio] = 0.0  # foot on ground if not in swing period
    return ref


def reward_quadratic_reference(
    env: ManagerBasedRLEnv,
    phase_freq_hz: float,
    swing_period_ratio: float,
    lf_asset_cfg: SceneEntityCfg,
    rf_asset_cfg: SceneEntityCfg,
    foot_max_height: float = 0.15,
    tracking_sigma: float = 0.02
):
    phase = conditioned_get_phase(env)
    lf_foot_z = env.scene[lf_asset_cfg.name].data.body_pos_w[:, lf_asset_cfg.body_ids, 2].reshape(-1)
    lf_foot_zref = quadratic_ref(phase, 0.25, swing_period_ratio, foot_max_height)
    rf_foot_z = env.scene[rf_asset_cfg.name].data.body_pos_w[:, rf_asset_cfg.body_ids, 2].reshape(-1)
    rf_foot_zref = quadratic_ref(phase, 0.75, swing_period_ratio, foot_max_height)

    return torch.exp(-torch.abs(lf_foot_z - lf_foot_zref) / tracking_sigma) \
        + torch.exp(-torch.abs(rf_foot_z - rf_foot_zref) / tracking_sigma)


def reward_stepfunc_reference(
    env: ManagerBasedRLEnv,
    swing_period_ratio: float,
    lf_asset_cfg: SceneEntityCfg,
    rf_asset_cfg: SceneEntityCfg,
    foot_max_height: float = 0.15,
    tracking_sigma: float = 0.05
):
    phase = conditioned_get_phase(env)
    lf_foot_z = env.scene[lf_asset_cfg.name].data.body_pos_w[:, lf_asset_cfg.body_ids, 2].reshape(-1)
    lf_foot_zref = stepfunc_ref(phase, 0.25, swing_period_ratio, foot_max_height)
    rf_foot_z = env.scene[rf_asset_cfg.name].data.body_pos_w[:, rf_asset_cfg.body_ids, 2].reshape(-1)
    rf_foot_zref = stepfunc_ref(phase, 0.75, swing_period_ratio, foot_max_height)
    return torch.exp(-torch.abs(lf_foot_z - lf_foot_zref) / tracking_sigma) \
        + torch.exp(-torch.abs(rf_foot_z - rf_foot_zref) / tracking_sigma)


def reward_sine_reference(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        ref_angle: float = 0.5,
        double_stand_phase: float = 0.5
) -> torch.Tensor:
    phase = conditioned_get_phase(env)
    sin_pos = torch.sin(2 * torch.pi * phase)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()
    ref_joint_pos = torch.zeros_like(asset.data.joint_pos[:, asset_cfg.joint_ids])

    # left foot stance phase set to default joint pos
    sin_pos_l[sin_pos_l > 0] = 0
    ref_joint_pos[:, 0] = ref_angle * sin_pos_l * 1  # + asset.data.default_joint_pos[:, 1]
    ref_joint_pos[:, 3] = -ref_angle * sin_pos_l * 2  # + asset.data.default_joint_pos[:, 4]
    ref_joint_pos[:, 4] = ref_angle * sin_pos_l * 1  # + asset.data.default_joint_pos[:, 5]

    # right foot stance phase set to default joint pos
    sin_pos_r[sin_pos_r < 0] = 0
    ref_joint_pos[:, 6] = -ref_angle * sin_pos_r * 1  # + asset.data.default_joint_pos[:, 7]
    ref_joint_pos[:, 9] = ref_angle * sin_pos_r * 2  # + asset.data.default_joint_pos[:, 10]
    ref_joint_pos[:, 10] = -ref_angle * sin_pos_r * 1  # + asset.data.default_joint_pos[:, 11]

    # Double support phase
    ref_joint_pos[torch.abs(sin_pos) < double_stand_phase] = 0

    diff = joint_pos - ref_joint_pos
    return torch.exp(-2 * torch.norm(diff, dim=1))  # - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)


def reward_foot_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    reward = torch.clip(torch.abs(ref_dist - foot_dist), min=0.0, max=0.1)

    return reward


def reward_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    phase = conditioned_get_phase(env)
    T_not_done = torch.abs(phase - torch.round(phase)) > 0.04  # True if phase is not done
    T_is_done = torch.abs(phase - torch.round(phase)) <= 0.04  # True if phase is done

    error = torch.norm(env.cycle_torque_diff, dim=-1)
    reward = torch.exp(-error / std**2) * T_is_done
    env.cycle_torque_diff = (env.cycle_torque_diff
                             + torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[:6]]) - torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids[6:]]) * T_not_done.unsqueeze(1)
                             )
    return reward


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


def flat_orientation_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using exp kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.exp(-torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) / std**2)


def joint_torque_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint torque and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.computed_torque[:, asset_cfg.joint_ids])
        - asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    out_of_limits = out_of_limits.clip_(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [0, 1]
):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider.
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]

    # Get the body IDs to use
    feet_quat = entity.data.body_quat_w[:, asset_cfg.body_ids, :]
    # feet_quat = entity.data.body_quat_w[:, feet_index, :]
    original_shape = feet_quat.shape
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    # yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), \
        pitch.reshape(original_shape[0], -1), \
        yaw.reshape(original_shape[0], -1)


def reward_feet_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # feet_index: list[int] = [22, 23]
) -> torch.Tensor:
    # Calculate roll angles from quaternions for the feet
    # feet_index = asset_cfg.body_ids
    _, feet_pitch, _ = _feet_rpy(
        env,
        asset_cfg=asset_cfg,
        # feet_index=feet_index
    )
    return torch.sum(torch.square(feet_pitch), dim=-1) * should_stand(env)


def reward_penalty_joint_deviation_hip_roll(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err_jnt = asset.data.joint_pos - asset.data.default_joint_pos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt[:, asset_cfg.joint_ids]), dim=1)
    cost = err_jnt_sqsum
    return cost


def reward_penalty_joint_deviation_hip_pitch(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err_jnt = asset.data.joint_pos - asset.data.default_joint_pos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt[:, asset_cfg.joint_ids]), dim=1)
    cost = err_jnt_sqsum
    return cost


def reward_penalty_pose(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint deviations from default pose when command is small (stand still cost).

    Returns the sum of absolute joint deviations, scaled by whether the command norm is below threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err_jnt = asset.data.joint_pos - asset.data.default_joint_pos
    err_jnt_sqsum = torch.sum(torch.abs(err_jnt[:, asset_cfg.joint_ids]), dim=1)
    return (err_jnt_sqsum).reshape(-1)  # only have this cost when stand


def feet_stance_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, swing_period: float) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    phase = conditioned_get_phase(env)

    # compute expected contact pattern based on phase
    swing_phase_1 = 0.25  # when left foot should be in swing
    swing_phase_2 = 0.75  # when right foot should be in swing
    swing_tolerance = 0.5 * swing_period

    # check if in swing phases
    in_swing_1 = torch.abs(phase - swing_phase_1) <= swing_tolerance
    in_swing_2 = torch.abs(phase - swing_phase_2) <= swing_tolerance

    # get left and right foot velocity
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_1 = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]  # type: ignore
    vel_1_norm = torch.norm(vel_1, dim=-1)
    ang_1 = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    ang_1_norm = torch.norm(ang_1, dim=-1)
    vel_2 = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[1], :]  # type: ignore
    vel_2_norm = torch.norm(vel_2, dim=-1)
    ang_2 = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[1], :]
    ang_2_norm = torch.norm(ang_2, dim=-1)

    combined_error = (vel_1_norm + ang_1_norm) * ~in_swing_1 \
        + (vel_2_norm + ang_2_norm) * ~in_swing_2
    return combined_error


def power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for energy-efficient locomotion by minimizing power consumption."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    power = torch.maximum(joint_vel * joint_torque, torch.zeros_like(joint_vel))
    return torch.sum(power, dim=1)


def reward_com_stability(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.25) -> torch.Tensor:
    """Advanced COM stability considering support polygon area."""
    asset: RigidObject = env.scene[asset_cfg.name]
    com_pos = asset.data.root_pos_w[:, :2]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    # support_area = torch.abs(foot_pos[:, 0, 0] - foot_pos[:, 1, 0]) * \
    #     torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    support_center = torch.mean(foot_pos, dim=1)
    com_error = torch.norm(com_pos - support_center, dim=1)
    # normalized_error = com_error / (support_area + 1e-6)
    # return torch.exp(-normalized_error / std) * should_stand(env)
    return torch.exp(-com_error / std) * should_stand(env)


def reward_take_a_step_on_base_move_when_stand(
    env: ManagerBasedRLEnv,
    liftoff_threshold: 0.6,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: RigidObject = env.scene[asset_cfg.name]
    should_take_a_step = torch.norm(asset.data.body_vel_w, dim=1) > liftoff_threshold

    # when should take a step, strongly reward liftoff
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    reward = torch.count_nonzero(contacts, dim=1) == 1
    conditioned_reward = reward * should_take_a_step * should_stand(env)  # only have this reward when :1. standing, 2. the push is large enough
    return conditioned_reward


def reward_foot_height_during_swing(
    env: ManagerBasedRLEnv, lf_asset_cfg: SceneEntityCfg,
    rf_asset_cfg: SceneEntityCfg, swing_period: float
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    phase = conditioned_get_phase(env)

    # compute expected contact pattern based on phase
    swing_phase_1 = 0.25  # when left foot should be in swing
    swing_phase_2 = 0.75  # when right foot should be in swing
    swing_tolerance = 0.5 * swing_period

    # check if in swing phases
    in_swing_1 = torch.abs(phase - swing_phase_1) <= swing_tolerance
    in_swing_2 = torch.abs(phase - swing_phase_2) <= swing_tolerance

    lf_foot_z = env.scene[lf_asset_cfg.name].data.body_pos_w[:, lf_asset_cfg.body_ids, 2].reshape(-1)
    rf_foot_z = env.scene[rf_asset_cfg.name].data.body_pos_w[:, rf_asset_cfg.body_ids, 2].reshape(-1)

    return lf_foot_z * in_swing_1 + rf_foot_z * in_swing_2


def stand_still_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * should_stand(env)


def reward_foot_pos_during_stand(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25
) -> torch.Tensor:
    # initialize default foot positions if not already set
    # FIXME: temporarily used fix constants for foot position
    if not hasattr(env.reward_manager, "foot_pose_b_default"):
        env.reward_manager.lf_posxy_b_default = torch.tensor([0.0409, 0.1062], device=env.device)
        env.reward_manager.rf_posxy_b_default = torch.tensor([0.0409, -0.1062], device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]

    # get base position
    base_pos_w = asset.data.body_pos_w[:, 0]  # base is body 0

    # get foot positions and transform to base frame (xy only)
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_feet, 3)
    foot_pos_b = foot_pos_w - base_pos_w.unsqueeze(1)  # simple translation
    foot_posxy_b = foot_pos_b[..., :2]  # take xy only

    # compute errors from default positions
    lf_error = torch.norm(foot_posxy_b[:, 0] - env.reward_manager.lf_posxy_b_default, dim=-1)
    rf_error = torch.norm(foot_posxy_b[:, 1] - env.reward_manager.rf_posxy_b_default, dim=-1)

    # reward for being close to default positions
    reward = torch.exp(-(lf_error + rf_error) / std)
    return reward * should_stand(env)


def reward_penalty_linvel_angvel_coupling(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    velcmd = env.command_manager.get_command(command_name)
    no_linvel = torch.norm(velcmd[:, :2], dim=1) < 0.01
    is_rot_only = no_linvel * should_walk(env)

    asset: RigidObject = env.scene[asset_cfg.name]
    # penalize linear velocity of the robot that is rotating in place
    linvel_b_norm = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return linvel_b_norm * is_rot_only


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 4 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def velocity_tracking_x_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in x direction."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std)


def velocity_tracking_y_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in y direction."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1])

    return torch.exp(-lin_vel_error / std)


def velocity_tracking_yaw_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std)


def orientation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def joint_torque_tiredness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids] / asset.data.joint_effort_limits), dim=1)


def base_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_acc_l2 = torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)
    ang_acc_l2 = torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)
    return lin_acc_l2 + ang_acc_l2


def feet_swing_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, swing_height: float):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - swing_height) * ~contacts
    return torch.sum(pos_error, dim=(1)) * should_walk(env)
