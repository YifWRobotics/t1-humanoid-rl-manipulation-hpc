"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors import ContactSensor

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from .T1_states import root_state_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def should_stand(env: ManagerBasedRLEnv, zero_threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) < zero_threshold


def should_walk(env: ManagerBasedRLEnv, zero_threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) >= zero_threshold


def conditioned_get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    phase = env.command_manager.get_command("phase")
    standing = should_stand(env)
    return torch.where(standing, torch.zeros_like(phase), phase)


def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = env.command_manager.get_command("phase")
    condition = should_walk(env).unsqueeze(1)  # shape: (num_envs, 1)
    return torch.cat(
        [
            torch.cos(2 * torch.pi * phase).unsqueeze(1) * condition,
            torch.sin(2 * torch.pi * phase).unsqueeze(1) * condition,
        ],
        dim=1,
    )


def body_acc_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_acc_w


def body_pose_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7*num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz]. Output is
            stacked horizontally per body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_poses = asset.data.body_state_w[:, asset_cfg.body_ids, :7]
    pose = asset.data.body_state_w[:, asset_cfg.body_ids, :7]

    pose[..., :3] = body_poses[..., :3] - pose[..., :3]
    pose[..., 3:] = math_utils.quat_mul(math_utils.quat_inv(body_poses[..., 3:7].unsqueeze(1)), pose[..., 3:])
    return pose.reshape(env.num_envs, -1)


def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_mass_scaled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the mass scale factor from the scale_base_mass event for the base body.

    This function returns the scale factor applied to the base body mass during
    the scale_base_mass event. The scale factor represents how much the mass was
    scaled from its default value.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The mass scale factor for the base body [num_envs, 1].
    """

    # Access the scale_base_mass event term from the event manager
    # We need to find the event term that uses randomize_rigid_body_mass_class
    scale_base_mass_term = None

    # Try to find the event term by looking through the startup mode terms
    # We'll use a more direct approach by checking if the event manager has the term
    try:
        # Get the event term configuration for scale_base_mass
        term_cfg = env.event_manager.get_term_cfg("scale_base_mass")
        if hasattr(term_cfg, 'func') and hasattr(term_cfg.func, '__name__'):
            if term_cfg.func.__name__ == 'randomize_rigid_body_mass_class':
                scale_base_mass_term = term_cfg.func
    except ValueError:
        # If scale_base_mass term is not found, try to find any mass randomization term
        for mode in ["startup", "reset", "interval"]:
            if mode in env.event_manager.active_terms:
                for term_name in env.event_manager.active_terms[mode]:
                    try:
                        term_cfg = env.event_manager.get_term_cfg(term_name)
                        if (hasattr(term_cfg, 'func') and hasattr(term_cfg.func, '__name__')
                                and term_cfg.func.__name__ == 'randomize_rigid_body_mass_class'):
                            scale_base_mass_term = term_cfg.func
                            break
                    except ValueError:
                        continue
                if scale_base_mass_term is not None:
                    break

    if scale_base_mass_term is None:
        # If no scale_base_mass event is found, return zeros
        return torch.zeros(env.num_envs, 1, device=env.device)

    # Get the mass noise (which for scale operation is scale_factor - 1.0)
    mass_noise = scale_base_mass_term.get_mass_noise_for_body(asset_cfg.body_ids[0])

    # Convert back to scale factor (add 1.0 to get the actual scale factor)
    scale_factor = mass_noise + 1.0

    return scale_factor.unsqueeze(-1)


def base_com_xyz_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the center of mass offset from the add_base_com_xyz event.

    This function returns the CoM offset that was applied by the add_base_com_xyz 
    event during initialization. It retrieves the stored noise values directly 
    from the event term instead of calculating the difference between current 
    and default CoM.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The CoM offset for the base body [num_envs, 3] (x, y, z).
    """
    # Access the add_base_com_xyz event term from the event manager
    # We need to find the event term that uses randomize_rigid_body_com_class
    add_base_com_xyz_term = None

    try:
        # Get the event term configuration for add_base_com_xyz
        term_cfg = env.event_manager.get_term_cfg("add_base_com_xyz")
        if hasattr(term_cfg, 'func'):
            # Check if it's a class instance (already instantiated)
            if hasattr(term_cfg.func, 'get_com_noise_for_body'):
                add_base_com_xyz_term = term_cfg.func
            # Check if it's a class that needs to be instantiated
            elif hasattr(term_cfg.func, '__name__') and term_cfg.func.__name__ == 'randomize_rigid_body_com_class':
                # Instantiate the class if it hasn't been instantiated yet
                add_base_com_xyz_term = term_cfg.func(cfg=term_cfg, env=env)
                # Update the term_cfg to store the instance
                term_cfg.func = add_base_com_xyz_term
    except ValueError:
        # If add_base_com_xyz term is not found, try to find any CoM randomization term
        for mode in ["startup", "reset", "interval"]:
            if mode in env.event_manager.active_terms:
                for term_name in env.event_manager.active_terms[mode]:
                    try:
                        term_cfg = env.event_manager.get_term_cfg(term_name)
                        if (hasattr(term_cfg, 'func')
                            and ((hasattr(term_cfg.func, 'get_com_noise_for_body')) or
                                 (hasattr(term_cfg.func, '__name__') and term_cfg.func.__name__ == 'randomize_rigid_body_com_class'))):
                            if hasattr(term_cfg.func, 'get_com_noise_for_body'):
                                add_base_com_xyz_term = term_cfg.func
                            else:
                                # Instantiate the class
                                add_base_com_xyz_term = term_cfg.func(cfg=term_cfg, env=env)
                                term_cfg.func = add_base_com_xyz_term
                            break
                    except ValueError:
                        continue
                if add_base_com_xyz_term is not None:
                    break

    if add_base_com_xyz_term is None:
        # If no add_base_com_xyz event is found, return zeros
        return torch.zeros(env.num_envs, 3, device=env.device)

    # Get the CoM noise values stored by the event term
    # The noise is stored as (num_envs, num_bodies, 3), we want the first body (base)
    com_noise = add_base_com_xyz_term.get_com_noise_for_body(asset_cfg.body_ids[0])

    return com_noise


@configclass
class CriticCfg(ObsGroup):
    """Observations for critic group."""

    # observation terms (order preserved)
    clock = ObsTerm(func=clock)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(func=mdp.last_action)
    # root_state_w = ObsTerm(func=root_state_w)
    # root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
    # root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
    # root_quat = ObsTerm(func=mdp.root_quat_w)
    # root_acc = ObsTerm(func=body_acc_w)
    # foot_pose = ObsTerm(
    #     func=body_pose_rel,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["left_foot", "right_foot"],
    #             preserve_order=True,
    #         )
    #     }
    # )

    # body_mass = ObsTerm(func=mdp.)
    # body_com = ObsTerm(func=mdp.)
    base_linear_velocity = ObsTerm(func=mdp.root_lin_vel_w)
    base_height = ObsTerm(func=base_height)
    push_wrench = ObsTerm(
        func=mdp.body_incoming_wrench,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["Trunk"]
            )
        }
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    clock = ObsTerm(
        func=clock,
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=1,
        noise=Gnoise(mean=0.0, std=0.1),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Gnoise(mean=0.0, std=0.01),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        scale=1,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        scale=1.0,
        noise=Gnoise(mean=0.0, std=0.02),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=0.1,
        noise=Gnoise(mean=0.0, std=0.05),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(func=mdp.last_action)
    # Potentially add foot position as observation.

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 8


@configclass
class BoosterGymPolicyCfg(ObsGroup):
    """Observations for BoosterGym group - matches t1.py observation structure."""

    # observation terms (order preserved)
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        scale=1.0,
        noise=Gnoise(mean=0.0, std=0.01),
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=1.0,
        noise=Gnoise(mean=0.0, std=0.1),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        scale=1.0,
        params={"command_name": "base_velocity"},
    )
    clock = ObsTerm(
        func=clock,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        scale=1.0,
        noise=Gnoise(mean=0.0, std=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=0.1,
        noise=Gnoise(mean=0.0, std=0.1),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(
        func=mdp.last_action,
        params={
            "action_name": "lower_joint_pos"
        }
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 8  # We use 8 but booster_gym only use 1


@configclass
class BoosterGymCriticCfg(ObsGroup):
    """Observations for BoosterGym critic group - includes privileged observations from t1.py."""

    # observation terms (order preserved)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    clock = ObsTerm(func=clock)
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
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
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(
        func=mdp.last_action,
        params={
            "action_name": "lower_joint_pos"
        }
    )
    
    # ------------------------------------------------------------#
    # Privileged observations
    base_mass_scaled = ObsTerm(
        func=base_mass_scaled,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["Trunk"]
            )
        }
    )  # size 1
    base_com_xyz_offset = ObsTerm(
        func=base_com_xyz_offset,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["Trunk"]
            )
        }
    )  # size 3
    root_lin_vel_b = ObsTerm(func=mdp.base_lin_vel)
    base_height = ObsTerm(func=base_height)  # TODO: compensate for terrain height for rough terrain.
    # push_wrench = ObsTerm( # TODO: this is zero if we push by setting velocity. BoosterGym does use force to push.
    #     func=mdp.body_incoming_wrench,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["Trunk"]
    #         )
    #     }
    # )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 8  # BoosterGym uses 1 instead of 8
