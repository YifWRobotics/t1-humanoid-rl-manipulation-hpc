# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Sequence
import torch
from isaaclab.assets.articulation.articulation import Articulation
import math
from dataclasses import MISSING
import os
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.math import matrix_from_quat

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.foot_track.foot_track_env_cfg import FootTrackEnvCfg

import isaaclab_tasks.manager_based.locomotion.foot_track.config.t1.mdp as t1_mdp
import isaaclab_tasks.manager_based.locomotion.foot_track.config.t1.terms as t1_terms
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.booster import T1_CFG  # isort: skip
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Environment configuration
##


@configclass
class T1FootTrackSceneCfg(InteractiveSceneCfg):
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


def get_countdown_command(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.cmd_countdown


def get_foot_indicator_command(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.cmd_foot_indicator


def get_stsw_pos_command(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.ref_T_stsw


def get_cmd_footstep(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.cmd_footstep


def get_T_blf_ref(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.ref_T_blf


def get_T_brf_ref(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    return term.ref_T_brf


def get_q_ref(env: ManagerBasedRLEnv):
    term: t1_terms.FootTrackCommand = env.command_manager.get_term("foot_pos_cmd")
    # Dataset has 36 joints, but robot has 29 joints - return only first 29
    return term.ref_q[:, -29:]


@configclass
class T1FootTrackObservations:
    """Observation specifications for the MDP."""

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        footstep_command = ObsTerm(func=get_cmd_footstep)
        countdown_command = ObsTerm(func=get_countdown_command)
        foot_indicator_command = ObsTerm(func=get_foot_indicator_command)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=t1_mdp.joint_vel_rel_w_zero,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                ),
                "zero_idxs": [4, 5, 10, 11],  # zero out ankle velocity
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        # Additional privileged terms for the critic
        joint_ref = ObsTerm(func=get_q_ref)
        left_foot_ref = ObsTerm(func=get_T_blf_ref)
        right_foot_ref = ObsTerm(func=get_T_brf_ref)
        left_foot_pose_current = ObsTerm(func=t1_mdp.body_pose_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link"])})
        right_foot_pose_current = ObsTerm(func=t1_mdp.body_pose_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_foot_link"])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        foot_vel = ObsTerm(func=t1_mdp.body_vel_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        footstep_command = ObsTerm(func=get_cmd_footstep)
        countdown_command = ObsTerm(func=get_countdown_command)
        foot_indicator_command = ObsTerm(func=get_foot_indicator_command)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class T1FootTrackCommands:
    """Command terms for the MDP."""

    foot_pos_cmd = t1_terms.FootTrackCommandCfg(
        foot_track_data_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "stepping_dataset_11.16_rand.npz"),
        asset_name="robot",
        resample_ratio_range=(0.0, 0.5),
        rel_random=0.0,
        resampling_time_range=(1e9, 1e9),
        left_foot_name="left_foot_link",
        right_foot_name="right_foot_link",
    )


@configclass
class T1VelocityToFootTrackCommands:
    """Command terms for the MDP."""

    foot_pos_cmd = t1_terms.VelocityToFootTrackCommandCfg(
        foot_track_data_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "stepping_dataset_11.11_rand.npz"),
        asset_name="robot",
        resample_ratio_range=(0.0, 0.0),
        rel_random=0.0,
        resampling_time_range=(1e9, 1e9),
        left_foot_name="left_foot_link",
        right_foot_name="right_foot_link",
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=(-0, 0),
        ),
    )


@configclass
class T1FootTrackTerminations:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})


def set_default_target(env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]

    # correct indexing: first select environments, then joints
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]

    asset.write_joint_position_to_sim(joint_pos, asset_cfg.joint_ids, env_ids)
    asset.set_joint_position_target(joint_pos, asset_cfg.joint_ids, env_ids)


@configclass
class T1FootTrackEvents:
    """Configuration for the foot track end-effector pose tracking environment."""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add",
        },
    )
    add_end_effector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.5),
            "dynamic_friction_range": (0.1, 1.5),
            "num_buckets": 64,
            "restitution_range": (0.0, 0.0),
            "make_consistent": True,
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
            ),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # "Waist",
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
            ),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
        },
    )
    reset_to_reference = EventTerm(
        func=t1_mdp.reset_to_reference,
        mode="reset",
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
            ),
        },
    )
    set_default_joint_pos_target_upper_body = EventTerm(
        func=set_default_target,
        mode="reset",
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
                ],
            )
        },
    )
    # reset_robot_upper_joints_from_limits = EventTerm(
    #     func=mdp.reset_robot_upper_joints_from_limits,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "AAHead_yaw",
    #                 "Head_pitch",
    #                 "Left_Shoulder_Pitch",
    #                 "Left_Shoulder_Roll",
    #                 "Left_Elbow_Pitch",
    #                 "Left_Elbow_Yaw",
    #                 "Left_Wrist_Pitch",
    #                 "Left_Wrist_Yaw",
    #                 "Left_Hand_Roll",
    #                 "Right_Shoulder_Pitch",
    #                 "Right_Shoulder_Roll",
    #                 "Right_Elbow_Pitch",
    #                 "Right_Elbow_Yaw",
    #                 "Right_Wrist_Pitch",
    #                 "Right_Wrist_Yaw",
    #                 "Right_Hand_Roll",
    #             ],
    #         )
    #     },
    # )
    # add_base_com_xyz = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
    #         "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.0, 0.0)},
    #     },
    # )
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,  # This is uniform where as BoosterGym uses gaussian.
    #     mode="interval",
    #     interval_range_s=(1.0, 2.0),
    #     params={
    #         "velocity_range": {
    #             "x": (-0.1, 0.1),
    #             "y": (-0.1, 0.1),
    #             "z": (-0.1, 0.1),
    #             "roll": (-0.1, 0.1),
    #             "pitch": (-0.1, 0.1),
    #             "yaw": (-0.1, 0.1),
    #         }  # on start
    #     },
    # )


@configclass
class T1FootTrackActions:
    foot_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            # "Waist", # No need to use waist
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
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class T1FootTrackRewards:
    # =============================================================================
    # Command Tracking
    # =============================================================================
    track_xy_exp = RewTerm(
        func=t1_mdp.track_pos_xy_exp,
        weight=3.0,
        params={
            "std": 0.05,
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True),
        },
    )
    track_xy_w_exp = RewTerm(
        func=t1_mdp.track_pos_xy_w_exp,
        weight=3.0,
        params={
            "std": 0.05,
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True),
        },
    )
    track_xy_w_exp_fine = RewTerm(
        func=t1_mdp.track_pos_xy_w_exp,
        weight=1.0,
        params={
            "std": 0.25,
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True),
        },
    )
    track_yaw_exp = RewTerm(
        func=t1_mdp.track_yaw_exp,
        weight=1.5,
        params={
            "std": 0.1,
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # =============================================================================
    # Reference Tracking
    # =============================================================================
    ref_pos_lf = RewTerm(
        func=t1_mdp.ref_pos_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "ref_T_blf",
            "std": 0.05,
        },
    )
    ref_yaw_lf = RewTerm(
        func=t1_mdp.ref_yaw_exp,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "ref_T_blf",
            "std": 0.10,
        },
    )
    ref_pos_rf = RewTerm(
        func=t1_mdp.ref_pos_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "ref_T_brf",
            "std": 0.10,
        },
    )
    ref_yaw_rf = RewTerm(
        func=t1_mdp.ref_yaw_exp,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "ref_T_brf",
            "std": 0.10,
        },
    )
    ref_linvel_b = RewTerm(
        func=t1_mdp.ref_linvel_exp,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "ref_v_b",
            "std": 0.20,
        },
    )

    ref_qpos = RewTerm(
        func=t1_mdp.joint_pos_tracking_exp,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # "Waist",
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
            ),
            "command_name": "foot_pos_cmd",
            "std": 0.30,
        },
    )
    ref_qvel = RewTerm(
        func=t1_mdp.joint_vel_tracking_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # "Waist",
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
            ),
            "command_name": "foot_pos_cmd",
            "std": 1.5,
        },
    )

    # =============================================================================
    # Regularization
    # =============================================================================
    survival = RewTerm(func=mdp.is_alive, weight=0.25)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    action_penalty = RewTerm(
        func=t1_mdp.action_countdown_penalty,
        weight=-0.1,
        params={
            "command_name": "foot_pos_cmd",
        },
    )
    # keep ankle joints near default pose for stability
    # hip_posture = RewTerm(
    #     func=t1_mdp.joint_posture,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_Hip_Pitch",
    #                 ".*_Hip_Roll",
    #                 ".*_Hip_Yaw",
    #             ],
    #             preserve_order=True,
    #         )
    #     },
    # )
    ankle_posture = RewTerm(
        func=t1_mdp.joint_posture,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True,
            )
        },
    )
    torque_limit = RewTerm(func=mdp.applied_torque_limits, weight=-1.0e-4)
    pos_limit = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    # base_height_range = RewTerm(
    #     func=t1_mdp.base_height_range,
    #     weight=-10.0,
    #     params={
    #         "min_height": 0.60,
    #         "max_height": 0.68,
    #     },
    # )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    base_orientation = RewTerm(
        func=t1_mdp.flat_link_orientation_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "std": 0.10,
        },
    )
    ankle_orientation = RewTerm(
        func=t1_mdp.flat_link_orientation_exp,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
            "std": 0.05,
        },
    )
    feet_slip = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link", preserve_order=True),
        },
    )
    feet_stumble = RewTerm(
        func=t1_mdp.feet_stumble,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link", preserve_order=True),
        },
    )
    feet_rotate = RewTerm(
        func=t1_mdp.feet_rotate,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link", preserve_order=True),
        },
    )
    # base_stability = RewTerm(
    #     func=t1_mdp.base_in_middle_during_standing,
    #     weight=5.0,
    #     params={
    #         "command_name": "foot_pos_cmd",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["left_foot_link", "right_foot_link"],
    #             preserve_order=True,
    #         ),
    #         "base_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
    #         "std": 0.5,
    #         "contact_sensor_cfg": SceneEntityCfg(
    #             "contact_forces", 
    #             body_names=["right_foot_link","left_foot_link"]
    #         ),
    #     },
    # )
    # =============================================================================
    # TASK SPACE REWARDS - End-effector position and trajectory tracking
    # =============================================================================
    correct_swing_foot_height = RewTerm(
        func=t1_mdp.correct_foot_height,
        weight=10.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "stance": False,
        },
    )
    correct_stance_foot_height = RewTerm(
        func=t1_mdp.correct_foot_height,
        weight=-10.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "stance": True,
        },
    )
    right_foot_contact = RewTerm(
        func=t1_mdp.correct_single_foot_contact,
        weight=-5,
        params={
            "command_name": "foot_pos_cmd",
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_foot_link"]),
            "foot_id": 1,
        },
    )
    left_foot_contact = RewTerm(
        func=t1_mdp.correct_single_foot_contact,
        weight=-5,
        params={
            "command_name": "foot_pos_cmd",
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link"]),
            "foot_id": 0,
        },
    )
    # feet_airtime = RewTerm(
    #     func=t1_mdp.feet_air_time,
    #     weight=10.0,
    #     params={
    #         "command_name": "foot_pos_cmd",
    #         "threshold": 0.7,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"], preserve_order=True),
    #     },
    # )
    left_feet_airttime = RewTerm(
        func=t1_mdp.feet_airtime_in_swing,
        weight=100.0,
        params={
            "command_name": "foot_pos_cmd",
            "threshold": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link"], preserve_order=True),
            "foot_id": 0,
        },
    )
    right_feet_airttime = RewTerm(
        func=t1_mdp.feet_airtime_in_swing,
        weight=100.0,
        params={
            "command_name": "foot_pos_cmd",
            "threshold": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_foot_link"], preserve_order=True),
            "foot_id": 1,
        },
    )
    # left_airtime_penalty = RewTerm(
    #     func=t1_mdp.feet_airtime_penalty,
    #     weight=-5.0,
    #     params={
    #         "command_name": "foot_pos_cmd",
    #         "min_air_time": 0.4,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link"], preserve_order=True),
    #         "foot_id": 0,
    #         "std": 0.05
    #     },
    # )
    # right_airtime_penalty = RewTerm(
    #     func=t1_mdp.feet_airtime_penalty,
    #     weight=-5.0,
    #     params={
    #         "command_name": "foot_pos_cmd",
    #         "min_air_time": 0.4,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_foot_link"], preserve_order=True),
    #         "foot_id": 1,
    #         "std": 0.05
    #     },
    # )
    left_feet_contacttime = RewTerm(
        func=t1_mdp.feet_contacttime_in_stance,
        weight=1.0,
        params={
            "command_name": "foot_pos_cmd",
            "threshold": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link"], preserve_order=True),
            "foot_id": 0,
        },
    )
    right_feet_contacttime = RewTerm(
        func=t1_mdp.feet_contacttime_in_stance,
        weight=1.0,
        params={
            "command_name": "foot_pos_cmd",
            "threshold": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_foot_link"], preserve_order=True),
            "foot_id": 1,
        },
    )
    swing_foot_height_left = RewTerm(
        func=t1_mdp.swing_foot_height_reward,
        weight=5.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link"], preserve_order=True),
            "target_height": 0.35, # from the dataset
            "std": 0.05,
            "foot_id": 0,
        },
    )
    swing_foot_height_right = RewTerm(
        func=t1_mdp.swing_foot_height_reward,
        weight=5.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_foot_link"], preserve_order=True),
            "target_height": 0.35, #from the dataset
            "std": 0.05,
            "foot_id": 1,
        },
    )

def linearly_alter_weight(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, start_weight: float, end_weight: float, start_step: int, end_step: int
) -> float:
    """Curriculum that linearly alters a reward weight between two steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        start_weight: The starting weight of the reward term.
        end_weight: The ending weight of the reward term.
        start_step: The step at which to start altering the weight.
        end_step: The step at which to end altering the weight.
    """
    if env.common_step_counter < start_step:
        weight = start_weight
    elif env.common_step_counter >= end_step:
        weight = end_weight
    else:
        alpha = (env.common_step_counter - start_step) / (end_step - start_step)
        weight = (1 - alpha) * start_weight + alpha * end_weight

    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight


def linearly_alter_std(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, start_std: float, end_std: float, start_step: int, end_step: int
) -> float:
    if env.common_step_counter < start_step:
        std = start_std
    elif env.common_step_counter >= end_step:
        std = end_std
    else:
        alpha = (env.common_step_counter - start_step) / (end_step - start_step)
        std = (1 - alpha) * start_std + alpha * end_std

    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.params["std"] = std
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return std


@configclass
class T1FootTrackCurriculum:
    """Curriculum terms for the MDP."""

    action_rate_currilcum = CurrTerm(
        func=linearly_alter_weight,
        params={"term_name": "action_rate", "start_weight": -0.001, "end_weight": -5.0, "start_step": 1000 * 24, "end_step": 6000 * 24},
    )
    joint_acc_penality = CurrTerm(
        func=linearly_alter_weight,
        params={"term_name": "joint_acc", "start_weight": -1e-7, "end_weight": -1e-5, "start_step": 1000 * 24, "end_step": 6000 * 24},
    )
    # joint_acc_curriculum = CurrTerm(
    #     func=linearly_alter_weight,
    #     params={"term_name": "joint_acc", "start_weight": -1e-7, "end_weight": -5e-6, "start_step": 1000 * 24, "end_step": 6000 * 24},
    # )
    # joint_power_curriculum = CurrTerm(
    #     func=linearly_alter_weight,
    #     params={"term_name": "joint_power", "start_weight": -1e-7, "end_weight": -1e-5, "start_step": 1000 * 24, "end_step": 6000 * 24},
    # )
    # feet_air_curriculum = CurrTerm(
    #     func=linearly_alter_weight,
    #     params={"term_name": "feet_airtime", "start_weight": 10, "end_weight": 15.0, "start_step": 1000 * 24, "end_step": 6000 * 24},
    # )
    qpos_curr = CurrTerm(
        func=linearly_alter_std,
        params={"term_name": "ref_qpos", "start_std": 0.8, "end_std": 0.3, "start_step": 1000 * 24, "end_step": 6000 * 24},
    )
    # slip_curriculum = CurrTerm(
    #     func=linearly_alter_weight,
    #     params={"term_name": "feet_slip", "start_weight": -10, "end_weight": -20.0, "start_step": 1000 * 24, "end_step": 6000 * 24},
    # )

@configclass
class T1FootTrackEnvCfg(FootTrackEnvCfg):
    scene: T1FootTrackSceneCfg = T1FootTrackSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: T1FootTrackObservations = T1FootTrackObservations()
    commands: T1FootTrackCommands = T1FootTrackCommands()
    terminations: T1FootTrackTerminations = T1FootTrackTerminations()
    curriculum: T1FootTrackCurriculum = T1FootTrackCurriculum()
    rewards: T1FootTrackRewards = T1FootTrackRewards()
    events: T1FootTrackEvents = T1FootTrackEvents()
    actions: T1FootTrackActions = T1FootTrackActions()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 3.0
        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 60000
        self.commands.foot_pos_cmd.debug_vis = True


@configclass
class T1FootTrackEnvCfg_PLAY(T1FootTrackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.episode_length_s = 10
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.foot_pos_cmd.debug_vis = True
        self.episode_length_s = 3.0
        self.commands.foot_pos_cmd.resample_ratio_range = (0.0, 0.0)
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"


@configclass
class T1VelocityToFootTrackEnvCfg_PLAY(T1FootTrackEnvCfg):
    commands: T1VelocityToFootTrackCommands = T1VelocityToFootTrackCommands()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.episode_length_s = 10
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.foot_pos_cmd.debug_vis = True
