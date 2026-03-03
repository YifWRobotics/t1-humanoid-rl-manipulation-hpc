# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp as T1_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.booster import T1_CFG  # isort: skip
from .terms import curriculums as T1_curriculum
from .terms import commands_cfg as T1_commands_cfg
from .terms import events as T1_events


@configclass
class T1Observations:
    """Observation specifications for the MDP."""

    policy: T1_mdp.PolicyCfg = T1_mdp.PolicyCfg()
    critic: T1_mdp.CriticCfg = T1_mdp.CriticCfg()


@configclass
class T1Actions:
    """Action terms for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(  # type: ignore
    #     asset_name="robot",
    #     joint_names=[
    #             "AAHead_yaw",
    #             "Head_pitch",
    #             "Left_Shoulder_Pitch",
    #             "Left_Shoulder_Roll",
    #             "Left_Elbow_Pitch",
    #             "Left_Elbow_Yaw",
    #             "Left_Wrist_Pitch",
    #             "Left_Wrist_Yaw",
    #             "Left_Hand_Roll",
    #             # "left_Link1",
    #             # "left_Link11",
    #             # "left_Link2",
    #             # "left_Link22",
    #             "Right_Shoulder_Pitch",
    #             "Right_Shoulder_Roll",
    #             "Right_Elbow_Pitch",
    #             "Right_Elbow_Yaw",
    #             "Right_Wrist_Pitch",
    #             "Right_Wrist_Yaw",
    #             "Right_Hand_Roll",
    #             # "right_Link1",
    #             # "right_Link11",
    #             # "right_Link2",
    #             # "right_Link22",
    #             "Waist",
    #             "Left_Hip_Pitch",
    #             "Left_Hip_Roll",
    #             "Left_Hip_Yaw",
    #             "Left_Knee_Pitch",
    #             "Left_Ankle_Pitch",
    #             "Left_Ankle_Roll",
    #             "Right_Hip_Pitch",
    #             "Right_Hip_Roll",
    #             "Right_Hip_Yaw",
    #             "Right_Knee_Pitch",
    #             "Right_Ankle_Pitch",
    #             "Right_Ankle_Roll",
    #     ],
    #     scale={
    #         "AAHead_yaw": 0.0,
    #         "Head_pitch": 0.0,
    #         "Left_Shoulder_Pitch": 0.0,
    #         "Left_Shoulder_Roll": 0.0,
    #         "Left_Elbow_Pitch": 0.0,
    #         "Left_Elbow_Yaw": 0.0,
    #         "Left_Wrist_Pitch": 0.0,
    #         "Left_Wrist_Yaw": 0.0,
    #         "Left_Hand_Roll": 0.0,
    #         # "left_Link1": 0.0,
    #         # "left_Link11": 0.0,
    #         # "left_Link2": 0.0,
    #         # "left_Link22": 0.0,
    #         "Right_Shoulder_Pitch": 0.0,
    #         "Right_Shoulder_Roll": 0.0,
    #         "Right_Elbow_Pitch": 0.0,
    #         "Right_Elbow_Yaw": 0.0,
    #         "Right_Wrist_Pitch": 0.0,
    #         "Right_Wrist_Yaw": 0.0,
    #         "Right_Hand_Roll": 0.0,
    #         # "right_Link1": 0.0,
    #         # "right_Link11": 0.0,
    #         # "right_Link2": 0.0,
    #         # "right_Link22": 0.0,
    #         "Waist": 0.0,
    #         "Left_Hip_Pitch": 1.0,
    #         "Left_Hip_Roll": 1.0,
    #         "Left_Hip_Yaw": 1.0,
    #         "Left_Knee_Pitch": 1.0,
    #         "Left_Ankle_Pitch": 0.0,   # constrained ankle pitch
    #         "Left_Ankle_Roll": 0.0,   # constrained ankle roll
    #         "Right_Hip_Pitch": 1.0,
    #         "Right_Hip_Roll": 1.0,
    #         "Right_Hip_Yaw": 1.0,
    #         "Right_Knee_Pitch": 1.0,
    #         "Right_Ankle_Pitch": 0.0,   # constrained ankle pitch
    #         "Right_Ankle_Roll": 0.0,   # constrained ankle roll
    #     },
    #     offset={
    #         "Head_pitch": 0.0,
    #         "AAHead_yaw": 0.0,
    #         "Left_Shoulder_Pitch": 0.0,
    #         "Left_Shoulder_Roll": -1.57,
    #         "Left_Elbow_Pitch": 0.0,
    #         "Left_Elbow_Yaw": -1.57,
    #         "Left_Wrist_Pitch": 0.0,
    #         "Left_Wrist_Yaw": 0.0,
    #         "Left_Hand_Roll": 0.0,
    #         # "left_Link1": 0.0,
    #         # "left_Link11": 0.0,
    #         # "left_Link2": 0.0,
    #         # "left_Link22": 0.0,
    #         "Right_Shoulder_Pitch": 0.0,
    #         "Right_Shoulder_Roll": 1.57,
    #         "Right_Elbow_Pitch": 0.0,
    #         "Right_Elbow_Yaw": 1.57,
    #         "Right_Wrist_Pitch": 0.0,
    #         "Right_Wrist_Yaw": 0.0,
    #         "Right_Hand_Roll": 0.0,
    #         # "right_Link1": 0.0,
    #         # "right_Link11": 0.0,
    #         # "right_Link2": 0.0,
    #         # "right_Link22": 0.0,
    #         "Waist": 0.0,
    #         "Left_Hip_Pitch": -0.1,
    #         "Left_Hip_Roll": 0.0,
    #         "Left_Hip_Yaw": 0.0,
    #         "Left_Knee_Pitch": 0.2,
    #         "Left_Ankle_Pitch": -0.1,
    #         "Left_Ankle_Roll": 0.0,
    #         "Right_Hip_Pitch": -0.1,
    #         "Right_Hip_Roll": 0.0,
    #         "Right_Hip_Yaw": 0.0,
    #         "Right_Knee_Pitch": 0.2,
    #         "Right_Ankle_Pitch": -0.1,
    #         "Right_Ankle_Roll": 0.0,
    #     },
    #     preserve_order=True,
    #     use_default_offset=False,
    # )

    lower_joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
        joint_names=[
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
        scale={
            "Waist": 0.0,
            "Left_Hip_Pitch": 1.0,
            "Left_Hip_Roll": 1.0,
            "Left_Hip_Yaw": 1.0,
            "Left_Knee_Pitch": 1.0,
            "Left_Ankle_Pitch": 1.0,   # constrained ankle pitch
            "Left_Ankle_Roll": 1.0,   # constrained ankle roll
            "Right_Hip_Pitch": 1.0,
            "Right_Hip_Roll": 1.0,
            "Right_Hip_Yaw": 1.0,
            "Right_Knee_Pitch": 1.0,
            "Right_Ankle_Pitch": 1.0,   # constrained ankle pitch
            "Right_Ankle_Roll": 1.0,   # constrained ankle roll
        },
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class T1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    # This seems useless. We can try remove this.
    alive = RewTerm(func=mdp.is_alive, weight=0.25)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.4},
    )

    track_lin_vel_xy_exp_finer = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.4},
    )

    track_ang_vel_z_exp = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.3}
    )

    track_ang_vel_z_exp_finer = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.3}
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "target_height": 0.68,
        }
    )

    base_height_exp = RewTerm(
        func=T1_mdp.base_height_exp,
        weight=0.5,
        params={
            "target_height": 0.68,
            "std": 0.25
        }
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    flat_orientation_exp = RewTerm(
        func=T1_mdp.flat_orientation_exp,
        weight=5.0,
        params={
            "std": 0.3,
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_torque_limits = RewTerm(
        func=T1_mdp.joint_torque_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
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
            "soft_ratio": 1.0},
    )

    base_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1)

    # Penalize deviation from default of the joints that are not essential for locomotion
    reward_penalty_hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw"
                ]
            ),
        },
    )

    # joint_deviation_hip_roll = RewTerm(
    #     func=T1_mdp.reward_penalty_joint_deviation_hip_roll,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_Hip_Roll",
    #             ],
    #             preserve_order=True,
    #         ),
    #     },
    # )
    # joint_deviation_hip_pitch = RewTerm(
    #     func=T1_mdp.reward_penalty_joint_deviation_hip_pitch,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_Hip_Pitch",
    #             ],
    #             preserve_order=True,
    #         ),
    #     },
    # )
    reward_penalty_pose = RewTerm(
        func=T1_mdp.reward_penalty_pose,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    ".*_Knee_Pitch",
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True,
            ),
        },
    )

    reward_penalty_stand_still_joint_vel = RewTerm(
        func=T1_mdp.stand_still_joint_vel_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    ".*_Knee_Pitch",
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True,
            ),
        },
    )

    energy_efficiency = RewTerm(
        func=T1_mdp.power,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    ".*_Knee_Pitch",
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True
            )
        },
    )
# joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #             joint_names=[
    #                 ".*_Shoulder_Pitch",
    #                 ".*_Shoulder_Roll",
    #                 ".*_Elbow_Pitch",
    #                 ".*_Elbow_Yaw",
    #                 ".*_Wrist_Pitch",
    #                 ".*_Wrist_Yaw",
    #                 ".*_Hand_Roll",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_Link1",
    #                 ".*_Link11",
    #                 ".*_Link2",
    #                 ".*_Link22",
    #             ],
    #         )
    #     },
    # )

    # Rewards the agent for having feet in the air (e.g., walking, running, not standing still).
    feet_air_time = RewTerm(
        # feet_air_time is for swing phase. Whereas feet_air_time_positive_biped is for both phase longer.
        func=vel_mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,  # swing phase encourage up tp 0.4s
        },
    )

    # Penalizes the agent for sliding feet while in contact (i.e., wants stepping, not dragging feet)
    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    feet_swing = RewTerm(
        func=T1_mdp.feet_swing,
        weight=0.5,
        params={
            "swing_period": 0.2,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True
            ),
        },
    )

    feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_yaw_mean = RewTerm(
        func=mdp.reward_feet_yaw_mean,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_roll = RewTerm(
        func=mdp.feet_roll,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # feet_stumble = RewTerm(
    #     func=T1_mdp.feet_stumble,
    #     weight=-10.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["left_foot_link", "right_foot_link"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    feet_pitch_diff = RewTerm(
        func=mdp.reward_feet_pitch_diff,
        weight=-1.0,
        params={
            "std": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    feet_pitch = RewTerm(
        func=T1_mdp.reward_feet_pitch,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    foot_distance = RewTerm(
        func=T1_mdp.reward_foot_distance,
        weight=-0.5,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    feet_contact_number = RewTerm(
        func=T1_mdp.reward_feet_contact_number,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "swing_period": 0.2,
            "pos_rw": 1.0,
            "neg_rw": -0.3,
        },
    )

    feet_stance_velocity = RewTerm(
        func=T1_mdp.feet_stance_vel,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "swing_period": 0.2
        },
    )

    track_foot_height = RewTerm(
        func=T1_mdp.track_foot_height,
        weight=1,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    foot_clearance = RewTerm(
        func=T1_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "target_height": 0.1,
            "std": 0.5,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link"),
        },
    )

    # Maybe separate left and right, each have one ref tracking reward.
    quadratic_reference = RewTerm(
        func=T1_mdp.reward_quadratic_reference,
        weight=3.0,
        params={
            "phase_freq_hz": 1.0 / 1.2,
            "swing_period_ratio": 0.20,
            "lf_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link"),
            "foot_max_height": 0.15,
            "tracking_sigma": 0.05
        }
    )

    # Another alternative: encourage foot height in swing phase.
    stepfunc_reference = RewTerm(
        func=T1_mdp.reward_stepfunc_reference,
        weight=3.0,
        params={
            "swing_period_ratio": 0.20,
            "lf_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link"),
            "foot_max_height": 0.20,
            "tracking_sigma": 0.025
        }
    )

    com_stability = RewTerm(func=T1_mdp.reward_com_stability, weight=1.0)
    foot_pos_standing = RewTerm(
        func=T1_mdp.reward_foot_pos_during_stand,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "std": 0.15
        }
    )
    linvel_angvel_coupling_penalty = RewTerm(
        func=T1_mdp.reward_penalty_linvel_angvel_coupling,
        params={
            "command_name": "base_velocity"
        }
    )
    feet_liftoff = RewTerm(
        func=T1_mdp.reward_foot_height_during_swing,
        params={
            "lf_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link"),
            "swing_period": 0.2
        }
    )


@configclass
class T1Events:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    add_end_effector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link", "right_hand_link"]),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
            ),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_and_target_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.4, 1.6),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot")
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.65, 0.68), "yaw": (-3.14, 3.14)},
             "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "z": (-0.4, 0.4),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.4, 0.4),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3)
            }
        },
    )

    reset_robot_upper_joints_from_limits = EventTerm(
        func=mdp.reset_robot_upper_joints_from_limits,
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

    # reset_robot_lower_joints_from_reference = EventTerm(
    #     func=T1_events.reset_from_trajectory_reference,
    #     mode="reset",
    #     params={
    #         "trajectory_npz": "source/isaaclab_assets/data/reset_traj.npz",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "Waist",
    #                 "Left_Hip_Pitch",
    #                 "Left_Hip_Roll",
    #                 "Left_Hip_Yaw",
    #                 "Left_Knee_Pitch",
    #                 "Left_Ankle_Pitch",
    #                 "Left_Ankle_Roll",
    #                 "Right_Hip_Pitch",
    #                 "Right_Hip_Roll",
    #                 "Right_Hip_Yaw",
    #                 "Right_Knee_Pitch",
    #                 "Right_Ankle_Pitch",
    #                 "Right_Ankle_Roll"
    #             ],
    #             preserve_order=True
    #         )
    #     }
    # )


@configclass
class T1Terminations:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[".*Trunk", ".*Hip.*", ".*Shank.*"]
        ),
            "threshold": 1.0},
    )

    base_too_low = DoneTerm(
        func=T1_mdp.root_height_below_minimum_adaptive,  # type: ignore
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*foot.*",
                ],
            ),
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.0
        }
    )


@configclass
class T1Curriculums:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)

    # command_range_alive_gt = CurrTerm(
    #     func=T1_curriculum.modify_command_range_on_greater_than_reward,
    #     params={
    #         "term_name": "base_velocity",
    #         "ranges": T1_commands_cfg.UniformVelocityCommandPlsCfg.Ranges(
    #             lin_vel_x=(-1.0, 1.0),
    #             lin_vel_y=(-0.5, 0.5),
    #             ang_vel_z=(-1.5, 1.5)
    #         ),
    #         "metric_name": "alive",
    #         "metric_threshold": 8.5
    #     },
    # )

    command_curriculum_x = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_x",
            "start_step": 2000 * 24,
            "end_step": 4000 * 24,
        }
    )
    command_curriculum_y = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_y",
            "start_step": 2000 * 24,
            "end_step": 4000 * 24,
        }
    )
    command_curriculum_z = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "ang_vel_Z",
            "start_step": 4000 * 24,
            "end_step": 6000 * 24,
        }
    )

    # track_lin_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "track_lin_vel_xy_exp", "weight": 15, "num_steps": 10000 * 24}
    # )

    # track_ang_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "track_ang_vel_z_exp", "weight": 10, "num_steps": 10000 * 24}
    # )

    # track_lin_vel_alive_gt = CurrTerm(
    #     func=T1_curriculum.modify_reward_weight_on_greater_than_reward,
    #     params={
    #         "term_name": "track_lin_vel_xy_exp_finer",
    #         "weight": 20,
    #         "metric_name": "alive",
    #         "metric_threshold": 8.5
    #     }
    # )

    # track_ang_vel_alive_gt = CurrTerm(
    #     func=T1_curriculum.modify_reward_weight_on_greater_than_reward,
    #     params={
    #         "term_name": "track_ang_vel_z_exp_finer",
    #         "weight": 15,
    #         "metric_name": "alive",
    #         "metric_threshold": 8.5
    #     }
    # )

    action_rate_penalty = CurrTerm(
        func=T1_curriculum.linearly_alter_weight,
        params={
            "term_name": "action_rate_l2",
            "start_step": 2000 * 24,
            "end_step": 4000 * 24,
        }
    )

    bad_orientation_alive_gt = CurrTerm(
        func=T1_curriculum.modify_done_term_on_greater_than_reward,
        params={
            "term_name": "bad_orientation",
            "term_param_name": "limit_angle",
            "value": 0.5,
            "metric_name": "alive",
            "metric_threshold": 8.5,
        }
    )

    push_curriculum = CurrTerm(
        func=T1_curriculum.modify_event_on_greater_than_reward,
        params={
            "term_name": "push_robot",
            "term_param_name": "velocity_range",
            "value": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.5, 0.5)},
            "metric_name": "alive",
            "metric_threshold": 8.5,
        }
    )


@configclass
class T1Commands:
    base_velocity = T1_commands_cfg.UniformVelocityCommandPlsCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        heading_command=False,
        debug_vis=True,
        ranges=T1_commands_cfg.UniformVelocityCommandPlsCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
        ),
    )

    phase = T1_commands_cfg.PhaseCommandCfg(
        resampling_time_range=(1e10, 1e10),  # resample on command resample
        phase_frequency_hz_range=[1.0, 2.0]
    )


@configclass
class T1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: T1Observations = T1Observations()
    rewards: T1Rewards = T1Rewards()
    terminations: T1Terminations = T1Terminations()
    events: T1Events = T1Events()
    actions: T1Actions = T1Actions()
    curriculum: T1Curriculums = T1Curriculums()
    commands: T1Commands = T1Commands()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_agregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregatge_pairs_capacity = 2**22

        # Scene
        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"


# ------------------------------------------------------------#
        # Randomization
        self.events.push_robot.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.add_end_effector_mass.params["mass_distribution_params"] = (0.0, 1.0)
        self.events.scale_actuator_gains.params["stiffness_distribution_params"] = (0.90, 1.10)
        self.events.scale_actuator_gains.params["damping_distribution_params"] = (0.90, 1.10)
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-0.3, 0.3),
            },
        }
        # self.events.reset_robot_lower_joints_from_reference.params["rel_default"] = 0.6


# ------------------------------------------------------------#
        # Rewards
        # Base height seem wrong.
        self.rewards.base_height_l2.params["target_height"] = 0.68
        self.rewards.base_height_l2.weight = -50.0
        self.rewards.base_height_exp.params["target_height"] = 0.68
        self.rewards.base_height_exp.weight = 15.0
        self.rewards.quadratic_reference.weight = 0.0
        self.rewards.stepfunc_reference.weight = 30.0
# ------------------------------------------------------------#
        self.rewards.undesired_contacts = None  # type: ignore
        self.rewards.alive.weight = 0.0
        self.rewards.termination_penalty.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 25.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_lin_vel_xy_exp_finer.weight = 12.5
        self.rewards.track_lin_vel_xy_exp_finer.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.weight = 10.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp_finer.weight = 5.0
        self.rewards.track_ang_vel_z_exp_finer.params["std"] = 0.15
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.flat_orientation_exp.weight = 10.0
        # WE may want to limit the policy output directly.
        self.rewards.dof_pos_limits.weight = -4.0
        self.rewards.dof_torque_limits.weight = -1.5
# ------------------------------------------------------------#
        # ZG: Rewards in this section should be fixed. Either removed or fixed.
        self.rewards.lin_vel_z_l2.weight = -2.0  # TODO: This is decreasing, which is wrong.
        self.rewards.ang_vel_xy_l2.weight = -1.0  # TODO: This is decreasing, which is wrong.
        self.rewards.reward_penalty_pose.weight = -2.0  # TODO: This is decreasing, which is wrong.
        self.rewards.reward_penalty_hip_deviation.weight = -0.5  # TODO: This is decreasing, which is wrong.
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*Hip.*",
                ".*Knee.*",
            ],
        )
        self.rewards.dof_torques_l2.weight = -1.5e-4
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*Hip.*",
                ".*Knee.*",
            ],
        )
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_vel_l2.weight = -1.0e-4
        self.rewards.base_acc_l2.weight = -1.0e-4
        self.rewards.action_rate_l2.weight = -0.35
        self.rewards.energy_efficiency.weight = -1.0e-4
# ------------------------------------------------------------#
        self.rewards.feet_swing.weight = 0.0
        self.rewards.feet_roll.weight = -5.0
        self.rewards.feet_pitch.weight = 0.0
        self.rewards.feet_pitch_diff.weight = -0.0
        self.rewards.feet_yaw_diff.weight = -0.0
        self.rewards.feet_yaw_mean.weight = -1.0
        self.rewards.foot_distance.weight = -0.0  # Does not matter. Just remove for now.
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_air_time.weight = 20.0
        self.rewards.feet_liftoff.weight = 10.0
        self.rewards.foot_clearance.weight = 0.0
        self.rewards.feet_contact_number.weight = 10.0
        self.rewards.feet_stance_velocity.weight = -5.0
        self.rewards.track_foot_height.weight = 0.0
        self.rewards.linvel_angvel_coupling_penalty.weight = 0.0
        # self.rewards.feet_stumble.weight = -10.0
        # self.rewards.joint_deviation_hip_roll.weight = -3.0
        # self.rewards.joint_deviation_hip_pitch.weight = -3.0
# ------------------------------------------------------------#
        # Standing specific rewards.
        # Regulate foot location wrt base during standing.
        self.rewards.foot_pos_standing.weight = 0.0
        # Add standing conditioned CoM in foot middle reward.
        self.rewards.com_stability.weight = 0.0
        self.rewards.reward_penalty_stand_still_joint_vel.weight = -1.5

# ------------------------------------------------------------#
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_rotonly_envs = 0.05


# ------------------------------------------------------------#
        # Curriculum
        # self.curriculum.command_range = None
        # self.curriculum.command_range.params["ranges"] = mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0),
        #     lin_vel_y=(-1.0, 1.0),
        #     ang_vel_z=(-1.0, 1.0),
        # )
        # self.curriculum.command_range.params["num_steps"] = 10000 * 20
        # self.curriculum.track_lin_vel.params['weight'] = 20
        # self.curriculum.track_ang_vel.params['weight'] = 15
        # self.curriculum.track_lin_vel.params["num_steps"] = 10000 * 20
        # self.curriculum.track_ang_vel.params["num_steps"] = 10000 * 20
        self.curriculum.action_rate_penalty.params["start_weight"] = -0.35
        self.curriculum.action_rate_penalty.params["end_weight"] = -0.50
        # self.curriculum.command_range_alive_gt.params["ranges"] = T1_commands_cfg.UniformVelocityCommandPlsCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0),
        #     lin_vel_y=(-0.8, 0.8),
        #     ang_vel_z=(-2.0, 2.0)
        # )
        alive_weight = self.rewards.alive.weight
        # self.curriculum.command_range_alive_gt.params["metric_threshold"] = alive_weight * 0.80
        self.curriculum.push_curriculum.params["value"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.5, 0.5)}
        self.curriculum.push_curriculum.params["metric_threshold"] = alive_weight * 0.90
        # self.curriculum.track_lin_vel_alive_gt.params["metric_threshold"] = alive_weight * 0.65
        # self.curriculum.track_ang_vel_alive_gt.params["metric_threshold"] = alive_weight * 0.65
        self.curriculum.bad_orientation_alive_gt.params["metric_threshold"] = alive_weight * 0.65
        self.curriculum.command_curriculum_x.params["start_value"] = 0.5
        self.curriculum.command_curriculum_x.params["end_value"] = 1.0
        self.curriculum.command_curriculum_y.params["start_value"] = 0.3
        self.curriculum.command_curriculum_y.params["end_value"] = 0.7
        self.curriculum.command_curriculum_z.params["start_value"] = 1.0
        self.curriculum.command_curriculum_z.params["end_value"] = 2.0


@configclass
class T1RoughEnvCfg_PLAY(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10000.0  # play non-stop
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
