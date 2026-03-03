# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Code transferred from BoosterGym.
Link: https://github.com/BoosterRobotics/booster_gym

Target Config: https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml
Environment: https://github.com/BoosterRobotics/booster_gym/blob/main/envs/t1.py
"""


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

from isaaclab_assets.robots.booster import T1_CFG  # isort: skip
from .terms import commands_cfg as T1_commands_cfg
from .terms import commands as T1_commands
from .terms import curriculums as T1_curriculum

import isaaclab.terrains as terrain_gen


@configclass
class T1Observations:
    policy: T1_mdp.PolicyCfg = T1_mdp.BoosterGymPolicyCfg()
    critic: T1_mdp.CriticCfg = T1_mdp.BoosterGymCriticCfg()


@configclass
class T1Actions:
    """Action terms for the MDP."""
    lower_joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
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
        scale={
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
class T1Events:
    """Configuration for events."""
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # BoosterGym uses gaussian but we use uniform.
        mode="reset",
        params={
            "position_range": (0., 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # This is additive.
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # This is uniform where as BoosterGym uses gaussian.
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1)
            }  # on start
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-15.0, 15.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
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
            ),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "num_buckets": 64,
            "restitution_range": (0.0, 0.0),
            # "make_consistent": True
        },
    )  # BoosterGym has compliance, not sure how to implement this.

    add_base_com_xyz = EventTerm(
        func=mdp.randomize_rigid_body_com_class,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.0, 0.0)},
        },
    )

    # We omit other link mass and com randomization.
    scale_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass_class,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",  # Uniform distribution
            "recompute_inertia": True,  # Recompute inertia tensors
        },
    )


@configclass
class T1Commands:
    base_velocity = T1_commands_cfg.UniformVelocityCommandPlsCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        heading_command=False,
        debug_vis=True,
        ranges=T1_commands_cfg.UniformVelocityCommandPlsCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
        ),
        rel_standing_envs=0.15,
        rel_rotonly_envs=0.2,
    )

    phase = T1_commands_cfg.PhaseCommandCfg(
        resampling_time_range=(1e10, 1e10),  # resample on command resample
        phase_frequency_hz_range=[0.8, 2.0]
    )


@configclass
class T1Terminations:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[".*Trunk", ".*Hip.*", ".*Shank.*"]
    #         ),
    #         "threshold": 10.0
    #     },
    # )

    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,  # type: ignore
        params={
            "minimum_height": 0.50,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.5
        }
    )


@configclass
class T1Rewards:
    # --------------------------------------------------
    # Locomotion Related
    # --------------------------------------------------
    survival = RewTerm(
        func=mdp.is_alive,
        weight=1.0
    )
    velocity_tracking_x = RewTerm(
        func=T1_mdp.velocity_tracking_x_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.15
        }
    )
    velocity_tracking_y = RewTerm(
        func=T1_mdp.velocity_tracking_y_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.15
        }
    )
    velocity_tracking_yaw = RewTerm(
        func=T1_mdp.velocity_tracking_yaw_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.15
        }
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-30.0,
        params={
            "target_height": 0.68,
        }
    )
    orientation = RewTerm(
        func=T1_mdp.orientation,
        weight=-7.0
    )

    # --------------------------------------------------
    # Regularization
    # --------------------------------------------------
    torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0
    )
    torque_tiredness = RewTerm(
        func=T1_mdp.joint_torque_tiredness,
        weight=0.0
    )
    power = RewTerm(
        func=T1_mdp.power,
        weight=-2e-4
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7
    )
    base_acc = RewTerm(
        func=T1_mdp.base_acc_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")},
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0005
    )
    joint_pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0
    )
    posture = RewTerm(
        func=T1_mdp.reward_penalty_pose,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    # ".*_Knee_Pitch", # don't penalize knee to allow lift off
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True,
            ),
        },
    )

    # --------------------------------------------------
    # Gait
    # --------------------------------------------------
    feet_slip = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link", preserve_order=True),
        },
    )
    feet_yaw = RewTerm(
        func=mdp.feet_yaw,
        weight=-1.0,
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
        weight=-1.0,
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
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    feet_distance = RewTerm(
        func=T1_mdp.reward_foot_distance,
        weight=-1.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    # feet_swing = RewTerm(
    #     func=T1_mdp.feet_swing,
    #     weight=1.0,  # BoosterGym uses 3.0. We reduce it to avoid foot lift.
    #     params={
    #         "swing_period": 0.20,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["left_foot_link", "right_foot_link"],
    #             preserve_order=True
    #         ),
    #     },
    # )
    feet_stepfunc_ref = RewTerm(
        func=T1_mdp.reward_stepfunc_reference,
        weight=3.0,
        params={
            "swing_period_ratio": 0.20,
            "lf_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link"),
            "foot_max_height": 0.1,
            "tracking_sigma": 0.020
        }
    )
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"]
            ),
            "command_name": "base_velocity",
            "threshold": 1.0 / 1.2 * 0.2
        },
    )
    feet_stance_linvel = RewTerm(
        func=T1_mdp.feet_stance_vel,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "swing_period": 0.2
        },
    )


@configclass
class T1Curriculums:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)

    action_rate_currilcum = CurrTerm(
        func=T1_curriculum.linearly_alter_weight,
        params={
            "term_name": "action_rate_l2",
            "start_weight": -0.0005,
            "end_weight": -0.8,
            "start_step": 1000 * 24,
            "end_step": 6000 * 24
        }
    )

    command_curriculum_x = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_x",
            "start_step": 300 * 24,
            "end_step": 5000 * 24,
            "start_value": 0.7,
            "end_value": 1.2
        }
    )
    command_curriculum_y = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "lin_vel_y",
            "start_step": 300 * 24,
            "end_step": 5000 * 24,
            "start_value": 0.3,
            "end_value": 0.6
        }
    )
    command_curriculum_z = CurrTerm(
        func=T1_curriculum.linear_alter_command_param,
        params={
            "term_name": "base_velocity",
            "subterm_name": "ang_vel_z",
            "start_step": 300 * 24,
            "end_step": 6000 * 24,
            "start_value": 0.5,
            "end_value": 2.5
        }
    )

    push_curriculum = CurrTerm(
        func=T1_curriculum.modify_event_on_greater_than_reward,
        params={
            "term_name": "push_robot",
            "term_param_name": "velocity_range",
            "value": {
                "x": (-1.1, 1.1),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "yaw": (-0.3, 0.3)
            },
            "metric_name": "survival",
            "metric_threshold": 1.0 * 0.80,
        }  # on first time reward > 75% survival
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
        self.scene.env_spacing = 2.0
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_agregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregatge_pairs_capacity = 2**22
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.enable_ccd = True

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator.sub_terrains = {
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.6
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.4, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.15
            )
        }
        self.scene.terrain.max_init_terrain_level = 5

        # Scene
        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"
        self.observations.policy.enable_corruption = True
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.debug_vis = True
        self.episode_length_s = 30.0


@configclass
class T1RoughEnvCfg_PLAY(T1RoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 10
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 1000000.0  # non-stop playing

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.3
        self.commands.base_velocity.rel_rotonly_envs = 0.1

        self.commands.base_velocity.debug_vis = True

        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"


@configclass
class T1FlatEnvCfg(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.enable_corruption = True
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.debug_vis = True
        self.episode_length_s = 30.0


@configclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 10
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 1000000.0  # non-stop playing

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.rel_rotonly_envs = 0.8
        self.curriculum = None

        self.commands.base_velocity.debug_vis = True

        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"
