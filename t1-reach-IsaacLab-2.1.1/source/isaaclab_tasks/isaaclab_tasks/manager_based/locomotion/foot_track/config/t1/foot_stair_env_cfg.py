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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from .foot_pos_env_cfg import T1FootTrackSceneCfg, T1FootTrackEvents, T1FootTrackCommands, T1FootTrackEnvCfg

@configclass
class T1FootTrackStairSceneCfg(T1FootTrackSceneCfg):
    # long box as collidable static mesh
    stair_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stair_box",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.23, 0.75, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
                max_depenetration_velocity = 1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            copy_from_source=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )
    right_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/right_box",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.23, 0.75, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=0.0,
                max_depenetration_velocity = 1.0,
                max_angular_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            copy_from_source=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )
    left_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/left_box",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.23, 0.75, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
                max_depenetration_velocity = 1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            copy_from_source=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

@configclass
class T1FootTrackStairCommands:
    """Command terms for the MDP."""

    foot_pos_cmd = t1_terms.FootTrackMotionTypeCommandCfg(
        # foot_track_data_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "1227_LIP_step.npz"),
        foot_track_data_path="/home/uchiwuwu/walking_ik_gen/0118_walkup_v2.npz",
        # foot_track_data_path="/workspace/isaaclab/datasets/0118_walkup_v1.npz",
        asset_name="robot",
        resample_ratio_range=(0.0, 0.5),
        rel_random=0.0,
        resampling_time_range=(1e9, 1e9),
        left_foot_name="left_foot_link",
        right_foot_name="right_foot_link",
    )
def set_default_target(env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]

    # correct indexing: first select environments, then joints
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]

    asset.write_joint_position_to_sim(joint_pos, asset_cfg.joint_ids, env_ids)
    asset.set_joint_position_target(joint_pos, asset_cfg.joint_ids, env_ids)

@configclass
class T1FootTrackStairEvents:
    """Events for the stair environment with randomized stair position."""

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
            "mass_distribution_params": (-0.2, 0.2),
            "operation": "add",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.4, 1.1),
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
    reset_to_reference = EventTerm(
        func=t1_mdp.reset_to_reference_stair,
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

    reset_stair_position = EventTerm(
        func=t1_mdp.reset_boxes_position,
        mode="reset",
        params={
            "stair_box": SceneEntityCfg("stair_box"),
            "left_box": SceneEntityCfg("left_box"),
            "right_box": SceneEntityCfg("right_box"),
            "command_name": "foot_pos_cmd",
        },
    )
    reset_robot_upper_joints_by_offset_default = EventTerm(
        func=mdp.reset_robot_upper_joints_by_offset_default,
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
            ),
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    add_base_com_xyz = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.0, 0.0)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # This is uniform where as BoosterGym uses gaussian.
        mode="interval",
        interval_range_s=(0.4, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            }  # on start
        },
    )
@configclass
class T1FootTrackStairRewards:
    # =============================================================================
    # Reference Tracking
    # =============================================================================
    ref_pos_lf = RewTerm(
        func=t1_mdp.ref_pos_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "T_blf",
            "std": 0.05,
        },
    )
    ref_yaw_lf = RewTerm(
        func=t1_mdp.ref_yaw_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "T_blf",
            "std": 0.20,
        },
    )
    ref_pos_rf = RewTerm(
        func=t1_mdp.ref_pos_exp,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "T_brf",
            "std": 0.05,
        },
    )
    ref_yaw_rf = RewTerm(
        func=t1_mdp.ref_yaw_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_foot_link"],
                preserve_order=True,
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "T_brf",
            "std": 0.20,
        },
    )
    ref_linvel_b = RewTerm(
        func=t1_mdp.ref_linvel_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "v_b",
            "std": 0.25,
        },
    )
    ref_yaw_b = RewTerm(
        func=t1_mdp.ref_base_yaw_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            ),
            "command_name": "foot_pos_cmd",
            "attr_name": "T_wbase",
            "std": 0.20,
        },
    )
    ref_pos_b_z = RewTerm(
        func=t1_mdp.base_z_position_exp,
        weight=1.0,
        params={
            "command_name": "foot_pos_cmd",
            "attr_name": "T_wbase",
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    ref_qpos = RewTerm(
        func=t1_mdp.joint_pos_tracking_sum_exp,
        weight=0.5,
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
            "std": 0.3,
        },
    )
    ref_qvel = RewTerm(
        func=t1_mdp.joint_vel_tracking_sum_exp,
        weight=0.1,
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
            "std": 1.0,
        },
    )

    # =============================================================================
    # Regularization
    # =============================================================================
    survival = RewTerm(func=mdp.is_alive, weight=0.25)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    ankle_posture = RewTerm(
        func=t1_mdp.joint_posture,
        weight=-2.0,
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
    base_height_range = RewTerm(
        func=t1_mdp.base_height_range_from_foot,
        weight=-10.0,
        params={
            "min_height": 0.58,
            "max_height": 0.62,
        },
    )
    base_height_exp = RewTerm(
        func=t1_mdp.base_height_exp_from_foot,
        weight=2.0,
        params={
            "target_height": 0.59,
            "std": 0.2,
        },
    )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    base_orientation = RewTerm(
        func=t1_mdp.flat_link_orientation_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "std": 0.2,
        },
    )
    # trunk_yaw_mid_feet = RewTerm(
    #     func=t1_mdp.trunk_yaw_align_with_feet,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["Trunk", "left_foot_link", "right_foot_link"],
    #             preserve_order=True,
    #         ),
    #         "std": 0.2,
    #     },
    # )
    left_foot_orientation = RewTerm(
        func=t1_mdp.flat_link_orientation_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link"]),
            "std": 0.2,
        },
    )
    right_foot_orientation = RewTerm(
        func=t1_mdp.flat_link_orientation_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_foot_link"]),
            "std": 0.2,
        },
    )
    feet_slip = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-15.0,
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
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link", preserve_order=True),
        },
    )
    # =============================================================================
    # TASK SPACE REWARDS - End-effector position and trajectory tracking
    # =============================================================================
    correct_swing_height = RewTerm(
        func=t1_mdp.swing_foot_height,
        weight=5.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "swing_scale": 0.1,
        },
    )
    stance_height_penalty = RewTerm(
        func=t1_mdp.stance_height_penalty_stair,
        weight=-5.0,
        params={
            "command_name": "foot_pos_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    right_foot_contact = RewTerm(
        func=t1_mdp.correct_single_foot_contact,
        weight=-5.0,
        params={
            "command_name": "foot_pos_cmd",
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_foot_link"]),
            "foot_id": 1,
        },
    )
    left_foot_contact = RewTerm(
        func=t1_mdp.correct_single_foot_contact,
        weight=-5.0,
        params={
            "command_name": "foot_pos_cmd",
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link"]),
            "foot_id": 0,
        },
    )
    feet_airtime = RewTerm(
        func=t1_mdp.feet_air_time_at_contact,
        weight=20.0,
        params={
            "threshold": 0.4,  # Swing duration below this penalize.
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
            ),
        },
    )

@configclass
class T1FootTrackStairEnvCfg(T1FootTrackEnvCfg):
    scene: T1FootTrackStairSceneCfg = T1FootTrackStairSceneCfg(num_envs=4096, env_spacing=1.0)
    commands: T1FootTrackStairCommands = T1FootTrackStairCommands()
    rewards: T1FootTrackStairRewards = T1FootTrackStairRewards()
    events: T1FootTrackStairEvents = T1FootTrackStairEvents()

    enable_random_upper_joints: bool = True
    enable_default_upper_joints: bool = False
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

        if not self.enable_default_upper_joints:
            self.events.reset_robot_upper_joints_by_offset_default = None
        if not self.enable_random_upper_joints:
            self.events.reset_robot_upper_joints_by_offset_default = None
            self.events.push_robot = None
            self.events.add_base_com_xyz = None


@configclass
class T1FootTrackStairEnvCfg_PLAY(T1FootTrackStairEnvCfg):
    enable_random_upper_joints: bool = False
    enable_default_upper_joints: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 1.5
        self.episode_length_s = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.foot_pos_cmd.debug_vis = True
        self.commands.foot_pos_cmd.resample_ratio_range = (0.0, 0.0)
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"
