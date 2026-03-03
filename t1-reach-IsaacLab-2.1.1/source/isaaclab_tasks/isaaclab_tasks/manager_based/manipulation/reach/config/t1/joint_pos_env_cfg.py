# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.envs.mdp.events import PREP_STATE
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
import isaaclab_tasks.manager_based.manipulation.reach.config.t1.mdp as t1_mdp
import isaaclab_tasks.manager_based.manipulation.reach.config.t1.terms as t1_terms
from .terms import curriculum as T1_reach_curriculum
##
# Pre-defined configs
##
from isaaclab_assets.robots.booster import T1_REACH_CFG  # isort: skip
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Environment configuration
##


@configclass
class T1ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
        ),
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


def get_left_hand_pose_command(env: ManagerBasedRLEnv):
    """Get left hand pose command as 7D (position + quaternion) for reward calculations."""
    term: t1_terms.SE3IKCommand = env.command_manager.get_term("se3_ik_cmd")
    return term.xyzwxyz_BLH


def get_right_hand_pose_command(env: ManagerBasedRLEnv):
    """Get right hand pose command as 7D (position + quaternion) for reward calculations."""
    term: t1_terms.SE3IKCommand = env.command_manager.get_term("se3_ik_cmd")
    return term.xyzwxyz_BRH


def get_joint_reference(env: ManagerBasedRLEnv):
    term: t1_terms.SE3IKCommand = env.command_manager.get_term("se3_ik_cmd")
    return term.joint_reference


@configclass
class T1ReachObservations:
    """Observation specifications for the MDP."""

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )

        left_hand_pose_command = ObsTerm(func=get_left_hand_pose_command)
        right_hand_pose_command = ObsTerm(func=get_right_hand_pose_command)
        joint_reference = ObsTerm(func=get_joint_reference)
        left_hand_pose_current = ObsTerm(
            func=t1_mdp.body_pose_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_hand_link"]
                ),
                "command_name": "se3_ik_cmd",
                "attr_name": "left_hand_pose_command"
            }
        )
        right_hand_pose_current = ObsTerm(
            func=t1_mdp.body_pose_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["right_hand_link"]
                ),
                "command_name": "se3_ik_cmd",
                "attr_name": "right_hand_pose_command"
            }
        )
        hand_vel = ObsTerm(
            func=t1_mdp.body_vel_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_hand_link", "right_hand_link"],
                )
            }
        )
        hand_acc = ObsTerm(
            func=t1_mdp.body_acc_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_hand_link", "right_hand_link"],
                )
            }
        )
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )

        left_hand_pose_command = ObsTerm(func=get_left_hand_pose_command)
        right_hand_pose_command = ObsTerm(func=get_right_hand_pose_command)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 8

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# @configclass
# class T1ReachCommands:
#     """Command terms for the MDP."""
#     se3_ik_cmd = t1_terms.SE3IKCommandCfg(
#         debug_vis=True,
#         path_dataset=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "ik_dataset_0.02.npz"),
#         resample_ratio_range=(0, 0.8),
#         qref_indices=[
#             2, 3, 4, 5, 6, 7, 8,
#             9, 10, 11, 12, 13, 14, 15
#         ],
#         asset_name="robot",
#         body_names=[
#             "left_hand",
#             "right_hand"
#         ],
#         ranges_lh=t1_terms.SE3IKCommandCfg.Ranges(
#             pos_x=(0.10, 0.30),
#             pos_y=(0.10, 0.30),
#             pos_z=(-0.3, 0.3),
#             roll=(-0.5, 0.5),
#             pitch=(-0.5, 0.5), ###
#             yaw=(-0.5, 0.5), ###
#         ),
#         ranges_rh=t1_terms.SE3IKCommandCfg.Ranges(
#             pos_x=(0.10, 0.30),
#             pos_y=(-0.30, -0.10),
#             pos_z=(-0.3, 0.3),
#             roll=(-0.5, 0.5),
#             pitch=(-0.5, 0.5), ###
#             yaw=(-0.5, 0.5), ####
#         ),
#         rel_random=0.1,

#     )
@configclass
class T1ReachCommands:
    """Command terms for the MDP (narrow initial ranges)."""

    se3_ik_cmd = t1_terms.SE3IKCommandCfg(
        debug_vis=True,
        path_dataset=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "ik_dataset_0.02.npz"),
        resample_ratio_range=(0.0, 0.4),
        qref_indices=[2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 12, 13, 14, 15],
        asset_name="robot",
        body_names=["left_hand_link", "right_hand_link"],
        # half-ranges
        ranges_lh=t1_terms.SE3IKCommandCfg.Ranges(
            pos_x=(PREP_STATE.pos[0] - 0.05, PREP_STATE.pos[0] + 0.05),
            pos_y=(PREP_STATE.pos[1] + 0.10 - 0.05, PREP_STATE.pos[1] + 0.10 + 0.05),
            pos_z=(PREP_STATE.pos[2] - 0.05, PREP_STATE.pos[2] + 0.05),
            roll=(-0.25, 0.25),
            pitch=(-0.25, 0.25),
            yaw=(-0.25, 0.25),
        ),
        ranges_rh=t1_terms.SE3IKCommandCfg.Ranges(
            pos_x=(PREP_STATE.pos[0] - 0.05, PREP_STATE.pos[0] + 0.05),
            pos_y=(PREP_STATE.pos[1] - 0.10 - 0.05, PREP_STATE.pos[1] - 0.10 + 0.05),
            pos_z=(PREP_STATE.pos[2] - 0.05, PREP_STATE.pos[2] + 0.05),
            roll=(-0.25, 0.25),
            pitch=(-0.25, 0.25),
            yaw=(-0.25, 0.25),
        ),
        rel_random=0,
    )


@configclass
class T1ReachTerminations:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Waist",
                    "AL.*",
                    "AR.*",
                    "H.*",
                    "Trunk"
                ]
            ),
            "threshold": 20.0
        },
    )


@configclass
class T1ReachEvents:
    """Configuration for the reach end-effector pose tracking environment."""
    # startup
    add_arm_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    add_end_effector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_hand_link", "right_hand_link"]
            ),
            "mass_distribution_params": (0.0, 3.5),
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
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.80, 1.20),
            "damping_distribution_params": (0.80, 1.20),
        },
    )

    # interval
    hand_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_hand_link", "right_hand_link"]
            ),
            "force_range": (-15.0, 15.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_to_prep,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class T1ReachActions:
    pass
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        scale=1.0,
        preserve_order=True,
        use_default_offset=True
    )


@configclass
class T1ReachRewards:
    """Reward terms for the MDP."""

    lh_position_tracking_exp = RewTerm(
        func=t1_mdp.position_command_error_exp,
        weight=1.5,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_hand_link"] 
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "left_hand_pose_command",
            "std": 0.10
        },
    )
    lh_position_tracking_exp_fine_grained = RewTerm(
        func=t1_mdp.position_command_error_exp,
        weight=3,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_hand_link"] 
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "left_hand_pose_command",
            "std": 0.30
        },
    )
    lh_orientation_tracking = RewTerm(
        func=t1_mdp.orientation_command_error_exp,
        weight=100,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_hand_link"] 
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "left_hand_pose_command",
            "std": 0.10
        },
    )

    rh_position_tracking_exp = RewTerm(
        func=t1_mdp.position_command_error_exp,
        weight=1.5,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_hand_link"]
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "right_hand_pose_command",
            "std": 0.10
        },
    )
    rh_position_tracking_exp_fine_grained = RewTerm(
        func=t1_mdp.position_command_error_exp,
        weight=3,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_hand_link"]
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "right_hand_pose_command",
            "std": 0.30
        },
    )
    rh_orientation_tracking = RewTerm(
        func=t1_mdp.orientation_command_error_exp,
        weight=100,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["right_hand_link"]
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "right_hand_pose_command",
            "std": 0.1,
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-2e-3 #Changed from -0.7e-3 to -2e-3
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7
    )
    power = RewTerm(
        func=t1_mdp.joint_power,
        weight=-1e-6
    )

    ee_accel = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-5e-4, # Changed from -1e-4 to -5e-4
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_hand_link", "right_hand_link"], 
            ),
        }
    )

    torque_limit = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-1.0e-4
    )
    pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5
    )

    follow_qref = RewTerm(
        func=t1_mdp.joint_tracking_exp,
        weight=3.0,

        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                preserve_order=True,
            ),
            "command_name": "se3_ik_cmd",
            "attr_name": "joint_reference",
            "std": 1.0
        }
    )

def linear_ramp(env, env_id, data, start_val, end_val, num_steps):
    step = env.common_step_counter
    if step >= num_steps:
        return end_val
    progress = step / num_steps
    return start_val + (end_val - start_val) * progress


@configclass
class T1ReachCurriculum:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 1500 * 20}
    # )
    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-4, "num_steps": 1500 * 20}
    # )
    # joint_acc = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_acc", "weight": -3e-7, "num_steps": 1500 * 20}
    # )


    action_rate = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight, params={
            "term_name": "action_rate",
            "start_step": 3000 * 20,
            "end_step": 6000 * 20,
        }
    )

    joint_vel = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight, params={
            "term_name": "joint_vel",
            "start_step": 3000 * 20,
            "end_step": 6000 * 20,
        }
    )       

    joint_acc = CurrTerm(
        func=T1_reach_curriculum.linearly_alter_weight, params={
            "term_name": "joint_acc",
            "start_step": 3000 * 20,
            "end_step": 6000 * 20,
        }
    )

    # rel_random = CurrTerm(
    #     func=mdp.modify_env_param, params={

    #         "address": "command_manager.cfg.se3_ik_cmd.rel_random",
    #         "modify_fn": linear_ramp,
    #         "modify_params": {
    #             "start_val": 0.1,
    #             "end_val": 0.3,
    #             "num_steps": 3000 * 20,
    #         }
    #     }
    # )
    
@configclass
class T1ReachEnvCfg(ReachEnvCfg):
    scene: T1ReachSceneCfg = T1ReachSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: T1ReachObservations = T1ReachObservations()
    commands: T1ReachCommands = T1ReachCommands()
    terminations: T1ReachTerminations = T1ReachTerminations()
    curriculum: T1ReachCurriculum = T1ReachCurriculum()
    rewards: T1ReachRewards = T1ReachRewards()
    events: T1ReachEvents = T1ReachEvents()
    actions: T1ReachActions = T1ReachActions()
    # Optionally delay the start of curriculum terms. Can be specified as a number of
    # environment steps (`curriculum_start_delay_steps`) or seconds
    # (`curriculum_start_delay_s`). If both are provided, `curriculum_start_delay_steps`
    # takes precedence. This allows "start later" behaviour for curriculum during
    # training without modifying the curriculum definitions.
    curriculum_start_delay_steps: int = 0
    curriculum_start_delay_s: float = 0.0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.scene.robot = T1_REACH_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 60000
        self.commands.se3_ik_cmd.debug_vis = True
        
        # Curriculum weight parameters 
        self.curriculum.action_rate.params["start_weight"] = -2e-3  # Current initial weight
        self.curriculum.action_rate.params["end_weight"] = -4e-3      # Target final weight
        self.curriculum.joint_vel.params["start_weight"] = -1e-4      # Current initial weight  
        self.curriculum.joint_vel.params["end_weight"] = -5e-4        # Target final weight
        self.curriculum.joint_acc.params["start_weight"] = -1e-7      # Current initial weight
        self.curriculum.joint_acc.params["end_weight"] = -3e-7        # Target final weight

        # Apply optional curriculum start delay (in steps or seconds)
        # If curriculum_start_delay_steps > 0 it is used directly. Otherwise,
        # convert curriculum_start_delay_s to steps using sim.dt and decimation.
        delay_steps = int(self.curriculum_start_delay_steps)
        if delay_steps <= 0 and self.curriculum_start_delay_s > 0.0:
            # steps_per_env_step = 1 / sim.dt, but policy steps are decimated: one action every `decimation` sim steps
            # so convert seconds -> sim steps -> policy steps
            sim_steps = int(round(self.curriculum_start_delay_s / self.sim.dt))
            delay_steps = sim_steps // max(1, self.decimation)

        if delay_steps > 0:
            # Safely add delay to any curriculum term that defines `start_step` in params
            for attr in ("action_rate", "joint_vel", "joint_acc", "rel_random"):
                term = getattr(self.curriculum, attr, None)
                if term is None:
                    continue
                params = getattr(term, "params", None)
                if not isinstance(params, dict):
                    continue
                if "start_step" in params and isinstance(params["start_step"], int):
                    params["start_step"] = params["start_step"] + delay_steps


@configclass
class T1ReachEnvCfg_PLAY(T1ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.se3_ik_cmd.debug_vis = True
        self.commands.se3_ik_cmd.rel_random = 0
