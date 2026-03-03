# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Booster robots.

The following configurations are available:

* :obj:`T1_CFG`: T1 humanoid robot with 7DOF Arms
* :obj:`T1_REACH_CFG`: Fixed base version of T1_CFG

Reference: https://booster.feishu.cn/wiki/UvowwBes1iNvvUkoeeVc3p5wnUg
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


##
# Configuration
##


T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/robots/USD/t1_29dof_1108_add_ee.usd",
        # Turns on contact sensors for collision/contact detection.
        activate_contact_sensors=True,
        # Physical properties for rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,  # Resistance to linear movement.
            angular_damping=0.0,  # Resistance to rotational movement.
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            # Head joints (indices 0-1)
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            
            # Waist (index 2)
            "Waist": 0.0,

            # Left arm
            "Left_Shoulder_Pitch": 0.376992,
            "Left_Shoulder_Roll": -0.816816,
            "Left_Elbow_Pitch": -0.15708,
            "Left_Elbow_Yaw": -1.88496,
            "Left_Wrist_Pitch": -0.816816,
            "Left_Wrist_Yaw": 0.0,
            "Left_Hand_Roll": 0.0,

            # Right arm
            "Right_Shoulder_Pitch": 0.376992,
            "Right_Shoulder_Roll": 0.816816,
            "Right_Elbow_Pitch": -0.15708,
            "Right_Elbow_Yaw": 1.88496,
            "Right_Wrist_Pitch": -0.816816,
            "Right_Wrist_Yaw": 0.0,
            "Right_Hand_Roll": 0.0,
            
            # Left leg (indices 17-22)
            "Left_Hip_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.2,
            "Left_Ankle_Roll": 0.0,
            
            # Right leg (indices 23-28)
            "Right_Hip_Pitch": -0.2,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.4,
            "Right_Ankle_Pitch": -0.2,
            "Right_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            effort_limit_sim=7,
            velocity_limit_sim=12.56,
            joint_names_expr=["Head_pitch", "AAHead_yaw"],
            stiffness=20.0,
            damping=0.2,
            armature=0.01,
        ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Yaw": 60.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Pitch": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": 10.9,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Pitch": 12.5,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            stiffness=200.0,
            damping=5.0,
            armature={
                ".*_Hip_.*": 0.01,
                ".*_Knee_Pitch": 0.01,
                "Waist": 0.01,
            },
            min_delay=10,
            max_delay=20,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            damping=1.0,
            armature=0.01,
            min_delay=10,
            max_delay=20,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",

            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 10.0,
                ".*_Elbow_.*": 10.0,
                ".*_Wrist_.*": 10.0,
                ".*_Hand_Roll": 10.0,

            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
                ".*_Wrist_.*": 18.84,
                ".*_Hand_Roll": 18.84,

            },
            
            stiffness=50.0,
            damping=3.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
                ".*_Wrist_.*": 0.001,
                ".*_Hand_Roll": 0.001,
            },

        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["left_Link1", "right_Link1"],
            effort_limit_sim=10,
            velocity_limit_sim=57,
            stiffness=40,
            damping=3,
            armature=0.001,
        ),
        "grippers_mimic": ImplicitActuatorCfg(
            joint_names_expr=["left_Link11", "left_Link2", "left_Link22", "right_Link11", "right_Link2", "right_Link22"],
            effort_limit_sim=10,
            velocity_limit_sim=57,
            stiffness=40,
            damping=3,
            armature=0.001,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""

T1_REACH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/isaaclab_assets/isaaclab_assets/robots/USD/T1_7dof_arms_with_gripper_Fixed_upper_only_reducedcollision.usd",
        # Turns on contact sensors for collision/contact detection.
        activate_contact_sensors=True,
        # Physical properties for rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,  # Resistance to linear movement.
            angular_damping=0.0,  # Resistance to rotational movement.
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            # Head joints (indices 0-1)
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            
            # Waist (index 2)
            "Waist": 0.0,

            # Left arm
            "Left_Shoulder_Pitch": 0.376992,
            "Left_Shoulder_Roll": -0.816816,
            "Left_Elbow_Pitch": -0.15708,
            "Left_Elbow_Yaw": -1.88496,
            "Left_Wrist_Pitch": -0.816816,
            "Left_Wrist_Yaw": 0.0,
            "Left_Hand_Roll": 0.0,

            # Right arm
            "Right_Shoulder_Pitch": 0.376992,
            "Right_Shoulder_Roll": 0.816816,
            "Right_Elbow_Pitch": -0.15708,
            "Right_Elbow_Yaw": 1.88496,
            "Right_Wrist_Pitch": -0.816816,
            "Right_Wrist_Yaw": 0.0,
            "Right_Hand_Roll": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim={
                ".*_Shoulder_.*": 10.0,
                ".*_Elbow_.*": 10.0,
                ".*_Wrist_.*": 10.0,
                ".*_Hand_Roll": 10.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_.*": 18.84,
                ".*_Elbow_.*": 18.84,
                ".*_Wrist_.*": 18.84,
                ".*_Hand_Roll": 18.84,
            },
            stiffness=50.0,
            damping=3.0,
            armature={
                ".*_Shoulder_.*": 0.01,
                ".*_Elbow_.*": 0.01,
                ".*_Wrist_.*": 0.001,
                ".*_Hand_Roll": 0.001,
            },
        ),

    },
)