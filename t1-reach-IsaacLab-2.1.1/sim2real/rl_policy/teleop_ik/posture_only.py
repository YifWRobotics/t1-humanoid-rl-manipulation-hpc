import os
import sys
from pathlib import Path
import time

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import threading
import yaml
from scipy.spatial.transform import Rotation as R

from termcolor import colored

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.arm_ik.mink_t1_29dof_teleop import T1_29_ArmIK_Teleop
from booster_robotics_sdk_python import GripperControlMode, GripperMotionParameter, B1HandIndex
import mink as ik


class TeleopXRoboLocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=1.0
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.stair_posture_lower = np.array([
            0.0,
            0.10405,
            0,
            0,
            0.0702,
            -0.1807,
            -0.022,
            -1.54725,
            0,
            0,
            1.5327,
            0.0084,
            0.0088
        ]).reshape(1, 13)

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7: 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            q_target[:, self.lower_dof_indices] = dof_pos[:, self.lower_dof_indices] + (self.stair_posture_lower - dof_pos[:, self.lower_dof_indices]) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data

        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7: 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            q_target[:, self.lower_dof_indices] = self.stair_posture_lower

        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(
                q_target[0],
                self.motor_pos_lower_limit_list,
                self.motor_pos_upper_limit_list
            )

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7: 7 + self.num_dofs]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get(
        "model_path")
    if not model_path:
        raise ValueError(
            "model_path must be provided either via --model_path argument or in config file")

    policy = TeleopXRoboLocoManipPolicy(
        config=config,
        model_path=model_path,
        rl_rate=50,
        policy_action_scale=1.0
    )
    policy.run()
