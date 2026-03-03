import os
import sys
from pathlib import Path
import time
import signal
import sys

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import threading
import yaml
from scipy.spatial.transform import Rotation as R

import pinocchio as pin
import xrobotoolkit_sdk as xrt
from termcolor import colored

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from booster_robotics_sdk_python import GripperControlMode, GripperMotionParameter, B1HandIndex, RobotMode
import mink as ik


class BoosterTeleop(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=1.0, headless=True
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.residual_upper_body_action = False
        self.obs_history = {}
        self.obs_history_len = 1
        self.single_obs_len = 47
        self.last_policy_action = np.zeros((1, 12))
        self.single_action_len = 12
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))

    def rl_inference(self, robot_state_data):
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        phase_time = self._get_obs_phase_time()

        obs_buffer_dict["command_lin_vel"] = self.lin_vel_command.reshape(1, 2)
        obs_buffer_dict["command_ang_vel"] = self.ang_vel_command.reshape(1, 1)
        obs_buffer_dict["dof_pos_lower"] = obs_buffer_dict["dof_pos"][:, self.lower_dof_indices]
        obs_buffer_dict["dof_vel_lower"] = obs_buffer_dict["dof_vel"][:, self.lower_dof_indices] * 0.1
        obs_buffer_dict["sin_phase"] = np.sin(2 * phase_time * np.pi) * self.stand_command[0, 0]
        obs_buffer_dict["cos_phase"] = np.cos(2 * phase_time * np.pi) * self.stand_command[0, 0]
        obs_buffer_dict["last_policy_action"] = self.last_policy_action.copy()

        obs_list = [
            "projected_gravity", "base_ang_vel",
            "command_lin_vel", "command_ang_vel",
            "cos_phase", "sin_phase",
            "dof_pos_lower", "dof_vel_lower",
            "last_policy_action"
        ]

        # initialize and roll observation history if needed
        for key in obs_list:
            if key not in self.obs_history.keys():
                self.obs_history[key] = np.zeros((self.obs_history_len, obs_buffer_dict[key].shape[1]))

            # roll history: oldest observations removed, make space for newest
            self.obs_history[key] = np.roll(self.obs_history[key], -1, axis=0)
            self.obs_history[key][-1] = obs_buffer_dict[key]

        # collect observations in specified layout
        collected_obs = []
        for key in obs_list:
            collected_obs.append(self.obs_history[key].flatten())

        # concatenate all observation types
        collected_obs = np.concatenate(collected_obs, axis=0)

        # run policy
        policy_action = self.policy({"obs": collected_obs.reshape(1, -1).astype(np.float32)})

        # WBC actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos
                - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        robot_state_data = self.state_processor.robot_state_data
        if self.state_processor.robot_state_data is None:
            return

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7: 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            true_act = scaled_policy_action + self.default_dof_angles[self.lower_dof_indices]
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            q_target[:, self.lower_dof_indices] = true_act

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

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == "m":
            self.lin_vel_command[:, 0] = -1.0
            self.logger.info(
                colored(f"lin_vel_command: {self.lin_vel_command}", "green"))
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)

    def _handle_base_height_control(self, keycode):
        """Handle base height control."""
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        """Handle joystick base height control."""
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")


def signal_handler(sig, frame):
    xrt.close()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--model_path", type=str, help="path to the .pt")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = BoosterTeleop(
        config=config,
        model_path=model_path,
        rl_rate=50,
        policy_action_scale=1.0
    )

    signal.signal(signal.SIGINT, signal_handler)
    policy.run()
