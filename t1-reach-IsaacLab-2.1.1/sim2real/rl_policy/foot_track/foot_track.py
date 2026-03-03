import os
import sys
from pathlib import Path
import time
import signal
import sys
from datetime import datetime
import csv
import json

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import yaml
from termcolor import colored
import mujoco as mj

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from booster_robotics_sdk_python import RobotMode


class BoosterTeleop(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=1.0, headless=True):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.residual_upper_body_action = False
        self.obs_history = {}
        self.obs_history_len = 8
        self.single_obs_len = 48
        self.last_policy_action = np.zeros((1, 12))
        self.single_action_len = 12
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))

        self.command_stsw_pose = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.command_countdown = np.array([[0.0]])
        self.command_foot_indicator = np.array([[0.0]])
        self.startup = True

        self.in_stepping = False
        self.step_length = 0.1
        self.countdown_takes_s = 1.0 # time in seconds from 1.0 to 0.0, includes 0.5s shifting and 0.5s swinging.
        self.start_stepping_time = None

        self.model = mj.MjModel.from_xml_path(self.config.get("ik_xml_path"))
        self.data = mj.MjData(self.model)
        mj.mj_resetDataKeyframe(self.model, self.data, 0)

    def get_st_pose(self):
        """Get stance foot pose w.r.t world frame."""
        p_lf = self.data.site("left_foot").xpos
        p_rf = self.data.site("right_foot").xpos
        R_lf = self.data.site("left_foot").xmat.reshape(3, 3)
        R_rf = self.data.site("right_foot").xmat.reshape(3, 3)

        if self.command_foot_indicator[0] == 0:
            # Left foot is stance
            p_st = p_lf
            R_st = R_lf
        else:
            # Right foot is stance
            p_st = p_rf
            R_st = R_rf

        # Extract yaw from rotation matrix
        yaw = np.arctan2(R_st[1, 0], R_st[0, 0])

        return np.array([[p_st[0], p_st[1], p_st[2], yaw]])

    def get_sw_pose(self):
        """Get swing foot pose w.r.t world frame."""
        p_lf = self.data.site("left_foot").xpos
        p_rf = self.data.site("right_foot").xpos
        R_lf = self.data.site("left_foot").xmat.reshape(3, 3)
        R_rf = self.data.site("right_foot").xmat.reshape(3, 3)

        if self.command_foot_indicator[0] == 0:
            # Right foot is swing
            p_sw = p_rf
            R_sw = R_rf
        else:
            # Left foot is swing
            p_sw = p_lf
            R_sw = R_lf

        # Extract yaw from rotation matrix
        yaw = np.arctan2(R_sw[1, 0], R_sw[0, 0])

        return np.array([[p_sw[0], p_sw[1], p_sw[2], yaw]])

    def get_stsw_pose(self):
        p_lf = self.data.site("left_foot").xpos
        p_rf = self.data.site("right_foot").xpos
        R_lf = self.data.site("left_foot").xmat.reshape(3, 3)
        R_rf = self.data.site("right_foot").xmat.reshape(3, 3)
        if self.command_foot_indicator[0] == 0:
            p_st = p_lf
            p_sw = p_rf
            R_st = R_lf
            y = -0.2
        else:
            p_st = p_rf
            p_sw = p_lf
            R_st = R_rf
            y = 0.2
        delta_pos_world = p_sw - p_st
        delta_pos_st = R_st.T @ delta_pos_world

        # x, y, z relative position in stance frame
        x = delta_pos_st[0]
        # y = delta_pos_st[1]
        z = delta_pos_st[2]
        # yaw: extract from stance foot rotation matrix
        yaw = np.arctan2(R_st[1, 0], R_st[0, 0])

        return np.array([[0, y, 0.0, 0]])

    def rl_inference(self, robot_state_data):
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        self.data.qpos[:] = obs_buffer_dict["dof_pos"]
        mj.mj_forward(self.model, self.data)

        if self.startup:
            self.command_stsw_pose = self.get_stsw_pose()
            self.command_countdown[0][0] = 0.0
            self.startup = False

        obs_buffer_dict["command_stsw_pose"] = self.command_stsw_pose
        obs_buffer_dict["command_countdown"] = self.command_countdown
        obs_buffer_dict["command_foot_indicator"] = self.command_foot_indicator
        # print(self.command_countdown,self.command_stsw_pose,self.command_foot_indicator)

        obs_buffer_dict["base_ang_vel_scaled"] = obs_buffer_dict["base_ang_vel"] * 0.2
        obs_buffer_dict["dof_pos_lower"] = obs_buffer_dict["dof_pos"][:, -12:]
        obs_buffer_dict["dof_vel_scaled"] = obs_buffer_dict["dof_vel"][:, -12:] * 0.025
        obs_buffer_dict["dof_vel_scaled"][:, [4, 5, 10, 11]] = 0.0
        obs_buffer_dict["last_policy_action"] = self.last_policy_action.copy()

        obs_list = [
            "command_stsw_pose",
            "command_countdown",
            "command_foot_indicator",
            "base_ang_vel_scaled",
            "projected_gravity",
            "dof_pos_lower",
            "dof_vel_scaled",
            "last_policy_action",
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
            scaled_policy_action[:, self.upper_dof_indices] += self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]

        return scaled_policy_action

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)
        if not self.in_stepping:
            if keycode in ["a", "d"]:
                self.in_stepping = True
                self.start_stepping_time = time.time_ns()
                print(f"{'Left' if keycode == 'a' else 'Right'}Foot Step.")
                self.command_foot_indicator[0][0] = 1.0 if keycode == "a" else 0.0
                self.command_stsw_pose[0] = self.get_stsw_pose()
                self.command_stsw_pose[0][0] += self.step_length
                self.command_countdown[0][0] = 1.0
                self._print_control_status()

    def _print_control_status(self):
        print(f"Current Stance: {self.command_foot_indicator}")

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        robot_state_data = self.state_processor.robot_state_data
        if self.state_processor.robot_state_data is None:
            return

        if self.in_stepping:
            now = time.time_ns()
            t_passed = (now - self.start_stepping_time) / 1e9
            # start: 0.0
            # middle: t_passsed is half step time, reaches 1, the final value is 1
            # end: 0.0
            self.command_countdown[0][0] = np.abs((t_passed / (self.countdown_takes_s) - 1.0))
            if t_passed > self.countdown_takes_s:
                self.command_countdown[0][0] = 0.0
                self.in_stepping = False  # end stepping based on time

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            true_act = scaled_policy_action + self.default_dof_angles[self.lower_dof_indices]
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            q_target[:, self.lower_dof_indices] = true_act

        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])

    def _handle_base_height_control(self, keycode):
        """Handle base height control."""
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1


def signal_handler(sig, frame):
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

    policy = BoosterTeleop(config=config, model_path=model_path, rl_rate=50, policy_action_scale=1.0)

    signal.signal(signal.SIGINT, signal_handler)
    policy.run()
