import os
import sys
import time

import numpy as np
import argparse
import yaml
import onnxruntime
from pathlib import Path

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


import pinocchio as pin
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.math import quat_rotate_inverse_numpy
from termcolor import colored
import matplotlib.pyplot as plt
from sim2real.utils.robot import Robot
from datetime import datetime


class LocoManipPolicyModel(DecLocomotionPolicy):
    def __init__(
            self, config,
            model_path_loco=None, model_path_manip=None,
            rl_rate_loco=50, rl_rate_manip=100, model_rate_manip=500,
            policy_action_scale=1
    ):
        self.config = config
        # Initialize robot config
        self._init_robot_config(disable_upper_body_pd=(model_path_manip is None))
        # Initialize SDK components
        self._init_sdk_components()
        # Initialize observation config
        self._init_obs_config()
        # Initialize communication components
        self._init_communication_components()
        # Initialize policy components

        self.loco_decimation = int(rl_rate_manip / rl_rate_loco)
        self.model_decimation = int(model_rate_manip / rl_rate_manip)
        self.loco_decimation_counter = 0
        self.model_decimation_counter = 0

        if model_path_loco is None and model_path_manip is None:
            print("Error: model_path_loco and model_path_manip cannot be None")

        self._init_policy_components(model_path_loco, model_path_manip,
                                     rl_rate_loco, rl_rate_manip,
                                     policy_action_scale)

        # self.init_upper_body_controller()
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()

        # self._handle_init_state()

        # self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        # self.ref_upper_dof_pos *= 0.0
        # self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        # self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)

        self.record_joint_pos = []
        self.record_joint_torque = []
        # self.record_base_acc = []

        self.cmd_q = np.zeros(self.num_dofs)
        self.cmd_dq = np.zeros(self.num_dofs)
        self.cmd_tau = np.zeros(self.num_dofs)

        # self.upper_body_controller = T1_LeftArm_OSC(False, self.config)
        self.upper_body_controller = None
        self.model_pos_Kp = self.config.get("model_pos_Kp", 200.0)
        self.model_pos_Kd = self.config.get("model_pos_Kd", 50.0)
        self.model_ori_Kp = self.config.get("model_ori_Kp", 200.0)
        self.model_ori_Kd = self.config.get("model_ori_Kd", 60.0)

    def _init_robot_config(self, disable_upper_body_pd=False):
        """Initialize robot configuration and parameters."""
        if disable_upper_body_pd:
            self.config['POLICY_KP'][2:9] = [2.0] * 7
            self.config['POLICY_KD'][2:9] = [0.5] * 7
        self.robot = Robot(self.config)
        self.num_dofs = self.robot.NUM_JOINTS
        self.default_dof_angles = np.array(self.robot.DEFAULT_DOF_ANGLES)
        self.num_upper_dofs = self.config.get("NUM_UPPER_BODY_JOINTS", 14)
        self.num_lower_dofs = self.config.get("NUM_LOWER_BODY_JOINTS", 13)

        # Initialize motor limits (only position limits are used)
        self.motor_pos_lower_limit_list = self.config.get("motor_pos_lower_limit_list", None)
        self.motor_pos_upper_limit_list = self.config.get("motor_pos_upper_limit_list", None)

        # Setup dof names and indices
        self._setup_dof_mappings()

    def _init_obs_config(self):
        """Initialize observation configuration and buffers."""
        self.obs_scales = self.config["obs_scales"]
        self.obs_dims = self.config["obs_dims"]
        self.obs_dict = self.config["obs_dict"]
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.config["history_length_dict"]

        # Initialize observation buffers
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key]))
            for key in self.obs_dim_dict
        }

    def _init_policy_components(self, model_path_loco, model_path_manip, rl_rate_loco, rl_rate_manip, policy_action_scale):
        """Initialize policy-related components."""
        self.setup_policy_loco(model_path_loco)
        # self.setup_policy_manip(model_path_manip)
        self.last_policy_action_loco = np.zeros((1, self.num_lower_dofs))
        self.last_policy_action_manip = np.zeros((1, 7))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        self.policy_action_scale = policy_action_scale

    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.get("DESIRED_BASE_HEIGHT", 0.78)
        self.gait_period = self.config.get("GAIT_PERIOD", 0.64)

        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.ee_command = np.array([[0.4, 0.15, 0.0,
                                     1.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0
                                     ]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])
        # self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        # self.ref_upper_dof_pos *= 0.0
        # self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))

        # Upper body controller

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 + 16 : 7 + self.num_dofs]  # - self.default_dof_angles # TODO: check
        current_obs_buffer_dict["dof_vel"] = robot_state_data[:, 7 + self.num_dofs + 6 + 16 : 7 + self.num_dofs + 6 + self.num_dofs]
        # Calculate projected gravity
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse_numpy(
            current_obs_buffer_dict["base_quat"], v
        )
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_ee"] = self.ee_command
        current_obs_buffer_dict["command_stand"] = self.stand_command
        current_obs_buffer_dict["command_base_height"] = self.base_height_command
        # current_obs_buffer_dict["command_waist_dofs"] = self.waist_dofs_command
        current_obs_buffer_dict["phase_time"] = self._get_obs_phase_time()
        current_obs_buffer_dict["sin_phase"] = np.sin(2 * np.pi * current_obs_buffer_dict["phase_time"])
        current_obs_buffer_dict["cos_phase"] = np.cos(2 * np.pi * current_obs_buffer_dict["phase_time"])
        # current_obs_buffer_dict["ref_upper_dof_pos"] = self.ref_upper_dof_pos
        current_obs_buffer_dict["last_actions_loco"] = self.last_policy_action_loco
        current_obs_buffer_dict["last_actions_manip"] = self.last_policy_action_manip
        current_obs_buffer_dict['joint_pos'] = robot_state_data[:, 7 + 2 : 7 + 2 + 7]  # - self.default_dof_angles[2:9] #TODO: check
        current_obs_buffer_dict['joint_vel'] = robot_state_data[:, 7 + self.num_dofs + 6 + 2 : 7 + self.num_dofs + 6 + 2 + 7]

        return current_obs_buffer_dict

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary."""
        current_obs_dict = {}
        for key in self.obs_dict:
            # obs_list = sorted(self.obs_dict[key])
            if key == 'loco':
                obs_list = ['sin_phase', 'cos_phase', 'base_ang_vel', 'projected_gravity', 'command_lin_vel', 'command_ang_vel',
                            'dof_pos', 'dof_vel', 'last_actions_loco']
            if key == 'manip':
                obs_list = ['command_ee', 'joint_pos', 'joint_vel', "last_actions_manip"]
            current_obs_dict[key] = np.concatenate(
                [current_obs_buffer_dict[obs_name] * self.obs_scales[obs_name] for obs_name in obs_list], axis=1
            )
        return current_obs_dict

    def setup_policy_loco(self, model_path_loco):
        """Setup ONNX policy model."""
        if model_path_loco is None:
            self.policy_loco = lambda x: np.zeros((1, self.num_lower_dofs), dtype=np.float32)
            self.onnx_policy_session_loco = None
            self.onnx_input_names_loco = None
            self.onnx_output_names_loco = None
            return
        self.onnx_policy_session_loco = onnxruntime.InferenceSession(model_path_loco)
        input_names_loco = [inp.name for inp in self.onnx_policy_session_loco.get_inputs()]
        output_names_loco = [out.name for out in self.onnx_policy_session_loco.get_outputs()]

        self.onnx_input_names_loco = input_names_loco
        self.onnx_output_names_loco = output_names_loco

        def policy_act_loco(obs_dict_loco):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {name: obs_dict_loco['loco'] for name in self.onnx_input_names_loco}
            outputs = self.onnx_policy_session_loco.run(self.onnx_output_names_loco, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy_loco = policy_act_loco

    def setup_policy_manip(self, model_path_manip):
        """Setup ONNX policy model."""
        if model_path_manip is None:
            self.policy_manip = lambda x: np.zeros((1, 7), dtype=np.float32)
            self.onnx_policy_session_manip = None
            self.onnx_input_names_manip = None
            self.onnx_output_names_manip = None
            return

        self.onnx_policy_session_manip = onnxruntime.InferenceSession(model_path_manip)
        input_names_manip = [inp.name for inp in self.onnx_policy_session_manip.get_inputs()]
        output_names_manip = [out.name for out in self.onnx_policy_session_manip.get_outputs()]

        self.onnx_input_names_manip = input_names_manip
        self.onnx_output_names_manip = output_names_manip

        h_in = np.zeros(*self.onnx_policy_session_manip.get_inputs()[1].shape, dtype=np.float32)
        c_in = np.zeros(*self.onnx_policy_session_manip.get_inputs()[2].shape, dtype=np.float32)

        def policy_act_manip(obs_dict_manip):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {'obs': obs_dict_manip['manip'], 'h_in': h_in, 'c_in': c_in}
            outputs = self.onnx_policy_session_manip.run(self.onnx_output_names_manip, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy_manip = policy_act_manip
        return

    def rl_inference(self, robot_state_data):
        obs_dict_loco, obs_dict_manip = self.prepare_obs_for_rl(robot_state_data)
        # import ipdb; ipdb.set_trace()
        policy_action_loco = self.policy_loco(obs_dict_loco)
        policy_action_manip = self.policy_manip(obs_dict_manip)
        # policy_action = np.clip(policy_action, -100, 100)

        # WBC actions
        self.last_policy_action_loco = policy_action_loco.copy()
        self.last_policy_action_manip = policy_action_manip.copy()

        scaled_policy_action_loco = policy_action_loco * self.policy_action_scale
        scaled_policy_action_loco[0][0] = 0.0  # Set Waist action to zero
        scaled_policy_action_manip = policy_action_manip * self.policy_action_scale

        # Set feet action to zero
        # scaled_policy_action[0][5] = 0.0
        # scaled_policy_action[0][6] = 0.0
        # scaled_policy_action[0][-1] = 0.0
        # scaled_policy_action[0][-2] = 0.0

        # if self.residual_upper_body_action:
        #     scaled_policy_action[:, self.upper_dof_indices] += (
        #         self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
        #     )

        return scaled_policy_action_loco, scaled_policy_action_manip

    def policy_action(self):
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data

        # Apply upper body controller
        alpha = 0.1
        if self.upper_body_controller:
            # Control upper qpos and tau
            if self.get_ready_state:
                pass
            elif not self.use_policy_action:
                pass
            else:
                q = robot_state_data[0, : 7 + self.num_dofs].copy()
                # q[7:] += self.default_dof_angles
                _, tau = self.upper_body_controller.compute_osc_torque(
                    target_left_ee_pose=self.ee_command[0],
                    current_left_arm_motor_q=q,
                    current_left_arm_motor_dq=robot_state_data[0, 7 + self.num_dofs : 7 + self.num_dofs + 6 + self.num_dofs],
                    Kp_pos=self.model_pos_Kp,
                    Kd_pos=self.model_pos_Kd,
                    Kp_ori=self.model_ori_Kp,
                    Kd_ori=self.model_ori_Kd,
                )
                self.cmd_tau[2:9] = tau[2:9] * alpha + self.cmd_tau[2:9] * (1 - alpha)

        # Get policy action

        if self.loco_decimation_counter == 0:
            scaled_policy_action_loco, scaled_policy_action_manip = self.rl_inference(robot_state_data)
            if self.get_ready_state:
                # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
            elif not self.use_policy_action:
                # 2. No Policy Action: set to zero
                q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
            else:
                # 3. Policy Action: apply policy action to current joint angles
                q_target_loco = scaled_policy_action_loco + self.default_dof_angles[16:]
                q_target_manip = scaled_policy_action_manip + self.default_dof_angles[2:9]
                q_target = np.concatenate([np.array([[0.0, 0.0]]),  # AAHead_yaw, Head_pitch
                                           q_target_manip,
                                           np.array([[
                                               -0.6249,  # Left_Shoulder_Pitch
                                               0.6995,  # Left_Shoulder_Roll
                                               1.3938,  # Left_Elbow_Pitch
                                               1.7929,  # Left_Elbow_Yaw
                                               -0.1551,  # Left_Wrist_Pitch
                                               -0.8901,  # Left_Wrist_Yaw
                                               -0.6900,  # Left_Hand_Roll
                                           ]]),
                                           q_target_loco
                                           ], axis=1)

                joint_pos = robot_state_data[:, 7 : 7 + self.num_dofs].copy()
                self.record_joint_pos.append(joint_pos[0])
                joint_torque = robot_state_data[:, 71 + 6 : 71 + 6 + self.num_dofs].copy()
                self.record_joint_torque.append(joint_torque[0])
            # import ipdb; ipdb.set_trace()
            # Clip q target
            if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
                q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

            # Send command
            self.cmd_q = q_target[0]

        self.loco_decimation_counter = (self.loco_decimation_counter + 1) % self.loco_decimation
        self.command_sender.send_command(self.cmd_q.copy(),
                                         self.cmd_dq.copy(),
                                         self.cmd_tau.copy(),
                                         robot_state_data[0, 7 : 7 + self.num_dofs])
        # print(f"flag: {self.command_sender.pd_choice}")

    def _handle_velocity_control(self, keycode):
        """Handle linear velocity control."""
        if not self.stand_command[0, 0]:
            return

        if keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1

    def _handle_angular_velocity_control(self, keycode):
        """Handle angular velocity control."""
        if keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1

    def _handle_ee_command_control(self, keycode):
        """Handle ee command control."""

        if keycode == "t":
            self.ee_command[0, 0] += 0.1
        elif keycode == "g":
            self.ee_command[0, 0] -= 0.1
        elif keycode == "f":
            self.ee_command[0, 1] += 0.1
        elif keycode == "h":
            self.ee_command[0, 1] -= 0.1
        elif keycode == "r":
            self.ee_command[0, 2] += 0.1
        elif keycode == "y":
            self.ee_command[0, 2] -= 0.1

    def update_waypoints(self):
        self.waypoints_left = [
            pin.SE3(self.EE_left_R.astype(np.float64), np.array([self.EE_left_x, self.EE_left_y, self.EE_left_z]))
        ]
        self.waypoints_right = [
            pin.SE3(self.EE_right_R.astype(np.float64), np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z]))
        ]

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == "m":
            self.lin_vel_command[:, 0] = -1.0
            self.logger.info(colored(f"lin_vel_command: {self.lin_vel_command}", "green"))
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)
        elif keycode in ["r", "t", "y", "f", "g", "h"]:
            self._handle_ee_command_control(keycode)

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        if cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "R1+up":
            self.EE_left_x += 0.05
            self.EE_right_x += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+down":
            self.EE_left_x -= 0.05
            self.EE_right_x -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+left":
            self.EE_left_y += 0.02
            self.EE_right_y -= 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "R1+right":
            self.EE_left_y -= 0.02
            self.EE_right_y += 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "X+up":
            self.EE_left_z += 0.05
            self.EE_right_z += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+down":
            self.EE_left_z -= 0.05
            self.EE_right_z -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+left":
            self.degrees -= 5
            self.theta = np.radians(self.degrees)
            self.EE_left_R = np.array(
                [
                    [np.cos(-self.theta), -np.sin(-self.theta), 0],
                    [np.sin(-self.theta), np.cos(-self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.EE_right_R = np.array(
                [[np.cos(self.theta), -np.sin(self.theta), 0], [np.sin(self.theta), np.cos(self.theta), 0], [0, 0, 1]]
            )
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "X+right":
            self.degrees += 5
            self.theta = np.radians(self.degrees)
            self.EE_left_R = np.array(
                [
                    [np.cos(-self.theta), -np.sin(-self.theta), 0],
                    [np.sin(-self.theta), np.cos(-self.theta), 0],
                    [0, 0, 1],
                ]
            )
            self.EE_right_R = np.array(
                [[np.cos(self.theta), -np.sin(self.theta), 0], [np.sin(self.theta), np.cos(self.theta), 0], [0, 0, 1]]
            )
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

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
        print(f"Linear velocity command: {self.lin_vel_command}")
        print(f"Angular velocity command: {self.ang_vel_command}")
        print(f"EE command: {self.ee_command}")

    def run(self):
        """Main run loop for the policy."""
        try:
            while True:
                if self.use_joystick and self.wc_msg is not None:
                    self.process_joystick_input()
                self.policy_action()
                self.rate_manip.sleep()
        except KeyboardInterrupt:
            self.plot_recorded_data()
            pass

    def plot_recorded_data(self):
        self.record_joint_pos = np.array(self.record_joint_pos)
        self.record_joint_torque = np.array(self.record_joint_torque)
        # self.record_base_acc = np.array(self.record_base_acc)

        if not hasattr(self, "model_path") or self.model_path is None:
            self.model_path = "test"

        policy_name = self.model_path.split('/')[-1].split('.')[0]

        plot_dir = f"record/{policy_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(plot_dir)

        np.savetxt(f"{plot_dir}/recorded_joint_pos.txt", self.record_joint_pos[:, 2:9], fmt='%.6f', delimiter='\t')
        np.savetxt(f"{plot_dir}/recorded_joint_torque.txt", self.record_joint_torque[:, 2:9], fmt='%.6f', delimiter='\t')
        # np.savetxt(f"{plot_dir}/recorded_base_acc.txt", self.record_base_acc, fmt='%.6f', delimiter='\t')

        print(f"Recorded data saved to {plot_dir}")

        fig, axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
        for i in range(7):
            axs[i].plot(self.record_joint_pos[:, i + 2], color='C' + str(i), label='left')
            # axs[i].plot(self.record_joint_pos[:, i+23], color='C'+str(i+6), label='right')
            axs[i].set_ylabel(f'arm joint {i+1}')
            axs[i].grid(True)
            axs[i].legend(loc='upper right')
            if i == 0:
                axs[i].set_title('Joint Positions Over Time')
        axs[-1].set_xlabel('Step')
        plt.savefig(f"{plot_dir}/recorded_joint_pos.png")
        plt.tight_layout()
        # plt.show()

        fig, axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
        for i in range(7):
            axs[i].plot(self.record_joint_torque[:, i + 2], color='C' + str(i), label='left')
            # axs[i].plot(self.record_joint_torque[:, i+23], color='C'+str(i+6), label='right')
            axs[i].set_ylabel(f'arm joint {i+1}')
            axs[i].grid(True)
            axs[i].legend(loc='upper right')
            if i == 0:
                axs[i].set_title('Joint Torques Over Time')
        axs[-1].set_xlabel('Step')
        plt.savefig(f"{plot_dir}/recorded_joint_torque.png")
        plt.tight_layout()
        # plt.show()

        # fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        # for i in range(3):
        #     axs[i].plot(self.record_base_acc[:, i], color='C'+str(i))
        #     axs[i].set_ylabel(f'base acceleration {i+1}')
        #     axs[i].grid(True)
        #     if i == 0:
        #         axs[i].set_title('Base Accelerations Over Time')
        # axs[-1].set_xlabel('Step')
        # plt.savefig(f"{plot_dir}/recorded_base_acc.png")
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/t1/t1_29dof_loco_manip.yaml", help="config file")
    parser.add_argument("--model_path_loco", type=str, help="path to the ONNX model file")
    parser.add_argument("--model_path_manip", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path_loco = args.model_path_loco if args.model_path_loco else config.get("model_path_loco")
    model_path_manip = args.model_path_manip if args.model_path_manip else config.get("model_path_manip")

    if model_path_loco == 'None':
        model_path_loco = None
    if model_path_manip == 'None':
        model_path_manip = None

    policy = LocoManipPolicyModel(
        config=config,
        model_path_loco=model_path_loco, model_path_manip=model_path_manip,
        rl_rate_loco=100, rl_rate_manip=100, model_rate_manip=500,
        policy_action_scale=1
    )
    policy.run()
