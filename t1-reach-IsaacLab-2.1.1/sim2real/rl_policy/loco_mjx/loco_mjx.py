import argparse
import functools
import os
import random
import sys
import time
from threading import Lock

import jax
import jax.numpy as jp
import mediapy as media
import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
import pinocchio as pin
import yaml
from absl import app, flags, logging
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from mujoco_playground._src.locomotion.t1 import t1_constants as consts
from orbax import checkpoint as ocp
from termcolor import colored

from digit_dprl.rl import registry
from digit_dprl.utils import clamp, env_setup
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.arm_ik.robot_arm_ik_g1_23dof import G1_29_ArmIK_NoWrists

env_setup()

sys.path.append("../")
sys.path.append("./rl_policy")

ENV_NAME = "T1_29DoF_Locomotion"


class LocoMjxPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = self.config.get(
            "residual_upper_body_action", False)

        self.upper_body_controller = None
        if self.config.get("use_upper_body_controller", False):
            self.init_upper_body_controller()

        self.rl_dt = 1.0 / self.config.get("rl_rate", 50)
        self.phase_dt = 2 * np.pi * self.rl_dt * 1.5
        self.phase = np.array([0.0, np.pi])

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command

        return current_obs_dict

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] == "g1_29dof":
            self.upper_body_controller = G1_29_ArmIK_NoWrists(
                Unit_Test=False, Visualization=False, robot_config=self.config
            )
        else:
            self.logger.error("Unsupported robot type: %s",
                              self.config["ROBOT_TYPE"])
        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        # Initialize waypoints
        self.degrees = -0
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array(
            [
                [np.cos(-self.theta), -np.sin(-self.theta), 0],
                [np.sin(-self.theta), np.cos(-self.theta), 0],
                [0, 0, 1],
            ]
        )
        self.EE_right_R = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), 0],
             [np.sin(self.theta), np.cos(self.theta), 0], [0, 0, 1]]
        )
        self.EE_left_x = 0.30
        self.EE_right_x = 0.30
        self.EE_left_y = 0.13
        self.EE_right_y = -0.13
        self.EE_left_z = 0.08
        self.EE_right_z = 0.08
        self.update_waypoints()
        # Initialize external force
        self.EE_efrc_L = np.array([0, 0, 0, 0, 0, 0])
        self.EE_efrc_R = np.array([0, 0, 0, 0, 0, 0])
        # Initialize interpolated positions and orientations
        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation,
        )

    def setup_policy(self, model_path):
        # Load environment configuration
        env_cfg = registry.get_env_config(ENV_NAME)
        ppo_params = registry.get_ppo_config(ENV_NAME)
        env_cfg.episode_length = 1000  # or any desired number of steps
        env_cfg.is_noise = False
        env_cfg.command_config.enable_sample = False
        env_cfg.push_config.enable = False
        env_cfg.noise_config.level = 0.0

        env = registry.get_env_class(ENV_NAME)(config=env_cfg)
        self.obs_size = env.observation_size
        self.act_size = env.action_size
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory,  # type: ignore
            preprocess_observations_fn=running_statistics.normalize,
        )

        # Load checkpoint
        ckpt_path = epath.Path(model_path).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt_data = checkpointer.restore(ckpt_path)
        ckpt_data = tuple(ckpt_data)

        # Convert ckpt_data to RunningStatisticsState
        running_stats = running_statistics.RunningStatisticsState(
            mean=ckpt_data[0]["mean"],
            std=ckpt_data[0]["std"],
            count=ckpt_data[0]["count"],
            summed_variance=ckpt_data[0]["summed_variance"],
        )

        policy_params = ckpt_data[1]
        value_params = ckpt_data[2]
        params = (running_stats, policy_params, value_params)

        ppo_network = network_factory(self.obs_size, self.act_size)
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_inference_fn(params, deterministic=True)
        self.jit_infer = inference_fn

        def policy_func(obs):
            return np.array(self.jit_infer({
                "state": jp.array(obs),
                "privileged_state": jp.zeros(self.obs_size["privileged_state"])
            }, jax.random.PRNGKey(0))[0]).reshape(1, -1)

        self.policy = policy_func

    def rl_inference(self, robot_state_data):
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        se2_cmd = np.concatenate([
            obs_buffer_dict["command_lin_vel"][0],
            obs_buffer_dict["command_ang_vel"][0]
        ])
        phase_tp1 = self.phase + self.phase_dt * 10
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(se2_cmd) < 0.01:
            self.phase = np.full(2, np.pi)
        phase_encoded = np.concatenate([
            np.cos(self.phase),
            np.sin(self.phase)
        ])
        obs = np.concatenate([
            obs_buffer_dict["base_ang_vel"][0],
            obs_buffer_dict["projected_gravity"][0],
            se2_cmd,
            obs_buffer_dict["dof_pos"][0],
            obs_buffer_dict["dof_vel"][0],
            self.last_policy_action[0],
            self.last_last_policy_action[0],
            phase_encoded
        ])

        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        # WBC actions
        self.last_last_policy_action = self.last_policy_action.copy()
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos -
                self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data
        # Apply upper body controller
        if self.upper_body_controller:
            # Control upper qpos and tau
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
            # Zero out wrist joints
            wrist_joint_indices = [19, 20, 21, 26, 27, 28]
            for idx in wrist_joint_indices:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

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
            q_target = scaled_policy_action + self.default_dof_angles
        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(
                q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7: 7 + self.num_dofs])

    def update_waypoints(self):
        self.waypoints_left = [
            pin.SE3(self.EE_left_R.astype(np.float64), np.array(
                [self.EE_left_x, self.EE_left_y, self.EE_left_z]))
        ]
        self.waypoints_right = [
            pin.SE3(self.EE_right_R.astype(np.float64), np.array(
                [self.EE_right_x, self.EE_right_y, self.EE_right_z]))
        ]

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

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        if cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "R1+up":
            self.EE_left_x += 0.05
            self.EE_right_x += 0.05
            self.update_waypoints()
            self.logger.info(
                colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+down":
            self.EE_left_x -= 0.05
            self.EE_right_x -= 0.05
            self.update_waypoints()
            self.logger.info(
                colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+left":
            self.EE_left_y += 0.02
            self.EE_right_y -= 0.02
            self.update_waypoints()
            self.logger.info(
                colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "R1+right":
            self.EE_left_y -= 0.02
            self.EE_right_y += 0.02
            self.update_waypoints()
            self.logger.info(
                colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "X+up":
            self.EE_left_z += 0.05
            self.EE_right_z += 0.05
            self.update_waypoints()
            self.logger.info(
                colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+down":
            self.EE_left_z -= 0.05
            self.EE_right_z -= 0.05
            self.update_waypoints()
            self.logger.info(
                colored(f"EE Z command: {self.EE_left_z}", "green"))
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
                [[np.cos(self.theta), -np.sin(self.theta), 0],
                 [np.sin(self.theta), np.cos(self.theta), 0], [0, 0, 1]]
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
                [[np.cos(self.theta), -np.sin(self.theta), 0],
                 [np.sin(self.theta), np.cos(self.theta), 0], [0, 0, 1]]
            )
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(
                colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(
                colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(
                colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str,
                        default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str,
                        help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get(
        "model_path")
    if not model_path:
        raise ValueError(
            "model_path must be provided either via --model_path argument or in config file")

    policy = LocoMjxPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=1.0
    )
    policy.run()
