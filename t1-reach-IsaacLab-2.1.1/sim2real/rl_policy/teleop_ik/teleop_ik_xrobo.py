import os
import sys
from pathlib import Path

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


import argparse
import asyncio
import numpy as np
import os
import sys
import threading
import time
import yaml
from asyncio import sleep
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import pinocchio as pin
import xrobotoolkit_sdk as xrt
from termcolor import colored
from vuer import Vuer, VuerSession
from vuer.schemas import MotionControllers

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.arm_ik.mink_t1_29dof_teleop import T1_29_ArmIK_Teleop


def pose_to_rpy(pose):
    x, y, z = pose[0], pose[1], pose[2]
    r = R.from_quat(pose[3:])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    return x, y, z, roll, pitch, yaw


def relative_pose(p_Target_W, p_Ref_W):
    """
    compute relative pose from reference to target
    input: 7-element arrays [x,y,z,qx,qy,qz,qw]
    output: relative pose [x,y,z,qx,qy,qz,qw]
    """
    # extract positions and quaternions
    pos_target = np.array(p_Target_W[:3])
    pos_ref = np.array(p_Ref_W[:3])
    quat_target = np.array(p_Target_W[3:])  # [qx,qy,qz,qw]
    quat_ref = np.array(p_Ref_W[3:])

    # relative position: rotate (target - ref) by inverse of ref rotation
    pos_diff = pos_target - pos_ref
    R_ref = quat_to_rotation_matrix(quat_ref)
    rel_pos = R_ref.T @ pos_diff

    # relative quaternion: q_rel = q_ref^(-1) * q_target
    quat_ref_inv = quat_inverse(quat_ref)
    rel_quat = quat_multiply(quat_ref_inv, quat_target)

    return np.concatenate([rel_pos, rel_quat])


def quat_to_rotation_matrix(q):
    """convert quaternion [qx,qy,qz,qw] to 3x3 rotation matrix"""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R


def quat_inverse(q):
    """compute quaternion inverse (conjugate for unit quaternions)"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1, q2):
    """multiply two quaternions [qx,qy,qz,qw]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])


class LocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = False
        self.init_upper_body_controller()

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command

        return current_obs_dict

    def init_upper_body_controller(self):
        # save commands
        xrt.init()
        self.xrt_first_sync = True
        self.lctrl_pose = np.zeros(7)
        self.lctrl_hold = False
        self.rctrl_pose = np.zeros(7)
        self.rctrl_hold = False
        self.lctrl_joystick = np.zeros(2)
        self.lctrl_x_prev = False
        self.headset_pose = np.zeros(7)

        self.upper_body_controller = T1_29_ArmIK_Teleop(
            self.config.get("ik_xml_path")
        )
        self.upper_body_controller.start_ik()
        threading.Thread(target=self.loop_xrobo, daemon=True).start()

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        # import ipdb; ipdb.set_trace()
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

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
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data
        # Apply upper body controller
        if self.upper_body_controller:
            # Control upper qpos and tau
            upper_body_qpos = self.upper_body_controller.get_qpos_upper()
            true_qpos = upper_body_qpos
            self.ref_upper_dof_pos = true_qpos.reshape(1, -1)

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
            true_act = scaled_policy_action + self.default_dof_angles
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            q_target[:, self.lower_dof_indices] = true_act[:,
                                                           self.lower_dof_indices]
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

    def loop_xrobo(self):
        try:
            while True:
                lctrl_pose = xrt.get_left_controller_pose()
                lctrl_axis = xrt.get_left_axis()
                rctrl_axis = xrt.get_right_axis()
                l_grip = xrt.get_left_grip()
                x_button = xrt.get_X_button()
                rctrl_pose = xrt.get_right_controller_pose()
                r_grip = xrt.get_right_grip()
                headset_pose = xrt.get_headset_pose()
                if all([x == 0 for x in lctrl_pose]) or all([x == 0 for x in rctrl_pose]) or all([x == 0 for x in headset_pose]):
                    self.logger.error("Failed to connect to XRoboTool.")
                    return  # all zeros, not connected

                with self.upper_body_controller.get_viewer().lock():
                    self.headset_pose[:] = headset_pose

                    _, _, _, r, p, y = pose_to_rpy(self.headset_pose)
                    if self.xrt_first_sync:
                        self.xrt_first_sync = False
                        self.upper_body_controller.reframe_mocap(
                            "head_target", "world",
                            np.zeros(3),
                            np.array([-y, -r, p])
                        )
                    # self.upper_body_controller.sync_mocap(
                    #     "head_target", "world",
                    #     np.zeros(3),
                    #     np.array([0.0, -r, p])
                    # )

                    # lctrl, rctrl pose should be in headset frame
                    self.lctrl_pose = relative_pose(
                        np.array(lctrl_pose), self.headset_pose
                    )
                    self.rctrl_pose = relative_pose(
                        np.array(rctrl_pose), self.headset_pose
                    )
                    self.lctrl_joystick[:] = lctrl_axis

                    if l_grip > 0.99:
                        lx, ly, lz, lroll, lpitch, lyaw = pose_to_rpy(
                            self.lctrl_pose)
                        if not self.lctrl_hold:
                            self.upper_body_controller.reframe_mocap(
                                "left_hand_target", "head",
                                np.array([-lz, -lx, ly]),
                                np.array([-lyaw, -lroll, lpitch]))
                            self.lctrl_hold = True
                        self.upper_body_controller.sync_mocap(
                            "left_hand_target", "head",
                            np.array([-lz, -lx, ly]),
                            np.array([-lyaw, -lroll, lpitch]))
                    else:
                        self.lctrl_hold = False

                    if r_grip > 0.99:
                        rx, ry, rz, rroll, rpitch, ryaw = pose_to_rpy(
                            self.rctrl_pose)
                        if not self.rctrl_hold:
                            self.upper_body_controller.reframe_mocap(
                                "right_hand_target", "head",
                                np.array([-rz, -rx, ry]),
                                np.array([-ryaw, -rroll, rpitch])
                            )
                            self.rctrl_hold = True
                        self.upper_body_controller.sync_mocap(
                            "right_hand_target", "head",
                            np.array([-rz, -rx, ry]),
                            np.array([-ryaw, -rroll, rpitch]))
                    else:
                        self.rctrl_hold = False

                if x_button:
                    if not self.lctrl_x_prev:
                        self.lctrl_x_prev = True
                        self.stand_command = np.array(
                            [[1 if self.stand_command[0][0] == 0 else 0]]
                        )  # switch stepping / standing
                else:
                    self.lctrl_x_prev = False

                self.lin_vel_command[0][1] = lctrl_axis[0] * 0.3
                self.lin_vel_command[0][0] = lctrl_axis[1] * 0.2
                self.ang_vel_command[0][0] = -rctrl_axis[0] * 0.5
                self.base_height_command[0][0] = np.clip(
                    self.base_height_command[0][0]
                    + 1e-4 * rctrl_axis[1],
                    0.5, 0.7
                )
        finally:
            xrt.close()


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

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
    policy.upper_body_controller.stop_ik()
