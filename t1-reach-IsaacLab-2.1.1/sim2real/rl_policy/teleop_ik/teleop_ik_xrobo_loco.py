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
from sim2real.utils.arm_ik.mink_t1_29dof_teleop import T1_29_ArmIK_Teleop
from booster_robotics_sdk_python import GripperControlMode, GripperMotionParameter, B1HandIndex, RobotMode
import mink as ik


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


def mat_to_quat(mat: np.array) -> np.array:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(mat)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    else:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quat_to_rpy(quat: np.array) -> np.array:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw angles."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def rpy_to_quat(rpy: np.array) -> np.array:
    """Convert roll-pitch-yaw angles to quaternion (w, x, y, z)."""
    roll, pitch, yaw = rpy

    # Half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # Quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quat_to_mat(quat):
    # quat is [w, x, y, z]
    w, x, y, z = quat

    # normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # rotation matrix from quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])


def headset_to_world(wxyzxyz, linear_scale_factor=0.6):
    # Extract position and quaternion
    headset_pos = wxyzxyz[4:]
    headset_quat = wxyzxyz[:4]

    world_pos = np.array([-headset_pos[2], -headset_pos[0], headset_pos[1]]) * linear_scale_factor
    transform_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0])

    # swap y and z components, then negate x and z to fix roll/yaw inversion
    adjusted_quat = np.array([headset_quat[0], -headset_quat[1], headset_quat[3], -headset_quat[2]])
    world_quat = quat_multiply(transform_quat, adjusted_quat)

    return np.concatenate([world_quat, world_pos])


def yaw_rotate(wxyzxyz, yaw_rad):
    """rotate pose around z-axis by yaw_rad"""
    quat = wxyzxyz[:4]
    pos = wxyzxyz[4:]

    # yaw rotation quaternion around z-axis (negated for correct direction)
    r, p, y = quat_to_rpy(quat)
    rotated_quat = rpy_to_quat(np.array([r, p, y + yaw_rad]))
    rotated_pos = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0.],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0.],
        [0., 0., 1.]
    ]) @ pos

    return np.concatenate([rotated_quat, rotated_pos])


class TeleopXRoboLocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=1.0, headless=True
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.residual_upper_body_action = False
        self.last_policy_action = np.zeros((1, 13))
        self.obs_history = {}
        self.obs_history_len = 8
        self.single_obs_len = 82  # action not include waist
        if headless:
            self.logger.warning("Using headless mode.")
        self.init_upper_body_controller(headless=headless)
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command

        return current_obs_dict

    def init_upper_body_controller(self, headless):
        # save commands
        xrt.init()
        self.xrt_first_sync = True
        self.lctrl_pose = np.zeros(7)
        self.lctrl_hold = False
        self.lctrl_trigger = 0
        self.rctrl_pose = np.zeros(7)
        self.rctrl_hold = False
        self.rctrl_trigger = 0
        self.lctrl_joystick = np.zeros(2)
        self.lctrl_x_prev = False
        self.rctrl_a_prev = False
        self.headset_pose = np.zeros(7)
        self.offset_yaw_rad = 0.0
        self.offset_yaw_rad_starting = None

        self.upper_body_controller = T1_29_ArmIK_Teleop(
            self.config.get("ik_xml_path"), headless
        )

        self.upper_body_controller.start_ik()
        threading.Thread(target=self.loop_xrobo).start()

    def rl_inference(self, robot_state_data):
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        phase_time = self._get_obs_phase_time()

        obs_buffer_dict["command_lin_vel"] = self.lin_vel_command.reshape(1, 2)
        obs_buffer_dict["command_ang_vel"] = self.ang_vel_command.reshape(1, 1)
        obs_buffer_dict["sin_phase"] = np.sin(2 * phase_time * np.pi)
        obs_buffer_dict["cos_phase"] = np.cos(2 * phase_time * np.pi)
        obs_buffer_dict["last_policy_action"] = self.last_policy_action.copy()

        obs_list = [
            'sin_phase', 'cos_phase',
            'base_ang_vel', 'projected_gravity',
            'command_lin_vel', 'command_ang_vel',
            'dof_pos', 'dof_vel', 'last_policy_action'
        ]

        # print(f"LinVel: {obs_buffer_dict['command_lin_vel']}, AngVel: {obs_buffer_dict['command_ang_vel']}")

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
            self.upper_body_controller.set_solving(False)
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
            with self.upper_body_controller.get_datalock():  # sync in init state
                self.upper_body_controller.sync_qpos_upper(self.state_processor.q[7: 7 + self.num_upper_dofs])
                self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")
                self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            self.upper_body_controller.set_solving(False)
            q_target = robot_state_data[:, 7: 7 + self.num_dofs]
            with self.upper_body_controller.get_datalock():  # sync in init state
                self.upper_body_controller.sync_qpos_upper(self.state_processor.q[7: 7 + self.num_upper_dofs])
                self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")
                self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")
        else:
            # 3. Policy Action: apply policy action to current joint angles
            self.upper_body_controller.set_solving(True)
            true_act = scaled_policy_action + self.default_dof_angles[self.lower_dof_indices]
            true_act[0][0] = 0.0
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
        # if self.config.get("INTERFACE") != "lo":
        #     lparam = GripperMotionParameter()
        #     lparam.position = int(self.lctrl_trigger)
        #     lparam.force = 50
        #     lparam.speed = 100

        #     rparam = GripperMotionParameter()
        #     rparam.position = int(self.rctrl_trigger)
        #     rparam.force = 50
        #     rparam.speed = 100
        #     print(lparam.position)
        #     self.command_sender.client.ControlGripper(lparam, GripperControlMode.kPosition, B1HandIndex.kLeftHand)
        #     self.command_sender.client.ControlGripper(rparam, GripperControlMode.kPosition, B1HandIndex.kRightHand)

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

    def loop_xrobo(self):
        while True:
            lctrl_pose = xrt.get_left_controller_pose()
            lctrl_axis = xrt.get_left_axis()
            rctrl_axis = xrt.get_right_axis()
            l_grip = xrt.get_left_grip()
            l_trigger = xrt.get_left_trigger()
            x_button = xrt.get_X_button()
            a_button = xrt.get_A_button()
            rctrl_pose = xrt.get_right_controller_pose()
            r_grip = xrt.get_right_grip()
            r_trigger = xrt.get_right_trigger()
            headset_pose = xrt.get_headset_pose()
            if all([x == 0 for x in lctrl_pose]) or all([x == 0 for x in rctrl_pose]) or all([x == 0 for x in headset_pose]):
                self.logger.error("Failed to connect to XRoboTool.")
                time.sleep(1.0)
                continue  # all zeros, not connected

            with self.upper_body_controller.get_datalock():
                self.headset_pose = headset_to_world(np.concatenate([headset_pose[3:], headset_pose[:3]]), linear_scale_factor=1.0)
                if self.offset_yaw_rad_starting is None:
                    self.offset_yaw_rad_starting = quat_to_rpy(self.headset_pose[:4])[2]
                self.lctrl_pose = headset_to_world(np.concatenate([lctrl_pose[3:], lctrl_pose[:3]]), linear_scale_factor=0.9)
                self.rctrl_pose = headset_to_world(np.concatenate([rctrl_pose[3:], rctrl_pose[:3]]), linear_scale_factor=0.9)
                offset = self.offset_yaw_rad - self.offset_yaw_rad_starting
                self.lctrl_pose = yaw_rotate(self.lctrl_pose, -offset)
                self.rctrl_pose = yaw_rotate(self.rctrl_pose, -offset)
                # self.upper_body_controller.set_mocap_wxyzxyz("left_hand_target", self.lctrl_pose)
                # self.upper_body_controller.set_mocap_wxyzxyz("right_hand_target", self.rctrl_pose)

                self.lctrl_joystick[:] = lctrl_axis

                if l_grip > 0.99:
                    if not self.lctrl_hold:
                        self.upper_body_controller.reframe_mocap("left_hand_target", self.lctrl_pose)
                        self.lctrl_hold = True
                    self.upper_body_controller.sync_mocap("left_hand_target", self.lctrl_pose)
                else:
                    self.lctrl_hold = False
                    self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")

                if r_grip > 0.99:
                    if not self.rctrl_hold:
                        self.upper_body_controller.reframe_mocap("right_hand_target", self.rctrl_pose)
                        self.rctrl_hold = True
                    self.upper_body_controller.sync_mocap("right_hand_target", self.rctrl_pose)
                else:
                    self.rctrl_hold = False
                    self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")

            if x_button:
                if not self.lctrl_x_prev:
                    self.lctrl_x_prev = True
                    self.stand_command = np.array(
                        [[1 if self.stand_command[0][0] == 0 else 0]]
                    )  # switch stepping / standing
                    self.logger.info(colored(f"Switched to: {'Standing' if self.stand_command[0][0] == 0 else 'Walking'}", "green"))
            else:
                self.lctrl_x_prev = False

            if a_button:
                if not self.rctrl_a_prev:
                    self.rctrl_a_prev = True
                    self.offset_yaw_rad = quat_to_rpy(self.headset_pose[:4])[2]
                    self.logger.info(colored("VR reference yaw angle resetted.", "green"))
            else:
                self.rctrl_a_prev = False

            self.lin_vel_command[0][0] = -lctrl_axis[0] * 0.8
            self.lin_vel_command[0][1] = lctrl_axis[1] * 1.0
            self.ang_vel_command[0][0] = -rctrl_axis[0] * 1.5
            self.base_height_command[0][0] = \
                np.clip(
                    self.base_height_command[0][0]
                    + 1e-4 * rctrl_axis[1],
                0.5, 0.7
            )
            self.lctrl_trigger = l_trigger * 800.0 + 100.0
            self.rctrl_trigger = r_trigger * 800.0 + 100.0


def signal_handler(sig, frame):
    policy.upper_body_controller.stop_ik()
    xrt.close()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX")
    parser.add_argument('--viewer', action='store_false', dest='headless', help='Enable viewer')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = TeleopXRoboLocoManipPolicy(
        config=config,
        model_path=model_path,
        rl_rate=50,
        policy_action_scale=1.0,
        headless=args.headless
    )

    signal.signal(signal.SIGINT, signal_handler)
    policy.run()
