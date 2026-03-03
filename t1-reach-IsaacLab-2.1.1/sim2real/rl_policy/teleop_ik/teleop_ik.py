import asyncio
import threading
from sim2real.utils.arm_ik.mink_t1_29dof_teleop import T1_29_ArmIK_Teleop
from termcolor import colored
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
import pinocchio as pin
import os
import sys
import time
from vuer import Vuer, VuerSession
from vuer.schemas import MotionControllers
from asyncio import sleep
from scipy.spatial.transform import Rotation as R

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

lx = ly = lz = 0.0
lroll = lpitch = lyaw = 0.0
lhold = False

rx = ry = rz = 0.0
rroll = rpitch = ryaw = 0.0
rhold = False


def pose_to_rpy(flat_pose):
    """
    Convert a 16-element flat 4x4 transformation matrix into position and roll-pitch-yaw angles.

    Args:
        flat_pose (list or array): 16 elements, 4x4 matrix in column-major (Vuer format)

    Returns:
        x, y, z : translation coordinates
        roll, pitch, yaw : in radians
    """
    mat = np.array(flat_pose).reshape((4, 4), order='F')
    t = mat[:3, 3]
    x, y, z = t[0], t[1], t[2]
    rot_mat = mat[:3, :3]
    r = R.from_matrix(rot_mat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    return x, y, z, roll, pitch, yaw


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
        self.vuer_app = Vuer()
        self.vuer_app.add_handler("CONTROLLER_MOVE", self.cb_controller_move)
        self.vuer_app.spawn(self.cb_spawn, start=False)
        threading.Thread(target=self.loop_run_vuer, daemon=True).start()

        self.upper_body_controller = T1_29_ArmIK_Teleop(
            self.config.get("ik_xml_path")
        )
        self.upper_body_controller.start_ik()

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
            # q_target = scaled_policy_action + self.default_dof_angles
            true_act = scaled_policy_action + self.default_dof_angles
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            # q_target[:, self.lower_dof_indices] = true_act[:, self.lower_dof_indices]
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

    def loop_run_vuer(self):
        self.vuer_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.vuer_loop)
        self.vuer_loop.run_until_complete(self.vuer_app.run())

    async def cb_controller_move(self, event, session):
        if event.key != "motion-controller":
            return
        global lx, ly, lz, lroll, lpitch, lyaw, lhold
        global rx, ry, rz, rroll, rpitch, ryaw, rhold
        ik = self.upper_body_controller

        if "squeeze" not in event.value['leftState']:
            return

        with self.upper_body_controller.get_viewer().lock():
            if event.value['leftState']["squeeze"]:
                lx, ly, lz, lroll, lpitch, lyaw = pose_to_rpy(
                    event.value['left'])
                if not lhold:
                    ik.reframe_mocap("left_hand_target", "head", np.array(
                        [-lz, -lx, ly]), np.array([-lyaw, -lroll, lpitch]))
                    lhold = True
                ik.sync_mocap("left_hand_target", "head", np.array(
                    [-lz, -lx, ly]), np.array([-lyaw, -lroll, lpitch]))
            else:
                lhold = False

            if event.value['rightState']["squeeze"]:
                rx, ry, rz, rroll, rpitch, ryaw = pose_to_rpy(
                    event.value['right'])
                if not rhold:
                    ik.reframe_mocap("right_hand_target", "head", np.array(
                        [-rz, -rx, ry]), np.array([-ryaw, -rroll, rpitch]))
                    rhold = True
                ik.sync_mocap("right_hand_target", "head", np.array(
                    [-rz, -rx, ry]), np.array([-ryaw, -rroll, rpitch]))
            else:
                rhold = False

    async def cb_spawn(self, session: VuerSession):
        # Important: enable streaming
        print("Vuer running!")
        session.upsert(MotionControllers(
            stream=True, key="motion-controller", left=True, right=True))
        while True:
            await sleep(1)


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

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
    policy.upper_body_controller.stop_ik()
