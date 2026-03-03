import os
import sys
from pathlib import Path
import time
import signal
import sys
from datetime import datetime
import csv

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import yaml
from termcolor import colored
import mujoco as mj
from scipy.spatial.transform import Rotation as R_scipy

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from booster_robotics_sdk_python import RobotMode


def quat_to_mat(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])


class BoosterTeleop(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=50, policy_action_scale=1.0, headless=True):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.residual_upper_body_action = False
        self.obs_history = {}
        self.obs_history_len = 8
        self.single_obs_len = 48
        self.last_policy_action = np.zeros((1, 12))
        self.single_action_len = 12
        self.last_scaled_action = np.zeros(12)
        self.last_actual_joint_pos = np.zeros(12)
        self.last_com = np.zeros(3)
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))
        # while True:
        #     if self.command_sender.client.ChangeMode(RobotMode.kCustom) == 0:
        #         break
        print("Sucessfully connected!")
        self.command_stsw_pose = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.command_countdown = np.array([[0.0]])
        self.command_foot_indicator = np.array([[0.0]])
        self.startup = True

        self.in_stepping = False
        self.step_length_x = 0.25
        self.step_length_y = 0.1
        self.step_length_yaw = 0.4
        self.step_time = 0.8
        self.cooldown_time = 0.8 # Cooldown period after each step
        self.start_stepping_time = None
        self.dir = 1
        self.left_stance_default = np.array([[0.0, -0.2, 0.0, 0.0]])
        self.right_stance_default = np.array([[0.0, 0.2, 0.0, 0.0]])

        self.model = mj.MjModel.from_xml_path(self.config.get("ik_xml_path"))
        self.data = mj.MjData(self.model)
        mj.mj_resetDataKeyframe(self.model, self.data, 0)

        # Initialize data logging
        self._init_data_logger()
        self.timestep = 0
        self.last_logged_foot_indicator = None

        # For velocity calculation via finite differences
        self.last_left_foot_pos = np.zeros(3)
        self.last_right_foot_pos = np.zeros(3)
        self.dt = 1.0 / 200.0  # Control frequency (200 Hz typical for MuJoCo)

        # For IMU data storage
        self.last_imu_gyro = np.zeros(3)
        self.last_imu_acc = np.zeros(3)

    def _init_data_logger(self):
        """Initialize data logging infrastructure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(self.config.get("log_dir", "./foot_track_logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.log_dir / f"foot_track_data_{timestamp}.csv"
        self.csv_writer = None
        self.csv_file_handle = None
        self._init_csv_file()

    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        try:
            self.csv_file_handle = open(self.csv_file, 'w', newline='')
            self.fieldnames = [
                'timestep',
                'keyboard_key',
                'command_foot_indicator',
                'command_countdown_0',
                'command_stsw_pose_0',
                'command_stsw_pose_1',
                'command_stsw_pose_2',
                'command_stsw_pose_3',
                'action_0', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5',
                'action_6', 'action_7', 'action_8', 'action_9', 'action_10', 'action_11',
                'scaled_action_0', 'scaled_action_1', 'scaled_action_2', 'scaled_action_3', 'scaled_action_4', 'scaled_action_5',
                'scaled_action_6', 'scaled_action_7', 'scaled_action_8', 'scaled_action_9', 'scaled_action_10', 'scaled_action_11',
                'joint_pos_rel_0', 'joint_pos_rel_1', 'joint_pos_rel_2', 'joint_pos_rel_3', 'joint_pos_rel_4', 'joint_pos_rel_5',
                'joint_pos_rel_6', 'joint_pos_rel_7', 'joint_pos_rel_8', 'joint_pos_rel_9', 'joint_pos_rel_10', 'joint_pos_rel_11',
                'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
                'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
                'left_foot_xpos_x', 'left_foot_xpos_y', 'left_foot_xpos_z',
                'left_foot_xmat_0', 'left_foot_xmat_1', 'left_foot_xmat_2',
                'left_foot_xmat_3', 'left_foot_xmat_4', 'left_foot_xmat_5',
                'left_foot_xmat_6', 'left_foot_xmat_7', 'left_foot_xmat_8',
                'left_foot_vel_x', 'left_foot_vel_y', 'left_foot_vel_z',
                'right_foot_xpos_x', 'right_foot_xpos_y', 'right_foot_xpos_z',
                'right_foot_xmat_0', 'right_foot_xmat_1', 'right_foot_xmat_2',
                'right_foot_xmat_3', 'right_foot_xmat_4', 'right_foot_xmat_5',
                'right_foot_xmat_6', 'right_foot_xmat_7', 'right_foot_xmat_8',
                'right_foot_vel_x', 'right_foot_vel_y', 'right_foot_vel_z',
                'swing_stance_delta_x', 'swing_stance_delta_y', 'swing_stance_delta_z',
                'com_x', 'com_y', 'com_z',
            ]
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
            self.csv_file_handle.flush()
            print(f"Initialized data logging to: {self.csv_file}")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")

    def _log_step_data(self, keyboard_key=None):
        """Log all step data every timestep."""
        if not self.csv_writer:
            return

        try:
            # Get foot positions and orientations from MuJoCo
            p_lf = self.data.site("left_foot").xpos.copy()
            R_lf = self.data.site("left_foot").xmat.reshape(3, 3)
            p_rf = self.data.site("right_foot").xpos.copy()
            R_rf = self.data.site("right_foot").xmat.reshape(3, 3)
            # print("left_foot: ", p_lf)
            # print("right_foot: ", p_rf)
            # Calculate foot velocities using finite differences
            v_lf = (p_lf - self.last_left_foot_pos) / self.dt if self.timestep > 0 else np.zeros(3)
            v_rf = (p_rf - self.last_right_foot_pos) / self.dt if self.timestep > 0 else np.zeros(3)

            # Update last positions for next iteration
            self.last_left_foot_pos = p_lf.copy()
            self.last_right_foot_pos = p_rf.copy()

            # Compute center of mass from MuJoCo
            # CoM is the weighted average of all body positions
            total_mass = 0
            com_pos = np.zeros(3)
            for i in range(self.model.nbody):
                mass = self.model.body_mass[i]
                if mass > 0:
                    body_pos = self.data.body(i).xpos
                    com_pos += mass * body_pos
                    total_mass += mass
            if total_mass > 0:
                com_pos /= total_mass
            self.last_com = com_pos.copy()

            # Prepare data row
            row_data = {
                'timestep': self.timestep,
                'keyboard_key': keyboard_key if keyboard_key else '',
                'command_foot_indicator': self.command_foot_indicator[0][0],
                'command_countdown_0': self.command_countdown[0][0],
                'command_stsw_pose_0': self.command_stsw_pose[0][0],
                'command_stsw_pose_1': self.command_stsw_pose[0][1],
                'command_stsw_pose_2': self.command_stsw_pose[0][2],
                'command_stsw_pose_3': self.command_stsw_pose[0][3],
            }

            # Add action outputs (12 DOF lower body)
            for i in range(12):
                row_data[f'action_{i}'] = self.last_policy_action[0][i] if self.last_policy_action is not None else 0.0

            # Add scaled action outputs (after scaling and residual adjustments) - only lower 12 DOFs
            for i in range(12):
                row_data[f'scaled_action_{i}'] = self.last_scaled_action[i]

            # Add actual joint position relative to default position - only lower 12 DOFs
            for i in range(12):
                row_data[f'joint_pos_rel_{i}'] = self.last_actual_joint_pos[i]

            # Add IMU data (gyro and accelerometer)
            row_data['imu_gyro_x'] = self.last_imu_gyro[0]
            row_data['imu_gyro_y'] = self.last_imu_gyro[1]
            row_data['imu_gyro_z'] = self.last_imu_gyro[2]
            row_data['imu_acc_x'] = self.last_imu_acc[0]
            row_data['imu_acc_y'] = self.last_imu_acc[1]
            row_data['imu_acc_z'] = self.last_imu_acc[2]

            # Add left foot position and orientation
            row_data['left_foot_xpos_x'] = p_lf[0]
            row_data['left_foot_xpos_y'] = p_lf[1]
            row_data['left_foot_xpos_z'] = p_lf[2]
            for i, val in enumerate(R_lf.flatten()):
                row_data[f'left_foot_xmat_{i}'] = val

            # Add left foot velocity (calculated via finite differences)
            row_data['left_foot_vel_x'] = v_lf[0]
            row_data['left_foot_vel_y'] = v_lf[1]
            row_data['left_foot_vel_z'] = v_lf[2]

            # Add right foot position and orientation
            row_data['right_foot_xpos_x'] = p_rf[0]
            row_data['right_foot_xpos_y'] = p_rf[1]
            row_data['right_foot_xpos_z'] = p_rf[2]
            for i, val in enumerate(R_rf.flatten()):
                row_data[f'right_foot_xmat_{i}'] = val

            # Add right foot velocity (calculated via finite differences)
            row_data['right_foot_vel_x'] = v_rf[0]
            row_data['right_foot_vel_y'] = v_rf[1]
            row_data['right_foot_vel_z'] = v_rf[2]

            # Compute swing-stance position difference w.r.t. stance foot frame
            if self.command_foot_indicator[0][0] == 0:
                # Left foot is stance, right foot is swing
                p_st = p_lf
                R_st = R_lf
                p_sw = p_rf
            else:
                # Right foot is stance, left foot is swing
                p_st = p_rf
                R_st = R_rf
                p_sw = p_lf

            # Compute swing position relative to stance foot in world frame
            delta_pos_world = p_sw - p_st
            # Transform to stance foot frame
            delta_pos_st = R_st.T @ delta_pos_world

            row_data['swing_stance_delta_x'] = delta_pos_st[0]
            row_data['swing_stance_delta_y'] = delta_pos_st[1]
            row_data['swing_stance_delta_z'] = delta_pos_st[2]

            # Add center of mass position
            row_data['com_x'] = self.last_com[0]
            row_data['com_y'] = self.last_com[1]
            row_data['com_z'] = self.last_com[2]
            # print("com :", com_pos)

            # Write to CSV
            self.csv_writer.writerow(row_data)
            self.csv_file_handle.flush()

        except Exception as e:
            print(f"Error logging step data: {e}")

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

    def rl_inference(self, robot_state_data):
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        self.data.qpos[:] = obs_buffer_dict["dof_pos"]
        mj.mj_forward(self.model, self.data)

        if self.startup:
            self.command_stsw_pose = self.left_stance_default.copy()
            self.command_countdown[0][0] = 0.0
            self.startup = False

        obs_buffer_dict["command_stsw_pose"] = self.command_stsw_pose
        obs_buffer_dict["command_countdown"] = self.command_countdown
        obs_buffer_dict["command_foot_indicator"] = self.command_foot_indicator
        # print(self.command_countdown,self.command_stsw_pose,self.command_foot_indicator)

        obs_buffer_dict["base_ang_vel_scaled"] = obs_buffer_dict["base_ang_vel"] * 0.2
        obs_buffer_dict["dof_pos_lower"] = obs_buffer_dict["dof_pos"][:, -12:]
        obs_buffer_dict["dof_vel_scaled"] = obs_buffer_dict["dof_vel"][:, -12:] * 0.05
        # obs_buffer_dict["dof_vel_scaled"][:, [4, 5, 10, 11]] = 0.0
        obs_buffer_dict["last_policy_action"] = self.last_policy_action.copy()

        # Compute foot poses in base frame via forward kinematics (following reach_gmr_FK_plot.py example)
        # Get foot poses in world frame from FK results
        p_lf_w = self.data.site("left_foot").xpos.copy()
        R_lf_w = self.data.site("left_foot").xmat.reshape(3, 3).copy()
        p_rf_w = self.data.site("right_foot").xpos.copy()
        R_rf_w = self.data.site("right_foot").xmat.reshape(3, 3).copy()

        # Get base pose in world frame
        base_site = self.data.site("imu")
        base_xpos = base_site.xpos.copy()
        base_xmat = base_site.xmat.reshape(3, 3).copy()

        # Transform foot positions to base frame
        p_lf_b = base_xmat.T @ (p_lf_w - base_xpos)
        p_rf_b = base_xmat.T @ (p_rf_w - base_xpos)

        # Transform foot orientations to base frame
        rot_base = R_scipy.from_matrix(base_xmat)
        rot_lf_w = R_scipy.from_matrix(R_lf_w)
        rot_rf_w = R_scipy.from_matrix(R_rf_w)
        rot_lf_b = rot_base.inv() * rot_lf_w
        rot_rf_b = rot_base.inv() * rot_rf_w
        R_lf_b = rot_lf_b.as_matrix()
        R_rf_b = rot_rf_b.as_matrix()

        # Format as pos[3] + rot_6d[6] = 9D per foot (following body_pose_b format)
        # Take first two columns of rotation matrix and flatten to 6D
        left_rot_6d = R_lf_b[:, :2].flatten()  # First two columns, flattened to 6D
        right_rot_6d = R_rf_b[:, :2].flatten()  # First two columns, flattened to 6D
        left_foot_pose_b = np.concatenate([p_lf_b, left_rot_6d])  # 3D pos + 6D rot = 9D
        right_foot_pose_b = np.concatenate([p_rf_b, right_rot_6d])  # 3D pos + 6D rot = 9D
        obs_buffer_dict["feet_pose_b"] = np.concatenate([left_foot_pose_b, right_foot_pose_b]).reshape(1, 18)  # 9D + 9D = 18D total

        obs_list = [
            "command_stsw_pose",
            "command_countdown",
            "command_foot_indicator",
            "base_ang_vel_scaled",
            "projected_gravity",
            # "feet_pose_b",
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
        self.last_policy_action = policy_action.copy() * 0.3 + self.last_policy_action.copy() * 0.7
        # self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]

        return scaled_policy_action

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)

        # Check cooldown period using start_stepping_time
        if self.start_stepping_time is not None:
            time_since_last_command = (time.time_ns() - self.start_stepping_time) / 1e9
            if time_since_last_command < self.cooldown_time:
                print(f"Cooldown active: {self.cooldown_time - time_since_last_command:.2f}s remaining")
                return

        if not self.in_stepping:
            if keycode in ["w", "s"]:
                self.in_stepping = True
                self.start_stepping_time = time.time_ns()
                dir = 1.0 if keycode == "w" else -1.0
                self.command_foot_indicator[0][0] = 1.0 if self.command_foot_indicator[0][0] == 0 else 0.0
                self.command_stsw_pose[0] = self.left_stance_default.copy() if self.command_foot_indicator[0][0] == 0 else self.right_stance_default.copy()
                self.command_stsw_pose[0][0] += dir * self.step_length_x
                self.command_countdown[0][0] = 1.0
            elif keycode in ["a","d"]:
                self.in_stepping = True
                self.start_stepping_time = time.time_ns()
                dir = 1.0 if keycode == "a" else -1.0
                self.command_foot_indicator[0][0] = 1.0 if self.command_foot_indicator[0][0] == 0 else 0.0
                self.command_stsw_pose[0] = self.left_stance_default.copy() if self.command_foot_indicator[0][0] == 0 else self.right_stance_default.copy()
                self.command_stsw_pose[0][1] += dir * self.step_length_y
                if self.command_stsw_pose[0][1] > 0:
                    self.command_stsw_pose[0][1] = max(self.command_stsw_pose[0][1], 0.2)
                else:
                    self.command_stsw_pose[0][1] = min(self.command_stsw_pose[0][1],-0.2)
                self.command_countdown[0][0] = 1.0
            elif keycode in ["q","e"]:
                self.in_stepping = True
                self.start_stepping_time = time.time_ns()
                dir = 1.0 if keycode == "q" else -1.0
                self.command_foot_indicator[0][0] = 1.0 if self.command_foot_indicator[0][0] == 0 else 0.0
                self.command_stsw_pose[0] = self.left_stance_default.copy() if self.command_foot_indicator[0][0] == 0 else self.right_stance_default.copy()
                self.command_stsw_pose[0][3] += dir * self.step_length_yaw
                self.command_countdown[0][0] = 1.0
            elif keycode in ["r", "f"]:
                self.in_stepping = True
                self.start_stepping_time = time.time_ns()
                dir = 1.0 if keycode == "r" else -1.0
                self.command_foot_indicator[0][0] = 1.0 if self.command_foot_indicator[0][0] == 0 else 0.0
                self.command_stsw_pose[0] = self.left_stance_default.copy() if self.command_foot_indicator[0][0] == 0 else self.right_stance_default.copy()
                self.command_stsw_pose[0][0] += self.step_length_x
                self.command_stsw_pose[0][2] += dir * 0.15
                self.command_countdown[0][0] = 1.0
            print("command_stsw_pose: ", self.command_stsw_pose)

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        robot_state_data = self.state_processor.robot_state_data
        if self.state_processor.robot_state_data is None:
            return

        # Increment timestep counter
        self.timestep += 1

        # Extract IMU data from robot_state_data
        # robot_state_data format: [q, dq, tau_est, ddq]
        # q: [base_pos(3), base_quat(4), joint_pos(num_dof)] - 7 + num_dof elements
        # dq: [base_lin_vel(3), base_ang_vel(3), joint_vel(num_dof)] - 3 + 3 + num_dof elements
        # tau_est: [base_lin_force(3), base_ang_torque(3), joint_torque(num_dof)] - 3 + 3 + num_dof elements
        # ddq: [base_lin_acc(3), base_ang_acc(3), joint_acc(num_dof)] - 3 + 3 + num_dof elements
        q_len = 7 + self.num_dofs
        dq_len = 3 + 3 + self.num_dofs
        tau_len = 3 + 3 + self.num_dofs
        # Extract IMU gyro (angular velocity) from dq[3:6]
        self.last_imu_gyro = robot_state_data[0, q_len + 3:q_len + 6].copy()
        # Extract IMU accelerometer from ddq[0:3]
        ddq_start = q_len + dq_len + tau_len
        self.last_imu_acc = robot_state_data[0, ddq_start:ddq_start + 3].copy()

        if self.in_stepping:
            now = time.time_ns()
            t_passed = (now - self.start_stepping_time) / 1e9
            # start: 0.0
            # middle: t_passsed is half step time, reaches 1, the final value is 1
            # end: 0.0
            self.command_countdown[0][0] = np.abs((t_passed / (self.step_time) - 1.0))
            if t_passed > self.step_time:
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

        # Store scaled action and actual joint positions for logging (lower body DOFs only)
        self.last_scaled_action = scaled_policy_action[0].copy()
        # Get current joint positions from MuJoCo (qpos includes free joint + all DOFs)
        # Free joint is first 7 DOFs, actual joints start at index 7
        current_joint_pos = self.data.qpos
        # Compute relative position for lower body DOFs only
        self.last_actual_joint_pos = current_joint_pos[self.lower_dof_indices] - self.default_dof_angles[self.lower_dof_indices]

        # Log data every timestep
        self._log_step_data()

    def _handle_base_height_control(self, keycode):
        """Handle base height control."""
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def close_logger(self):
        """Close the data logger and CSV file."""
        if self.csv_file_handle:
            try:
                self.csv_file_handle.close()
                print(f"Data logging closed. File saved to: {self.csv_file}")
            except Exception as e:
                print(f"Error closing CSV file: {e}")


def signal_handler(sig, frame, policy=None):
    if policy:
        policy.close_logger()
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

    def signal_handler_wrapper(sig, frame):
        policy.close_logger()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler_wrapper)
    policy.run()