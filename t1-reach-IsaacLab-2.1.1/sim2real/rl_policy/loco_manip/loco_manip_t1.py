import argparse
import csv
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xrobotoolkit_sdk as xrt
import yaml
import mujoco as mj
from scipy.spatial.transform import Rotation as R_scipy
from termcolor import colored
import onnxruntime

from booster_robotics_sdk_python import RobotMode

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy
from sim2real.utils.arm_ik.mink_t1_29dof_teleop import T1_29_ArmIK_Teleop




def quat_inverse(q):
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
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
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])


def headset_to_world(wxyzxyz, linear_scale_factor=0.6):
    """Transform headset pose from VR space to world frame."""
    headset_pos = wxyzxyz[4:]
    headset_quat = wxyzxyz[:4]
    world_pos = np.array([-headset_pos[2], -headset_pos[0], headset_pos[1]]) * linear_scale_factor
    transform_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    # Swap y/z components and negate x/z to fix roll/yaw inversion
    adjusted_quat = np.array([headset_quat[0], -headset_quat[1], headset_quat[3], -headset_quat[2]])
    world_quat = quat_multiply(transform_quat, adjusted_quat)
    return np.concatenate([world_quat, world_pos])


def yaw_rotate(wxyzxyz, yaw_rad):
    """Rotate pose around z-axis by yaw_rad."""
    quat = wxyzxyz[:4]
    pos = wxyzxyz[4:]
    r, p, y = quat_to_rpy(quat)
    rotated_quat = rpy_to_quat(np.array([r, p, y + yaw_rad]))
    rotated_pos = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0.],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0.],
        [0., 0., 1.]
    ]) @ pos
    return np.concatenate([rotated_quat, rotated_pos])


# Workspace limits in base frame (matching training configuration)
WORKSPACE_LIMITS = {
    "left": {
        "x": (0.0, 0.5),
        "y": (0.0, 0.3),
        "z": (-0.1, 0.3),
    },
    "right": {
        "x": (0.0, 0.5),
        "y": (-0.3, 0.0),
        "z": (-0.1, 0.3),
    }
}


def clamp_hand_pose_to_workspace(pos_xyzwxyz: np.array, arm: str) -> np.array:
    """Clamp hand position to workspace limits in base frame [x, y, z, qw, qx, qy, qz]."""
    limits = WORKSPACE_LIMITS[arm]
    clamped = pos_xyzwxyz.copy()
    clamped[0] = np.clip(pos_xyzwxyz[0], limits["x"][0], limits["x"][1])
    clamped[1] = np.clip(pos_xyzwxyz[1], limits["y"][0], limits["y"][1])
    clamped[2] = np.clip(pos_xyzwxyz[2], limits["z"][0], limits["z"][1])
    return clamped


def clamp_pose_world_frame(upper_body_controller, pose_wxyzxyz: np.array, arm: str) -> np.array:
    """Clamp world frame pose [w,x,y,z,x,y,z] to workspace limits."""
    quat_w = pose_wxyzxyz[:4]
    pos_w = pose_wxyzxyz[4:]

    base_site = upper_body_controller.mj_data.site("imu")
    base_xpos = base_site.xpos.copy()
    base_xmat = base_site.xmat.reshape(3, 3).copy()

    # Transform to base frame
    pos_b = base_xmat.T @ (pos_w - base_xpos)
    rot_base = R_scipy.from_matrix(base_xmat)
    rot_w = R_scipy.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])
    rot_b = rot_base.inv() * rot_w
    quat_b_xyzw = rot_b.as_quat()
    quat_b_wxyz = np.array([quat_b_xyzw[3], quat_b_xyzw[0], quat_b_xyzw[1], quat_b_xyzw[2]])

    pose_b = np.concatenate([pos_b, quat_b_wxyz])
    clamped_pose_b = clamp_hand_pose_to_workspace(pose_b, arm)

    # Transform back to world frame
    clamped_pos_b = clamped_pose_b[:3]
    clamped_quat_b = clamped_pose_b[3:]
    clamped_pos_w = base_xpos + base_xmat @ clamped_pos_b

    rot_b_clamped = R_scipy.from_quat([clamped_quat_b[1], clamped_quat_b[2], clamped_quat_b[3], clamped_quat_b[0]])
    rot_w_clamped = rot_base * rot_b_clamped
    quat_w_clamped_xyzw = rot_w_clamped.as_quat()
    quat_w_clamped_wxyz = np.array([quat_w_clamped_xyzw[3], quat_w_clamped_xyzw[0], quat_w_clamped_xyzw[1], quat_w_clamped_xyzw[2]])

    return np.concatenate([quat_w_clamped_wxyz, clamped_pos_w])


def clamp_mocap_to_workspace(upper_body_controller, mocap_name: str, arm: str):
    """Clamp mocap target in MuJoCo to workspace limits."""
    mocap_pose_b = upper_body_controller.get_mocap_pose_b(mocap_name)
    clamped_pose_b = clamp_hand_pose_to_workspace(mocap_pose_b, arm)

    if not np.allclose(mocap_pose_b[:3], clamped_pose_b[:3], atol=1e-4):
        base_site = upper_body_controller.mj_data.site("imu")
        base_xpos = base_site.xpos.copy()
        base_xmat = base_site.xmat.reshape(3, 3).copy()

        pos_b = clamped_pose_b[:3]
        quat_b = clamped_pose_b[3:]
        pos_w = base_xpos + base_xmat @ pos_b

        rot_base = R_scipy.from_matrix(base_xmat)
        rot_b = R_scipy.from_quat([quat_b[1], quat_b[2], quat_b[3], quat_b[0]])
        rot_w = rot_base * rot_b
        quat_w_xyzw = rot_w.as_quat()
        quat_w_wxyz = np.array([quat_w_xyzw[3], quat_w_xyzw[0], quat_w_xyzw[1], quat_w_xyzw[2]])

        mocap_id = upper_body_controller.mj_model.body(mocap_name).mocapid[0]
        upper_body_controller.mj_data.mocap_pos[mocap_id] = pos_w
        upper_body_controller.mj_data.mocap_quat[mocap_id] = quat_w_wxyz


class TeleopXRoboLocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, upper_body_model_path, lower_body_model_path=None,
        rl_rate=50, policy_action_scale=1.0, headless=True, enable_logging=False
    ):
        # Initialize with upper body model path (for base initialization)
        super().__init__(config, upper_body_model_path, rl_rate, policy_action_scale)

        # Upper body policy (already loaded by super().__init__)
        self.upper_body_policy = self.policy
        self.upper_body_policy_session = self.onnx_policy_session

        # Load lower body policy if provided
        if lower_body_model_path is not None:
            self.setup_lower_body_policy(lower_body_model_path)
        else:
            self.lower_body_policy = None
            self.lower_body_policy_session = None

        self.residual_upper_body_action = False

        # Upper body policy tracking
        self.last_upper_policy_action = np.zeros((1, 14))
        self.obs_history_upper = {}
        self.obs_history_len_upper = 5

        # Lower body policy tracking
        self.last_lower_policy_action = np.zeros((1, 12))
        self.obs_history_lower = {}
        self.obs_history_len_lower = 8

        self.starting_pose = self.default_dof_angles.copy()
        self._debug_iter = 0
        self._vr_left_was_out = False
        self._vr_right_was_out = False
        self._vr_left_out_count = 0
        self._vr_right_out_count = 0

        # Lower body control variables from foot_track_plot.py
        self.command_stsw_pose = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.command_countdown = np.array([[0.0]])
        self.command_foot_indicator = np.array([[0.0]])
        self.startup_lower = True
        self.in_stepping = False
        self.step_length = 0.1
        self.step_time = 1.0
        self.start_stepping_time = None
        self.dir = 1

        # MuJoCo model for lower body foot tracking
        if lower_body_model_path is not None:
            self.mj_model = mj.MjModel.from_xml_path(self.config.get("ik_xml_path"))
            self.mj_data = mj.MjData(self.mj_model)
            mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        else:
            self.mj_model = None
            self.mj_data = None

        self.enable_logging = enable_logging
        self.csv_file = None
        self.csv_writer = None
        if self.enable_logging:
            self._init_csv_file()

        if headless:
            self.logger.warning("Using headless mode.")
        self.init_upper_body_controller(headless=headless)
        print(self.command_sender.client.ChangeMode(RobotMode.kCustom))

    def setup_lower_body_policy(self, model_path):
        """Setup ONNX policy model for lower body."""
        self.lower_body_policy_session = onnxruntime.InferenceSession(model_path)
        input_names = [
            inp.name for inp in self.lower_body_policy_session.get_inputs()]
        output_names = [
            out.name for out in self.lower_body_policy_session.get_outputs()]

        self.lower_body_onnx_input_names = input_names
        self.lower_body_onnx_output_names = output_names

        def lower_body_policy_act(obs_dict):
            input_feed = {name: obs_dict[name]
                          for name in self.lower_body_onnx_input_names}
            outputs = self.lower_body_policy_session.run(
                self.lower_body_onnx_output_names, input_feed)
            return outputs[0]

        self.lower_body_policy = lower_body_policy_act

    def _init_csv_file(self):
        """Initialize CSV file for hand tracking data logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / "hand_tracking_logs"
        log_dir.mkdir(exist_ok=True)
        csv_path = log_dir / f"hand_track_{timestamp}.csv"

        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        header = [
            'timestamp', 'iteration',
            # Left hand target (position + quaternion)
            'left_target_x', 'left_target_y', 'left_target_z',
            'left_target_qw', 'left_target_qx', 'left_target_qy', 'left_target_qz',
            # Left hand current (position + quaternion)
            'left_current_x', 'left_current_y', 'left_current_z',
            'left_current_qw', 'left_current_qx', 'left_current_qy', 'left_current_qz',
            # Left hand position error norm (from observation)
            'left_pos_error_norm',
            # Right hand target
            'right_target_x', 'right_target_y', 'right_target_z',
            'right_target_qw', 'right_target_qx', 'right_target_qy', 'right_target_qz',
            # Right hand current
            'right_current_x', 'right_current_y', 'right_current_z',
            'right_current_qw', 'right_current_qx', 'right_current_qy', 'right_current_qz',
            # Right hand position error norm (from observation)
            'right_pos_error_norm',
            # Policy action (14 DOF)
            *[f'policy_action_{i}' for i in range(14)],
            # Policy enabled flag
            'policy_enabled', 'init_mode'
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()

        print(f"📊 Hand tracking logging enabled. Data will be saved to: {csv_path}")

    def _log_hand_tracking_data(
        self, left_hand_target, left_hand_current,
        right_hand_target, right_hand_current,
        left_pos_error, right_pos_error, policy_action
    ):
        """Log hand tracking data to CSV file."""
        if not self.enable_logging or self.csv_writer is None:
            return

        row = [
            time.time(), self._debug_iter,
            *left_hand_target[:3], *left_hand_target[3:],
            *left_hand_current[:3], *left_hand_current[3:],
            np.linalg.norm(left_pos_error),
            *right_hand_target[:3], *right_hand_target[3:],
            *right_hand_current[:3], *right_hand_current[3:],
            np.linalg.norm(right_pos_error),
            *policy_action.flatten(),
            int(self.use_policy_action), int(self.get_ready_state)
        ]
        self.csv_writer.writerow(row)

        if self._debug_iter % 50 == 0:
            self.csv_file.flush()

    def close_logger(self):
        """Close CSV file if logging is enabled."""
        if self.csv_file is not None:
            self.csv_file.close()
            print("📊 Hand tracking log file closed.")

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_upper_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command
        return current_obs_dict

    def init_upper_body_controller(self, headless):
        """Initialize VR controller and upper body IK solver."""
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

        self.upper_body_controller = T1_29_ArmIK_Teleop(self.config.get("ik_xml_path"), headless)
        self.upper_body_controller.start_ik()
        threading.Thread(target=self.loop_xrobo).start()

    def rl_inference_upper(self, robot_state_data):
        """RL inference for upper body policy."""
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        obs_buffer_dict["last_policy_action"] = self.last_upper_policy_action.copy()

        # Get hand target poses in base frame [x,y,z,w,x,y,z]
        left_hand_pose_target_xyzwxyz_unclamped = self.upper_body_controller.get_mocap_pose_b("left_hand_target").reshape(7)
        right_hand_pose_target_xyzwxyz_unclamped = self.upper_body_controller.get_mocap_pose_b("right_hand_target").reshape(7)

        # Clamp to workspace limits
        left_hand_pose_target_xyzwxyz = clamp_hand_pose_to_workspace(left_hand_pose_target_xyzwxyz_unclamped, "left")
        right_hand_pose_target_xyzwxyz = clamp_hand_pose_to_workspace(right_hand_pose_target_xyzwxyz_unclamped, "right")

        # Clamp mocap targets in sim to prevent IK solver from targeting unreachable poses
        if not np.allclose(left_hand_pose_target_xyzwxyz_unclamped[:3], left_hand_pose_target_xyzwxyz[:3], atol=1e-4):
            clamp_mocap_to_workspace(self.upper_body_controller, "left_hand_target", "left")
        if not np.allclose(right_hand_pose_target_xyzwxyz_unclamped[:3], right_hand_pose_target_xyzwxyz[:3], atol=1e-4):
            clamp_mocap_to_workspace(self.upper_body_controller, "right_hand_target", "right")

        # Compute current hand poses via FK (dof_pos are deltas from default)
        all_joint_pos = (obs_buffer_dict["dof_pos"] + self.default_dof_angles).flatten()
        left_hand_pose_current_xyzwxyz, right_hand_pose_current_xyzwxyz = self.upper_body_controller.compute_hand_fk_from_all_joints(all_joint_pos)

        # Build observations: convert poses to (pos[3] + rotmat[9]) format
        left_quat_target = left_hand_pose_target_xyzwxyz[3:]
        left_pos_target = left_hand_pose_target_xyzwxyz[:3]
        right_quat_target = right_hand_pose_target_xyzwxyz[3:]
        right_pos_target = right_hand_pose_target_xyzwxyz[:3]

        left_rotmat_target = quat_to_mat(left_quat_target)
        left_rot_9d_target = left_rotmat_target.flatten()
        right_rotmat_target = quat_to_mat(right_quat_target)
        right_rot_9d_target = right_rotmat_target.flatten()

        obs_buffer_dict["left_hand_pose_command"] = np.concatenate([left_pos_target, left_rot_9d_target]).reshape(1, 12)
        obs_buffer_dict["right_hand_pose_command"] = np.concatenate([right_pos_target, right_rot_9d_target]).reshape(1, 12)

        left_pos_current = left_hand_pose_current_xyzwxyz[:3]
        right_pos_current = right_hand_pose_current_xyzwxyz[:3]
        left_rotmat_current = quat_to_mat(left_hand_pose_current_xyzwxyz[3:])
        right_rotmat_current = quat_to_mat(right_hand_pose_current_xyzwxyz[3:])
        left_rot_9d_current = left_rotmat_current.flatten()
        right_rot_9d_current = right_rotmat_current.flatten()

        left_hand_pose_current = np.concatenate([left_pos_current, left_rot_9d_current])
        right_hand_pose_current = np.concatenate([right_pos_current, right_rot_9d_current])
        obs_buffer_dict["hands_pose_b"] = np.concatenate([left_hand_pose_current, right_hand_pose_current]).reshape(1, 24)

        # Compute errors: pos delta + rotation error (R_current^T @ R_target)
        left_pos_error = left_pos_target - left_pos_current
        left_rot_error = left_rotmat_current.T @ left_rotmat_target
        obs_buffer_dict["left_hand_error"] = np.concatenate([left_pos_error, left_rot_error.flatten()]).reshape(1, 12)

        right_pos_error = right_pos_target - right_pos_current
        right_rot_error = right_rotmat_current.T @ right_rotmat_target
        obs_buffer_dict["right_hand_error"] = np.concatenate([right_pos_error, right_rot_error.flatten()]).reshape(1, 12)

        if self.use_policy_action:
            self._log_hand_tracking_data(
                left_hand_target=left_hand_pose_target_xyzwxyz,
                left_hand_current=left_hand_pose_current_xyzwxyz,
                right_hand_target=right_hand_pose_target_xyzwxyz,
                right_hand_current=right_hand_pose_current_xyzwxyz,
                left_pos_error=left_pos_error,
                right_pos_error=right_pos_error,
                policy_action=self.last_upper_policy_action
            )

        obs_buffer_dict["dof_pos_upper"] = obs_buffer_dict["dof_pos"][:, self.upper_dof_indices[2:]]
        obs_buffer_dict["dof_vel_upper"] = obs_buffer_dict["dof_vel"][:, self.upper_dof_indices[2:]]
        obs_buffer_dict["dof_vel_upper_scaled"] = obs_buffer_dict["dof_vel_upper"] * 0.1

        # Build observation history buffer for upper body
        obs_list_upper = [
            'dof_pos_upper', 'dof_vel_upper_scaled',
            'left_hand_pose_command', 'right_hand_pose_command',
            'hands_pose_b', 'left_hand_error', 'right_hand_error',
            'last_policy_action'
        ]
        for key in obs_list_upper:
            if key not in self.obs_history_upper.keys():
                self.obs_history_upper[key] = np.zeros((self.obs_history_len_upper, obs_buffer_dict[key].shape[1]))
            self.obs_history_upper[key] = np.roll(self.obs_history_upper[key], -1, axis=0)
            self.obs_history_upper[key][-1] = obs_buffer_dict[key]

        collected_obs_upper = np.concatenate([self.obs_history_upper[key].flatten() for key in obs_list_upper], axis=0)
        policy_action_upper = self.upper_body_policy({"obs": collected_obs_upper.reshape(1, -1).astype(np.float32)})

        # Smooth policy action with EMA filter
        self.last_upper_policy_action = policy_action_upper.copy() * 0.1 + self.last_upper_policy_action.copy() * 0.9
        scaled_policy_action_upper = policy_action_upper * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action_upper[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action_upper

    def rl_inference_lower(self, robot_state_data):
        """RL inference for lower body policy."""
        if self.lower_body_policy is None:
            return None

        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)

        # Update MuJoCo model with current joint positions
        if self.mj_data is not None and self.mj_model is not None:
            self.mj_data.qpos[:] = obs_buffer_dict["dof_pos"]
            mj.mj_forward(self.mj_model, self.mj_data)

        if self.startup_lower:
            self.command_stsw_pose = self.get_stsw_pose()
            self.command_countdown[0][0] = 0.0
            self.startup_lower = False

        obs_buffer_dict["command_stsw_pose"] = self.command_stsw_pose
        obs_buffer_dict["command_countdown"] = self.command_countdown
        obs_buffer_dict["command_foot_indicator"] = self.command_foot_indicator

        obs_buffer_dict["base_ang_vel_scaled"] = obs_buffer_dict["base_ang_vel"] * 0.2
        obs_buffer_dict["dof_pos_lower"] = obs_buffer_dict["dof_pos"][:, -12:]
        obs_buffer_dict["dof_vel_scaled"] = obs_buffer_dict["dof_vel"][:, -12:] * 0.025
        obs_buffer_dict["dof_vel_scaled"][:, [4, 5, 10, 11]] = 0.0
        obs_buffer_dict["last_policy_action"] = self.last_lower_policy_action.copy()

        obs_list_lower = [
            "command_stsw_pose",
            "command_countdown",
            "command_foot_indicator",
            "base_ang_vel_scaled",
            "projected_gravity",
            "dof_pos_lower",
            "dof_vel_scaled",
            "last_policy_action",
        ]

        # Initialize and roll observation history for lower body
        for key in obs_list_lower:
            if key not in self.obs_history_lower.keys():
                self.obs_history_lower[key] = np.zeros((self.obs_history_len_lower, obs_buffer_dict[key].shape[1]))
            self.obs_history_lower[key] = np.roll(self.obs_history_lower[key], -1, axis=0)
            self.obs_history_lower[key][-1] = obs_buffer_dict[key]

        # Collect observations in specified layout
        collected_obs_lower = []
        for key in obs_list_lower:
            collected_obs_lower.append(self.obs_history_lower[key].flatten())

        # Concatenate all observation types
        collected_obs_lower = np.concatenate(collected_obs_lower, axis=0)

        # Run lower body policy
        policy_action_lower = self.lower_body_policy({"obs": collected_obs_lower.reshape(1, -1).astype(np.float32)})

        # Smooth policy action with EMA filter
        self.last_lower_policy_action = policy_action_lower.copy() * 0.1 + self.last_lower_policy_action.copy() * 0.9
        scaled_policy_action_lower = policy_action_lower * self.policy_action_scale

        return scaled_policy_action_lower

    def get_stsw_pose(self):
        """Get swing-stance pose for lower body control."""
        if self.mj_data is None:
            return np.array([[0.0, 0.0, 0.0, 0.0]])

        p_lf = self.mj_data.site("left_foot").xpos
        p_rf = self.mj_data.site("right_foot").xpos
        R_lf = self.mj_data.site("left_foot").xmat.reshape(3, 3)
        R_rf = self.mj_data.site("right_foot").xmat.reshape(3, 3)

        if self.command_foot_indicator[0][0] == 0:
            p_st = p_lf
            p_sw = p_rf
            R_st = R_lf
            y = -0.2
        else:
            p_st = p_rf
            p_sw = p_lf
            R_st = R_rf
            y = 0.2

        # Compute swing position relative to stance foot (not used but kept for consistency)
        delta_pos_world = p_sw - p_st
        _ = R_st.T @ delta_pos_world  # Transform to stance foot frame

        return np.array([[0, y, 0.0, 0]])

    def get_init_target(self, robot_state_data):
        """Get init target by interpolating from current to default pose."""
        dof_pos = robot_state_data[:, 7: 7 + self.num_dofs]
        if self.get_ready_state:
            q_target = dof_pos + (self.starting_pose - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def policy_action(self):
        robot_state_data = self.state_processor.robot_state_data

        # Update upper body controller reference
        if self.upper_body_controller:
            upper_body_qpos = self.upper_body_controller.get_qpos_upper()
            self.ref_upper_dof_pos = upper_body_qpos.reshape(1, -1)
        if robot_state_data is None:
            return

        # Get upper body policy action
        scaled_policy_action_upper = self.rl_inference_upper(robot_state_data)

        # Get lower body policy action
        scaled_policy_action_lower = None
        if self.lower_body_policy is not None:
            # Update stepping countdown
            if self.in_stepping:
                now = time.time_ns()
                t_passed = (now - self.start_stepping_time) / 1e9
                self.command_countdown[0][0] = np.abs((t_passed / (self.step_time) - 1.0))
                if t_passed > self.step_time:
                    self.command_countdown[0][0] = 0.0
                    self.in_stepping = False

            scaled_policy_action_lower = self.rl_inference_lower(robot_state_data)

        if self.get_ready_state:
            # Init mode: interpolate to starting pose
            print("  [BRANCH] Init state mode - interpolating to starting_pose")
            self.upper_body_controller.set_solving(False)
            q_target = self.get_init_target(robot_state_data)
            with self.upper_body_controller.get_datalock():
                self.upper_body_controller.sync_qpos_upper(self.state_processor.q[7: 7 + self.num_upper_dofs])
                self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")
                self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # No policy: stay at current position
            print("  [BRANCH] No policy - staying at current position")
            self.upper_body_controller.set_solving(False)
            q_target = robot_state_data[:, 7: 7 + self.num_dofs]
            with self.upper_body_controller.get_datalock():
                self.upper_body_controller.sync_qpos_upper(self.state_processor.q[7: 7 + self.num_upper_dofs])
                self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")
                self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")
        else:
            # Policy enabled: apply RL actions
            print("  [BRANCH] Policy enabled - using RL actions")
            self.upper_body_controller.set_solving(True)
            true_act_upper = scaled_policy_action_upper + self.default_dof_angles[self.upper_dof_indices[2:]]
            q_target = robot_state_data[:, 7: 7 + self.num_dofs].copy()
            q_target[:, self.upper_dof_indices[2:]] = true_act_upper
            q_target[0][0] = 0
            q_target[0][1] = 0

            # Apply lower body policy action if available
            if scaled_policy_action_lower is not None:
                true_act_lower = scaled_policy_action_lower + self.default_dof_angles[self.lower_dof_indices]
                q_target[:, self.lower_dof_indices] = true_act_lower

            if self._debug_iter % 100 == 0:
                left_err = self.obs_history_upper['left_hand_error'][-1]
                right_err = self.obs_history_upper['right_hand_error'][-1]
                left_rot_mat = left_err[3:].reshape(3, 3)
                right_rot_mat = right_err[3:].reshape(3, 3)
                left_rot_angle = np.arccos(np.clip((np.trace(left_rot_mat) - 1) / 2, -1, 1))
                right_rot_angle = np.arccos(np.clip((np.trace(right_rot_mat) - 1) / 2, -1, 1))

        # Clip and send command
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        cmd_q = q_target[0]
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7: 7 + self.num_dofs])
        self._debug_iter += 1

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
        elif self.lower_body_policy is not None:
            # Lower body stepping controls from foot_track_plot.py
            if not self.in_stepping:
                if keycode in ["w"]:
                    self.dir = 1
                elif keycode in ["s"]:
                    self.dir = -1
                if keycode in ["a", "d"]:
                    self.in_stepping = True
                    self.start_stepping_time = time.time_ns()
                    print(f"{'Left' if keycode == 'a' else 'Right'}Foot Step.")
                    self.command_foot_indicator[0][0] = 1.0 if keycode == "a" else 0.0
                    if self.mj_data is not None:
                        self.command_stsw_pose[0] = self.get_stsw_pose()
                        self.command_stsw_pose[0][0] += self.dir * self.step_length
                    self.command_countdown[0][0] = 1.0
                    self._print_control_status()

    def _handle_base_height_control(self, keycode):
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _print_control_status(self):
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")
        if self.lower_body_policy is not None:
            print(f"Current Stance: {self.command_foot_indicator}")

    def loop_xrobo(self):
        """VR controller loop - reads controller state and updates robot commands."""
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
                continue

            with self.upper_body_controller.get_datalock():
                self.headset_pose = headset_to_world(np.concatenate([headset_pose[3:], headset_pose[:3]]), linear_scale_factor=1.0)
                if self.offset_yaw_rad_starting is None:
                    self.offset_yaw_rad_starting = quat_to_rpy(self.headset_pose[:4])[2]
                self.lctrl_pose = headset_to_world(np.concatenate([lctrl_pose[3:], lctrl_pose[:3]]), linear_scale_factor=0.9)
                self.rctrl_pose = headset_to_world(np.concatenate([rctrl_pose[3:], rctrl_pose[:3]]), linear_scale_factor=0.9)
                offset = self.offset_yaw_rad - self.offset_yaw_rad_starting
                self.lctrl_pose = yaw_rotate(self.lctrl_pose, -offset)
                self.rctrl_pose = yaw_rotate(self.rctrl_pose, -offset)

                self.lctrl_joystick[:] = lctrl_axis
                lctrl_pose_clamped = self.lctrl_pose
                rctrl_pose_clamped = self.rctrl_pose

                if l_grip > 0.99:
                    if not self.lctrl_hold:
                        self.upper_body_controller.reframe_mocap("left_hand_target", lctrl_pose_clamped)
                        self.lctrl_hold = True
                    self.upper_body_controller.sync_mocap("left_hand_target", lctrl_pose_clamped)
                else:
                    self.lctrl_hold = False
                    self.upper_body_controller.move_mocap_to("left_hand_target", "left_hand")

                if r_grip > 0.99:
                    if not self.rctrl_hold:
                        self.upper_body_controller.reframe_mocap("right_hand_target", rctrl_pose_clamped)
                        self.rctrl_hold = True
                    self.upper_body_controller.sync_mocap("right_hand_target", rctrl_pose_clamped)
                else:
                    self.rctrl_hold = False
                    self.upper_body_controller.move_mocap_to("right_hand_target", "right_hand")

            # X button: toggle standing/walking
            if x_button:
                if not self.lctrl_x_prev:
                    self.lctrl_x_prev = True
                    self.stand_command = np.array([[1 if self.stand_command[0][0] == 0 else 0]])
                    self.logger.info(colored(f"Switched to: {'Standing' if self.stand_command[0][0] == 0 else 'Walking'}", "green"))
            else:
                self.lctrl_x_prev = False

            # A button: reset yaw reference
            if a_button:
                if not self.rctrl_a_prev:
                    self.rctrl_a_prev = True
                    self.offset_yaw_rad = quat_to_rpy(self.headset_pose[:4])[2]
                    self.logger.info(colored("VR reference yaw angle resetted.", "green"))
            else:
                self.rctrl_a_prev = False

            # Update movement commands from joystick
            self.lin_vel_command[0][0] = -lctrl_axis[0] * 0.8
            self.lin_vel_command[0][1] = lctrl_axis[1] * 1.0
            self.ang_vel_command[0][0] = -rctrl_axis[0] * 1.5
            self.base_height_command[0][0] = np.clip(
                self.base_height_command[0][0] + 1e-4 * rctrl_axis[1], 0.5, 0.7
            )
            self.lctrl_trigger = l_trigger * 800.0 + 100.0
            self.rctrl_trigger = r_trigger * 800.0 + 100.0


def signal_handler(sig, frame):
    print("\n🛑 Shutting down...")
    if hasattr(policy, 'close_logger'):
        policy.close_logger()
    policy.upper_body_controller.stop_ik()
    xrt.close()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--upper_body_model_path", type=str, help="path to the upper body ONNX model")
    parser.add_argument("--lower_body_model_path", type=str, default=None, help="path to the lower body ONNX model (optional)")
    parser.add_argument('--viewer', action='store_false', dest='headless', help='Enable viewer')
    parser.add_argument('--log', action='store_true', help='Enable hand tracking data logging to CSV')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Get model paths from args or config
    upper_body_model_path = args.upper_body_model_path if args.upper_body_model_path else config.get("upper_body_model_path")
    lower_body_model_path = args.lower_body_model_path if args.lower_body_model_path else config.get("lower_body_model_path")

    # Fallback to old model_path for backward compatibility
    if upper_body_model_path is None:
        upper_body_model_path = config.get("model_path")

    if not upper_body_model_path:
        raise ValueError("upper_body_model_path must be provided either via --upper_body_model_path argument or in config file")

    policy = TeleopXRoboLocoManipPolicy(
        config=config,
        upper_body_model_path=upper_body_model_path,
        lower_body_model_path=lower_body_model_path,
        rl_rate=50,
        policy_action_scale=1.0,
        headless=args.headless,
        enable_logging=args.log
    )

    signal.signal(signal.SIGINT, signal_handler)
    policy.run()
