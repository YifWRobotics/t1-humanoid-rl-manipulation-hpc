import mujoco as mj
import mujoco.viewer as mj_viewer
import mink as ik
import threading
from loop_rate_limiters import RateLimiter
import time
import numpy as np


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


def quat_to_mat(quat: np.array) -> np.array:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quat

    # normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return mat


def quat_multiply(q1: np.array, q2: np.array) -> np.array:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def clamp_quat_rpy(quat, quat_ref, rpy):
    """
    clamp quaternion so its rpy angles don't exceed limits from reference quaternion

    args:
        quat: quaternion to clamp (w, x, y, z)
        quat_ref: reference quaternion (w, x, y, z)
        r, p, y: max allowed deviation in roll, pitch, yaw (radians)

    returns:
        clamped quaternion (w, x, y, z)
    """
    r, p, y = rpy
    # convert quaternions to rotation matrices
    mat_current = quat_to_mat(quat)
    mat_ref = quat_to_mat(quat_ref)

    # compute relative rotation: R_rel = R_ref^T * R_current
    mat_rel = mat_ref.T @ mat_current

    # convert relative rotation to rpy
    rpy_rel = mat_to_rpy(mat_rel)

    # clamp the relative rpy angles
    rpy_clamped = np.array([
        np.clip(rpy_rel[0], -r, r),  # roll
        np.clip(rpy_rel[1], -p, p),  # pitch
        np.clip(rpy_rel[2], -y, y)   # yaw
    ])

    # convert back to rotation matrix
    mat_rel_clamped = rpy_to_mat(rpy_clamped)

    # compute clamped absolute rotation: R_clamped = R_ref * R_rel_clamped
    mat_clamped = mat_ref @ mat_rel_clamped

    # convert back to quaternion
    quat_clamped = mat_to_quat(mat_clamped)

    return quat_clamped


def mat_to_rpy(mat: np.array) -> np.array:
    """convert 3x3 rotation matrix to roll-pitch-yaw angles."""
    # extract rpy using standard convention (ZYX euler angles)
    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(mat[2, 1], mat[2, 2])  # roll
        y = np.arctan2(-mat[2, 0], sy)        # pitch
        z = np.arctan2(mat[1, 0], mat[0, 0])  # yaw
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])  # roll
        y = np.arctan2(-mat[2, 0], sy)         # pitch
        z = 0                                   # yaw

    return np.array([x, y, z])


def rpy_to_mat(rpy: np.array) -> np.array:
    """convert roll-pitch-yaw angles to 3x3 rotation matrix."""
    roll, pitch, yaw = rpy

    # individual rotation matrices
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # combined rotation: R = Rz * Ry * Rx
    return rz @ ry @ rx


class T1_29_ArmIK_Teleop:
    def __init__(self, model_path, headless=True):
        self.mj_model = mj.MjModel.from_xml_path(model_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Create a separate MjData instance for FK computation to avoid threading conflicts
        self.mj_data_fk = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data_fk, 0)  # Initialize to home keyframe

        self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)
        self.tasks = [
            posture_task := ik.PostureTask(self.mj_model, cost=0.05),
            lh_task := ik.FrameTask(
                frame_name="left_hand",
                frame_type="site",
                position_cost=6.0,
                orientation_cost=2.0,
                lm_damping=0.03,
            ),
            rh_task := ik.FrameTask(
                frame_name="right_hand",
                frame_type="site",
                position_cost=6.0,
                orientation_cost=2.0,
                lm_damping=0.03,
            ),
            damping_task := ik.DampingTask(
                self.mj_model,
                cost=np.array([0.1] * 2 + (([0.5] * 4 + [0.1] * 3) * 2) + [0.1] * 13)
            ),
        ]
        self.posture_task = posture_task
        self.lh_task = lh_task
        self.rh_task = rh_task
        self.damping_task = damping_task

        # fmt: off
        self.collision_pairs = [
            (["al1", "al2", "al4", "al5", "al6", "al7", "ar1", "ar2", "ar4", "ar5", "ar6", "ar7", "al3_collision", "al4_collision", "ar3_collision", "ar4_collision"], ["trunk"]),
            (["al1", "al2", "al4", "al5", "al6", "al7", "ar1", "ar2","ar4", "ar5", "ar6","ar7", "al3_collision", "al4_collision", "ar3_collision", "ar4_collision", ], ["lowerbody_box"]),
            (["al3_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["al4_collision"], ["al3_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["left_hand_collision"], ["al3_collision", "al4_collision", "ar3_collision", "ar4_collision", "right_hand_collision"]),
            (["ar3_collision"], ["al4_collision", "left_hand_collision", "al3_collision", "ar4_collision", "right_hand_collision"]),
            (["ar4_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "al3_collision", "right_hand_collision"]),
            (["right_hand_collision"], ["al4_collision", "left_hand_collision", "ar3_collision", "ar4_collision", "al3_collision"]),
        ]
        # fmt: on

        factor = 0.7
        self.limits = [
            ik.ConfigurationLimit(
                self.mj_model, min_distance_from_limits=0.1
            ),
            ik.CollisionAvoidanceLimit(
                self.mj_model,
                self.collision_pairs,
                minimum_distance_from_collisions=0.04,
                collision_detection_distance=0.10,
            ),
            ik.VelocityLimit(
                self.mj_model,
                {
                    "Left_Shoulder_Pitch": 5.0 * factor,
                    "Left_Shoulder_Roll": 5.0 * factor,
                    "Left_Elbow_Pitch": 5.0 * factor,
                    "Left_Elbow_Yaw": 5.0 * factor,
                    "Left_Wrist_Pitch": 2.5 * 1.0,
                    "Left_Wrist_Yaw": 2.5 * 1.0,
                    "Left_Hand_Roll": 2.5 * 1.0,

                    "Right_Shoulder_Pitch": 5.0 * factor,
                    "Right_Shoulder_Roll": 5.0 * factor,
                    "Right_Elbow_Pitch": 5.0 * factor,
                    "Right_Elbow_Yaw": 5.0 * factor,
                    "Right_Wrist_Pitch": 2.5 * 1.0,
                    "Right_Wrist_Yaw": 2.5 * 1.0,
                    "Right_Hand_Roll": 2.5 * 1.0,

                    "AAHead_yaw": 6.0,
                    "Head_pitch": 5.0
                }
            ),
        ]

        self.synced_mocap = {}
        self.limited_mocap = {}

        # thread state
        self.datalock = threading.RLock()
        if not headless:
            self.viewer = mj_viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            self.viewer = None
        # self.viewer.cam.fixedcamid = self.mj_model.camera("teleop").id
        # self.viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.rate = None
        self.is_running = False
        self.is_ready = False
        self.is_solving = True

    def start_ik(self):
        if self.is_running:
            return
        self.is_running = True
        self.is_ready = False
        self.ik_thread = threading.Thread(target=self._solve_ik_loop, name="ArmIKThread")
        self.ik_thread.start()

    def set_solving(self, status: bool):
        self.is_solving = status

    def stop_ik(self):
        print("Stopping IK solver...")
        self.shutdown_requested = True
        self.is_running = False

        # Wait for thread to finish
        if hasattr(self, 'ik_thread') and self.ik_thread.is_alive():
            self.ik_thread.join(timeout=2.0)  # Wait max 2 seconds
            if self.ik_thread.is_alive():
                print("Warning: IK thread did not stop cleanly")

        # Close viewer if it exists
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass

    def get_qpos_upper(self):
        with self.datalock:
            res = np.zeros(16)
            for i, jnt_name in enumerate([
                "AAHead_yaw",
                "Head_pitch",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",
                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",
            ]):
                res[i] = self.mj_data.joint(jnt_name).qpos

            return res

    def sync_qpos_upper(self, qpos_upper):
        for i, jnt_name in enumerate([
            "AAHead_yaw",
            "Head_pitch",
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
        ]):
            self.mj_data.joint(jnt_name).qpos = qpos_upper[i]

    def set_mocap_wxyzxyz(self, name: str, wxyz_xyz):
        mocap_id = self.mj_model.body(name).mocapid[0]
        self.mj_data.mocap_pos[mocap_id] = wxyz_xyz[4:]
        self.mj_data.mocap_quat[mocap_id] = wxyz_xyz[:4]

    def reframe_mocap(self, name: str, wxyz_xyz, relative_site_name: str = "world"):
        if not self.is_ready:
            return

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        mocap_id = self.mj_model.body(name).mocapid[0]

        # get current mocap pose in world frame
        mocap_xpos_W = self.mj_data.mocap_pos[mocap_id].copy()
        mocap_quat_W = self.mj_data.mocap_quat[mocap_id].copy()

        # transform current mocap pose to site frame
        site_quat = mat_to_quat(site_xmat)
        site_quat_inv = np.array(
            [site_quat[0], -site_quat[1], -site_quat[2], -site_quat[3]])

        # position offset: current mocap position relative to site minus desired position
        pos_rel_to_site = site_xmat.T @ (mocap_xpos_W - site_xpos)
        pos_offset = pos_rel_to_site - pos

        desired_quat = quat
        current_quat_rel_to_site = quat_multiply(site_quat_inv, mocap_quat_W)
        # offset such that: desired_quat * quat_offset = current_quat_rel_to_site
        # so: quat_offset = desired_quat_inv * current_quat_rel_to_site
        desired_quat_inv = np.array(
            [desired_quat[0], -desired_quat[1], -desired_quat[2], -desired_quat[3]])
        quat_offset = quat_multiply(desired_quat_inv, current_quat_rel_to_site)

        self.synced_mocap[name] = {
            "pos_offset": pos_offset,
            "quat_offset": quat_offset
        }

    def get_datalock(self):
        return self.datalock

    def sync_mocap(self, name: str, wxyz_xyz, relative_site_name: str = "world"):
        if not self.is_ready:
            return

        if name not in self.synced_mocap.keys():
            self.reframe_mocap(name, relative_site_name, wxyz_xyz)

        mocap_id = self.mj_model.body(name).mocapid[0]
        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        pos_offset = self.synced_mocap[name]["pos_offset"]
        quat_offset = self.synced_mocap[name]["quat_offset"]

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        # apply offset
        pos_corrected = pos + pos_offset
        input_quat = quat
        relative_quat_corrected = quat_multiply(input_quat, quat_offset)

        # convert to world frame
        mocap_xpos_W = site_xpos + site_xmat @ pos_corrected
        site_quat = mat_to_quat(site_xmat)
        mocap_quat_W = quat_multiply(site_quat, relative_quat_corrected)

        self.mj_data.mocap_pos[mocap_id] = mocap_xpos_W
        self.mj_data.mocap_quat[mocap_id] = \
            clamp_quat_rpy(mocap_quat_W, self.limited_mocap[name]["quat_init_W"], self.limited_mocap[name]["limit"]) \
            if name in self.limited_mocap.keys() \
            else mocap_quat_W

    def move_mocap_to(self, name: str, target_site_name: str):
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, name, target_site_name, "site")

    def get_mocap_pose_w(self, name: str):
        mocap_id = self.mj_model.body(name).mocapid[0]
        return np.concatenate([self.mj_data.mocap_pos[mocap_id], self.mj_data.mocap_quat[mocap_id]])

    def get_mocap_pose_b(self, name: str):
        mocap_id = self.mj_model.body(name).mocapid[0]
        site = self.mj_data.site("imu")
        g_wb = ik.SE3(np.concatenate([mat_to_quat(site.xmat.reshape(3, 3)), site.xpos]))
        g_wm = ik.SE3(np.concatenate([self.mj_data.mocap_quat[mocap_id], self.mj_data.mocap_pos[mocap_id]]))
        g_bm = g_wb.inverse().multiply(g_wm)
        return np.concatenate([g_bm.wxyz_xyz[4:], g_bm.wxyz_xyz[:4]])

    def compute_hand_fk_from_all_joints(self, all_joint_pos):
        """Compute forward kinematics for hand sites given all joint positions.

        Args:
            all_joint_pos: All joint positions (29 values: upper body + lower body)
                Order: head(2) + left_arm(7) + right_arm(7) + waist(1) + left_leg(6) + right_leg(6)

        Returns:
            tuple: (left_hand_pose_b, right_hand_pose_b), each as [x,y,z,w,x,y,z]
        """
        # Use separate FK data to avoid threading conflicts with IK solver
        # Set ALL joint positions in FK data
        joint_names = [
            "AAHead_yaw",
            "Head_pitch",
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
            "Waist",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ]

        for i, jnt_name in enumerate(joint_names):
            self.mj_data_fk.joint(jnt_name).qpos = all_joint_pos[i]

        # Compute forward kinematics
        mj.mj_forward(self.mj_model, self.mj_data_fk)

        # Get hand site poses in world frame
        left_hand_site_id = self.mj_model.site("left_hand").id
        right_hand_site_id = self.mj_model.site("right_hand").id

        left_hand_xpos = self.mj_data_fk.site_xpos[left_hand_site_id].copy()
        left_hand_xmat = self.mj_data_fk.site_xmat[left_hand_site_id].reshape(3, 3).copy()
        right_hand_xpos = self.mj_data_fk.site_xpos[right_hand_site_id].copy()
        right_hand_xmat = self.mj_data_fk.site_xmat[right_hand_site_id].reshape(3, 3).copy()

        # Get base (imu) frame
        base_site = self.mj_data_fk.site("imu")
        base_xpos = base_site.xpos.copy()
        base_xmat = base_site.xmat.reshape(3, 3).copy()

        # Transform hand poses to base frame using SE3
        g_wb = ik.SE3(np.concatenate([mat_to_quat(base_xmat), base_xpos]))

        # Left hand
        g_wl = ik.SE3(np.concatenate([mat_to_quat(left_hand_xmat), left_hand_xpos]))
        g_bl = g_wb.inverse().multiply(g_wl)
        left_hand_pose_b = np.concatenate([g_bl.wxyz_xyz[4:], g_bl.wxyz_xyz[:4]])  # [x,y,z,w,x,y,z]

        # Right hand
        g_wr = ik.SE3(np.concatenate([mat_to_quat(right_hand_xmat), right_hand_xpos]))
        g_br = g_wb.inverse().multiply(g_wr)
        right_hand_pose_b = np.concatenate([g_br.wxyz_xyz[4:], g_br.wxyz_xyz[:4]])  # [x,y,z,w,x,y,z]

        return left_hand_pose_b, right_hand_pose_b

    def get_ik_solution_hand_poses(self):
        """Get the IK solution's end effector poses in base frame.

        This computes FK on the IK solver's internal state (mj_data) to get
        the hand poses that the IK solution actually achieves.

        Returns:
            tuple: (left_hand_pose_b, right_hand_pose_b), each as [x,y,z,w,x,y,z]
        """
        with self.datalock:
            # Get hand site poses from IK solver's mj_data (in world frame)
            left_hand_site_id = self.mj_model.site("left_hand").id
            right_hand_site_id = self.mj_model.site("right_hand").id

            left_hand_xpos = self.mj_data.site_xpos[left_hand_site_id].copy()
            left_hand_xmat = self.mj_data.site_xmat[left_hand_site_id].reshape(3, 3).copy()
            right_hand_xpos = self.mj_data.site_xpos[right_hand_site_id].copy()
            right_hand_xmat = self.mj_data.site_xmat[right_hand_site_id].reshape(3, 3).copy()

            # Get base (imu) frame
            base_site = self.mj_data.site("imu")
            base_xpos = base_site.xpos.copy()
            base_xmat = base_site.xmat.reshape(3, 3).copy()

        # Transform hand poses to base frame using SE3
        g_wb = ik.SE3(np.concatenate([mat_to_quat(base_xmat), base_xpos]))

        # Left hand
        g_wl = ik.SE3(np.concatenate([mat_to_quat(left_hand_xmat), left_hand_xpos]))
        g_bl = g_wb.inverse().multiply(g_wl)
        left_hand_pose_b = np.concatenate([g_bl.wxyz_xyz[4:], g_bl.wxyz_xyz[:4]])  # [x,y,z,w,x,y,z]

        # Right hand
        g_wr = ik.SE3(np.concatenate([mat_to_quat(right_hand_xmat), right_hand_xpos]))
        g_br = g_wb.inverse().multiply(g_wr)
        right_hand_pose_b = np.concatenate([g_br.wxyz_xyz[4:], g_br.wxyz_xyz[:4]])  # [x,y,z,w,x,y,z]

        return left_hand_pose_b, right_hand_pose_b

    def config_mocap_max_rpy_from_init(self, name: str, max_r_deg: float, max_p_deg: float, max_y_deg: float):
        self.limited_mocap[name] = {}
        self.limited_mocap[name]["limit"] = np.deg2rad(np.array([max_r_deg, max_p_deg, max_y_deg]))

    # -------------- ik loop --------------
    def _solve_ik_loop(self):
        mj.mj_forward(self.mj_model, self.mj_data)
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "trunk_target", "trunk", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "head_target", "head", "site")

        for name in self.limited_mocap.keys():
            mocap_id = self.mj_model.body(name).mocapid[0]
            self.limited_mocap[name]["quat_init_W"] = self.mj_data.mocap_quat[mocap_id].copy()

        self.rate = RateLimiter(100)

        while self.is_running:
            # update mj_data (forward kinematics, sensors, cam/light) under lock
            with self.datalock:
                # read mocap targets under lock, then release for IK solve
                lh_T = ik.SE3.from_mocap_name(
                    self.mj_model, self.mj_data, "left_hand_target"
                )
                rh_T = ik.SE3.from_mocap_name(
                    self.mj_model, self.mj_data, "right_hand_target"
                )
                self.posture_task.set_target_from_configuration(
                    self.configuration
                )
                self.lh_task.set_target(lh_T)
                self.rh_task.set_target(rh_T)

                if self.is_solving:
                    vel = ik.solve_ik(
                        self.configuration,
                        self.tasks,
                        self.rate.dt,
                        "daqp",
                        1e-2,
                        safety_break=True,
                        limits=self.limits,
                    )
                    self.configuration.integrate_inplace(vel, self.rate.dt)
                    self.mj_data.qpos[:] = self.configuration.q
                else:
                    self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])

                mj.mj_forward(self.mj_model, self.mj_data)

            if self.viewer is not None:
                with self.viewer.lock():
                    self.viewer.sync()

            self.is_ready = True
            self.rate.sleep()
        self.is_running = False
        self.is_ready = False


if __name__ == "__main__":
    ik_solver = T1_29_ArmIK_Teleop(
        "/home/zimengchai/Documents/digit_dprl/digit_dprl/rl/envs/t1/xmls/scene_t1_ik.xml"
    )
    ik_solver.start_ik()

    # keep the main thread alive while the viewer/thread are running
    try:
        while True:
            if not ik_solver.is_running:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        ik_solver.stop_ik()
        # give the worker a moment to exit cleanly
        time.sleep(0.2)
