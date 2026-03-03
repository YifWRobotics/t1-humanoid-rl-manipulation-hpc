import mujoco as mj
import mujoco.viewer as mj_viewer
import mink as ik
import threading
from loop_rate_limiters import RateLimiter
import time
import numpy as np


def rpy_to_quat(rpy: np.array) -> np.array:
    """Convert roll-pitch-yaw angles to quaternion (w, x, y, z)."""
    roll, pitch, yaw = rpy

    # half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


class T1_29_ArmIK:
    def __init__(self, model_path):
        self.mj_model = mj.MjModel.from_xml_path(model_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        self.configuration = ik.Configuration(self.mj_model)
        self.tasks = [
            posture_task := ik.PostureTask(self.mj_model, cost=1e-2),
            lh_task := ik.FrameTask(
                frame_name="left_hand",
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=0.001,
            ),
            rh_task := ik.FrameTask(
                frame_name="right_hand",
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=0.001,
            ),
            head_task := ik.FrameTask(
                frame_name="head",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=5.0,
                lm_damping=0.001,
            ),
            trunk_task := ik.FrameTask(
                frame_name="trunk",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=5.0,
                lm_damping=0.001,
            )
        ]
        self.posture_task = posture_task
        self.lh_task = lh_task
        self.rh_task = rh_task
        self.trunk_task = trunk_task
        self.head_task = head_task

        self.limits = [ik.ConfigurationLimit(self.mj_model)]

        # thread state
        self.viewer = mj_viewer.launch_passive(self.mj_model, self.mj_data)
        self.rate = None

        self.is_running = False

    def start_ik(self):
        if self.is_running:
            return
        self.is_running = True
        self.ik_thread = threading.Thread(
            target=self._solve_ik_loop, name="ArmIKThread", daemon=True
        )
        # start the background IK + viewer thread
        self.ik_thread.start()

    def stop_ik(self):
        self.is_running = False

    def get_qpos_upper(self):
        with self.viewer.lock():
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
        with self.viewer.lock():
            self.mj_data.qpos[7:7 + 2 + 7 * 2] = qpos_upper

    def sync_mocap(self, name: str, pos, rpy, relative_site_name):
        with self.viewer.lock():
            print("Syncing...")
            mocap_id = self.mj_model.body(name).mocapid[0]
            site_id = self.mj_model.site(relative_site_name).id
            site_xpos = self.mj_data.site_xpos[site_id]
            site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

            # convert relative position to world frame
            mocap_xpos_W = site_xpos + site_xmat @ pos

            # convert relative orientation to world frame
            relative_quat = rpy_to_quat(rpy)
            site_quat = mj.mju_mat2Quat(site_xmat.flatten())
            mocap_quat_W = mj.mju_mulQuat(site_quat, relative_quat)

            self.mj_data.mocap_pos[mocap_id] = mocap_xpos_W
            self.mj_data.mocap_quat[mocap_id] = mocap_quat_W

    # -------------- ik loop --------------

    def _solve_ik_loop(self):
        mj.mj_forward(self.mj_model, self.mj_data)
        ik.move_mocap_to_frame(self.mj_model, self.mj_data,
                               "left_hand_target", "left_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data,
                               "right_hand_target", "right_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data,
                               "trunk_target", "trunk", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data,
                               "head_target", "head", "site")
        self.rate = RateLimiter(100)

        while self.viewer.is_running() and self.is_running:
            # keep config near home as a soft prior
            self.configuration.update_from_keyframe("home")

            # update mj_data (forward kinematics, sensors, cam/light) under lock
            with self.viewer.lock():
                # read mocap targets under lock, then release for IK solve
            lh_T = ik.SE3.from_mocap_name(
                self.mj_model, self.mj_data, "left_hand_target"
            )
            rh_T = ik.SE3.from_mocap_name(
                self.mj_model, self.mj_data, "right_hand_target"
            )
            trunk_T = ik.SE3.from_mocap_name(
                self.mj_model, self.mj_data, "trunk_target"
            )
            head_T = ik.SE3.from_mocap_name(
                self.mj_model, self.mj_data, "head_target"
            )
            self.posture_task.set_target_from_configuration(
                self.configuration
            )
            self.lh_task.set_target(lh_T)
            self.rh_task.set_target(rh_T)
            self.trunk_task.set_target(trunk_T)
            self.head_task.set_target(head_T)

            vel = ik.solve_ik(
                self.configuration,
                self.tasks,
                self.rate.dt,
                "daqp",
                1e-1,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, self.rate.dt)

            self.mj_data.qpos[:] = self.configuration.q
            mj.mj_forward(self.mj_model, self.mj_data)
            self.viewer.sync()

            self.rate.sleep()
        self.is_running = False


if __name__ == "__main__":
    ik_solver = T1_29_ArmIK(
        "/home/zimengchai/Documents/digit_dprl/digit_dprl/rl/envs/t1/xmls/scene_t1_ik.xml"
    )
    ik_solver.start_ik()

    # keep the main thread alive while the viewer/thread are running
    try:
        while True:
            if not ik_solver.is_running:
                break
            v = ik_solver.viewer
            if v is not None and not v.is_running():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        ik_solver.stop_ik()
        # give the worker a moment to exit cleanly
        time.sleep(0.2)
