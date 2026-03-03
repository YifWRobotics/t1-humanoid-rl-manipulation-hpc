"""
Sample valid hand pose commands for T1 robot using IK (IsaacLab-compatible).

Output format matches IsaacLab FKCommand conventions (positions and quaternions in base/Trunk frame, xyzw order).
"""

import numpy as np
import mujoco
import mink as ik
import os
from scipy.spatial.transform import Rotation as R

MODEL_XML = "source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml"
N_SAMPLES = 200000
OUTPUT_NPZ = "source/isaaclab_assets/data/hand_pose_commands.npz"
OUTPUT_DATASET_NPZ = "source/isaaclab_assets/data/feasible_poses_dataset.npz"

LEFT_HAND_BODY = "left_hand_ee"
RIGHT_HAND_BODY = "right_hand_ee"
BASE_BODY = "Trunk"
LEFT_HAND_JOINTS = [
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
]
RIGHT_HAND_JOINTS = [
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
]

# Workspace bounds (relative to base)
WORKSPACE_X_RANGE = (-0.2, 1.5)  # Extended forward reach in +X direction
WORKSPACE_Y_RANGE = (-0.15, 0.4)
WORKSPACE_Z_RANGE = (-0.2, 1.0)  # Target: reach at least +0.5 in Z

DT = 0.02


def get_joint_indices(model, joint_names):
    return [model.joint(name).qposadr for name in joint_names]


def pose_within_cube(pos, base_pos, x_range, y_range, z_range):
    rel = pos - base_pos
    x, y, z = rel
    return (x_range[0] <= x <= x_range[1]) and \
           (y_range[0] <= y <= y_range[1]) and \
           (z_range[0] <= z <= z_range[1])


def sample_target_pose(T_init, delta_pos, delta_rot_deg):
    pos = T_init.wxyz_xyz[4:] + np.random.uniform(delta_pos[0], delta_pos[1], size=3)
    delta_r = np.deg2rad(np.random.uniform(delta_rot_deg[0], delta_rot_deg[1]))
    delta_p = np.deg2rad(np.random.uniform(delta_rot_deg[0], delta_rot_deg[1]))
    delta_y = np.deg2rad(np.random.uniform(delta_rot_deg[0], delta_rot_deg[1]))
    cr, sr = np.cos(delta_r / 2), np.sin(delta_r / 2)
    cp, sp = np.cos(delta_p / 2), np.sin(delta_p / 2)
    cy, sy = np.cos(delta_y / 2), np.sin(delta_y / 2)
    delta_quat = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])
    # quat = init * delta (wxyz)
    w1, x1, y1, z1 = T_init.wxyz_xyz[:4]
    w2, x2, y2, z2 = delta_quat
    quat = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])
    quat = quat / np.linalg.norm(quat)
    return np.concatenate([quat, pos])


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    ik_model = model
    configuration = ik.Configuration(ik_model, ik_model.keyframe("arms_extended_front").qpos)
    arm_joint_ids = get_joint_indices(model, LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS)

    KEYFRAMES = ["arms_extended_front"]

    # Fresh Configuration per keyframe for correct FK transforms
    keyframe_transforms = {}
    print("\nVerifying keyframe symmetry:")
    for kf_name in KEYFRAMES:
        kf_config = ik.Configuration(ik_model, ik_model.keyframe(kf_name).qpos)
        T_lh_init = kf_config.get_transform_frame_to_world("left_hand", "site")
        T_rh_init = kf_config.get_transform_frame_to_world("right_hand", "site")

        lh_quat_wxyz = T_lh_init.wxyz_xyz[:4]
        rh_quat_wxyz = T_rh_init.wxyz_xyz[:4]
        lh_quat_xyzw = np.array([lh_quat_wxyz[1], lh_quat_wxyz[2], lh_quat_wxyz[3], lh_quat_wxyz[0]])
        rh_quat_xyzw = np.array([rh_quat_wxyz[1], rh_quat_wxyz[2], rh_quat_wxyz[3], rh_quat_wxyz[0]])
        lh_rot = R.from_quat(lh_quat_xyzw)
        rh_rot = R.from_quat(rh_quat_xyzw)
        lh_euler = lh_rot.as_euler('ZYX', degrees=True)
        rh_euler = rh_rot.as_euler('ZYX', degrees=True)
        # Y-pitch asymmetry: left_y should be -right_y for symmetry
        initial_y_rot_asymmetry = (lh_euler[1] + rh_euler[1]) / 2

        keyframe_transforms[kf_name] = {
            "left": T_lh_init,
            "right": T_rh_init,
            "qpos": ik_model.keyframe(kf_name).qpos.copy(),
            "initial_y_rot_asymmetry": initial_y_rot_asymmetry
        }
        left_pos = keyframe_transforms[kf_name]['left'].wxyz_xyz[4:]
        right_pos = keyframe_transforms[kf_name]['right'].wxyz_xyz[4:]

        sym_error_y = abs(left_pos[1] + right_pos[1])
        sym_error_x = abs(left_pos[0] - right_pos[0])
        sym_error_z = abs(left_pos[2] - right_pos[2])

        print(f"  {kf_name}: left=({left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}), "
              f"right=({right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f})")
        print(f"    Symmetry errors: Y_mirror={sym_error_y:.4f}, X_match={sym_error_x:.4f}, Z_match={sym_error_z:.4f}")
        print(f"    Initial Y-rotation asymmetry: {initial_y_rot_asymmetry:.4f}° (will be compensated)")

        if sym_error_y > 0.01 or sym_error_x > 0.01 or sym_error_z > 0.01:
            print(f"    WARNING: Keyframe '{kf_name}' is not symmetric!")

    tasks = [
        ik.PostureTask(ik_model, cost=0.5),
        ik.FrameTask("left_hand", "site", position_cost=10.0, orientation_cost=20.0, lm_damping=0.1),
        ik.FrameTask("right_hand", "site", position_cost=10.0, orientation_cost=20.0, lm_damping=0.1),
    ]

    # Joint limits: MUST match soft_joint_pos_limit_factor in training (booster.py:98)
    joint_limits = ik_model.jnt_range.copy()
    factor = 0.9  # Match training environment soft_joint_pos_limit_factor
    margin = ((1 - factor) * (joint_limits[:, 1] - joint_limits[:, 0]) / 2).min()
    limits = [
        ik.ConfigurationLimit(ik_model, min_distance_from_limits=margin),
    ]

    valid_qref = []
    valid_body_poses = []

    print(f"Sampling {N_SAMPLES} IK-feasible hand pose commands from keyframes: {KEYFRAMES}")

    attempts = 0
    while len(valid_qref) < N_SAMPLES:
        kf_name = np.random.choice(KEYFRAMES)
        kf_data = keyframe_transforms[kf_name]
        T_lh_init = kf_data["left"]
        T_rh_init = kf_data["right"]

        delta_x_base = np.random.uniform(-0.4, 0.1)
        delta_z_base = np.random.uniform(-0.5, 0.5)

        # Symmetric sampling: shared X/Z, mirrored Y
        noise_scale_pos = 0.05  # 5cm
        shared_delta_x = delta_x_base + np.random.uniform(-noise_scale_pos, noise_scale_pos)
        shared_delta_z = delta_z_base + np.random.uniform(-noise_scale_pos, noise_scale_pos)

        # Y: negative = toward centerline, positive = outward
        delta_y_abs = np.random.uniform(-0.25, 0.3)
        delta_y_left = delta_y_abs
        delta_y_right = -delta_y_abs  # mirrored

        # Lock orientation
        delta_rot_deg_lh = np.zeros(3)
        delta_rot_deg_rh = np.zeros(3)

        delta_pos_lh = np.array([shared_delta_x, delta_y_left, shared_delta_z])
        delta_pos_rh = np.array([shared_delta_x, delta_y_right, shared_delta_z])

        lh_pos = T_lh_init.wxyz_xyz[4:] + delta_pos_lh
        delta_r, delta_p, delta_y = np.deg2rad(delta_rot_deg_lh)
        cr, sr = np.cos(delta_r / 2), np.sin(delta_r / 2)
        cp, sp = np.cos(delta_p / 2), np.sin(delta_p / 2)
        cy, sy = np.cos(delta_y / 2), np.sin(delta_y / 2)
        delta_quat = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
        w1, x1, y1, z1 = T_lh_init.wxyz_xyz[:4]
        w2, x2, y2, z2 = delta_quat
        lh_quat = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
        lh_quat = lh_quat / np.linalg.norm(lh_quat)
        lh_target = np.concatenate([lh_quat, lh_pos])

        rh_pos = T_rh_init.wxyz_xyz[4:] + delta_pos_rh
        delta_r_rh, delta_p_rh, delta_y_rh = np.deg2rad(delta_rot_deg_rh)
        cr_rh, sr_rh = np.cos(delta_r_rh / 2), np.sin(delta_r_rh / 2)
        cp_rh, sp_rh = np.cos(delta_p_rh / 2), np.sin(delta_p_rh / 2)
        cy_rh, sy_rh = np.cos(delta_y_rh / 2), np.sin(delta_y_rh / 2)
        delta_quat_rh = np.array([
            cr_rh * cp_rh * cy_rh + sr_rh * sp_rh * sy_rh,
            sr_rh * cp_rh * cy_rh - cr_rh * sp_rh * sy_rh,
            cr_rh * sp_rh * cy_rh + sr_rh * cp_rh * sy_rh,
            cr_rh * cp_rh * sy_rh - sr_rh * sp_rh * cy_rh
        ])

        w1, x1, y1, z1 = T_rh_init.wxyz_xyz[:4]
        w2, x2, y2, z2 = delta_quat_rh
        rh_quat = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
        rh_quat = rh_quat / np.linalg.norm(rh_quat)
        rh_target = np.concatenate([rh_quat, rh_pos])

        # Set IK targets
        tasks[0].set_target(kf_data["qpos"])
        tasks[1].set_target(ik.SE3(lh_target))
        tasks[2].set_target(ik.SE3(rh_target))
        configuration.q[:] = kf_data["qpos"]
        try:
            vel = ik.solve_ik(configuration, tasks, DT, "daqp", 1e-4, limits=limits)
            configuration.integrate_inplace(vel, DT)
        except Exception:
            attempts += 1
            continue

        data.qpos[:] = configuration.q
        mujoco.mj_forward(model, data)

        lh_body_id = model.body(LEFT_HAND_BODY).id
        rh_body_id = model.body(RIGHT_HAND_BODY).id
        base_id = model.body(BASE_BODY).id

        lh_pos = data.xpos[lh_body_id].copy()
        rh_pos = data.xpos[rh_body_id].copy()
        lh_mat = data.xmat[lh_body_id].reshape(3, 3)
        rh_mat = data.xmat[rh_body_id].reshape(3, 3)
        base_mat = data.xmat[base_id].reshape(3, 3)
        base_pos = data.xpos[base_id].copy()

        # Transform to Trunk frame (IsaacLab convention)
        lh_pos_body = base_mat.T @ (lh_pos - base_pos)
        rh_pos_body = base_mat.T @ (rh_pos - base_pos)
        lh_quat_body = R.from_matrix(base_mat.T @ lh_mat).as_quat()  # xyzw
        rh_quat_body = R.from_matrix(base_mat.T @ rh_mat).as_quat()  # xyzw

        correction_lh = R.from_euler('yx', [180, -90], degrees=True)
        lh_quat_body = (correction_lh * R.from_quat(lh_quat_body)).as_quat()

        correction_rh = R.from_euler('yx', [180, 90], degrees=True)
        rh_quat_body = (correction_rh * R.from_quat(rh_quat_body)).as_quat()

        if lh_quat_body[3] < 0:
            lh_quat_body = -lh_quat_body
        if rh_quat_body[3] < 0:
            rh_quat_body = -rh_quat_body

        lh_pose = np.concatenate([lh_pos_body, lh_quat_body])
        rh_pose = np.concatenate([rh_pos_body, rh_quat_body])
        valid_body_poses.append((lh_pose, rh_pose))
        valid_qref.append(configuration.q[arm_joint_ids].copy())

        if len(valid_qref) % 100 == 0:
            print(f"Collected {len(valid_qref)}/{N_SAMPLES} samples...")

        attempts += 1

    print(f"Success rate: {100*len(valid_qref)/attempts:.1f}%")

    # Post-generation symmetry verification
    print("\n=== Symmetry Verification ===")
    all_left_y = [pose[0][1] for pose in valid_body_poses]  # Y positions of left hands
    all_right_y = [pose[1][1] for pose in valid_body_poses]  # Y positions of right hands

    # Check if distributions are mirrored (left_y should be -right_y)
    symmetry_errors = [abs(ly + ry) for ly, ry in zip(all_left_y, all_right_y)]
    mean_sym_error = np.mean(symmetry_errors)
    max_sym_error = np.max(symmetry_errors)

    print(f"Left Y range: [{np.min(all_left_y):.3f}, {np.max(all_left_y):.3f}]")
    print(f"Right Y range: [{np.min(all_right_y):.3f}, {np.max(all_right_y):.3f}]")
    print(f"Y-axis symmetry error: mean={mean_sym_error*100:.2f}cm, max={max_sym_error*100:.2f}cm")

    # Check workspace coverage
    all_left_x = [pose[0][0] for pose in valid_body_poses]
    all_left_z = [pose[0][2] for pose in valid_body_poses]
    print("\nWorkspace coverage (left hand):")
    print(f"  X: [{np.min(all_left_x):.3f}, {np.max(all_left_x):.3f}] (target: 0.05-0.5)")
    print(f"  Y: [{np.min(all_left_y):.3f}, {np.max(all_left_y):.3f}] (target: -0.1-0.4)")
    print(f"  Z: [{np.min(all_left_z):.3f}, {np.max(all_left_z):.3f}] (target: -0.2-0.5)")

    # Check rotation symmetry (Y-rotation should be mirrored)
    all_left_quats = [pose[0][3:] for pose in valid_body_poses]
    all_right_quats = [pose[1][3:] for pose in valid_body_poses]
    y_rot_diffs = []
    for lq, rq in zip(all_left_quats, all_right_quats):
        l_euler = R.from_quat(lq).as_euler('ZYX', degrees=True)
        r_euler = R.from_quat(rq).as_euler('ZYX', degrees=True)
        # For symmetry, Y-rotation should be mirrored: left_y ≈ -right_y
        y_rot_diff = abs(l_euler[1] + r_euler[1])
        y_rot_diffs.append(y_rot_diff)

    mean_y_rot_diff = np.mean(y_rot_diffs)
    max_y_rot_diff = np.max(y_rot_diffs)
    print(f"\nY-rotation symmetry: mean_diff={mean_y_rot_diff:.2f}°, max_diff={max_y_rot_diff:.2f}°")

    if mean_sym_error > 0.01:  # 1cm
        print(f"\nWARNING: High position symmetry error ({mean_sym_error*100:.2f}cm mean). Dataset may be asymmetric!")
    if mean_y_rot_diff > 5.0:  # 5 degrees
        print(f"WARNING: High rotation symmetry error ({mean_y_rot_diff:.2f}° mean). Dataset may be asymmetric!")

    left_poses_body = np.array([pose[0] for pose in valid_body_poses])
    right_poses_body = np.array([pose[1] for pose in valid_body_poses])
    poses_body = np.stack([left_poses_body, right_poses_body], axis=-1)  # (N, 7, 2)
    qref = np.array(valid_qref)
    xyzwxyz_BLH = left_poses_body
    xyzwxyz_BRH = right_poses_body
    num_samples = len(valid_qref)
    starting_steps = np.arange(num_samples)
    ending_steps = np.arange(num_samples)
    dt = DT

    output_dir = os.path.dirname(OUTPUT_NPZ)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    np.savez(OUTPUT_NPZ, poses=poses_body)
    print(f"\n[1/2] Saved {num_samples} hand pose command samples to {OUTPUT_NPZ}")
    print(f"      Output shape: {poses_body.shape} (N, 7, 2) in Trunk frame")

    np.savez(
        OUTPUT_DATASET_NPZ,
        qref=qref,
        xyzwxyz_BLH=xyzwxyz_BLH,
        xyzwxyz_BRH=xyzwxyz_BRH,
        starting_steps=starting_steps,
        ending_steps=ending_steps,
        dt=dt
    )
    print(f"[2/2] Saved dataviewer dataset to {OUTPUT_DATASET_NPZ}")
    print(f"      qref: {qref.shape}, BLH: {xyzwxyz_BLH.shape}, BRH: {xyzwxyz_BRH.shape}")


if __name__ == "__main__":
    main()
