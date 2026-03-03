
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

MODEL_XML = "source/isaaclab_assets/isaaclab_assets/robots/xmls/scene_t1_ik.xml" # Update with your path if needed
LEFT_HAND_BODY = "left_hand_ee"
RIGHT_HAND_BODY = "right_hand_ee"
BASE_BODY = "Trunk"

model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)

data.qpos[:] = model.keyframe("prep3").qpos
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

print("\n=== prep3 Keyframe Hand Poses (Base/Trunk Frame) ===\n")
print("Left Hand:")
print(f"  Position: [{lh_pos_body[0]:.6f}, {lh_pos_body[1]:.6f}, {lh_pos_body[2]:.6f}]")
print(f"  Quaternion (xyzw): [{lh_quat_body[0]:.6f}, {lh_quat_body[1]:.6f}, {lh_quat_body[2]:.6f}, {lh_quat_body[3]:.6f}]")
print()
print("Right Hand:")
print(f"  Position: [{rh_pos_body[0]:.6f}, {rh_pos_body[1]:.6f}, {rh_pos_body[2]:.6f}]")
print(f"  Quaternion (xyzw): [{rh_quat_body[0]:.6f}, {rh_quat_body[1]:.6f}, {rh_quat_body[2]:.6f}, {rh_quat_body[3]:.6f}]")
print()

print("\n=== For commands.py FKCommand.__init__ ===")
print(f"""
self.default_left_hand_pos = torch.tensor(
    [{lh_pos_body[0]:.6f}, {lh_pos_body[1]:+.6f}, {lh_pos_body[2]:.6f}], dtype=torch.float32, device=self.device
)
self.default_left_hand_quat = torch.tensor(
    [{lh_quat_body[0]:+.6f}, {lh_quat_body[1]:+.6f}, {lh_quat_body[2]:+.6f}, {lh_quat_body[3]:+.6f}], dtype=torch.float32, device=self.device
)

self.default_right_hand_pos = torch.tensor(
    [{rh_pos_body[0]:.6f}, {rh_pos_body[1]:+.6f}, {rh_pos_body[2]:.6f}], dtype=torch.float32, device=self.device
)
self.default_right_hand_quat = torch.tensor(
    [{rh_quat_body[0]:+.6f}, {rh_quat_body[1]:+.6f}, {rh_quat_body[2]:+.6f}, {rh_quat_body[3]:+.6f}], dtype=torch.float32, device=self.device
)
""")
