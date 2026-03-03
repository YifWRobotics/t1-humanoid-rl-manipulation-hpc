import torch
import numpy as np


def get_joint_ref_indices():
    """
    Create a mapping from robot joint indices to dataset joint indices.

    Dataset q order (36 total: 7 base + 29 joints):
    0-6: base [x, y, z, qw, qx, qy, qz]
    7: AAHead_yaw
    8: Head_pitch
    9: Left_Shoulder_Pitch
    10: Left_Shoulder_Roll
    11: Left_Elbow_Pitch
    12: Left_Elbow_Yaw
    13: Left_Wrist_Pitch
    14: Left_Wrist_Yaw
    15: Left_Hand_Roll
    16: Right_Shoulder_Pitch
    17: Right_Shoulder_Roll
    18: Right_Elbow_Pitch
    19: Right_Elbow_Yaw
    20: Right_Wrist_Pitch
    21: Right_Wrist_Yaw
    22: Right_Hand_Roll
    23: Waist
    24: Left_Hip_Pitch
    25: Left_Hip_Roll
    26: Left_Hip_Yaw
    27: Left_Knee_Pitch
    28: Left_Ankle_Pitch
    29: Left_Ankle_Roll
    30: Right_Hip_Pitch
    31: Right_Hip_Roll
    32: Right_Hip_Yaw
    33: Right_Knee_Pitch
    34: Right_Ankle_Pitch
    35: Right_Ankle_Roll

    Dataset qd order (35 total: 6 base + 29 joints):
    0-5: base velocities [vx, vy, vz, wx, wy, wz]
    6-34: same joint order as above (indices shifted by -1)

    Robot internal joint order (actual order from robot):
    0: AAHead_yaw
    1: Left_Shoulder_Pitch
    2: Right_Shoulder_Pitch
    3: Waist
    4: Head_pitch
    5: Left_Shoulder_Roll
    6: Right_Shoulder_Roll
    7: Left_Hip_Pitch
    8: Right_Hip_Pitch
    9: Left_Elbow_Pitch
    10: Right_Elbow_Pitch
    11: Left_Hip_Roll
    12: Right_Hip_Roll
    13: Left_Elbow_Yaw
    14: Right_Elbow_Yaw
    15: Left_Hip_Yaw
    16: Right_Hip_Yaw
    17: Left_Wrist_Pitch
    18: Right_Wrist_Pitch
    19: Left_Knee_Pitch
    20: Right_Knee_Pitch
    21: Left_Wrist_Yaw
    22: Right_Wrist_Yaw
    23: Left_Ankle_Pitch
    24: Right_Ankle_Pitch
    25: Left_Hand_Roll
    26: Right_Hand_Roll
    27: Left_Ankle_Roll
    28: Right_Ankle_Roll

    Returns:
        tuple: (q_mapping, qd_mapping)
            - q_mapping: (29,) torch tensor - maps robot joints to dataset q columns (7-35)
            - qd_mapping: (29,) torch tensor - maps robot joints to dataset qd columns (6-34)
    """

    # Mapping for q (positions) - maps robot joint index to dataset q column
    robot_to_dataset_q = np.array([
        7,   # 0: AAHead_yaw -> dataset q[7]
        9,   # 1: Left_Shoulder_Pitch -> dataset q[9]
        16,  # 2: Right_Shoulder_Pitch -> dataset q[16]
        23,  # 3: Waist -> dataset q[23]
        8,   # 4: Head_pitch -> dataset q[8]
        10,  # 5: Left_Shoulder_Roll -> dataset q[10]
        17,  # 6: Right_Shoulder_Roll -> dataset q[17]
        24,  # 7: Left_Hip_Pitch -> dataset q[24]
        30,  # 8: Right_Hip_Pitch -> dataset q[30]
        11,  # 9: Left_Elbow_Pitch -> dataset q[11]
        18,  # 10: Right_Elbow_Pitch -> dataset q[18]
        25,  # 11: Left_Hip_Roll -> dataset q[25]
        31,  # 12: Right_Hip_Roll -> dataset q[31]
        12,  # 13: Left_Elbow_Yaw -> dataset q[12]
        19,  # 14: Right_Elbow_Yaw -> dataset q[19]
        26,  # 15: Left_Hip_Yaw -> dataset q[26]
        32,  # 16: Right_Hip_Yaw -> dataset q[32]
        13,  # 17: Left_Wrist_Pitch -> dataset q[13]
        20,  # 18: Right_Wrist_Pitch -> dataset q[20]
        27,  # 19: Left_Knee_Pitch -> dataset q[27]
        33,  # 20: Right_Knee_Pitch -> dataset q[33]
        14,  # 21: Left_Wrist_Yaw -> dataset q[14]
        21,  # 22: Right_Wrist_Yaw -> dataset q[21]
        28,  # 23: Left_Ankle_Pitch -> dataset q[28]
        34,  # 24: Right_Ankle_Pitch -> dataset q[34]
        15,  # 25: Left_Hand_Roll -> dataset q[15]
        22,  # 26: Right_Hand_Roll -> dataset q[22]
        29,  # 27: Left_Ankle_Roll -> dataset q[29]
        35,  # 28: Right_Ankle_Roll -> dataset q[35]
    ], dtype=np.int32)

    # Mapping for qd (velocities) - same joint order but shifted by -1
    # (dataset qd has 6 base values instead of 7)
    robot_to_dataset_qd = robot_to_dataset_q - 1

    # Convert to torch tensors
    robot_to_dataset_q_tensor = torch.from_numpy(robot_to_dataset_q).long()
    robot_to_dataset_qd_tensor = torch.from_numpy(robot_to_dataset_qd).long()

    return robot_to_dataset_q_tensor, robot_to_dataset_qd_tensor

