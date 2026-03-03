import xrobotoolkit_sdk as xrt

xrt.init()

while True:
    left_pose = xrt.get_left_controller_pose()
    right_pose = xrt.get_right_controller_pose()
    headset_pose = xrt.get_headset_pose()

    # print(f"Left Controller Pose: {left_pose}")
    # print(f" {xrt.get_left_axis()}")
    print(f"Headset Pose: {headset_pose}")


xrt.close()
