import sys
import booster_robotics_sdk_python as sdk


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        sys.exit(-1)

    sdk.ChannelFactory.Instance().Init(0, sys.argv[1])
    client = sdk.B1LocoClient()
    client.Init()
    client.ChangeMode(sdk.RobotMode.kCustom)
    motion_param = sdk.GripperMotionParameter()

    while True:
        input_cmd = input().strip()
        transform = sdk.Transform()
        print(client.GetFrameTransform(sdk.Frame.kBody, sdk.Frame.kLeftHand, transform))
        print(transform.position.x, transform.position.y, transform.position.z)

        if input_cmd == "open":
            print("Open Gripper")
            motion_param.position = 500
            motion_param.force = 100
            motion_param.speed = 100
            res = client.ControlGripper(motion_param, sdk.GripperControlMode.kPosition, sdk.B1HandIndex.kLeftHand)
            res = client.ControlGripper(motion_param, sdk.GripperControlMode.kPosition, sdk.B1HandIndex.kRightHand)
            print(res)

        elif input_cmd == "close":
            print("Closing Gripper")
            motion_param.position = 100
            motion_param.force = 100
            motion_param.speed = 100
            res = client.ControlGripper(motion_param, sdk.GripperControlMode.kPosition, sdk.B1HandIndex.kLeftHand)
            res = client.ControlGripper(motion_param, sdk.GripperControlMode.kPosition, sdk.B1HandIndex.kRightHand)
            print(res)

        else:
            print("Invalid Command.")


if __name__ == "__main__":
    main()
