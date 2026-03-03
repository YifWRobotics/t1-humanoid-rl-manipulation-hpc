import sys
import booster_robotics_sdk_python as sdk
import time


def handler(msg):
    for joint in msg.motor_state_serial:
        print(f"{joint.q},")
    print("-------")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        sys.exit(-1)

    sdk.ChannelFactory.Instance().Init(0, sys.argv[1])
    client = sdk.B1LocoClient()
    client.Init()
    client.ChangeMode(sdk.RobotMode.kCustom)
    robot_lowstate_subscriber = sdk.B1LowStateSubscriber(handler)
    robot_lowstate_subscriber.InitChannel()

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
