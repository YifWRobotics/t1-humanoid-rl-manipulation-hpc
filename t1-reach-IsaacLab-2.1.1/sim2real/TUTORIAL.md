# Sim2Real and VR Teleopration Documentation

This document is for Sim2Real Transfer using the `IsaacLab - FALCON MuJoCo - Hardware` pipeline, as well as for VR Teleoperation using the XRoboToolkit.


# Installation
## Install booster_robotics_sdk for Booster T1 deployment
Note that the official [booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) does not provide state publisher and command receiver, so I improve the repo a bit and add them in my [forked repo](https://github.com/hang0610/booster_robotics_sdk). Also booster sdk is NOT supported on Mac OS yet.
```bash
git clone https://github.com/hang0610/booster_robotics_sdk
cd booster_robotics_sdk
pip3 install pybind11
pip3 install pybind11-stubgen
# Build & Install
mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDING=on
make
sudo make install
```
## Install xrobotoolkit  on PC
**Install XRoboToolkit-PC-Service**  
   - Download [deb package for ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb), or build from the [repo source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).
   - To install, use command
     ```bash
      sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
      ```
**Install XRoboToolkit-PC-Service-Pybind**  
This is tricky. The installation in readme of this repo is kinda not accurate. Please ask Zimeng or Zhaoyuan. 
   - [XRoboToolkit-PC-Service-Pybind](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)

## Install the Dependencies Required for the Sim2Real Pipeline in FALCON

```bash
cd sim2real
pip install -r requirements.txt
```


## Sim2Sim and Sim2Real share the same pipeline

After getting a `.pt` file from logs / wandb, do:

1. Navigate to "Run and Debug" on the side panel. Choose the launch target to be `Play: Local` and click start.
2. Two prompts will show up, enter the environment name and the relative path (from the workspace directory) to the `.pt` weight file.
3. IsaacLab environment will show up. Check the performance of the policy.
4. After running the task, there will be an `exported` folder under the directory of `.pt` weight file. Inside the folder contains the policy in both `.pt` and `.onnx` format. Copy the onnx format to `models` folder under `sim2real`, and change the `model_path` under `sim2real/config/t1/t1_29dof_loco.yaml`. 
5. If we run the simulator, then it's sim2sim. If we don't start the simulator, then it's sim2real. 
6. For sim2sim, evaluate the performance of the policy in the simulator. For simulator / policy control, please refer to **keyboard shortcuts**. We start the simulator by running:

```python
python sim2real/sim_env/loco_manip.py --config=sim2real/config/t1/t1_29dof_loco.yaml 
```

You should see a window popping up.

7. Start the policy by running

```python
python sim2real/rl_policy/teleop_ik/teleop_ik_xrobo_loco.py --config=sim2real/config/t1/t1_29dof_loco.yaml
```

The policy can be changed by selecting different python file under `rl_policy`. For teleopration policy with MuJoCo visualization, it can be enabled by passing in `--headless false`.

8. After evaluating in simulation, we do sim2real. For sim2real, don't start simulation, directly start policy. 
Connect to robot following the [T1 manual](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f)
Deploy on hardware by running:
```python
python sim2real/rl_policy/teleop_ik/teleop_ik_xrobo_loco.py --config=sim2real/config/t1/t1_29dof_loco_hardware.yaml
```
Notice the only change is the extra "hardware" in the name of the configuration file.

Here are some **keyboard shortcuts**:

<details>
<summary>Keyboard Shortcuts in Mujoco</summary>

- `7`: raise elastic band height
- `8`: lower elastic band height
- `9`: toggle elastic band
- `backspace`: reset simulation

</details>

<details>
<summary>Keyboard Shortcuts in Policy Terminal</summary>

- `]`: start using policy actions
- `o`: disabling policy actions
- `i`: start initial state
- `=`: switch between standing and stepping
- `w`: increase linear velocity in `x` direction
- `s`: decrease linear velocity in `x` direction
- `a`: increase linear velocity in `y` direction
- `d`: decrease linear velocity in `y` direction
- `q`: decrease angular velocity in `z` direction
- `e`: increase angular velocity in `z` direction
- `z`: set velocity to zero
- `1`: increase base height (if the policy allows)
- `2`: decrease base height (if the policy allows)
- `5`: decrease kp scale by 0.01
- `6`: increase kp scale by 0.01
- `4`: decrease kp scale by 0.1
- `7`: increase kp scale by 0.1
- `0`: reset kp scale to 1.0
  
</details>

## VR Teleoperation

1. Install [XRoboToolkit-Service-Pybind](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind), follow the instruction in the repository.
2. Make sure the policy computer and the VR headset is connected under the same network. Currently the setup is to use the TP-Link Router, SSID `LIDAR_WiFi`, password `12345678`.
3. Start XRoboToolkit on the policy computer. Start XRoboToolkit Quest Client on the VR headset. (To install Quest Client, utilize the Meta Developer Hub, enable developer mode and install the pre-built `.apk`.)
4. On the panel, click "Connect", select the correct IP. Then, check "Headset" and "Controller", as well as "Send". The status should turn in to green "WORKING".
5. Start the policy. Here's a general guide for teleop:
   - Trigger - Grippper Control
   - Hold Grip - Move the End-Effector
   - Release Grip - Hold the End-Effector
   - A - Reframe Headset Positive Yaw Direction
   - X - Standing / Walking Switch
   - Left Axis - Translational Movement
   - Right X Axis - Rotational Movement
