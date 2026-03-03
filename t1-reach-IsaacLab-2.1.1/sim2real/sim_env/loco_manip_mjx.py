from sim2real.utils.math import quat_rotate_numpy
from sim2real.sim_env.base_sim_mjx import BaseSimulatorMJX
import sys

import mujoco
import mujoco.viewer
import numpy as np
import yaml
import argparse
import jax.numpy as jp

sys.path.append("../")
sys.path.append("./sim2real")


def quat_rotate_jax(q, v):
    """Quaternion rotation (forward rotation)"""
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, jp.newaxis]
    b = jp.cross(q_vec, v) * q_w[:, jp.newaxis] * 2.0
    dot_product = jp.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a + b + c


class LocoManipSimulatorMJX(BaseSimulatorMJX):
    def __init__(self, config):
        super().__init__(config)
        self.EE_xfrc = 0
        self.t = 0
        self.left_hand_link_name = self.config.get(
            "left_hand_link_name", "left_hand_link")
        self.right_hand_link_name = self.config.get(
            "right_hand_link_name", "right_hand_link")

    def init_scene(self):
        super().init_scene()
        NUM_FEET_SENSORS = 8
        # Assuming you know the order of the sensors in the XML
        self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3
        # Customize the viewer options
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()

        mjx_data = self.mjx_data
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                elastic_force = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
                new_xfrc_applied = mjx_data.xfrc_applied \
                    .at[self.band_attached_link, :3] \
                    .set(elastic_force)
                mjx_data = mjx_data.replace(xfrc_applied=new_xfrc_applied)

        if self.elastic_band.estimate:
            self.EE_xfrc = self.elastic_band.apply_force
            base_target_axis = jp.array([-1.0, 0.0, 0.0])
            body_quat = self.mj_data.qpos[3:7]
            force_in_global = quat_rotate_jax(
                jp.array([body_quat]), base_target_axis * self.EE_xfrc)

            left_hand_id = self.mj_model.body(self.left_hand_link_name).id
            right_hand_id = self.mj_model.body(self.right_hand_link_name).id

            new_xfrc_applied = mjx_data.xfrc_applied \
                .at[left_hand_id, 0:3] \
                .set(force_in_global) \
                .at[right_hand_id, 0:3] \
                .set(force_in_global)
            mjx_data = mjx_data.replace(xfrc_applied=new_xfrc_applied)
            self.t += self.dt

        self.compute_torques()

        if self.robot_bridge.free_base:
            ctrl = jp.concatenate((jp.zeros(6), self.torques))
        else:
            ctrl = self.torques
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # Step simulation and update
        self.mjx_data = self.jit_step(mjx_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str,
                        default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = LocoManipSimulatorMJX(config)
    simulation.sim_thread.start()
