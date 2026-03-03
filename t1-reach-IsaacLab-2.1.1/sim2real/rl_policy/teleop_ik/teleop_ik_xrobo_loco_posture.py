import sys
from pathlib import Path

sys.path.append("../")
sys.path.append("./rl_policy")
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import yaml


from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy


class TeleopXRoboLocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=1.0
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        print(self.num_upper_dofs)
        print(self.upper_dof_indices)
        print(self.ref_upper_dof_pos)
        self.residual_upper_body_action = False
        self.last_policy_action = np.zeros((1, 13))
        self.obs_history = {}
        self.obs_history_len = 5
        self.single_obs_len = 82  # action not include waist

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        current_obs_dict["command_base_height"] = self.base_height_command

        return current_obs_dict

    def rl_inference(self, robot_state_data):
        np.set_printoptions(precision=3, suppress=True)
        obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        phase_time = self._get_obs_phase_time()
        obs_buffer_dict["command_lin_vel"] = self.lin_vel_command.reshape(1, 2)
        obs_buffer_dict["command_ang_vel"] = self.ang_vel_command.reshape(1, 1)
        obs_buffer_dict["sin_phase"] = np.sin(2 * phase_time * np.pi)
        obs_buffer_dict["cos_phase"] = np.cos(2 * phase_time * np.pi)

        if np.linalg.norm(np.concatenate([self.lin_vel_command[0], self.ang_vel_command[0]])) < 0.01:
            obs_buffer_dict["sin_phase"] = np.sin(2 * 0 * np.pi).reshape(1, 1)
            obs_buffer_dict["cos_phase"] = np.cos(2 * 0 * np.pi).reshape(1, 1)

        obs_buffer_dict["last_policy_action"] = self.last_policy_action.copy()

        obs_list = [
            'sin_phase', 'cos_phase',
            'base_ang_vel', 'projected_gravity',
            'command_lin_vel', 'command_ang_vel',
            'dof_pos', 'dof_vel', 'last_policy_action'
        ]

        # print(f"LinVel: {obs_buffer_dict['command_lin_vel']}, AngVel: {obs_buffer_dict['command_ang_vel']}")

        # initialize and roll observation history if needed
        for key in obs_list:
            if key not in self.obs_history.keys():
                self.obs_history[key] = np.zeros((self.obs_history_len, obs_buffer_dict[key].shape[1]))

            # roll history: oldest observations removed, make space for newest
            self.obs_history[key] = np.roll(self.obs_history[key], -1, axis=0)
            self.obs_history[key][-1] = obs_buffer_dict[key]

        # collect observations in specified layout
        collected_obs = []
        for key in obs_list:
            collected_obs.append(self.obs_history[key].flatten())

        # concatenate all observation types
        collected_obs = np.concatenate(collected_obs, axis=0)

        # run policy
        policy_action = self.policy({"obs": collected_obs.reshape(1, -1).astype(np.float32)})

        # WBC actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos
                - self.default_dof_angles[self.upper_dof_indices]
            )

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data
        # Apply upper body controller
        if self.upper_body_controller:
            # Control upper qpos and tau
            upper_body_qpos = self.upper_body_controller.get_qpos_upper()
            true_qpos = upper_body_qpos
            self.ref_upper_dof_pos = true_qpos.reshape(1, -1)

        if self.state_processor.robot_state_data is None:
            return

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7: 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            stair_posture = np.array([
                0.0,
                0.10405,
                0,
                0,
                0.0702,
                -0.1807,
                -0.022,
                -1.54725,
                0,
                0,
                1.5327,
                0.0084,
                0.0088,]).reshape(1, 13)
            true_act = stair_posture
            true_act[0][0] = 0.0  # FIXME: temporarily no waist
            q_target = self.get_init_target(robot_state_data)
            q_target[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            q_target[:, self.lower_dof_indices] = true_act

        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(
                q_target[0],
                self.motor_pos_lower_limit_list,
                self.motor_pos_upper_limit_list
            )

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(
            cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7: 7 + self.num_dofs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get(
        "model_path")
    if not model_path:
        raise ValueError(
            "model_path must be provided either via --model_path argument or in config file")

    policy = TeleopXRoboLocoManipPolicy(
        config=config,
        model_path=model_path,
        rl_rate=50,
        policy_action_scale=1.0
    )
    policy.run()
