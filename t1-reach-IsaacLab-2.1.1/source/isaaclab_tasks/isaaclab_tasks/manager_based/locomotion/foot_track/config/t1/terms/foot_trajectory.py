from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import torch
import numpy as np
from typing import Any, Mapping, Sequence, Tuple


from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FOOT_TARGET_MARKER_CFG
from omni.isaac.lab.utils.math import quat_rotate, yaw_quat

# The position correction coefficients in Raibert's formula.
_KP = np.array([0.01, 0.01, 0.01]) * 3
# At the end of swing, we leave a small clearance to prevent unexpected foot
# collision.
MAX_CLEARANCE = 0.12
SINGLE_SWING_PERIOD = 0.4


def _gen_parabola(phase:  torch.tensor, start: torch.tensor, mid:  torch.tensor, end:  torch.tensor) ->  torch.tensor:
    """Gets a point on a parabola y = a x^2 + b x + c.

    The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
    the plane.

    Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

    Returns:
    The y value at x == phase.
    """
    mid_phase = 0.5
    delta_1 = mid - start
    delta_2 = end - start
    delta_3 = mid_phase**2 - mid_phase
    coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
    coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
    coef_c = start

    return coef_a * phase[:,None]**2 + coef_b * phase[:,None] + coef_c


def _gen_swing_foot_trajectory(input_phase:  torch.tensor, start_pos:  torch.tensor,
                               end_pos:  torch.tensor) ->  torch.tensor:
    """Generates the swing trajectory using a parabola.

    Args:
    input_phase: the swing/stance phase value between [0, 1].
    start_pos: The foot's position at the beginning of swing cycle.
    end_pos: The foot's desired position at the end of swing cycle.

    Returns:
    The desired foot position at the current phase.
    """
    # We augment the swing speed using the below formula. For the first half of
    # the swing cycle, the swing leg moves faster and finishes 80% of the full
    # swing trajectory. The rest 20% of trajectory takes another half swing
    # cycle. Intuitely, we want to move the swing foot quickly to the target
    # landing location and stay above the ground, in this way the control is more
    # robust to perturbations to the body that may cause the swing foot to drop
    # onto the ground earlier than expected. This is a common practice similar
    # to the MIT cheetah and Marc Raibert's original controllers.
    phase = input_phase
    # phase = torch.where(input_phase <= 0.5,
    #                     0.8* torch.sin(input_phase * math.pi),
    #                     0.8+ (input_phase-0.5)*0.4)
    
    skewness = 0.9
    mid = start_pos.clone().detach()
    mid += skewness*(end_pos - start_pos)
    mid[:, 2] = start_pos[:, 2] + MAX_CLEARANCE

    # mid = torch.max(end_pos[:,2], start_pos[:,2]) + max_clearance
    
    midair = _gen_parabola(phase, start_pos, mid, end_pos)
    midair[:,2] += phase * (end_pos[:, 2] - start_pos[:, 2])

    # PyType detects the wrong return type here.
    return midair
    # return torch.stack([x, y, z], dim=1)  # pytype: disable=bad-return-type

class T1GaitGenerator(ManagerTermBase):
    """Gait enforcing reward term for Digit.

    Based on Raibert heuristics
    """

    def __init__(self, cfg: RewardTermCfg, 
                env: ManagerBasedRLEnv,
                left_hip_offset: torch.Tensor,
    right_hip_offset: torch.Tensor):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.debug_vis = True
        self.left_hip_offset = left_hip_offset
        self.right_hip_offset = right_hip_offset

        if self.debug_vis:
            if not hasattr(self, "contact_visualizer"):
                self.foot_track_visualizer = VisualizationMarkers(FOOT_TARGET_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor"))
            self.foot_track_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.foot_track_visualizer.set_visibility(False)

        self.env = env
        self.reset_clock()

        # Temporarily, set body_ids[[0]] as starting foot
        self.LEFT=0
        self.RIGHT=1

        self.left_foot_start_pos = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids[self.LEFT], :3]
        self.right_foot_start_pos = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids[self.RIGHT], :3]

        self.init_foot_target()

        # debug purpose
        self.midair_foot_target_pos = self.right_foot_start_pos.clone().detach()

    def reset_clock(self, env_ids=None):
        if env_ids is None:
            self.init_clock = self.env.get_phase().to(torch.float)
            clock = self.env.get_phase().to(torch.float) - self.init_clock
            self.prev_clock = (clock - torch.floor(clock)).to(torch.float)
            self.curr_clock = self.prev_clock[:]
        else:
            self.init_clock[env_ids] = self.env.get_phase().to(torch.float)[env_ids]
            clock = self.env.get_phase().to(torch.float)[env_ids] - self.init_clock[env_ids]
            self.prev_clock[env_ids] = (clock - torch.floor(clock)).to(torch.float)
            self.curr_clock[env_ids] = self.prev_clock[env_ids]
    
    def update_clock(self, env_ids=None):
        if env_ids is None:
            clock = self.env.get_phase().to(torch.float) - self.init_clock
            self.prev_clock = self.curr_clock[:]
            self.curr_clock = clock - torch.floor(clock)
            # self.curr_clock = self.prev_clock[:]
        else:
            clock = self.env.get_phase().to(torch.float)[env_ids] - self.init_clock[env_ids]
            self.prev_clock[env_ids] = self.curr_clock[env_ids]
            self.curr_clock[env_ids] = clock - torch.floor(clock)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.reset_clock()

            self.left_foot_start_pos = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids[self.LEFT], :3]
            self.right_foot_start_pos = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids[self.RIGHT], :3]

            self.left_foot_target_pos = self.left_foot_start_pos.clone().detach()
            self.right_foot_target_pos = self.right_foot_start_pos.clone().detach()
            
            return
        
        self.reset_clock(env_ids)
        
        self.left_foot_start_pos[env_ids] = self.asset.data.body_pos_w[env_ids, self.asset_cfg.body_ids[self.LEFT], :3]
        self.right_foot_start_pos[env_ids] = self.asset.data.body_pos_w[env_ids, self.asset_cfg.body_ids[self.RIGHT], :3]

        self.left_foot_target_pos[env_ids] = self.left_foot_start_pos[env_ids]
        self.right_foot_target_pos[env_ids] = self.right_foot_start_pos[env_ids]
        
        return

    def _compute_reward(self):
        l_phase = torch.nonzero(torch.sin(self.curr_clock*2*math.pi) > 0, as_tuple=True)[0]
        r_phase = torch.nonzero(torch.sin(self.curr_clock*2*math.pi) <= 0, as_tuple=True)[0]

        assert len(l_phase) + len(r_phase) == self.num_envs, "...?"
        if len(l_phase) > 0:
            current_pos = self.asset.data.body_pos_w[l_phase, self.asset_cfg.body_ids[self.LEFT], :3] # current foot pos
            target_pos = self.compute_midair_foot_target(l_phase, is_left=True)  # target midair foot pos
            self.midair_foot_target_pos[l_phase] = target_pos[:]
            error = torch.norm(current_pos[:,:2] - target_pos[:, :2], dim=1) + \
                        5 * torch.norm(current_pos[:, 2:] - target_pos[:, 2:], dim=1)
            self.reward_buf[l_phase] = torch.exp(-error/0.05)
        if len(r_phase) > 0:
            current_pos = self.asset.data.body_pos_w[r_phase, self.asset_cfg.body_ids[self.RIGHT], :3] # current foot pos
            target_pos = self.compute_midair_foot_target(r_phase, is_left=False)  # target midair foot pos
            self.midair_foot_target_pos[r_phase] = target_pos[:]
            error = torch.norm(current_pos[:,:2] - target_pos[:, :2], dim=1) + \
                        5 * torch.norm(current_pos[:, 2:] - target_pos[:, 2:], dim=1)
            self.reward_buf[r_phase] = torch.exp(-error/0.05)
        if self.debug_vis:
            marker_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            frame_origins = self.midair_foot_target_pos
            self.foot_track_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

    def compute_midair_foot_target(self, env_ids, is_left=True):
        if is_left:
            gait_phase = self.curr_clock[env_ids] / 0.5
            normalized_phase = gait_phase - torch.floor(gait_phase)
            start_pos = self.left_foot_start_pos[env_ids]
            end_pos = self.left_foot_target_pos[env_ids]
        if not is_left:
            gait_phase = self.curr_clock[env_ids] / 0.5
            normalized_phase = gait_phase - torch.floor(gait_phase)
            start_pos = self.right_foot_start_pos[env_ids]
            end_pos = self.right_foot_target_pos[env_ids]

        
        # phase = torch.linspace(0, 1, 10).to(self.device)
        # total_plot = []
        # for p in phase:
        #     total_plot.append(_gen_swing_foot_trajectory(p.unsqueeze(dim=0), start_pos[:1,:], end_pos[:1,:]))
        # total_plot = torch.stack(total_plot, dim=0).detach().cpu().numpy()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')


        # # Plot the 3D parabolic line
        # ax.plot(total_plot[:,0,0], total_plot[:,0,1], total_plot[:,0,2], color='blue', linewidth=2, label="3D Parabolic Line")
        # ax.scatter(total_plot[:,0,0], total_plot[:,0,1], total_plot[:,0,2], color='red', s=50, label="Points on the line")
        # # Add labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()

        # # Show plot
        # plt.show()

        return _gen_swing_foot_trajectory(normalized_phase,
                                          start_pos,
                                          end_pos)

    def update_foot_target(self):
        # compute a target trajectory
        # self.asset.body_names.index("Hip_Pitch_Left")
        # 3
        # self.asset.body_names.index("Hip_Pitch_Right")
        # 4
        left_swing_begin = torch.nonzero(
                                torch.logical_and(
                                    torch.sin(self.prev_clock*2*math.pi)<=0, 
                                    torch.sin(self.curr_clock*2*math.pi) > 0
                                ), 
                                as_tuple=True)[0]
        right_swing_begin = torch.nonzero(
                                torch.logical_and(
                                    torch.sin(self.prev_clock*2*math.pi)>0,
                                    torch.sin(self.curr_clock*2*math.pi) <= 0
                                ),
                                as_tuple=True)[0]
        
        if len(left_swing_begin)>0:
            self.left_foot_target_pos[left_swing_begin] = self.compute_foot_target(left_swing_begin, "Hip_Pitch_Left")
            self.left_foot_start_pos[left_swing_begin] = self.asset.data.body_pos_w[left_swing_begin, self.asset_cfg.body_ids[self.LEFT], :3]
        
        if len(right_swing_begin)>0:
            self.right_foot_target_pos[right_swing_begin] = self.compute_foot_target(right_swing_begin, "Hip_Pitch_Right")
            self.right_foot_start_pos[right_swing_begin] = self.asset.data.body_pos_w[right_swing_begin, self.asset_cfg.body_ids[self.RIGHT], :3]

    def init_foot_target(self):
        self.left_foot_target_pos = self.compute_foot_target(offset_body_name="Hip_Pitch_Left")
        self.right_foot_target_pos = self.compute_foot_target(offset_body_name="Hip_Pitch_Right")

    def compute_foot_target(self, env_ids=None, foot_indicator="left"):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        # hip_ids = self.asset.body_names.index(offset_body_name)
        cmd_vel = self.env.command_manager.get_command("base_velocity")
        vyaw = cmd_vel[env_ids, 2]
        
        if foot_indicator == "left":
            hip_offset = self.left_hip_offset[env_ids]
        else:
            hip_offset = self.right_hip_offset[env_ids]

        hip_normal = torch.vstack([ -hip_offset[:,1], 
                                    hip_offset[:,0], 
                                    # torch.zeros(hip_offset.shape[:-1], device=hip_offset.device)
                                   ]).transpose(0,1)
        # shape len(env_ids)X3
        
        yaw_component = vyaw[:, None] * hip_normal
        cmd_vel_w = quat_rotate(
            yaw_quat(self.asset.data.root_quat_w), cmd_vel[:, :3]
        )   
        xy_component = cmd_vel_w[env_ids, 0:2]
        
        target_pos_xy = (yaw_component + xy_component)*SINGLE_SWING_PERIOD
        target_pos_z = torch.zeros(hip_offset.shape[:-1], device=hip_offset.device)
        # target_pos_z += 0.01 + 0.0630
        
        return torch.concat([target_pos_xy, target_pos_z.unsqueeze(-1)], dim=1)
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        self.update_clock()
        self.update_foot_target()
        self._compute_reward()

        return self.reward_buf.clone().detach()
