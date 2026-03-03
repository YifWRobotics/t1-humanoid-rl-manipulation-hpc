# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshRandomGridTerrainCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from .rough_env_cfg import T1RoughEnvCfg


@configclass
class T1FlatEnvCfg(T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # self.scene.terrain.terrain_type = "generator"
        # self.scene.terrain.terrain_generator.sub_terrains = {
        #     "plane": terrain_gen.MeshPlaneTerrainCfg(
        #         proportion=0.2
        #     ),
        #     "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #         proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0, size=(8.0, 8.0)
        #     ),
        #     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #         proportion=0.2, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.25
        #     ),
        #     "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #         proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        #     ),
        #     "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #         proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        #     )
        # }
        # self.scene.terrain.max_init_terrain_level = 5

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.enable_corruption = True
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.debug_vis = False
        self.episode_length_s = 20.0


@configclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 10
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        # self.events.base_external_force_torque = None
        self.episode_length_s = 1000000.0  # non-stop playing

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.3
        self.commands.base_velocity.rel_rotonly_envs = 0.1

        self.commands.base_velocity.debug_vis = True

        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = "performance"

        # self.actions = None
        # self.rewards = None
        # self.termination = None
        # self.curriculum = None
