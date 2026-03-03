from __future__ import annotations


import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass


import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from pathlib import Path

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from .commands import FootPosCommand, FootTrackCommand, VelocityToFootTrackCommand, FootTrackMotionTypeCommand

@configclass
class FootPosCommandCfg(CommandTermCfg):
    class_type: type = FootPosCommand

    path_dataset: str = MISSING
    """
    Target dataset path.
    """

    resample_ratio_range: tuple[float, float] = (0.0, 0.8)
    """
    At which section of the trajectory do resampling happens. Default to [0.0, 0.8].
    """

    qref_indices: Sequence[int] = MISSING
    """
    Reference traj indices.
    """

    asset_name: str = "robot"
    """
    Robot asset name.
    """
    body_names: list[str] = MISSING
    """
    Body of left hand and right hand.
    """
    rel_random: float = 0.1
    """
    The proportion of random poses not from the trajectory dataset.
    Increase this lead to more unstable traning, slower convergence, but more generalizability.
    """

    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # never resamples actively, only resample on reset

    EE_target_viz_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/EE_target",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1)
            )
        }
    )
    EE_curr_viz_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/EE_curr",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1)
            )
        }
    )
    random_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/random_viz"
    )

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges_lf: Ranges = MISSING
    ranges_rf: Ranges = MISSING
    """Ranges for the commands."""


@configclass
class FootTrackCommandCfg(CommandTermCfg):
    """Configuration for foot tracking command generator.

    Loads footstep tracking data from NPZ files containing joint positions/velocities,
    foot transforms, and footstep commands for locomotion tasks.
    """

    class_type: type = FootTrackCommand

    foot_track_data_path: str = MISSING
    """Path to foot tracking NPZ file.

    The NPZ file should contain:
    - q: (n, nq) - joint positions with base
    - qd: (n, nv) - joint velocities
    - T_blf: (n, 4) - body frame to left foot frame transform (x, y, z, yaw)
    - T_brf: (n, 4) - body frame to right foot frame transform (x, y, z, yaw)
    - T_stsw: (n, 4) - stance foot to swing foot transform (x, y, z, yaw)
    - p_wcom: (n, 3) - CoM position in world frame
    - T_wbase: (n, 7) - base transform in world frame (x, y, z, qw, qx, qy, qz)
    - v_b: (n, 6) - base velocity in base frame (linear xyz, angular xyz)
    - cmd_footstep: (n, 4) - [x, y, sin(yaw), cos(yaw)] in stance foot frame
    - cmd_stance: (n, 1) - 0=left stance, 1=right stance
    - cmd_countdown: (n, 1) - countdown timer: 0 during wait, 0->1->0 during step
    - traj: (k,) - starting indices of each trajectory
    - traj_dt: float - time step between frames
    """

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    resample_ratio_range: tuple[float, float] = (0.0, 0.8)
    """Range for sampling the starting position ratio within each trajectory.
    Values should be in [0, 1] where 0 is the start and 1 is the end of the trajectory."""

    rel_random: float = 0.0
    """The proportion of random poses not from the trajectory dataset.
    Increase this leads to more unstable training, slower convergence, but more generalizability.
    Set to 0.0 to disable random sampling."""

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Time range for active resampling. Default never resamples actively, only on reset."""

    left_foot_name: str | None = None
    """Name of the left foot body in the robot articulation (optional, for visualization)."""

    right_foot_name: str | None = None
    """Name of the right foot body in the robot articulation (optional, for visualization)."""

    foot_target_viz_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_target",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08)
            )
        }
    )
    """Visualization marker configuration for foot targets."""

    foot_cmd_viz_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_cmd",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.08, 0.08, 0.04),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    """Visualization marker configuration for foot command positions (green boxes)."""

    random_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_random_viz"
    )
    """Visualization marker configuration for random environment indicator."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for random foot transform commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges_lf: Ranges | None = None
    """Ranges for left foot random transforms (optional)."""

    ranges_rf: Ranges | None = None
    """Ranges for right foot random transforms (optional)."""


@configclass
class VelocityToFootTrackCommandCfg(CommandTermCfg):
    class_type: type = VelocityToFootTrackCommand

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    ranges: Ranges = MISSING
    """Ranges for the velocity commands."""

    # foot_track_data_path: str = MISSING
    
    resample_ratio_range: tuple[float, float] = (0.0, 0.8)
    """Range for sampling the starting position ratio within each trajectory.
    Values should be in [0, 1] where 0 is the start and 1 is the end of the trajectory."""

    rel_random: float = 0.0
    """The proportion of random poses not from the trajectory dataset.
    Increase this leads to more unstable training, slower convergence, but more generalizability.
    Set to 0.0 to disable random sampling."""

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Time range for active resampling. Default never resamples actively, only on reset."""

    base_left: tuple[float, float, float] = (0.0, 0.1, 0.0)
    """Base left position in local frame (COM = 0)."""

    base_right: tuple[float, float, float] = (0.0, -0.1, 0.0)
    """Base right position in local frame (COM = 0)."""

    countdown_takes_s: float = 1.0

    """Single swing period in seconds."""

    left_foot_name: str | None = None
    """Name of the left foot body in the robot articulation (optional, for visualization)."""

    right_foot_name: str | None = None
    """Name of the right foot body in the robot articulation (optional, for visualization)."""
    
    auto_resample_velocity: bool = True
    """Whether to auto resample velocity commands. Defaults to True."""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for random foot transform commands."""

        vel_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        vel_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        vel_yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""


    foot_target_viz_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_target",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08)
            )
        }
    )

    random_viz_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_random_viz"
    )

    foot_cmd_viz_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_cmd",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.08, 0.08, 0.04),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )

@configclass
class FootTrackMotionTypeCommandCfg(FootTrackCommandCfg):
    class_type: type = FootTrackMotionTypeCommand
