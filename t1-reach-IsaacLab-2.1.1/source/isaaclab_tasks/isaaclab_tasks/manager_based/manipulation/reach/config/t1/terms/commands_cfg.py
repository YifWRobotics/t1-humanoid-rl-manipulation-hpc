from __future__ import annotations


import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
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
from .commands import SE3IKCommand
from isaaclab.envs.mdp.events import PREP_STATE


@configclass
class SE3IKCommandCfg(CommandTermCfg):
    class_type: type = SE3IKCommand

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
    rel_random: float = 0.0
    """
    The proportion of random poses from the trajectory dataset.
    """
    rel_default: float = 0.0
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

    ranges_lh: Ranges = MISSING
    ranges_rh: Ranges = MISSING
    """Ranges for the commands."""