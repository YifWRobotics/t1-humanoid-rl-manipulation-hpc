from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from .commands import PhaseCommand


class reset_from_trajectory_reference(ManagerTermBase):
    """
    Randomize starting pose of the robot from a given trajectory.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # load trajectory
        path_data = Path(cfg.params["trajectory_npz"])
        if not path_data.exists() or not path_data.is_file() or path_data.suffix != ".npz":
            raise ValueError(f"Trajectory npz file {path_data} is invalid.")
        data = np.load(path_data)
        self.traj_qref = torch.tensor(data["q"], device=env.device, dtype=torch.float32)  # shape: (step_traj, nqref)
        self.traj_phase = torch.tensor(data["phase"], device=env.device, dtype=torch.float32)  # shape: (step_traj, )
        self.traj_root_pose = torch.tensor(data["root_pose"], device=env.device, dtype=torch.float32)  # shape: (step_traj, 7)
        self.traj_max_step = int(data["phase"].shape[0])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        trajectory_npz: str,  # no use, as it is already loaded
        asset_cfg: SceneEntityCfg,
        rel_default: float = 0.6
    ):
        asset: Articulation = env.scene[asset_cfg.name]

        # get joint numbers for resetting
        joint_ids = asset_cfg.joint_ids
        step_idx = torch.randint_like(env_ids, low=0, high=self.traj_max_step)
        reset_q = self.traj_qref[step_idx][:, joint_ids]  # shape: (n, nqref)
        reset_phase = self.traj_phase[step_idx]  # shape: (n, )

        # determine which to reset from default
        flag_default = torch.empty_like(env_ids).random_(0, 1) < rel_default
        default_q = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
        reset_q = torch.where(flag_default.unsqueeze(1), default_q, reset_q)
        reset_phase = torch.where(flag_default, torch.zeros_like(reset_phase), reset_phase)

        # write to sim and finish the reset
        self.asset.write_joint_position_to_sim(
            reset_q,
            joint_ids, env_ids
        )
        phase_command: PhaseCommand = env.command_manager.get_term("phase")
        phase_command.set_phase(reset_phase, env_ids)
