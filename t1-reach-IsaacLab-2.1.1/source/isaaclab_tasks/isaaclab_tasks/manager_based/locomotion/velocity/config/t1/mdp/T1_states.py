import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

ZERO_THRESHOLD = 0.10


def root_state_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root state in world frame."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1).to(
        env.device
    )


def should_stand(env: ManagerBasedRLEnv, zero_threshold: float = ZERO_THRESHOLD) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) < zero_threshold


def should_walk(env: ManagerBasedRLEnv, zero_threshold: float = ZERO_THRESHOLD) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) >= zero_threshold
