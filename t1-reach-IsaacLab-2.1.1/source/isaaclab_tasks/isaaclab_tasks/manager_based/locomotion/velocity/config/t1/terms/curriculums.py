
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_command_range(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, ranges, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.command_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.ranges = ranges
        env.command_manager.set_term_cfg(term_name, term_cfg)


class modify_reward_weight_on_greater_than_reward(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        metric_name: str,
        metric_threshold: float
    ) -> float:
        # update term settings
        if "log" in env.extras.keys() and env.extras["log"][f"Episode_Reward/{metric_name}"] > metric_threshold:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight


def modify_command_range_on_greater_than_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str, ranges: float,
    metric_name: str,
    metric_threshold: float
) -> float:
    # update term settings
    if "log" in env.extras.keys() and env.extras["log"][f"Episode_Reward/{metric_name}"] > metric_threshold:
        term_cfg = env.command_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.ranges = ranges
        env.command_manager.set_term_cfg(term_name, term_cfg)


def modify_done_term_on_greater_than_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    term_param_name : str,
    value,
    metric_name: str,
    metric_threshold: float
) -> float:
    # update term settings
    if "log" in env.extras.keys() and env.extras["log"][f"Episode_Reward/{metric_name}"] > metric_threshold:
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # update term settings
        setattr(term_cfg, term_param_name, value)
        env.termination_manager.set_term_cfg(term_name, term_cfg)


def modify_event_on_greater_than_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    term_param_name : str,
    value,
    metric_name: str,
    metric_threshold: float
) -> float:
    # update term settings
    if "log" in env.extras.keys() and env.extras["log"][f"Episode_Reward/{metric_name}"] > metric_threshold:
        event_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        setattr(event_cfg, term_param_name, value)
        env.event_manager.set_term_cfg(term_name, event_cfg)
        return 1.0  # show curriculum is activated
    return 0.0  # show curriculum is not activated


def linearly_alter_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int
) -> float:
    """Curriculum that linearly alters a reward weight between two steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        start_weight: The starting weight of the reward term.
        end_weight: The ending weight of the reward term.
        start_step: The step at which to start altering the weight.
        end_step: The step at which to end altering the weight.
    """
    if env.common_step_counter < start_step:
        weight = start_weight
    elif env.common_step_counter > end_step:
        weight = end_weight
    else:
        alpha = (env.common_step_counter - start_step) / (end_step - start_step)
        weight = (1 - alpha) * start_weight + alpha * end_weight

    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight


def linear_alter_command_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    subterm_name: str,
    start_value: dict,
    end_value: dict,
    start_step: int,
    end_step: int
) -> float:
    if env.common_step_counter < start_step:
        value = start_value
    elif env.common_step_counter > end_step:
        value = end_value
    else:
        alpha = (env.common_step_counter - start_step) / (end_step - start_step)
        value = (1 - alpha) * start_value + alpha * end_value

    term_cfg = env.command_manager.get_term_cfg(term_name)
    setattr(term_cfg.ranges, subterm_name, (-value, value))
    env.command_manager.set_term_cfg(term_name, term_cfg)
    return value
