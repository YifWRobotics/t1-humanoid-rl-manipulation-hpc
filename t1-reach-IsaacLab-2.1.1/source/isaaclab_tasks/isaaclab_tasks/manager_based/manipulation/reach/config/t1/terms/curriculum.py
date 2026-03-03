from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
import isaaclab.envs.mdp as mdp


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def linearly_change_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    attr_name: str,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int
):
    """Curriculum helper that linearly changes a parameter value over time.

    This function is designed to be used with mdp.modify_term_cfg to change
    command term parameters like rel_random.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        data: Current value of the parameter (not modified, just returned if no change).
        start_value: The starting value of the parameter.
        end_value: The ending value of the parameter.
        start_step: The step at which to start changing the parameter.
        end_step: The step at which to end changing the parameter.

    Returns:
        The new parameter value, or mdp.modify_term_cfg.NO_CHANGE if before start_step.

    Example:
        To change gmr command's rel_random from 0.0 to 1.0:

        gmr_random_ratio = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.gmr.rel_random",
                "modify_fn": linearly_change_param,
                "modify_params": {
                    "start_value": 0.0,
                    "end_value": 1.0,
                    "start_step": 1000 * 24,
                    "end_step": 10000 * 24,
                }
            }
        )
    """
    if env.common_step_counter < start_step:
        # Before curriculum starts, don't change
        value = start_value
    elif env.common_step_counter > end_step:
        # After curriculum ends, use end value
        value = end_value
    else:
        # Linear interpolation during curriculum
        alpha = (env.common_step_counter - start_step) / (end_step - start_step)
        value = (1 - alpha) * start_value + alpha * end_value

    cfg = env.command_manager.get_term_cfg(term_name)
    setattr(cfg, attr_name, value)
    env.command_manager.set_term_cfg(term_name, cfg)
    return value
