# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for t1."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original, left-right. The symmetry transformations are beneficial 
    for reinforcement learning tasks by providing additional diverse data without requiring 
    additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        num_envs = obs.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = torch.zeros(num_envs * 2, obs.shape[1], device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- left-right
        obs_aug[num_envs : 2 * num_envs] = _transform_obs_left_right(env.unwrapped, obs, obs_type)
    else:
        obs_aug = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(num_envs * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the t1 robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # ang vel
    obs[:, 10:13] = obs[:, 10:13] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 13:16] = obs[:, 13:16] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 16:19] = obs[:, 16:19] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 19:22] = obs[:, 19:22] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 22:25] = obs[:, 22:25] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 25:28] = obs[:, 25:28] * torch.tensor([1, -1, 1], device=device)
    obs[:, 28:31] = obs[:, 28:31] * torch.tensor([1, -1, 1], device=device)
    obs[:, 31:34] = obs[:, 31:34] * torch.tensor([1, -1, 1], device=device)
    obs[:, 34:37] = obs[:, 34:37] * torch.tensor([1, -1, 1], device=device)
    obs[:, 37:40] = obs[:, 37:40] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 40:43] = obs[:, 40:43] * torch.tensor([1, -1, -1], device=device)
    obs[:, 43:46] = obs[:, 43:46] * torch.tensor([1, -1, -1], device=device)
    obs[:, 46:49] = obs[:, 46:49] * torch.tensor([1, -1, -1], device=device)
    obs[:, 49:52] = obs[:, 49:52] * torch.tensor([1, -1, -1], device=device)
    obs[:, 52:55] = obs[:, 52:55] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 55:84] = _switch_anymal_joints_left_right_29D(obs[:, 55:84])
    obs[:, 84:113] = _switch_anymal_joints_left_right_29D(obs[:, 84:113])
    obs[:, 113:142] = _switch_anymal_joints_left_right_29D(obs[:, 113:142])
    obs[:, 142:171] = _switch_anymal_joints_left_right_29D(obs[:, 142:171])
    obs[:, 171:200] = _switch_anymal_joints_left_right_29D(obs[:, 171:200])
    # joint vel
    obs[:, 200:229] = _switch_anymal_joints_left_right_29D(obs[:, 200:229])
    obs[:, 229:258] = _switch_anymal_joints_left_right_29D(obs[:, 229:258])
    obs[:, 258:287] = _switch_anymal_joints_left_right_29D(obs[:, 258:287])
    obs[:, 287:316] = _switch_anymal_joints_left_right_29D(obs[:, 287:316])
    obs[:, 316:345] = _switch_anymal_joints_left_right_29D(obs[:, 316:345])
    # last actions
    obs[:, 345:358] = _switch_anymal_joints_left_right_13D(obs[:, 345:358])
    obs[:, 358:371] = _switch_anymal_joints_left_right_13D(obs[:, 358:371])
    obs[:, 371:384] = _switch_anymal_joints_left_right_13D(obs[:, 371:384])
    obs[:, 384:397] = _switch_anymal_joints_left_right_13D(obs[:, 384:397])
    obs[:, 397:410] = _switch_anymal_joints_left_right_13D(obs[:, 397:410])

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    t1 robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_left_right_13D(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]
"""


def _switch_anymal_joints_left_right_29D(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [7, 11, 15, 19, 23, 27]] = joint_data[..., [8, 12, 16, 20, 24, 28]]
    # right <-- left
    joint_data_switched[..., [8, 12, 16, 20, 24, 28]] = joint_data[..., [7, 11, 15, 19, 23, 27]]

    return joint_data_switched


def _switch_anymal_joints_left_right_13D(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [1, 2, 3, 4, 5, 6]] = joint_data[..., [7, 8, 9, 10, 11, 12]]
    # right <-- left
    joint_data_switched[..., [7, 8, 9, 10, 11, 12]] = joint_data[..., [1, 2, 3, 4, 5, 6]]

    return joint_data_switched
