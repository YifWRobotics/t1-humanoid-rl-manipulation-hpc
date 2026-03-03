# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RL agents for the reach task with T1 robot."""

from .rsl_rl_ppo_cfg import T1ReachPPORunnerCfg
from .rsl_rl_distill_cfg import T1ReachDistillRunnerCfg

__all__ = ["T1ReachPPORunnerCfg", "T1ReachDistillRunnerCfg"]

