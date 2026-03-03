# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
)
from rsl_rl.modules import StudentTeacher

# Legacy checkpoints for the reach task were trained with a 300-dimensional
# observation vector for the teacher policy. The current environment delivers a
# longer observation (448), so we trim the additional components on-the-fly.
_LEGACY_TEACHER_OBS_DIM = 300

class T1ReachStudentTeacher(StudentTeacher):
    """Student-teacher network that trims teacher observations to legacy size."""

    def __init__(self, num_student_obs, num_teacher_obs, num_actions, **kwargs):
        super().__init__(num_student_obs, _LEGACY_TEACHER_OBS_DIM, num_actions, **kwargs)

    def evaluate(self, teacher_observations):
        trimmed_obs = teacher_observations[..., :_LEGACY_TEACHER_OBS_DIM]
        return super().evaluate(trimmed_obs)


_CLASS_EXPR = (
    "__import__('isaaclab_tasks.manager_based.manipulation.reach.config.t1.agents.rsl_rl_distill_cfg', "
    "fromlist=['T1ReachStudentTeacher']).T1ReachStudentTeacher"
)


@configclass
class T1ReachDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "reach_t1_distill"
    empirical_normalization = False

    policy = RslRlDistillationStudentTeacherCfg(
        class_name=_CLASS_EXPR,
        init_noise_std=1.0,
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=6,
        learning_rate=1e-3,
        gradient_length=24,
        max_grad_norm=1.0,
    )
