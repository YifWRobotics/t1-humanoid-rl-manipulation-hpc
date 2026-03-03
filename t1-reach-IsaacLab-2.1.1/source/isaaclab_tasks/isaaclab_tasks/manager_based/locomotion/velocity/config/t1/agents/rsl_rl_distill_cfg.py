# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg, RslRlOnPolicyRunnerCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import t1


@configclass
class T1FlatSinglePolicyDistillCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "t1_flat"
    empirical_normalization = False
    
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=6,
        learning_rate=1e-3,
        gradient_length=24,
        max_grad_norm=1.0
    )
