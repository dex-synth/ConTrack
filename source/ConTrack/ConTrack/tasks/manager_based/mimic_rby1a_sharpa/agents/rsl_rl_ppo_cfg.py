# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rsl_rl_contrack.sharpa_recipe import ALGORITHM, MAX_ITERATIONS, NUM_STEPS_PER_ENV, POLICY

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoAlgorithmCmdpCfg(RslRlPpoAlgorithmCfg):
    cmdp_alpha = 0.9
    cmdp_lambda_lr = 5e-2
    cmdp_warmup_iters = 0
    cmdp_vg_window = 100
    cmdp_vg_star_window = 20
    cmdp_lambda_init = 0.0


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = NUM_STEPS_PER_ENV
    max_iterations = MAX_ITERATIONS
    save_interval = 1_000
    video_length = 500
    experiment_name = "Rby1aSharpa-Mimic"
    policy = RslRlPpoActorCriticCfg(**POLICY)
    algorithm = RslRlPpoAlgorithmCmdpCfg(**ALGORITHM)
