"""Public Sharpa PPO settings with the paper's fixed update budget."""

NUM_STEPS_PER_ENV = 16
MAX_ITERATIONS = 5_000
TRAINING_BUDGET = dict(
    default_iterations=MAX_ITERATIONS,
    source='https://arxiv.org/html/2606.03177v3#Pt0.A6',
    scope='PPO updates only; parallel environments retain the public Sharpa setting',
    upstream_default_iterations=100_000,
)
POLICY = dict(init_noise_std=1.0, actor_obs_normalization=True, critic_obs_normalization=True,
              actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation='elu')
ALGORITHM = dict(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                 entropy_coef=0.001, num_learning_epochs=10, num_mini_batches=16,
                 learning_rate=5e-4, schedule='adaptive', gamma=0.99, lam=0.95,
                 desired_kl=0.01, max_grad_norm=1.0, cmdp_alpha=0.9, cmdp_lambda_lr=5e-2,
                 cmdp_warmup_iters=0, cmdp_vg_window=100, cmdp_vg_star_window=20, cmdp_lambda_init=0.0)

# README command-line overrides, not the environment/runner class defaults.
NUM_ENVS = 6000
SAVE_INTERVAL = 200
CONTROL_HZ = 60.0
PHYSICS_HZ = 120.0
RECIPE_ID = 'contrack-public-sharpa-h2r-v2'
PROVENANCE = dict(
    name=RECIPE_ID, upstream_commit='f5efe18e4b1f986566b252265c4bf63f3b67c6c6',
    isaaclab_commit='42e61645c96bac08135566634785cdc87728d5ab',
    entry='scripts/rsl_rl/train.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0',
    configuration='mimic_rby1a_sharpa/{rby1a_sharpa_env_cfg.py,agents/rsl_rl_ppo_cfg.py}',
    readme_overrides={'num_envs': NUM_ENVS, 'save_interval': SAVE_INTERVAL},
    paper_recipe='not established', preparation='FlashCHORD IK and H2R contacts; not official HDF5 generation',
    velocity_history='native reset-only cache (upstream indexed copy does not write back)',
    empty_return_window='native ZeroDivisionError', terminal_action='included in private training only',
)
