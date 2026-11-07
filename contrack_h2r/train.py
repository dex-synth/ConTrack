"""Train the H2R adaptation with ConTrack's original PPO/CMDP runner."""

import argparse
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch

from rsl_rl_contrack.runners import OnPolicyRunner

from .inputs import digest, implementation_hashes, write_json
from .mjwarp_env import MJWarpEnv


def training_config():
    return dict(num_steps_per_env=16, save_interval=1000, logger='tensorboard',
                obs_groups={'policy': ['policy'], 'critic': ['critic']},
                policy=dict(class_name='ActorCritic', init_noise_std=1.0,
                            actor_obs_normalization=True, critic_obs_normalization=True,
                            actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation='elu'),
                algorithm=dict(class_name='PPO', value_loss_coef=1.0, use_clipped_value_loss=True,
                               clip_param=0.2, entropy_coef=0.001, num_learning_epochs=10, num_mini_batches=16,
                               learning_rate=0.0005, schedule='adaptive', gamma=0.99, lam=0.95,
                               desired_kl=0.01, max_grad_norm=1.0, cmdp_alpha=0.9, cmdp_lambda_lr=0.05,
                               cmdp_warmup_iters=0, cmdp_vg_window=100, cmdp_vg_star_window=20, cmdp_lambda_init=0.0))


class H2RRunner(OnPolicyRunner):
    def __init__(self, env, cfg, output, seed):
        self.deployment = dict(format='contrack-h2r-deployment-v1', prepared_id=env.binding['prepared_id'],
                               layout=env.layout, task_id=env.binding['task']['task_id'],
                               train_config=deepcopy(cfg), seed=seed, training_physics=env.audit,
                               implementation_sha256=implementation_hashes())
        super().__init__(env, deepcopy(cfg), str(output), env.device)

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos.update(h2r=self.deployment, reset_library=self.env.library.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        super().save(path, infos)
        write_json(Path(path).with_suffix('.json'), {**self.deployment, 'checkpoint_sha256': digest(path)})

    def resume(self, path):
        # User-provided local checkpoints are trusted pickle inputs, never fetched automatically.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        metadata = checkpoint['infos']['h2r']
        for key in ('prepared_id', 'layout', 'train_config'):
            if metadata[key] != self.deployment[key]:
                raise ValueError(f'Checkpoint differs from this training task: {key}')
        infos = self.load(path, map_location=self.device)
        self.env.library.load_state_dict(infos['reset_library'])
        # Resume optimization and adaptive state; live simulator worlds start fresh episodes.
        self.env.reset(torch.arange(self.env.num_envs, device=self.device))
        torch.set_rng_state(infos['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in infos['cuda_rng']])
        np.random.set_state(infos['numpy_rng'])
        random.setstate(infos['python_rng'])
        self.current_learning_iteration += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--contacts-per-world', type=int, default=512)
    parser.add_argument('--constraints-per-world', type=int, default=4096)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = MJWarpEnv(args.prepared, args.num_envs, args.device, args.contacts_per_world, args.constraints_per_world)
    args.output.mkdir(parents=True, exist_ok=False)
    runner = H2RRunner(env, training_config(), args.output, args.seed)
    if args.resume:
        runner.resume(args.resume)
    write_json(args.output / 'training_physics.json', env.audit)
    write_json(args.output / 'deployment.json', runner.deployment)
    runner.learn(args.iterations)
    runner.writer.close()


if __name__ == '__main__':
    main()
