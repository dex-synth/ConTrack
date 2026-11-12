"""Train the H2R adaptation with ConTrack's original PPO/CMDP runner."""

import argparse
from copy import deepcopy
from pathlib import Path
import random
import time

import numpy as np
import torch

from rsl_rl_contrack.runners import OnPolicyRunner
from rsl_rl_contrack.sharpa_recipe import (
    ALGORITHM, CONTROL_HZ, MAX_ITERATIONS, NUM_ENVS, NUM_STEPS_PER_ENV,
    PHYSICS_HZ, POLICY, PROVENANCE, SAVE_INTERVAL, TRAINING_BUDGET,
)

from .inputs import DEPLOYMENT_FORMAT, digest, implementation_hashes, write_json
from .mjlab_env import make_env


def training_config():
    return dict(num_steps_per_env=NUM_STEPS_PER_ENV, save_interval=SAVE_INTERVAL, logger='tensorboard',
                obs_groups={'policy': ['policy'], 'critic': ['critic']},
                policy=dict(class_name='ActorCritic', **deepcopy(POLICY)),
                algorithm=dict(class_name='PPO', **deepcopy(ALGORITHM)))


class H2RRunner(OnPolicyRunner):
    def __init__(self, env, cfg, output, seed):
        self.task = env.unwrapped.task_state
        self.deployment = dict(format=DEPLOYMENT_FORMAT, recipe=PROVENANCE,
                               prepared_id=self.task.binding['prepared_id'],
                               layout=self.task.layout, task_id=self.task.binding['task']['task_id'],
                               train_config=deepcopy(cfg), seed=seed, training_physics=self.task.audit,
                               implementation_sha256=implementation_hashes())
        super().__init__(env, deepcopy(cfg), str(output), env.device)

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos.update(h2r=self.deployment, reset_library=self.task.library.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        super().save(path, infos)
        write_json(Path(path).with_suffix('.json'), {**self.deployment, 'checkpoint_sha256': digest(path)})

    def resume(self, path):
        # User-provided local checkpoints are trusted pickle inputs, never fetched automatically.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        metadata = checkpoint['infos']['h2r']
        for key in ('format', 'recipe', 'prepared_id', 'layout', 'train_config'):
            if metadata.get(key) != self.deployment[key]:
                raise ValueError(f'Checkpoint differs from this training task: {key}')
        for key in ('environment', 'timestep_s', 'control_ticks', 'num_envs'):
            if metadata['training_physics'].get(key) != self.task.audit[key]:
                raise ValueError(f'Checkpoint training setting differs: {key}')
        infos = self.load(path, map_location=self.device)
        self.task.library.load_state_dict(infos['reset_library'])
        # Resume optimization and adaptive state; live simulator worlds start fresh episodes.
        self.env.reset()
        torch.set_rng_state(infos['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in infos['cuda_rng']])
        np.random.set_state(infos['numpy_rng'])
        random.setstate(infos['python_rng'])
        self.current_learning_iteration += 1

    def log(self, locs, *args, **kwargs):
        # Original runner diagnostics; the official MJLab wrapper owns the VecEnv interface.
        for name in ('it_obj_pos_err_sum', 'it_obj_rot_err_sum', 'it_step_count'):
            setattr(self.env, name, getattr(self.task, name))
        super().log(locs, *args, **kwargs)
        self.task.it_step_count = 0


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS)
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS)
    parser.add_argument('--physics-hz', type=float, default=PHYSICS_HZ)
    parser.add_argument('--control-hz', type=float, default=CONTROL_HZ, choices=[CONTROL_HZ])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--contacts-per-world', type=int, default=512)
    parser.add_argument('--constraints-per-world', type=int, default=4096)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error('--iterations and --num-envs must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    # Match the selected native training entry's numerical settings.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = make_env(args.prepared, args.num_envs, args.device, args.contacts_per_world, args.constraints_per_world,
                   physics_hz=args.physics_hz, control_hz=args.control_hz, seed=args.seed)
    environment_s = time.perf_counter() - started
    runner = H2RRunner(env, training_config(), args.output, args.seed)
    if args.resume:
        runner.resume(args.resume)
    write_json(args.output / 'training_physics.json', runner.task.audit)
    write_json(args.output / 'deployment.json', runner.deployment)
    write_json(args.output / 'run_config.json', {
        'recipe': PROVENANCE, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'budget_override': args.iterations != MAX_ITERATIONS or args.num_envs != NUM_ENVS,
        'budget_provenance': TRAINING_BUDGET,
        'transitions_per_update': args.num_envs * NUM_STEPS_PER_ENV,
        'init_at_random_ep_len': True, 'logger_override': 'tensorboard',
    })
    initialization_s = time.perf_counter() - started
    started = time.perf_counter()
    completed = False
    try:
        runner.learn(args.iterations, init_at_random_ep_len=True)
        completed = True
    finally:
        torch.cuda.synchronize(env.device)
        write_json(args.output / 'timing.json', {
            'environment_initialization_s': environment_s, 'initialization_s': initialization_s,
            'training_s': time.perf_counter() - started, 'completed': completed,
            'completed_updates': args.iterations if completed else None,
            'optimizer_update_attempts': runner.alg.num_updates,
            'control_steps_per_world': env.unwrapped.common_step_counter,
            'physics_steps_per_world': env.unwrapped._sim_step_counter,
            'control_updates_per_task_second': 1 / env.unwrapped.step_dt,
            'physics_steps_per_task_second': 1 / env.unwrapped.physics_dt,
        })
        if runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == '__main__':
    main()
