"""Deterministic closed-loop rollout of a checkpoint in its own training setup.

This diagnoses training quality only. It uses the checkpoint's MJLab/MJWarp
physics and clocks, the native observations and pose-break termination with
the reset library and thresholds frozen at the checkpoint, and starts every
world at the first reference frame. It is not a benchmark result: the common
evaluation (other physics rate, zero initial velocity, bench metrics) stays in
h2r_bench.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inputs import load_prepared, write_json
from .mjlab_env import make_env
from .policy import Checkpoint


def rollout(prepared, checkpoint, *, num_envs=1, device='cuda:0', initial_velocity='reference',
            contacts_per_world=512, constraints_per_world=4096, physics_states=False):
    """Summary and per-control-step arrays of the rollout.

    `physics_states` adds `physics_qpos` (1 + control steps x substeps, worlds, nq): the initial
    state, then qpos after every physics step. A finished world restarts and its later rows are not
    part of its rollout.
    """
    binding, _ = load_prepared(prepared)
    model = Checkpoint(checkpoint, binding, device)
    physics = model.metadata['training_physics']
    env = make_env(prepared, num_envs, device, contacts_per_world, constraints_per_world,
                   physics_hz=1 / model.timestep_s, seed=model.metadata['seed'])
    try:
        task = env.unwrapped.task_state
        if task.audit['timestep_s'] != physics['timestep_s'] or task.audit['control_dt_s'] != physics['control_dt_s']:
            raise ValueError('Rollout clocks differ from the checkpoint training clocks')
        task.library.load_state_dict(model.saved['infos']['reset_library'])
        task.rollout = {'start': 0, 'velocity': initial_velocity}
        env.unwrapped.reset()
        if physics_states:
            task.physics_qpos = [task.tensor(task.data, 'qpos').clone()]
        observations = env.get_observations()
        policy = model.policy(observations)
        length = len(task.clock)
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        end = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        reason = [None] * num_envs
        steps = []
        for _ in range(length):
            with torch.inference_mode():
                actions = policy.act_inference(observations)
            observations, _, dones, _ = env.step(actions)
            step = task.last_step
            if not torch.equal(step['frame'][active], torch.full_like(step['frame'][active], len(steps))):
                raise RuntimeError('Rollout lost the fixed reference clock')
            steps.append({k: v.detach().cpu().numpy() for k, v in step.items()})
            for i in (active & dones.bool()).nonzero().flatten().tolist():
                end[i] = len(steps) - 1
                reason[i] = 'pose_break' if bool(step['failed'][i]) else 'reference_end'
            active &= ~dones.bool()
            if not active.any():
                break
        arrays = {k: np.stack([s[k] for s in steps]) for k in steps[0]}
        if physics_states:
            arrays['physics_qpos'] = torch.stack(task.physics_qpos).cpu().numpy()
        thresholds = [float(v) for v in task.library.thresholds()]
    finally:
        env.close()
    dt = model.control_dt
    worlds = []
    for i in range(num_envs):
        n = int(end[i]) + 1 if end[i] >= 0 else len(steps)
        position, angle = arrays['position'][:n, i], arrays['angle'][:n, i]
        worlds.append(dict(
            end_frame=n - 1, end_time_s=(n - 1) * dt, reason=reason[i] or 'unfinished',
            reference_coverage=(n - 1) / (length - 1),
            object_position_error_m={'mean': float(position.mean()), 'max': float(position.max())},
            object_angle_error_rad={'mean': float(angle.mean()), 'max': float(angle.max())}))
    summary = dict(
        format='contrack-h2r-native-rollout-v1', scope='training-setup diagnostic; not a benchmark result',
        checkpoint=model.identity(), prepared_id=binding['prepared_id'], task_id=binding['task_id'],
        physics={'backend': physics['backend'], 'timestep_s': model.timestep_s, 'control_dt_s': dt},
        start_frame=0, initial_velocity=initial_velocity, num_envs=num_envs,
        policy='ActorCritic.act_inference (deterministic)',
        termination={'rule': 'native pose_break: any part position > threshold_m or angle > threshold_rad',
                     'threshold_m': thresholds[0], 'threshold_rad': thresholds[1],
                     'source': 'checkpoint reset library, frozen'},
        reference_frames=length, worlds=worlds)
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--initial-velocity', choices=('reference', 'zero'), default='reference',
                        help='reference: native reset state at frame 0; zero: the benchmark initialization')
    parser.add_argument('--contacts-per-world', type=int, default=512)
    parser.add_argument('--constraints-per-world', type=int, default=4096)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error('--num-envs must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    summary, arrays = rollout(args.prepared, args.checkpoint, num_envs=args.num_envs, device=args.device,
                              initial_velocity=args.initial_velocity, contacts_per_world=args.contacts_per_world,
                              constraints_per_world=args.constraints_per_world)
    np.savez_compressed(args.output / 'rollout.npz', **arrays)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary['worlds'], indent=2))


if __name__ == '__main__':
    main()
