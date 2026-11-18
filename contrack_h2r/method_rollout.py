"""Hand a checkpoint's rollout in its own training simulator to the benchmark (method-sim).

Next to its independent evaluation, h2r_bench reports how a method does in the simulator it
trained in: the same checkpoint, the evaluation initial state, no assistance and no disturbance.
This entry runs the native rollout (contrack_h2r.native_rollout: the checkpoint's MJLab/MJWarp
physics, 60 Hz control, native observations, deterministic actions) from the first reference frame
at zero velocity and writes the state after every physics step as an `h2r-method-rollout-v1`
package. MJWarp on a GPU is not fully deterministic, so several worlds start from the same state
and each is one trajectory. The benchmark imports and scores the package itself.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_warp
import numpy as np

from .controller import METHOD
from .h2r_io.method_rollout import state_names, write_method_rollout
from .inputs import load_prepared, prepared_task, read_json
from .native_rollout import rollout

WORLDS = 10
# How this rollout is run. The benchmark accepts a method rollout only as such.
CONDITIONS = {
    'initial_state': 'evaluation',  # checked below: the client's proposed robot pose, objects at frame 0
    'initial_velocity': 'zero',
    'assistance': 'none',  # the task's actuators only; reset library and thresholds stay frozen
    'disturbance': 'none',  # the recipe has no randomization or push events
    'action': 'deterministic',  # ActorCritic.act_inference
}


def export(prepared, checkpoint, output, *, num_envs=WORLDS, device='cuda:0'):
    """Write the method rollout of one checkpoint; each world is one trajectory.

    A world ends early at its native pose-break. The native rollout also takes one more action at
    the last reference frame; the task ends before it, so its states are not written, but a
    pose-break there still fails the world's verdict.
    """
    binding, reference = load_prepared(prepared)
    task = prepared_task(prepared, binding)
    summary, arrays = rollout(prepared, checkpoint, num_envs=num_envs, device=device, initial_velocity='zero',
                              physics_states=True)
    trained = read_json(Path(checkpoint).with_suffix('.json'))  # rollout() verified it against the checkpoint
    runtime = {'mjlab': version('mjlab'), 'mujoco_warp': mujoco_warp.__version__}
    if any(trained['training_physics'][name] != value for name, value in runtime.items()):
        raise ValueError(f'The checkpoint was not trained with this simulator: {runtime}')
    physics_dt, control_dt = summary['physics']['timestep_s'], summary['physics']['control_dt_s']
    substeps, steps = round(control_dt / physics_dt), round(task.duration_s / physics_dt)

    # MJLab keeps the generalized coordinates of the task model (checked field by field in TaskState).
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    names, objects = state_names(task), task.object_binding
    address = lambda joints: np.array([model.joint(name).qposadr[0] for name in joints], dtype=int)
    qpos = arrays['physics_qpos'].astype(float)  # (1 + control steps x substeps, worlds, nq)
    states = {'robot_q': qpos[..., address(names['joint_names'])],
              'object_root_poses': qpos[..., address(r['joint'] for r in objects['roots'].values())[:, None]
                                        + np.arange(7)],
              'object_q': qpos[..., address(objects['joints'].values())]}
    # The evaluation initial state: what the client proposes in start, objects at the first reference frame.
    start = task.initial_object_state
    proposed = dict(zip(binding['layout']['joint_names'], reference['robot_q'][0]))
    initial = {'robot_q': [proposed[name] for name in names['joint_names']],
               'object_root_poses': list(start['object_roots'].values()),
               'object_q': list(start['object_joints'].values())}
    for key, value in initial.items():
        if not np.allclose(states[key][0], np.reshape(value, states[key].shape[2:]), rtol=0, atol=1e-6):
            raise RuntimeError(f'The rollout did not start from the evaluation initial state: {key}')

    episodes = []
    for i, world in enumerate(summary['worlds']):
        count = min((world['end_frame'] + 1) * substeps, steps)  # physics steps of this world inside the task
        success = world['reason'] == 'reference_end'
        episodes.append({
            'states': {'time_s': np.arange(count + 1) * physics_dt,
                       **{key: value[:count + 1, i] for key, value in states.items()}},
            'verdict': {'success': success, 'reason': None if success else world['reason'],
                        'ended_at_s': count * physics_dt if count < steps else None,
                        'metrics': {'reference_coverage': world['reference_coverage'],
                                    'object_position_error_mean_m': world['object_position_error_m']['mean'],
                                    'object_position_error_max_m': world['object_position_error_m']['max'],
                                    'object_angle_error_mean_rad': world['object_angle_error_rad']['mean'],
                                    'object_angle_error_max_rad': world['object_angle_error_rad']['max']}}})
    limits = summary['termination']
    return write_method_rollout(
        output, task, method=METHOD,
        # As reported to the evaluation in start (contrack_h2r.controller).
        controller={**summary['checkpoint'], 'prepared_id': summary['prepared_id']},
        training_seed=trained['seed'],
        simulator={'name': 'mjwarp', 'version': f"{runtime['mujoco_warp']} (mjlab {runtime['mjlab']})",
                   'physics_dt_s': physics_dt, 'control_dt_s': control_dt, 'device': device, 'deterministic': False},
        conditions=CONDITIONS,
        criteria='Native rule of the training setup: the world reaches the end of the reference without a '
                 f"pose-break, i.e. no object part is ever more than {limits['threshold_m']:.3g} m or "
                 f"{limits['threshold_rad']:.3g} rad from its reference at a 60 Hz control step "
                 '(thresholds frozen at the checkpoint).',
        episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('prepared', 'checkpoint', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(export(args.prepared, args.checkpoint, args.output, device=args.device))


if __name__ == '__main__':
    main()
