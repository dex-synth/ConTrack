"""Run the frozen FlashCHORD IK in isolated MANO and Pinocchio environments."""

import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from .inputs import check_files, read_json, write_json


VENDOR = Path(__file__).parent / '_vendor' / 'v2d'
# Method diagnostic only: the evaluation applies its own initial-penetration threshold.
PENETRATION_REPORT_M = 0.02
RECIPE = dict(name='flashchord-h2r-sharpa-ik-v1', solver='daqp', max_iter=100,
              frequency=200.0, frame_tasks_converged_threshold=1e-6,
              use_relative_frames=False, source_to_robot_scale=1.0)


def run_worker(python, module, *arguments):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(VENDOR.resolve()), str(VENDOR.parents[2].resolve())))
    env['PYTHONNOUSERSITE'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    subprocess.run([str(python), '-m', module, *map(str, arguments)], env=env, check=True)


def retarget(scene, task, qpos, *, output, mano_root, loader_python, ik_python):
    """Both workers read the task through the vendored loader, in their own environments."""
    source = read_json(VENDOR / 'SOURCE.json')
    check_files(VENDOR, {name: item['sha256'] for name, item in source['files'].items()})
    output = Path(output)
    output.mkdir()
    write_json(output / 'source.json', source)
    write_json(output / 'recipe.json', RECIPE)
    write_json(output / 'layout.json', scene.layout)
    run_worker(loader_python, 'contrack_h2r.mano_features', '--task', task.path,
               '--mano-root', mano_root, '--output', output)
    run_worker(ik_python, 'contrack_h2r.ik_worker', '--task', task.path, '--output', output)
    with np.load(output / 'solution.npz', allow_pickle=False) as data:
        names = data['joint_names'].tolist()
        if len(names) != len(set(names)) or set(names) != set(scene.layout['joint_names']):
            raise ValueError('IK output differs from the complete controlled joint set')
        if not np.array_equal(data['time_s'], task.demo.time_s):
            raise ValueError('IK changed the reference clock')
        q = data['joint_q'][:, [names.index(n) for n in scene.layout['joint_names']]]
    if (q.shape != (len(qpos), len(scene.qids)) or not np.isfinite(q).all()
            or np.any(q < scene.limits[:, 0] - 1e-5) or np.any(q > scene.limits[:, 1] + 1e-5)):
        raise ValueError('Nonfinite, out-of-range or malformed IK solution')
    qpos[:, scene.qids] = q
    return qpos, read_json(output / 'report.json')


def penetration_report(scene, qpos, threshold=PENETRATION_REPORT_M):
    """Observe the resulting trajectory without changing any IK solution."""
    m, d = scene.model, mujoco.MjData(scene.model)
    depths, worst = [], None
    for frame, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        deepest = max(d.contact, key=lambda c: -c.dist, default=None)
        depth = max(0., -float(deepest.dist)) if deepest is not None else 0.
        depths.append(depth)
        if depth > 0 and (worst is None or depth > worst['depth_m']):
            worst = dict(frame=frame, depth_m=depth,
                         geom1=m.geom(deepest.geom1).name, geom2=m.geom(deepest.geom2).name)
    return dict(threshold_m=threshold, max_depth_m=max(depths),
                frames_above_threshold=[i for i, depth in enumerate(depths) if depth > threshold],
                depth_per_frame_m=depths, worst=worst)
