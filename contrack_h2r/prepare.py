"""Prepare a sealed benchmark bundle for ConTrack: python -m contrack_h2r.prepare."""

import argparse
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np

from .contacts import reference_contacts
from .inputs import implementation_hashes, load_inputs, seal_prepared, write_json
from .layout import make_layout
from .mjwarp_scene import Scene
from .retarget import retarget


def prepare(bundle, robot, task, output, *, ik_iterations=40, contact_distance=0.008):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(Path(p).resolve()) for p in (bundle, robot, task)):
        raise ValueError('Output must be outside sealed inputs')
    manifest, spec, reference, arrays = load_inputs(bundle, robot, task)
    if ik_iterations < 1 or not np.isfinite(contact_distance) or contact_distance <= 0:
        raise ValueError('Invalid preparation parameters')
    layout = make_layout(spec, reference['object_geometry']['part_order'])
    scene = Scene(Path(task) / spec['scene'], spec, layout)
    qpos, fk_error = scene.object_qpos(arrays)
    print('Solving H2R Sharpa IK...', flush=True)
    qpos, errors = retarget(scene, arrays, qpos, ik_iterations)
    print('Generating part-local skin contacts...', flush=True)
    contact, points = reference_contacts(scene, arrays, contact_distance)
    qvel = np.empty((len(qpos), scene.model.nv))
    for i in range(len(qpos)-1):
        mujoco.mj_differentiatePos(scene.model, qvel[i], arrays['time_s'][i+1]-arrays['time_s'][i], qpos[i], qpos[i+1])
    qvel[-1] = qvel[-2]
    data = mujoco.MjData(scene.model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(scene.model, data)
    collisions = [{'geom1': scene.model.geom(c.geom1).name, 'geom2': scene.model.geom(c.geom2).name,
                   'depth_m': float(-c.dist)} for c in data.contact if -c.dist > spec['initialization']['max_penetration_m']]
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(task, output / 'task')
        np.savez_compressed(output / 'reference.npz', time_s=arrays['time_s'],
                            robot_q=qpos[:, scene.qids], qpos=qpos, qvel=qvel,
                            part_pos=arrays['part_pos_m'], part_quat=arrays['part_quat_wxyz'],
                            contact=contact, contact_points=points)
        write_json(output / 'prepare_report.json', {'recipe': 'contrack-h2r-geometric-ik-v1',
                   'ik': errors, 'object_fk_max_error_m': fk_error,
                   'initialization_accepted': not collisions, 'initial_collisions': collisions,
                   'contact_positive_slots': int(contact.sum()),
                   'contact_recipe': 'nearest MANO bone skin segmentation; mean surface points within threshold; 15-to-10 OR/mean'})
        write_json(output / 'training_physics.json', {'source': 'unmodified common task MJCF',
                   'mujoco_version': mujoco.__version__, 'compiled': scene.audit()})
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        seal_prepared(output, dict(format='contrack-h2r-v1', source_revision=revision,
                      implementation_sha256=implementation_hashes(),
                      bundle_id=manifest['bundle_id'], scene_id=manifest['scene_id'],
                      robot_manifest_sha256=spec['robot_manifest_sha256'], task=spec, layout=layout,
                      preparation={'ik_iterations': ik_iterations, 'contact_distance_m': contact_distance,
                                   'legacy_hand_declaration': 'robot task hands' if 'task' not in reference else None}))
    except Exception:
        shutil.rmtree(output)
        raise
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'robot', 'task', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--ik-iterations', type=int, default=40)
    parser.add_argument('--contact-distance', type=float, default=0.008)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
