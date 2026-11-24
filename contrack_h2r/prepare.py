"""Prepare one sealed benchmark task for ConTrack: python -m contrack_h2r.prepare."""

import argparse
from pathlib import Path
import shutil
import subprocess

from h2r_io import export_task, load_task
import mujoco
import numpy as np

from .contacts import DEFAULT_CONTACT_DISTANCE_M, reference_contacts
from .inputs import PREPARED_FORMAT, implementation_hashes, seal_prepared, write_json
from .layout import make_layout
from .mjwarp_scene import Scene, options, require_runtime_mujoco
from .retarget import PENETRATION_REPORT_M, RECIPE, penetration_report, retarget
from .reset_reference import reset_reference
from rsl_rl_contrack.sharpa_recipe import RECIPE_ID


def prepare(task, output, *, mano_root, loader_python, ik_python,
            contact_distance=DEFAULT_CONTACT_DISTANCE_M):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(contact_distance) or contact_distance <= 0:
        raise ValueError('Invalid preparation parameters')
    source = load_task(task)
    require_runtime_mujoco(source)
    if any(output.is_relative_to(p) for p in (source.path, source.demo.path, source.robot.path)):
        raise ValueError('Output must be outside sealed inputs')
    output.mkdir(parents=True)
    try:
        # The preparation keeps a standalone copy of its task (same task_id) and works from it.
        task = load_task(export_task(source, output / 'task'))
        demo = task.demo
        layout = make_layout(task)
        scene = Scene(task.scene_xml, layout)
        qpos, fk_error = scene.object_qpos(demo)
        print('Solving the frozen FlashCHORD Sharpa IK...', flush=True)
        qpos, errors = retarget(scene, task, qpos, output=output / 'ik',
                                mano_root=mano_root, loader_python=loader_python, ik_python=ik_python)
        print('Generating part-local skin contacts and penetration diagnostics...', flush=True)
        contact, points = reference_contacts(scene, task, contact_distance)
        reference_arrays = dict(time_s=demo.time_s, robot_q=qpos[:, scene.qids], qpos=qpos,
                                part_pos=demo.arrays['part_pos_m'], part_quat=demo.arrays['part_quat_wxyz'],
                                contact=contact, contact_points=points)
        reset_qpos, reset_qvel, reset_report = reset_reference(scene.model, layout, reference_arrays)
        write_json(output / 'ik/penetration.json', penetration_report(scene, qpos))
        data = mujoco.MjData(scene.model)
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(scene.model, data)
        collisions = [{'geom1': scene.model.geom(c.geom1).name, 'geom2': scene.model.geom(c.geom2).name,
                       'depth_m': float(-c.dist)} for c in data.contact if -c.dist > PENETRATION_REPORT_M]
        np.savez_compressed(output / 'reference.npz', **reference_arrays,
                            reset_qpos=reset_qpos, reset_qvel=reset_qvel)
        write_json(output / 'prepare_report.json', {'recipe': RECIPE['name'],
                   'ik': errors, 'reset_reference': reset_report, 'object_fk_max_error_m': fk_error,
                   'initial_collisions': collisions, 'initial_collision_report_threshold_m': PENETRATION_REPORT_M,
                   'contact_positive_slots': int(contact.sum()),
                   'contact_recipe': 'nearest MANO bone skin segmentation; mean surface points within threshold; 15-to-10 OR/mean'})
        write_json(output / 'training_physics.json', {'source': 'unmodified common task MJCF',
                   'mujoco_version': mujoco.__version__, 'options': options(scene.model),
                   'compiled': scene.audit()})
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        seal_prepared(output, dict(format=PREPARED_FORMAT, reference_recipe=RECIPE_ID, source_revision=revision,
                      implementation_sha256=implementation_hashes(),
                      task_id=task.task_id, demo_id=demo.demo_id, robot_id=task.robot.robot_id,
                      dataset=task.dataset, profile=task.profile, layout=layout,
                      preparation={'ik': RECIPE, 'contact_distance_m': contact_distance}))
    except Exception:
        shutil.rmtree(output)
        raise
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'output', 'mano-root', 'loader-python', 'ik-python'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--contact-distance', type=float, default=DEFAULT_CONTACT_DISTANCE_M)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
