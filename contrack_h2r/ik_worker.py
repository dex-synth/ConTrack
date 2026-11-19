"""Run unchanged V2D IK and convert its free wrist to common named joints."""

import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

from h2r_io import load_task
import mujoco
import numpy as np

from flash_chord.h2r.assets import numbers, portable_xml
from flash_chord.h2r.checks import check_ik_model
from flash_chord.h2r.inputs import DIGITS
from flash_chord.h2r.wrist import wrist_coordinates
from robotic_grounding.retarget import SHARPA_WAVE_XMLS_DIR
from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
from robotic_grounding.retarget.retarget_utils import wrist_pose_from_mano_joint0


def prepare_models(package, output):
    """FlashCHORD's common-hand subtree and native-site construction."""
    robot = portable_xml(package.mjcf, output / 'robot.xml')
    for actuator in robot.findall('actuator/*'):
        joint = package.compiled['joints'][actuator.attrib['joint']]
        actuator.attrib.pop('dampratio', None)
        actuator.attrib.pop('inheritrange', None)
        actuator.set('kv', numbers([joint['kv']]))
        actuator.set('kp', numbers([joint['kp']]))
        for field in ('ctrl', 'force'):
            actuator.set(field + 'limited', str(joint[field + '_limited']).lower())
            actuator.set(field + 'range', numbers(joint[field + 'range']))
    ET.ElementTree(robot).write(output / 'robot.xml')
    model = mujoco.MjModel.from_xml_path(str(output / 'robot.xml'))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    for side, hand in package.semantics['hands'].items():
        ik = copy.deepcopy(robot)
        palm = copy.deepcopy(ik.find(f".//body[@name='{hand['wrist_body']}']"))
        palm.attrib.pop('pos', None)
        palm.attrib.pop('quat', None)
        for joint in list(palm.findall('joint')):
            palm.remove(joint)
        world = ik.find('worldbody')
        world.clear()
        world.append(palm)
        for tag in ('contact', 'actuator', 'keyframe'):
            for node in ik.findall(tag):
                ik.remove(node)
        defaults = ik.find('default')
        ik.remove(defaults)
        ik.insert(1, defaults)
        template = SHARPA_WAVE_XMLS_DIR / f'{side}_sharpawave.xml'
        for body in ET.parse(template).findall('.//body'):
            target = ik.find(f".//body[@name='{body.attrib['name']}']")
            for site in body.findall('site'):
                target.append(copy.deepcopy(site))
        for digit in DIGITS:
            distal = state.body(f'{side}_{digit}_DP')
            tip = state.body(f'{side}_{digit}_fingertip')
            local = distal.xmat.reshape(3, 3).T @ (tip.xpos - distal.xpos)
            ik.find(f".//site[@name='{side}_{digit}_tip_site']").set('pos', numbers(local))
        ET.ElementTree(ik).write(output / f'ik_{side}.xml')
    return model, state


def solve(task, output):
    package = load_task(task).robot
    layout = json.loads((output / 'layout.json').read_text())
    recipe = json.loads((output / 'recipe.json').read_text())
    with np.load(output / 'features.npz', allow_pickle=False) as data:
        features = dict(data)
    model, state = prepare_models(package, output)
    joint_q = np.zeros((len(features['time_s']), len(layout['joint_names'])))
    report = {'python': platform.python_version(), 'recipe': recipe,
              'versions': {n: version(n) for n in ('pin', 'pin-pink', 'daqp', 'qpsolvers', 'numpy', 'scipy', 'mujoco')}}
    for side in layout['hands']:
        solver = SharpaHandKinematics(side=side, robot_asset_path=str(output / f'ik_{side}.xml'),
                                      source_model='mano', **{k: recipe[k] for k in
                                      ('solver', 'max_iter', 'frequency', 'use_relative_frames',
                                       'frame_tasks_converged_threshold')})
        fingers = list(solver.robot_finger_joint_names.values())
        hand = package.semantics['hands'][side]
        wrist_names = hand['wrist_translation_joints'] + hand['wrist_rotation_joints']
        wrist_ids = [model.joint(name).qposadr[0] for name in wrist_names]
        state.qpos[:] = 0
        mujoco.mj_forward(model, state)
        base = state.body(hand['wrist_body']).xpos.copy()
        fk_error = check_ik_model(solver, model, state, side, fingers, wrist_ids, base)
        q, previous = None, None
        errors, iterations, tip_errors = [], [], []
        for t in range(len(joint_q)):
            if q is None:
                pos, quat = wrist_pose_from_mano_joint0(
                    features[f'{side}_joints'][t, 0], features[f'{side}_joints_wxyz'][t, 0],
                    link_to_site_quat_xyzw=np.array([-0.5, 0.5, 0.5, 0.5]))
                q = solver.robot.q0.copy()
                q[:3], q[3:7] = pos, quat
            try:
                result = solver.compute(features[f'{side}_joints'][t], features[f'{side}_joints_wxyz'][t],
                                        source_to_robot_scale=recipe['source_to_robot_scale'], qpos=q)
            except Exception as error:
                raise RuntimeError(f'IK failed for {side}, frame {t}, time {features["time_s"][t]}') from error
            q = result['q']
            wrist = wrist_coordinates(q[:3], q[[6, 3, 4, 5]], base, previous)
            previous = wrist[3:]
            for name, value in zip(wrist_names + fingers, np.r_[wrist, q[7:]]):
                joint_q[t, layout['joint_names'].index(name)] = value
                state.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(model, state)
            palm = state.body(hand['wrist_body'])
            np.testing.assert_allclose(palm.xpos, q[:3], atol=2e-6, rtol=0)
            if abs(np.dot(palm.xquat, q[[6, 3, 4, 5]])) < 1 - 1e-9:
                raise ValueError(f'Wrist FK mismatch at {side} frame {t}')
            targets = features[f'{side}_joints'][t, [4, 8, 12, 16, 20]]
            tips = np.array([state.body(f'{side}_{digit}_fingertip').xpos for digit in DIGITS])
            tip_errors.append(float(np.linalg.norm(tips - targets, axis=-1).mean()))
            errors.append(result['frame_task_errors'])
            iterations.append(result['num_optimization_iterations'])
            if t % 100 == 0:
                print(f'FlashCHORD IK {side}: {t + 1}/{len(joint_q)} frames', flush=True)
        report[side] = dict(fk_max_error_m=fk_error, frame_task_names=list(solver.frame_tasks),
                            frame_task_errors_m=errors, iterations=iterations,
                            mean_tip_error_m=float(np.mean(tip_errors)), max_tip_error_m=max(tip_errors))
    ranges = np.array([model.joint(n).range for n in layout['joint_names']])
    if (not np.isfinite(joint_q).all() or np.any(joint_q < ranges[:, 0] - 1e-5)
            or np.any(joint_q > ranges[:, 1] + 1e-5)):
        raise ValueError('Invalid IK joint targets')
    np.savez_compressed(output / 'solution.npz', joint_q=joint_q,
                        joint_names=layout['joint_names'], time_s=features['time_s'])
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    solve(**vars(parser.parse_args()))
