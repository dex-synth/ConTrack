import json

from h2r_io import demo as demo_io, robot as robot_io, task as task_io
import mujoco
import numpy as np
import pytest

from contrack_h2r.inputs import PREPARED_FORMAT, seal_prepared
from contrack_h2r.reset_reference import reset_reference
from rsl_rl_contrack.sharpa_recipe import RECIPE_ID

# A three-joint hand and a hinged two-part object, at the tasks' nominal 300 Hz step.
SCENE = '''<mujoco><option timestep="0.0033333333333333335" integrator="implicitfast"/>
<worldbody><geom type="plane" size="1 1 .01"/>
  <body name="hand" pos="0 0 .5"><joint name="x" type="slide" axis="1 0 0"/>
    <joint name="yaw" axis="0 0 1"/><joint name="finger" axis="0 1 0"/>
    <geom name="hand_geom" type="sphere" size=".05" mass="1"/></body>
  <body name="object" pos=".09 0 .5"><freejoint name="object_free"/>
    <geom type="sphere" size=".05" mass="1"/>
    <body name="child" pos="0 0 .15"><joint name="hinge" axis="0 1 0"/>
      <geom type="sphere" size=".03" mass=".1"/></body></body>
</worldbody><actuator><position joint="x" kp="100" kv="10"/>
<position joint="yaw" kp="10" kv="1"/><position joint="finger" kp="10" kv="1"/></actuator></mujoco>'''
LAYOUT = dict(hands=['right'], joint_names=['x', 'yaw', 'finger'], joint_units=['m', 'rad', 'rad'],
              groups={'wrist_translation': [0], 'wrist_rotation': [1], 'fingers': [2]},
              parts=['root', 'child'], part_bodies=['object', 'child'], contact_bodies={'hand': [0, 0]},
              action_scales={'wrist_translation': 2/30, 'wrist_rotation': 6.4/30, 'fingers': .25},
              wrist_error_scales={'translation_m': 1., 'rotation_rad': 1.})


def write_task(root, reference_hz=30., frames=4):
    """Seal a standalone toy task with h2r_io; returns the loaded Task."""
    demo, robot = root / 'demo', root / 'robot'
    demo.mkdir(parents=True)
    robot.mkdir()
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = [model.body(name).id for name in LAYOUT['part_bodies']]
    unit = np.array([1., 0, 0, 0])
    time = np.arange(frames) / reference_hz
    np.savez(demo / 'reference.npz', time_s=time, source_time_s=time, joint_hinge_rad=np.zeros(frames),
             part_pos_m=np.tile(data.xpos[bodies], (frames, 1, 1)),
             part_quat_wxyz=np.tile(data.xquat[bodies], (frames, 1, 1)),
             right_joints_m=np.zeros((frames, 21, 3)), right_vertices_m=np.zeros((frames, 778, 3)),
             right_faces=np.zeros((2, 3), np.int32), right_mano_quat=np.tile(unit, (frames, 16, 1)),
             right_mano_trans=np.zeros((frames, 3)), right_mano_shape=np.zeros(10))
    (demo / 'reference.json').write_text(json.dumps({
        'dataset': 'synthetic', 'profile': 'clip', 'task': {'hand_sides': ['right']},
        'reference_hz': reference_hz, 'duration_s': float(time[-1]),
        'mano': {'convention': 'h2r-mano-v1'}, 'quaternion_convention': 'wxyz-unit-continuous-v1',
        'object_geometry': {'part_order': LAYOUT['parts']}}))
    (demo / 'scene.json').write_text(json.dumps({
        'interaction_objects': {'box': {
            'root': {'part': 'root', 'body': 'object', 'joint': 'object_free', 'motion': 'free'},
            'parts': {'root': 'object', 'child': 'child'},
            'joints': {'hinge': {'type': 'hinge', 'parent': 'root', 'child': 'child',
                                 'mjcf_name': 'hinge', 'reference_key': 'joint_hinge_rad'}}}},
        'scoring_parts': ['root'], 'environment': {'table': {}}, 'settings': {'timestep_s': 1 / 300}}))
    (demo / 'scene.xml').write_text('<mujoco/>')
    demo_io.seal_demo(demo, input_key='test')
    for name in ('robot.xml', 'robot.urdf'):
        (robot / name).write_text('<robot/>')
    (robot / 'compiled.json').write_text('{}')
    (robot / 'semantics.json').write_text(json.dumps({
        'joint_names': LAYOUT['joint_names'], 'joint_units': LAYOUT['joint_units'], 'body_names': ['hand'],
        'joint_groups': {'wrist_translation': ['x'], 'wrist_rotation': ['yaw'], 'fingers': ['finger']},
        'hands': {'right': {'wrist_body': 'hand', 'wrist_translation_joints': ['x'],
                            'wrist_rotation_joints': ['yaw'], 'finger_joints': ['finger']}}}))
    robot_io.seal_robot(robot, 'toy_hand', ['right'], mjcf='robot.xml', urdf='robot.urdf',
                        hand_urdfs={'right': 'robot.urdf'})
    (root / 'scene.xml').write_text(SCENE)
    (root / 'compiled.json').write_text('{}')
    task_io.seal_task(root, 'demo', 'robot')
    return task_io.load_task(root)


def write_prepared(root, **task_options):
    """A prepared directory around a sealed toy task, with the toy layout written by hand."""
    task = write_task(root / 'task', **task_options)
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    frames = len(task.demo.time_s)
    arrays = dict(time_s=task.demo.time_s, robot_q=np.zeros((frames, 3)),
                  part_pos=task.demo.arrays['part_pos_m'], part_quat=task.demo.arrays['part_quat_wxyz'],
                  contact=np.zeros((frames, 1, 2, 10), bool), contact_points=np.zeros((frames, 1, 2, 10, 3)),
                  qpos=np.tile(model.qpos0, (frames, 1)))
    qpos, qvel, _ = reset_reference(model, LAYOUT, arrays)
    np.savez(root / 'reference.npz', **arrays, reset_qpos=qpos, reset_qvel=qvel)
    seal_prepared(root, dict(format=PREPARED_FORMAT, reference_recipe=RECIPE_ID, layout=LAYOUT, task_id=task.task_id,
                             demo_id=task.demo.demo_id, robot_id=task.robot.robot_id))
    return root


@pytest.fixture
def prepared(tmp_path):
    return write_prepared(tmp_path / 'prepared')
