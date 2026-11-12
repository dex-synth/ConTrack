import mujoco
import numpy as np
import pytest

from contrack_h2r.inputs import seal_prepared, write_json
from contrack_h2r.reset_reference import reset_reference
from rsl_rl_contrack.sharpa_recipe import RECIPE_ID


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'prepared'
    (root / 'task').mkdir(parents=True)
    xml = '''<mujoco><option timestep="0.004166666666666667" integrator="implicitfast"/>
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
    (root / 'task/scene.xml').write_text(xml)
    layout = dict(hands=['right'], joint_names=['x', 'yaw', 'finger'], joint_units=['m', 'rad', 'rad'],
                  groups={'wrist_translation': [0], 'wrist_rotation': [1], 'fingers': [2]},
                  parts=['root', 'child'], part_bodies=['object', 'child'], contact_bodies={'hand': [0, 0]},
                  action_scales={'wrist_translation': 2/30, 'wrist_rotation': 6.4/30, 'fingers': .25},
                  wrist_error_scales={'translation_m': 1., 'rotation_rad': 1.})
    task = dict(scene='scene.xml', dt_s=1/240, control_ticks=8, end_tick=24, task_id='test-task',
                bundle_id='test-bundle', robot_manifest_sha256='test-robot',
                object={'parts': {'root': 'object', 'child': 'child'}},
                observation={'contacts': True, 'contact_schema': 'h2r-body-contacts-v1',
                             'quaternion': 'wxyz', 'body_velocity': 'world_at_body_origin',
                             'bodies': ['object', 'child']})
    write_json(root / 'task/task.json', task)
    m = mujoco.MjModel.from_xml_path(str(root / 'task/scene.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bids = [m.body(n).id for n in layout['part_bodies']]
    arrays = dict(time_s=np.arange(4)/30, robot_q=np.zeros((4, 3)),
             part_pos=np.tile(d.xpos[bids], (4, 1, 1)), part_quat=np.tile(d.xquat[bids], (4, 1, 1)),
             contact=np.zeros((4, 1, 2, 10), bool), contact_points=np.zeros((4, 1, 2, 10, 3)),
             qpos=np.tile(m.qpos0, (4, 1)))
    qpos, qvel, _ = reset_reference(m, layout, arrays)
    np.savez(root / 'reference.npz', **arrays, reset_qpos=qpos, reset_qvel=qvel)
    seal_prepared(root, dict(format='contrack-h2r-v2', reference_recipe=RECIPE_ID, task=task, layout=layout))
    return root
