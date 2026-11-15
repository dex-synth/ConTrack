"""Contact boundaries and full Sharpa wrist residuals in the H2R preparation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from contrack_h2r import contacts, prepare
from contrack_h2r.inputs import load_prepared
from contrack_h2r.layout import CHAINS, FINGERS, make_layout
from contrack_h2r.reference import Reference


def test_contact_distance_boundary_and_cli_default(monkeypatch):
    mesh = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    monkeypatch.setattr(contacts, 'part_meshes', lambda scene, surface_geoms: [mesh])
    joints = np.zeros((21, 3))
    for finger, chain in enumerate(CHAINS):
        joints[list(chain)] = [[finger * .2, link * .02, 0] for link in range(4)]
    distances = np.array([.0049, .005, .0051, .0079])
    vertices = np.zeros((len(distances), 1, 3))
    vertices[:, 0, 1], vertices[:, 0, 2] = .01, distances
    scene = SimpleNamespace(layout={'hands': ['right'], 'parts': ['object']})
    hand = SimpleNamespace(vertices_m=vertices, joints_m=np.tile(joints, (len(distances), 1, 1)))
    arrays = dict(time_s=np.arange(len(distances))/30, part_pos_m=np.zeros((len(distances), 1, 3)),
                  part_quat_wxyz=np.tile([1., 0, 0, 0], (len(distances), 1, 1)))
    task = SimpleNamespace(demo=SimpleNamespace(arrays=arrays, hands={'right': hand}),
                           object_binding={'surface_geoms': None})
    flags, points = contacts.reference_contacts(scene, task)
    np.testing.assert_array_equal(flags[:, 0, 0, 0], [True, True, False, False])
    assert not flags[..., 1:].any()
    np.testing.assert_allclose(points[:2, 0, 0, 0], [[0, .01, 0]]*2, atol=1e-8)
    wider, _ = contacts.reference_contacts(scene, task, threshold=.008)
    assert wider[:, 0, 0, 0].all()

    captured = {}
    monkeypatch.setattr(prepare, 'prepare', lambda **kwargs: captured.update(kwargs))
    argv = ['prepare']
    for name in ('task', 'output', 'mano-root', 'loader-python', 'ik-python'):
        argv.extend(['--' + name, '/unused'])
    monkeypatch.setattr('sys.argv', argv)
    prepare.main()
    assert captured['contact_distance'] == .005


@pytest.mark.parametrize('hands', [('left',), ('right',), ('left', 'right')])
def test_full_sharpa_layout_decodes_wrist_ranges(prepared, hands):
    names, bodies = [], []
    groups = {'wrist_translation': [], 'wrist_rotation': [], 'fingers': []}
    semantics = {'hands': {}, 'joint_groups': groups}
    for side in hands:
        fingers = [f'{side}_finger_{i}' for i in range(22)]
        translation = [f'{side}_{axis}_joint' for axis in ('x', 'y', 'z')]
        rotation = [f'{side}_{axis}_joint' for axis in ('roll', 'pitch', 'yaw')]
        names.extend(translation + rotation + fingers)
        groups['wrist_translation'].extend(translation)
        groups['wrist_rotation'].extend(rotation)
        groups['fingers'].extend(fingers)
        semantics['hands'][side] = {'wrist_translation_joints': translation, 'wrist_rotation_joints': rotation,
                                    'finger_joints': fingers}
        bodies.extend(f'{side}_{finger}_{suffix}' for finger in FINGERS
                      for suffix in ('PP', 'DP', 'elastomer', 'fingertip'))
    semantics.update(joint_names=names[::-1], body_names=bodies,
                     joint_units=['m' if n in groups['wrist_translation'] else 'rad' for n in names[::-1]])
    _, arrays = load_prepared(prepared)
    # The robot is judged by its name, hands and semantics; the object comes from the demo.
    task = SimpleNamespace(robot=SimpleNamespace(name='sharpa_wave', hands=hands, semantics=semantics),
                           demo=SimpleNamespace(part_order=('root', 'child')),
                           object_binding={'parts': {'child': 'object_child', 'root': 'object_root'}})
    layout = make_layout(task)
    assert layout['hands'] == list(hands) and layout['joint_names'] == names
    assert layout['part_bodies'] == ['object_root', 'object_child']
    arrays['robot_q'] = np.tile(np.arange(len(names)) * .001, (len(arrays['time_s']), 1))
    reference = Reference(layout, arrays, 'cpu', 1/30)
    actions = torch.tensor([-2., -1., 0., 1., 2.])[:, None].expand(-1, len(names))
    targets = reference.targets(actions, torch.zeros(5))
    residuals = targets - torch.tensor(arrays['robot_q'][0], dtype=torch.float32)
    for side in hands:
        for axes, scale in [(('x', 'y', 'z'), 2/30), (('roll', 'pitch', 'yaw'), 6.4/30)]:
            for axis in axes:
                column = names.index(f'{side}_{axis}_joint')
                torch.testing.assert_close(residuals[:, column], torch.tensor([-scale, -scale, 0., scale, scale]))
        column = names.index(f'{side}_finger_0')
        torch.testing.assert_close(residuals[:, column], torch.tensor([-.25, -.25, 0., .25, .25]))
