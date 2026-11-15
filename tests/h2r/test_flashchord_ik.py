import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from contrack_h2r.inputs import implementation_hashes
from contrack_h2r.retarget import VENDOR, penetration_report


def test_frozen_sources_and_manifest_are_bound():
    source = json.loads((VENDOR / 'SOURCE.json').read_text())
    hashes = implementation_hashes()
    for name, record in source['files'].items():
        raw = (VENDOR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        assert hashes['contrack_h2r/_vendor/v2d/' + name] == record['sha256']
        if 'extracts' in record:
            lines = raw.decode().splitlines()
            for node in ast.parse(raw).body:
                names = [node.name] if isinstance(node, ast.FunctionDef) else (
                    [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
                for extracted in set(names) & record['extracts'].keys():
                    body = '\n'.join(lines[node.lineno-1:node.end_lineno]) + '\n'
                    assert hashlib.sha256(body.encode()).hexdigest() == record['extracts'][extracted]


def test_penetration_diagnostic_does_not_modify_reference(prepared):
    model = mujoco.MjModel.from_xml_path(str(prepared / 'task/scene.xml'))
    scene = SimpleNamespace(model=model)
    qpos = np.tile(model.qpos0, (2, 1))
    qpos[1, model.joint('x').qposadr[0]] = -.1
    original = qpos.copy()
    report = penetration_report(scene, qpos, threshold=.005)
    np.testing.assert_array_equal(qpos, original)
    assert report['frames_above_threshold'] == [0]
    assert report['max_depth_m'] == pytest.approx(.01)


@pytest.mark.skipif(not os.environ.get('CONTRACK_IK_PARITY_ROOT'), reason='requires independent FlashCHORD outputs')
def test_full_native_preparation_parity():
    root = Path(os.environ['CONTRACK_IK_PARITY_ROOT'])
    with np.load(root / 'native-arctic/features.npz') as native, np.load(root / 'prepared-arctic/ik/features.npz') as copied:
        for side in ('left', 'right'):
            for field in ('joints', 'joints_wxyz'):
                np.testing.assert_array_equal(native[f'{side}_{field}'], copied[f'{side}_{field}'])
    with np.load(root / 'native-arctic/robot_reference.npz') as native, np.load(root / 'prepared-arctic/ik/solution.npz') as copied:
        order = [copied['joint_names'].tolist().index(n) for n in native['joint_names']]
        np.testing.assert_allclose(native['joint_q'], copied['joint_q'][:, order], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(native['time_s'], copied['time_s'])
    native = json.loads((root / 'native-arctic/checks/ik.json').read_text())
    copied = json.loads((root / 'prepared-arctic/ik/report.json').read_text())
    for side in ('left', 'right'):
        assert native[side]['iterations'] == copied[side]['iterations']
        np.testing.assert_allclose(native[side]['frame_task_errors_m'], copied[side]['frame_task_errors_m'],
                                   atol=1e-9, rtol=0)
