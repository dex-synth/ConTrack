"""Small, strict readers for sealed benchmark inputs and method artifacts."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from rsl_rl_contrack.sharpa_recipe import RECIPE_ID

# v3: per-step body refresh runs kinematics/com_pos/com_vel only (v2 ran a second full forward).
DEPLOYMENT_FORMAT = 'contrack-h2r-deployment-v3'
STATE_REFRESH = 'kinematics-com_pos-com_vel-v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    files = {p for package in ('contrack_h2r', 'rsl_rl_contrack') for p in (root / package).rglob('*.py')}
    files.update(p for p in (root / 'contrack_h2r/_vendor').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    files.update((root / 'h2r').glob('*requirements.lock'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def local_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Dependency escapes artifact: {name}')
    return path


def check_files(root, files):
    for name, expected in files.items():
        if digest(local_file(root, name)) != expected:
            raise ValueError(f'Changed dependency: {name}')


def check_xml(root, scene, files):
    pending, seen = [scene], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in files:
            raise ValueError(f'Unsealed MJCF: {name}')
        for node in ET.parse(local_file(root, name)).iter():
            if node.tag == 'compiler' and any(node.get(k) not in (None, '', '.') for k in ('meshdir', 'texturedir', 'assetdir')):
                raise ValueError('Expected materialized MJCF resource paths, without compiler directories')
            if 'file' in node.attrib:
                path = local_file(root, str(Path(name).parent / node.attrib['file']))
                child = str(path.relative_to(Path(root).resolve()))
                if child not in files:
                    raise ValueError(f'Unsealed MJCF dependency: {child}')
                if node.tag == 'include':
                    pending.append(child)


def load_inputs(bundle, robot, task):
    bundle, robot, task = map(lambda p: Path(p).resolve(), (bundle, robot, task))
    manifest = read_json(bundle / 'bundle.json')
    if manifest['format'] not in {f'h2r-preprocess-v{i}' for i in (1, 2, 3)}:
        raise ValueError('Unsupported bundle format')
    if manifest.get('kind', 'physical') != 'physical':
        raise ValueError('ConTrack requires a physical bundle')
    if identity(manifest['files']) != manifest['bundle_id']:
        raise ValueError('Changed bundle identity')
    check_files(bundle, manifest['files'])
    check_xml(bundle, 'scene.xml', manifest['files'])
    robot_manifest = read_json(robot / 'robot.json')
    check_files(robot, robot_manifest['files_sha256'])
    task_spec = read_json(task / 'task.json')
    unsigned = {k: v for k, v in task_spec.items() if k != 'task_id'}
    if (task_spec['format'] not in {f'h2r-evaluation-v{i}' for i in (1, 2, 3)}
            or identity(unsigned) != task_spec['task_id']):
        raise ValueError('Unsupported or changed task')
    check_files(task, task_spec['files'])
    check_xml(task, task_spec['scene'], task_spec['files'])
    for key in ('bundle_id', 'scene_id'):
        if task_spec[key] != manifest[key]:
            raise ValueError(f'Task differs from bundle: {key}')
    if task_spec['robot_manifest_sha256'] != digest(robot / 'robot.json'):
        raise ValueError('Task differs from robot')
    if task_spec['semantics'] != read_json(robot / 'semantics.json'):
        raise ValueError('Task differs from robot semantics')
    if not task_spec['observation']['contacts']:
        raise ValueError('Prepare a common task with --contacts')
    reference = read_json(bundle / 'reference.json')
    if (not reference.get('mano', {}).get('geometry_decoded') or
            reference['mano'].get('joint_order') != 'MANO 16 + thumb/index/middle/ring/little tips'):
        raise ValueError('Unsupported decoded human joint ordering')
    for name in ('reference.npz', 'reference.json'):
        if digest(task / name) != manifest['files'][name]:
            raise ValueError('Task reference differs from bundle')
    if reference['reference_sha256'] != manifest['files']['reference.npz']:
        raise ValueError('Reference identity differs from bundle')
    with np.load(bundle / 'reference.npz', allow_pickle=False) as source:
        arrays = dict(source)
    time = arrays['time_s']
    if (time.ndim != 1 or len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0)
            or not np.isclose(time[-1], reference['duration_s'], atol=1e-10, rtol=0)
            or not np.isclose(time[-1], task_spec['end_tick'] * task_spec['dt_s'], atol=1e-10, rtol=0)):
        raise ValueError('Invalid or inconsistent reference clock')
    if any(not np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype, np.number)):
        raise ValueError('Nonfinite reference')
    sides = list(task_spec['semantics']['hands'])
    declared = reference.get('task', {}).get('hand_sides', sides)
    if set(declared) != set(sides):
        raise ValueError('Task and robot must contain the same hands')
    for side in sides:
        for suffix, shape in [('joints_m', (21, 3)), ('vertices_m', (778, 3))]:
            if arrays[f'{side}_{suffix}'].shape != (len(time), *shape):
                raise ValueError(f'Unsupported decoded MANO layout: {side}_{suffix}')
    parts = reference['object_geometry']['part_order']
    if set(parts) != set(task_spec['object']['parts']):
        raise ValueError('Part names differ between task and reference')
    for key, width in [('part_pos_m', 3), ('part_quat_wxyz', 4)]:
        if arrays[key].shape != (len(time), len(parts), width):
            raise ValueError(f'Invalid {key} layout')
    if not np.allclose(np.linalg.norm(arrays['part_quat_wxyz'], axis=-1), 1, atol=1e-5):
        raise ValueError('Non-unit object reference quaternion')
    return manifest, task_spec, reference, arrays


def seal_prepared(root, binding):
    binding['files'] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and p != root / 'binding.json'}
    binding['prepared_id'] = identity(binding)
    write_json(root / 'binding.json', binding)


def load_prepared(root):
    root = Path(root).resolve()
    binding = read_json(root / 'binding.json')
    if binding['format'] != 'contrack-h2r-v2' or binding.get('reference_recipe') != RECIPE_ID:
        raise ValueError('Legacy or unsupported preparation; rerun prepare for the native H2R recipe')
    if (binding['prepared_id'] !=
            identity({k: v for k, v in binding.items() if k != 'prepared_id'})):
        raise ValueError('Unsupported or changed ConTrack preparation')
    check_files(root, binding['files'])
    check_xml(root, 'task/' + binding['task']['scene'], binding['files'])
    with np.load(root / 'reference.npz', allow_pickle=False) as data:
        arrays = dict(data)
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite prepared reference')
    return binding, arrays
