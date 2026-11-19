"""Method artifacts bound to one sealed benchmark task; the task itself is read by h2r_io."""

import hashlib
import json
from pathlib import Path

from h2r_io import load_task
import numpy as np

from rsl_rl_contrack.sharpa_recipe import RECIPE_ID

# v3: prepared from one task package (h2r-task-v1) and bound to task_id, demo_id and robot_id.
PREPARED_FORMAT = 'contrack-h2r-v3'
# v3: per-step body refresh runs kinematics/com_pos/com_vel only (v2 ran a second full forward).
# v4: bound to the task package; the control period is stored in seconds, never as a tick count.
DEPLOYMENT_FORMAT = 'contrack-h2r-deployment-v4'
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
    # The copied frozen IK is bound whole, with its SOURCE.json.
    files.update(p for p in (root / 'contrack_h2r' / '_vendor').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    # The locks pin every installed dependency, the benchmark loader h2r-io among them.
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


def seal_prepared(root, binding):
    """Bind the method files; the task copy in task/ is sealed by its own task_id."""
    binding['files'] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and p != root / 'binding.json' and root / 'task' not in p.parents}
    binding['prepared_id'] = identity(binding)
    write_json(root / 'binding.json', binding)


def load_prepared(root):
    root = Path(root).resolve()
    binding = read_json(root / 'binding.json')
    if binding.get('format') != PREPARED_FORMAT or binding.get('reference_recipe') != RECIPE_ID:
        raise ValueError(f'Unsupported preparation {binding.get("format")!r}: rerun contrack_h2r.prepare '
                         f'on the task directory ({PREPARED_FORMAT})')
    if (binding['prepared_id'] !=
            identity({k: v for k, v in binding.items() if k != 'prepared_id'})):
        raise ValueError('Changed ConTrack preparation')
    check_files(root, binding['files'])
    with np.load(root / 'reference.npz', allow_pickle=False) as data:
        arrays = dict(data)
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite prepared reference')
    return binding, arrays


def prepared_task(root, binding):
    """The standalone task copy this preparation was made from, verified file by file."""
    task = load_task(Path(root) / 'task')
    if task.task_id != binding['task_id']:
        raise ValueError('Prepared directory holds a different task')
    return task
