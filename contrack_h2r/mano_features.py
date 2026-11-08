"""Bundle boundary for FlashCHORD's unchanged MANO/AxisLayerFK decoder."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation


def mano_parameters(reference, arrays, side):
    """Convert dataset parameter representations, keeping the benchmark world."""
    prefix = f'{side}_mano_'
    if prefix + 'quat' in arrays:
        quat = arrays[prefix + 'quat']
        if quat.shape[1:] != (16, 4) or not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-5):
            raise ValueError('Expected sixteen unit MANO quaternions in wxyz order')
        pose = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(-1, 48)
        orient, fingers = pose[:, :3], pose[:, 3:]
    else:
        orient, fingers = arrays[prefix + 'rot'], arrays[prefix + 'pose']
    trans = arrays[prefix + 'trans']
    if reference['mano'].get('parameter_coordinates') == 'source_world':
        world = np.asarray(reference['coordinates']['source_to_benchmark'])
        orient = (Rotation.from_matrix(world[:3, :3]) * Rotation.from_rotvec(orient)).as_rotvec()
        trans = trans @ world[:3, :3].T + world[:3, 3]
    return orient, fingers, trans


def decode(bundle, mano_root, output):
    import torch
    from v2d.task_library_loader.lib.read_mano import MANO

    torch.set_num_threads(1)
    reference = json.loads((bundle / 'reference.json').read_text())
    layout = json.loads((output / 'layout.json').read_text())
    with np.load(bundle / 'reference.npz', allow_pickle=False) as data:
        arrays = dict(data)
    with tempfile.TemporaryDirectory() as temp:
        models = Path(temp) / 'models'
        models.mkdir()
        for side in ('LEFT', 'RIGHT'):
            name = f'MANO_{side}.pkl'
            source = (mano_root / name).resolve()
            expected = next(a['sha256'] for a in reference['mano']['assets'] if a['name'] == name)
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f'MANO asset differs from the common bundle: {name}')
            (models / name).symlink_to(source)
        mano = MANO(temp, device=torch.device('cpu'), flat_hand_mean=reference['mano']['flat_hand_mean'],
                    center_idx=0 if reference['mano'].get('wrist_recentered') else None)
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    result = {'time_s': arrays['time_s']}
    checks = {'python': platform.python_version(),
              'versions': {n: version(n) for n in ('torch', 'numpy', 'scipy', 'manotorch')}}
    for side in layout['hands']:
        orient, fingers, trans = mano_parameters(reference, arrays, side)
        with torch.no_grad():
            decoded = mano.forward(side, *[torch.as_tensor(a, dtype=torch.float32) for a in
                                          (arrays[f'{side}_mano_shape'], orient, trans, fingers)])
        joints = decoded['joints'].cpu().numpy()
        differences = np.linalg.norm(joints - arrays[f'{side}_joints_m'][:, order], axis=-1)
        tip_slots = [4, 8, 12, 16, 20]
        error = float(np.max(np.delete(differences, tip_slots, axis=1)))
        vertices_error = float(np.max(np.linalg.norm(
            decoded['vertices'].cpu().numpy() - arrays[f'{side}_vertices_m'], axis=-1)))
        if max(error, vertices_error) > 1e-4:
            raise ValueError(f'{side} MANO geometry differs: joints={error}m, surface={vertices_error}m')
        native_tips = [745, 317, 444 if side == 'right' else 445, 556, 673]
        np.testing.assert_allclose(joints[:, tip_slots], arrays[f'{side}_vertices_m'][:, native_tips],
                                   atol=1e-6, rtol=0)
        result[f'{side}_joints'] = joints
        result[f'{side}_joints_wxyz'] = decoded['joints_wxyz'].cpu().numpy()
        checks[side] = dict(shared_joint_max_error_m=error, vertex_max_error_m=vertices_error,
                            native_tip_vertex_ids=native_tips,
                            native_vs_bench_tip_max_distance_m=float(differences[:, tip_slots].max()))
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite MANO features')
    np.savez_compressed(output / 'features.npz', **result)
    (output / 'mano.json').write_text(json.dumps(checks, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'mano-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    decode(**vars(parser.parse_args()))
