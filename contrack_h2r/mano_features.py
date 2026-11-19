"""Task boundary for FlashCHORD's unchanged MANO/AxisLayerFK decoder.

The IK targets are the decoder's own output: joint positions with its fingertip vertices
(NATIVE_TIPS, not the task's tip vertices) and AxisLayerFK joint orientations, which the task
package does not carry. The decoded mesh and MANO joints must reproduce the task's geometry.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

from h2r_io import load_task
import numpy as np
from scipy.spatial.transform import Rotation

# Decoder joint order (wrist, then thumb..pinky with their tips) as indices into the task's joints_m.
ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TIP_SLOTS = [4, 8, 12, 16, 20]
NATIVE_TIPS = {'left': [745, 317, 445, 556, 673], 'right': [745, 317, 444, 556, 673]}


def decode(task, mano_root, output):
    import torch
    from v2d.task_library_loader.lib.read_mano import MANO

    torch.set_num_threads(1)
    demo = load_task(task).demo
    convention = demo.reference['mano']
    with tempfile.TemporaryDirectory() as temp:
        models = Path(temp) / 'models'
        models.mkdir()
        for asset in convention['assets']:
            source = (mano_root / asset['name']).resolve()
            if hashlib.sha256(source.read_bytes()).hexdigest() != asset['sha256']:
                raise ValueError(f'MANO asset differs from the task: {asset["name"]}')
            (models / asset['name']).symlink_to(source)
        # h2r-mano-v1: absolute rotations on the flat mean hand; mano_trans is the wrist position.
        mano = MANO(temp, device=torch.device('cpu'), flat_hand_mean=True, center_idx=0)
    if convention['left_shapedirs_x_mirrored']:
        left, right = mano.left_mano_layer.th_shapedirs, mano.right_mano_layer.th_shapedirs
        # As h2r_io.mano.model: the legacy left asset repeats the right hand's x shape directions.
        if (left[:, 0] - right[:, 0]).abs().sum() < 1:
            left[:, 0] *= -1
    result = {'time_s': demo.time_s}
    checks = {'python': platform.python_version(), 'native_tip_vertex_ids': NATIVE_TIPS,
              'versions': {n: version(n) for n in ('torch', 'numpy', 'scipy', 'manotorch')}}
    for side, hand in demo.hands.items():
        pose = Rotation.from_quat(hand.mano_quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(-1, 48)
        with torch.no_grad():
            decoded = mano.forward(side, *[torch.as_tensor(a, dtype=torch.float32) for a in
                                          (hand.mano_shape, pose[:, :3], hand.mano_trans, pose[:, 3:])])
        joints = decoded['joints'].cpu().numpy()
        differences = np.linalg.norm(joints - hand.joints_m[:, ORDER], axis=-1)
        error = float(np.delete(differences, TIP_SLOTS, axis=1).max())
        vertices_error = float(np.linalg.norm(decoded['vertices'].cpu().numpy() - hand.vertices_m, axis=-1).max())
        if max(error, vertices_error) > 1e-4:
            raise ValueError(f'{side} MANO geometry differs: joints={error}m, surface={vertices_error}m')
        result[f'{side}_joints'] = joints
        result[f'{side}_joints_wxyz'] = decoded['joints_wxyz'].cpu().numpy()
        checks[side] = dict(mano_joint_max_error_m=error, vertex_max_error_m=vertices_error,
                            native_vs_task_tip_max_distance_m=float(differences[:, TIP_SLOTS].max()))
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite MANO features')
    np.savez_compressed(output / 'features.npz', **result)
    (output / 'mano.json').write_text(json.dumps(checks, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'mano-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    decode(**vars(parser.parse_args()))
