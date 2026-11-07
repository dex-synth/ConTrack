"""H2R-specific geometric Sharpa IK; this is not the unpublished native HDF5 generator."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .layout import CHAINS, FINGERS


def palm_rotation(joints, side):
    z = joints[4] - joints[0]
    if np.linalg.norm(z) < 1e-6:
        raise ValueError('Degenerate human palm')
    z /= np.linalg.norm(z)
    y = (joints[1] - joints[7]) * (1 if side == 'right' else -1)
    y -= z * np.dot(z, y)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('Degenerate human palm')
    y /= np.linalg.norm(y)
    return np.stack((np.cross(y, z), y, z), axis=1)


def retarget(scene, arrays, qpos, iterations=40):
    m, layout = scene.model, scene.layout
    d = mujoco.MjData(m)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    errors = []
    for h, side in enumerate(layout['hands']):
        names = layout['joint_names'][h*28:(h+1)*28]
        qi = scene.qids[h*28:(h+1)*28]
        vi = scene.vids[h*28:(h+1)*28]
        limits = scene.limits[h*28:(h+1)*28]
        wrist = m.body(f'{side}_hand_C_MC').id
        tips = [m.body(f'{side}_{finger}_fingertip').id for finger in FINGERS]
        previous = np.clip(m.qpos0[qi], limits[:, 0], limits[:, 1])
        hand_errors = []
        for frame, joints in enumerate(arrays[f'{side}_joints_m']):
            d.qpos[:] = qpos[frame]
            d.qpos[qi] = previous
            desired_rotation = palm_rotation(joints, side)
            targets = joints[[chain[-1] for chain in CHAINS]]
            for _ in range(iterations):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                residual, jacobian = [], []
                for body, target, weight in [(wrist, joints[0], 2.0), *zip(tips, targets, [1.0]*5)]:
                    mujoco.mj_jacBody(m, d, jp, jr, body)
                    residual.extend(weight * (target-d.xpos[body]))
                    jacobian.extend(weight * jp[:, vi])
                mujoco.mj_jacBody(m, d, jp, jr, wrist)
                angular = Rotation.from_matrix(desired_rotation @ d.xmat[wrist].reshape(3, 3).T).as_rotvec()
                residual.extend(0.08 * angular)
                jacobian.extend(0.08 * jr[:, vi])
                jacobian, residual = np.asarray(jacobian), np.asarray(residual)
                # Small temporal regularizer; all joints retain their named units and limits.
                damping = 1e-5
                delta = np.linalg.solve(jacobian.T @ jacobian + damping*np.eye(28),
                                        jacobian.T @ residual + damping*(previous-d.qpos[qi]))
                delta[:3] = np.clip(delta[:3], -0.04, 0.04)
                delta[3:] = np.clip(delta[3:], -0.25, 0.25)
                d.qpos[qi] = np.clip(d.qpos[qi]+delta, limits[:, 0], limits[:, 1])
                if np.max(np.abs(delta)) < 1e-5:
                    break
            mujoco.mj_kinematics(m, d)
            previous = d.qpos[qi].copy()
            qpos[frame, qi] = previous
            hand_errors.append(float(np.linalg.norm(d.xpos[tips]-targets, axis=-1).mean()))
        errors.append({'hand': side, 'mean_tip_error_m': float(np.mean(hand_errors)),
                       'max_tip_error_m': float(np.max(hand_errors)), 'joints': names})
    return qpos, errors
