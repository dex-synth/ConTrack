"""Convert native control-grid reset states to the common MJCF coordinates."""

import mujoco
import numpy as np
import torch

from rsl_rl_contrack.mimic import angular_velocity, linear_velocity, resample_reference
from rsl_rl_contrack.sharpa_recipe import CONTROL_HZ

from .reference import Reference


def reset_reference(model, layout, arrays):
    reference = Reference(layout, arrays, 'cpu', 1 / CONTROL_HZ)
    values = reference.data
    source = torch.as_tensor(arrays['qpos'], dtype=torch.float32)
    # Scalar joints use the same interpolation as the robot reference. Free-root
    # quaternion columns are replaced separately, using the native nlerp branch.
    free = [j for j, kind in enumerate(model.jnt_type) if kind == mujoco.mjtJoint.mjJNT_FREE]
    quaternion_addresses = [int(model.jnt_qposadr[j]) + 3 for j in free]
    if any(kind == mujoco.mjtJoint.mjJNT_BALL for kind in model.jnt_type):
        raise ValueError('Ball joints are not supported by the common Sharpa reset mapping')
    quaternions = (torch.stack([source[:, a:a+4] for a in quaternion_addresses], 1)
                   if free else torch.tensor([1., 0, 0, 0]).expand(len(source), 1, 4))
    sampled = resample_reference(dict(qpos=source[:, None], object_pos=torch.as_tensor(arrays['part_pos']).float(),
                                     object_quat_wxyz=quaternions, fps=torch.tensor([(len(source)-1)/reference.duration]),
                                     hand_is_contact=torch.zeros(len(source), 1),
                                     hand_contact_points=torch.zeros(len(source), 1)), CONTROL_HZ)
    qpos = sampled['qpos'][:, 0].numpy().astype(np.float64)
    for i, address in enumerate(quaternion_addresses):
        qpos[:, address:address+4] = sampled['object_quat_wxyz'][:, i]
    qvel = np.zeros((len(qpos), model.nv))
    scalar_velocity = linear_velocity(torch.from_numpy(qpos).float(), CONTROL_HZ).numpy()
    for j, kind in enumerate(model.jnt_type):
        # MuJoCo 4 enums do not equal NumPy integers from the left side, as `in` compares them.
        if int(kind) in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            qvel[:, model.jnt_dofadr[j]] = scalar_velocity[:, model.jnt_qposadr[j]]

    part_ids = [model.body(name).id for name in layout['part_bodies']]
    linear = linear_velocity(values['part_pos'], CONTROL_HZ).numpy()
    angular = angular_velocity(values['part_quat'], CONTROL_HZ).numpy()
    data = mujoco.MjData(model)
    for j in free:
        part = part_ids.index(int(model.jnt_bodyid[j]))
        address = model.jnt_dofadr[j]
        qvel[:, address:address+3] = linear[:, part]
        for frame, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            rotation = data.xmat[part_ids[part]].reshape(3, 3)
            qvel[frame, address+3:address+6] = rotation.T @ angular[frame, part]

    # Hinges must obey the shared joint tree. Report the resulting body-velocity
    # difference from independently differenced part tracks, without changing it.
    max_linear, max_angular = 0., 0.
    for frame, (q, v) in enumerate(zip(qpos, qvel)):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        for part, body in enumerate(part_ids):
            # mj_objectVelocity(BODY) reports COM velocity; references and the
            # shared observation protocol use the body's frame origin instead.
            spatial = data.cvel[body]
            offset = data.xpos[body] - data.subtree_com[model.body_rootid[body]]
            actual_linear = spatial[3:] + np.cross(spatial[:3], offset)
            max_linear = max(max_linear, float(np.linalg.norm(actual_linear - linear[frame, part])))
            max_angular = max(max_angular, float(np.linalg.norm(spatial[:3] - angular[frame, part])))
    report = dict(recipe='native-control-grid-central-difference-v1', control_hz=CONTROL_HZ,
                  frames=len(qpos), max_part_linear_velocity_difference_m_s=max_linear,
                  max_part_angular_velocity_difference_rad_s=max_angular,
                  mapping='scalar joint differences; free-root world linear/local angular velocity; constrained hinge FK')
    return qpos, qvel, report
