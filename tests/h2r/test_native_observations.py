"""Compare feature order and values with the original simulator-free callbacks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from contrack_h2r.reference import Reference


def test_native_feature_layout():
    path = Path(__file__).parents[2] / 'source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/mdp/observations.py'
    spec = importlib.util.spec_from_file_location('native_observations', path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    torch.manual_seed(5)
    length, n, p = 7, 3, 2
    frame = torch.tensor([0, 2, 6])
    q = torch.randn(length, 2, 29)
    pos = torch.randn(length, p, 3)
    quat = torch.nn.functional.normalize(torch.randn(length, p, 4), dim=-1)
    # Native callbacks consume the already resampled quaternion branch. Use a
    # continuous branch here; antipodal interpolation is tested separately.
    for i in range(1, length):
        quat[i] = torch.where((quat[i-1]*quat[i]).sum(-1, keepdim=True) < 0, -quat[i], quat[i])
    contacts = torch.rand(length, 2, p, 10) > .5
    dq, velocity = torch.randn(n, 58), torch.randn(n, p, 6)
    class Scene(dict):
        env_origins = torch.zeros(n, 3)
    scene = Scene(objects=NS(data=NS(object_pos_w=pos[frame], object_quat_w=quat[frame],
                                     object_lin_vel_w=velocity[..., :3], object_ang_vel_w=velocity[..., 3:])))
    keys = {}
    for h, side in enumerate(('left', 'right')):
        scene[side] = NS(data=NS(joint_pos=q[frame, h], joint_vel=dq[:, h*29:(h+1)*29]))
        keys[side] = []
        for slot in range(10):
            key = f'{side}-{slot}'
            points = torch.zeros(n, 1, p, 3)
            points[~contacts[frame, h, :, slot][:, None]] = torch.nan
            scene[key] = NS(data=NS(contact_pos_w=points))
            keys[side].append(key)
    env = NS(scene=scene, ref_hands=[('left', list(range(29)), 0), ('right', list(range(29)), 1)],
             num_envs=n, device='cpu', frame_idx=frame, frame_idx_f=frame.float(), ref_len=length, ref_step_frames=1,
             is_rhand=[False, True], hand_contact_sensor_keys=keys, ref_qpos=q, ref_object_pos=pos,
             ref_object_quat=quat, ref_hand_is_contact=contacts.permute(0, 2, 1, 3))
    expected = torch.cat([getattr(native, name)(env) for name in
                          ('all_qpos', 'all_qvel', 'hand_contact', 'ref_all_qpos', 'ref_hand_contact', 'phase')], -1)
    arrays = dict(time_s=np.arange(length)/30, robot_q=q.reshape(length, 58), part_pos=pos, part_quat=quat,
                  contact=contacts, contact_points=torch.zeros(length, 2, p, 10, 3))
    binding = dict(task={'dt_s': 1/240, 'control_ticks': 8},
                   layout={'joint_names': list(range(58)), 'groups': {}, 'action_scales': {}})
    reference = Reference(binding, arrays, 'cpu')
    state = dict(robot_q=q[frame].reshape(n, 58), robot_dq=dq, part_pose=torch.cat((pos[frame], quat[frame]), -1),
                 part_velocity=velocity, contact=contacts[frame])
    actual = reference.observations(state, reference.time[frame])['policy']
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
