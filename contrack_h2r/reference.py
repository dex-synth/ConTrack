"""Native control-grid references and shared training/deployment features."""

import torch
from tensordict import TensorDict

from rsl_rl_contrack.mimic import resample_reference


class Reference:
    def __init__(self, binding, arrays, device, control_dt=None):
        self.layout = binding['layout']
        self.dt = control_dt if control_dt is not None else binding['task']['dt_s'] * binding['task']['control_ticks']
        source_time = torch.as_tensor(arrays['time_s'], dtype=torch.float64)
        self.duration = float(source_time[-1])
        source_fps = (len(source_time) - 1) / self.duration
        expected_time = torch.arange(len(source_time), dtype=torch.float64) / source_fps
        if source_time[0] != 0 or not torch.allclose(source_time, expected_time, atol=1e-7, rtol=0):
            raise ValueError('Native reference requires a uniform source clock starting at zero')
        data = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in arrays.items()}
        native = resample_reference(dict(
            qpos=data['robot_q'][:, None], object_pos=data['part_pos'], object_quat_wxyz=data['part_quat'],
            hand_is_contact=data['contact'], hand_contact_points=data['contact_points'],
            fps=torch.tensor([source_fps], device=device)), 1 / self.dt)
        self.data = dict(robot_q=native['qpos'][:, 0], part_pos=native['object_pos'],
                         part_quat=native['object_quat_wxyz'], contact=native['hand_is_contact'].float(),
                         contact_points=native['hand_contact_points'])
        length = len(self.data['robot_q'])
        if abs((length - 1) * self.dt - self.duration) > 1e-8:
            raise ValueError('Task duration must contain complete training control intervals')
        self.time = torch.arange(length, device=device) * self.dt
        self.scale = torch.zeros(len(self.layout['joint_names']), device=device)
        for group, indices in self.layout['groups'].items():
            self.scale[indices] = self.layout['action_scales'][group]

    def indices(self, time):
        phase = time.clamp(0, self.duration) / self.dt
        index = phase.round().long().clamp(0, len(self.time) - 1)
        if not torch.allclose(phase, index.float(), atol=2e-4, rtol=0):
            raise ValueError('Reference queries must use the policy control grid')
        return index

    def sample(self, time):
        index = self.indices(time)
        return {key: values[index] for key, values in self.data.items()}

    def observations(self, state, time):
        index = self.indices(time)
        future_index = (index[:, None] + torch.arange(6, device=time.device)).clamp(max=len(self.time) - 1)
        future = {key: self.data[key][future_index] for key in ('robot_q', 'part_pos', 'part_quat')}
        future_pose = torch.cat((future['part_pos'], future['part_quat']), -1).flatten(2)
        values = torch.cat((state['robot_q'], state['part_pose'].flatten(1),
                            state['robot_dq'], state['part_velocity'].flatten(1),
                            state['contact'].flatten(1).float(),
                            torch.cat((future['robot_q'], future_pose), -1).flatten(1),
                            self.data['contact'][index].flatten(1),
                            (index.float() / (len(self.time) - 1))[:, None]), -1)
        return TensorDict({'policy': values, 'critic': values}, batch_size=[len(time)])

    def targets(self, actions, time):
        return self.sample(time)['robot_q'] + self.scale * actions.clamp(-1, 1)
