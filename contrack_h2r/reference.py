"""Reference interpolation and full policy features in physical task time."""

import torch
from tensordict import TensorDict


class Reference:
    def __init__(self, binding, arrays, device):
        self.layout = binding['layout']
        self.dt = binding['task']['dt_s'] * binding['task']['control_ticks']
        self.data = {k: torch.as_tensor(v, device=device, dtype=torch.float32)
                     for k, v in arrays.items()}
        self.time = self.data['time_s']
        self.scale = torch.zeros(len(self.layout['joint_names']), device=device)
        for group, indices in self.layout['groups'].items():
            self.scale[indices] = self.layout['action_scales'][group]

    def sample(self, time):
        time = time.clamp(0, self.time[-1])
        hi = torch.searchsorted(self.time, time.contiguous(), right=True).clamp(1, len(self.time)-1)
        lo = hi - 1
        alpha = (time - self.time[lo]) / (self.time[hi] - self.time[lo])
        result = {}
        for key in ('robot_q', 'part_pos', 'part_quat', 'contact', 'contact_points'):
            values = self.data[key]
            a, b = values[lo], values[hi]
            weight = alpha.reshape(*alpha.shape, *([1] * (a.ndim-alpha.ndim)))
            if key in ('contact', 'contact_points'):
                result[key] = torch.where(weight <= 0.5, a, b)
            else:
                if key == 'part_quat':
                    b = torch.where((a*b).sum(-1, keepdim=True) < 0, -b, b)
                value = a + weight*(b-a)
                result[key] = torch.nn.functional.normalize(value, dim=-1) if key == 'part_quat' else value
        return result

    def observations(self, state, time):
        now = self.sample(time)
        future = self.sample(time[:, None] + torch.arange(6, device=time.device)[None] * self.dt)
        future_pose = torch.cat((future['part_pos'], future['part_quat']), -1).flatten(2)
        values = torch.cat((state['robot_q'], state['part_pose'].flatten(1),
                            state['robot_dq'], state['part_velocity'].flatten(1),
                            state['contact'].flatten(1).float(),
                            torch.cat((future['robot_q'], future_pose), -1).flatten(1),
                            now['contact'].flatten(1),
                            (time / self.time[-1]).clamp(0, 1)[:, None]), -1)
        return TensorDict({'policy': values, 'critic': values}, batch_size=[len(time)])

    def targets(self, actions, time):
        return self.sample(time)['robot_q'] + self.scale * actions.clamp(-1, 1)
