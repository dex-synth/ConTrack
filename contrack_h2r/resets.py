"""ConTrack adaptive reset library in consistent MuJoCo generalized coordinates."""

import torch


class ResetLibrary:
    def __init__(self, qpos, qvel, num_envs):
        self.state = torch.cat((qpos, qvel), -1).clone()
        self.length = len(qpos)
        self.device = qpos.device
        self.track = torch.zeros(self.length, device=self.device)
        self.best = torch.zeros_like(self.track)
        self.success_ema = torch.zeros((), device=self.device)
        self.history = torch.zeros((num_envs, self.length, self.state.shape[-1]), device=self.device)
        self.start = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset_count = torch.zeros(self.length, dtype=torch.long, device=self.device)
        self.terminate_count = torch.zeros_like(self.reset_count)

    def sample(self, ids):
        horizon = self.length-torch.arange(self.length-1, device=self.device)
        ratio = self.track[:-1]/horizon
        gradient = torch.zeros_like(ratio)
        gradient[:-1] = ratio[1:]-ratio[:-1]
        worst = gradient.topk(min(64, len(gradient))).indices
        if not (worst == 0).any():
            worst[-1] = 0
        start = torch.multinomial(torch.softmax(-ratio, 0), len(ids), replacement=True)
        hard = torch.rand(len(ids), device=self.device) < 0.5
        candidate = (worst[torch.randint(len(worst), (len(ids),), device=self.device)]-100).clamp(min=0)
        start = torch.where(hard, candidate, start)
        self.reset_count += torch.bincount(start, minlength=self.length)
        self.start[ids] = start
        self.history[ids, start] = self.state[start]
        return start, self.state[start]

    def record(self, frame, state):
        self.history[torch.arange(len(frame), device=self.device), frame] = state

    def finish(self, ids, end):
        if not len(ids):
            return
        start = self.start[ids]
        self.success_ema.mul_(0.95).add_(0.05*(end == self.length-1).float().mean())
        self.terminate_count += torch.bincount(end, minlength=self.length)
        grid = torch.arange(self.length, device=self.device)[None].expand(len(ids), -1)
        valid = (grid >= start[:, None]) & (grid <= end[:, None])
        frames = grid[valid]
        remaining = (end[:, None]-grid+1)[valid].float()
        envs = ids[:, None].expand_as(grid)[valid]
        count = torch.bincount(frames, minlength=self.length)
        mean = torch.bincount(frames, weights=remaining, minlength=self.length)/count.clamp(min=1)
        self.track.copy_(torch.where(count > 0, 0.95*self.track+0.05*mean, self.track))
        self.best.mul_(0.99999)
        better = (frames != 0) & (remaining > self.best[frames])
        frames, remaining, envs = frames[better], remaining[better], envs[better]
        order = torch.argsort(frames*(self.length+1)-remaining)
        frames, remaining, envs = frames[order], remaining[order], envs[order]
        keep = torch.ones_like(frames, dtype=torch.bool)
        keep[1:] = frames[1:] != frames[:-1]
        frames, remaining, envs = frames[keep], remaining[keep], envs[keep]
        self.state[frames] = self.history[envs, frames]
        self.best[frames] = remaining

    def thresholds(self):
        weight = ((self.success_ema-0.9)/(0.999-0.9)).clamp(0, 1)
        return 0.1-0.05*weight, 1.0-0.5*weight

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in
                ('state', 'track', 'best', 'success_ema', 'reset_count', 'terminate_count')}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Incomplete reset-library checkpoint')
        for name, value in state.items():
            target = getattr(self, name)
            if target.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid reset-library field: {name}')
            target.copy_(value)
