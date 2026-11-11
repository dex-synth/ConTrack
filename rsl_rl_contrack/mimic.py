"""Simulator-independent reference operations from the public Sharpa MDP.

Extracted from mimic_rby1a_sharpa/mdp/state.py at f5efe18. Keep its sampling,
finite differences and random draw order; both simulator adapters use these.
"""

import torch


def resample_reference(ref, action_fps):
    raw_len = int(ref['qpos'].shape[0])
    raw_fps = float(torch.max(ref['fps']).item())
    action_len = int((raw_len - 1) * action_fps / raw_fps) + 1
    t = torch.arange(action_len, device=ref['qpos'].device, dtype=torch.float32) * (raw_fps / action_fps)
    i0 = t.to(torch.long)
    i1 = torch.clamp(i0 + 1, max=raw_len - 1)
    w = (t - i0.to(torch.float32)).view(-1, 1, 1)
    q0, q1 = ref['object_quat_wxyz'][i0], ref['object_quat_wxyz'][i1]
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
    q = q0 * (1.0 - w) + q1 * w
    nn = torch.clamp((t + 0.5).to(torch.long), max=raw_len - 1)
    return dict(qpos=ref['qpos'][i0] * (1.0 - w) + ref['qpos'][i1] * w,
                object_pos=ref['object_pos'][i0] * (1.0 - w) + ref['object_pos'][i1] * w,
                object_quat_wxyz=q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8),
                hand_is_contact=ref['hand_is_contact'][nn].to(torch.bool),
                hand_contact_points=ref['hand_contact_points'][nn])


def linear_velocity(position, fps):
    velocity = torch.zeros_like(position)
    velocity[1:-1] = 0.5 * (position[2:] - position[:-2]) * fps
    velocity[0] = (position[1] - position[0]) * fps
    velocity[-1] = (position[-1] - position[-2]) * fps
    return velocity


def angular_velocity(q, fps):
    """Native shortest relative rotation, expressed in the world frame (wxyz)."""
    q_prev, q_next = torch.empty_like(q), torch.empty_like(q)
    q_prev[0], q_next[0] = q[0], q[1]
    q_prev[-1], q_next[-1] = q[-2], q[-1]
    q_prev[1:-1], q_next[1:-1] = q[:-2], q[2:]
    w0, x0, y0, z0 = q_prev.unbind(-1)
    w1, x1, y1, z1 = q_next.unbind(-1)
    rel = torch.stack([w1*w0 + x1*x0 + y1*y0 + z1*z0,
                       -w1*x0 + x1*w0 - y1*z0 + z1*y0,
                       -w1*y0 + x1*z0 + y1*w0 - z1*x0,
                       -w1*z0 - x1*y0 + y1*x0 + z1*w0], dim=-1)
    rel = torch.where(rel[..., :1] < 0.0, -rel, rel)
    w, v = rel[..., 0], rel[..., 1:]
    s = torch.linalg.norm(v, dim=-1)
    a = 2.0 * torch.atan2(s, w.clamp(min=1e-8))
    rotvec = v * (a / s.clamp(min=1e-8)).unsqueeze(-1)
    scale = torch.full((len(q), 1, 1), 0.5 * fps, device=q.device)
    scale[0] = fps
    scale[-1] = fps
    return rotvec * scale


def sample_start_frames(track, n, *, topk=64, delta=100, epsilon=0.5, tau=1.0):
    length, device = len(track), track.device
    horizon = (length - torch.arange(length - 1, device=device)).to(torch.float32)
    ratio = track[:-1] / horizon
    gradient = torch.zeros_like(ratio)
    gradient[:-1] = ratio[1:] - ratio[:-1]
    k = min(topk, len(gradient))
    worst = torch.topk(gradient, k, largest=True).indices
    if not (worst == 0).any().item():
        worst[-1] = 0
    start = torch.multinomial(torch.softmax(-ratio / tau, dim=0), n, replacement=True)
    hard = torch.rand((n,), device=device) < epsilon
    start[hard] = torch.clamp(
        worst[torch.randint(0, k, (int(hard.sum().item()),), device=device)] - delta, min=0)
    return start
