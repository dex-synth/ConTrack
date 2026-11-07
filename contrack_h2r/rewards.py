"""ConTrack's task/style/penalty equations, detached from Isaac managers."""

import torch


def pose_error(state, ref):
    position = torch.linalg.vector_norm(state['part_pose'][..., :3]-ref['part_pos'], dim=-1)
    dot = (state['part_pose'][..., 3:]*ref['part_quat']).sum(-1).abs().clamp(0, 1)
    return position, 2*torch.acos(dot)


def reward_groups(state, previous_velocity, ref, layout, dt):
    """Return rg, rs, rp with the control-period multiplier applied exactly once."""
    _, angle = pose_error(state, ref)
    pos_mse = (state['part_pose'][..., :3]-ref['part_pos']).square().mean((-1, -2))
    rg = 500*torch.exp(-pos_mse/0.05**2) + 500*torch.exp(-angle.mean(-1).square()/0.5**2)
    groups = layout['groups']
    wrist = groups['wrist_translation'] + groups['wrist_rotation']
    fingers = groups['fingers']
    scale = torch.ones(state['robot_q'].shape[-1], device=state['robot_q'].device)
    scale[groups['wrist_translation']] = layout['wrist_error_scales']['translation_m']
    scale[groups['wrist_rotation']] = layout['wrist_error_scales']['rotation_rad']
    error = ((state['robot_q']-ref['robot_q'])/scale).square()
    rs = 10*torch.exp(-4*error[:, wrist].mean(-1)) + 10*torch.exp(-4*error[:, fingers].mean(-1))
    shared = state['contact'].bool() & ref['contact'].bool()  # world, hand, part, slot
    rs += 100*shared.any(2).float().mean((1, 2))
    distance = (state['contact_points']-ref['contact_points']).square().sum(-1)
    count = shared.sum(2)
    mean_distance = (distance*shared).sum(2)/count.clamp(min=1)
    valid = count > 0
    per_hand = (torch.exp(-mean_distance/0.03**2)*valid).sum(-1)/valid.sum(-1).clamp(min=1)
    rs += 100*per_hand.mean(-1)
    velocities = [state['robot_dq'][:, wrist]/scale[wrist], state['robot_dq'][:, fingers],
                  state['part_velocity'][..., :3].flatten(1), state['part_velocity'][..., 3:].flatten(1)]
    rp = torch.zeros_like(rg)
    acceleration_dt = dt[:, None] if isinstance(dt, torch.Tensor) and dt.ndim else dt
    for v, before, wv, wa in zip(velocities, previous_velocity, (1, 1, 1, 0.1), (10, 10, 100, 10)):
        rp -= wv * (1-torch.exp(-v.square().mean(-1)))
        rp -= wa * (1-torch.exp(-((v-before)/acceleration_dt).square().mean(-1)/50**2))
    return {'rg': rg*dt, 'rs': rs*dt, 'rp': rp*dt}, velocities


def velocity_groups(state, layout):
    groups = layout['groups']
    wrist = groups['wrist_translation'] + groups['wrist_rotation']
    scale = torch.tensor([layout['wrist_error_scales']['translation_m']]*len(groups['wrist_translation'])
                         + [layout['wrist_error_scales']['rotation_rad']]*len(groups['wrist_rotation']),
                         device=state['robot_dq'].device)
    return [state['robot_dq'][:, wrist]/scale, state['robot_dq'][:, groups['fingers']],
            state['part_velocity'][..., :3].flatten(1), state['part_velocity'][..., 3:].flatten(1)]
