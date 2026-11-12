"""Named Sharpa layout, shared by preparation, training and deployment."""

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
# MANO's 16 joints followed by thumb/index/middle/ring/little tips.
CHAINS = ((13, 14, 15, 16), (1, 2, 3, 17), (4, 5, 6, 18),
          (10, 11, 12, 19), (7, 8, 9, 20))


def make_layout(task, parts):
    if task['robot_name'] not in ('virtual_wrist_sharpa_wave_bimanual', 'virtual_wrist_sharpa_wave_right'):
        raise ValueError('This adaptation supports only the common virtual-wrist Sharpa robots')
    semantics = task['semantics']
    hands = [side for side in ('left', 'right') if side in semantics['hands']]
    names = []
    for side in hands:
        names.extend(f'{side}_{axis}_joint' for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'))
        fingers = semantics['hands'][side]['finger_joints']
        if len(fingers) != 22:
            raise ValueError('Expected 22 Sharpa finger joints')
        names.extend(fingers)
    if len(set(names)) != len(names) or set(names) != set(semantics['joint_names']):
        raise ValueError('Incomplete controlled robot layout')
    groups = {k: [names.index(n) for n in v] for k, v in semantics['joint_groups'].items()}
    slots = {}
    for h, side in enumerate(hands):
        for f, finger in enumerate(FINGERS):
            # Fixed pads inherit the distal slot, but articulated middle links do not.
            for suffix, slot in [('PP', 2*f), ('DP', 2*f+1), ('elastomer', 2*f+1), ('fingertip', 2*f+1)]:
                body = f'{side}_{finger}_{suffix}'
                if body not in semantics['body_names']:
                    raise ValueError(f'Missing Sharpa contact body: {body}')
                slots[body] = [h, slot]
    return dict(hands=hands, joint_names=names, groups=groups,
                joint_units=[semantics['joint_units'][semantics['joint_names'].index(n)] for n in names],
                parts=list(parts), part_bodies=[task['object']['parts'][p] for p in parts],
                contact_bodies=slots, contact_order='hand,part,finger(PP,DP)',
                # Match LeanRetarget's Sharpa per-action wrist ranges; policy remains 60 Hz.
                action_scales={'wrist_translation': 0.06666666666666667,
                               'wrist_rotation': 0.21333333333333335, 'fingers': 0.25},
                wrist_error_scales={'translation_m': 1.0, 'rotation_rad': 1.0})
