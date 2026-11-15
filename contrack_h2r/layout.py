"""Named robot layout, shared by preparation, training and deployment."""

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
# MANO's 16 joints followed by thumb/index/middle/ring/little tips.
CHAINS = ((13, 14, 15, 16), (1, 2, 3, 17), (4, 5, 6, 18),
          (10, 11, 12, 19), (7, 8, 9, 20))
# Robots with a fixed IK and contact mapping: link suffix -> slot within a finger (proximal, distal).
# Fixed pads inherit the distal slot, but articulated middle links do not.
CONTACT_LINKS = {'sharpa_wave': (('PP', 0), ('DP', 1), ('elastomer', 1), ('fingertip', 1))}


def make_layout(task):
    robot, semantics = task.robot, task.robot.semantics
    if robot.name not in CONTACT_LINKS:
        raise ValueError(f'ConTrack has an IK and contact mapping only for {sorted(CONTACT_LINKS)}, not {robot.name!r}')
    hands = [side for side in ('left', 'right') if side in robot.hands]
    names = []
    for side in hands:
        hand = semantics['hands'][side]
        names.extend(hand['wrist_translation_joints'] + hand['wrist_rotation_joints'] + hand['finger_joints'])
    if len(set(names)) != len(names) or set(names) != set(semantics['joint_names']):
        raise ValueError('Incomplete controlled robot layout')
    groups = {k: [names.index(n) for n in v] for k, v in semantics['joint_groups'].items()}
    slots = {}
    for h, side in enumerate(hands):
        for f, finger in enumerate(FINGERS):
            for suffix, slot in CONTACT_LINKS[robot.name]:
                body = f'{side}_{finger}_{suffix}'
                if body not in semantics['body_names']:
                    raise ValueError(f'Missing contact body: {body}')
                slots[body] = [h, 2*f + slot]
    parts = list(task.demo.part_order)
    return dict(hands=hands, joint_names=names, groups=groups,
                joint_units=[semantics['joint_units'][semantics['joint_names'].index(n)] for n in names],
                parts=parts, part_bodies=[task.object_binding['parts'][p] for p in parts],
                contact_bodies=slots, contact_order='hand,part,finger(PP,DP)',
                # Match LeanRetarget's Sharpa per-action wrist ranges; policy remains 60 Hz.
                action_scales={'wrist_translation': 0.06666666666666667,
                               'wrist_rotation': 0.21333333333333335, 'fingers': 0.25},
                wrist_error_scales={'translation_m': 1.0, 'rotation_rad': 1.0})
