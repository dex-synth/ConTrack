"""Independent equation regression against the original Isaac callbacks, without Isaac."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from contrack_h2r.rewards import reward_groups


def test_native_reward_equations_and_dt():
    path = Path(__file__).parents[2] / 'source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/mdp/rewards.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module.startswith('isaaclab'))]
    # Identity object orientations make the inverse transform independently trivial here.
    native = {'quat_apply_inverse': lambda quat, value: value}
    exec(compile(tree, str(path), 'exec'), native)
    torch.manual_seed(3)
    n, p = 4, 2
    q, dq, previous = [torch.randn(n, 58)*.1 for _ in range(3)]
    refq = torch.randn(1, 2, 29)*.1
    pos, refpos = torch.randn(n, p, 3)*.02, torch.zeros(1, p, 3)
    quat = torch.tensor([1., 0, 0, 0]).expand(n, p, 4)
    velocity, before = torch.randn(n, p, 6)*.1, torch.randn(n, p, 6)*.1
    actual = torch.rand(n, 2, p, 10)>.5
    contact = torch.rand(1, 2, p, 10)>.5
    points = torch.randn(n, 2, p, 10, 3)*.01
    refpoints = torch.randn(1, 2, p, 10, 3)*.01
    scene = {'objects': NS(data=NS(object_pos_w=pos, object_quat_w=quat,
                                   object_lin_vel_w=velocity[..., :3], object_ang_vel_w=velocity[..., 3:]))}
    keys = {}
    for h, side in enumerate(('left', 'right')):
        scene[side] = NS(data=NS(joint_pos=q[:, h*29:(h+1)*29], joint_vel=dq[:, h*29:(h+1)*29]))
        keys[side] = []
        for slot in range(10):
            key = f'{side}-{slot}'
            world = points[:, h, :, slot]+pos
            world = torch.where(actual[:, h, :, slot, None], world, torch.nan)
            scene[key] = NS(data=NS(contact_pos_w=world[:, None]))
            keys[side].append(key)
    class Scene(dict):
        env_origins = torch.zeros(n, 3)
    wrist = list(range(7))+list(range(29, 36))
    fingers = list(range(7, 29))+list(range(36, 58))
    env = NS(scene=Scene(scene), frame_idx=torch.zeros(n, dtype=torch.long), ref_len=1,
             ref_hands=[('left', list(range(29)), 0), ('right', list(range(29)), 1)],
             is_rhand=[False, True], ref_qpos=refq, ref_object_pos=refpos, ref_object_quat=quat[:1],
             ref_hand_is_contact=contact.permute(0, 2, 1, 3), ref_hand_contact_points=refpoints.permute(0, 2, 1, 3, 4),
             hand_contact_sensor_keys=keys, num_envs=n, device='cpu', cfg=NS(sim=NS(dt=1/240), decimation=8),
             prev_arm_qvel=previous[:, wrist], prev_finger_qvel=previous[:, fingers],
             prev_object_lin_vel=before[..., :3], prev_object_ang_vel=before[..., 3:])
    state = dict(robot_q=q, robot_dq=dq, part_pose=torch.cat((pos, quat), -1),
                 part_velocity=velocity, contact=actual, contact_points=points)
    ref = dict(robot_q=refq.reshape(1, 58), part_pos=refpos, part_quat=quat[:1], contact=contact, contact_points=refpoints)
    layout = dict(groups={'wrist_translation': wrist[:3], 'wrist_rotation': wrist[3:], 'fingers': fingers},
                  wrist_error_scales={'translation_m': 1, 'rotation_rad': 1})
    out, _ = reward_groups(state, [previous[:, wrist], previous[:, fingers], before[..., :3].flatten(1),
                                   before[..., 3:].flatten(1)], ref, layout, 1/30)
    call = lambda name, **kw: native[name](env, **kw)
    rg = 500*call('tracking_obj_pos', std=.05)+500*call('tracking_obj_rot', std=.5)
    rs = 10*call('tracking_arm')+10*call('tracking_finger')+100*call('contact_reward')+100*call('contact_distance_reward')
    rp = sum(-w*call('qvel_penalty_'+name) for name, w in zip(('arm', 'finger', 'obj_pos', 'obj_rot'), (1, 1, 1, .1)))
    rp += sum(-w*call('qacc_penalty_'+name) for name, w in zip(('arm', 'finger', 'obj_pos', 'obj_rot'), (10, 10, 100, 10)))
    for key, expected in [('rg', rg), ('rs', rs), ('rp', rp)]:
        torch.testing.assert_close(out[key], expected/30, atol=1e-6, rtol=1e-5)


def test_ppo_before_first_completed_episode():
    from tensordict import TensorDict
    from rsl_rl_contrack.algorithms import PPO
    from rsl_rl_contrack.modules import ActorCritic

    obs = TensorDict({'policy': torch.zeros(4, 8)}, batch_size=[4])
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 2,
                         actor_hidden_dims=[16], critic_hidden_dims=[16])
    algorithm = PPO(policy, num_learning_epochs=1, num_mini_batches=1)
    algorithm.init_storage('rl', 4, 4, obs, [2])
    with torch.inference_mode():
        for _ in range(4):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool),
                                       {'rg': torch.ones(4), 'rs': torch.ones(4), 'rp': torch.zeros(4)})
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert not algorithm.vg_buffer and not algorithm.vg_hat_buffer
    assert algorithm.lambda_raw.item() == 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
