import os

import mujoco
import numpy as np
import pytest
import torch

from contrack_h2r.mjwarp_env import MJWarpEnv
from contrack_h2r.controller import Controller
from contrack_h2r.train import H2RRunner, training_config
from contrack_h2r.inputs import load_prepared, seal_prepared, write_json

pytestmark = pytest.mark.skipif(os.environ.get('CONTRACK_TEST_GPU') != '1', reason='Set CONTRACK_TEST_GPU=1 for GPU integration')


def test_real_contact_body_velocity_and_mask_reset(prepared):
    env = MJWarpEnv(prepared, num_envs=2)
    assert env.state['contact'][:, 0, 0, 0].all()  # actual overlapping spheres, independent of reference
    env.tensor(env.data, 'qvel')[0] = torch.arange(env.cpu_model.nv, device=env.device)*.01
    contacts = (env.state['contact'], env.state['contact_points'])
    state = env._snapshot(env.data, contacts)
    cpu = mujoco.MjData(env.cpu_model)
    cpu.qpos[:] = env.tensor(env.data, 'qpos')[0].cpu().numpy()
    cpu.qvel[:] = env.tensor(env.data, 'qvel')[0].cpu().numpy()
    mujoco.mj_forward(env.cpu_model, cpu)
    for p, bid in enumerate(env.scene.part_ids):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(env.cpu_model, cpu, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
        np.testing.assert_allclose(state['part_velocity'][0, p].cpu(), np.r_[velocity[3:], velocity[:3]], atol=1e-6)
    fields = ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied')
    before = {k: env.tensor(env.data, k)[0].clone() for k in fields}
    env.reset(torch.tensor([1], device=env.device))
    for k in fields:
        torch.testing.assert_close(env.tensor(env.data, k)[0], before[k], atol=0, rtol=0)
    obs, reward, done, extras = env.step(torch.zeros((2, 3), device=env.device))
    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    assert {'rg', 'rs', 'rp', 'time_outs'} <= extras.keys()


def test_ppo_resume_and_complete_controller_parity(prepared, tmp_path):
    env = MJWarpEnv(prepared, num_envs=2)
    cfg = training_config()
    cfg['algorithm'].update(num_learning_epochs=2, num_mini_batches=2)
    output = tmp_path / 'train'
    output.mkdir()
    runner = H2RRunner(env, cfg, output, 0)
    runner.learn(1)
    runner.writer.close()
    checkpoint = output / 'model_0.pt'
    controller = Controller(prepared, checkpoint, 'cuda:0')
    task, layout = env.binding['task'], env.layout
    # Deliberately permute evaluator joints; training and deployment must still agree.
    order = [2, 0, 1]
    hello = dict(type='hello', protocol='h2r-evaluation-v1',
                 **{k: task[k] for k in ('task_id', 'bundle_id', 'robot_manifest_sha256', 'dt_s',
                                          'control_ticks', 'end_tick', 'object', 'observation')},
                 joint_names=[layout['joint_names'][i] for i in order],
                 joint_units=[layout['joint_units'][i] for i in order],
                 episodes=[{'episode_id': str(i)} for i in range(2)])
    controller.start(hello)
    runner.alg.policy.eval()
    for _ in range(3):
        frames = []
        for i in range(2):
            tick = int(env.frame[i])*task['control_ticks']
            # The fixture's training auto-resets are separate from the server lifecycle.
            controller.next_tick[str(i)] = tick
            state = env.state
            records = []
            for body, (hand, slot) in layout['contact_bodies'].items():
                for p, part in enumerate(layout['part_bodies']):
                    if state['contact'][i, hand, p, slot]:
                        records.append({'body1': part, 'body2': body})  # swap pair orientation
            frames.append(dict(episode_id=str(i), tick=tick, time_s=tick*task['dt_s'], done=False,
                               robot_q=state['robot_q'][i, order].tolist(), robot_dq=state['robot_dq'][i, order].tolist(),
                               bodies={b: torch.cat((state['part_pose'][i, p], state['part_velocity'][i, p])).tolist()
                                       for p, b in enumerate(layout['part_bodies'])},
                               contacts={'solve_tick': tick-1 if tick else None, 'records': records}))
        with torch.inference_mode():
            expected = env.reference.targets(runner.alg.policy.act_inference(env.get_observations()), env.clock[env.frame])
        result = controller.step({'type': 'frames', 'episodes': frames, 'done': False})
        actual = torch.tensor([[result['commands'][str(i)]['joint_targets'][n] for n in layout['joint_names']]
                               for i in range(2)], device=env.device)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        env.step(torch.zeros((2, 3), device=env.device))
    resume_dir = tmp_path / 'resume'
    resume_dir.mkdir()
    resumed = H2RRunner(env, cfg, resume_dir, 0)
    resumed.resume(checkpoint)
    assert resumed.current_learning_iteration == 1
    assert resumed.alg.num_updates == 1
    saved = torch.load(checkpoint, weights_only=False)
    torch.testing.assert_close(env.library.track.cpu(), saved['infos']['reset_library']['track'].cpu())
    resumed.learn(1)
    resumed.writer.close()
    assert (resume_dir / 'model_1.pt').exists()


def test_fractional_final_control_interval(prepared):
    binding, arrays = load_prepared(prepared)
    binding['task']['end_tick'] = 22  # two full 8-tick intervals, then exactly 6 ticks
    arrays['time_s'][-1] = 22/240
    np.savez(prepared / 'reference.npz', **arrays)
    write_json(prepared / 'task/task.json', binding['task'])
    seal_prepared(prepared, {k: v for k, v in binding.items() if k not in ('files', 'prepared_id')})
    env = MJWarpEnv(prepared, num_envs=1)
    env.frame[:] = 2
    cpu = mujoco.MjData(env.cpu_model)
    cpu.qpos[:] = env.tensor(env.data, 'qpos')[0].cpu().numpy()
    cpu.qvel[:] = env.tensor(env.data, 'qvel')[0].cpu().numpy()
    for _ in range(6):
        mujoco.mj_step(env.cpu_model, cpu)
    _, _, done, extras = env.step(torch.zeros((1, 3), device=env.device))
    assert done.all() and extras['time_outs'].all()
    terminal_q = env.library.history[0, -1, :env.cpu_model.nq].cpu().numpy()
    np.testing.assert_allclose(terminal_q, cpu.qpos, atol=1e-4, rtol=1e-4)
