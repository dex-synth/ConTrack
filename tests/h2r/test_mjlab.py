import os

import mujoco
import numpy as np
import pytest
import torch

from contrack_h2r.mjlab_env import ConTrackEnv, capacity_health, make_env
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from contrack_h2r.controller import Controller
from contrack_h2r.train import H2RRunner, training_config
from contrack_h2r.inputs import load_prepared, seal_prepared, write_json

pytestmark = pytest.mark.skipif(os.environ.get('CONTRACK_TEST_GPU') != '1', reason='Set CONTRACK_TEST_GPU=1 for GPU integration')


def test_real_contact_body_velocity_and_mask_reset(prepared):
    wrapped = make_env(prepared, num_envs=2)
    env = wrapped.unwrapped.task_state
    assert env.state['contact'][:, 0, 0, 0].all()  # actual overlapping spheres, independent of reference
    env.tensor(env.data, 'qvel')[0] = torch.arange(env.cpu_model.nv, device=env.device)*.01
    wrapped.unwrapped.sim.forward()
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
    wrapped.unwrapped.reset(env_ids=torch.tensor([1], device=env.device))
    for k in fields:
        torch.testing.assert_close(env.tensor(env.data, k)[0], before[k], atol=0, rtol=0)
    obs, reward, done, extras = wrapped.step(torch.zeros((2, 3), device=env.device))
    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    assert {'rg', 'rs', 'rp', 'time_outs'} <= extras.keys()


def test_ppo_resume_and_complete_controller_parity(prepared, tmp_path):
    binding, arrays = load_prepared(prepared)
    binding['task']['control_ticks'] = 4
    write_json(prepared / 'task/task.json', binding['task'])
    seal_prepared(prepared, {k: v for k, v in binding.items() if k not in ('files', 'prepared_id')})
    wrapped = make_env(prepared, num_envs=2)
    env = wrapped.unwrapped.task_state
    cfg = training_config()
    cfg['algorithm'].update(num_learning_epochs=2, num_mini_batches=2)
    output = tmp_path / 'train'
    output.mkdir()
    runner = H2RRunner(wrapped, cfg, output, 0)
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
            expected = env.reference.targets(runner.alg.policy.act_inference(wrapped.get_observations()), env.clock[env.frame])
        result = controller.step({'type': 'frames', 'episodes': frames, 'done': False})
        actual = torch.tensor([[result['commands'][str(i)]['joint_targets'][n] for n in layout['joint_names']]
                               for i in range(2)], device=env.device)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        wrapped.step(torch.zeros((2, 3), device=env.device))
    resume_dir = tmp_path / 'resume'
    resume_dir.mkdir()
    resumed = H2RRunner(wrapped, cfg, resume_dir, 0)
    resumed.resume(checkpoint)
    assert resumed.current_learning_iteration == 1
    assert resumed.alg.num_updates == 1
    saved = torch.load(checkpoint, weights_only=False)
    torch.testing.assert_close(env.library.track.cpu(), saved['infos']['reset_library']['track'].cpu())
    resumed.learn(1)
    resumed.writer.close()
    assert (resume_dir / 'model_1.pt').exists()


def test_incomplete_training_interval_is_rejected(prepared):
    binding, arrays = load_prepared(prepared)
    binding['task']['end_tick'] = 22
    arrays['time_s'][-1] = 22/240
    np.savez(prepared / 'reference.npz', **arrays)
    write_json(prepared / 'task/task.json', binding['task'])
    seal_prepared(prepared, {k: v for k, v in binding.items() if k not in ('files', 'prepared_id')})
    with pytest.raises(ValueError, match='complete training control intervals'):
        make_env(prepared, num_envs=1)


def test_native_clock_and_mjlab_graph_parity(prepared):
    eager = make_env(prepared, num_envs=2, seed=4)
    eager.unwrapped.sim.use_cuda_graph = False
    graph = make_env(prepared, num_envs=2, seed=4)
    task = graph.unwrapped.task_state
    assert isinstance(graph, RslRlVecEnvWrapper)
    assert ConTrackEnv.step is ManagerBasedRlEnv.step
    assert ConTrackEnv.reset is ManagerBasedRlEnv.reset
    assert graph.unwrapped.sim.use_cuda_graph
    assert task.cpu_model.opt.timestep == 1/120
    assert task.substeps == 2 and task.reference.dt == 1/60
    assert task.binding['task']['dt_s'] == 1/240
    assert task.binding['task']['control_ticks'] == 8
    assert graph.max_episode_length == 6
    for i in range(8):
        action = torch.full((2, 3), .1, device=graph.device)
        torch.manual_seed(i)
        a = eager.step(action)
        torch.manual_seed(i)
        b = graph.step(action)
        torch.testing.assert_close(a[0]['policy'], b[0]['policy'], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(a[1], b[1], atol=2e-5, rtol=1e-4)
        assert torch.equal(a[2], b[2])


def test_transient_substep_overflow_is_not_lost(prepared):
    wrapped = make_env(prepared, num_envs=2)
    env = wrapped.unwrapped
    task = env.task_state
    task.tensor(task.data, 'overflow')[1] = 1
    capacity_health(env)
    task.tensor(task.data, 'overflow').zero_()
    with pytest.raises(RuntimeError, match='during a physics substep'):
        task.finish_step()
