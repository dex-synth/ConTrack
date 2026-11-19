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
    binding, layout = env.binding, env.layout
    # The evaluation runs its own 300 Hz step and announces no control period; the client
    # declares the trained 60 Hz one. Deliberately permute evaluator joints as well.
    order = [2, 0, 1]
    dt, end_tick = 1/300, 30
    hello = dict(type='hello', protocol='h2r-evaluation-v2', dt_s=dt, end_tick=end_tick,
                 **{k: binding[k] for k in ('task_id', 'demo_id', 'robot_id')},
                 joint_names=[layout['joint_names'][i] for i in order], observation={'contacts': True},
                 episodes=[{'episode_id': str(i)} for i in range(2)])
    start = controller.start(hello)
    assert start['requirements'] == {'bodies': layout['part_bodies'], 'contacts': True, 'control_dt_s': 1/60}
    assert start['binding']['method'] == 'ConTrack-H2R' and start['task_id'] == binding['task_id']
    assert start['binding']['controller']['completed_iterations'] == 1
    assert controller.control_ticks == 5
    for change, message in [({'protocol': 'h2r-evaluation-v1'}, 'Unsupported evaluator protocol'),
                            ({'task_id': 'other'}, 'different task'), ({'dt_s': 1/100}, 'not a whole number')]:
        with pytest.raises(ValueError, match=message):
            controller.start({**hello, **change})
    runner.alg.policy.eval()
    for _ in range(3):
        frames = []
        for i in range(2):
            tick = int(env.frame[i])*controller.control_ticks
            # The fixture's training auto-resets are separate from the server lifecycle.
            controller.next_tick[str(i)] = tick
            state = env.state
            records = []
            for body, (hand, slot) in layout['contact_bodies'].items():
                for p, part in enumerate(layout['part_bodies']):
                    if state['contact'][i, hand, p, slot]:
                        records.append({'body1': part, 'body2': body})  # swap pair orientation
            done = tick >= end_tick or str(i) in controller.finished
            frames.append(dict(episode_id=str(i), tick=tick, time_s=tick*dt, done=done,
                               robot_q=state['robot_q'][i, order].tolist(), robot_dq=state['robot_dq'][i, order].tolist(),
                               bodies={b: torch.cat((state['part_pose'][i, p], state['part_velocity'][i, p])).tolist()
                                       for p, b in enumerate(layout['part_bodies'])},
                               contacts={'solve_tick': tick-1 if tick else None, 'records': records}))
        with torch.inference_mode():
            expected = env.reference.targets(runner.alg.policy.act_inference(wrapped.get_observations()), env.clock[env.frame])
        active = [i for i, frame in enumerate(frames) if not frame['done']]
        result = controller.step({'type': 'frames', 'episodes': frames, 'done': not active})
        if not active:
            assert result is None  # The private terminal action must not reach bench.
            break
        assert set(result['commands']) == {str(i) for i in active}
        actual = torch.tensor([[result['commands'][str(i)]['joint_targets'][n] for n in layout['joint_names']]
                               for i in active], device=env.device)
        torch.testing.assert_close(actual, expected[active], atol=1e-6, rtol=1e-5)
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
    wrapped.close()


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
    assert graph.unwrapped.h2r_task.timestep_s == 1/300  # the task's nominal step is not inherited
    assert task.audit['timestep_s'] == 1/120 and task.audit['control_dt_s'] == 1/60
    assert graph.max_episode_length == 8  # native raw_len/raw_fps declaration
    assert len(task.clock) == 7  # including the private terminal reference action
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


@pytest.mark.parametrize('use_graph', [False, True])
def test_body_refresh_matches_full_forward_without_resolving_contacts(prepared, use_graph):
    wrapped = make_env(prepared, num_envs=2)
    sim, task = wrapped.unwrapped.sim, wrapped.unwrapped.task_state
    sim.use_cuda_graph = use_graph
    try:
        # Move away from an existing contact and articulate the object. Contact
        # records must stay at the last solve while body frames become current.
        contacts = task._contacts(task.data)
        assert contacts[0][:, 0, 0, 0].all()
        root = task.cpu_model.joint('object_free').qposadr[0]
        qpos, qvel = task.tensor(task.data, 'qpos'), task.tensor(task.data, 'qvel')
        qpos[:, root] += .2
        qpos[:, root + 3:root + 7] = torch.tensor([.8, 0., 0., .6], device=task.device)
        qpos[:, -1] = .4
        qvel[:] = torch.arange(task.cpu_model.nv, device=task.device) * .1
        data_fields = ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'qfrc_constraint', 'nacon')
        before = {name: task.tensor(task.data, name).clone() for name in data_fields}
        contact_before = {name: task.tensor(task.data.contact, name).clone()
                          for name in ('geom', 'pos', 'efc_address')}
        task._refresh_body_state()
        for name, expected in before.items():
            torch.testing.assert_close(task.tensor(task.data, name), expected, atol=0, rtol=0)
        for name, expected in contact_before.items():
            torch.testing.assert_close(task.tensor(task.data.contact, name), expected, atol=0, rtol=0)
        actual = task._snapshot(task.data, contacts)
        sim.forward()
        expected = task._snapshot(task.data, contacts)
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-6)
        assert not task._contacts(task.data)[0][:, 0, 0, 0].any()
    finally:
        wrapped.close()


def test_one_full_forward_keeps_multistep_rewards_contacts_and_resets(prepared, monkeypatch):
    old = make_env(prepared, num_envs=2, seed=11)
    new = make_env(prepared, num_envs=2, seed=11)
    # Reconstruct the previous extra full-forward path as the comparison oracle.
    monkeypatch.setattr(old.unwrapped.task_state, '_refresh_body_state', old.unwrapped.sim.forward)
    forwards = []
    full_forward = new.unwrapped.sim.forward

    def count_forward():
        forwards.append(new.unwrapped.common_step_counter)
        full_forward()

    monkeypatch.setattr(new.unwrapped.sim, 'forward', count_forward)
    resets = 0
    try:
        for step in range(16):
            action = torch.full((2, 3), .1, device=new.device)
            torch.manual_seed(100 + step)
            expected = old.step(action)
            torch.manual_seed(100 + step)
            actual = new.step(action)
            assert len(forwards) == step + 1
            torch.testing.assert_close(actual[0]['policy'], expected[0]['policy'], atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=1e-4)
            assert torch.equal(actual[2], expected[2])
            resets += int(actual[2].sum())
            for name in ('rg', 'rs', 'rp'):
                torch.testing.assert_close(actual[3][name], expected[3][name], atol=2e-5, rtol=1e-4)
            a, b = new.unwrapped.task_state, old.unwrapped.task_state
            for name in a.state:
                torch.testing.assert_close(a.state[name], b.state[name], atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(a.tensor(a.data, 'qacc_warmstart'),
                                       b.tensor(b.data, 'qacc_warmstart'), atol=2e-5, rtol=1e-4)
        assert resets > 0
    finally:
        old.close()
        new.close()


def test_native_rollout_starts_at_first_frame_with_frozen_library(prepared, tmp_path):
    from contrack_h2r.native_rollout import rollout
    wrapped = make_env(prepared, num_envs=2)
    output = tmp_path / 'train'
    output.mkdir()
    cfg = training_config()
    cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
    runner = H2RRunner(wrapped, cfg, output, 0)
    runner.learn(1)
    runner.writer.close()
    wrapped.close()
    checkpoint = output / 'model_0.pt'
    saved = torch.load(checkpoint, weights_only=False)['infos']['reset_library']
    for velocity in ('reference', 'zero'):
        summary, arrays = rollout(prepared, checkpoint, num_envs=2, initial_velocity=velocity)
        assert summary['checkpoint']['completed_iterations'] == 1
        frames = arrays['frame']
        assert 'physics_qpos' not in arrays  # Only the method rollout export asks for physics-step states.
        assert (frames[:, 0] == np.arange(len(frames))).all()  # Fixed clock from the first frame.
        world = summary['worlds'][0]
        assert world['reason'] in ('pose_break', 'reference_end') and world['end_frame'] == len(frames) - 1
        ema = float(saved['success_ema'])
        weight = min(max((ema - 0.9) / (0.999 - 0.9), 0), 1)
        assert summary['termination']['threshold_m'] == pytest.approx(0.1 - 0.05 * weight)


def test_method_rollout_exports_every_physics_step_from_the_evaluation_start(prepared, tmp_path, monkeypatch):
    from contrack_h2r import method_rollout
    from h2r_io.method_rollout import CONDITIONS, load_method_rollout
    wrapped = make_env(prepared, num_envs=2)
    output = tmp_path / 'train'
    output.mkdir()
    cfg = training_config()
    cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
    runner = H2RRunner(wrapped, cfg, output, 0)
    runner.learn(1)
    runner.writer.close()
    wrapped.close()
    checkpoint = output / 'model_0.pt'
    native_rollout, native = method_rollout.rollout, {}

    def rollout(*args, **kwargs):
        native['summary'], native['arrays'] = result = native_rollout(*args, **kwargs)
        return result

    monkeypatch.setattr(method_rollout, 'rollout', rollout)
    package = method_rollout.export(prepared, checkpoint, tmp_path / 'package', num_envs=2)
    controller = Controller(prepared, checkpoint)
    task = method_rollout.prepared_task(prepared, controller.binding)
    manifest, episodes = load_method_rollout(package, task)
    hello = dict(type='hello', protocol='h2r-evaluation-v2', dt_s=1/300, end_tick=30, task_id=task.task_id,
                 demo_id=task.demo.demo_id, robot_id=task.robot.robot_id, joint_names=manifest['joint_names'],
                 observation={'contacts': True}, episodes=[{'episode_id': '0'}])
    start = controller.start(hello)
    # The same method identity and deployed checkpoint as the evaluation client reports.
    assert manifest['binding'] == start['binding']
    assert manifest['conditions'] == CONDITIONS and manifest['training_seed'] == 0
    simulator = manifest['simulator']
    assert (simulator['physics_dt_s'], simulator['control_dt_s'], simulator['deterministic']) == (1/120, 1/60, False)
    steps = 12  # 0.1 s of task at 120 Hz; the native terminal action lies beyond it
    assert len(episodes) == 2
    for i, (states, entry, world) in enumerate(zip(episodes, manifest['episodes'], native['summary']['worlds'])):
        verdict, count = entry['verdict'], len(states['time_s']) - 1
        np.testing.assert_allclose(states['time_s'], np.arange(count + 1) / 120)
        assert count == min(2 * (world['end_frame'] + 1), steps)
        assert verdict['ended_at_s'] == (count / 120 if count < steps else None)
        assert verdict['success'] == (world['reason'] == 'reference_end')
        # The evaluation initial state: the pose proposed in start, objects at the first reference frame.
        proposed = start['initial_robot']['0']
        np.testing.assert_allclose(states['robot_q'][0], [proposed[n] for n in manifest['joint_names']], atol=1e-6)
        np.testing.assert_allclose(states['object_root_poses'][0, 0],
                                   task.initial_object_state['object_roots']['box'], atol=1e-6)
        np.testing.assert_allclose(states['object_q'][0], [0], atol=1e-6)
        # Every second physics state is the state the native rollout scored at that control step.
        np.testing.assert_array_equal(states['robot_q'][2::2], native['arrays']['robot_q'][:count // 2, i])
