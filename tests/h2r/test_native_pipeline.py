"""Behavior checks against the pinned, unmodified public Sharpa callbacks."""

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from contrack_h2r.reference import Reference
from contrack_h2r.resets import ResetLibrary


ROOT = Path(__file__).parents[2]
NATIVE_STATE = 'source/ConTrack/ConTrack/tasks/manager_based/mimic_rby1a_sharpa/mdp/state.py'
UPSTREAM = 'f5efe18e4b1f986566b252265c4bf63f3b67c6c6'


def native_callbacks(source=None):
    if source is None:
        source = subprocess.check_output(['git', 'show', f'{UPSTREAM}:{NATIVE_STATE}'], cwd=ROOT, text=True)
    tree = ast.parse(source)
    functions = {'prepare_reference_state', 'reset_reference_episode', 'advance_reference_frame'}
    namespace = dict(torch=torch, RBY1A_RIGHT_JOINT_NAMES=list(range(7)),
                     SHARPA_RIGHT_JOINT_NAMES=list(range(7, 29)), RBY1A_LEFT_JOINT_NAMES=[],
                     SHARPA_LEFT_JOINT_NAMES=[], DEFAULT_RBY1A_QPOS=[0.] * 7)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'DELTA', 'EMA_ALPHA', 'HARD_RESET_TOPK', 'EPSILON_GREEDY', 'TAU', '_HAND_CONTACT_ORDER'}:
                namespace[name] = ast.literal_eval(node.value)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    exec(compile(tree, NATIVE_STATE, 'exec'), namespace)
    return namespace


def source_reference(length=3):
    torch.manual_seed(17)
    q = torch.randn(length, 1, 29) * .01
    quat = torch.nn.functional.normalize(torch.randn(length, 1, 4), dim=-1)
    return dict(qpos=q, object_pos=torch.randn(length, 1, 3) * .01, object_quat_wxyz=quat,
                fps=torch.tensor([30.]), urdf_name=['sharpa'], is_rhand=torch.tensor([True]),
                hand_is_contact=torch.arange(length)[:, None, None, None].expand(-1, 1, 1, 10) % 2,
                hand_contact_points=torch.randn(length, 1, 1, 10, 3) * .01)


def make_native(source, num_envs=3, callbacks=None):
    callbacks = native_callbacks() if callbacks is None else callbacks
    class Scene(dict):
        env_origins = torch.zeros(num_envs, 3)
    hand = NS(data=NS(joint_pos=torch.zeros(num_envs, 29), joint_vel=torch.zeros(num_envs, 29)),
              find_joints=lambda names, **kw: (names, names), _ALL_INDICES=torch.arange(num_envs))
    def write_joints(q, v, joint_ids, env_ids):
        hand.data.joint_pos[env_ids[:, None], joint_ids] = q
        hand.data.joint_vel[env_ids[:, None], joint_ids] = v
    hand.write_joint_state_to_sim = write_joints
    objects = NS(data=NS(object_pos_w=torch.zeros(num_envs, 1, 3),
                         object_quat_w=torch.tensor([1., 0, 0, 0]).expand(num_envs, 1, 4).clone(),
                         object_lin_vel_w=torch.zeros(num_envs, 1, 3), object_ang_vel_w=torch.zeros(num_envs, 1, 3)))
    def write_objects(state, env_ids):
        for name, section in [('object_pos_w', slice(0, 3)), ('object_quat_w', slice(3, 7)),
                              ('object_lin_vel_w', slice(7, 10)), ('object_ang_vel_w', slice(10, 13))]:
            getattr(objects.data, name)[env_ids] = state[..., section]
    objects.write_object_state_to_sim = write_objects
    cfg = NS(reference_path='synthetic', sim=NS(dt=1/120), decimation=2, threshold_curriculum_enabled=True,
             object_pos_threshold_max=.1, object_rot_threshold_max=1., object_pos_threshold_min=.05,
             object_rot_threshold_min=.5, threshold_curriculum_success_ema_alpha=.05,
             threshold_curriculum_sr_low=.9, threshold_curriculum_sr_high=.999)
    env = NS(device='cpu', num_envs=num_envs, cfg=cfg, scene=Scene(rby1a_sharpa=hand, objects=objects),
             reset_buf=torch.zeros(num_envs, dtype=torch.bool))
    callbacks['reference_tensors'] = lambda *args: source
    callbacks['prepare_reference_state'](env, None)
    return env, callbacks


def h2r_reference(source):
    # The original-arm oracle uses its original scales, not the H2R wrist defaults.
    layout = dict(joint_names=list(range(29)), groups={'wrist_translation': list(range(3)),
                  'wrist_rotation': list(range(3, 7)), 'fingers': list(range(7, 29))},
                  action_scales={'wrist_translation': .01, 'wrist_rotation': .01, 'fingers': .25},
                  wrist_error_scales={'translation_m': 1., 'rotation_rad': 1.})
    arrays = dict(time_s=np.arange(len(source['qpos']))/30, robot_q=source['qpos'].flatten(1),
                  part_pos=source['object_pos'], part_quat=source['object_quat_wxyz'],
                  contact=source['hand_is_contact'].transpose(1, 2),
                  contact_points=source['hand_contact_points'].transpose(1, 2))
    return Reference(layout, arrays, 'cpu', 1/60)


def test_reference_matches_native_control_grid():
    source = source_reference()
    native, _ = make_native(source)
    reference = h2r_reference(source)
    time = torch.arange(native.ref_len)/native.ref_fps
    result = reference.sample(time)
    for key, expected in [('robot_q', native.ref_qpos.flatten(1)), ('part_pos', native.ref_object_pos),
                          ('part_quat', native.ref_object_quat),
                          ('contact', native.ref_hand_is_contact.transpose(1, 2)),
                          ('contact_points', native.ref_hand_contact_points.transpose(1, 2))]:
        torch.testing.assert_close(result[key].float(), expected.float(), atol=1e-7, rtol=1e-6)


def test_reset_sampling_matches_native_rng():
    native, callbacks = make_native(source_reference(160), num_envs=12)
    lib = ResetLibrary(native.reset_lib_state.clone(), torch.empty(native.ref_len, 0), 12)
    lib.track[:] = torch.linspace(0, 100, lib.length)
    native.reset_lib_track_len.copy_(lib.track)
    ids = torch.arange(12)
    torch.manual_seed(23)
    callbacks['reset_reference_episode'](native, ids)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(23)
    start, state = lib.sample(ids)
    torch.testing.assert_close(start, native.frame_idx)
    torch.testing.assert_close(state, native.reset_lib_state[native.frame_idx])
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_native_empty_return_window_is_not_silently_fixed():
    from rsl_rl_contrack.algorithms import PPO
    from rsl_rl_contrack.modules import ActorCritic
    from tensordict import TensorDict
    obs = TensorDict({'policy': torch.zeros(2, 4)}, batch_size=[2])
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 2,
                         actor_hidden_dims=[8], critic_hidden_dims=[8])
    ppo = PPO(policy)
    with pytest.raises(ZeroDivisionError):
        ppo.update()


def test_extracted_native_reference_and_reset_callbacks():
    from rsl_rl_contrack import mimic
    source = source_reference(90)
    expected, original = make_native(source)
    callbacks = native_callbacks((ROOT / NATIVE_STATE).read_text())
    callbacks.update({name: getattr(mimic, name) for name in
                      ('resample_reference', 'linear_velocity', 'angular_velocity', 'sample_start_frames')})
    actual, extracted = make_native(source, callbacks=callbacks)
    for name in ('ref_qpos', 'ref_object_pos', 'ref_object_quat', 'ref_hand_is_contact',
                 'ref_hand_contact_points', 'reset_lib_state'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
    for seed in (3, 17):
        torch.manual_seed(seed)
        original['reset_reference_episode'](expected, None)
        rng = torch.get_rng_state()
        torch.manual_seed(seed)
        extracted['reset_reference_episode'](actual, None)
        assert torch.equal(torch.get_rng_state(), rng)
        torch.testing.assert_close(actual.frame_idx, expected.frame_idx)
        torch.testing.assert_close(actual.reset_lib_state, expected.reset_lib_state)


def test_reset_velocity_coordinates(prepared):
    import mujoco
    from scipy.spatial.transform import Rotation
    from contrack_h2r.inputs import load_prepared
    from contrack_h2r.reset_reference import reset_reference
    from rsl_rl_contrack.mimic import angular_velocity, linear_velocity

    binding, arrays = load_prepared(prepared)
    model = mujoco.MjModel.from_xml_path(str(prepared / 'task/scene.xml'))
    model.body_ipos[model.body('object').id] = [.02, -.01, .03]
    data = mujoco.MjData(model)
    arrays['robot_q'][:, 0] = [0, .01, 0, -.005]
    arrays['qpos'][:, 0] = arrays['robot_q'][:, 0]
    root_q = model.joint('object_free').qposadr[0]
    rotation = Rotation.from_euler('xyz', [[.2, .1, .3], [.22, .12, .32], [.24, .14, .34], [.26, .16, .36]])
    arrays['qpos'][:, root_q+3:root_q+7] = rotation.as_quat(scalar_first=True)
    arrays['qpos'][:, root_q] += np.array([0, .01, 0, -.005])
    hinge_q, hinge_v = model.joint('hinge').qposadr[0], model.joint('hinge').dofadr[0]
    arrays['qpos'][:, hinge_q] = [0, .01, .02, -.01]
    for frame, q in enumerate(arrays['qpos']):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for p, name in enumerate(binding['layout']['part_bodies']):
            arrays['part_pos'][frame, p] = data.body(name).xpos
            arrays['part_quat'][frame, p] = data.body(name).xquat
    qpos, qvel, report = reset_reference(model, binding['layout'], arrays)
    ref = Reference(binding['layout'], arrays, 'cpu', 1/60)
    expected_joint = linear_velocity(ref.data['robot_q'], 60).numpy()
    np.testing.assert_allclose(qvel[:, :3], expected_joint, atol=1e-7)
    np.testing.assert_allclose(qvel[:3, 0], [.3, .3, 0], atol=1e-7)
    expected_hinge = linear_velocity(torch.tensor(qpos[:, hinge_q]).float(), 60).numpy()
    np.testing.assert_allclose(qvel[:, hinge_v], expected_hinge, atol=1e-7)
    expected_linear = linear_velocity(ref.data['part_pos'], 60).numpy()
    expected_angular = angular_velocity(ref.data['part_quat'], 60).numpy()
    for frame, (q, v) in enumerate(zip(qpos, qvel)):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        body = model.body('object').id
        actual = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, actual, 0)
        origin_velocity = actual[3:] + np.cross(actual[:3], data.xpos[body] - data.xipos[body])
        np.testing.assert_allclose(origin_velocity, expected_linear[frame, 0], atol=1e-7)
        np.testing.assert_allclose(actual[:3], expected_angular[frame, 0], atol=1e-7)
    assert np.isfinite(list(v for k, v in report.items() if k.startswith('max_'))).all()


def synthetic_task(source, native):
    """Drive real H2R task callbacks using a physical-state pack in place of MJWarp."""
    from contrack_h2r.mjlab_env import TaskState
    from contrack_h2r.rewards import velocity_groups
    n = native.num_envs
    task = TaskState.__new__(TaskState)
    task.env = NS(common_step_counter=0, sim=NS(forward=lambda: None), extras={'log': {}})
    task.device, task.num_envs = 'cpu', n
    task.reference = h2r_reference(source)
    task.layout, task.clock = task.reference.layout, task.reference.time
    task.step_dt, task.max_episode_length = 1/60, native.ref_len + 1
    task.frame = torch.zeros(n, dtype=torch.long)
    task.cpu_model = NS(nq=71)
    task.qids = task.aids = torch.arange(29)
    task.data = NS(qpos=torch.zeros(n, 71), qvel=torch.empty(n, 0), ctrl=torch.zeros(n, 29))
    task.tensor = lambda data, name: getattr(data, name)
    task._contacts = lambda data: None
    task._refresh_body_state = lambda: None
    def snapshot(data, contacts):
        return dict(robot_q=data.qpos[:, :29].clone(), robot_dq=data.qpos[:, 29:58].clone(),
                    part_pose=data.qpos[:, 58:65, None].transpose(1, 2).clone(),
                    part_velocity=data.qpos[:, 65:71, None].transpose(1, 2).clone(),
                    contact=torch.zeros(n, 1, 1, 10, dtype=torch.bool), contact_points=torch.zeros(n, 1, 1, 10, 3))
    task._snapshot = snapshot
    task.state = snapshot(task.data, None)
    task.previous_velocity = [v.clone() for v in velocity_groups(task.state, task.layout)]
    task.library = ResetLibrary(native.reset_lib_state.clone(), torch.empty(native.ref_len, 0), n)
    task.pending_reset = torch.ones(n, dtype=torch.bool)
    task.capacity_overflow = torch.zeros(n, dtype=torch.bool)
    task.it_obj_pos_err_sum = torch.tensor(0.)
    task.it_obj_rot_err_sum = torch.tensor(0.)
    task.it_step_count, task.finished_step = 0, -1
    return task


def test_multistep_actions_termination_history_and_curriculum():
    source = source_reference(5)
    native, callbacks = make_native(source)
    task = synthetic_task(source, native)
    ids = torch.arange(native.num_envs)
    torch.manual_seed(7)
    callbacks['reset_reference_episode'](native, ids)
    torch.manual_seed(7)
    task.reset_reference(ids)
    task.observe()
    assert set(native.frame_idx.tolist()) != {0}  # include random mid-clip starts

    path = NATIVE_STATE.replace('state.py', 'actions.py')
    tree = ast.parse(subprocess.check_output(['git', 'show', f'{UPSTREAM}:{path}'], cwd=ROOT, text=True))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Rby1aSharpaResidualAction')
    process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_actions')
    action_namespace = {'torch': torch}
    exec(compile(ast.Module(body=[process], type_ignores=[]), path, 'exec'), action_namespace)
    action_term = NS(_env=native, _hands=[(None, list(range(29)), 0)], _sizes=[29],
                     _action_scales=[torch.tensor([.01]*7 + [.25]*22)],
                     _processed_actions=torch.zeros(native.num_envs, 29), _targets=[None])
    total_timeouts = 0
    for step in range(24):
        frame = native.frame_idx.clone()
        action = torch.linspace(-2, 2, 29).expand(native.num_envs, -1)
        action_namespace['process_actions'](action_term, action)
        torch.testing.assert_close(task.reference.targets(action, task.clock[task.frame]), action_term._targets[0])

        # Prescribe the same post-action physical state to each implementation.
        physical = native.reset_lib_state[frame].clone()
        physical[:, :29] += .001 * (step + 1)
        physical[:, 29:58] += .01 * (step + 1)
        physical[:, 58:61] = native.ref_object_pos[frame, 0]
        physical[:, 61:65] = native.ref_object_quat[frame, 0]
        if step % 4 == 1:
            physical[0, 58] += .2  # early failure, including near the endpoint
        hand = native.scene['rby1a_sharpa']
        hand.write_joint_state_to_sim(physical[:, :29], physical[:, 29:58], list(range(29)), ids)
        native.scene['objects'].write_object_state_to_sim(physical[:, 58:, None].transpose(1, 2), ids)
        position = torch.linalg.vector_norm(physical[:, 58:61] - native.ref_object_pos[frame, 0], dim=-1)
        failed = position > native.object_pos_threshold
        timeout = frame >= native.ref_len - 1
        native.reset_buf = failed | timeout
        old_velocity = [v.clone() for v in task.previous_velocity]
        task.data.qpos.copy_(physical)
        task.env.common_step_counter += 1
        task.finish_step()
        torch.testing.assert_close(task.failed, failed)
        torch.testing.assert_close(task.timeout, timeout)
        for actual, before in zip(task.previous_velocity, old_velocity):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)
        done_ids = native.reset_buf.nonzero().flatten()
        if len(done_ids):
            torch.manual_seed(100 + step)
            callbacks['reset_reference_episode'](native, done_ids)
            rng = torch.get_rng_state()
            torch.manual_seed(100 + step)
            task.reset_reference(done_ids)
            assert torch.equal(torch.get_rng_state(), rng)
        callbacks['advance_reference_frame'](native, None)
        task.observe()
        torch.testing.assert_close(task.frame, native.frame_idx)
        for actual, name in [(task.library.state, 'reset_lib_state'), (task.library.track, 'reset_lib_track_len'),
                             (task.library.best, 'reset_lib_track_len_best'),
                             (task.library.success_ema, 'threshold_curriculum_success_ema')]:
            torch.testing.assert_close(actual, getattr(native, name), atol=2e-6, rtol=1e-6)
        for actual, name in zip(task.previous_velocity, ('prev_arm_qvel', 'prev_finger_qvel',
                                                        'prev_object_lin_vel', 'prev_object_ang_vel')):
            torch.testing.assert_close(actual, getattr(native, name).flatten(1))
        pt, rt = task.library.thresholds()
        torch.testing.assert_close(pt.expand(native.num_envs), native.object_pos_threshold)
        torch.testing.assert_close(rt.expand(native.num_envs), native.object_rot_threshold)
        total_timeouts += int(timeout.sum())
    assert total_timeouts > 0


def test_public_recipe_defaults_and_non_native_clock_rejected(prepared):
    import inspect
    from contrack_h2r.mjlab_env import ConTrackEnv
    from contrack_h2r.train import argument_parser, training_config
    from rsl_rl_contrack.sharpa_recipe import MAX_ITERATIONS, NUM_ENVS, TRAINING_BUDGET
    args = argument_parser().parse_args(['--prepared', str(prepared), '--output', 'unused'])
    assert args.num_envs == inspect.signature(ConTrackEnv).parameters['num_envs'].default == NUM_ENVS == 6000
    assert args.iterations == MAX_ITERATIONS == TRAINING_BUDGET['default_iterations'] == 5000
    cfg = training_config()
    assert cfg['save_interval'] == 200 and cfg['num_steps_per_env'] * args.num_envs == 96000
    with pytest.raises(ValueError, match='60 Hz'):
        ConTrackEnv(prepared, control_hz=30)


def test_legacy_artifacts_are_not_reinterpreted(prepared, tmp_path):
    from contrack_h2r.controller import Controller
    from contrack_h2r.inputs import digest, load_prepared, seal_prepared, write_json
    checkpoint = tmp_path / 'old.pt'
    metadata = {'format': 'contrack-h2r-deployment-v3'}
    torch.save({'infos': {'h2r': metadata}}, checkpoint)
    write_json(checkpoint.with_suffix('.json'), {**metadata, 'checkpoint_sha256': digest(checkpoint)})
    with pytest.raises(ValueError, match='Unsupported checkpoint .contrack-h2r-deployment-v3.: retrain'):
        Controller(prepared, checkpoint)
    binding, _ = load_prepared(prepared)
    binding['format'] = 'contrack-h2r-v2'
    seal_prepared(prepared, {k: v for k, v in binding.items() if k not in ('files', 'prepared_id')})
    with pytest.raises(ValueError, match="Unsupported preparation 'contrack-h2r-v2': rerun contrack_h2r.prepare"):
        load_prepared(prepared)
