"""ConTrack task terms on MJLab's environment and official RSL-RL wrapper."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_warp
import numpy as np
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from .inputs import load_prepared
from .mjwarp_scene import Scene
from .reference import Reference
from .resets import ResetLibrary
from .rewards import pose_error, reward_groups, velocity_groups


class TaskState:
    """Method reference, reward and adaptive-reset state; no simulation step loop."""

    def __init__(self, env, prepared, binding, arrays):
        self.env, self.binding = env, binding
        self.layout, task = binding['layout'], binding['task']
        self.device, self.num_envs = env.device, env.num_envs
        self.scene = Scene(Path(prepared) / 'task' / task['scene'], task, self.layout)
        self.cpu_model = m = env.sim.mj_model
        self.data, self.model = env.sim.wp_data, env.sim.wp_model
        self.ctrl = env.sim.data.ctrl  # Create the shared bridge outside PPO's inference-mode scope.
        self.dt, self.substeps = env.physics_dt, env.cfg.decimation
        self.step_dt = env.step_dt
        checked = []
        for name, values in self.scene.audit().items():
            expected = np.asarray(values)
            actual = getattr(m, name)
            gpu = getattr(self.model, name).numpy().reshape(expected.shape)
            if (not np.allclose(actual, expected, atol=1e-7, rtol=1e-5)
                    or not np.allclose(gpu, expected, atol=1e-7, rtol=1e-5)):
                raise ValueError(f'MJLab changed the shared asset field: {name}')
            checked.append(name)
        self.audit = dict(backend='mujoco_warp', environment='mjlab', mjlab=version('mjlab'),
                          mujoco=mujoco.__version__, mujoco_warp=mujoco_warp.__version__,
                          warp=wp.__version__, torch=torch.__version__, gpu_checked=checked,
                          timestep_s=self.dt, control_ticks=self.substeps, num_envs=self.num_envs,
                          integrator=int(m.opt.integrator), solver=int(m.opt.solver), iterations=m.opt.iterations,
                          cone=int(m.opt.cone), impratio=m.opt.impratio,
                          contacts_per_world=env.cfg.sim.nconmax, constraints_per_world=env.cfg.sim.njmax,
                          prepared_id=binding['prepared_id'], use_cuda_graph=env.sim.use_cuda_graph,
                          training_randomization='disabled (native default)')
        self.reference = Reference(binding, arrays, self.device, self.step_dt)
        self.qids = torch.as_tensor(self.scene.qids, device=self.device)
        self.vids = torch.as_tensor(self.scene.vids, device=self.device)
        self.aids = torch.as_tensor(self.scene.aids, device=self.device)
        self.part_ids = torch.as_tensor(self.scene.part_ids, device=self.device)
        self.root_ids = torch.as_tensor(m.body_rootid[self.scene.part_ids], device=self.device)
        self.geom_body = torch.as_tensor(m.geom_bodyid, device=self.device)
        self.body_slots = torch.as_tensor(self.scene.body_slots, device=self.device)
        self.body_parts = torch.as_tensor(self.scene.body_parts, device=self.device)
        self.max_episode_length = env.max_episode_length
        self.clock = torch.arange(self.max_episode_length + 1, device=self.device) * self.step_dt
        self.clock[-1] = task['end_tick'] * task['dt_s']
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        qpos, qvel = self._reset_reference(arrays)
        self.library = ResetLibrary(qpos, qvel, self.num_envs)
        self.pending_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.capacity_overflow = torch.zeros_like(self.pending_reset)
        self.tensor(self.data, 'qpos')[:] = qpos[0]
        self.tensor(self.data, 'qvel')[:] = qvel[0]
        env.sim.forward()
        self.state = self._snapshot(self.data, self._contacts(self.data))
        self.previous_velocity = [v.clone() for v in velocity_groups(self.state, self.layout)]
        self.it_obj_pos_err_sum = torch.zeros((), device=self.device)
        self.it_obj_rot_err_sum = torch.zeros((), device=self.device)
        self.it_step_count = 0
        self.finished_step = -1

    @staticmethod
    def tensor(data, name):
        return wp.to_torch(getattr(data, name))

    def observe(self):
        if self.pending_reset.any():
            state = self._snapshot(self.data, self._contacts(self.data))
            for key in self.state:
                self.state[key][self.pending_reset] = state[key][self.pending_reset]
            for old, new in zip(self.previous_velocity, velocity_groups(state, self.layout)):
                old[self.pending_reset] = new[self.pending_reset]
            self.pending_reset.zero_()
        return self.reference.observations(self.state, self.clock[self.frame])['policy']

    def finish_step(self):
        if self.finished_step == self.env.common_step_counter:
            return
        self.finished_step = self.env.common_step_counter
        if self.capacity_overflow.any():
            raise RuntimeError('MJWarp contact/constraint capacity exceeded during a physics substep')
        # Keep solve-time contacts, then refresh post-integration poses via MJLab.
        contacts = self._contacts(self.data)
        self.env.sim.forward()
        state = self._snapshot(self.data, contacts)
        for key in self.state:
            self.state[key].copy_(state[key])
        if not all(torch.isfinite(v).all() for v in self.state.values()):
            raise RuntimeError('Nonfinite training state')
        ref = self.reference.sample(self.clock[self.frame])
        groups, velocity = reward_groups(self.state, self.previous_velocity, ref, self.layout, self.step_dt)
        for previous, current in zip(self.previous_velocity, velocity):
            previous.copy_(current)
        self.env.extras.update(groups)
        self.frame += 1
        position, angle = pose_error(self.state, ref)
        self.it_obj_pos_err_sum += position.mean()
        self.it_obj_rot_err_sum += angle.mean()
        self.it_step_count += 1
        pt, rt = self.library.thresholds()
        self.failed = ((position > pt) | (angle > rt)).any(-1)
        self.timeout = (self.frame >= self.max_episode_length) & ~self.failed
        state = torch.cat((self.tensor(self.data, 'qpos'), self.tensor(self.data, 'qvel')), -1)
        self.library.record(self.frame, state)
        ids = (self.failed | self.timeout).nonzero().flatten()
        self.library.finish(ids, self.frame[ids])
        self.env.extras['log'].update({'Tracking/position_m': position.mean(), 'Tracking/angle_rad': angle.mean(),
                                      'Curriculum/success_ema': self.library.success_ema.clone()})

    def reset_reference(self, ids):
        start, state = self.library.sample(ids)
        nq = self.cpu_model.nq
        self.tensor(self.data, 'qpos')[ids] = state[:, :nq]
        self.tensor(self.data, 'qvel')[ids] = state[:, nq:]
        self.tensor(self.data, 'ctrl')[ids[:, None], self.aids] = state[:, self.qids]
        self.frame[ids] = start
        self.pending_reset[ids] = True

    def _reset_reference(self, arrays):
        source = self.reference.time
        hi = torch.searchsorted(source, self.clock, right=True).clamp(1, len(source)-1)
        lo = hi-1
        weight = ((self.clock-source[lo])/(source[hi]-source[lo]))[:, None]
        q = torch.as_tensor(arrays['qpos'], device=self.device, dtype=torch.float32)
        a, b = q[lo], q[hi].clone()
        for j, kind in enumerate(self.cpu_model.jnt_type):
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                address = self.cpu_model.jnt_qposadr[j]+3
                section = slice(address, address+4)
                b[:, section] = torch.where((a[:, section]*b[:, section]).sum(-1, keepdim=True)<0,
                                             -b[:, section], b[:, section])
        result = a+weight*(b-a)
        for j, kind in enumerate(self.cpu_model.jnt_type):
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                address = self.cpu_model.jnt_qposadr[j]+3
                result[:, address:address+4] = torch.nn.functional.normalize(result[:, address:address+4], dim=-1)
        v = torch.as_tensor(arrays['qvel'], device=self.device, dtype=torch.float32)
        return result, v[lo]+weight*(v[hi]-v[lo])

    def _check_capacity(self, data):
        if ((self.tensor(data, 'nacon')[0] > data.naconmax)
                | (self.tensor(data, 'nefc') > data.njmax).any() | self.tensor(data, 'overflow').any()):
            raise RuntimeError('MJWarp contact/constraint capacity exceeded; increase the explicit capacities')

    def _contacts(self, data):
        """Mean contact midpoint in the object's solve-time local frame, without host copies."""
        self._check_capacity(data)
        contact = data.contact
        geom = self.tensor(contact, 'geom').long()
        world = self.tensor(contact, 'worldid').long().clamp(0, self.num_envs-1)
        valid = torch.arange(data.naconmax, device=self.device) < self.tensor(data, 'nacon')[0]
        valid &= (geom >= 0).all(-1) & (self.tensor(contact, 'efc_address')[:, 0] >= 0)
        bodies = self.geom_body[geom.clamp(0, self.cpu_model.ngeom-1)]
        h, p = len(self.layout['hands']), len(self.part_ids)
        count = torch.zeros(self.num_envs*h*p*10, device=self.device)
        total = torch.zeros((len(count), 3), device=self.device)
        for robot_column, object_column in ((0, 1), (1, 0)):
            hand_slot = self.body_slots[bodies[:, robot_column]].long()
            part = self.body_parts[bodies[:, object_column]].long()
            mask = valid & (hand_slot[:, 0] >= 0) & (part >= 0)
            w, hs, ps = world[mask], hand_slot[mask], part[mask]
            body = self.part_ids[ps]
            delta = self.tensor(contact, 'pos')[mask]-self.tensor(data, 'xpos')[w, body]
            local = torch.einsum('nji,nj->ni', self.tensor(data, 'xmat')[w, body], delta)
            index = ((w*h+hs[:, 0])*p+ps)*10+hs[:, 1]
            count.index_add_(0, index, torch.ones(len(index), device=self.device))
            total.index_add_(0, index, local)
        return (count.reshape(self.num_envs, h, p, 10)>0,
                (total/count.clamp(min=1)[:, None]).reshape(self.num_envs, h, p, 10, 3))

    def _snapshot(self, data, contacts):
        # Caller refreshes poses through MJLab and supplies the desired solve-time contacts.
        pos = self.tensor(data, 'xpos')[:, self.part_ids]
        quat = self.tensor(data, 'xquat')[:, self.part_ids]
        spatial = self.tensor(data, 'cvel')[:, self.part_ids]
        offset = pos-self.tensor(data, 'subtree_com')[:, self.root_ids]
        linear = spatial[..., 3:]+torch.linalg.cross(spatial[..., :3], offset)
        return dict(robot_q=self.tensor(data, 'qpos')[:, self.qids].clone(),
                    robot_dq=self.tensor(data, 'qvel')[:, self.vids].clone(),
                    part_pose=torch.cat((pos, quat), -1),
                    part_velocity=torch.cat((linear, spatial[..., :3]), -1),
                    contact=contacts[0], contact_points=contacts[1])


@dataclass(kw_only=True)
class ResidualActionCfg(ActionTermCfg):
    # The sealed MJCF is attached as a whole scene, without rebuilding its entities or actuators.
    entity_name: str = ''

    def build(self, env):
        return ResidualAction(self, env)


class ResidualAction(ActionTerm):
    def __init__(self, cfg, env):
        ManagerTermBase.__init__(self, env)
        self.cfg = cfg
        self.task = env.task_state
        self._raw_action = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self.targets = torch.zeros_like(self._raw_action)

    @property
    def action_dim(self):
        return len(self.task.layout['joint_names'])

    @property
    def raw_action(self):
        return self._raw_action

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite policy action')
        self._raw_action.copy_(actions)
        self.task.capacity_overflow.zero_()
        self.targets.copy_(self.task.reference.targets(actions, self.task.clock[self.task.frame]))

    def apply_actions(self):
        self.task.ctrl[:, self.task.aids] = self.targets


def observation(env):
    return env.task_state.observe()


def pose_break(env):
    env.task_state.finish_step()
    return env.task_state.failed


def reference_end(env):
    return env.task_state.timeout


def reward(env):
    return env.extras['rg'] + env.extras['rs'] + env.extras['rp']


def reset_reference(env, env_ids):
    env.task_state.reset_reference(env_ids)


def capacity_health(env):
    task, data = env.task_state, env.sim.wp_data
    overflow = ((task.tensor(data, 'nacon')[0] > data.naconmax)
                | (task.tensor(data, 'nefc') > data.njmax) | (task.tensor(data, 'overflow') != 0))
    task.capacity_overflow.logical_or_(overflow)
    return overflow.float()


class ConTrackEnv(ManagerBasedRlEnv):
    """MJLab environment with ConTrack task terms; inherits its step and reset unchanged."""

    def __init__(self, prepared, num_envs=4096, device='cuda:0', contacts_per_world=512,
                 constraints_per_world=4096, *, physics_hz=120.0, control_hz=60.0, seed=0):
        self.prepared = Path(prepared)
        self.binding, self.arrays = load_prepared(prepared)
        task = self.binding['task']
        if not np.isfinite([physics_hz, control_hz]).all() or min(physics_hz, control_hz) <= 0:
            raise ValueError('Training frequencies must be positive and finite')
        duration = task['dt_s'] * task['end_tick']
        decimation = round(physics_hz / control_hz)
        if (decimation < 1 or not np.isclose(decimation / physics_hz, 1 / control_hz, atol=1e-12, rtol=0)
                or not np.isclose(round(duration * control_hz) / control_hz, duration, atol=1e-10, rtol=0)):
            raise ValueError('Task duration must contain complete training control intervals')
        path = self.prepared / 'task' / task['scene']
        source = mujoco.MjSpec.from_file(str(path))
        opt = source.option

        def attach_scene(spec):
            spec.option = source.option
            spec.nkey = source.nkey
            spec.attach(source, prefix='', frame=spec.worldbody.add_frame())

        physics = MujocoCfg(timestep=1 / physics_hz,
                            integrator=mujoco.mjtIntegrator(opt.integrator).name.removeprefix('mjINT_').lower(),
                            cone=mujoco.mjtCone(opt.cone).name.removeprefix('mjCONE_').lower(),
                            solver=mujoco.mjtSolver(opt.solver).name.removeprefix('mjSOL_').lower(),
                            jacobian=mujoco.mjtJacobian(opt.jacobian).name.removeprefix('mjJAC_').lower(),
                            impratio=opt.impratio, iterations=opt.iterations, tolerance=opt.tolerance,
                            ls_iterations=opt.ls_iterations, ls_tolerance=opt.ls_tolerance,
                            ccd_iterations=opt.ccd_iterations, gravity=tuple(opt.gravity))
        group = lambda: ObservationGroupCfg(terms={'features': ObservationTermCfg(func=observation)})
        cfg = ManagerBasedRlEnvCfg(
            seed=seed, decimation=decimation, episode_length_s=duration,
            scene=SceneCfg(num_envs=num_envs, env_spacing=0, spec_fn=attach_scene),
            sim=SimulationCfg(nconmax=contacts_per_world, njmax=constraints_per_world, mujoco=physics),
            observations={'policy': group(), 'critic': group()}, actions={'residual': ResidualActionCfg()},
            rewards={'tracking': RewardTermCfg(func=reward, weight=1.0)}, scale_rewards_by_dt=False,
            metrics={'capacity_overflow': MetricsTermCfg(func=capacity_health, per_substep=True, reduce='max')},
            terminations={'pose_break': TerminationTermCfg(func=pose_break),
                          'reference_end': TerminationTermCfg(func=reference_end, time_out=True)},
            events={'reference_reset': EventTermCfg(func=reset_reference, mode='reset')})
        super().__init__(cfg, device)

    def load_managers(self):
        self.task_state = TaskState(self, self.prepared, self.binding, self.arrays)
        super().load_managers()


def make_env(*args, **kwargs):
    # Keep Torch task terms and MJLab's Warp graphs on the same stream, including PPO inference.
    device = kwargs.get('device', args[2] if len(args) > 2 else 'cuda:0')
    wp.init()
    torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(device)))
    return RslRlVecEnvWrapper(ConTrackEnv(*args, **kwargs))
