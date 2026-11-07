"""Direct MJWarp VecEnv for the repository's unmodified PyTorch PPO/CMDP."""

from pathlib import Path

import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

from rsl_rl_contrack.env import VecEnv

from .inputs import load_prepared
from .mjwarp_scene import Scene
from .reference import Reference
from .resets import ResetLibrary
from .rewards import pose_error, reward_groups, velocity_groups


class MJWarpEnv(VecEnv):
    def __init__(self, prepared, num_envs=64, device='cuda:0', contacts_per_world=512, constraints_per_world=4096):
        if num_envs < 1 or contacts_per_world < 1 or constraints_per_world < 1:
            raise ValueError('Positive environment count and capacities required')
        if not str(device).startswith('cuda'):
            raise ValueError('The MJWarp training environment requires CUDA')
        self.binding, arrays = load_prepared(prepared)
        self.layout, task = self.binding['layout'], self.binding['task']
        self.scene = Scene(Path(prepared) / 'task' / task['scene'], task, self.layout)
        self.cpu_model = m = self.scene.model
        self.device, self.num_envs = str(device), num_envs
        self.num_actions = len(self.layout['joint_names'])
        self.cfg = dict(backend='mujoco_warp', prepared_id=self.binding['prepared_id'], num_envs=num_envs,
                        contacts_per_world=contacts_per_world, constraints_per_world=constraints_per_world)
        self.dt, self.substeps = task['dt_s'], task['control_ticks']
        self.step_dt = self.dt*self.substeps
        if not np.isclose(m.opt.timestep, self.dt, atol=1e-12, rtol=0):
            raise ValueError('MJCF time step differs from task')
        # All Warp kernels and shared Torch buffers use the same CUDA stream.
        wp.init()
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        with wp.ScopedStream(self.stream):
            self.model = mw.put_model(m)
            self.data = mw.make_data(m, nworld=num_envs, nconmax=contacts_per_world, njmax=constraints_per_world)
            # Forward on scratch data publishes reset contacts without solving active worlds twice.
            self.reset_data = mw.make_data(m, nworld=num_envs, nconmax=contacts_per_world, njmax=constraints_per_world)
        self.audit = self._audit_gpu()
        self.reference = Reference(self.binding, arrays, device)
        self.qids = torch.as_tensor(self.scene.qids, device=device)
        self.vids = torch.as_tensor(self.scene.vids, device=device)
        self.aids = torch.as_tensor(self.scene.aids, device=device)
        self.part_ids = torch.as_tensor(self.scene.part_ids, device=device)
        self.root_ids = torch.as_tensor(m.body_rootid[self.scene.part_ids], device=device)
        self.geom_body = torch.as_tensor(m.geom_bodyid, device=device)
        self.body_slots = torch.as_tensor(self.scene.body_slots, device=device)
        self.body_parts = torch.as_tensor(self.scene.body_parts, device=device)
        self.max_episode_length = (task['end_tick']+self.substeps-1)//self.substeps
        self.clock = torch.arange(self.max_episode_length+1, device=device)*self.step_dt
        self.clock[-1] = task['end_tick']*self.dt
        self.frame = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_length_buf = torch.zeros_like(self.frame)
        q, v = self._reset_reference(arrays)
        self.library = ResetLibrary(q, v, num_envs)
        self.previous_velocity = None
        self.state = None
        self.it_obj_pos_err_sum = torch.zeros((), device=device)
        self.it_obj_rot_err_sum = torch.zeros((), device=device)
        self.it_step_count = 0
        self.reset(torch.arange(num_envs, device=device))

    @staticmethod
    def tensor(data, name):
        return wp.to_torch(getattr(data, name))

    def _audit_gpu(self):
        checked = []
        for key, values in self.scene.audit().items():
            target = getattr(self.model, key, None)
            if target is None:
                raise ValueError(f'MJWarp cannot audit required model field: {key}')
            actual = target.numpy() if isinstance(target, wp.array) else np.asarray(target)
            expected = np.asarray(values)
            if actual.size == expected.size:
                actual = actual.reshape(expected.shape)
            equal = (np.array_equal(actual, expected) if np.issubdtype(expected.dtype, np.integer)
                     else np.allclose(actual, expected, atol=1e-7, rtol=1e-5))
            if not equal:
                raise ValueError(f'MJWarp model differs from compiled MJCF: {key}')
            checked.append(key)
        return dict(mujoco=mujoco.__version__, mujoco_warp=mw.__version__,
                    warp=wp.__version__, torch=torch.__version__, gpu_checked=checked,
                    timestep_s=self.dt, integrator=int(self.cpu_model.opt.integrator),
                    solver=int(self.cpu_model.opt.solver), iterations=self.cpu_model.opt.iterations,
                    training_randomization='disabled (native default)', **self.cfg)

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
        if (self.tensor(data, 'nacon')[0] > data.naconmax or
                (self.tensor(data, 'nefc') > data.njmax).any() or self.tensor(data, 'overflow').any()):
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
        # Refresh post-integration kinematics only, preserving the last solve/contact buffers.
        with wp.ScopedStream(self.stream):
            mw.fwd_kinematics(self.model, data)
            mw.com_vel(self.model, data)
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

    def reset(self, ids):
        if not len(ids):
            return
        start, state = self.library.sample(ids)
        mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        mask[ids] = True
        with wp.ScopedStream(self.stream):
            mw.reset_data(self.model, self.data, wp.from_torch(mask))
            mw.reset_data(self.model, self.reset_data)
        nq = self.cpu_model.nq
        self.tensor(self.reset_data, 'qpos')[:] = self.library.state[0, :nq]
        self.tensor(self.reset_data, 'qvel')[:] = self.library.state[0, nq:]
        for data in (self.data, self.reset_data):
            self.tensor(data, 'qpos')[ids] = state[:, :nq]
            self.tensor(data, 'qvel')[ids] = state[:, nq:]
            self.tensor(data, 'ctrl')[ids[:, None], self.aids] = state[:, self.qids]
        with wp.ScopedStream(self.stream):
            mw.forward(self.model, self.reset_data)
        reset_state = self._snapshot(self.reset_data, self._contacts(self.reset_data))
        if self.state is None:
            self.state = reset_state
        else:
            for key in self.state:
                self.state[key][ids] = reset_state[key][ids]
        self.frame[ids] = start
        self.episode_length_buf[ids] = 0
        velocity = velocity_groups(self.state, self.layout)
        if self.previous_velocity is None:
            self.previous_velocity = [v.clone() for v in velocity]
        else:
            for previous, current in zip(self.previous_velocity, velocity):
                previous[ids] = current[ids]

    def get_observations(self):
        return self.reference.observations(self.state, self.clock[self.frame])

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions) or not torch.isfinite(actions).all():
            raise ValueError('Invalid policy action')
        time = self.clock[self.frame]
        targets = self.reference.targets(actions, time)
        self.tensor(self.data, 'ctrl')[:, self.aids] = targets
        tail = self.binding['task']['end_tick'] % self.substeps
        short = (self.frame == self.max_episode_length-1) if tail else torch.zeros_like(self.frame, dtype=torch.bool)
        terminal = None
        with wp.ScopedStream(self.stream):
            for substep in range(1, self.substeps+1):
                mw.step(self.model, self.data)
                self._check_capacity(self.data)
                if substep == tail and short.any():
                    # Preserve the exact endpoint of a fractional final interval.
                    # Extra batched work for these terminated worlds is discarded before reset.
                    terminal = (self._snapshot(self.data, self._contacts(self.data)),
                                self.tensor(self.data, 'qpos').clone(), self.tensor(self.data, 'qvel').clone())
        contacts = self._contacts(self.data)
        self.state = self._snapshot(self.data, contacts)
        if terminal is not None:
            for key in self.state:
                self.state[key][short] = terminal[0][key][short]
            self.tensor(self.data, 'qpos')[short] = terminal[1][short]
            self.tensor(self.data, 'qvel')[short] = terminal[2][short]
        if not all(torch.isfinite(value).all() for value in self.state.values()):
            raise RuntimeError('Nonfinite MJWarp state')
        # As in the native manager path, score the transition against its action-time reference.
        ref = self.reference.sample(time)
        elapsed = self.clock[self.frame+1]-time
        extras, velocity = reward_groups(self.state, self.previous_velocity, ref, self.layout, elapsed)
        self.previous_velocity = [v.clone() for v in velocity]
        self.frame += 1
        self.episode_length_buf += 1
        position, angle = pose_error(self.state, ref)
        self.it_obj_pos_err_sum += position.mean()
        self.it_obj_rot_err_sum += angle.mean()
        self.it_step_count += 1
        pt, rt = self.library.thresholds()
        failed = ((position > pt) | (angle > rt)).any(-1)
        timeout = self.frame >= self.max_episode_length
        done = failed | timeout
        state = torch.cat((self.tensor(self.data, 'qpos'), self.tensor(self.data, 'qvel')), -1)
        self.library.record(self.frame, state)
        ids = done.nonzero().flatten()
        self.library.finish(ids, self.frame[ids])
        extras.update(time_outs=timeout & ~failed,
                      log={'Tracking/position_m': position.mean(), 'Tracking/angle_rad': angle.mean(),
                           'Curriculum/success_ema': self.library.success_ema.clone()})
        reward = extras['rg']+extras['rs']+extras['rp']
        self.reset(ids)
        return self.get_observations(), reward, done, extras
