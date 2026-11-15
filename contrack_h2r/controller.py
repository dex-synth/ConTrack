"""Complete feedback controller; deliberately imports no simulator."""

import numpy as np
import torch

from .h2r_io import client
from .inputs import load_prepared
from .policy import Checkpoint
from .reference import Reference

METHOD = 'ConTrack-H2R'


class Controller:
    def __init__(self, prepared, checkpoint, device='cpu'):
        self.binding, arrays = load_prepared(prepared)
        self.layout = self.binding['layout']
        self.device = device
        self.checkpoint = Checkpoint(checkpoint, self.binding, device)
        # The policy period is the trained one; the evaluation's physics step arrives with hello.
        self.reference = Reference(self.layout, arrays, device, self.checkpoint.control_dt)
        # Shapes are determined by the common feature function; no physical world is created.
        h, p, n = len(self.layout['hands']), len(self.layout['parts']), len(self.layout['joint_names'])
        zero = lambda *shape: torch.zeros(shape, device=device)
        dummy = dict(robot_q=zero(1, n), robot_dq=zero(1, n), part_pose=zero(1, p, 7),
                     part_velocity=zero(1, p, 6), contact=zero(1, h, p, 10))
        self.policy = self.checkpoint.policy(self.reference.observations(dummy, zero(1)))
        self.initial = dict(zip(self.layout['joint_names'], arrays['robot_q'][0].tolist()))
        self.next_tick = {}
        self.finished = set()

    def start(self, hello):
        if hello.get('type') != 'hello' or hello.get('protocol') != client.PROTOCOL:
            raise ValueError(f'Unsupported evaluator protocol {hello.get("protocol")!r}; expected {client.PROTOCOL}')
        if hello['task_id'] != self.binding['task_id']:
            raise ValueError('Evaluator runs a different task than the prepared one')
        if not hello['observation']['contacts']:
            raise ValueError('ConTrack requires contact observations; the evaluation has them disabled')
        self.control_ticks = client.control_ticks(hello, self.reference.dt)
        self.dt_s, self.end_tick = hello['dt_s'], hello['end_tick']
        # Joints are matched by name; the task_id already fixes the set, units and object semantics.
        self.joint_order = [hello['joint_names'].index(n) for n in self.layout['joint_names']]
        self.next_tick = {e['episode_id']: 0 for e in hello['episodes']}
        if len(self.next_tick) != len(hello['episodes']) or not self.next_tick:
            raise ValueError('Invalid episode IDs')
        return client.start_request(
            hello, method=METHOD, controller={**self.checkpoint.identity(), 'prepared_id': self.binding['prepared_id']},
            initial_robot={e: self.initial for e in self.next_tick}, control_dt_s=self.reference.dt,
            bodies=self.layout['part_bodies'], contacts=True)

    def state(self, frames):
        h, p = len(self.layout['hands']), len(self.layout['parts'])
        contacts = np.zeros((len(frames), h, p, 10), dtype=np.float32)
        bodies = {body: i for i, body in enumerate(self.layout['part_bodies'])}
        for i, frame in enumerate(frames):
            records = frame['contacts']
            if records is None or records['solve_tick'] != (frame['tick']-1 if frame['tick'] else None):
                raise ValueError('Missing or mistimed actual contacts')
            for record in records['records']:
                for robot, obj in ((record['body1'], record['body2']), (record['body2'], record['body1'])):
                    if robot in self.layout['contact_bodies'] and obj in bodies:
                        hand, slot = self.layout['contact_bodies'][robot]
                        contacts[i, hand, bodies[obj], slot] = 1
        body_state = np.asarray([[f['bodies'][b] for b in self.layout['part_bodies']] for f in frames])
        state = dict(robot_q=np.asarray([f['robot_q'] for f in frames])[:, self.joint_order],
                     robot_dq=np.asarray([f['robot_dq'] for f in frames])[:, self.joint_order],
                     part_pose=body_state[..., :7], part_velocity=body_state[..., 7:13], contact=contacts)
        if body_state.shape != (len(frames), p, 13) or not all(np.isfinite(v).all() for v in state.values()):
            raise ValueError('Invalid physical feedback')
        if not np.allclose(np.linalg.norm(body_state[..., 3:7], axis=-1), 1, atol=1e-4):
            raise ValueError('Invalid physical quaternion')
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in state.items()}

    @torch.inference_mode()
    def step(self, message):
        if message.get('type') == 'error':
            raise ValueError(f"Evaluator rejected controller: {message['message']}")
        if message.get('type') != 'frames':
            raise ValueError('Expected evaluator frames')
        frames = message['episodes']
        if {f['episode_id'] for f in frames} != set(self.next_tick) or len(frames) != len(self.next_tick):
            raise ValueError('Episode set changed')
        active = []
        for frame in frames:
            eid = frame['episode_id']
            if frame['done']:
                self.finished.add(eid)
                continue
            if (eid in self.finished or frame['tick'] != self.next_tick[eid]
                    or frame['tick'] >= self.end_tick
                    or not np.isclose(frame['time_s'], frame['tick']*self.dt_s, atol=1e-10, rtol=0)):
                raise ValueError('Invalid control tick or episode lifecycle')
            active.append(frame)
        if bool(message['done']) != (len(self.finished) == len(self.next_tick)):
            raise ValueError('Inconsistent batch termination')
        if not active:
            return None
        time = torch.tensor([f['time_s'] for f in active], device=self.device, dtype=torch.float32)
        obs = self.reference.observations(self.state(active), time)
        actions = self.policy.act_inference(obs)
        targets = self.reference.targets(actions, time).cpu().numpy()
        commands = {}
        for frame, target in zip(active, targets):
            eid = frame['episode_id']
            commands[eid] = dict(tick=frame['tick'], joint_targets=dict(zip(self.layout['joint_names'], target.tolist())))
            self.next_tick[eid] = min(self.end_tick, frame['tick']+self.control_ticks)
        return dict(type='step', commands=commands)
