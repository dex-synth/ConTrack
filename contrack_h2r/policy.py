"""Sealed H2R checkpoints for inference: identity checks and the original actor."""

from pathlib import Path

import torch

from rsl_rl_contrack.modules import ActorCritic
from rsl_rl_contrack.sharpa_recipe import PROVENANCE

from .inputs import DEPLOYMENT_FORMAT, STATE_REFRESH, digest, read_json


class Checkpoint:
    """A model_N.pt bound to its model_N.json and to one prepared directory."""

    def __init__(self, path, binding, device='cpu'):
        self.path = Path(path).resolve()
        manifest = read_json(self.path.with_suffix('.json'))
        self.sha256 = manifest['checkpoint_sha256']
        if self.sha256 != digest(self.path):
            raise ValueError('Checkpoint checksum changed')
        # Trusted local pickle produced by contrack_h2r.train.
        self.saved = torch.load(self.path, map_location=device, weights_only=False)
        self.metadata = metadata = self.saved['infos']['h2r']
        if (metadata.get('format') != DEPLOYMENT_FORMAT or metadata.get('recipe') != PROVENANCE
                or metadata['training_physics'].get('state_refresh') != STATE_REFRESH):
            raise ValueError(f'Unsupported checkpoint ({metadata.get("format")}): retrain with contrack_h2r.train '
                             f'for {DEPLOYMENT_FORMAT} and the current recipe')
        if (metadata['prepared_id'] != binding['prepared_id'] or metadata['layout'] != binding['layout']
                or metadata != {k: v for k, v in manifest.items() if k != 'checkpoint_sha256'}):
            raise ValueError('Checkpoint does not match preparation/deployment metadata')
        self.device = device
        self.layout = binding['layout']
        physics = metadata['training_physics']
        self.timestep_s = physics['timestep_s']
        self.control_dt = physics['control_dt_s']
        # The runner saves model_{it}.pt after update index it.
        self.completed_updates = int(self.saved['iter']) + 1

    def identity(self):
        return {'checkpoint': str(self.path), 'sha256': self.sha256,
                'completed_iterations': self.completed_updates}

    def policy(self, observations):
        """The original ActorCritic in eval mode; shapes come from example observations."""
        cfg = dict(self.metadata['train_config']['policy'])
        if cfg.pop('class_name') != 'ActorCritic':
            raise ValueError('Only the verified nonrecurrent ConTrack policy is supported')
        policy = ActorCritic(observations, self.metadata['train_config']['obs_groups'],
                             len(self.layout['joint_names']), **cfg).to(self.device).eval()
        policy.load_state_dict(self.saved['model_state_dict'])
        return policy
