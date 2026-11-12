import json
import socket
import struct
from threading import Thread

import pytest
import torch

from contrack_h2r.inputs import load_prepared, local_file
from contrack_h2r.protocol import receive
from contrack_h2r.reference import Reference
from contrack_h2r.resets import ResetLibrary


def test_sealed_files_and_paths(prepared):
    load_prepared(prepared)
    with pytest.raises(ValueError, match='escapes'):
        local_file(prepared, '../outside')
    (prepared / 'task/scene.xml').write_text('<mujoco/>')
    with pytest.raises(ValueError, match='Changed dependency'):
        load_prepared(prepared)


def test_reference_time_antipodes_contacts_and_endpoint(prepared):
    binding, arrays = load_prepared(prepared)
    arrays['part_quat'][1] *= -1
    arrays['robot_q'][:, 0] = arrays['time_s']*2
    arrays['contact'][1, 0, 0, 0] = True
    ref = Reference(binding, arrays, 'cpu')
    sample = ref.sample(torch.tensor([1/30, .1, .2]))
    torch.testing.assert_close(sample['robot_q'][:, 0], torch.tensor([2/30, .2, .2]))
    assert sample['contact'][0, 0, 0, 0] == 1
    torch.testing.assert_close(sample['part_quat'][..., 0].abs(), torch.ones(3, 2))
    targets = ref.targets(torch.tensor([[2., -2., 1.]]), torch.tensor([.1]))
    torch.testing.assert_close(targets, torch.tensor([[.2 + 2/30, -6.4/30, .25]]))


def test_protocol_fragmentation_and_duplicate_keys():
    a, b = socket.socketpair()
    with a, b:
        payload = json.dumps({'type': 'hello', 'n': 3}).encode()
        wire = struct.pack('!I', len(payload))+payload
        thread = Thread(target=lambda: [a.sendall(bytes([c])) for c in wire])
        thread.start()
        assert receive(b) == {'type': 'hello', 'n': 3}
        thread.join()
        payload = b'{"tick":0,"tick":1}'
        a.sendall(struct.pack('!I', len(payload))+payload)
        with pytest.raises(ValueError, match='Duplicate'):
            receive(b)
        a.sendall(struct.pack('!I', 33*1024*1024))
        with pytest.raises(ValueError, match='size'):
            receive(b)


def test_reset_library_keeps_first_frame_and_best_suffix():
    lib = ResetLibrary(torch.zeros(5, 2), torch.zeros(5, 1), 2)
    lib.start[:] = torch.tensor([0, 1])
    lib.history[0] = 2
    lib.history[1] = 3
    lib.finish(torch.tensor([0, 1]), torch.tensor([4, 2]))
    assert torch.all(lib.state[0] == 0)
    assert torch.all(lib.state[1:] == 2)  # longer successful suffix wins the duplicate frames
    saved = lib.state_dict()
    other = ResetLibrary(torch.zeros(5, 2), torch.zeros(5, 1), 2)
    other.load_state_dict(saved)
    torch.testing.assert_close(other.best, lib.best)
    assert other.success_ema > 0
