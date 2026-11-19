import json

from h2r_io import load_task
import pytest
import torch

from contrack_h2r.inputs import implementation_hashes, load_prepared, local_file, prepared_task
from contrack_h2r.layout import make_layout
from contrack_h2r.mjwarp_scene import Scene
from contrack_h2r.reference import Reference
from contrack_h2r.resets import ResetLibrary

from conftest import LAYOUT, write_prepared


def test_sealed_files_and_paths(prepared):
    binding, _ = load_prepared(prepared)
    assert prepared_task(prepared, binding).task_id == binding['task_id']
    with pytest.raises(ValueError, match='escapes'):
        local_file(prepared, '../outside')
    (prepared / 'task/scene.xml').write_text('<mujoco/>')
    with pytest.raises(ValueError, match='Task scene changed'):
        prepared_task(prepared, binding)
    (prepared / 'reference.npz').write_bytes(b'')
    with pytest.raises(ValueError, match='Changed dependency'):
        load_prepared(prepared)


def test_task_package_is_the_only_input(prepared, tmp_path):
    task = load_task(prepared / 'task')
    scene = Scene(task.scene_xml, LAYOUT)
    assert task.timestep_s == scene.model.opt.timestep == 1 / 300  # nominal; never the training step
    qpos, error = scene.object_qpos(task.demo)
    assert qpos.shape == (len(task.demo.time_s), scene.model.nq) and error < 1e-9
    with pytest.raises(ValueError, match="only for \\['sharpa_wave'\\], not 'toy_hand'"):
        make_layout(task)
    # A task of the former format is refused by the loader before any method code runs.
    old = tmp_path / 'old-task'
    old.mkdir()
    (old / 'task.json').write_text(json.dumps({'format': 'h2r-evaluation-v3', 'task_id': 'x', 'control_ticks': 5}))
    with pytest.raises(ValueError, match='Expected h2r-task-v1'):
        load_task(old)
    assert 'h2r/requirements.lock' in implementation_hashes()


def test_reference_must_fill_whole_policy_periods(tmp_path):
    # 3/50 s is not a whole number of 60 Hz policy periods; nothing is rounded.
    with pytest.raises(ValueError, match='complete training control intervals'):
        write_prepared(tmp_path / 'prepared', reference_hz=50.)


def test_reference_time_antipodes_contacts_and_endpoint(prepared):
    binding, arrays = load_prepared(prepared)
    arrays['part_quat'][1] *= -1
    arrays['robot_q'][:, 0] = arrays['time_s']*2
    arrays['contact'][1, 0, 0, 0] = True
    ref = Reference(binding['layout'], arrays, 'cpu', 1/30)
    sample = ref.sample(torch.tensor([1/30, .1, .2]))
    torch.testing.assert_close(sample['robot_q'][:, 0], torch.tensor([2/30, .2, .2]))
    assert sample['contact'][0, 0, 0, 0] == 1
    torch.testing.assert_close(sample['part_quat'][..., 0].abs(), torch.ones(3, 2))
    targets = ref.targets(torch.tensor([[2., -2., 1.]]), torch.tensor([.1]))
    torch.testing.assert_close(targets, torch.tensor([[.2 + 2/30, -6.4/30, .25]]))


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
