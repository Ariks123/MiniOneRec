"""Runtime and data-only DDP regression checks; no optimizer or backward."""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_runtime import runtime_environment


def test_nproc_configuration_alone_does_not_launch():
    assert runtime_environment({}) == {'rank': 0, 'local_rank': 0,
                                        'world_size': 1, 'local_world_size': 1}


def test_torchrun_environment_and_invalid_ranks():
    env = {'WORLD_SIZE': '4', 'LOCAL_WORLD_SIZE': '4', 'RANK': '3', 'LOCAL_RANK': '3',
           'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12345'}
    assert runtime_environment(env)['rank'] == 3
    with pytest.raises(ValueError, match='Invalid'):
        runtime_environment(dict(env, LOCAL_RANK='4'))
    with pytest.raises(ValueError, match='torchrun'):
        runtime_environment({'WORLD_SIZE': '4'})


def test_cpu_override_does_not_touch_cuda_even_when_available(monkeypatch):
    torch = pytest.importorskip('torch')
    from qwen_runtime import _configure_device
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'QWEN_DDP_BACKEND'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5,6,7')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def forbidden(*args):
        raise AssertionError('CPU override must not call cuda.set_device')
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    config = {'nproc': 4, 'gpu_id': '0,1,2,3', 'use_gpu': True}
    _configure_device(config, 'cpu')
    assert config['single_spec'] is True
    assert config['device'] == torch.device('cpu')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'
    assert config['qwen_requested_nproc'] == 4


@pytest.mark.parametrize('visible', [None, '4,5,6,7'])
@pytest.mark.parametrize('requested, expected', [('cuda:2', 'cuda:2'), ('cuda', 'cuda:0')])
def test_explicit_cuda_logical_device_preserves_visibility(monkeypatch, visible, requested, expected):
    torch = pytest.importorskip('torch')
    from qwen_runtime import _configure_device
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'QWEN_DDP_BACKEND'):
        monkeypatch.delenv(key, raising=False)
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    selected = []
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    config = {'nproc': 1, 'gpu_id': '0', 'use_gpu': True}
    _configure_device(config, requested)
    assert selected == [torch.device(expected)]
    assert config['device'] == torch.device(expected)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible


def test_process_group_helpers_are_idempotent_without_group():
    pytest.importorskip('torch')
    from qwen_runtime import (distributed_rank, distributed_world_size,
                              is_primary, cleanup_distributed)
    assert distributed_rank() == 0 and distributed_world_size() == 1 and is_primary()
    cleanup_distributed()
    cleanup_distributed()


def test_runtime_config_is_picklable():
    pytest.importorskip('recbole')
    import pickle
    from qwen_runtime import _config_class
    cls = _config_class()
    value = object.__new__(cls)
    value.final_config_dict = {'single_spec': True}
    restored = pickle.loads(pickle.dumps(value))
    assert restored['single_spec'] is True


def test_loader_preparation_builds_once_and_shards_train_only(monkeypatch):
    pytest.importorskip('recbole')
    import recbole.data.utils as utils
    from types import SimpleNamespace
    from qwen_runtime import _config_class, _prepare_loaders
    config = object.__new__(_config_class())
    config.final_config_dict = {'single_spec': False, 'shuffle': True}
    dataset = SimpleNamespace(build_calls=0)
    def build():
        dataset.build_calls += 1
        return ('train-dataset', 'valid-dataset', 'test-dataset')
    dataset.build = build
    monkeypatch.setattr(utils, 'create_samplers', lambda *args: (None, None, None))
    def get_loader(config, phase):
        return lambda used_config, used_dataset, sampler, shuffle: SimpleNamespace(
            config=used_config, dataset=used_dataset, phase=phase, shuffle=shuffle)
    monkeypatch.setattr(utils, 'get_dataloader', get_loader)
    train, valid, test = _prepare_loaders(config, dataset)
    assert dataset.build_calls == 1
    assert not train.config['single_spec']
    assert valid.config['single_spec'] and test.config['single_spec']
    assert config['single_spec'] is False
    assert train.shuffle and not valid.shuffle and not test.shuffle
