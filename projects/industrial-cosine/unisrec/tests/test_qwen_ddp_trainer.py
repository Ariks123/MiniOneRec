"""Policies and rank coordination only; no real Trainer, optimizer, or backward."""

import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_schedule import epoch_learning_rate


def test_warmup_and_cosine_endpoints():
    values = [epoch_learning_rate(.002, i, 100, 'warmup_cosine', 5, .0002)
              for i in range(100)]
    assert values[:5] == pytest.approx([.0004, .0008, .0012, .0016, .002])
    assert values[5] == pytest.approx(.002)
    assert values[-1] == pytest.approx(.0002)
    assert all(a >= b for a, b in zip(values[5:], values[6:]))
    assert epoch_learning_rate(.001, 50, 100) == .001


@pytest.mark.parametrize('kwargs', [
    {'base_lr': 0}, {'epoch_index': -1}, {'total_epochs': 0},
    {'warmup_epochs': 100}, {'min_learning_rate': .003}, {'schedule': 'unknown'},
])
def test_invalid_schedule_is_rejected(kwargs):
    settings = dict(base_lr=.002, epoch_index=0, total_epochs=100,
                    schedule='warmup_cosine', warmup_epochs=5, min_learning_rate=.0002)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        epoch_learning_rate(**settings)


@pytest.fixture
def trainer_module():
    pytest.importorskip('torch')
    pytest.importorskip('recbole')
    import qwen_train
    return qwen_train


def fake_trainer(module):
    trainer = object.__new__(module.QwenTrainer)
    trainer.config = {'single_spec': False, 'exclude_seen': False,
        'learning_rate': .002, 'lr_schedule': 'warmup_cosine', 'warmup_epochs': 5,
        'min_learning_rate': .0002, 'epochs': 100, 'train_batch_size': 512}
    trainer.current_epoch_lr = .0004
    trainer.saved_model_file = 'checkpoint.pth'
    trainer.cur_step = 0
    trainer.best_valid_score = .1
    trainer.model = SimpleNamespace(state_dict=lambda: {'stub': True},
        other_parameter=lambda: None, qwen_provenance={'features': 'known'})
    trainer.optimizer = SimpleNamespace(state_dict=lambda: {'stub': True})
    trainer.logger = logging.getLogger('test')
    return trainer


def test_checkpoint_directory_exists_before_recbole_initializer(trainer_module, monkeypatch, tmp_path):
    module = trainer_module
    directory = tmp_path/'nested'/'weights'
    config = {'checkpoint_dir': str(directory), 'learning_rate': .002}
    monkeypatch.setattr(module, 'distributed_world_size', lambda: 1)
    monkeypatch.setattr(module, 'is_primary', lambda: True)
    def parent_init(self, supplied_config, model):
        assert directory.is_dir()
        self.saved_model_file = str(directory/'checkpoint.pth')
    monkeypatch.setattr(module.Trainer, '__init__', parent_init)
    trainer = module.QwenTrainer(config, object())
    assert trainer.saved_model_file == str(directory/'checkpoint.pth')


def set_rank(module, monkeypatch, primary):
    monkeypatch.setattr(module, 'distributed_world_size', lambda: 4)
    monkeypatch.setattr(module, 'is_primary', lambda: primary)


def test_checkpoint_only_primary_writes_and_all_ranks_barrier(trainer_module, monkeypatch, tmp_path):
    module = trainer_module
    trainer = fake_trainer(module)
    trainer.saved_model_file = str(tmp_path / 'checkpoint.pth')
    saved, barriers, payload = [], [], []
    def save_fixture(state, path, **kwargs):
        saved.append((state, path))
        Path(path).write_bytes(b'synthetic checkpoint fixture')
    monkeypatch.setattr(module.torch, 'save', save_fixture)
    monkeypatch.setattr(module.torch.distributed, 'barrier', lambda: barriers.append(True))
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: payload.extend(obj))
    set_rank(module, monkeypatch, True)
    trainer._save_checkpoint(0)
    assert len(saved) == 1 and len(barriers) == 1
    assert Path(trainer.saved_model_file).read_bytes() == b'synthetic checkpoint fixture'
    assert not Path(trainer.saved_model_file + '.partial').exists()
    assert saved[0][0]['qwen_training_setup']['per_rank_batch_size'] == 128
    assert saved[0][0]['qwen_training_setup']['current_epoch_lr'] == .0004
    set_rank(module, monkeypatch, False)
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: obj.__setitem__(0, payload[0]))
    trainer._save_checkpoint(0)
    assert len(saved) == 1 and len(barriers) == 2


def test_primary_eval_is_complete_and_broadcast_without_reduce(trainer_module, monkeypatch):
    module = trainer_module
    trainer = fake_trainer(module)
    calls, payload = [], []
    set_rank(module, monkeypatch, True)
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: payload.extend(obj))
    def evaluate(self, data, **kwargs):
        assert self.config['single_spec'] is True
        assert data == 'full-test-4533'
        calls.append(data)
        return {'ndcg@10': .25}
    monkeypatch.setattr(module.Trainer, 'evaluate', evaluate)
    result = trainer.evaluate('full-test-4533', load_best_model=False)
    assert result == {'ndcg@10': .25} and trainer.config['single_spec'] is False
    set_rank(module, monkeypatch, False)
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: obj.__setitem__(0, payload[0]))
    assert trainer.evaluate('full-test-4533', load_best_model=False) == result
    assert calls == ['full-test-4533']


def test_epoch_policy_sets_lr_without_running_training(trainer_module, monkeypatch):
    module = trainer_module
    trainer = fake_trainer(module)
    trainer.learning_rate = .002
    trainer.epochs = 100
    trainer.optimizer = SimpleNamespace(param_groups=[{'lr': .002}])
    trainer.tensorboard = SimpleNamespace(add_scalar=lambda *args: None)
    monkeypatch.setattr(module, 'is_primary', lambda: True)
    calls = []
    def no_training(self, data, epoch_idx, **kwargs):
        calls.append((data, epoch_idx, self.optimizer.param_groups[0]['lr']))
        return 'not-trained'
    monkeypatch.setattr(module.Trainer, '_train_epoch', no_training)
    assert trainer._train_epoch('fake-data', 0) == 'not-trained'
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(.0004)
    assert calls == [('fake-data', 0, .0004)]


def test_eval_configuration_is_restored_after_failure(trainer_module, monkeypatch):
    module = trainer_module
    trainer = fake_trainer(module)
    set_rank(module, monkeypatch, True)
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: None)
    def fail(self, data, **kwargs):
        assert self.config['single_spec'] is True
        raise OSError('evaluation failed')
    monkeypatch.setattr(module.Trainer, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        trainer.evaluate('complete-data', load_best_model=False)
    assert trainer.config['single_spec'] is False


def test_primary_failure_reaches_every_rank(trainer_module, monkeypatch):
    module = trainer_module
    set_rank(module, monkeypatch, True)
    payload = []
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: payload.extend(obj))
    def fail():
        raise OSError('disk full')
    with pytest.raises(RuntimeError, match='disk full'):
        module.primary_call(fail, 'save')
    set_rank(module, monkeypatch, False)
    monkeypatch.setattr(module.torch.distributed, 'broadcast_object_list', lambda obj, src: obj.__setitem__(0, payload[0]))
    with pytest.raises(RuntimeError, match='disk full'):
        module.primary_call(lambda: None, 'save')


def test_four_gpu_training_rejects_plain_python(trainer_module, monkeypatch, tmp_path):
    module = trainer_module
    config = tmp_path / 'four.yaml'
    config.write_text('nproc: 4\n')
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='torchrun'):
        module.validate_training_launch(config)
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match='CUDA/NCCL'):
        module.validate_training_launch(config, device='cpu')
