"""Synthetic inference fixtures only: no optimizer, backward, or real checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_vector_eval import evaluate_checkpoint, evaluate_model_vectors, main


def test_missing_checkpoint_fails_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    with pytest.raises(FileNotFoundError, match='Trained checkpoint'):
        evaluate_checkpoint('not-needed.yaml', tmp_path / 'missing.pth', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_only_held_out_and_single_process_evaluation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='valid or test'):
        evaluate_checkpoint('unused', 'unused', tmp_path / 'out', split='train')
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match='do not launch with torchrun'):
        evaluate_checkpoint('unused', 'unused', tmp_path / 'out')


def test_checkpoint_is_required_by_cli():
    with pytest.raises(SystemExit):
        main(['--output', '/tmp/unused-vector-eval'])


@pytest.fixture
def fixture_model():
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')

    class Batch:
        def __init__(self, fields):
            self.fields = fields

        def __getitem__(self, key):
            if isinstance(key, str):
                return self.fields[key]
            return Batch({name: values[key] for name, values in self.fields.items()})

        def __contains__(self, key):
            return key in self.fields

        def __len__(self):
            return len(next(iter(self.fields.values())))

    class ToyDataset:
        iid_field, uid_field, item_num = 'item_id', 'user_id', 4
        field2id_token = {'item_id': ['[PAD]', '7', '0', '2'],
                          'user_id': ['[PAD]', 'u0', 'u1']}
        train_item_ids = torch.tensor([1, 2])
        inter_feat = Batch({
            'item_id_list': torch.tensor([[1, 0], [2, 1]]),
            'item_length': torch.tensor([1, 2]),
            'item_id': torch.tensor([2, 3]),
            'user_id': torch.tensor([1, 2]),
        })

        def __len__(self):
            return len(self.inter_feat)

    class ToyModel(torch.nn.Module):
        ITEM_SEQ, ITEM_SEQ_LEN, POS_ITEM_ID = 'item_id_list', 'item_length', 'item_id'

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.qwen_provenance = {'synthetic_fixture': True}
            # Deliberately includes the cold item, like train_catalog_only:false.
            self.train_candidate_ids = torch.tensor([1, 2, 3])
            self.calls = []

        def encode_items(self, ids):
            assert not self.training and not torch.is_grad_enabled()
            self.calls.append(('item', ids.detach().cpu().tolist()))
            table = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [-1., 0.]], device=ids.device)
            return table[ids]

        def encode_history(self, sequences, lengths):
            assert not self.training and not torch.is_grad_enabled()
            self.calls.append(('query', len(sequences)))
            x = torch.where(sequences[:, 0] == 1, 1., -1.)
            return torch.stack((x, torch.zeros_like(x)), dim=1)

    return np, torch, ToyModel(), ToyDataset()


def test_evaluation_preserves_state_rows_raw_zero_and_actual_cold_groups(fixture_model):
    np, torch, model, dataset = fixture_model
    before = model.anchor.detach().clone()
    result = evaluate_model_vectors(model, dataset, temperature=.1, topk=(1, 2),
                                    batch_size=1, item_batch_size=2, include_vectors=True)
    assert model.training
    torch.testing.assert_close(model.anchor, before)
    assert all(parameter.grad is None for parameter in model.parameters())
    np.testing.assert_array_equal(result['per_row']['target_item_id'], ['0', '2'])
    np.testing.assert_array_equal(result['per_row']['target_internal_id'], [2, 3])
    np.testing.assert_array_equal(result['per_row']['target_train_seen'], [True, False])
    np.testing.assert_array_equal(result['per_row']['history_length'], [1, 2])
    np.testing.assert_array_equal(result['per_row']['user_id'], ['u0', 'u1'])
    assert result['summary']['groups']['train_seen']['count'] == 1
    assert result['summary']['groups']['train_unseen']['count'] == 1
    assert result['summary']['positive_cosine_mean'] == pytest.approx(1.)
    assert result['summary']['pairwise_accuracy_mean'] == pytest.approx(1.)
    assert result['evaluation']['checkpoint_verified'] is False
    assert result['evaluation']['candidate_count'] == 3
    np.testing.assert_array_equal(result['vectors']['queries'], result['vectors']['targets'])
    assert ('item', [1, 2]) in model.calls and ('item', [3]) in model.calls


def test_model_mode_is_restored_after_encoder_failure(fixture_model, monkeypatch):
    _, _, model, dataset = fixture_model
    def fail(*args):
        raise ValueError('synthetic encoder failure')
    monkeypatch.setattr(model, 'encode_history', fail)
    with pytest.raises(ValueError, match='synthetic encoder failure'):
        evaluate_model_vectors(model, dataset, temperature=.1)
    assert model.training


def test_padding_target_is_rejected_before_inference(fixture_model):
    _, _, model, dataset = fixture_model
    dataset.inter_feat.fields['item_id'] = dataset.inter_feat['item_id'].clone()
    dataset.inter_feat.fields['item_id'][0] = 0
    with pytest.raises(ValueError, match='non-padding'):
        evaluate_model_vectors(model, dataset, temperature=.1)
    assert model.calls == []


def synthetic_runtime(tmp_path, fixture_model, monkeypatch, epoch=2):
    _, _, model, dataset = fixture_model
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    config_path = tmp_path / 'synthetic.yaml'
    config_path.write_text('synthetic fixture only\n')
    checkpoint_path = tmp_path / 'synthetic.pth'
    checkpoint_path.write_bytes(b'synthetic fixture; no model training took place')
    catalog_path = tmp_path / 'catalog.json'
    # Colliding SID and null ASIN must not merge distinct target item IDs.
    catalog_path.write_text(json.dumps([
        {'embedding_row': 0, 'item_id': '0', 'asin': None,
         'sid_tokens': ['<a_1>'], 'sid': '<a_1>'},
        {'embedding_row': 1, 'item_id': '7', 'asin': None,
         'sid_tokens': ['<a_2>'], 'sid': '<a_2>'},
        {'embedding_row': 2, 'item_id': '2', 'asin': None,
         'sid_tokens': ['<a_1>'], 'sid': '<a_1>'},
    ]))
    module = ModuleType('qwen_runtime')
    module.ROOT = tmp_path
    config = {'temperature': .1, 'topk': [1, 2], 'catalog_path': str(catalog_path)}
    loaders = [None, SimpleNamespace(dataset=dataset), SimpleNamespace(dataset=dataset)]
    module.load_experiment = lambda *args, **kwargs: (config, dataset, loaders, model)
    module.load_trained_checkpoint = lambda *args: {'epoch': epoch}
    module.sha256_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, 'qwen_runtime', module)
    return config_path, checkpoint_path


def test_checkpoint_report_preserves_colliding_sids_and_exports_aligned_vectors(
        fixture_model, tmp_path, monkeypatch):
    np, _, _, _ = fixture_model
    config, checkpoint = synthetic_runtime(tmp_path, fixture_model, monkeypatch)
    output = evaluate_checkpoint(config, checkpoint, tmp_path / 'report', save_vectors=True)
    report = json.loads((output / 'report.json').read_text())
    rows = [json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines()]
    assert report['evaluation']['checkpoint_verified'] is True
    assert report['evaluation']['checkpoint_epoch'] == 2
    assert report['evaluation']['training_started'] is False
    assert [row['target_item_id'] for row in rows] == ['0', '2']
    assert [row['target_sid'] for row in rows] == ['<a_1>', '<a_1>']
    assert [row['row_index'] for row in rows] == [0, 1]
    np.testing.assert_array_equal(np.load(output / 'query_vectors.npy'), [[1., 0.], [-1., 0.]])
    np.testing.assert_array_equal(np.load(output / 'target_item_vectors.npy'), [[1., 0.], [-1., 0.]])
    with pytest.raises(FileExistsError, match='overwrite'):
        evaluate_checkpoint(config, checkpoint, output)


@pytest.mark.parametrize('epoch', [-1, True, '0'])
def test_untrained_or_invalid_checkpoint_epoch_is_rejected(
        fixture_model, tmp_path, monkeypatch, epoch):
    config, checkpoint = synthetic_runtime(tmp_path, fixture_model, monkeypatch, epoch=epoch)
    with pytest.raises(ValueError, match='trained checkpoint'):
        evaluate_checkpoint(config, checkpoint, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
