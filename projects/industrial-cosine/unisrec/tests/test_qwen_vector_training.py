"""Vector-focused selection and logging checks; no training or optimizer."""

import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_vector_nll_is_minimized_after_recbole_defaults(monkeypatch):
    pytest.importorskip('recbole')
    from qwen_model import QwenUniSRec
    from qwen_runtime import ROOT, _config_class
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'QWEN_DDP_BACKEND'):
        monkeypatch.delenv(key, raising=False)
    config = _config_class()(model=QwenUniSRec, dataset='Industrial_and_Scientific',
        config_file_list=[str(ROOT/'props/UniSRec.yaml'), str(ROOT/'props/qwen_industrial_4gpu.yaml')],
        config_dict={'qwen_requested_device': 'cpu'})
    assert config['valid_metric'] == 'vector_nll'
    assert config['valid_metric_bigger'] is False
    assert config['metrics'] == ['Recall', 'NDCG', 'Hit']
    from recbole.utils import calculate_valid_score, early_stopping
    score = calculate_valid_score({'vector_nll': 2.0}, config['valid_metric'])
    best, _, _, update = early_stopping(score, 3.0, 0, max_step=10,
                                       bigger=config['valid_metric_bigger'])
    assert best == 2.0 and update
    with pytest.raises(ValueError, match='requires vector_quality_eval'):
        _config_class()(model=QwenUniSRec, dataset='Tiny',
            config_file_list=[str(ROOT/'props/UniSRec.yaml'), str(ROOT/'props/qwen_industrial_4gpu.yaml')],
            config_dict={'qwen_requested_device': 'cpu', 'valid_metric': 'vector_nll',
                         'vector_quality_eval': False})


def test_training_evaluation_adds_vector_quality_without_training(monkeypatch, tmp_path):
    pytest.importorskip('recbole')
    import qwen_train as module
    import qwen_vector_eval
    trainer = object.__new__(module.QwenTrainer)
    trainer.config = {'single_spec': False, 'vector_quality_eval': True,
                      'temperature': .07, 'topk': [10, 20, 50], 'eval_batch_size': 128}
    trainer.model = object()
    trainer.logger = logging.getLogger('vector-eval-test')
    trainer.tensorboard = SimpleNamespace(add_scalar=lambda *args: None)
    trainer.saved_model_file = str(tmp_path/'checkpoint.pth')
    trainer.current_epoch_index = 3
    summary = {'vector_nll': 2.0, 'positive_cosine_mean': .7,
               'margin_vs_hardest_negative_mean': .2, 'pairwise_accuracy_mean': .9,
               'alignment_squared_l2_mean': .6}
    class Dataset:
        def __len__(self):
            return 2
    dataset = Dataset()
    monkeypatch.setattr(module, 'distributed_world_size', lambda: 1)
    monkeypatch.setattr(module, 'is_primary', lambda: True)
    monkeypatch.setattr(module.Trainer, 'evaluate', lambda *args, **kwargs: {'recall@10': .5})
    calls = []
    def measured(model, data, **kwargs):
        assert trainer.config['single_spec'] is True
        assert data is dataset
        calls.append(kwargs)
        return {'summary': summary, 'geometry': {'queries': {'collapsed': False}}}
    monkeypatch.setattr(qwen_vector_eval, 'evaluate_model_vectors', measured)
    result = trainer.evaluate(SimpleNamespace(dataset=dataset), load_best_model=False)
    assert result['recall@10'] == .5 and result['vector_nll'] == 2.0
    assert trainer.config['single_spec'] is False
    assert trainer.last_vector_quality['epoch_index'] == 3
    assert len(calls) == 1
    logged = json.loads((tmp_path/'checkpoint.vector_quality.jsonl').read_text())
    assert logged['summary']['margin_vs_hardest_negative_mean'] == .2
    assert not (tmp_path/'checkpoint.pth').exists()
