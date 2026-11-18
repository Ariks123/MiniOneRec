import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_runtime import validate_prepared


def test_modified_prepared_split_is_rejected(tmp_path):
    names = ['Tiny' + suffix for suffix in (
        '.item', '.train.inter', '.valid.inter', '.test.inter',
        '.emb-qwen-td.npy', '.emb-qwen-td.npy.ids.json')]
    names.extend(['catalog.json', 'train_item_ids.json'])
    hashes = {}
    for name in names:
        (tmp_path / name).write_text('original')
        hashes[name] = hashlib.sha256(b'original').hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'dataset': 'Tiny', 'max_history_len': 10, 'prepared_sha256': hashes}))
    validate_prepared(tmp_path, 'Tiny', 10)
    (tmp_path / 'Tiny.train.inter').write_text('changed')
    import pytest
    with pytest.raises(ValueError, match='differs from manifest'):
        validate_prepared(tmp_path, 'Tiny', 10)


def test_history_filter_is_explicit_and_repeat_targets_remain_by_default():
    import torch
    from types import SimpleNamespace
    from recbole.data.interaction import Interaction
    from qwen_train import QwenTrainer
    # Build only the evaluation method's state; do not construct a Trainer/optimizer.
    trainer = object.__new__(QwenTrainer)
    trainer.config = {'exclude_seen': False}
    trainer.device = torch.device('cpu')
    trainer.tot_item_num = 4
    trainer.model = SimpleNamespace(ITEM_SEQ='item_id_list',
        full_sort_predict=lambda interaction: torch.ones((len(interaction), 4)))
    interaction = Interaction({'item_id_list': torch.tensor([[1, 2, 0], [3, 0, 0]])})
    batch = interaction, None, torch.tensor([0, 1]), torch.tensor([2, 3])
    _, scores, _, _ = trainer._full_sort_batch_eval(batch)
    assert torch.isneginf(scores[:, 0]).all()
    assert scores[0, 2] == 1 and scores[1, 3] == 1
    trainer.config['exclude_seen'] = True
    _, scores, _, _ = trainer._full_sort_batch_eval(batch)
    assert torch.isneginf(scores[0, 1:3]).all() and torch.isneginf(scores[1, 3])
    assert scores[0, 3] == 1
