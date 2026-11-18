"""Forward-only checks: these tests never create an optimizer or update weights."""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("recbole")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recbole.config import Config

from qwen_dataset import QwenUniSRecDataset
from qwen_model import QwenUniSRec


@pytest.fixture(params=[np.float32, np.float16], ids=["float32", "float16"])
def fixture_data(tmp_path, request):
    folder = tmp_path / "Tiny"
    folder.mkdir()
    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
    for split, rows in {
        "train": "u\t0\t10\nu\t0 10\t20\n",
        "valid": "u\t0 10 20\t30\n",
        "test": "u\t0 10 20 30\t10\n",
    }.items():
        (folder / f"Tiny.{split}.inter").write_text(header + rows)
    (folder / "Tiny.item").write_text("item_id:token\n0\n10\n20\n30\n")
    features = np.random.default_rng(7).normal(size=(4, 2560)).astype(request.param)
    feature_path = folder / "Tiny.emb-qwen-td.npy"
    np.save(feature_path, features)
    (folder / "Tiny.emb-qwen-td.npy.ids.json").write_text(json.dumps([20, 0, 30, 10]))
    (folder / "train_item_ids.json").write_text(json.dumps([0, 10, 20]))
    config_dict = {
        "data_path": str(tmp_path),
        "benchmark_filename": ["train", "valid", "test"],
        "alias_of_item_id": ["item_id_list"],
        "load_col": {"inter": ["user_id", "item_id_list", "item_id"], "item": ["item_id"]},
        "train_neg_sample_args": None,
        "train_stage": "inductive_ft",
        "train_catalog_only": True,
        "MAX_ITEM_LIST_LENGTH": 4,
        "use_gpu": False,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_size": 300,
        "inner_size": 64,
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "loss_type": "CE",
        "lambda": 1e-3,
        "plm_size": 2560,
        "adaptor_layers": [2560, 300],
        "adaptor_dropout_prob": 0.0,
        "temperature": 0.07,
        "n_exps": 2,
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "group_by": "user", "mode": "full"},
    }
    config = Config(model=QwenUniSRec, dataset="Tiny", config_dict=config_dict)
    dataset = QwenUniSRecDataset(config)
    train, valid, test = dataset.build()
    model = QwenUniSRec(config, train).eval()
    return config, dataset, train, valid, test, model, features


def test_feature_mapping_and_supplied_splits(fixture_data):
    _, dataset, train, valid, test, model, features = fixture_data
    assert [len(part) for part in (train, valid, test)] == [2, 1, 1]
    raw_zero = dataset.field2token_id[dataset.iid_field]["0"]
    assert raw_zero != 0
    torch.testing.assert_close(dataset.plm_embedding.weight[raw_zero], torch.from_numpy(features[1]).float())
    assert torch.count_nonzero(dataset.plm_embedding.weight[0]) == 0
    assert not model.plm_embedding.weight.requires_grad
    assert model.item_embedding is None
    assert all(parameter.requires_grad for parameter in model.moe_adaptor.parameters())
    assert all(parameter.requires_grad for parameter in model.trm_encoder.parameters())


def test_query_item_and_full_sort_scores_agree(fixture_data):
    _, dataset, _, valid, _, model, features = fixture_data
    interaction = valid.inter_feat
    with torch.no_grad():
        queries = model.encode_history(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
        candidates = model.encode_items()
        scores = model.full_sort_predict(interaction)
        external = model.encode_features(features)
        assert queries.shape == (1, 300)
        assert candidates.shape == (5, 300)
        assert external.shape == (4, 300)
        torch.testing.assert_close(scores[:, 1:], (queries @ candidates.T)[:, 1:])
        assert torch.isneginf(scores[:, 0]).all()
        torch.testing.assert_close(queries.norm(dim=-1), torch.ones(1))
        torch.testing.assert_close(candidates[1:].norm(dim=-1), torch.ones(4))
        raw_20 = dataset.field2token_id[dataset.iid_field]["20"]
        torch.testing.assert_close(external[0], candidates[raw_20])
        torch.testing.assert_close(
            model.predict(interaction),
            scores.gather(1, interaction[model.ITEM_ID].unsqueeze(1)).squeeze(1),
        )


def test_ce_uses_train_catalog_without_cold_or_pad(fixture_data):
    _, dataset, train, _, _, model, _ = fixture_data
    cold_id = dataset.field2token_id[dataset.iid_field]["30"]
    assert cold_id not in model.train_candidate_ids.tolist()
    assert 0 not in model.train_candidate_ids.tolist()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        interaction = train.inter_feat
        queries = model.encode_history(interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN])
        candidates = model.encode_items(model.train_candidate_ids)
        expected = torch.nn.functional.cross_entropy(
            queries @ candidates.T / model.temperature,
            model.train_class_by_item[interaction[model.POS_ITEM_ID]],
        )
        loss = model.calculate_loss(interaction)
        torch.testing.assert_close(loss, expected)
        assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_invalid_stage_and_features_are_rejected(fixture_data):
    config, _, train, _, _, model, _ = fixture_data
    with pytest.raises(ValueError, match="last dimension"):
        model.encode_features(torch.zeros(1, 768))
    config["train_stage"] = "transductive_ft"
    with pytest.raises(ValueError, match="inductive_ft"):
        QwenUniSRec(config, train)
