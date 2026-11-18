"""CPU loading and forward checks only: no optimizer, backward, or trainer.fit."""

import argparse
import json
from pathlib import Path


def preflight(config_path='props/qwen_industrial.yaml'):
    import torch
    from qwen_runtime import load_experiment
    torch.set_num_threads(2)
    config, dataset, loaders, model = load_experiment(config_path, device='cpu')
    model.eval()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert model.item_embedding is None
    assert not model.plm_embedding.weight.requires_grad
    assert all(p.requires_grad for p in model.moe_adaptor.parameters())
    assert all(p.requires_grad for p in model.trm_encoder.parameters())
    assert torch.count_nonzero(model.plm_embedding.weight[0]) == 0
    raw_zero = dataset.field2token_id[dataset.iid_field].get('0')
    if raw_zero is not None:
        assert raw_zero > 0
        assert torch.count_nonzero(model.plm_embedding.weight[raw_zero]) > 0
    counts = [len(loader.dataset) for loader in loaders]
    # Benchmark loaders must retain exactly the supplied rows.
    import csv
    expected = []
    for name in ('train', 'valid', 'test'):
        with (Path(config['data_path']) / (config['dataset'] + '.' + name + '.inter')).open() as handle:
            expected.append(sum(1 for _ in csv.reader(handle, delimiter='\t')) - 1)
    assert counts == expected, (counts, expected)
    batch = loaders[0].dataset.inter_feat[:2].to('cpu')
    with torch.no_grad():
        query = model.encode_history(batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
        items = model.encode_items()
        scores = model.full_sort_predict(batch)
        torch.testing.assert_close(scores[:, 1:], (query @ items.T)[:, 1:], rtol=1e-4, atol=1e-5)
        assert torch.isneginf(scores[:, 0]).all()
        torch.testing.assert_close(items[1:].norm(dim=-1), torch.ones(dataset.item_num - 1),
                                   rtol=1e-4, atol=1e-5)
        assert query.shape == (2, 300)
        loss = model.calculate_loss(batch)
        assert torch.isfinite(loss)
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())
    assert all(value.grad is None for value in model.parameters())
    return {'status': 'passed', 'training_started': False, 'optimizer_created': False,
            'backward_called': False, 'split_rows': dict(zip(('train','valid','test'), counts)),
            'catalog_items': dataset.item_num - 1,
            'train_candidates': model.train_candidate_ids.numel(),
            'fixed_feature_shape': list(model.plm_embedding.weight.shape),
            'query_dimension': query.shape[-1], 'raw_item_zero_internal_id': raw_zero,
            'score_consistency': 'query @ normalized_items.T equals full_sort_predict',
            'parameters_unchanged': True, 'qwen_provenance': model.qwen_provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='props/qwen_industrial.yaml')
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    result = preflight(args.config)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
