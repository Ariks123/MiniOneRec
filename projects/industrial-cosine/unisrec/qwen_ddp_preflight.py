"""Distributed loading/communication/forward checks; never trains or updates weights."""

import argparse
import hashlib
import json
from pathlib import Path


def _parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preflight(config_path='props/qwen_industrial_4gpu.yaml', expected_world_size=4):
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from qwen_runtime import (load_experiment, runtime_environment, distributed_rank,
                              distributed_world_size)

    torch.set_num_threads(2)
    if runtime_environment()['world_size'] != expected_world_size:
        raise ValueError(f'Launch with torchrun --nproc-per-node={expected_world_size}')
    config, dataset, (train, valid, test), model = load_experiment(config_path)
    model.eval()
    before = _parameter_digest(model)
    world_size, rank = distributed_world_size(), distributed_rank()
    assert not config['single_spec']
    assert isinstance(train.sampler, DistributedSampler)
    assert train.batch_size * world_size == config['train_batch_size']
    assert not isinstance(valid.sampler, DistributedSampler)
    assert not isinstance(test.sampler, DistributedSampler)
    # Full evaluation must retain every supplied example exactly once.
    valid_rows = sum(len(batch[0]) for batch in valid)
    test_rows = sum(len(batch[0]) for batch in test)
    assert valid_rows == len(valid.dataset) and test_rows == len(test.dataset)
    assert model.item_embedding is None and not model.plm_embedding.weight.requires_grad

    train.sampler.set_epoch(0)
    indices = list(iter(train.sampler))
    batch = train.dataset.inter_feat[indices[:2]].to(config['device'])
    with torch.no_grad():
        queries = model.encode_history(batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
        candidate_ids = model.train_candidate_ids[:32]
        items = model.encode_items(candidate_ids)
        scores = queries @ items.T
        assert queries.shape == (2, 300) and scores.shape == (2, len(candidate_ids))
        assert torch.isfinite(scores).all()
        torch.testing.assert_close(queries.norm(dim=-1), torch.ones(2, device=queries.device))
        torch.testing.assert_close(items.norm(dim=-1), torch.ones(len(candidate_ids), device=items.device))
    assert _parameter_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())

    record = {'rank': rank, 'device': str(config['device']), 'local_batch_size': train.batch_size,
              'train_shard_indices': indices, 'parameter_sha256': before,
              'qwen_provenance': model.qwen_provenance,
              'valid_rows': valid_rows, 'test_rows': test_rows}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, record)
    assert all(row['qwen_provenance'] == gathered[0]['qwen_provenance'] for row in gathered)
    assert all(row['parameter_sha256'] == gathered[0]['parameter_sha256'] for row in gathered)
    all_indices = [index for row in gathered for index in row['train_shard_indices']]
    assert set(all_indices) == set(range(len(train.dataset)))
    assert len({tuple(row['train_shard_indices']) for row in gathered}) == world_size
    # Exercise the actual trainer's rank-0 result/error dispatcher without
    # constructing a Trainer or optimizer, and without running a training loop.
    from qwen_train import primary_call
    message = primary_call(lambda: {'valid_rows': valid_rows, 'test_rows': test_rows,
                                    'stop': False, 'executed_rank': rank}, 'preflight broadcast')
    assert message == {'valid_rows': valid_rows, 'test_rows': test_rows,
                       'stop': False, 'executed_rank': 0}
    def expected_failure():
        raise ValueError('expected preflight error propagation')
    caught_expected_error = False
    try:
        primary_call(expected_failure, 'preflight error broadcast')
    except RuntimeError as error:
        caught_expected_error = 'expected preflight error propagation' in str(error)
    assert caught_expected_error
    return {'status': 'passed', 'training_started': False, 'optimizer_created': False,
            'backward_called': False, 'parameters_unchanged': True,
            'world_size': world_size, 'backend': dist.get_backend(),
            'gpu_allocated': config['device'].type == 'cuda',
            'global_batch_size': config['train_batch_size'],
            'per_rank_batch_size': train.batch_size,
            'train_rows': len(train.dataset),
            'training_sampler_padding_rows': len(all_indices) - len(set(all_indices)),
            'validation_rows_exact': valid_rows, 'test_rows_exact': test_rows,
            'evaluation_sharded': False, 'query_dimension': 300,
            'rank_parameter_digests_equal': True,
            'rank_provenance_equal': True, 'rank0_broadcast_passed': True,
            'rank0_error_broadcast_passed': True,
            'ranks': [{key: value for key, value in row.items()
                       if key not in ('train_shard_indices', 'qwen_provenance')} for row in gathered],
            'qwen_provenance': model.qwen_provenance}


def main():
    from qwen_runtime import cleanup_distributed, is_primary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='props/qwen_industrial_4gpu.yaml')
    parser.add_argument('--expect-world-size', type=int, default=4)
    parser.add_argument('--report')
    args = parser.parse_args()
    try:
        result = preflight(args.config, args.expect_world_size)
        if is_primary():
            output = json.dumps(result, ensure_ascii=False, indent=2)
            if args.report:
                path = Path(args.report)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(output + '\n', encoding='utf-8')
            print(output)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
