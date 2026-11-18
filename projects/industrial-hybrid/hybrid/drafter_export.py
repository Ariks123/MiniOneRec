"""Export the validated UniSRec history encoder from its original RecBole runtime."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


class PortableHistory(torch.nn.Module):
    """Use the trained modules with batch/device-independent position IDs."""
    def __init__(self, model):
        super().__init__()
        self.table, self.adaptor = model.plm_embedding, model.moe_adaptor
        self.positions, self.norm = model.position_embedding, model.LayerNorm
        self.encoder = model.trm_encoder
        self.register_buffer('pos', torch.arange(model.max_seq_length).unsqueeze(0))

    def forward(self, internal_ids, lengths):
        item_vectors = self.adaptor(self.table(internal_ids))
        hidden = self.norm(item_vectors + self.positions(self.pos))
        mask = torch.tril(internal_ids.ne(0)[:, None, None, :].expand(-1, 1, internal_ids.shape[1], -1))
        mask = torch.where(mask, 0.0, -10000.0)
        last = self.encoder(hidden, mask, output_all_encoded_layers=True)[-1]
        gather = (lengths - 1).reshape(-1, 1, 1).expand(-1, 1, last.size(-1))
        return torch.nn.functional.normalize(last.gather(1, gather).squeeze(1), dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'config', 'checkpoint', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--expected-checkpoint-sha256',
                    default='50f8bd01b5bdb5251ee1019ba0a3a003a6fd8ecfc987179cd3f16d13c36a64cb')
    args = ap.parse_args()
    sys.path.insert(0, str(args.source))
    from qwen_runtime import load_experiment, load_trained_checkpoint
    torch.set_num_threads(4)
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError('UniSRec checkpoint differs from selected checkpoint hash')
    _, dataset, _, model = load_experiment(args.config, device='cpu')
    checkpoint = load_trained_checkpoint(model, args.checkpoint)
    model.eval()
    mapping = dataset.field2token_id[dataset.iid_field]
    internal_ids = [int(mapping[str(i)]) for i in range(3686)]
    assert len(set(internal_ids)) == 3686 and min(internal_ids) == 1
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        catalog = torch.cat([model.encode_items(chunk) for chunk in torch.tensor(internal_ids).split(128)])
        np.save(args.output / 'item_vectors.npy', catalog.numpy(), allow_pickle=False)
        wrapper = PortableHistory(model).eval()
        ids = torch.randint(1, 3687, (4, 10))
        lengths = torch.tensor([3, 4, 5, 10])
        ids = ids.masked_fill(torch.arange(10)[None, :] >= lengths[:, None], 0)
        traced = torch.jit.trace(wrapper, (ids, lengths))
        traced.save(str(args.output / 'encoder.pt'))
        restored = torch.jit.load(str(args.output / 'encoder.pt')).eval()
        errors = []
        for batch in (1, 4, 16):
            sample = torch.randint(1, 3687, (batch, 10))
            lens = torch.arange(batch) % 10 + 1
            sample = sample.masked_fill(torch.arange(10)[None, :] >= lens[:, None], 0)
            golden = model.encode_history(sample, lens)
            observed = restored(sample, lens)
            torch.testing.assert_close(observed, golden, atol=1e-6, rtol=1e-5)
            errors.append((golden-observed).abs().max().item())
        # A GPU loader can check parity against the original RecBole output too.
        np.savez(args.output / 'parity.npz', internal_ids=ids.numpy(), lengths=lengths.numpy(),
                 expected=model.encode_history(ids, lengths).numpy())
    info = {'internal_ids_by_item_id': internal_ids, 'checkpoint_sha256': checkpoint_sha,
            'checkpoint_epoch_zero_based': checkpoint['epoch'], 'provenance': model.qwen_provenance,
            'checksums': {n:sha(args.output/n) for n in ('item_vectors.npy','encoder.pt','parity.npz')},
            'max_abs_error_cpu': max(errors), 'item_count': len(internal_ids)}
    (args.output/'metadata.json').write_text(json.dumps(info))
    print('UniSRec export verified', info['item_count'], info['max_abs_error_cpu'])


if __name__ == '__main__':
    main()
