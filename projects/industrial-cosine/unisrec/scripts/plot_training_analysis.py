#!/usr/bin/env python3
"""Render completed-run diagnostics with Matplotlib; no model loading/training."""

import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    run = args.run_dir
    analysis = json.loads((run/'analysis/analysis.json').read_text())
    records = [json.loads(line) for path in (run/'checkpoints').glob('*.vector_quality.jsonl')
               for line in path.read_text().splitlines()]
    valid = [row for row in records if not row['checkpoint_loaded']]
    epochs = [row['epoch_index'] + 1 for row in valid]
    best = min(valid, key=lambda row: row['summary']['vector_nll'])['epoch_index'] + 1
    log = (run/'train.log').read_text(errors='replace')
    rates = {int(epoch): float(rate) for epoch, rate in re.findall(
        r'Epoch (\d+) learning rate: ([\deE.+-]+)', log)}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), constrained_layout=True)
    colors = ['#2563a6', '#d47720', '#278568']
    ax = axes[0, 0]
    ax.plot(epochs, [r['summary']['vector_nll'] for r in valid], 'o-',
            color=colors[0], markersize=3, label='Validation vector NLL')
    ax.axvline(best, color='#6b7280', linestyle='--', linewidth=1)
    ax.set(title='Validation improves early, then regresses', xlabel='Epoch', ylabel='Vector NLL (lower is better)')
    twin = ax.twinx()
    twin.plot(epochs, [rates[e] for e in epochs], color=colors[1], alpha=.8, label='Learning rate')
    twin.set_ylabel('Learning rate', color=colors[1])
    twin.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    ax.annotate(f'Best epoch: {best}', (best, min(r['summary']['vector_nll'] for r in valid)),
                xytext=(2, 7.24), arrowprops={'arrowstyle': '->', 'color': '#6b7280'}, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles+h2, labels+l2, loc='upper left', fontsize=8)

    ax = axes[0, 1]
    for key, label, color in [('queries', 'History queries', colors[0]),
                              ('candidates', 'Catalog items', colors[2])]:
        ax.plot(epochs, [r['geometry'][key]['covariance_effective_rank'] for r in valid],
                'o-', markersize=3, label=label, color=color)
    ax.axvline(best, color='#6b7280', linestyle='--', linewidth=1)
    ax.set(title='Representation concentration increases (dimension = 300)',
           xlabel='Epoch', ylabel='Covariance effective rank')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    models = [('trained', 'Trained UniSRec'), ('raw_qwen_history_mean', 'Raw Qwen mean'),
              ('raw_qwen_last_item', 'Raw Qwen last item')]
    positions = np.arange(2)
    for index, (key, label) in enumerate(models):
        summary = analysis['trained'] if key == 'trained' else analysis['baselines'][key]
        values = [summary['vector_nll'], summary['groups']['train_unseen']['vector_nll']]
        bars = ax.bar(positions + (index-1)*.24, values, width=.24, color=colors[index], label=label)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    ax.axhline(analysis['trained']['uniform_catalog_nll'], color='#9ca3af',
               linestyle=':', label='Uniform catalog')
    ax.set_xticks(positions, ['All test rows (n=4533)', 'Cold target rows (n=147)'])
    ax.set(title='Same catalog and temperature; CPU baselines, no training', ylabel='Test vector NLL')
    ax.set_ylim(0, max(9, ax.get_ylim()[1]))
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[1, 1]
    for key, label, color in [('positive_cosine_mean', 'Observed target', colors[0]),
                             ('hardest_negative_cosine_mean', 'Hardest alternative', colors[1]),
                             ('negative_mean_cosine_mean', 'Mean alternative', colors[2])]:
        ax.plot(epochs, [r['summary'][key] for r in valid], label=label, color=color)
    ax.axvline(best, color='#6b7280', linestyle='--', linewidth=1)
    ax.set(title='Absolute cosine alone does not establish matching quality', xlabel='Epoch', ylabel='Validation cosine')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('UniSRec Industrial: training and vector-quality diagnostics', fontsize=15)
    output = run/'analysis/figures'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(output/f'diagnostics.{suffix}', dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
