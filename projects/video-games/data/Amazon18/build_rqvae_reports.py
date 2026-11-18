#!/usr/bin/env python3
"""Build source-driven RQ-VAE training and SID validation PDF reports."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import csv
import hashlib
import json
import math
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter, NullFormatter, LogLocator
from build_embedding_similarity_report import clean, label, line, style, table, paragraph, INK, MUTED, TEAL, BLUE, GOLD, RED, GRID

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / 'output/reports/Video_Games'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def scalar(value, digits=4, percent=False):
    if value is None:
        return '未记录'
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return '不可用'
    if percent:
        return f'{value:.{digits}%}'
    if isinstance(value, (int, np.integer)):
        return f'{value:,}'
    if isinstance(value, (float, np.floating)):
        return f'{value:,.{digits}f}'
    return clean(value)


def header(title, subtitle, page, page_count, stage):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.add_artist(Rectangle((.065, .947), .05, .006, transform=fig.transFigure, color=TEAL, lw=0))
    label(fig, .065, .925, title, 22)
    label(fig, .065, .883, subtitle, 8.6, MUTED)
    line(fig, .055)
    label(fig, .065, .041, f'MiniOneRec  |  Video Games  |  {stage}', 7.8, MUTED)
    label(fig, .94, .041, f'{page} / {page_count}', 7.8, MUTED, ha='right')
    return fig


def metric_cards(fig, metrics, y=.837):
    count = len(metrics)
    for index, (name, value, context) in enumerate(metrics):
        x = .065 + index * .875 / count
        label(fig, x, y, name, 9.2, MUTED)
        units=sum(2 if __import__('unicodedata').east_asian_width(char) in 'WF' else 1 for char in str(value))
        value_size=min(23 if count<=3 else 20, (.875/count*595.44)/(max(1,units)*.53))
        label(fig, x, y-.027, value, value_size, TEAL)
        label(fig, x, y-.068, context, 8.1, MUTED)


def chart_line(ax, xs, values, name, color=TEAL, *, percent=False, log=False):
    pairs = [(x, y) for x, y in zip(xs, values) if y is not None and np.isfinite(y)]
    if pairs:
        x, y = np.asarray(pairs, dtype=float).T
        ax.plot(x, y, color=color, linewidth=1.6, label=name)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.3g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    style(ax)
    return len(pairs)


def plot_missing(ax, message='该统计未记录'):
    ax.axis('off')
    ax.text(.5, .5, message, color=MUTED, ha='center', va='center', transform=ax.transAxes, fontsize=10)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def information_table(fig, rows, area, *, font_size=8.8):
    clean_rows = [['\n'.join(textwrap.wrap(clean(name), width=12)), '\n'.join(textwrap.wrap(clean(value), width=78, break_long_words=True, break_on_hyphens=False))] for name, value in rows]
    return table(fig, ['项目', '记录'], clean_rows, area, [.25, .75], font_size, ['left', 'left'])


class Report:
    def __init__(self, output_dir, stem, title, source, pages):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stem, self.source, self.page_count = stem, Path(source), pages
        self.pdf_path = self.output_dir / f'{stem}.pdf'
        self.pdf = PdfPages(self.pdf_path, metadata={'Title': title, 'Author': 'MiniOneRec', 'Subject': '实际运行统计、可视化与方法边界'})
        self.images = []
        self.tables = []

    def save(self, fig):
        page = len(self.images) + 1
        self.pdf.savefig(fig)
        path = self.output_dir / f'{self.stem}-{page}.png'
        fig.savefig(path, dpi=160, facecolor='white')
        self.images.append(path.name)
        plt.close(fig)

    def csv(self, name, rows, fieldnames=None):
        path = self.output_dir / name
        write_csv(path, rows, fieldnames)
        self.tables.append(path.name)
        return path

    def finish(self, details):
        self.pdf.close()
        if len(self.images) != self.page_count:
            raise ValueError(f'Expected {self.page_count} pages, generated {len(self.images)}')
        manifest = {'pdf': str(self.pdf_path), 'pages': self.page_count, 'page_images': self.images,
                    'tables': self.tables, 'source': str(self.source.resolve()), 'source_sha256': sha256(self.source),
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), **details}
        (self.output_dir / 'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return manifest



def load_training(run_dir):
    run = Path(run_dir).resolve(strict=True)
    manifest = read_json(run / 'manifest.json')
    summary_path = Path(manifest['artifacts']['training_summary'])
    summary = read_json(summary_path)
    if summary['status'] != 'completed':
        raise ValueError('The training report requires a completed training summary')
    evaluations = read_json(manifest['artifacts']['evaluation_history'])
    with Path(manifest['artifacts']['epoch_metrics']).open(encoding='utf-8-sig', newline='') as source:
        epochs = [{key: (int(value) if key == 'epoch' else float(value)) for key, value in row.items()} for row in csv.DictReader(source)]
    if not epochs or not evaluations:
        raise ValueError('Completed training metrics are empty')
    if epochs[-1]['epoch'] != summary['epochs_completed']:
        raise ValueError('Epoch history does not match the completed summary')
    if not math.isclose(summary['selected_metrics']['val']['mse'], summary['best_val_mse'], rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError('Selected validation MSE does not match the recorded best checkpoint')
    return manifest, summary, evaluations, epochs, summary_path


def training_report(run_dir, output_dir):
    manifest, summary, evaluations, epochs, source = load_training(run_dir)
    config, selected = summary['config'], summary['selected_metrics']
    report = Report(output_dir, 'video-games-rqvae-training-report', 'Video Games RQ-VAE 训练报告', source, 4)
    ex = [row['epoch'] for row in evaluations]
    tx = [row['epoch'] for row in epochs]
    stop_labels = {'validation_plateau': '验证误差进入平台期，触发早停', 'maximum_epochs': '达到训练轮数上限'}
    stop_reason = stop_labels.get(summary['stop_reason'], clean(summary['stop_reason']))
    stamp = datetime.fromisoformat(summary['completed_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    fig = header(f"RQ-VAE 训练：采用 epoch {summary['best_epoch']:,}", f"实际完成 {summary['epochs_completed']:,} 轮  |  当前采用验证最佳权重  |  {stamp} 北京时间", 1, 4, 'RQ-VAE Training')
    metric_cards(fig, [('最佳验证 MSE', scalar(summary['best_val_mse'], 6), '原始商品向量重构误差'),
                       ('验证 NMSE', scalar(selected['val']['nmse'], 4), '相对训练均值预测器；越低越好'),
                       ('已完成轮数 / 最佳轮次', f"{summary['epochs_completed']:,} / {summary['best_epoch']:,}", '选择验证误差最低的 checkpoint')])
    label(fig, .065, .728, '01  训练与验证重构误差', 13)
    ax = fig.add_axes([.12, .428, .79, .253])
    chart_line(ax, tx, [row['train_recon'] for row in epochs], '训练重构 MSE', TEAL, log=True)
    chart_line(ax, ex, [row['val']['mse'] for row in evaluations], '验证 MSE', BLUE, log=True)
    ax.axhline(summary['baseline_mse']['val'], color=GOLD, ls='--', lw=1.2, label='验证集训练均值基线')
    ax.axvline(summary['best_epoch'], color=RED, ls=':', lw=1.1, label='选中轮次')
    ax.set_xlabel('训练 epoch', fontsize=9)
    ax.set_ylabel('每元素 MSE（对数纵轴）', fontsize=9)
    ax.legend(frameon=False, fontsize=8.0, loc='upper right')
    label(fig, .065, .388, f"停止原因：{stop_reason}。曲线含所有实际记录，不仅展示最佳片段。", 8.7, MUTED)
    label(fig, .065, .349, '02  选中权重的重构质量', 13)
    rows = []
    for key, title in [('train', '训练商品'), ('val', '验证商品'), ('all', '全部商品')]:
        metric = selected[key]
        rows.append([title, scalar(summary['split_counts'][key]), scalar(metric['mse'], 6),
                     scalar(metric['baseline_mse'], 6), scalar(metric['nmse'], 4), scalar(metric['cosine_mean'], 4)])
    table(fig, ['范围', '商品数', '模型 MSE', '均值基线 MSE', 'NMSE', '重构 cos 均值'], rows,
          [.065, .192, .875, .127], [.17,.12,.18,.20,.13,.20], 8.5)
    paragraph(fig, .065, .155, 'NMSE = 模型 MSE / 均值基线 MSE。基线只用训练商品向量的均值；NMSE 小于 1 表示优于该基线。验证商品未参与梯度训练和码本初始化。', 8.7, width=113)
    label(fig, .065, .091, '重构与量化质量不等同于推荐准确率；本阶段没有训练 SFT 或 RL 推荐模型。', 8.5, MUTED)
    report.save(fig)

    fig = header('码本使用量与原始编码碰撞', '曲线来自每次评估的全部商品  |  未进行最终 SID 后缀去重', 2, 4, 'RQ-VAE Training')
    label(fig, .065, .838, '03  三层码本使用量随训练变化', 13)
    ax = fig.add_axes([.12, .598, .79, .193])
    for level, color in enumerate([TEAL, BLUE, GOLD]):
        chart_line(ax, ex, [row['all']['codebook'][level]['used'] for row in evaluations], f'第 {level+1} 层', color)
    ax.set_xlabel('训练 epoch', fontsize=9)
    ax.set_ylabel('有商品使用的码字数', fontsize=9)
    ax.axhline(config['num_emb_list'][0], color=GRID, ls='--')
    ax.legend(frameon=False, ncol=3, loc='lower right', fontsize=8.3)
    label(fig, .065, .554, '04  选中权重的码本分布', 13)
    usage_rows = [[f'第 {index+1} 层', f"{metric['used']} / {config['num_emb_list'][index]}", scalar(metric['dead']),
                   scalar(metric['entropy'], 3), scalar(metric['perplexity'], 2), scalar(metric['max_share'], 2, percent=True)]
                  for index, metric in enumerate(selected['all']['codebook'])]
    table(fig, ['层', '使用码字 / 容量', '未使用码字', '熵（nats）', '困惑度', '最大单码占比'], usage_rows,
          [.065, .404, .875, .117], [.12,.25,.16,.15,.16,.16], 8.5)
    label(fig, .065, .378, '困惑度 = exp(熵)，是使用均匀程度的描述；未使用码字不自动代表训练失败。', 8.3, MUTED)
    label(fig, .065, .340, '05  最近码字编码的碰撞率', 13)
    ax = fig.add_axes([.12, .143, .79, .155])
    for key, title, color in [('train', '训练商品', TEAL), ('val', '验证商品', BLUE), ('all', '全部商品', GOLD)]:
        chart_line(ax, ex, [row[key]['collision_rate'] for row in evaluations], title, color, percent=True)
    ax.set_xlabel('训练 epoch', fontsize=9)
    ax.set_ylabel('原始三层码碰撞率', fontsize=9)
    ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=8)
    label(fig, .065, .100, '碰撞率 = 1 - 唯一三层编码数 / 商品数；不含后续 Sinkhorn 调整与身份后缀。', 8.4, MUTED)
    report.save(fig)

    fig = header('优化目标、学习率与初始化', '损失曲线使用全部 epoch；初始化只使用训练商品', 3, 4, 'RQ-VAE Training')
    label(fig, .065, .838, '06  训练损失分量', 13)
    ax = fig.add_axes([.12, .541, .79, .248])
    all_positive = all(row[key] > 0 for row in epochs for key in ('train_total', 'train_recon', 'train_quant'))
    for key, title, color in [('train_total', '总损失', TEAL), ('train_recon', '重构损失', BLUE), ('train_quant', '量化损失', GOLD)]:
        chart_line(ax, tx, [row[key] for row in epochs], title, color, log=all_positive)
    ax.set_xlabel('训练 epoch', fontsize=9)
    ax.set_ylabel('损失' + ('（对数纵轴）' if all_positive else ''), fontsize=9)
    ax.legend(frameon=False, ncol=3, fontsize=8.3)
    label(fig, .065, .500, f"总损失 = 重构损失 + {config['quant_loss_weight']:g} × 量化损失；commitment beta = {config['beta']:g}。", 8.6, MUTED)
    label(fig, .065, .462, '07  KMeans++ 残差码本初始化', 13)
    rows = [[f"第 {row['level']} 层", scalar(row['training_items']), scalar(row['k']), scalar(row['n_init']),
             scalar(row['iterations']), scalar(row['residual_mse_after'], 8)] for row in summary['initialization']]
    table(fig, ['层', '训练商品数', '码字数', 'n_init', '实际迭代', '残差 MSE'], rows,
          [.065, .306, .875, .124], [.12,.20,.15,.15,.16,.22], 8.6)
    label(fig, .065, .280, '每层对上一层残差初始化；迭代上限与随机种子记录在配置及源 JSON 中。', 8.3, MUTED)
    label(fig, .065, .242, '08  实际学习率', 13)
    ax = fig.add_axes([.12, .112, .79, .095])
    chart_line(ax, tx, [row['lr'] for row in epochs], '学习率', TEAL)
    ax.set_xlabel('训练 epoch', fontsize=9)
    ax.set_ylabel('lr', fontsize=9)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    report.save(fig)

    fig = header('训练配置、资源与复现记录', '保留原 RQ-VAE 结构；新增分组验证、最佳权重选择与早停', 4, 4, 'RQ-VAE Training')
    label(fig, .065, .838, '09  数据与模型配置', 13)
    rows = [
        ('输入', f"{summary['split_counts']['all']:,} 件商品 × 2560 维；原始 float32 向量"),
        ('训练 / 验证', f"{summary['split_counts']['train']:,} / {summary['split_counts']['val']:,} 件商品"),
        ('分组留出', '相同清洗后标题+描述归为同组；分组划分，减少相同文本跨集泄漏'),
        ('编码器 / 解码器', '2560 -> ' + ' -> '.join(map(str, config['layers'])) + f" -> {config['e_dim']}；解码器对称"),
        ('码本', ' / '.join(map(str, config['num_emb_list'])) + f"；每码字维度 {config['e_dim']}"),
        ('优化器', f"{config['learner']}；lr={config['lr']:g}；weight_decay={config['weight_decay']:g}"),
        ('批量 / warmup', f"batch_size={config['batch_size']:,}；warmup={config['warmup_epochs']} epochs"),
        ('初始化', f"KMeans++；n_init={config['kmeans_n_init']}；每层至多 {config['kmeans_iters']} 轮；seed={config['seed']}"),
        ('评估频率', f"每 {config['eval_step']} epochs，记录 train / val / all"),
        ('停止条件', f"最少 {config['min_epochs']:,}，最多 {config['epochs']:,} epochs；连续 {config['patience']} 次评估无至少 {config['relative_improvement']:.2%} 相对改进"),
        ('权重选择', f"验证 MSE 最低；选中 epoch {summary['best_epoch']:,}；{Path(summary['selected_checkpoint']).name}"),
    ]
    information_table(fig, rows, [.065, .398, .875, .402], font_size=8.3)
    label(fig, .065, .365, '10  资源与运行结果', 13)
    resources = [('停止原因', stop_reason), ('GPU', summary.get('gpu', '未记录')),
                 ('训练器记录耗时', f"{summary['training_seconds']/60:.2f} 分钟 / {summary['training_seconds']:,.2f} 秒"),
                 ('checkpoint 恢复', f"{len(manifest.get('resume_events', []))} 次；事件记录见 manifest.resume_events"),
                 ('已记录的训练阶段峰值显存', scalar(summary.get('peak_gpu_allocated_GiB'), 2) + ' GiB')]
    information_table(fig, resources, [.065, .179, .875, .148], font_size=8.3)
    label(fig, .065, .153, '输入向量 SHA-256', 8.8, MUTED)
    label(fig, .065, .131, summary['embedding_sha256'], 8)
    label(fig, .065, .105, '复现文件：manifest.json / split.json / epoch_metrics.csv / evaluation_history.json / training_summary.json', 7.8, MUTED)
    label(fig, .065, .081, '早停与最佳权重选择依据验证重构；码本碰撞和后续 SID 唯一性需分别检查。', 8.3, MUTED)
    report.save(fig)
    report.csv('epoch_metrics.csv', epochs)
    report.csv('evaluation_metrics.csv', [{'epoch': row['epoch'], 'split': key,
               **{metric: row[key][metric] for metric in ('mse','cosine_mean','baseline_mse','nmse','unique_sids','collision_rate')}}
              for row in evaluations for key in ('train','val','all')])
    report.csv('selected_codebook_usage.csv', [{'level': level+1, 'code': code, 'items': count}
              for level, metric in enumerate(selected['all']['codebook']) for code, count in enumerate(metric['counts'])])
    return report.finish({'stage': 'rqvae_training', 'epochs_completed': summary['epochs_completed'],
                          'best_epoch': summary['best_epoch'], 'selected_epoch': summary['best_epoch'], 'best_val_mse': summary['best_val_mse'],
                          'validation_nmse': selected['val']['nmse'], 'stop_reason': summary['stop_reason']})

def warning_groups(warnings, max_lines=39):
    if len(warnings) <= 3:
        return []
    chunks, current, length = [], [], 0
    for warning in warnings:
        units = sum(2 if __import__('unicodedata').east_asian_width(c) in 'WF' else 1 for c in clean(warning))
        lines = max(1, math.ceil(units / 108)) + 1
        if current and length + lines > max_lines:
            chunks.append(current); current, length = [], 0
        current.append(warning); length += lines
    if current:
        chunks.append(current)
    return chunks


def nearest_comparison_case(fig, case, top, bottom):
    label(fig, .065, top, f"商品 ID {case['anchor_id']}  |  {case['title']}", 10.2, TEAL)
    y = top - .031
    columns = [('original', '原始商品向量'), ('raw_reconstruction', '原始三层码重构'), ('semantic_reconstruction', 'SK 调整码重构')]
    for x, (_, title) in zip([.130, .402, .674], columns):
        label(fig, x, y, title, 8.7, MUTED)
    label(fig, .065, y, '#', 8.7, MUTED)
    line(fig, y-.015); y -= .025
    for rank in range(5):
        cells = []
        for stage, _ in columns:
            neighbor = case['neighbors'][stage][rank]
            cells.append([f"ID {neighbor['neighbor_id']} | 原始 cos {neighbor['original_cosine']:.3f}"] + textwrap.wrap(clean(neighbor['title']), width=43, break_long_words=True, break_on_hyphens=False))
        height = max(map(len, cells)) * .0095 + .007
        label(fig, .065, y, rank+1, 8)
        for x, lines in zip([.130, .402, .674], cells):
            for index, value in enumerate(lines):
                label(fig, x, y-index*.0095, value, 6.9, MUTED if index == 0 else INK)
        line(fig, y-height+.003); y -= height
    if y < bottom:
        raise ValueError(f"Neighbor case {case['anchor_id']} overflow: {y:.3f} < {bottom:.3f}")


def comparison_artifact(run, key, filename):
    """Resolve diagnostics from a training run or its endpoint export folder."""
    run = Path(run)
    local = run / filename
    if local.exists():
        return local.resolve()
    manifest_path = run / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    recorded = manifest.get('artifacts', {}).get(key)
    if recorded and Path(recorded).exists():
        return Path(recorded).resolve()
    if (run.parent / filename).exists():
        return (run.parent / filename).resolve()
    raise FileNotFoundError(f'Cannot find comparison artifact {filename}: {run}')


def load_integrated_comparison(data, endpoint_run_dir, comparison_path=None):
    """Require already-computed, source-matched diagnostics; never run inference."""
    baseline = Path(data['run_directory']).resolve(strict=True)
    endpoint = Path(endpoint_run_dir).resolve(strict=True)
    comparison_path = Path(comparison_path or REPORT_ROOT/'05_rqvae_10000_comparison/comparison.json').resolve(strict=True)
    comparison = read_json(comparison_path)
    sources = []
    result = {'comparison': comparison, 'comparison_path': comparison_path}
    for name, run in [('baseline', baseline), ('endpoint', endpoint)]:
        part = {'run_directory': str(run)}
        for key, filename in [('sid_validation', 'sid_validation.json'), ('export_stats', 'export_stats.json'), ('training_summary', 'training_summary.json'), ('evaluation_history', 'evaluation_history.json')]:
            path = comparison_artifact(run, key, filename)
            part[key] = read_json(path)
            sources.append({'run': name, 'artifact': key, 'path': str(path), 'sha256': sha256(path)})
        part['epoch'] = int(part['export_stats']['parameters']['checkpoint_epoch'])
        result[name] = part
    a, b = result['baseline'], result['endpoint']
    av, bv = a['sid_validation'], b['sid_validation']
    checks = dict(comparison['fairness_checks'])
    checks['selected_analysis_matches_baseline'] = data['sources']['checkpoint']['sha256'] == av['sources']['checkpoint']['sha256']
    for source in ['embeddings', 'item_metadata', 'split', 'initial_checkpoint']:
        checks['integrated_identical_' + source] = av['sources'][source]['sha256'] == bv['sources'][source]['sha256']
    checks['integrated_identical_query_ids'] = av['knn']['query_ids'] == bv['knn']['query_ids']
    checks['integrated_different_checkpoint_hashes'] = av['sources']['checkpoint']['sha256'] != bv['sources']['checkpoint']['sha256']
    checks['integrated_endpoint_epoch_10000'] = b['epoch'] == 10000
    for name, part in [('baseline', a), ('endpoint', b)]:
        source_hash = part['sid_validation']['sources']['checkpoint']['sha256']
        checks[name + '_comparison_checkpoint_matches'] = comparison[name]['checkpoint_sha256'] == source_hash
        checks[name + '_comparison_epoch_matches'] = comparison[name]['checkpoint_epoch'] == part['epoch']
        checks[name + '_export_validation_checkpoint_matches'] = part['export_stats']['sources']['checkpoint']['sha256'] == source_hash
        checks[name + '_validation_passed'] = part['sid_validation']['validation'] == 'passed'
    if not all(checks.values()):
        raise ValueError('Integrated comparison source/fairness checks failed: ' + ', '.join(k for k,v in checks.items() if not v))
    sources.append({'run': 'comparison', 'artifact': 'comparison', 'path': str(comparison_path), 'sha256': sha256(comparison_path)})
    result.update(sources=sources, fairness_checks=checks, selected_epoch=a['epoch'])
    return result


def integrated_comparison_pages(report, comparison, pages):
    """Append four source-driven comparison pages before the references page."""
    a, b = comparison['baseline'], comparison['endpoint']
    vals = [a['sid_validation'], b['sid_validation']]
    summaries = [a['training_summary'], b['training_summary']]
    epochs = [a['epoch'], b['epoch']]
    names = [f"采用 {epochs[0]:,} 轮权重", f"候选 {epochs[1]:,} 轮终点"]
    metrics = {row['metric']: row for row in comparison['comparison']['metrics']}
    source_data = comparison['comparison']
    def recon(v, stage, split, metric='mse'):
        row = v['reconstruction'][stage][split]
        return row['cosine']['mean'] if metric == 'cosine' else row['nmse_vs_train_mean'] if metric == 'nmse' else row[metric]
    def nn(v, stage, k=10):
        return v['knn']['metrics'][stage+'_reconstruction']['val_queries'][f'overlap_at_{k}']
    def metric_table(fig, keys, area, size=8.1):
        rows = []
        for key, title in keys:
            row = metrics[key]
            percent = row['kind'] == 'percent'
            value = lambda x: f'{x:.2%}' if percent else f'{x:.6f}'
            delta = f"{row['delta']*100:+.2f} pp" if percent else f"{row['delta']:+.6f}"
            rows.append([title, value(row['baseline']), value(row['endpoint']), delta])
        return table(fig, ['指标', f'{epochs[0]:,} 轮', f'{epochs[1]:,} 轮', '终点 - 原权重'], rows, area, [.43,.18,.18,.21], size)
    def new_page(title, subtitle):
        return header(title, subtitle, len(report.images)+1, pages, 'SID Validation + Comparison')

    fig = new_page(f"当前采用 epoch {epochs[0]:,} 的 SID", '验证报告内的 10,000 轮对比  |  采用原权重作为保守基线，保留终点候选')
    metric_cards(fig, [('当前采用权重', f"{epochs[0]:,}", f"原 run 实际完成 {summaries[0]['epochs_completed']:,} 轮"),
                       ('采用权重验证 MSE', scalar(recon(vals[0], 'raw', 'val'),6), '原始三层最近码字重构'),
                       ('采用权重 SK Top10', scalar(nn(vals[0],'semantic'),2,percent=True), '相同原始向量近邻；不是推荐准确率')])
    label(fig,.065,.727,'19  选择结论与主要取舍',13)
    paragraph(fig,.065,.689,f"当前使用 {epochs[0]:,} 轮权重对应的商品语义码和完整 SID。{epochs[1]:,} 轮的原空间近邻保持更高，但验证重构误差也更高，部分前缀覆盖与完整码噪声稳定性较低。采用原权重是保守的项目基线选择；未证明其用户 HR / NDCG 或生成推荐效果最好。",8.7,width=112)
    metric_table(fig, [('raw_val_mse','Raw 验证 MSE（低更好）'), ('semantic_val_mse','SK 验证 MSE（低更好）'),
                       ('raw_val_top10_original','Raw 原空间 Top10 保留（高更好）'), ('semantic_val_top10_original','SK 原空间 Top10 保留（高更好）'),
                       ('raw_train_mse','Raw 训练 MSE（仅描述拟合）')], [.065,.409,.875,.173])
    label(fig,.065,.371,'20  公平性与真实终点核验',13)
    y=.332
    for text in [
        '商品向量、元数据、分组划分和初始化权重的 SHA-256 相同；近邻查询 ID、候选商品库、模型与优化器核心配置相同。',
        f"原 run 的最后训练轮次为 {summaries[0]['epochs_completed']:,}，选中权重为 {epochs[0]:,}；从原最后权重续训到 {epochs[1]:,}，继承原验证历史，关闭早停。这里导出的是实际终点。",
        f"续训 run 的最佳验证轮次仍为 {summaries[1]['best_epoch']:,}，验证 MSE 为 {summaries[1]['best_val_mse']:.6f}；该最佳值没有冒充 {epochs[1]:,} 轮终点。两套选用 checkpoint 的 SHA-256 不同。",
        '同一随机种子下的单一训练轨迹；没有跨种子重复实验。统一碰撞修复算法及参数，前三层作语义诊断，身份后缀只负责商品区分。']:
        y=paragraph(fig,.065,y,'- '+text,8.5,width=113)-.014
    report.save(fig)

    fig = new_page('重构泛化、近邻与修复代价', '原始向量邻域保留描述商品内容结构；不等于用户推荐 Recall / NDCG')
    label(fig,.065,.838,'21  原最佳附近至 10,000 轮的训练 / 验证 MSE',13)
    hist=b['evaluation_history']
    start=max(0,epochs[0]-700)
    visible=[x for x in hist if x['epoch']>=start]
    ax=fig.add_axes([.12,.558,.79,.227])
    chart_line(ax,[x['epoch'] for x in visible],[x['train']['mse'] for x in visible],'训练 Raw MSE',TEAL)
    chart_line(ax,[x['epoch'] for x in visible],[x['val']['mse'] for x in visible],'验证 Raw MSE',BLUE)
    ax.axvline(epochs[0],color=GOLD,ls='--',lw=1,label=f'采用 {epochs[0]:,}')
    ax.axvline(epochs[1],color=RED,ls=':',lw=1.2,label=f'终点 {epochs[1]:,}')
    ax.set_xlim(start,epochs[1]+80); ax.set_xlabel('训练 epoch',fontsize=8.5); ax.set_ylabel('Raw 重构 MSE',fontsize=8.5)
    ax.legend(frameon=False,ncol=2,fontsize=7.8,loc='center right'); ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
    paragraph(fig,.065,.517,f"训练 MSE {recon(vals[0],'raw','train'):.6f} -> {recon(vals[1],'raw','train'):.6f}，验证 MSE {recon(vals[0],'raw','val'):.6f} -> {recon(vals[1],'raw','val'):.6f}：在重构目标上出现泛化退化。但原空间近邻保持改善，不能概括为所有质量指标均变差。",8.4,width=115)
    label(fig,.065,.448,'22  验证查询的原始 2560D 全库余弦近邻保留',12)
    rows=[]
    for index,v in enumerate(vals):
        for stage,name in [('raw','Raw'),('semantic','SK')]:
            rows.append([f'{epochs[index]:,}',name,*[f'{nn(v,stage,k):.2%}' for k in [5,10,20]]])
    table(fig,['权重轮次','阶段','Top5','Top10','Top20'],rows,[.065,.290,.875,.119],[.23,.14,.21,.21,.21],8.2)
    label(fig,.065,.257,'23  SK 碰撞修复的代价（统一算法与参数）',12)
    metric_table(fig,[('repair_all_mse_delta','全库 SK 重构 MSE 增量'),('repair_val_top10_drop','验证 Top10 因 SK 下降')],[.065,.144,.875,.081],8.1)
    added=source_data['endpoint'].get('additional_training_seconds',summaries[1]['training_seconds']-summaries[0]['training_seconds'])
    paragraph(fig,.065,.114,f"新增续训 {added:.1f} 秒，累计训练 {summaries[1]['training_seconds']:.1f} 秒，不含导出与验证。TopK = 同一原空间近邻交集 / K，排除自身；32D 潜空间参照随编码器改变，仅作内部诊断，不用于本页跨权重选优。",8.0,width=117)
    report.save(fig)

    fig = new_page('前缀覆盖、组内相似与稳定性', '覆盖份额与组内相似度回答不同问题；下降不能相互替代解释')
    label(fig,.065,.838,'24  原向量 Top10 近邻共享前缀：覆盖了多少相似商品',12)
    prefix_rows=[]
    for stage,depth,title in [('raw',1,'前 1 层（Raw / SK 相同）'),('raw',2,'前 2 层（Raw / SK 相同）'),('raw',3,'前 3 层 Raw'),('semantic',3,'前 3 层 SK')]:
        values=[]
        for v in vals:
            row=next(x for x in v['prefix_neighbor_retention']['stages'][stage]['val_queries'] if x['k']==10 and x['prefix_depth']==depth)
            values.append(row['observed'])
        prefix_rows.append([title,f'{values[0]:.3%}',f'{values[1]:.3%}',f'{(values[1]-values[0])*100:+.3f} pp'])
    table(fig,['原始近邻的共享比例',f'{epochs[0]:,}',f'{epochs[1]:,}','变化'],prefix_rows,[.065,.649,.875,.144],[.46,.18,.18,.18],8.1)
    label(fig,.065,.614,'25  LCP 恰好为 2 的商品对：组内原向量余弦均值',12)
    lcp_rows=[]
    lcp_csv=[]
    for stage,name in [('raw','Raw 三层码分组'),('semantic','SK 三层码分组')]:
        pair=[next(x for x in v['prefix_similarity'][stage] if x['lcp']==2) for v in vals]
        lcp_rows.append([name,*[f"{x['original_cosine']['mean']:.6f}" for x in pair],f"{pair[0]['sampled_pairs']:,} / {pair[1]['sampled_pairs']:,}"])
        lcp_csv.extend([{'epoch':epoch,'stage':stage,**{key:item[key] for key in ['lcp','eligible_distinct_text_pairs','sampled_pairs']},'mean_original_cosine':item['original_cosine']['mean']} for epoch,item in zip(epochs,pair)])
    table(fig,['分组阶段',f'{epochs[0]:,}',f'{epochs[1]:,}','各自抽样对数'],lcp_rows,[.065,.509,.875,.069],[.38,.19,.19,.24],8.2)
    paragraph(fig,.065,.488,'前2层近邻覆盖下降，不能写成组内相似度下降；此处均值反而更高。两套分组成员与抽样商品对不同，且排除相同清洗文本，只能作描述比较，不能声称同一商品对的相似度改善或统计显著。',8.0,width=117)
    label(fig,.065,.411,'26  完整三层码噪声稳定性、码本占用与碰撞',12)
    rows=[]
    for index,level in enumerate(vals[0]['noise_stability']['levels']):
        other=vals[1]['noise_stability']['levels'][index]
        rows.append([f"{level['relative_l2_noise']:.1%} 噪声：Raw 三层码保持",f"{level['prefix_3_retention']:.2%}",f"{other['prefix_3_retention']:.2%}"])
    for title,getter,kind in [
        ('Raw L1 使用码字 / 256',lambda v:v['codebook_usage']['raw']['all'][0]['used'],'used'),
        ('Raw L1 困惑度',lambda v:v['codebook_usage']['raw']['all'][0]['perplexity'],'number'),
        ('Raw 三层码碰撞率',lambda v:v['collisions']['raw']['collision_rate'],'percent'),
        ('SK 三层码碰撞率',lambda v:v['collisions']['semantic']['collision_rate'],'percent'),
        ('SK 修改商品数',lambda v:v['collisions']['repair_distortion']['changed_items'],'int')]:
        f=lambda x:f'{x:.2%}' if kind=='percent' else f'{int(x):,} / 256' if kind=='used' else f'{int(x):,}' if kind=='int' else f'{x:.2f}'
        rows.append([title,*[f(getter(v)) for v in vals]])
    table(fig,['诊断',f'{epochs[0]:,}',f'{epochs[1]:,}'],rows,[.065,.151,.875,.222],[.52,.24,.24],8.1)
    paragraph(fig,.065,.122,'噪声每幅度仅一次实现；完整三层码稳定性下降，不表示每个单层前缀都下降。L2/L3 两套均使用 256 个码字。完整 SID 均唯一，身份后缀不参与这些语义指标；占用更均匀、碰撞更少不自动等于推荐更好。',7.9,width=119)
    report.csv('comparison_lcp_exactly_two.csv',lcp_csv)
    report.save(fig)

    fig = new_page('选用口径、推荐适配与资产来源', '当前已选定 2,900 轮 SID；10000 轮产物作为候选保留，尚未做下游训练')
    y=.838
    for title,body in [
        ('27  两种“相似”与不同参照', '重构 MSE / NMSE / cos 检查向量恢复；原空间 TopK 检查内容近邻；近邻共享前缀描述覆盖，LCP 分组余弦描述该组内部相似度。32D 潜空间邻域由各自编码器定义，参照不固定，不能和原空间保留率混成一个质量分数。'),
        ('28  从可用 SID 到生成推荐', '这套 SID 可作为后续生成推荐的商品目标标识。仍需按用户时间顺序构造历史 -> 下一商品的样本，使用同一商品 ID / ASIN / SID 映射，并让训练与推理使用一致的 SID token 和合法商品目录。本阶段未执行 SFT / RL，也未证明哪套 SID 的 HR / NDCG 更优。'),
        ('29  选择的边界', f"{epochs[0]:,} 轮具有较低验证重构误差，部分前缀覆盖和完整码扰动保持也更高，适合作为当前保守基线；{epochs[1]:,} 轮在原空间邻域保持、码本占用和碰撞等诊断上有优势。当前选择不是单指标普遍最优证明，更不是下游推荐效果结论。")]:
        label(fig,.065,y,title,11.5,TEAL)
        y=paragraph(fig,.065,y-.028,body,8.5,width=114)-.022
    label(fig,.065,y,'30  采用与候选资产的可复核来源',11.5,TEAL)
    y-=.032
    asset_rows=[]
    for name,part in [('采用',a),('候选',b)]:
        v=part['sid_validation']
        asset_rows.append([f"{name} {part['epoch']:,}",Path(v['sources']['checkpoint']['path']).name,v['sources']['checkpoint']['sha256'][:20]+'...'])
    table(fig,['用途 / epoch','权重文件','SHA-256（前20位）'],asset_rows,[.065,y-.088,.875,.082],[.23,.34,.43],8.0)
    y-=.120
    paragraph(fig,.065,y,'采用商品语义码与最终四 token SID 来自原 run 的 Video_Games.semantic.index.json / Video_Games.index.json；身份后缀只用于唯一识别。所有源文件完整路径和 SHA-256 见 comparison_sources.csv 与报告 manifest；完整对比数值见 comparison_checkpoint_metrics.csv。',8.0,width=117)
    report.csv('comparison_checkpoint_metrics.csv',source_data['metrics'])
    report.csv('comparison_sources.csv',comparison['sources'])
    snapshot={'selected_epoch':a['epoch'],'candidate_epoch':b['epoch'],'selection_scope':'User-selected conservative project baseline; no downstream recommendation evaluation',
              'fairness_checks':comparison['fairness_checks'],'sources':comparison['sources'],'baseline':source_data['baseline'],'endpoint':source_data['endpoint']}
    (report.output_dir/'comparison_selection.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if y<.145:
        raise ValueError(f'Integrated comparison source page overflow: {y}')
    report.save(fig)


def sid_report(analysis_path, output_dir, comparison_endpoint_run_dir=None, comparison_json=None):
    data = read_json(analysis_path)
    if data['validation'] != 'passed' or not data['identity_suffix_excluded_from_semantic_metrics']:
        raise ValueError('SID validation must pass, with identity suffix excluded from semantic metrics')
    n, collision, reconstruction, knn = data['items'], data['collisions'], data['reconstruction'], data['knn']
    quality, warnings = data['quality_assessment'], data['quality_assessment']['warnings']
    warning_pages = warning_groups(warnings)
    comparison = load_integrated_comparison(data, comparison_endpoint_run_dir, comparison_json) if comparison_endpoint_run_dir else None
    export_path=Path(data['run_directory'])/'export_stats.json'
    selected_epoch=comparison['selected_epoch'] if comparison else (read_json(export_path)['parameters']['checkpoint_epoch'] if export_path.exists() else None)
    pages = 9 + len(warning_pages) + (4 if comparison else 0)
    report = Report(output_dir, 'video-games-sid-validation-report', 'Video Games SID 验证报告', analysis_path, pages)
    quality_name = '定义内诊断支持' if quality['quality_status'] == 'supported_on_defined_diagnostics' else '语义质量需复核'
    stamp = datetime.fromisoformat(data['created_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    first_title=f'{selected_epoch:,} 轮 SID 验证与选型' if comparison else 'SID 唯一性与语义质量验证'
    fig = header(first_title, f"{n:,} 件商品  |  {data['dimensions']:,} 维向量  |  {stamp} 北京时间原验证记录", 1, pages, 'SID Validation')
    metric_cards(fig, [('最终完整 SID', scalar(n) if collision['final_four_token_unique'] else '存在重复', '四个 token；含身份后缀'),
                       ('原始三层碰撞率', scalar(collision['raw']['collision_rate'], 2, percent=True), '最近码字，不含后缀'),
                       ('诊断结论', quality_name, f"结构检查通过；{len(warnings)} 条质量告警")])
    label(fig, .065, .727, '01  编码阶段的唯一性', 13)
    rows = []
    for stage, name in [('raw', '原始最近码字 / 3 语义 token'), ('semantic', 'Sinkhorn 调整 / 3 语义 token')]:
        c = collision[stage]
        rows.append([name, scalar(c['unique_semantic_sids']), scalar(c['excess_items']), scalar(c['items_in_collision_groups']), scalar(c['collision_rate'], 2, percent=True)])
    rows.append(['最终完整 SID / 3 语义 + 1 身份 token', scalar(n) if collision['final_four_token_unique'] else '不唯一', '0' if collision['final_four_token_unique'] else '见源数据', '0' if collision['final_four_token_unique'] else '见源数据', '0.00%' if collision['final_four_token_unique'] else '见源数据'])
    table(fig, ['阶段', '唯一编码数', '多余商品数', '碰撞组内商品', '碰撞率'], rows,
          [.065, .535, .875, .154], [.40,.15,.15,.16,.14], 8.2)
    paragraph(fig, .065, .505, '碰撞率 = (商品数 - 唯一编码数) / 商品数。身份后缀保证完整 SID 可区分商品，但前三层语义码仍可能相同；后缀不参与本报告的语义相似度、重构或邻域指标。', 8.7, width=112)
    label(fig, .065, .426, '02  碰撞调整的重构代价', 13)
    repair = collision['repair_distortion']
    delta = repair['mse_delta']
    rows = [['编码改变的商品数', scalar(repair['changed_items'])], ['被改变商品的原始 MSE', scalar(repair['raw_mse'], 6)],
            ['被改变商品的 SK MSE', scalar(repair['semantic_mse'], 6)], ['被改变商品 MSE 变化均值', scalar(delta['mean'], 6)],
            ['MSE 变化第 5 / 50 / 95 百分位', ' / '.join(scalar(delta[k], 6) for k in ('p05','median','p95'))]]
    information_table(fig, rows, [.065, .204, .875, .187], font_size=8.5)
    paragraph(fig, .065, .170, f"质量状态：{quality_name}。结构检查仅确认产物有效；语义质量另由重构、邻域与前缀诊断评估，不代表推荐准确率。详细判据见报告末尾。", 8.7, width=111)
    first_note=f"当前采用 epoch {selected_epoch:,}；与 10,000 轮对比见第 {9+len(warning_pages)}-{12+len(warning_pages)} 页；尚无用户推荐评测。" if comparison else '本报告验证量化与商品语义结构，没有运行用户推荐 SFT 或 RL 效果评测。'
    label(fig, .065, .091, first_note, 8.4, MUTED)
    report.save(fig)

    fig = header('重构质量与训练均值基线', '初始化、原始码与碰撞调整后的比较  |  验证商品未参与梯度训练', 2, pages, 'SID Validation')
    val = {stage: reconstruction[stage]['val'] for stage in ('initial','raw','semantic')}
    metric_cards(fig, [('初始化后验证 NMSE', scalar(val['initial']['nmse_vs_train_mean'], 4), '梯度训练前的模型'),
                       ('原始码验证 NMSE', scalar(val['raw']['nmse_vs_train_mean'], 4), '已训练，最近码字'),
                       ('调整码验证 NMSE', scalar(val['semantic']['nmse_vs_train_mean'], 4), '同一模型，SK 调整编码')])
    label(fig, .065, .728, '03  验证集的相对重构误差', 13)
    ax = fig.add_axes([.12, .489, .79, .193])
    stages = [('initial', '初始化后'), ('raw', '原始三层码'), ('semantic', 'SK 调整码')]
    values = [val[stage]['nmse_vs_train_mean'] for stage, _ in stages]
    bars = ax.bar(range(3), values, color=['#B6C8D0', TEAL, BLUE], width=.50)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(1, color=GOLD, ls='--', label='训练均值基线 NMSE=1')
    ax.set_xticks(range(3), [name for _, name in stages]); ax.set_ylabel('验证 NMSE', fontsize=9)
    ax.legend(frameon=False, fontsize=8); style(ax)
    label(fig, .065, .448, 'NMSE = 模型 MSE / 训练均值基线 MSE；小于 1 表示优于该常量预测器。', 8.5, MUTED)
    label(fig, .065, .408, '04  各数据范围的完整重构统计', 13)
    rows = []
    for stage, name in stages:
        for split, split_name in [('train','训练'),('val','验证'),('all','全部')]:
            row = reconstruction[stage][split]
            rows.append([name, split_name, scalar(row['items']), scalar(row['mse'],6), scalar(row['nmse_vs_train_mean'],4), scalar(row['cosine']['mean'],4), scalar(row['zero_reconstruction_rows'])])
    table(fig, ['编码 / 权重', '范围', '商品数', '重构 MSE', 'NMSE', '重构 cos 均值', '零重构'], rows,
          [.065, .130, .875, .241], [.19,.09,.12,.19,.13,.19,.09], 8.3)
    label(fig, .065, .093, 'MSE 按原始 2560 维向量逐元素计算；原向量及训练均值基线均未额外归一化。', 8.3, MUTED)
    report.save(fig)

    fig = header('量化前后的邻域保持', '重构参照原始 2560 维；潜空间参照编码器 32 维；两者不直接比较优劣', 3, pages, 'SID Validation')
    label(fig, .065, .838, '05  各自量化前 TopK 邻居集合的保留份额', 13)
    stages = [('initial_reconstruction', '初始化后重构', '#A7BAC3'), ('raw_reconstruction', '原始码重构', TEAL),
              ('semantic_reconstruction', 'SK 码重构', BLUE), ('raw_quantized_latent', '原始量化潜空间', GOLD),
              ('semantic_quantized_latent', 'SK 量化潜空间', RED)]
    ks = [5,10,20]
    panels = [([.12,.551,.34,.240], '重构：参照原始 2560 维向量', stages[:3]),
              ([.60,.551,.34,.240], '潜空间：参照编码器 32 维表示', stages[3:])]
    for area, title, selected_stages in panels:
        ax = fig.add_axes(area)
        for stage, name, color in selected_stages:
            row = knn['metrics'][stage]['all_queries']
            ax.plot(ks, [row[f'overlap_at_{k}'] for k in ks], marker='o', ms=4, color=color, label=name, lw=1.5, ls='--' if stage=='initial_reconstruction' else '-')
        ax.plot(ks, [knn['random_overlap_baseline'][str(k)] for k in ks], ls=':', color=INK, label='随机期望')
        ax.set_xticks(ks, [f'Top{k}' for k in ks]); ax.set_ylim(0,1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title,fontsize=8.9,pad=10)
        ax.set_ylabel('各自参照邻域的交集数量 / K', fontsize=8.4)
        ax.legend(frameon=False, fontsize=7.1, loc='upper left'); style(ax)
    all_queries = knn['metrics']['raw_reconstruction']['all_queries']['queries']
    val_queries = knn['metrics']['raw_reconstruction']['val_queries']['queries']
    paragraph(fig,.065,.511,f"两图合并 {all_queries:,} 个查询、候选 {knn['candidate_items']:,} 件；下表为 {val_queries:,} 个验证查询。参照空间不同，不能直接比较左右数值高低。",8.2,width=113)
    rows = [[name, '原始 2560 维' if 'reconstruction' in stage else '编码器 32 维'] + [scalar(knn['metrics'][stage]['val_queries'][f'overlap_at_{k}'],2,percent=True) for k in ks] for stage,name,_ in stages]
    table(fig, ['验证查询', '参照邻域', 'Top5 保留', 'Top10 保留', 'Top20 保留'], rows,
          [.065, .327, .875, .150], [.30,.22,.16,.16,.16], 8.3)
    raw_top10=knn['metrics']['raw_reconstruction']['val_queries']['overlap_at_10']
    sk_top10=knn['metrics']['semantic_reconstruction']['val_queries']['overlap_at_10']
    paragraph(fig,.065,.308,f'SK 调整后，验证 Top10 邻域保持率从 {raw_top10:.2%} 变为 {sk_top10:.2%}（{(sk_top10-raw_top10)*100:+.2f} 个百分点）；碰撞减少伴随原空间邻域损失。',8.2,width=113)
    label(fig, .065, .257, '06  四类硬件子集内的同标签份额', 13)
    weak = data['weak_label_retrieval']
    rows = [[name] + [scalar(weak['metrics'][stage]['macro'][f'same_label_at_{k}'],2,percent=True) for k in [1,5,10]]
            for stage,name in [('original','原始商品向量'),('raw_reconstruction','原始码重构'),('semantic_reconstruction','SK 码重构')]]
    table(fig, ['四类等权宏平均', 'Top1', 'Top5', 'Top10'], rows,
          [.065, .112, .875, .107], [.40,.20,.20,.20], 8.8)
    paragraph(fig, .065, .093, f"弱标签候选仅 {weak['labeled_items']:,} 件（{weak['coverage']:.2%}），按四类等权平均；同标题及近重复商品可能提高份额。", 8.1, width=113)
    report.save(fig)

    fig = header('共享语义前缀的商品是否相似？', '按最长公共前缀 LCP 分组  |  商品对排除相同清洗文本', 4, pages, 'SID Validation')
    label(fig, .065, .838, '07  不同 LCP 长度的原向量余弦相似度', 13)
    prefixes = data['prefix_similarity']
    for stage, name, area in [('raw','原始最近码字',[.12,.548,.34,.234]), ('semantic','SK 调整码',[.60,.548,.34,.234])]:
        ax = fig.add_axes(area)
        values = [row['cosine_samples'] for row in prefixes[stage]]
        for position, (row, sample) in enumerate(zip(prefixes[stage], values), start=1):
            if sample:
                bp = ax.boxplot([sample], positions=[position], widths=.50, patch_artist=True, showfliers=False,
                    medianprops={'color':INK}, boxprops={'facecolor':[TEAL,BLUE,GOLD,RED][position-1], 'alpha':.65})
            else:
                ax.text(position,.04,'无样本',fontsize=7,ha='center',transform=ax.get_xaxis_transform())
        ax.set_xticks(range(1,5),[str(row['lcp']) for row in prefixes[stage]])
        ax.set_xlim(.4,4.6); ax.set_ylim(0,1.02); ax.set_title(name,fontsize=10,pad=10)
        ax.set_xlabel('最长公共前缀层数 LCP',fontsize=8.6)
        if stage=='raw':ax.set_ylabel('原始商品向量 cos',fontsize=9)
        style(ax)
    label(fig,.065,.507,'箱体为 25-75 百分位、须为 1.5 IQR；为清晰起见未画离群散点。',8.3,MUTED)
    rows=[]
    for index in range(4):
        raw,sem=prefixes['raw'][index],prefixes['semantic'][index]
        rows.append([raw['lcp'],scalar(raw['sampled_pairs']),scalar(raw['original_cosine']['mean'],4),scalar(sem['sampled_pairs']),scalar(sem['original_cosine']['mean'],4)])
    table(fig,['LCP','原始码抽样对数','原始码 cos 均值','SK 码抽样对数','SK 码 cos 均值'],rows,[.065,.301,.875,.162],[.10,.23,.22,.23,.22],8.7)
    label(fig,.065,.262,'08  如何解读前缀分组',13)
    paragraph(fig,.065,.229,'LCP 是从第一层开始连续相同的码字层数，范围 0-3；码字数字只是类别标识，没有“编号差越小越相近”的含义。LCP=3 代表完整三层语义码相同；身份后缀被排除。',9.0,width=111)
    paragraph(fig,.065,.156,'两组编码分别抽样，不能当作同一组配对数据。若 LCP=3 没有不同清洗文本的商品对，则显示无样本；样本不足时不补造数据。下一页用保持码本占用的置换对照检验近邻共享前缀的份额。',8.8,width=112)
    report.save(fig)

    fig = header('前缀邻域共享率与码本占用', '置换对照保持每个码字的占用量  |  参照原向量余弦 Top10 邻域', 5, pages, 'SID Validation')
    retention = data['prefix_neighbor_retention']
    label(fig, .065, .838, '09  验证查询的邻居共享前缀份额', 13)
    ax = fig.add_axes([.12,.569,.79,.221])
    selected_prefix = {}
    for stage, name, color, offset in [('raw','原始码',TEAL,-.10),('semantic','SK 码',BLUE,.10)]:
        records = [row for row in retention['stages'][stage]['val_queries'] if row['k']==10]
        selected_prefix[stage] = {row['prefix_depth']:row for row in records}
        for index, row in enumerate(records):
            x = row['prefix_depth'] + offset
            ax.scatter(x,row['observed'],color=color,s=30,zorder=4,label=name+'：观察值' if index==0 else None)
            ax.vlines(x,row['null_p025'],row['null_p975'],color=color,alpha=.45,lw=4,zorder=2)
            ax.scatter(x,row['null_mean'],color=color,marker='_',s=110,zorder=3,label=name+'：置换区间/均值' if index==0 else None)
            if not row['applicable']:
                ax.annotate('不可判别',(x,row['observed']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=6.8,color=MUTED)
    ax.set_xticks([1,2,3],['前 1 层','前 2 层','前 3 层']); ax.set_ylim(-.015,1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylabel('原向量 Top10 邻居共享前缀的份额',fontsize=8.7)
    ax.legend(frameon=False,ncol=2,fontsize=7.7,loc='upper right'); style(ax)
    label(fig,.065,.529,f"区间为 {retention['permutations']} 次置换的经验 2.5-97.5 百分位，不是观察值的置信区间。",8.4,MUTED)
    rows=[]
    for depth in (1,2,3):
        raw,sem=selected_prefix['raw'][depth],selected_prefix['semantic'][depth]
        rows.append([depth,scalar(raw['observed'],2,percent=True),scalar(raw['null_p975'],2,percent=True),
                     '是' if raw['above_null_p975'] else ('否' if raw['applicable'] else '不可判别'),
                     scalar(sem['observed'],2,percent=True),scalar(sem['null_p975'],2,percent=True),
                     '是' if sem['above_null_p975'] else ('否' if sem['applicable'] else '不可判别')])
    table(fig,['前缀层数','原始观察','null 97.5%','超过 null?','SK 观察','null 97.5%','超过 null?'],rows,
          [.065,.372,.875,.119],[.13,.145,.145,.14,.145,.145,.15],8.1)
    label(fig,.065,.335,'10  每个码字被多少商品使用？',13)
    counts=np.array([entry['counts'] for stage in ('raw','semantic') for entry in data['codebook_usage'][stage]['all']],dtype=float)
    ax=fig.add_axes([.13,.154,.74,.139]); image=ax.imshow(np.ma.masked_where(counts==0,np.log1p(counts)),aspect='auto',interpolation='nearest',cmap='YlGnBu')
    ax.set_yticks(range(6),['原始 1层','原始 2层','原始 3层','SK 1层','SK 2层','SK 3层'],fontsize=8)
    ax.set_xlabel('码字编号（0-255，仅标识，无连续数值距离）',fontsize=8.4)
    ax.set_xticks([0,64,128,192,255]); ax.tick_params(length=0)
    cax=fig.add_axes([.89,.154,.018,.139]); fig.colorbar(image,cax=cax).set_label('log(1+商品数)',fontsize=7.8)
    sk_usage=data['codebook_usage']['semantic']['all']
    label(fig,.065,.099,f"SK 第1层使用 {sk_usage[0]['used']}/{sk_usage[0]['codewords']} 码字、{sk_usage[0]['dead']} 个未使用；后两层使用 {sk_usage[1]['used']}/{sk_usage[1]['codewords']}、{sk_usage[2]['used']}/{sk_usage[2]['codewords']}。白色为未使用。",8.1,MUTED)
    report.save(fig)

    fig=header('输入扰动稳定性与编码示例','稳定性测试使用原始最近码字三层语义码；排除身份后缀',6,pages,'SID Validation')
    noise=data['noise_stability']; levels=noise['levels']
    label(fig,.065,.838,'11  轻微输入变化后，语义前缀是否保持？',13)
    ax=fig.add_axes([.12,.586,.79,.206])
    for depth,color in zip((1,2,3),(TEAL,BLUE,GOLD)):
        ax.plot([row['relative_l2_noise']*100 for row in levels],[row[f'prefix_{depth}_retention'] for row in levels],marker='o',color=color,label=f'前 {depth} 层保持')
    ax.set_xticks([row['relative_l2_noise']*100 for row in levels],[f"{row['relative_l2_noise']:.1%}" for row in levels]); ax.set_ylim(0,1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlabel('噪声 L2 / 原向量 L2',fontsize=9); ax.set_ylabel('验证商品前缀保持份额',fontsize=9)
    ax.legend(frameon=False,ncol=3,fontsize=8.2); style(ax)
    paragraph(fig,.065,.544,f"使用 {levels[0]['items']:,} 件验证商品；独立高斯方向被缩放到指定相对 L2 幅度。每个幅度仅 {noise['repetitions_per_level']} 次随机实现，seed={noise['seed']}；没有估计置信区间。SK 调整码依赖目录级碰撞处理，此处没有将其作为扰动稳定性基线。",8.5,width=112)
    label(fig,.065,.438,'12  预定义锚点的三层语义编码',13)
    rows=[]
    for case in data['examples']:
        rows.append([case['anchor_id'],'\n'.join(textwrap.wrap(clean(case['title']),width=62,break_long_words=True,break_on_hyphens=False)),
                     str(case['raw_sid']),str(case['semantic_sid'])])
    table(fig,['商品 ID','完整商品标题','原始三层码','SK 调整三层码'],rows,[.065,.170,.875,.235],[.09,.49,.21,.21],7.7)
    paragraph(fig,.065,.132,'表中只展示前三层语义码，最终完整 SID 另加 <d_...> 身份后缀。同前缀只是类别化的编码一致；码字编号的大小差不构成商品距离。',8.3,width=113)
    report.save(fig)

    fig=header('原始与重构近邻对照','元数据预定义锚点；逐列完整 Top5，保留同标题和近重复版本',7,pages,'SID Validation')
    by_id={case['anchor_id']:case for case in data['examples']}
    chosen=[by_id[3],by_id[7]]
    nearest_comparison_case(fig,chosen[0],.837,.478)
    nearest_comparison_case(fig,chosen[1],.453,.095)
    label(fig,.065,.078,'各列按各自空间的 cos 排名；单元格 cos 是近邻与锚点在原始商品向量空间的相似度。',7.5,MUTED)
    report.save(fig)

    for offset, chunk in enumerate(warning_pages, start=8):
        fig=header('质量诊断告警与判定边界',f'结构检查通过  |  语义质量状态：{quality_name}  |  告警记录 {offset-7} / {len(warning_pages)}',offset,pages,'SID Validation')
        label(fig,.065,.838,'13  未满足预定义判据的项目',13)
        y=.796
        for warning in chunk:
            y=paragraph(fig,.065,y,'• '+warning,8.8,width=109,leading=1.48,color=INK)-.012
        if y<.105:raise ValueError('Quality warnings overflow')
        label(fig,.065,.095,'告警依据是预定义诊断阈值与置换参考，不等于统计显著性检验或推荐效果结论。',8.1,MUTED)
        report.save(fig)

    fig=header('方法、质量边界与复现文件','完整 JSON 和 CSV 保留所有查询、分组与阶段统计',len(report.images)+1,pages,'SID Validation')
    y=.838
    if not warning_pages:
        label(fig,.065,y,'13  质量诊断',12); y-=.029
        if warnings:
            for warning in warnings:
                y=paragraph(fig,.065,y,'• '+warning,8.7,width=111)-.009
        else:
            y=paragraph(fig,.065,y,'没有触发预定义质量告警。该结论只适用于已计算的诊断；量化仍有损，碰撞调整后的邻域保持率也可能下降，不表示每项指标都改善。',8.7,width=111)-.012
    sections=[('14  训练与留出边界',f"{data['split']['train']:,} 件训练商品、{data['split']['val']:,} 件验证商品；相同清洗后标题和描述归为同组，两集相同文本重叠为 {data['split']['exact_cleaned_text_overlap']}。初始化只用训练商品，验证集不参与梯度训练。完整 SID 的去重后缀只用于身份区分。"),
              ('15  邻域与前缀口径','全库余弦近邻排除自身。重构邻域参照原始 2560 维向量；量化潜空间邻域参照编码器 32 维连续表示。各自计算 TopK 交集数量 / K，两种参照不能直接比较优劣。前缀置换保持码字占用，复用固定原向量近邻；经验区间不是观察值的置信区间。'),
              ('16  解释限制','相同标题和近重复商品可能提高近邻指标。类别弱标签只覆盖四类商品子集，包含类别噪声。三层前缀全唯一时，共享率恒为零、没有鉴别力；不据此宣称语义好或坏。轻微噪声可能跨越量化边界；单次噪声测试不提供总体置信区间。'),
              ('17  结构通过不等于下游效果','结构检查覆盖输入形状/数值、数据划分、ID 与编码对齐、码字范围、从 checkpoint 复现原始编码以及最终四 token 唯一性。重构、近邻和前缀诊断没有使用用户推荐准确率，不替代 SFT/RL 或人工相关性评测。')]
    for title,body in sections:
        label(fig,.065,y,title,11.6); y=paragraph(fig,.065,y-.028,body,8.7,width=111,leading=1.48)-.018
    label(fig,.065,y,'18  来源 SHA-256',11.6); y-=.029
    for key,name in [('checkpoint','选中 RQ-VAE 权重'),('embeddings','商品向量'),('final_sid','最终完整 SID')]:
        label(fig,.065,y,name,8.4,MUTED); label(fig,.065,y-.019,data['sources'][key]['sha256'],7.7); y-=.046
    if y<.075:raise ValueError(f'Methods page overflow at {y}')
    label(fig,.065,y,'全部统计与原始表格：sid_validation.json 和同目录 CSV；来源路径见报告 manifest。',8,MUTED)
    report.save(fig)

    if comparison:
        integrated_comparison_pages(report,comparison,pages)

    fig=header('诊断设计参考文献','参考初始化、前缀语义和多维诊断；不是对所有论文算法的完整复现',pages,pages,'SID Validation')
    references=[
        ('TIGER - NeurIPS 2023','https://arxiv.org/html/2305.05065v3#S3.SS1','KMeans 初始化、语义前缀及追加区分 token 的思路。本文检查三层语义码与身份后缀的不同作用。'),
        ('MiniOneRec','https://arxiv.org/html/2510.24431v1#S3.SS2','RQ-VAE 构建商品语义 ID 的项目流程。本文沿用原结构，并记录训练、碰撞和语义质量诊断。'),
        ('Purely Semantic Indexing','https://arxiv.org/html/2509.16446v1','帮助理解身份后缀的非语义属性及其取舍；本文没有完整复现该文算法。'),
        ('DRQ diagnostics','https://arxiv.org/html/2606.01844v1','借鉴几何结构、码本占用和扰动鲁棒性的多维诊断视角；本文不是该算法或通用标准的完整复现。')]
    for y,(title,url,context) in zip([.838,.662,.486,.310],references):
        label(fig,.065,y,title,13,TEAL)
        label(fig,.065,y-.034,url,8.2,BLUE,url=url)
        paragraph(fig,.065,y-.066,context,9.1,width=110)
        line(fig,y-.145)
    paragraph(fig,.065,.136,'这些参考资料支持诊断设计的背景解释。本文报告的具体阈值、抽样和置换次数由本次脚本定义；不据此宣称符合某个统一质量标准，也不以商品内容诊断替代用户推荐评测。',8.7,width=111)
    report.save(fig)

    validation_dir=Path(data['run_directory'])/'validation'
    for csv_path in sorted(validation_dir.glob('*.csv')):
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream); rows=list(reader); fields=reader.fieldnames
        report.csv(csv_path.name,rows,fields)
    (report.output_dir/'sid_validation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report.finish({'stage':'sid_validation','structural_validation':data['validation'],
                          'quality_status':quality['quality_status'],'quality_warning_count':len(warnings),
                          'selected_epoch':selected_epoch,
                          'comparison':None if not comparison else {'endpoint_run_directory':str(Path(comparison_endpoint_run_dir).resolve()),'endpoint_epoch':comparison['endpoint']['epoch'],
                                                                   'source':str(comparison['comparison_path']),'source_sha256':sha256(comparison['comparison_path']),
                                                                   'source_files':comparison['sources'],'fairness_checks':comparison['fairness_checks'],
                                                                   'page_start':9+len(warning_pages),'page_end':12+len(warning_pages)},
                          'items':n,'raw_collision_rate':collision['raw']['collision_rate'],
                          'semantic_collision_rate':collision['semantic']['collision_rate'],
                          'final_four_token_unique':collision['final_four_token_unique']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run-dir',type=Path,help='Completed RQ-VAE run directory')
    parser.add_argument('--sid-analysis',type=Path,help='Completed sid_validation.json')
    parser.add_argument('--comparison-endpoint-run-dir',type=Path,help='Optional completed 10000-epoch endpoint: append four comparison pages to SID report')
    parser.add_argument('--comparison-json',type=Path,help='Optional existing comparison.json; defaults to report stage 05')
    parser.add_argument('--training-output-dir',type=Path,default=REPORT_ROOT/'03_rqvae_training')
    parser.add_argument('--sid-output-dir',type=Path,default=REPORT_ROOT/'04_sid_validation')
    args=parser.parse_args()
    if not args.training_run_dir and not args.sid_analysis:
        parser.error('Provide --training-run-dir and/or --sid-analysis')
    if (args.comparison_endpoint_run_dir or args.comparison_json) and not args.sid_analysis:
        parser.error('Comparison options require --sid-analysis')
    manifests=[]
    if args.training_run_dir:manifests.append(training_report(args.training_run_dir,args.training_output_dir))
    if args.sid_analysis:manifests.append(sid_report(args.sid_analysis,args.sid_output_dir,args.comparison_endpoint_run_dir,args.comparison_json))
    print(json.dumps(manifests,ensure_ascii=False))


if __name__=='__main__':main()
