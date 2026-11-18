#!/usr/bin/env python3
"""Compare a selected RQ-VAE checkpoint with a true 10,000-epoch endpoint.

Reads JSON/CSV diagnostics only. No model loading, vector inference, or GPU use.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import math
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from build_embedding_similarity_report import label, paragraph, table, style, TEAL, BLUE, GOLD, RED, MUTED, INK, GRID
from build_rqvae_reports import ROOT, REPORT_ROOT, Report, header, metric_cards, scalar, read_json, sha256, write_csv

STEM = 'video-games-rqvae-10000-comparison-report'
COLORS = [TEAL, BLUE]
NAMES = ['原最佳权重', '10,000 轮终点']
PRIMARY = ['raw_val_mse', 'semantic_val_top10_original']


def artifact(run, manifest, key, filename):
    local = run / filename
    if local.exists():
        return local
    recorded = manifest.get('artifacts', {}).get(key)
    if recorded and Path(recorded).exists():
        return Path(recorded).resolve(strict=True)
    # An endpoint export may live below its completed continuation training run.
    parent = run.parent / filename
    if parent.exists():
        return parent.resolve(strict=True)
    raise FileNotFoundError(f'Cannot locate {filename} for {run}')


def load_run(path):
    run = Path(path).resolve(strict=True)
    manifest = read_json(run / 'manifest.json')
    paths = {key: artifact(run, manifest, key, name) for key, name in [
        ('training_summary', 'training_summary.json'), ('evaluation_history', 'evaluation_history.json'),
        ('sid_validation', 'sid_validation.json'), ('export_stats', 'export_stats.json')]}
    data = {key: read_json(value) for key, value in paths.items()}
    data.update(run=run, manifest=manifest, paths=paths)
    if data['training_summary'].get('status') != 'completed':
        raise ValueError(f'Training not completed: {run}')
    if data['sid_validation'].get('validation') != 'passed':
        raise ValueError(f'Structural SID validation not passed: {run}')
    data['epoch'] = int(data['export_stats']['parameters']['checkpoint_epoch'])
    export_hash = data['export_stats']['sources']['checkpoint']['sha256']
    if export_hash != data['sid_validation']['sources']['checkpoint']['sha256']:
        raise ValueError(f'Export and validation checkpoint SHA-256 differ: {run}')
    return data


def fairness(a, b, expected):
    av, bv = a['sid_validation'], b['sid_validation']
    checks = {}
    for source in ['embeddings', 'item_metadata', 'split', 'initial_checkpoint']:
        checks[f'identical_{source}_sha256'] = av['sources'][source]['sha256'] == bv['sources'][source]['sha256']
    # Same-directory smoke tests are allowed; distinct exported epochs must have
    # distinct actual checkpoint bytes, not only different manifest labels.
    checks['different_checkpoint_sha256_for_different_epochs'] = (
        a['epoch'] == b['epoch'] or
        av['sources']['checkpoint']['sha256'] != bv['sources']['checkpoint']['sha256'])
    checks['identical_split_counts'] = av['split'] == bv['split']
    checks['identical_query_ids'] = av['knn']['query_ids'] == bv['knn']['query_ids']
    checks['identical_knn_candidate_scope'] = av['knn']['candidate_items'] == bv['knn']['candidate_items'] and av['knn']['exclude_self'] == bv['knn']['exclude_self']
    checks['identical_repair_configuration'] = all(a['export_stats'].get(key) == b['export_stats'].get(key) for key in ['algorithm', 'max_collision_rounds']) and a['export_stats']['parameters']['repair_sk_epsilons'] == b['export_stats']['parameters']['repair_sk_epsilons']
    checks['endpoint_epoch_matches_request'] = b['epoch'] == expected
    checks['training_reached_endpoint'] = b['training_summary']['epochs_completed'] >= expected
    checks['identity_suffix_excluded'] = av['identity_suffix_excluded_from_semantic_metrics'] and bv['identity_suffix_excluded_from_semantic_metrics']
    ac, bc = a['training_summary']['config'], b['training_summary']['config']
    model_keys = ['seed', 'lr', 'batch_size', 'learner', 'lr_scheduler_type', 'warmup_epochs', 'weight_decay', 'dropout_prob', 'bn', 'loss_type', 'num_emb_list', 'e_dim', 'quant_loss_weight', 'beta', 'layers']
    checks['identical_model_and_optimizer_configuration'] = all(ac.get(key) == bc.get(key) for key in model_keys)
    aeval = {x['epoch']: x for x in a['evaluation_history']}
    beval = {x['epoch']: x for x in b['evaluation_history']}
    checks['inherited_evaluation_history_matches'] = all(epoch in beval and math.isclose(row['val']['mse'], beval[epoch]['val']['mse'], rel_tol=1e-7, abs_tol=1e-9) for epoch, row in aeval.items())
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError('Comparison is not controlled: ' + ', '.join(failed))
    return checks


def metric_value(v, stage, split, metric):
    row = v['reconstruction'][stage][split]
    if metric == 'cosine':
        return row['cosine']['mean']
    if metric == 'nmse':
        return row['nmse_vs_train_mean']
    return row[metric]


def comparison_rows(a, b):
    rows = []
    def add(key, title, values, direction, kind='number', reference='', scope='val'):
        left, right = [float(value) for value in values]
        delta = right - left
        tolerance = max(abs(left), abs(right), 1) * 1e-10
        winner = '持平' if abs(delta) <= tolerance else ('10,000 轮' if (delta < 0) == (direction == 'lower') else '原最佳')
        if direction == 'diagnostic':
            winner = '诊断；不据此排序'
        rows.append(dict(metric=key, label=title, scope=scope, reference=reference, direction=direction,
                         baseline=left, endpoint=right, delta=delta,
                         relative_change=delta / left if left else None, kind=kind, preferred=winner))
    vals = [a['sid_validation'], b['sid_validation']]
    for stage, title in [('raw', '原始三层码'), ('semantic', 'SK 调整后')]:
        for metric, metric_title, direction, kind in [('mse', 'MSE', 'lower', 'number'), ('nmse', 'NMSE', 'lower', 'number'), ('cosine', '平均余弦', 'higher', 'number')]:
            add(f'{stage}_val_{metric}', f'{title}验证 {metric_title}', [metric_value(v, stage, 'val', metric) for v in vals], direction, kind, '原始 2560D 商品向量')
        for k in [5, 10, 20]:
            add(f'{stage}_val_top{k}_original', f'{title}验证 Top{k} 保留', [v['knn']['metrics'][f'{stage}_reconstruction']['val_queries'][f'overlap_at_{k}'] for v in vals], 'higher', 'percent', '同一原始 2560D 全库余弦近邻')
        add(f'{stage}_val_top10_latent', f'{title}验证潜空间 Top10 保留', [v['knn']['metrics'][f'{stage}_quantized_latent']['val_queries']['overlap_at_10'] for v in vals], 'diagnostic', 'percent', '各 checkpoint 编码器 32D 近邻；参照随模型改变')
        add(f'{stage}_collision_rate', f'{title}全库碰撞率', [v['collisions'][stage]['collision_rate'] for v in vals], 'lower', 'percent', '(商品数 - 不同三层码数) / 商品数', 'all')
        for layer in range(3):
            for metric, direction, kind in [('used', 'diagnostic', 'integer'), ('perplexity', 'diagnostic', 'number'), ('maximum_share', 'diagnostic', 'percent')]:
                add(f'{stage}_L{layer+1}_{metric}', f'{title}码本 L{layer+1} {metric}', [v['codebook_usage'][stage]['all'][layer][metric] for v in vals], direction, kind, '全库码字占用；不是准确率', 'all')
    add('raw_train_mse', '原始三层码训练 MSE', [metric_value(v, 'raw', 'train', 'mse') for v in vals], 'lower', reference='训练商品；仅看它会忽略泛化', scope='train')
    add('repair_changed_items', 'SK 修改商品数', [v['collisions']['repair_distortion']['changed_items'] for v in vals], 'diagnostic', 'integer', '被调整商品集合不同，不能作同组统计', 'all')
    add('repair_changed_mse_delta', '被修改商品平均 MSE 增量', [v['collisions']['repair_distortion']['mse_delta']['mean'] for v in vals], 'diagnostic', reference='各自被修改商品；不是全库均值', scope='changed_items')
    add('repair_all_mse_delta', '全库 SK MSE 增量', [metric_value(v, 'semantic', 'all', 'mse') - metric_value(v, 'raw', 'all', 'mse') for v in vals], 'lower', reference='同一完整商品库', scope='all')
    add('repair_val_top10_drop', '验证 Top10 因 SK 的下降', [v['knn']['metrics']['raw_reconstruction']['val_queries']['overlap_at_10'] - v['knn']['metrics']['semantic_reconstruction']['val_queries']['overlap_at_10'] for v in vals], 'lower', 'percent', '原始 2560D 商品向量近邻')
    for ai, level in enumerate(vals[0]['noise_stability']['levels']):
        other = vals[1]['noise_stability']['levels'][ai]
        if level['relative_l2_noise'] != other['relative_l2_noise']:
            raise ValueError('Noise levels differ')
        for depth in [1, 2, 3]:
            add(f"noise_{level['relative_l2_noise']}_prefix{depth}", f"{level['relative_l2_noise']:.1%} 噪声前 {depth} 层保持", [x[f'prefix_{depth}_retention'] for x in [level, other]], 'higher', 'percent', '原始三层码；每幅度一次噪声实现')
    return rows


def fmt(value, kind='number', delta=False):
    if kind == 'integer':
        return f'{value:+,.0f}' if delta else f'{value:,.0f}'
    if kind == 'percent':
        return f'{value*100:+.2f} pp' if delta else f'{value:.2%}'
    return f'{value:+.6f}' if delta else f'{value:.6f}'


def decision(rows):
    by = {x['metric']: x for x in rows}
    left, right = [by[key]['preferred'] for key in PRIMARY]
    if left == right and left in ['原最佳', '10,000 轮']:
        title = '10,000 轮在两项主指标上更好' if left == '10,000 轮' else '保留原最佳权重更合适'
        body = '两项主指标为验证集原始三层码重构 MSE（越低越好）和 SK 调整后原始向量 Top10 近邻保留率（越高越好）。本次方向一致，可据此选择该权重做后续候选。'
    else:
        title = '主指标存在取舍，不能只按轮数选'
        body = '验证原始重构 MSE 与最终语义码邻域质量没有一致占优。当前保留原权重，将 10,000 轮权重作为候选；尚无下游推荐结果。本报告不将异质指标合成总分。'
    return dict(headline=title, interpretation=body, primary_metrics=PRIMARY,
                primary_preferences={key: by[key]['preferred'] for key in PRIMARY},
                recommendation_scope='商品内容诊断；不是用户推荐准确率或 SFT/RL 效果')


def grouped_bars(ax, labels, arrays, *, percent=False, names=None, ylim=None, decimals=2):
    xs = np.arange(len(labels))
    width = .34
    for index, (values, color, name) in enumerate(zip(arrays, COLORS, names or NAMES)):
        bars = ax.bar(xs + (index-.5)*width, values, width, color=color, alpha=.88, label=name)
        for bar, value in zip(bars, values):
            display = f'{value:.{decimals}%}' if percent else f'{value:.{decimals}f}'
            ax.annotate(display, (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7)
    ax.set_xticks(xs, labels)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(max(values) for values in arrays)*1.23)
    style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)


def row_table(fig, rows, area, size=8.3):
    return table(fig, ['指标', '原最佳', '10,000 轮', '变化（终点 - 原）', '更优'],
                 [[row['label'].replace('原始三层码', 'Raw').replace('SK 调整后', 'SK'), fmt(row['baseline'], row['kind']), fmt(row['endpoint'], row['kind']), fmt(row['delta'], row['kind'], True), row['preferred']] for row in rows],
                 area, [.35, .15, .15, .20, .15], size)


def build_report(baseline_dir, endpoint_dir, output_dir, expected_endpoint_epoch=10000):
    a, b = load_run(baseline_dir), load_run(endpoint_dir)
    checks = fairness(a, b, expected_endpoint_epoch)
    rows = comparison_rows(a, b)
    by = {row['metric']: row for row in rows}
    outcome = decision(rows)
    vals = [a['sid_validation'], b['sid_validation']]
    exports = [a['export_stats'], b['export_stats']]
    summaries = [a['training_summary'], b['training_summary']]
    names = [f'原最佳 epoch {a["epoch"]:,}', f'终点 epoch {b["epoch"]:,}']
    global NAMES
    NAMES = names
    out = Path(output_dir)
    report = Report(out, STEM, 'Video Games RQ-VAE 10000 轮对比报告', b['paths']['sid_validation'], 6)
    report.csv('checkpoint_metric_comparison.csv', rows)
    source_rows = []
    for name, run in zip(['baseline', 'endpoint'], [a, b]):
        for kind, path in run['paths'].items():
            source_rows.append(dict(run=name, artifact=kind, path=str(path), sha256=sha256(path)))
    report.csv('comparison_sources.csv', source_rows)
    stamp = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')

    fig = header('训练到 10,000 轮，效果更好吗？', f'同一 Video Games 商品库、相同划分与查询  |  生成时间 {stamp}', 1, 6, 'RQ-VAE Comparison')
    metric_cards(fig, [('对比权重', f'{a["epoch"]:,} / {b["epoch"]:,}', '原最佳 / 真正训练终点'),
                       ('商品 / 验证商品', f'{vals[0]["items"]:,} / {vals[0]["split"]["val"]:,}', '10% 按清洗后的相同文本分组留出'),
                       ('续训 run 最佳轮次', f'{summaries[1]["best_epoch"]:,}', '最佳验证权重与 10,000 轮终点分开记录')])
    label(fig, .065, .724, outcome['headline'], 17, TEAL)
    paragraph(fig, .065, .684, outcome['interpretation'], 9, width=111)
    row_table(fig, [by[key] for key in ['raw_val_mse', 'semantic_val_top10_original', 'semantic_val_mse', 'raw_train_mse']], [.065, .435, .875, .175], 8.2)
    label(fig, .065, .392, '如何读这份比较', 13)
    y = .355
    bullets = [
        f"原实验共完成 {summaries[0]['epochs_completed']:,} 轮，但生产 SID 来自验证最优的 epoch {a['epoch']:,}；新实验显式导出 epoch {b['epoch']:,}，不是拿最佳权重冒充终点。",
        f"续训 run 的最低验证 MSE 是 {summaries[1]['best_val_mse']:.6f}，发生在 epoch {summaries[1]['best_epoch']:,}。它只作为训练过程信息，未替代本页终点值。",
        '最终四 token SID 全唯一表示商品身份可区分；它不能说明语义邻域更好。语义指标只使用前三层码，排除身份后缀。',
        '变化列统一为“10,000 轮终点 - 原最佳”；百分比指标变化使用百分点（pp）。码本利用与潜空间指标是辅助诊断，不做加权总分。']
    for text in bullets:
        y = paragraph(fig, .065, y, '- ' + text, 8.7, width=111) - .015
    report.save(fig)

    fig = header('训练曲线：更久是否仍在泛化？', '每 50 轮验证一次；选权重依据验证集，不以训练误差最低为准', 2, 6, 'RQ-VAE Comparison')
    hist = b['evaluation_history']
    xs = [x['epoch'] for x in hist]
    for area, limits, title in [([.12, .575, .79, .235], None, '完整训练轨迹'), ([.12, .245, .79, .235], (max(0, a['epoch']-700), b['epoch']), '原最佳之后的局部轨迹')]:
        ax = fig.add_axes(area)
        ax.plot(xs, [x['train']['mse'] for x in hist], color=TEAL, label='训练 Raw MSE', lw=1.5)
        ax.plot(xs, [x['val']['mse'] for x in hist], color=BLUE, label='验证 Raw MSE', lw=1.5)
        ax.axvline(a['epoch'], color=GOLD, ls='--', lw=1, label=f'原最佳 {a["epoch"]:,}')
        ax.axvline(b['epoch'], color=RED, ls=':', lw=1.3, label=f'终点 {b["epoch"]:,}')
        if limits:
            ax.set_xlim(*limits)
            visible = [x for x in hist if x['epoch'] >= limits[0]]
            lo, hi = min(x['train']['mse'] for x in visible), max(x['val']['mse'] for x in visible)
            ax.set_ylim(lo*.95, hi*1.05)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('原始向量重构 MSE', fontsize=8)
        ax.set_title(title, fontsize=10, loc='left', pad=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        style(ax)
        ax.legend(frameon=False, fontsize=7.6, ncol=2)
    improvement = by['raw_val_mse']
    paragraph(fig, .065, .175, f"验证 MSE 从 {improvement['baseline']:.6f} 变为 {improvement['endpoint']:.6f}（相对变化 {improvement['relative_change']:+.2%}）。训练误差下降只说明拟合训练商品更充分；若验证误差上升，不能据此判断新权重更好。", 8.8, width=111)
    added_seconds = max(0, summaries[1]['training_seconds'] - summaries[0]['training_seconds'])
    paragraph(fig, .065, .108, f"新增续训耗时 {added_seconds:.1f} 秒；累计训练 {summaries[1]['training_seconds']:.1f} 秒（含原先训练至 4,100 轮的 {summaries[0]['training_seconds']:.1f} 秒），不含导出与验证。曲线为 Raw；SK 是训练后修复。", 8.1, width=116)
    report.save(fig)

    fig = header('留出商品：重构与邻域保持', '先看未参与梯度更新的商品；两个近邻参照空间分别解释', 3, 6, 'RQ-VAE Comparison')
    label(fig, .065, .837, '01  相同原始 2560D 全库余弦近邻：越高越好', 12)
    ax = fig.add_axes([.12, .588, .79, .193])
    grouped_bars(ax, ['Raw Top5', 'Raw Top10', 'SK Top5', 'SK Top10'], [[v['knn']['metrics'][stage]['val_queries'][f'overlap_at_{k}'] for stage, k in [('raw_reconstruction', 5), ('raw_reconstruction', 10), ('semantic_reconstruction', 5), ('semantic_reconstruction', 10)]] for v in vals], percent=True)
    ax.set_ylabel('TopK 交集 / K', fontsize=8)
    row_table(fig, [by[key] for key in ['raw_val_mse', 'raw_val_nmse', 'raw_val_cosine', 'semantic_val_mse', 'semantic_val_nmse', 'semantic_val_cosine']], [.065, .324, .875, .203], 7.9)
    label(fig, .065, .283, '02  各 checkpoint 的 32D 编码器近邻：辅助诊断', 12)
    ax = fig.add_axes([.12, .129, .43, .107])
    grouped_bars(ax, ['Raw Top10', 'SK Top10'], [[v['knn']['metrics'][stage]['val_queries']['overlap_at_10'] for stage in ['raw_quantized_latent', 'semantic_quantized_latent']] for v in vals], percent=True)
    ax.tick_params(labelsize=8)
    # Keep the legend above the right-hand explanation, clear of bar labels.
    ax.legend(loc='upper left', bbox_to_anchor=(1.10, 1.27), fontsize=7,
              ncol=1, borderaxespad=0, handlelength=1.2, labelspacing=.3,
              frameon=False)
    paragraph(fig, .605, .233, '这里的连续潜空间由各自编码器生成，参照邻域会随模型改变。数值可说明各模型内部的量化损失，不可与上图混为同一基准，也不作为跨模型选优的主指标。', 8.2, width=43)
    label(fig, .065, .078, f"验证查询 {vals[0]['knn']['metrics']['raw_reconstruction']['val_queries']['queries']:,}；候选 {vals[0]['knn']['candidate_items']:,}；排除自身。相同标题和近重复版本仍保留。", 7.6, MUTED)
    report.save(fig)

    fig = header('码本占用与碰撞修复代价', 'Raw = 最近码字；SK = 最后一层 Sinkhorn 调整；身份后缀只作区分', 4, 6, 'RQ-VAE Comparison')
    label(fig, .065, .837, '03  全库原始码本：有效码字数 / 256', 12)
    ax = fig.add_axes([.12, .620, .79, .16])
    grouped_bars(ax, ['L1', 'L2', 'L3'], [[row['used'] for row in v['codebook_usage']['raw']['all']] for v in vals], ylim=(0, 305), decimals=0)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.19), fontsize=7.5,
              ncol=2, borderaxespad=0, frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    used_rows = []
    for index, v in enumerate(vals):
        for stage in ['raw', 'semantic']:
            used_rows.append([names[index], stage.upper(), *[f"{x['perplexity']:.1f}" for x in v['codebook_usage'][stage]['all']]])
    table(fig, ['权重', '阶段', 'L1 困惑度', 'L2 困惑度', 'L3 困惑度'], used_rows, [.065, .446, .875, .125], [.32, .14, .18, .18, .18], 8.1)
    collision_rows = []
    for index, v in enumerate(vals):
        for stage in ['raw', 'semantic']:
            c = v['collisions'][stage]
            collision_rows.append([names[index], stage.upper(), f"{c['unique_semantic_sids']:,}", f"{c['collision_rate']:.2%}", f"{c['maximum_group_size']:,}"])
    table(fig, ['权重', '阶段', '不同三层码', '碰撞率', '最大同码组'], collision_rows, [.065, .264, .875, .125], [.32, .14, .20, .18, .16], 8.1)
    row_table(fig, [by[key] for key in ['repair_all_mse_delta', 'repair_val_top10_drop']], [.065, .151, .875, .09], 7.8)
    paragraph(fig, .065, .125, f"SK 修改商品：原 {int(by['repair_changed_items']['baseline']):,} / 终点 {int(by['repair_changed_items']['endpoint']):,}。被改集合平均 MSE 增量 {by['repair_changed_mse_delta']['baseline']:.4f} / {by['repair_changed_mse_delta']['endpoint']:.4f}；两组集合不同，不能作同一商品组的配对比较。两套四 token SID 均唯一。", 7.8, width=119)
    report.save(fig)

    fig = header('前缀语义与扰动稳定性', '前缀是离散类别一致性；码字编号的数值差没有距离含义', 5, 6, 'RQ-VAE Comparison')
    label(fig, .065, .837, '04  原向量 Top10 近邻共享前缀（验证集）', 12)
    prefix_rows = []
    for index, v in enumerate(vals):
        for stage in ['raw', 'semantic']:
            selected = [row for row in v['prefix_neighbor_retention']['stages'][stage]['val_queries'] if row['k'] == 10]
            for p in selected:
                prefix_rows.append(dict(checkpoint=names[index], stage=stage, **p))
    report.csv('prefix_top10_with_permutation_null.csv', prefix_rows)
    table(fig, ['权重 / 阶段', '前缀', '观察共享率', '置换均值', '置换 2.5%-97.5%'],
          [[row['checkpoint'].replace('epoch ', '')+' / '+row['stage'].upper(), str(row['prefix_depth']), f"{row['observed']:.3%}", f"{row['null_mean']:.3%}", f"{row['null_p025']:.3%} - {row['null_p975']:.3%}"] for row in prefix_rows],
          [.065, .468, .875, .312], [.36, .08, .17, .16, .23], 7.5)
    paragraph(fig, .065, .439, '置换保持每个码字占用，只打乱商品与码的对应；经验范围不是观察共享率的置信区间。第三层经去碰撞后趋于唯一，共享率可能接近零；它不单独证明语义好坏。', 8.1, width=116)
    label(fig, .065, .362, '05  轻微噪声后原始三层完整码保持率', 12)
    ax = fig.add_axes([.12, .160, .79, .150])
    for v, name, color in zip(vals, names, COLORS):
        levels = v['noise_stability']['levels']
        ax.plot([row['relative_l2_noise']*100 for row in levels], [row['prefix_3_retention'] for row in levels], color=color, marker='o', label=name)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([x['relative_l2_noise']*100 for x in vals[0]['noise_stability']['levels']], [f"{x['relative_l2_noise']:.1%}" for x in vals[0]['noise_stability']['levels']])
    ax.set_xlabel('噪声 L2 / 原商品向量 L2', fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style(ax)
    paragraph(fig, .065, .098, f"每个幅度仅 {vals[0]['noise_stability']['repetitions_per_level']} 次独立高斯方向实现、seed={vals[0]['noise_stability']['seed']}，覆盖全部验证商品。该测试只含 Raw 三层码；SK 为目录级修复，未作同口径噪声测试。完整前 1/2/3 层结果见 CSV。", 8, width=116)
    report.save(fig)

    fig = header('比较边界与复核依据', '只生成 RQ-VAE 与 SID 的诊断报告；没有执行后续 SFT 或 RL', 6, 6, 'RQ-VAE Comparison')
    label(fig, .065, .837, '06  受控比较检查', 12)
    y = paragraph(fig, .065, .800, '已核对商品向量、元数据、划分与初始化权重 SHA-256 相同，两个不同轮次的选用权重 SHA-256 不同；训练 / 验证规模、近邻查询 ID、候选范围、模型及优化器核心配置相同；原验证曲线完整继承，碰撞修复算法和参数相同。', 8.7, width=111)-.019
    for title, text in [
        ('训练轮次与选模规则', f"原 run 训练到 {summaries[0]['epochs_completed']:,} 轮后选择 epoch {a['epoch']:,}；续训到 epoch {b['epoch']:,}，本报告严格使用这个终点的导出与验证结果。续训过程 best-val 为 epoch {summaries[1]['best_epoch']:,}，验证 MSE {summaries[1]['best_val_mse']:.6f}。两者不能混淆。"),
        ('训练 / 验证 / 目录级处理', '码本初始化和梯度训练仅使用训练商品。最终近邻检索与碰撞修复使用整个已知商品目录，因此这是目录内商品泛化诊断，不是对完全未知新目录的部署保证。'),
        ('指标解释与局限', 'MSE、NMSE 和余弦描述向量重构；TopK 保留描述内容空间邻域。占用高或碰撞少不保证相关商品聚集。前缀、噪声和码本统计辅助解释取舍，报告不把它们压成单一总分。'),
        ('没有下游效果结论', '没有训练推荐模型，没有用户 Recall/NDCG，也没有 SFT/RL 评测。相似标题和近重复商品可能提高近邻分数。本次为相同随机种子下的同一训练轨迹，不是跨种子独立重复实验，不能估计跨种子方差。')]:
        label(fig, .065, y, title, 10.8, TEAL)
        y = paragraph(fig, .065, y-.026, text, 8.6, width=111)-.020
    label(fig, .065, y, '07  源文件与校验', 12)
    y -= .031
    for name, run in zip(names, [a, b]):
        label(fig, .065, y, name, 8.6, MUTED)
        label(fig, .065, y-.021, run['sid_validation']['sources']['checkpoint']['sha256'], 7.4)
        y -= .052
    paragraph(fig, .065, y, 'comparison.json 保留结论、检查、指标和完整来源校验；checkpoint_metric_comparison.csv 含所有可复核数值。训练曲线、前缀置换与码本占用分别导出 CSV。完整诊断位于各 run 的 sid_validation.json。', 8.2, width=115)
    report.save(fig)

    curve_rows = []
    for run_name, run in zip(['baseline', 'endpoint'], [a, b]):
        for row in run['evaluation_history']:
            curve_rows.append(dict(run=run_name, epoch=row['epoch'], train_mse=row['train']['mse'], val_mse=row['val']['mse'], val_nmse=row['val']['nmse'], val_cosine=row['val']['cosine_mean']))
    report.csv('training_curve_comparison.csv', curve_rows)
    usage_rows = []
    for run_name, v in zip(['baseline', 'endpoint'], vals):
        for stage in ['raw', 'semantic']:
            for split in ['train', 'val', 'all']:
                for layer in v['codebook_usage'][stage][split]:
                    for codeword, count in enumerate(layer['counts']):
                        usage_rows.append(dict(checkpoint=run_name, stage=stage, split=split, layer=layer['layer'], codeword=codeword, count=count, used=layer['used'], perplexity=layer['perplexity'], maximum_share=layer['maximum_share']))
    report.csv('codebook_usage_comparison.csv', usage_rows)
    data = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), fairness_checks=checks,
                baseline=dict(run_directory=str(a['run']), checkpoint_epoch=a['epoch'], training_completed_epoch=summaries[0]['epochs_completed'], checkpoint_sha256=vals[0]['sources']['checkpoint']['sha256']),
                endpoint=dict(run_directory=str(b['run']), checkpoint_epoch=b['epoch'], training_completed_epoch=summaries[1]['epochs_completed'], best_validation_epoch=summaries[1]['best_epoch'], best_validation_mse=summaries[1]['best_val_mse'], checkpoint_sha256=vals[1]['sources']['checkpoint']['sha256'], cumulative_training_seconds=summaries[1]['training_seconds'], additional_training_seconds=added_seconds),
                decision=outcome, metrics=rows, source_files=source_rows,
                important_interpretations=['No aggregate score.', 'Original-space and checkpoint-specific latent-space kNN have different references.', 'Permutation range is not confidence interval.', 'Final identity suffix is excluded from semantic diagnostics.', 'No downstream recommendation claim.'])
    (out / 'comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report.finish(dict(stage='rqvae_10000_comparison', baseline_run=str(a['run']), endpoint_run=str(b['run']), baseline_epoch=a['epoch'], endpoint_epoch=b['epoch'], decision=outcome, fairness_checks=checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run-dir', required=True, type=Path)
    parser.add_argument('--endpoint-run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=REPORT_ROOT/'05_rqvae_10000_comparison')
    parser.add_argument('--expected-endpoint-epoch', type=int, default=10000, help='Must match exported endpoint; override only for report smoke tests')
    args = parser.parse_args()
    print(json.dumps(build_report(args.baseline_run_dir, args.endpoint_run_dir, args.output_dir, args.expected_endpoint_epoch), ensure_ascii=False))


if __name__ == '__main__':
    main()
