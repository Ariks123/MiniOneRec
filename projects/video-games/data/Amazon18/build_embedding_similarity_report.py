#!/usr/bin/env python3
"""Build a source-driven six-page PDF report of Video Games embedding similarity."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import html
import json
import textwrap
import unicodedata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/reports/Video_Games/02_embedding_similarity'
FONT = Path('C:/Windows/Fonts/simhei.ttf')
if FONT.exists():
    fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 9.5, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                     'text.color': '#18323C', 'axes.labelcolor': '#536C77',
                     'xtick.color': '#536C77', 'ytick.color': '#536C77'})
INK, MUTED, TEAL, BLUE, GOLD, RED, GRID = '#18323C', '#637C86', '#16808E', '#537FAE', '#C68D36', '#B96B5B', '#E1EAEE'
COLORS = [BLUE, TEAL, GOLD, RED]


def clean(value):
    return html.unescape(str(value)).replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').replace('\u00ae', '(R)').replace('\u2122', '(TM)')


def label(fig, x, y, value, size=9.5, color=INK, ha='left', **kwargs):
    return fig.text(x, y, clean(value), fontsize=size, color=color, va='top', ha=ha, **kwargs)


def line(fig, y, x0=.065, x1=.94):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=GRID, lw=.8))


def header(title, subtitle, page):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.add_artist(Rectangle((.065, .947), .05, .006, transform=fig.transFigure, color=TEAL, lw=0))
    label(fig, .065, .925, title, 22)
    label(fig, .065, .883, subtitle, 8.8, MUTED)
    line(fig, .055)
    label(fig, .065, .041, 'MiniOneRec  |  Video Games  |  Qwen3-Embedding-4B', 7.8, MUTED)
    label(fig, .94, .041, f'{page} / 6', 7.8, MUTED, ha='right')
    return fig


def style(ax, grid_axis='y'):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, lw=.7)


def boxplot(ax, arrays, labels, colors):
    bp = ax.boxplot(arrays, labels=labels, widths=.50, patch_artist=True, showmeans=True,
                    medianprops={'color': INK, 'linewidth': 1.3},
                    meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': INK, 'markersize': 4},
                    whiskerprops={'color': MUTED}, capprops={'color': MUTED},
                    flierprops={'marker': '.', 'markersize': 2.2, 'alpha': .35, 'markeredgecolor': MUTED})
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=.72, edgecolor=color)
    style(ax)
    return bp


def table(fig, columns, rows, area, widths=None, fontsize=9, aligns=None):
    ax = fig.add_axes(area)
    ax.axis('off')
    tb = ax.table(cellText=rows, colLabels=columns, colWidths=widths, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(fontsize)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(.6)
        cell.PAD = .08
        if r == 0:
            cell.set_facecolor('#EEF4F6')
            cell.set_text_props(color=INK)
        else:
            cell.set_facecolor('white')
        if aligns and c < len(aligns):
            cell.set_text_props(ha=aligns[c])
    return tb


def paragraph(fig, x, y, text, size=9.2, width=110, leading=1.55, color=MUTED):
    text = clean(text)
    lines = []
    current = ''
    units = 0
    for char in text:
        unit = 2 if unicodedata.east_asian_width(char) in 'WF' else 1
        if char == '\n' or units + unit > width:
            lines.append(current)
            current, units = '', 0
            if char == '\n':
                continue
        current += char
        units += unit
    if current:
        lines.append(current)
    for index, value in enumerate(lines):
        label(fig, x, y - index * size * leading / 841.68, value, size, color)
    return y - len(lines) * size * leading / 841.68


def neighbor_case(fig, case, y_top, y_bottom):
    label(fig, .065, y_top, f"{case['group_zh']}  |  商品 ID {case['anchor_id']}  |  {case['brand']}", 11, TEAL)
    y = y_top - .024
    anchor_lines = textwrap.wrap(clean(case['title']), width=105, break_long_words=True, break_on_hyphens=False)
    for value in anchor_lines:
        label(fig, .065, y, value, 8.7)
        y -= .013
    if case['description_discarded_by_original_rule']:
        label(fig, .065, y, '原描述按项目长度规则丢弃；本次向量主要由标题编码。', 7.9, GOLD)
        y -= .015
    y -= .005
    for x, value, align in [(.067, '#', 'left'), (.128, '商品 ID', 'left'), (.205, '近邻商品完整标题', 'left'),
                             (.785, 'cos', 'right'), (.872, '余弦距离', 'right'), (.94, '原始 L2', 'right')]:
        label(fig, x, y, value, 7.8, MUTED, ha=align)
    line(fig, y - .014)
    y -= .023
    for neighbor in case['neighbors']:
        full_title = clean(neighbor['neighbor_title'])
        title_lines = textwrap.wrap(full_title, width=80, break_long_words=True, break_on_hyphens=False)
        row_height = max(.024, len(title_lines) * .0106 + .008)
        label(fig, .067, y, neighbor['rank'], 8)
        label(fig, .128, y, neighbor['neighbor_id'], 8)
        for index, value in enumerate(title_lines):
            label(fig, .205, y - index * .0106, value, 7.4)
        label(fig, .785, y, f"{neighbor['cosine_similarity']:.4f}", 8, ha='right')
        label(fig, .872, y, f"{neighbor['cosine_distance']:.4f}", 8, ha='right')
        label(fig, .94, y, f"{neighbor['raw_l2_distance']:.2f}", 8, ha='right')
        line(fig, y - row_height + .003)
        y -= row_height
    if y < y_bottom:
        raise ValueError(f"Case {case['anchor_id']} overflow: {y:.3f} < {y_bottom:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUT / 'analysis.json')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding='utf-8'))
    s, d, brand, retrieval, proj, examples = [data[key] for key in ('summary', 'distributions', 'brand_control', 'retrieval', 'projection', 'examples')]
    groups = s['by_group']
    assert len(groups) == 4 and len(examples) == 6
    assert sum(group['anchors'] for group in groups) == s['pair_anchors']
    assert all(len(d[key]) == s['pair_anchors'] for key in ('same_cosine', 'different_cosine', 'same_l2', 'different_l2'))
    assert all(len(case['neighbors']) == 5 for case in examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / 'video-games-embedding-similarity-report.pdf'
    stamp = datetime.fromisoformat(data['provenance']['created_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    generated = []
    def save(pdf, fig, page):
        pdf.savefig(fig)
        path = args.output_dir / f'video-games-embedding-similarity-report-{page}.png'
        fig.savefig(path, dpi=160, facecolor='white')
        generated.append(path.name)
        plt.close(fig)
    with PdfPages(pdf_path, metadata={'Title': 'Video Games 商品向量相似度报告', 'Author': 'MiniOneRec', 'Subject': '元数据弱标签、距离分布、近邻案例与可复现统计'}) as pdf:
        fig = header('相似商品的向量是否更近？', f"全量 {s['items']:,} 件商品 / {s['dimensions']:,} 维  |  实际分析  |  {stamp} 北京时间", 1)
        cards = [('同类 cos 更高的胜率', f"{s['macro_cosine_win_rate']:.2%}", '四类等权宏平均'),
                 ('Top5 同标签份额', f"{retrieval['cosine_macro'][1]:.2%}", '候选仅四类子集；宏平均'),
                 ('弱标签覆盖率', f"{s['labeled_coverage']:.2%}", f"{s['labeled_items']:,} / {s['items']:,} 件商品")]
        for x, (title, value, note) in zip([.065, .365, .665], cards):
            label(fig, x, .837, title, 9.4, MUTED)
            label(fig, x, .810, value, 25, TEAL)
            label(fig, x, .768, note, 8.3, MUTED)
        fig.add_artist(Rectangle((.065, .667), .875, .070, transform=fig.transFigure, facecolor='#EEF6F6', edgecolor='none'))
        label(fig, .085, .721, '在这四类硬件上，同类商品通常更相近。', 13)
        label(fig, .085, .691, '这些结果衡量类别语义一致性，不代表用户推荐准确率。', 9.5, MUTED)
        label(fig, .065, .635, '01  同类与异类商品对的距离分布', 13)
        ax = fig.add_axes([.115, .345, .34, .242])
        boxplot(ax, [d['same_cosine'], d['different_cosine']], ['同类', '异类'], [TEAL, BLUE])
        ax.set_ylabel('余弦相似度 cos（越高越近）', fontsize=9)
        ax.set_ylim(max(0, min(min(d['same_cosine']), min(d['different_cosine'])) - .035), 1.02)
        ax = fig.add_axes([.60, .345, .34, .242])
        boxplot(ax, [d['same_l2'], d['different_l2']], ['同类', '异类'], [TEAL, BLUE])
        ax.set_ylabel('原始向量欧氏距离 L2（越低越近）', fontsize=9)
        label(fig, .065, .306, f"{s['pair_anchors']:,} 个锚点，各配 1 个同类和 1 个异类候选，共 {2*s['pair_anchors']:,} 个商品对。", 9)
        label(fig, .065, .282, '箱线图合并抽样商品对（pooled）；与上方四类等权宏平均的口径不同。', 8.7, MUTED)
        rows = [['余弦相似度均值', f"{np.mean(d['same_cosine']):.4f}", f"{np.mean(d['different_cosine']):.4f}"],
                ['原始 L2 距离均值', f"{np.mean(d['same_l2']):.2f}", f"{np.mean(d['different_l2']):.2f}"]]
        table(fig, ['合并抽样统计', '同类商品对', '异类商品对'], rows, [.065, .150, .875, .105], [.46, .27, .27], 9.5)
        label(fig, .065, .119, '箱体为第 25-75 百分位；须长为 1.5 IQR；菱形为均值，散点为离群值。', 8.2, MUTED)
        label(fig, .065, .095, f"欧氏距离宏平均胜率为 {s['macro_l2_win_rate']:.2%}；两种指标在此检验中表现接近。", 8.8, MUTED)
        save(pdf, fig, 1)

        fig = header('分品类结果与品牌因素', '主对照按四类等权汇总；品牌分组是观察性分析', 2)
        label(fig, .065, .839, '02  四类硬件的配对对照', 13)
        rows = [[g['group_zh'], f"{g['catalog_items']:,}", f"{g['anchors']:,}", f"{g['same_cos_mean']:.4f}",
                 f"{g['different_cos_mean']:.4f}", f"{g['cos_gap']:+.4f}", f"{g['cosine_win_rate']:.2%}", f"{g['l2_win_rate']:.2%}"] for g in groups]
        rows.append(['四类宏平均', '-', '-', f"{np.mean([g['same_cos_mean'] for g in groups]):.4f}",
                     f"{np.mean([g['different_cos_mean'] for g in groups]):.4f}", f"{s['macro_cosine_gap']:+.4f}",
                     f"{s['macro_cosine_win_rate']:.2%}", f"{s['macro_l2_win_rate']:.2%}"])
        table(fig, ['类目', '商品数', '锚点数', '同类 cos', '异类 cos', 'cos 差值', 'cos 胜率', 'L2 胜率'], rows,
              [.065, .642, .875, .163], [.14,.10,.10,.13,.13,.13,.135,.135], 8.3)
        label(fig, .065, .618, '胜率：同一锚点的同类 cos 高于异类，或同类 L2 低于异类；不是分类准确率。', 8.4, MUTED)
        label(fig, .065, .580, '03  品牌与品类的四组观察性对照', 13)
        ax = fig.add_axes([.12, .316, .79, .216])
        boxplot(ax, [g['cos_samples'] for g in brand], ['同类\n同品牌','同类\n异品牌','异类\n同品牌','异类\n异品牌'], [TEAL, '#64A3AC', '#8DAAC8', BLUE])
        ax.set_ylabel('余弦相似度 cos', fontsize=9)
        rows = [[g['condition_zh'], f"{g['sampled_pairs']:,}", f"{g['cos_mean']:.4f}", f"{g['cos_median']:.4f}", f"{g['l2_mean']:.2f}"] for g in brand]
        table(fig, ['观察分组', '抽样商品对', 'cos 均值', 'cos 中位数', '原始 L2 均值'], rows,
              [.065, .122, .875, .143], [.28,.18,.18,.18,.18], 9)
        label(fig, .065, .095, f"共 {sum(g['sampled_pairs'] for g in brand):,} 对；排除空品牌与相同标题。未匹配全部混杂因素，不作因果解释。", 8.2, MUTED)
        save(pdf, fig, 2)

        fig = header('二维结构与子集内近邻检索', '二维投影帮助观察结构；指标在原始 2560 维空间计算', 3)
        label(fig, .065, .839, '04  四类硬件的 PCA 投影', 13)
        ax = fig.add_axes([.12, .484, .79, .311])
        points = proj['points']
        for index, group in enumerate(groups):
            selected = [p for p in points if p['group'] == index]
            ax.scatter([p['x'] for p in selected], [p['y'] for p in selected], s=13, alpha=.62,
                       c=COLORS[index], linewidths=0, label=f"{group['group_zh']} ({len(selected):,})")
        ratios = proj['explained_variance_ratio']
        ax.set_xlabel(f'PC1（解释方差 {ratios[0]:.2%}）', fontsize=9)
        ax.set_ylabel(f'PC2（解释方差 {ratios[1]:.2%}）', fontsize=9)
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='white', fontsize=8.3, markerscale=1.6)
        style(ax, 'both')
        label(fig, .065, .444, f"投影仅保留 {sum(ratios):.2%} 方差；图上点距不等于 2560 维距离，不能用于判定单个近邻。", 8.4, MUTED)
        label(fig, .065, .407, '05  TopK 中同标签商品的份额', 13)
        ax = fig.add_axes([.12, .150, .79, .211])
        x = np.arange(len(retrieval['ks']))
        series = [('余弦检索', retrieval['cosine_macro'], TEAL), ('原始 L2 检索', retrieval['l2_macro'], BLUE),
                  ('随机期望', [retrieval['random_macro']] * len(x), '#BACAD0')]
        for offset, (name, values, color) in zip([-.25, 0, .25], series):
            bars = ax.bar(x + offset, values, width=.23, color=color, label=name)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.019, f'{value:.2%}', fontsize=7.5, ha='center', color=INK)
        ax.set_xticks(x, [f'Top{k}' for k in retrieval['ks']])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('同标签份额（四类宏平均）', fontsize=9)
        ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(-.01, 1.18), frameon=False, fontsize=8.2)
        style(ax)
        label(fig, .065, .110, f"检索的查询和候选都仅含 {retrieval['candidate_items']:,} 件四类商品；排除自身，按类等权宏平均。", 8.4, MUTED)
        label(fig, .065, .087, '类别标签不是人工相关性判断；这些份额不是全库召回率，也不是用户推荐效果。', 8.4, MUTED)
        save(pdf, fig, 3)

        for page, cases in [(4, examples[:3]), (5, examples[3:])]:
            fig = header('全库近邻案例' + ('：手柄、耳机、鼠标' if page == 4 else '：键盘与游戏作品'),
                         f"预定义锚点  |  在 {s['items']:,} 件商品中按 cos 精确检索，排除自身，完整展示 Top5", page)
            for case, top, bottom in zip(cases, [.837, .580, .323], [.595, .338, .081]):
                neighbor_case(fig, case, top, bottom)
            label(fig, .065, .070, '排名仅按 cos 降序；表中 L2 为同一组近邻的原始距离，并非 L2 排名。', 7.8, MUTED)
            save(pdf, fig, page)

        fig = header('方法、边界与可复现文件', '全部图表由 analysis.json 驱动；输入商品向量保持原样', 6)
        y = .839
        sections = [
            ('06  距离如何定义',
             '余弦相似度 cos(x,y) = (x·y) / (||x|| ||y||)，越高表示方向越接近；余弦距离 = 1-cos，越低越近。原始欧氏距离 L2(x,y) = ||x-y||，同时受方向与向量长度影响。分析时仅临时对副本做 L2 归一化计算余弦和 PCA，源向量没有修改。'),
            ('07  对照如何抽样',
             f"标签由元数据中明确的 Controllers / Headsets / Gaming Mice / Gaming Keyboards 类目 token 定义，先做 HTML 反转义。固定随机种子 {s['seed']}；每类最多抽取 300 个不同锚点，各选 1 个不同标题的同类商品和 1 个随机异类候选。主指标先在每类计算，再对四类等权平均。"),
            ('08  结果适用于哪里',
             f"弱标签仅覆盖 {s['labeled_items']:,} 件商品（{s['labeled_coverage']:.2%}），含部分适配器等类别噪声，不能推广为全类目质量保证。品牌四组只反映观察性相关，商品对可能共享商品，不能视为独立因果实验。游戏作品案例只作展示，不参与四类统计。未使用人工相关性标注或推荐效果评测。近邻检索排除自身，但保留同标题和近重复版本，可能提高同类占比；配对比较已排除同标题。"),
            ('09  文本编码口径',
             '使用 Qwen3-Embedding-4B 对商品标题与描述编码，沿用项目 masked mean pooling，不额外归一化。原文本规则会丢弃清洗后达到 2000 字符的单个字段；案例已标出受影响锚点。品牌、类目和用户行为未作为文本编码字段输入。')]
        for title, body in sections:
            label(fig, .065, y, title, 12)
            y = paragraph(fig, .065, y-.031, body, 9.1, width=111, leading=1.55) - .022
        label(fig, .065, y, '10  输入文件 SHA-256', 12)
        y -= .030
        label(fig, .065, y, 'Video_Games.emb-qwen-td.npy', 8.7, MUTED)
        label(fig, .065, y-.020, data['provenance']['embedding_sha256'], 8)
        label(fig, .065, y-.044, 'Video_Games.item.json', 8.7, MUTED)
        label(fig, .065, y-.064, data['provenance']['metadata_sha256'], 8)
        y -= .106
        label(fig, .065, y, '11  报告与表格文件', 12)
        files = ['video-games-embedding-similarity-report.pdf + 6 页 PNG',
                 'analysis.json / analysis_validation.json',
                 'sampled_pairs.csv / brand_controls.csv / labeled_retrieval.csv / nearest_neighbors.csv']
        for index, value in enumerate(files):
            label(fig, .065, y-.031-index*.022, value, 8.2, MUTED)
        save(pdf, fig, 6)
    manifest = {'pdf': str(pdf_path), 'pages': 6, 'page_images': generated, 'source': str(args.source),
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model_id': s['model_id'],
                'macro_cosine_win_rate': s['macro_cosine_win_rate'], 'top5_same_label_macro': retrieval['cosine_macro'][1]}
    (args.output_dir / 'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
