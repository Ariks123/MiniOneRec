#!/usr/bin/env python3
"""Build the offline landing page for both Video Games visualization reports."""
from pathlib import Path
from html import escape, unescape
import json

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/reports/Video_Games'
NEW=BASE/'02_embedding_similarity'
r=json.loads((NEW/'analysis.json').read_text(encoding='utf-8'))
s=r['summary']; retrieval=r['retrieval']

def clean(value):return escape(unescape(str(value)))

cases=[]
for ex in r['examples']:
    rows=''.join(f'<tr><td>{n["rank"]}</td><td>{n["neighbor_id"]}</td><td>{clean(n["neighbor_title"])}</td><td>{n["cosine_similarity"]:.4f}</td><td>{n["cosine_distance"]:.4f}</td><td>{n["raw_l2_distance"]:.2f}</td></tr>' for n in ex['neighbors'])
    note='<p class="muted">该锚点描述因原项目的 2000 字符规则被丢弃，本次主要由标题编码。</p>' if ex['description_discarded_by_original_rule'] else ''
    cases.append(f'<details><summary>ID {ex["anchor_id"]} · {clean(ex["title"])}</summary>{note}<div class="scroll"><table><thead><tr><th>排名</th><th>商品 ID</th><th>近邻商品完整标题</th><th>余弦相似度 ↑</th><th>余弦距离 ↓</th><th>原始 L2 距离 ↓</th></tr></thead><tbody>{rows}</tbody></table></div></details>')
class_rows=''.join(f'<tr><td>{g["group_zh"]}</td><td>{g["catalog_items"]}</td><td>{g["anchors"]}</td><td>{g["same_cos_mean"]:.4f}</td><td>{g["different_cos_mean"]:.4f}</td><td>{g["cosine_win_rate"]:.2%}</td><td>{g["l2_win_rate"]:.2%}</td></tr>' for g in s['by_group'])
brand_rows=''.join(f'<tr><td>{b["condition_zh"]}</td><td>{b["sampled_pairs"]:,}</td><td>{b["cos_mean"]:.4f}</td><td>{b["l2_mean"]:.2f}</td></tr>' for b in r['brand_control'])
new_pages=''.join(f'<a href="02_embedding_similarity/video-games-embedding-similarity-report-{i}.png"><img loading="lazy" src="02_embedding_similarity/video-games-embedding-similarity-report-{i}.png" alt="向量相似性报告第 {i} 页"></a>' for i in range(1,7))
old_pages=''.join(f'<a href="01_cleaning/video-games-cleaning-report-{i}.png"><img loading="lazy" src="01_cleaning/video-games-cleaning-report-{i}.png" alt="数据清洗报告第 {i} 页"></a>' for i in range(1,4))
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Video Games 数据与向量报告</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f6;color:#172d35;font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 26px}}h1{{font-size:34px;line-height:1.3;margin:8px 0 12px}}h2{{font-size:23px;margin:0 0 16px}}h3{{font-size:18px}}p{{margin:8px 0 15px}}a{{color:#126f80;text-decoration:none}}a:hover{{text-decoration:underline}}.eyebrow{{letter-spacing:.14em;color:#167d8d;font-size:13px;font-weight:700}}.muted{{color:#657983;font-size:14px}}.cards,.kpis{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin:24px 0}}.kpis{{grid-template-columns:repeat(3,1fr)}}.card,section{{background:white;border:1px solid #dae6e9;border-radius:14px;padding:26px;margin-bottom:24px}}.card{{margin:0}}.kpis .card b{{display:block;font-size:31px;color:#167d8d}}.button{{display:inline-block;background:#167d8d;color:white;padding:9px 16px;border-radius:7px;margin:8px 8px 0 0}}.button.secondary{{background:#edf5f7;color:#126f80}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #e3ebee;vertical-align:top}}th{{background:#eef5f7;font-weight:600;white-space:nowrap}}td:nth-last-child(-n+3){{font-variant-numeric:tabular-nums}}summary{{cursor:pointer;font-weight:600;padding:16px 0}}details{{border-top:1px solid #e3ebee}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.gallery img{{width:100%;border:1px solid #dbe6e9;box-shadow:0 3px 12px #172d3514}}code{{background:#edf5f7;padding:2px 6px;border-radius:4px}}footer{{color:#657983;font-size:13px;padding:8px 0 25px}}@media(max-width:760px){{main{{padding:24px 16px}}.cards,.kpis,.gallery{{grid-template-columns:1fr}}h1{{font-size:29px}}.card,section{{padding:19px}}}}</style>
<main><div class="eyebrow">MINIONEREC · VIDEO GAMES</div><h1>数据清洗与商品向量报告</h1><p class="muted">所有报告、页面预览与数据表统一存放于此目录，可离线浏览。来源：本地实际清洗结果及 Qwen3-Embedding-4B 商品向量。</p>
<div class="cards"><article class="card"><div class="eyebrow">01 · 已有报告</div><h2>数据清洗报告</h2><p>1–20 条历史窗口、清洗漏斗、输出文件及修改前后对比。</p><a class="button" href="01_cleaning/video-games-cleaning-report.pdf">打开 3 页 PDF</a><a class="button secondary" href="#cleaning">查看页面预览</a></article><article class="card"><div class="eyebrow">02 · 新报告</div><h2>向量距离与相似商品</h2><p>类内/类间距离、品牌对照、PCA、四类子集检索及全库近邻。</p><a class="button" href="02_embedding_similarity/video-games-embedding-similarity-report.pdf">打开 6 页 PDF</a><a class="button secondary" href="#embedding">查看图表与数据</a></article></div>
<section id="embedding"><h2>结果概览</h2><p>这四类商品在向量中表现出明显的类别结构；同类商品通常比异类商品更接近。</p><div class="kpis"><div class="card"><span>同类余弦更高 · 类宏平均</span><b>{s['macro_cosine_win_rate']:.2%}</b><span class="muted">833 个锚点，各配一对同类与异类商品</span></div><div class="card"><span>子集 Top-5 同标签占比</span><b>{retrieval['cosine_macro'][1]:.2%}</b><span class="muted">仅 1,159 件四类商品内；随机基线 {retrieval['random_macro']:.2%}</span></div><div class="card"><span>有明确类目弱标签的覆盖</span><b>{s['labeled_coverage']:.2%}</b><span class="muted">1,159 / 16,572 件，非人工相关性评测</span></div></div><p class="muted">余弦相似度越高表示方向越接近；余弦距离 = 1 − 相似度。L2 为原始未归一化向量间欧氏距离，越小越近。源向量没有被修改。</p></section>
<section><h2>按类目比较</h2><div class="scroll"><table><thead><tr><th>类目</th><th>商品数</th><th>锚点数</th><th>同类平均余弦</th><th>异类平均余弦</th><th>余弦同类更近</th><th>L2 同类更近</th></tr></thead><tbody>{class_rows}</tbody></table></div><p class="muted">标签来自元数据的 Controllers / Headsets / Gaming Mice / Gaming Keyboards 明确 token。按类等权的宏平均可减少类目规模差异的影响。</p></section>
<section><h2>品牌对照</h2><div class="scroll"><table><thead><tr><th>对照条件</th><th>抽样商品对</th><th>平均余弦相似度</th><th>平均原始 L2 距离</th></tr></thead><tbody>{brand_rows}</tbody></table></div><p class="muted">同类异品牌仍比异类同品牌更接近；品牌也会影响距离。这里是观察性对照，不能把均值差异解释为因果作用。</p></section>
<section><h2>六个固定锚点的全库 Top-5 近邻</h2><p class="muted">锚点由元数据预先选定，近邻从全部 16,572 件商品中精确检索；点击标题展开完整商品表。</p>{''.join(cases)}</section>
<section><h2>向量报告页面预览</h2><div class="gallery">{new_pages}</div></section>
<section><h2>下载数据表</h2><p><a class="button secondary" href="02_embedding_similarity/sampled_pairs.csv">同类/异类商品对 CSV</a><a class="button secondary" href="02_embedding_similarity/nearest_neighbors.csv">全库近邻 CSV</a><a class="button secondary" href="02_embedding_similarity/brand_controls.csv">品牌对照 CSV</a><a class="button secondary" href="02_embedding_similarity/labeled_retrieval.csv">子集检索 CSV</a></p><p><a href="02_embedding_similarity/analysis.json">完整分析数据 JSON</a> · <a href="02_embedding_similarity/analysis_validation.json">独立计算核验记录</a></p><p class="muted">本报告只检验向量中的商品内容结构，不等同于推荐准确率。类别中仍存在配件混入或标注噪声；近邻排除自身，但保留同标题与近重复版本，可能提高同类占比。配对比较已排除同标题。PCA 二维图只供观察，不能代替原 2560 维距离。</p></section>
<section id="cleaning"><h2>数据清洗报告页面预览</h2><div class="gallery">{old_pages}</div></section><footer>分析输入 SHA-256：{r['provenance']['embedding_sha256']}<br>原输出位置保留；本目录归档清洗报告，并收录新向量报告。</footer></main></html>'''
active_path=ROOT/'output/rqvae/Video_Games/active_run.json'
if active_path.exists():
    active=json.loads(active_path.read_text(encoding='utf-8'))
    run=Path(active['run_dir'])
    summary_path=run/'training_summary.json';validation_path=run/'sid_validation.json'
    if summary_path.exists() and validation_path.exists():
        training=json.loads(summary_path.read_text(encoding='utf-8'));validation=json.loads(validation_path.read_text(encoding='utf-8'))
        specs=[('03_rqvae_training','video-games-rqvae-training-report','RQ-VAE 训练报告',f"训练 {training['epochs_completed']:,} 轮后早停，选中第 {training['best_epoch']:,} 轮；分组留出 1,653 件商品。"),
               ('04_sid_validation','video-games-sid-validation-report','SID 验证与 10,000 轮对比','当前采用第 2,900 轮 SID；含完整语义诊断、同口径 10,000 轮对比及选型依据。')]
        cards=[];galleries=[]
        for folder,stem,title,description in specs:
            pdf=BASE/folder/(stem+'.pdf')
            if not pdf.exists():continue
            images=sorted((BASE/folder).glob(stem+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[-1]))
            cards.append(f'<article class="card"><div class="eyebrow">{folder[:2]} · 新报告</div><h2>{title}</h2><p>{description}</p><a class="button" href="{folder}/{stem}.pdf">打开 {len(images)} 页 PDF</a><a class="button secondary" href="#{folder}">查看页面预览</a></article>')
            gallery=''.join(f'<a href="{folder}/{p.name}"><img loading="lazy" src="{folder}/{p.name}" alt="{title}第 {i} 页"></a>' for i,p in enumerate(images,1))
            galleries.append(f'<section id="{folder}"><h2>{title}页面预览</h2><div class="gallery">{gallery}</div></section>')
        if cards:
            raw=validation['knn']['metrics']['raw_reconstruction']['val_queries']['overlap_at_10']
            repaired=validation['knn']['metrics']['semantic_reconstruction']['val_queries']['overlap_at_10']
            initial=validation['knn']['metrics']['initial_reconstruction']['val_queries']['overlap_at_10']
            insight=f'<section><h2>训练与 SID 验证结果</h2><p>全部 16,572 件商品获得唯一四 token SID。共享的语义前缀越长，商品原向量通常越相似。</p><p>留出商品 Top-10 近邻保持率：初始化 {initial:.2%} → 原始三层码 {raw:.2%} → 碰撞修复后三层码 {repaired:.2%}。碰撞修复提高唯一性，也带来额外近邻损失；第一级码本使用 {validation["codebook_usage"]["semantic"]["all"][0]["used"]}/{validation["codebook_usage"]["semantic"]["all"][0]["codewords"]}，仍有优化空间。</p><p class="muted">第四个 token 仅区分身份，不计入语义评分；这些结果不是推荐准确率。</p></section>'
            page=page.replace('<section id="embedding">','<div class="cards">'+''.join(cards)+'</div>'+insight+'<section id="embedding">')
            page=page.replace('<section id="cleaning">',''.join(galleries)+'<section id="cleaning">')
comparison_dir=BASE/'05_rqvae_10000_comparison'
comparison_stem='video-games-rqvae-10000-comparison-report'
comparison_pdf=comparison_dir/(comparison_stem+'.pdf')
if comparison_pdf.exists():
    images=sorted(comparison_dir.glob(comparison_stem+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[-1]))
    card=f'<div class="cards"><article class="card"><div class="eyebrow">05 · 轮数对比</div><h2>第 2,900 轮与第 10,000 轮</h2><p>相同数据划分与初始化轨迹；对比留出误差、相似商品近邻保留、码本和碰撞修复代价。</p><a class="button" href="05_rqvae_10000_comparison/{comparison_stem}.pdf">打开 {len(images)} 页 PDF</a><a class="button secondary" href="#comparison_10000">查看页面预览</a><a class="button secondary" href="05_rqvae_10000_comparison/comparison.json">对比数据 JSON</a></article></div>'
    gallery=''.join(f'<a href="05_rqvae_10000_comparison/{p.name}"><img loading="lazy" src="05_rqvae_10000_comparison/{p.name}" alt="10000 轮对比报告第 {i} 页"></a>' for i,p in enumerate(images,1))
    page=page.replace('<section id="embedding">',card+'<section id="embedding">')
    page=page.replace('<section id="cleaning">',f'<section id="comparison_10000"><h2>10,000 轮训练对比页面预览</h2><div class="gallery">{gallery}</div></section><section id="cleaning">')
selection_path=ROOT/'output/rqvae/Video_Games/selected_sid.json'
if selection_path.exists():
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    selected_epoch=selection['selected_epoch']
    selected_items=selection['items']
    selected_banner=f'<section id="selected_sid"><div class="eyebrow">当前项目采用</div><h2>第 {selected_epoch:,} 轮 SID</h2><p>原训练完成 4,100 轮，按验证 MSE 选中第 {selected_epoch:,} 轮。全部 {selected_items:,} 件商品具有唯一完整 SID。验证报告包含与第 10,000 轮的对比；当前选择为保守基线，尚无下游 HR / NDCG 优胜结论。</p><p><a class="button" href="03_rqvae_training/video-games-rqvae-training-report.pdf">对应训练报告</a><a class="button" href="04_sid_validation/video-games-sid-validation-report.pdf">验证报告（含 10,000 轮对比）</a><a class="button secondary" href="../../../data/Amazon18/Video_Games/Video_Games.index.json">当前 SID JSON</a><a class="button secondary" href="../../rqvae/Video_Games/selected_sid.json">选用记录</a></p></section>'
    page=page.replace('<div class="cards">',selected_banner+'<div class="cards">',1)
(BASE/'index.html').write_text(page,encoding='utf-8')
print(BASE/'index.html')
