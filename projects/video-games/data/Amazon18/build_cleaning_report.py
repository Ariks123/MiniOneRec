#!/usr/bin/env python3
"""Build a three-page visual report from the completed cleaning manifest."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
FONT = Path('C:/Windows/Fonts/simhei.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                     'text.color': '#172D35', 'axes.labelcolor': '#49616B',
                     'xtick.color': '#657983', 'ytick.color': '#172D35'})
PAGE_COUNT = 3
INK, MUTED, TEAL, BLUE, GOLD, GRID = '#172D35', '#657983', '#167D8D', '#527DAB', '#C58D34', '#E3EBEE'

def text(fig, x, y, value, size=10, color=INK, **kw):
    return fig.text(x, y, value, fontsize=size, color=color, va='top', **kw)

def rule(fig, y):
    fig.add_artist(plt.Line2D([.07,.93],[y,y],transform=fig.transFigure,color=GRID,lw=.8))

def header(fig, title, subtitle, page):
    fig.patch.set_facecolor('white')
    fig.add_artist(Rectangle((.07,.944),.052,.006,transform=fig.transFigure,color=TEAL,lw=0))
    text(fig,.07,.925,title,23)
    text(fig,.07,.883,subtitle,9.2,MUTED)
    rule(fig,.055)
    text(fig,.07,.041,'MiniOneRec  |  Amazon Review Data (2018) / Video Games',8,MUTED)
    text(fig,.93,.041,f'{page} / {PAGE_COUNT}',8,MUTED,ha='right')

def axis_style(ax):
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both',length=0,pad=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True,color=GRID,lw=.7)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/Amazon18/Video_Games/cleaning_report.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/pdf')
    args=parser.parse_args()
    r=json.loads(args.source.read_text(encoding='utf-8'))
    assert r['validation']=='passed' and r['execution']['exit_code']==0
    stages=r['filter_stages']; raw=stages[0]; final=stages[-1]
    removed=raw['interactions']-r['interactions']
    assert sum(s['removed_interactions'] for s in stages)==removed
    assert sum(r['split_examples'].values())==r['examples']==r['interactions']-r['users']
    history_min,history_max=r['history_length_range']
    history_stats=r['history_stats']; comparison=r['history_comparison']
    history_lengths=list(range(history_min,history_max+1))
    history_counts=[history_stats['histogram'][str(length)] for length in history_lengths]
    assert sum(history_counts)==r['examples']
    assert sum(length*count for length,count in zip(history_lengths,history_counts))==history_stats['total_history_items']
    assert history_stats['samples_over_10']==sum(count for length,count in zip(history_lengths,history_counts) if length>10)
    assert comparison['current_max']==history_max
    args.output_dir.mkdir(parents=True,exist_ok=True)
    destination=args.output_dir/'video-games-cleaning-report.pdf'
    executed=datetime.fromisoformat(r['execution']['finished_at_utc'].replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8)))
    stamp=executed.strftime('%Y-%m-%d %H:%M')+' 北京时间'
    with PdfPages(destination,metadata={'Title':'Video Games 数据清洗报告','Author':'MiniOneRec','Subject':'数据过滤、样本划分、文件清单、质量检查与历史长度对比'}) as pdf:
        fig=plt.figure(figsize=(8.27,11.69))
        header(fig,'Video Games 数据清洗报告','清洗概览  |  实际执行结果  |  '+stamp,1)
        cards=[('保留用户',r['users'],f"原始 {raw['users']:,}  /  保留 {r['users']/raw['users']:.2%}"),
               ('保留商品',r['items'],f"原始 {raw['items']:,}  /  保留 {r['items']/raw['items']:.2%}"),
               ('保留交互',r['interactions'],f"原始 {raw['interactions']:,}  /  保留 {r['interactions']/raw['interactions']:.2%}"),
               ('推荐样本',r['examples'],'每位用户首条交互仅作为历史')]
        for x,(label,value,context) in zip([.07,.293,.516,.739],cards):
            text(fig,x,.837,label,10,MUTED)
            text(fig,x,.812,f'{value:,}',23)
            text(fig,x,.773,context,7.7,MUTED)
        rule(fig,.748)
        text(fig,.07,.725,'01  交互数据保留路径',14)
        text(fig,.93,.723,f'累计保留 {r["interactions"]/raw["interactions"]:.2%}',11,TEAL,ha='right')
        ax=fig.add_axes([.215,.449,.68,.242])
        labels=['原始评论','有商品元数据','标题合格','时间窗口内','迭代 5-core 后']
        values=[s['interactions'] for s in stages]
        ax.barh(range(5),values,color=['#C8DDE1','#ABCBD2','#88B6C0','#60A0AE',TEAL],height=.52,zorder=3)
        ax.set_yticks(range(5),labels); ax.invert_yaxis(); ax.set_xlim(0,620000)
        ax.set_xticks([0,100000,200000,300000,400000,500000]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(x/10000)}'))
        ax.set_xlabel('交互数量（万条）',labelpad=8,fontsize=9)
        for i,v in enumerate(values): ax.text(v+8500,i,f'{v:,}',va='center',fontsize=10)
        axis_style(ax)
        text(fig,.07,.405,'02  删除原因分解',14)
        text(fig,.93,.403,f'累计删除 {removed:,} 条 / {removed/raw["interactions"]:.2%}',11,TEAL,ha='right')
        ax=fig.add_axes([.215,.19,.68,.178])
        reasons=['缺少元数据','标题不合格','时间窗口外','5-core 迭代']
        loss=[s['removed_interactions'] for s in stages[1:]]
        ax.barh(range(4),loss,color=TEAL,height=.48,zorder=3)
        ax.set_yticks(range(4),reasons); ax.invert_yaxis(); ax.set_xlim(0,18000)
        ax.set_xticks([0,5000,10000,15000]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{int(x):,}'))
        ax.set_xlabel('删除交互数（条）',fontsize=9,labelpad=7)
        for i,v in enumerate(loss):
            ax.text(v+260,i,f'{v:,}  ({v/removed:.2%})',va='center',fontsize=9.2)
        axis_style(ax)
        text(fig,.07,.136,'删除原因百分比的分母为累计删除的 26,480 条交互；时间过滤实际删除 4 条。',8.7,MUTED)
        text(fig,.07,.113,'时间窗口：1996-10-01 00:00 至 2018-10-01 00:00（UTC+08:00，包含端点）。',8.7,MUTED)
        text(fig,.07,.09,'沿用项目规则：保留全部 1-5 星评分及重复交互；5-core 约束作用于完整清洗集。',8.7,MUTED)
        pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(8.27,11.69))
        header(fig,'样本划分与输出文件','运行成功  |  缺失元数据商品已全部移除  |  样本与文件校验通过',2)
        text(fig,.07,.838,'03  推荐样本与时间划分',14)
        text(fig,.07,.806,f"{r['interactions']:,} 条交互 - {r['users']:,} 位用户 = {r['examples']:,} 条下一商品预测样本",10)
        text(fig,.07,.781,f'按用户时间顺序构造 {history_min}-{history_max} 条历史，再按全局目标时间排序切分。',9,MUTED)
        ax=fig.add_axes([.07,.704,.86,.052]); ax.set_xlim(0,r['examples']); ax.set_ylim(-.5,.5); ax.axis('off')
        start=0
        colors=[TEAL,BLUE,GOLD]
        for label,split,color in zip(['训练','验证','测试'],['train','valid','test'],colors):
            n=r['split_examples'][split]
            ax.barh(0,n,left=start,height=.8,color=color)
            ax.text(start+n/2,0,f'{label} {n/r["examples"]:.0%}',ha='center',va='center',fontsize=10,color='white')
            start+=n
        for x,label,split in zip([.07,.37,.67],['训练集','验证集','测试集'],['train','valid','test']):
            b=r['split_boundaries'][split]
            text(fig,x,.691,f'{label}  {r["split_examples"][split]:,} 条',11)
            text(fig,x,.669,b['first_target_utc'][:10]+' 至 '+b['last_target_utc'][:10],8.2,MUTED)
        text(fig,.07,.642,'以上日期为 UTC；相同时间戳可能跨切分边界。并非按用户留最后两条。',8.7,MUTED)
        rule(fig,.619)
        text(fig,.07,.599,'04  清洗后文件清单',14)
        text(fig,.93,.596,f"8 个数据文件 / {r['total_data_file_bytes']/1e6:.2f} MB",10,TEAL,ha='right')
        purposes=['训练样本','验证样本','测试样本','用户完整序列','商品特征','评论与摘要','用户 ID 映射','商品 ASIN 映射']
        y=.556
        for x,label,ha in [(.07,'文件名','left'),(.51,'内容','left'),(.795,'条目数','right'),(.93,'MB','right')]:
            text(fig,x,y,label,9.2,MUTED,ha=ha)
        rule(fig,.535)
        for idx,(f,purpose) in enumerate(zip(r['files'],purposes)):
            y=.520-idx*.027
            text(fig,.07,y,f['name'],8.8)
            text(fig,.51,y,purpose,9)
            text(fig,.795,y,f"{f['records']:,}",9,ha='right')
            text(fig,.93,y,f"{f['bytes']/1e6:.2f}",9,ha='right')
            rule(fig,y-.020)
        text(fig,.07,.293,'目录：data/Amazon18/Video_Games/；样本行数不含表头；MB 按十进制计。',8.5,MUTED)
        text(fig,.07,.264,'05  校验结果与数据口径',14)
        notes=[
            f"缺少元数据的 {r['missing_metadata_asins_removed']} 件商品已全部剔除；输出中缺失元数据商品为 0。",
            f"完整清洗集：用户最少 {r['min_user_interactions']} 条交互，商品最少 {r['min_item_interactions']} 条交互；不保证训练集内也满足。",
            f"{r['items_with_nonempty_categories']:,} 件商品类别非空；仍有 {r['items']-r['items_with_nonempty_categories']:,} 件类别为空。已兼容 category 字段。",
            f"评论文本表为 {r['review_text_records']:,} 个唯一键，不代表非空评论数；同键覆盖 {r['same_user_item_timestamp_collisions']:,} 次。",
            '文本键为（用户、商品、时间戳）；覆盖仅影响文本表，交互和训练样本仍保留。',
            '数据来源：Kaggle 评论文件备份及商品元数据镜像 v2；原始下载保留在 raw/。',
        ]
        for i,note in enumerate(notes): text(fig,.07,.232-i*.024,note,8.6,MUTED)
        text(fig,.07,.077,'统计依据：cleaning_report.json 与 download_manifest.json；全部数值来自本次实际清洗。',8,MUTED)
        pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(8.27,11.69))
        header(fig,'历史长度扩展与样本对比',f'历史窗口 {history_min}-{history_max} 条  |  基于三个切分文件的实际逐行统计',3)
        cards=[('最大历史长度',f'{history_max} 条',f"修改前最多 {comparison['previous_max']} 条"),
               ('平均历史长度',f"{history_stats['mean']:.2f} 条",f"修改前 {comparison['previous_mean']:.2f} 条"),
               ('历史超过 10 条',f"{history_stats['samples_over_10']:,}",f"占全部样本 {history_stats['samples_over_10']/r['examples']:.2%}")]
        for x,(label,value,context) in zip([.07,.37,.67],cards):
            text(fig,x,.837,label,10,MUTED)
            text(fig,x,.812,value,23)
            text(fig,x,.773,context,8.5,MUTED)
        rule(fig,.748)
        text(fig,.07,.725,'06  历史长度分布',14)
        text(fig,.93,.723,f"合计 {r['examples']:,} 条样本",10,TEAL,ha='right')
        text(fig,.07,.701,'蓝色：历史不超过 10 条；青色：扩展后可使用更多历史的样本。',8.6,MUTED)
        ax=fig.add_axes([.105,.438,.81,.222])
        ax.bar(history_lengths,history_counts,color=[BLUE if length<=10 else TEAL for length in history_lengths],width=.68,zorder=3)
        ax.set_xticks(history_lengths)
        ax.set_xlim(history_min-.65,history_max+.65)
        ax.set_ylim(0,max(history_counts)*1.43)
        ax.tick_params(axis='both',length=0,pad=5,labelsize=8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True,color=GRID,lw=.7)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value/10000:g}'))
        ax.set_ylabel('样本数（万条）',fontsize=9,labelpad=8)
        ax.set_xlabel('每条样本包含的历史商品数',fontsize=9,labelpad=8)
        for length,count in zip(history_lengths,history_counts):
            ax.text(length,count+max(history_counts)*.025,f'{count:,}',ha='left',va='bottom',rotation=60,fontsize=7)
        text(fig,.07,.385,'每条预测样本只计一次；历史达到窗口上限时，仅保留目标之前最近的商品。',8.5,MUTED)
        rule(fig,.367)
        text(fig,.07,.348,'07  修改前后对比',14)
        for x,label,ha in [(.07,'指标','left'),(.63,f"最多 {comparison['previous_max']} 条历史",'right'),(.93,f"最多 {comparison['current_max']} 条历史",'right')]:
            text(fig,x,.313,label,9.2,MUTED,ha=ha)
        rule(fig,.292)
        rows=[('平均历史长度',f"{comparison['previous_mean']:.2f} 条",f"{comparison['current_mean']:.2f} 条"),
              ('历史超过 10 条的样本','0',f"{history_stats['samples_over_10']:,}"),
              ('推荐样本总数',f"{r['examples']:,}",f"{r['examples']:,}"),
              ('三个切分文件合计',f"{comparison['previous_split_file_bytes']/1e6:.2f} MB",f"{comparison['current_split_file_bytes']/1e6:.2f} MB")]
        for index,(label,previous,current) in enumerate(rows):
            y=.277-index*.032
            text(fig,.07,y,label,9.3)
            text(fig,.63,y,previous,10,ha='right')
            text(fig,.93,y,current,10,TEAL,ha='right')
            rule(fig,y-.021)
        unchanged=[('用户数','users_unchanged'),('商品数','items_unchanged'),('交互数','interactions_unchanged'),('样本总数','examples_unchanged'),('目标与切分','targets_and_splits_unchanged')]
        check_text='；'.join(label+('一致' if comparison[key] else '有变化') for label,key in unchanged)+'。'
        text(fig,.07,.134,check_text,8.3,MUTED)
        text(fig,.07,.110,f"{comparison['samples_with_longer_history']:,} 条样本的历史变长；样本数量与每条样本的目标商品保持一致。",8.5,MUTED)
        text(fig,.07,.086,'平均值按全部样本加权；文件大小按十进制 MB 计，历史变长会增加存储与训练序列长度。',8.2,MUTED)
        pdf.savefig(fig); plt.close(fig)
    summary={'pdf':str(destination),'source':str(args.source),'pages':PAGE_COUNT,'retained_interactions':r['interactions'],'retention':r['interactions']/raw['interactions'],'removed':removed}
    (args.output_dir/'video-games-report-manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
