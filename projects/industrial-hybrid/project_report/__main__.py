"""Build, audit, and package the three Industrial and Scientific reports."""

import argparse
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

from hybrid.rho_bundle import digest, publish
from .provenance import audit


NAME = "Industrial-2018-Algorithm-Training-Validation-2K"
PAGES = {
    "index.html": "总览",
    "algorithm.html": "算法文档",
    "training.html": "训练报告",
    "validation.html": "验证报告",
}


def page(filename, title, description, body, sections):
    navigation = "".join(
        f'<a href="{href}" aria-current="page">{label}</a>' if href == filename
        else f'<a href="{href}">{label}</a>'
        for href, label in PAGES.items()
    )
    toc = "".join(f'<li><a href="#{escape(anchor, quote=True)}">{escape(label)}</a></li>'
                  for anchor, label in sections)
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)} · Amazon Reviews 2018</title>'
        '<link rel="stylesheet" href="report.css"></head><body>'
        f'<nav class="top" aria-label="报告导航"><div class="frame">{navigation}</div></nav>'
        '<header class="hero" id="top"><div class="frame">'
        '<p class="dataset">Amazon Reviews 2018 · Industrial and Scientific · UniSRec 2K</p>'
        f'<h1>{escape(title)}</h1><p class="summary">{escape(description)}</p></div></header>'
        '<main class="frame"><nav class="toc" aria-label="章节目录">'
        f'<p class="toc-title">目录</p><ol>{toc}</ol></nav>{body}'
        '<footer><a href="#top">回到页首</a> · <a href="index.html">报告总览</a> · '
        '<a href="data/provenance.json">源文件和检查点校验清单</a></footer></main></body></html>'
    )


def overview(lineage):
    counts = lineage["rows_by_split"]
    train = counts["train"]
    valid = counts["valid"]
    test = counts["test"]
    body = f"""
<section><h2 id="reports">报告目录</h2>
<div class="report-list">
  <article><h3><a href="algorithm.html">算法文档</a></h3>
  <p>从数据和三位商品 SID 到 SFT、GRPO、UniSRec 训练。详解 2K 联合推理的流程、模型结构、公式、伪代码和复现参数。</p></article>
  <article><h3><a href="training.html">可视化训练报告</a></h3>
  <p>对照真实曲线和日志，解释训练任务、学习率、早停、选点及不同训练设置的结果。</p></article>
  <article><h3><a href="validation.html">可视化验证报告与结果分析</a></h3>
  <p>逐 K 展示 2K 的质量和速度，复核纯 beam、UniSRec 原排、每轮取 K 与 2K 的差异。</p></article>
</div></section>
<section><h2 id="protocol">方案与实验范围</h2>
<p>最终推理每轮由 UniSRec 提出最多 2K 个商品。Qwen2.5-0.5B 的 GRPO 检查点为这些商品的完整三位 SID 打分，
按已验证的阈值与排序策略给出 K 个推荐。覆盖 K=1、5、10、20、50。实验输入有 {lineage['catalog_items']:,} 件商品、
{train:,} 条训练、{valid:,} 条验证和 {test:,} 条测试请求。测试中有 {lineage['cold_test_rows']:,} 条冷目标。</p>
<p class="note">上游说明预置工业品 SID 通过 RQ-KMeans 得到。算法文档同时说明 RQ-VAE 的参考实现。
验证报告列出统一 2K 的最终结果和逐 K 对照。</p></section>
<section><h2 id="evidence">检查点与证据</h2>
<p>从独立线性三轮 SFT 组的 {lineage['sft_checkpoint']} 进入 GRPO，由 β=0.001 组的 {lineage['rl_checkpoint']}
提供推理打分。UniSRec 采用 20 轮组的第 {lineage['unisrec_epoch']} 轮。输入、导出权重和推理清单已核对。
报告基于已完成的实验，所有图表和结果均来自归档记录及离线复算。</p>
<p><a href="data/provenance.json">下载源文件相对路径、大小和 SHA-256 清单</a></p></section>
"""
    return body, [("reports", "报告目录"), ("protocol", "方案与实验范围"), ("evidence", "检查点与证据")]


class Document(HTMLParser):
    """Collect IDs, links and embedded assets for offline integrity checks."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.ids = set()
        self.headings = set()
        self.links = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        anchor = attributes.get("id")
        if anchor:
            if anchor in self.ids:
                raise ValueError(f"Duplicate anchor {anchor} in {self.path.name}")
            self.ids.add(anchor)
            if tag == "h2":
                self.headings.add(anchor)
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append((tag, attributes[key]))
        if tag == "img" and not attributes.get("alt"):
            raise ValueError(f"Missing image alt text in {self.path.name}")


def validate_documents(root, contents):
    documents = {path.resolve(): Document(path) for path in root.glob("*.html")}
    for path, page_info in contents.items():
        doc = documents[(root / path).resolve()]
        expected = {anchor for anchor, _ in page_info}
        if not expected <= doc.headings or not expected <= {
            href[1:] for tag, href in doc.links if tag == "a" and href.startswith("#")
        }:
            raise ValueError(f"Missing heading or table of contents entry in {path}")
        for tag, link in doc.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                if tag == "a" and parsed.scheme == "https":
                    continue
                raise ValueError(f"Non-local asset in {path}: {link}")
            target = (doc.path.parent / unquote(parsed.path)).resolve() if parsed.path else doc.path.resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError(f"Missing local target in {path}: {link}")
            if parsed.fragment and (target not in documents or unquote(parsed.fragment) not in documents[target].ids):
                raise ValueError(f"Missing link target in {path}: {link}")


def pack(root, archive):
    members = [(path, str(Path(NAME) / path.relative_to(root))) for path in sorted(root.rglob("*"))
               if path.is_file()]
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as package:
        for source, target in members:
            package.write(source, target)
    with ZipFile(archive) as package:
        if package.testzip() is not None or package.namelist() != [target for _, target in members]:
            raise ValueError("Offline report ZIP failed verification")
    return len(members)


def build(workspace, destination):
    from . import algorithm, training, validation

    lineage = audit(workspace)
    with tempfile.TemporaryDirectory(prefix="industrial-2018-report-") as work:
        root = Path(work) / NAME
        (root / "figures").mkdir(parents=True)
        (root / "data").mkdir()
        shutil.copyfile(Path(__file__).with_name("report.css"), root / "report.css")
        (root / "data/provenance.json").write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        chapters = {
            "index.html": ("工业品推荐项目全流程报告",
                           "数据、训练、评估与 2K 联合推理的归档复核。", overview(lineage)),
            "algorithm.html": ("算法文档",
                               "从商品语义编码到 UniSRec 检索和 MiniOneRec 打分。", algorithm.render(workspace, root)),
            "training.html": ("可视化训练报告",
                              "原始训练曲线、实际训练预算、历史对照与检查点来源。", training.render(workspace, root)),
            "validation.html": ("可视化验证报告与结果分析",
                                "固定数据上的 2K 质量、冷暖分组、时延与吞吐。", validation.render(workspace, root)),
        }
        for filename, (title, description, (body, sections)) in chapters.items():
            (root / filename).write_text(page(filename, title, description, body, sections), encoding="utf-8")
        validate_documents(root, {name: item[2][1] for name, item in chapters.items()})

        archive = Path(work) / f"{NAME}.zip"
        count = pack(root, archive)
        publish(archive, destination)
        checksum = digest(archive)
        if digest(destination) != checksum:
            raise OSError(f"Final ZIP readback differs: {destination}")
        return {"path": str(destination), "sha256": checksum, "files": count,
                "contents": {name: len(item[2][1]) for name, item in chapters.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build(args.workspace.resolve(), args.destination.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
