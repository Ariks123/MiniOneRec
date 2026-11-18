"""Build the latest Industrial RQ-VAE reports and verify the offline bundle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from hybrid.rho_bundle import digest, publish
from project_report.__main__ import validate_documents


NAME = "Industrial-2018-RQVAE-Speculative-Recommendation-Report"
SASREC_NAME = "Industrial-2018-RQVAE-Speculative-Recommendation-SASRec-Report"
LATEST = Path("weights/checkpoints/MiniOneRec/Industrial_and_Scientific_RQVAE_2K_20261118")
PAGES = {
    "index.html": "报告总览",
    "algorithm.html": "算法报告",
    "training.html": "训练报告",
    "validation.html": "验证报告",
}
DATASET = "Amazon Reviews 2018 · Industrial and Scientific"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matching_artifacts(run):
    for name in ("unisrec-stage.status", "inference/stage.status", "tpot-stage.status"):
        if (run / name).read_text().strip() != "completed":
            raise ValueError(f"Report input is unfinished: {name}")
    data = load(run / "data/preparation_manifest.json")
    rq = load(run / "rqvae/training_summary.json")
    sid = load(run / "sid_export_summary.json")
    sft = load(run / "epochs3/sft/training_summary.json")
    rl = load(run / "rl/training_summary.json")
    uni = load(run / "unisrec/status.json")
    export = load(run / "unisrec/export/metadata.json")
    assets = load(run / "inference/asset_manifest.json")
    quality = load(run / "inference/report/quality_comparison.json")
    records = load(run / "inference/report/comparison_status.json")
    post = load(run / "posttraining_status.json")
    if (sft["status"], rl["status"], uni["stage"], post["status"], rq["status"]) != ("completed",) * 5:
        raise ValueError("One or more training stages did not complete")
    if (data["original_sid"], data["items"], data["unique_sids"], data["complete_sid_tokens"]) != (False, 3686, 3686, 3):
        raise ValueError("Latest report requires the regenerated unique three-token product catalog")
    if (data["rows"] != {"train": 36259, "valid": 4532, "test": 4533}
            or data["rows"] != assets["rows"] or data["rows"] != sid["rows"]
            or (quality["test_rows"], quality["cold_test_rows"]) != (4533, 147)):
        raise ValueError("Row counts differ across the archived training and test stages")
    if not (data["expected_sid_sha256"] == sid["sid_sha256"] == uni["sid_sha256"]
            == rl["parent_sft"]["provenance"]["sid_index_sha256"] == assets["files"]["sid.json"]["sha256"]):
        raise ValueError("Experiment stages used different SID catalogs")
    if not (data["rqvae_model_sha256"] == rq["model_sha256"] == sid["rqvae_model_sha256"]
            and data["selected_sid_epoch"] == rq["best_epoch"] == sid["selected_epoch"]):
        raise ValueError("The report SID was not exported from the selected quantizer")
    if not (uni["checkpoint_sha256"] == export["checkpoint_sha256"] == assets["unisrec_checkpoint_sha256"]
            and uni["best_epoch"] == export["checkpoint_epoch_zero_based"] + 1):
        raise ValueError("The report retrieval index was not exported from the selected UniSRec checkpoint")
    if (sft["best_checkpoint"] != rl["parent_sft"]["selected_checkpoint"]
            or Path(rl["best_checkpoint"]).name != assets["rl_selected_checkpoint"]
            or records["status"] != "passed"):
        raise ValueError("SFT, GRPO, and validation origins do not match")
    report = Path("inference/report")
    expected = {
        "RQ-VAE 选中权重": (Path("rqvae/best_val_model.pth"), rq["model_sha256"]),
        "商品 SID 目录": (Path("catalog/Industrial_and_Scientific.index.json"), sid["sid_sha256"]),
        "SFT 归档权重": (Path("epochs3/sft/final_checkpoint/model.safetensors"), rl["parent_sft"]["model_sha256"]),
        "GRPO 归档权重": (Path("rl/final_checkpoint/model.safetensors"), assets["files"]["rl/model.safetensors"]["sha256"]),
        "UniSRec 选中权重": (Path("unisrec") / uni["checkpoint"], uni["checkpoint_sha256"]),
        "UniSRec 商品向量": (Path("unisrec/export/item_vectors.npy"), export["checksums"]["item_vectors.npy"]),
        "质量对照数据": (report / "quality_comparison.json", records["quality_sha256"]),
        "性能对照数据": (report / "performance_comparison.json", records["performance_sha256"]),
    }
    for split, label in (("train", "训练数据"), ("valid", "验证数据"), ("test", "测试数据")):
        name = f"data/{split}.csv"
        checksum = data["outputs"][f"{split}.csv"]["sha256"]
        if checksum != assets["files"][name]["sha256"]:
            raise ValueError(f"Staged input checksum differs: {label}")
        expected[label] = (Path(name), checksum)
    tpot = load(run / report / "tpot_rqvae.json")
    protocol = run / report / "benchmark_protocol.json"
    if not (tpot["status"] == "passed" and tpot["variant"] == "rqvae"
            and tpot["model_sha256"] == assets["files"]["rl/model.safetensors"]["sha256"]
            and tpot["sid_sha256"] == sid["sid_sha256"]
            and tpot["benchmark_protocol_sha256"] == digest(protocol)
            and tpot["sample_row_ids"] == load(protocol)["sample_row_ids"]
            and tpot["summary"]["samples"] == 768):
        raise ValueError("TPOT measured a different model, SID catalog, or benchmark protocol")
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda pair: digest(run / pair[0]), expected.values()))
    checksums = {}
    for (label, (relative, checksum)), measured in zip(expected.items(), actual, strict=True):
        if measured != checksum:
            raise ValueError(f"Archived data differs from its fingerprint: {relative}")
        checksums[label] = measured
    return {
        "数据集": DATASET,
        "商品数": data["items"],
        "唯一 SID 数": data["unique_sids"],
        "各切分样本数": data["rows"],
        "冷目标请求数": quality["cold_test_rows"],
        "RQ-VAE 选中轮次": rq["best_epoch"],
        "SFT 选中步数": rl["parent_sft"]["selected_step"],
        "GRPO 选中步数": int(assets["rl_selected_checkpoint"].rsplit("-", 1)[1]),
        "UniSRec 选中轮次": uni["best_epoch"],
        "已核验 SHA-256": checksums,
    }


def overview(run, evidence):
    quality = load(run / "inference/report/quality_comparison.json")
    core = {r["k"]: r["new"] for r in quality["primary"]
            if r["stratum"] == "overall" and r["method"] == "unisrec_2k"}
    cold = {r["k"]: r["new"] for r in quality["primary"]
            if r["stratum"] == "cold" and r["method"] == "unisrec_2k"}
    order = [("reports", "三份报告"), ("experiment", "实验与最终模型"), ("summary", "主要验证结果")]
    rows = evidence["各切分样本数"]
    body = f"""
<section><h2 id="reports">三份报告</h2>
<div class="report-list">
  <article><h3><a href="algorithm.html">算法报告</a></h3><p>模型架构、关键公式以及从草拟、验证到商品排序的联合投机推理流程。</p></article>
  <article><h3><a href="training.html">训练报告</a></h3><p>最新 RQ-VAE、SFT、GRPO 和 UniSRec 的损失、曲线、选点与解释。</p></article>
  <article><h3><a href="validation.html">验证报告</a></h3><p>SID 产物对比、各 K 的精确商品排名、端到端延迟、吞吐和 token 计时。</p></article>
</div></section>
<section><h2 id="experiment">实验与最终模型</h2>
<p>实验覆盖 {evidence['商品数']:,} 件商品。训练、验证、测试分别保留 {rows['train']:,}、{rows['valid']:,}、
{rows['test']:,} 条交互请求。RQ-VAE 导出同等数量的唯一三位 SID。已核验从商品索引到训练选点、检索向量和推理实测的产物关联。</p>
<p>用于联合投机推理的生成权重来自线性三轮 SFT 第 {evidence['SFT 选中步数']:,} 步和
GRPO 第 {evidence['GRPO 选中步数']:,} 步。UniSRec 采用第 {evidence['UniSRec 选中轮次']} 轮。</p>
<p><a href="data/experiment-integrity.json">查看数据、权重与测试产物的校验摘要</a></p></section>
<section><h2 id="summary">主要验证结果</h2>
<p>联合投机推理的商品 HR@1 为 {core[1]['item_hr']:.3%}。K=50 时，HR 为 {core[50]['item_hr']:.3%}，
NDCG 为 {core[50]['item_ndcg']:.3%}，{evidence['冷目标请求数']} 条冷目标请求命中 {cold[50]['item_hits']} 条。</p>
<p>各页可直接离线打开。图表旁提供相应的训练或验证数据链接。</p></section>"""
    return body, order


def page(current, title, subtitle, body, anchors):
    links = "".join(
        f'<a href="{filename}" aria-current="page">{label}</a>' if filename == current
        else f'<a href="{filename}">{label}</a>'
        for filename, label in PAGES.items()
    )
    toc = "".join(f'<li><a href="#{escape(anchor, quote=True)}">{escape(label)}</a></li>'
                  for anchor, label in anchors)
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)} · {DATASET}</title><link rel="stylesheet" href="style.css">'
        '</head><body><nav class="top" aria-label="报告导航">'
        f'<div class="frame">{links}</div></nav>'
        '<header class="hero" id="top"><div class="frame">'
        f'<p class="dataset">{DATASET} · RQ-VAE</p>'
        f'<h1>{escape(title)}</h1><p class="summary">{escape(subtitle)}</p></div></header>'
        '<main class="frame"><nav class="toc" aria-label="章节目录">'
        f'<p class="toc-title">目录</p><ol>{toc}</ol></nav>{body}'
        '<footer><a href="#top">返回页首</a> · <a href="index.html">报告总览</a></footer>'
        '</main></body></html>'
    )


def with_sasrec(run, baseline, output, reports):
    from sasrec_baseline.report import render

    comparison = load(baseline / "comparison.json")
    frozen = run / "inference/report"
    sources = comparison["source_sha256"]
    if (comparison["status"] != "passed"
            or comparison["test_rows"] != 4533
            or comparison["hybrid_configs"] != load(frozen / "selection.json")["winners"]
            or sources["hybrid_selection"] != digest(frozen / "selection.json")
            or sources["hybrid_test_predictions"] != digest(frozen / "predictions/validation_winners.jsonl.gz")
            or sources["sasrec_selection"] != digest(baseline / "selection.json")
            or sources["sasrec_test_predictions"] != digest(baseline / "test_predictions.jsonl")):
        raise ValueError("SASRec comparison is from a different experiment or selection")
    additional, toc = render(run, baseline, output)
    title, subtitle, (body, sections) = reports["validation.html"]
    reports["validation.html"] = (title, subtitle, (body + additional, [*sections, *toc]))
    title, subtitle, (body, sections) = reports["index.html"]
    at_ten = next(row for row in comparison["comparisons"] if row["k"] == 10 and row["stratum"] == "overall")
    scores = at_ten["sasrec"]["item_ndcg"], at_ten["hybrid"]["item_ndcg"]
    baseline_summary = (
        '<section><h2 id="sasrec-baseline">独立 SASRec 基线</h2>'
        f'<p>同一批测试请求上的商品 NDCG@10，SASRec 为 {scores[0]:.3%}，联合投机推理为 {scores[1]:.3%}。'
        f'<a href="validation.html#{toc[0][0]}">查看训练选点、排名差值、推理时延和吞吐</a></p></section>'
    )
    reports["index.html"] = (title, subtitle, (body + baseline_summary, [*sections, ("sasrec-baseline", "独立 SASRec 基线")]))


def build(workspace, run, destination, sasrec_run=None):
    from . import algorithm, training, validation

    evidence = matching_artifacts(run)
    name = SASREC_NAME if sasrec_run else NAME
    with tempfile.TemporaryDirectory(prefix="industrial-rqvae-standalone-") as folder:
        output = Path(folder) / name
        (output / "data").mkdir(parents=True)
        (output / "figures").mkdir()
        shutil.copyfile(Path(__file__).with_name("style.css"), output / "style.css")
        shutil.copyfile(Path(__file__).parents[1] / "project_report/report.css", output / "base.css")
        (output / "data/experiment-integrity.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        chapters = {
            "index.html": ("工业品 RQ-VAE 全流程报告",
                           "从商品语义编码、模型训练到联合投机推理的最新实验。", overview(run, evidence)),
            "algorithm.html": ("算法报告", "模型结构、训练目标与联合投机推理流程。",
                               algorithm.render(workspace, run, output)),
            "training.html": ("可视化训练报告", "四个训练阶段的真实曲线、损失函数与检查点选择。",
                              training.render(workspace, run, output)),
            "validation.html": ("可视化验证报告", "商品语义编码对比、排名质量与推理性能。",
                                validation.render(workspace, run, output)),
        }
        if sasrec_run:
            with_sasrec(run, sasrec_run, output, chapters)
        for path, (title, subtitle, (body, anchors)) in chapters.items():
            if not anchors or len({key for key, _ in anchors}) != len(anchors):
                raise ValueError(f"Invalid chapter list: {path}")
            (output / path).write_text(page(path, title, subtitle, body, anchors), encoding="utf-8")
        validate_documents(output, {name: item[2][1] for name, item in chapters.items()})
        archive = Path(folder) / (name + ".zip")
        files = [path for path in sorted(output.rglob("*")) if path.is_file()]
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(output.parent))
        with ZipFile(archive) as bundle:
            if bundle.testzip() is not None or len(bundle.namelist()) != len(files):
                raise ValueError("Archive did not pass CRC and member-count checks")
        publish(archive, destination)
        if digest(destination) != digest(archive):
            raise OSError("Published checksum changed during copy")
        return {"archive": str(destination), "sha256": digest(destination), "files": len(files),
                "sections": {key: len(value[2][1]) for key, value in chapters.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--sasrec-run", type=Path, help="include the completed independent SASRec baseline")
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/industrial-rqvae-standalone-mpl")
    workspace = args.workspace.resolve()
    print(json.dumps(build(workspace, (args.run_dir or workspace / LATEST).resolve(), args.destination.resolve(),
                            args.sasrec_run.resolve() if args.sasrec_run else None), ensure_ascii=False))


if __name__ == "__main__":
    main()
