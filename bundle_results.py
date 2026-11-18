"""Bundle the selected 2018 results without modifying their original files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
import shutil
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
STAGING = HERE
LIMIT = 95_000_000


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    records = json.loads((HERE / "sources_results.json").read_text())
    skipped = json.loads((HERE / "results_skipped_large.json").read_text())
    records += [{k: v for k, v in row.items() if k != "reason"} for row in skipped]
    records = list({row["target"]: row for row in records}.values())
    groups = defaultdict(list)
    for row in records:
        target = Path(row["target"])
        parts = target.parts
        if parts[2] in {"MiniOneRec", "UniSRec"}:
            family, run = parts[2:4]
            row["member"] = "/".join(parts[4:])
            name = ("UniSRec-" if family == "UniSRec" else "") + run
            row["original_result_root"] = "/".join(parts[:4])
        elif parts[2:4] == ("rqvae", "Video_Games"):
            run = parts[4]
            row["member"] = "/".join(parts[5:])
            name = "Video_Games-RQVAE-" + run
            row["original_result_root"] = "/".join(parts[:5])
        else:
            raise ValueError(f"Unrecognized result target: {target}")
        row["sha256"] = digest(Path(row["source"]))
        groups[name].append(row)

    bundles = STAGING / "results/bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    previous_manifest = json.loads((bundles / "manifest.json").read_text()) if (bundles / "manifest.json").exists() else []
    temporary = Path(tempfile.mkdtemp(prefix="amazon2018-results-"))
    manifest = []

    def write_bundle(name: str, rows: list[dict], suffix: str = "") -> None:
        output = bundles / f"{name}{suffix}.zip"
        local_output = temporary / output.name
        with zipfile.ZipFile(local_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for row in rows:
                archive.write(row["source"], arcname=row["member"])
        if local_output.stat().st_size >= LIMIT:
            local_output.unlink()
            if len(rows) <= 1:
                raise RuntimeError(f"Single compressed result exceeds archive size limit: {name}")
            halfway = sum(row["size_bytes"] for row in rows) / 2
            running = 0
            midpoint = 1
            for index, row in enumerate(rows[:-1], start=1):
                running += row["size_bytes"]
                midpoint = index
                if running >= halfway:
                    break
            write_bundle(name, rows[:midpoint], suffix + "-part1")
            write_bundle(name, rows[midpoint:], suffix + "-part2")
            return
        with zipfile.ZipFile(local_output) as archive:
            problem = archive.testzip()
            if problem is not None:
                raise RuntimeError(f"CRC failed: {output}: {problem}")
            if len(archive.infolist()) != len(rows):
                raise RuntimeError(f"Archive count mismatch: {output}")
            total = sum(member.file_size for member in archive.infolist())
            if total != sum(row["size_bytes"] for row in rows):
                raise RuntimeError(f"Archive byte-count mismatch: {output}")
        relative = str(output.relative_to(STAGING))
        local_sha256 = digest(local_output)
        shutil.copy2(local_output, output)
        if digest(output) != local_sha256:
            raise RuntimeError(f"Published archive hash mismatch: {output}")
        local_output.unlink()
        for row in rows:
            row["bundle"] = relative
        manifest.append({
            "path": relative,
            "sha256": local_sha256,
            "compressed_bytes": output.stat().st_size,
            "uncompressed_bytes": total,
            "file_count": len(rows),
            "original_result_roots": sorted({row["original_result_root"] for row in rows}),
            "crc_verified": True,
        })
        print(json.dumps(manifest[-1]), flush=True)

    for name, rows in sorted(groups.items()):
        write_bundle(name, rows)

    produced = {item["path"] for item in manifest}
    for item in previous_manifest:
        old_path = STAGING / item["path"]
        if item["path"] not in produced and old_path.parent == bundles and old_path.is_file():
            old_path.unlink()

    (HERE / "sources_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    public_records = [{**row, "source": str(Path(row["source"]).relative_to(HERE.parent.parent))}
                      for row in records]
    (STAGING / "results/sources_results.json").write_text(json.dumps(public_records, ensure_ascii=False, indent=2) + "\n")
    (bundles / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    readme = [
        "# 2018 训练与评估原始结果", "",
        "每个 ZIP 对应一个正式实验目录，保留训练配置、状态、曲线数据、评估指标和逐行预测。",
        "所有 ZIP 使用 DEFLATE level 6，单包小于 95 MB，并已校验 CRC、文件数和未压缩总量。",
        "`sources_results.json` 记录每个原文件的工作区相对来源、SHA256、ZIP 路径与包内成员路径；",
        "`bundles/manifest.json` 记录各包的 SHA256 与大小。解压时按下表目录恢复。",
        "权重、优化器、中间 checkpoint、smoke、失败 partial、运行环境副本及已取消的 2023 实验不在结果包中。",
        "", "| ZIP | 原结果目录 | 文件数 | 压缩后字节 |", "| --- | --- | ---: | ---: |",
    ]
    for item in manifest:
        filename = Path(item["path"]).name
        readme.append(f"| [{filename}](bundles/{filename}) | {'; '.join(item['original_result_roots'])} | {item['file_count']} | {item['compressed_bytes']} |")
    (STAGING / "results/README.md").write_text("\n".join(readme) + "\n")
    # Only delete copies in the publication staging area after every bundle passed validation.
    if (STAGING / "results/checkpoints").exists():
        shutil.rmtree(STAGING / "results/checkpoints")
    summary = {
        "file_count": len(records),
        "bundle_count": len(manifest),
        "uncompressed_bytes": sum(row["size_bytes"] for row in records),
        "compressed_bytes": sum(item["compressed_bytes"] for item in manifest),
        "largest_bundle_bytes": max(item["compressed_bytes"] for item in manifest),
        "crc_all_verified": True,
    }
    (HERE / "bundles_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.rmtree(temporary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
