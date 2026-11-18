#!/usr/bin/env python3
"""Publish selected Amazon 2018 checkpoints without copying or changing originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("ZIWEIZHAO_WORKDIR", "/mnt/ossdata/ziweizhao"))
DEPS = WORKSPACE / ".tools/publish-deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))


def hashes(path):
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id")
    parser.add_argument("--staging", type=Path, default=Path("/tmp/amazon2018-models-ziweizhao"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    mapping = json.loads((ROOT / "weight_sources.json").read_text())
    files = {}
    for entry in mapping:
        source = Path(entry["source"])
        if not source.exists():
            raise SystemExit(f"Missing source: {source}")
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix() if source.is_dir() else ""
            target = str(PurePosixPath(entry["target"]) / relative)
            if PurePosixPath(target).is_absolute() or ".." in PurePosixPath(target).parts:
                raise SystemExit(f"Unsafe target: {target}")
            if target in files and files[target] != path:
                raise SystemExit(f"Conflicting target: {target}")
            files[target] = path

    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    cache_file = runtime / "weight_hash_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    manifest = []
    for number, (target, source) in enumerate(sorted(files.items()), 1):
        stat = source.stat()
        key = str(source)
        old = cache.get(key, {})
        if old.get("size") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            sha256, blob_sha = old["sha256"], old["git_blob_sha1"]
        else:
            sha256, blob_sha = hashes(source)
            if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
                raise SystemExit(f"Source changed while hashing: {source}")
            cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256, "git_blob_sha1": blob_sha}
            cache_file.write_text(json.dumps(cache, indent=2) + "\n")
        destination = args.staging / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise SystemExit(f"Staging link points elsewhere: {destination}")
        elif destination.exists():
            raise SystemExit(f"Staging destination already exists: {destination}")
        else:
            destination.symlink_to(source.resolve())
        manifest.append({"path": target, "size": stat.st_size, "sha256": sha256, "git_blob_sha1": blob_sha})
        print(f"Prepared {number}/{len(files)}: {target}", flush=True)

    expected = {item["path"] for item in manifest} | {"README.md", "weight_manifest.json", "weight_sources.json"}
    unexpected = [p.relative_to(args.staging).as_posix() for p in args.staging.rglob("*")
                  if p.is_file() and p.relative_to(args.staging).as_posix() not in expected and ".cache" not in p.relative_to(args.staging).parts]
    if unexpected:
        raise SystemExit(f"Unexpected staging files: {unexpected}")
    document = {"created_at": datetime.now(timezone.utc).isoformat(), "file_count": len(manifest),
                "bytes": sum(item["size"] for item in manifest), "files": manifest}
    (args.staging / "weight_manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "weight_manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    for name, source in [("README.md", ROOT / "publication/model_card.md"), ("weight_sources.json", ROOT / "weight_sources.json")]:
        shutil.copy2(source, args.staging / name)
    print(f"Ready: {len(manifest)} files, {document['bytes']:,} bytes", flush=True)
    if args.prepare_only:
        return
    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        raise SystemExit("NO_HF_WRITE_CREDENTIAL: run hf auth login or configure HF_TOKEN locally.")
    api = HfApi(token=token)
    identity = api.whoami()
    repo_id = args.repo_id or f"{identity['name']}/Amazon2018-MiniOneRec-Checkpoints"
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Uploading to https://huggingface.co/{repo_id}", flush=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=args.staging,
                      commit_message="Archive Amazon Reviews 2018 trained checkpoints and SID assets")
    remote = {item.rfilename: item for item in api.model_info(repo_id, files_metadata=True).siblings}
    errors = []
    for item in manifest:
        found = remote.get(item["path"])
        if found is None or found.size != item["size"]:
            errors.append(item["path"] + ": missing or size mismatch")
        elif found.lfs and found.lfs.sha256 != item["sha256"]:
            errors.append(item["path"] + ": SHA-256 mismatch")
        elif not found.lfs and found.blob_id != item["git_blob_sha1"]:
            errors.append(item["path"] + ": Git blob SHA-1 mismatch")
    if errors:
        raise SystemExit("Remote verification failed: " + "; ".join(errors))
    receipt = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
               "revision": api.model_info(repo_id).sha, "verified_files": len(manifest),
               "bytes": document["bytes"], "verified_at": datetime.now(timezone.utc).isoformat()}
    (runtime / "huggingface_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
