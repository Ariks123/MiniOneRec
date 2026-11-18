#!/usr/bin/env python3
"""Verify the Amazon Reviews 2018 Video Games Kaggle downloads (stdlib only).

Usage:
    python data/Amazon18/verify_download.py
    python data/Amazon18/verify_download.py --raw-dir PATH

Accepts gzip or plain JSONL, preferring .json.gz when both files exist. Reads to
EOF, verifying JSON records and gzip CRC/trailer for compressed inputs. SHA-256
values are local fingerprints, not checksums supplied by the dataset publisher.
Source URLs document the mirrors selected for this project; byte identity with
the upstream files is not independently verified. No network access, extraction,
or modification of source data is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile


REVIEW_FILE = "Video_Games_5.json.gz"
META_FILE = "meta_Video_Games.json.gz"
SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/"
SOURCES = {
    "Video_Games_5.json": {
        "source_url": "https://www.kaggle.com/idrimadrid/projet-text-mining-amazon-reviews/output",
        "source_type": "Kaggle notebook output",
        "upstream_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
    },
    "meta_Video_Games.json": {
        "source_url": "https://www.kaggle.com/datasets/gabrielfreddi/amazon-reviews-de-vdeo-games/versions/2",
        "source_type": "Kaggle dataset",
        "source_version": 2,
        "upstream_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz",
    },
}


def resolve_inputs(raw_dir: Path) -> tuple[Path, Path]:
    paths = []
    for compressed_name in (REVIEW_FILE, META_FILE):
        candidates = (raw_dir / compressed_name, raw_dir / compressed_name[:-3])
        for candidate in candidates:
            if candidate.is_file():
                paths.append(candidate)
                break
        else:
            raise FileNotFoundError(f"Missing input: expected {candidates[0]} or {candidates[1]}")
    return paths[0], paths[1]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        **SOURCES[path.name[:-3] if path.suffix == ".gz" else path.name],
        "bytes": path.stat().st_size,
        "compression": "gzip" if path.suffix == ".gz" else "none",
        "gzip_crc_and_trailer": "passed" if path.suffix == ".gz" else "not_applicable",
        "sha256": digest.hexdigest(),
    }


def records(path: Path):
    # Reading gzip inputs to EOF forces CRC and uncompressed-size validation.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            yield line_number, value


def identifier(record: dict, key: str, filename: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{filename}:{line_number}: missing or invalid {key}")
    return value


def signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def verify(raw_dir: Path, expected_reviews: int = 497577) -> dict:
    review_path, meta_path = resolve_inputs(raw_dir)
    paths = [review_path, meta_path]
    before = {path: signature(path) for path in paths}

    users: set[str] = set()
    item_review_counts: Counter[str] = Counter()
    review_count = 0
    for line_number, record in records(review_path):
        users.add(identifier(record, "reviewerID", review_path.name, line_number))
        item_review_counts[identifier(record, "asin", review_path.name, line_number)] += 1
        review_count += 1
    if expected_reviews and review_count != expected_reviews:
        raise ValueError(
            f"Expected {expected_reviews:,} Amazon 2018 5-core reviews; read {review_count:,}. "
            "Check that the correct file has finished downloading."
        )

    metadata_items: set[str] = set()
    metadata_count = 0
    for line_number, record in records(meta_path):
        metadata_items.add(identifier(record, "asin", meta_path.name, line_number))
        metadata_count += 1

    files = [fingerprint(path) for path in paths]
    for path in paths:
        if signature(path) != before[path]:
            raise RuntimeError(f"{path.name} changed during verification; retry after download finishes")

    review_items = set(item_review_counts)
    matched_items = review_items & metadata_items
    matched_review_count = sum(item_review_counts[item] for item in matched_items)
    missing_items = sorted(review_items - metadata_items)
    return {
        "dataset": "Amazon Review Data (2018) - Video Games (5-core)",
        "source_page": SOURCE_PAGE,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_directory": str(raw_dir.resolve()),
        "validation": {
            "gzip_crc_and_trailer": {
                entry["filename"]: entry["gzip_crc_and_trailer"] for entry in files
            },
            "json_objects_and_required_ids": "passed",
            "expected_review_count": expected_reviews or None,
            "sha256_note": "Locally computed fingerprints; no publisher checksum comparison performed",
        },
        "files": files,
        "reviews": {
            "records": review_count,
            "unique_users": len(users),
            "unique_items": len(review_items),
        },
        "metadata": {
            "records": metadata_count,
            "unique_items": len(metadata_items),
            "duplicate_asin_records": metadata_count - len(metadata_items),
        },
        "coverage": {
            "review_items_with_metadata": len(matched_items),
            "review_items_without_metadata": len(missing_items),
            "review_item_coverage_fraction": len(matched_items) / len(review_items) if review_items else None,
            "review_records_with_metadata": matched_review_count,
            "review_record_coverage_fraction": matched_review_count / review_count if review_count else None,
            "missing_item_examples": missing_items[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parent / "raw")
    parser.add_argument("--output", type=Path, help="Default: RAW_DIR/download_manifest.json")
    parser.add_argument("--expected-reviews", type=int, default=497577,
                        help="Expected review count; 0 disables this check (for fixtures only)")
    args = parser.parse_args()
    if args.expected_reviews < 0:
        parser.error("--expected-reviews must be nonnegative")
    input_paths = resolve_inputs(args.raw_dir)
    output = args.output or args.raw_dir / "download_manifest.json"
    protected_paths = {path.resolve() for path in input_paths}
    protected_paths.update(
        (args.raw_dir / (name + suffix)).resolve()
        for name in SOURCES for suffix in ("", ".gz")
    )
    if output.resolve() in protected_paths:
        parser.error("--output cannot overwrite a source data file")
    report = verify(args.raw_dir, args.expected_reviews)
    # Publish the report only after every input and record has passed checks.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as dest:
            temporary_path = Path(dest.name)
            json.dump(report, dest, ensure_ascii=False, indent=2)
            dest.write("\n")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(json.dumps({"manifest": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
