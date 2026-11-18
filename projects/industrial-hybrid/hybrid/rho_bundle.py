"""Publish the offline Industrial comparison and keep the previously published report accessible."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile


NAME = "Industrial-Hybrid-Validation-Report"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key not in ("src", "href") or not value:
                continue
            url = urlsplit(value)
            if not url.path or url.scheme or url.netloc:
                continue
            self.paths.add(unquote(url.path))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def unpack_previous(archive, directory):
    with ZipFile(archive) as source:
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.parts[:1] != (NAME,) or ".." in path.parts:
                raise ValueError(f"Unexpected path in previous report: {path}")
            if info.is_dir():
                continue
            dest = directory.joinpath(*path.parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as inp, dest.open("wb") as out:
                shutil.copyfileobj(inp, out)


def linked_files(report):
    report = report.resolve()
    pending = [report / "index.html"]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_relative_to(report) or not path.is_file():
            raise ValueError(f"Broken or external local report path: {path}")
        found.add(path)
        if path.suffix == ".html":
            refs = Links()
            refs.feed(path.read_text(encoding="utf-8"))
            pending.extend(path.parent / link for link in refs.paths)
    return sorted(found)


def publish(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".report-", suffix=".partial",
                                     delete=False) as out:
        temporary = Path(out.name)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != digest(source):
            raise OSError(f"Copy verification failed for {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def bundle(root, previous_zip, destination):
    report = (root / "report").resolve()
    status = json.loads((report / "report_status.json").read_text())
    if (not status["completed"] or status["quality"] != "passed"
            or status["performance"] != "passed"
            or status["configurations_measured"] != status["configurations_planned"]):
        raise ValueError("The comparison report must include all quality and live measurements before publication")
    previous = report / "previous"
    if previous_zip.resolve() == (root / (NAME + ".zip")).resolve():
        raise ValueError("Previous and current report archives must have separate paths")
    unpack_previous(previous_zip, previous)
    files = linked_files(report)
    if previous / "index.html" not in files:
        raise ValueError("Comparison report must link the previous report")
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / (NAME + ".zip")
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=5) as packed:
            for path in files:
                packed.write(path, str(PurePosixPath(NAME) / path.relative_to(report)))
        with ZipFile(archive) as packed:
            if packed.testzip() is not None or len(packed.infolist()) != len(files):
                raise ValueError("Report archive failed integrity checks")
        local = root / archive.name
        publish(archive, local)
        if destination is not None:
            publish(archive, destination)
        print(f"{len(files)} linked files, {archive.stat().st_size} bytes, SHA-256 {digest(archive)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous-zip", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    bundle(args.run_dir, args.previous_zip, args.destination)
