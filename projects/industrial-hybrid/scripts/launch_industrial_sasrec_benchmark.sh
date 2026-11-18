#!/usr/bin/env bash
set -euo pipefail

run="$(realpath "${1:?usage: launch_industrial_sasrec_benchmark.sh RUN_DIR}")"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd -- "$repo/../.." && pwd)"
archive="$workspace/weights/checkpoints/MiniOneRec/Industrial_and_Scientific_RQVAE_2K_20261118"
container=industrial-rqvae-2k-infer-20261118
started=false

if [[ "$run" == "$archive" ]]; then
    printf '%s\n' "Choose a separate output directory for the SASRec benchmark." >&2
    exit 2
fi
for artifact in selection.json model.ts test_predictions.jsonl; do
    test -s "$run/$artifact"
done
exec 9>"/tmp/$container.lock"
flock -n 9
cleanup() {
    if [[ "$started" == true ]]; then
        docker stop --timeout 10 "$container"
    fi
}
trap cleanup EXIT
if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
    docker start "$container"
    started=true
fi
active="$(docker exec "$container" nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$active" ]]; then
    printf '%s\n' "All four GPUs must be idle before measuring the SASRec benchmark." >&2
    exit 1
fi

staging="$(docker exec "$container" mktemp -d /run/sasrec-benchmark.XXXXXX)"
docker exec "$container" mkdir -p /work/sasrec_baseline "$staging/input" "$staging/results"
for module in __init__.py benchmark.py data.py; do
    docker cp "$repo/sasrec_baseline/$module" "$container:/work/sasrec_baseline/$module"
done
for artifact in selection.json model.ts test_predictions.jsonl; do
    docker cp "$run/$artifact" "$container:$staging/input/$artifact"
done
docker cp "$archive/data/test.csv" "$container:$staging/test.csv"
docker cp "$archive/inference/asset_manifest.json" "$container:$staging/asset_manifest.json"
docker cp "$archive/inference/report/benchmark_protocol.json" "$container:$staging/archived_protocol.json"
docker exec -w /work -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 "$container" \
    python -m torch.distributed.run --standalone --nproc-per-node=4 -m sasrec_baseline.benchmark \
    --run-dir "$staging/input" --archive-protocol "$staging/archived_protocol.json" \
    --asset-manifest "$staging/asset_manifest.json" --test-csv "$staging/test.csv" --output-dir "$staging/results"

cd "$repo"
python3 -B - "$run" "$container" "$staging/results" <<'PY'
from pathlib import Path
import json
import subprocess
import sys
import tempfile

from hybrid.rho_bundle import publish

run = Path(sys.argv[1])
source = f"{sys.argv[2]}:{sys.argv[3]}/."
with tempfile.TemporaryDirectory(prefix="sasrec-benchmark-publish-") as work:
    staging = Path(work)
    subprocess.run(["docker", "cp", source, work], check=True)
    metrics = json.loads((staging / "benchmarks.json").read_text(encoding="utf-8"))
    if len(metrics) != 5 or any(entry["latency_samples"] != 768 for entry in metrics):
        raise ValueError("The newly measured SASRec results are incomplete")
    for name in ("benchmark_latency.csv", "benchmark_protocol.json", "benchmarks.json"):
        publish(staging / name, run / name)
PY
