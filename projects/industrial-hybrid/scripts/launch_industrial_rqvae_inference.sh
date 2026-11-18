#!/usr/bin/env bash
# Run the original four-GPU inference protocol once post-training completes.
set -euo pipefail

run="$(realpath "${1:?usage: launch_industrial_rqvae_inference.sh RUN_DIR}")"
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd -- "$project/../.." && pwd)"
legacy_data="$workspace/MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey"
legacy_run="$workspace/weights/checkpoints/MiniOneRec/Industrial_hybrid_beta001_inference_20261118"
timing_run="$workspace/weights/checkpoints/MiniOneRec/Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118"
container=industrial-rqvae-2k-infer-20261118
image=minionerec:h20-original-torch28
assets=/run/rqvae-assets
inference=/run/rqvae-inference
original=/run/rqkmeans
host="$run/inference"
dataset=Industrial_and_Scientific

if [[ "$run" == "$legacy_run" || "$run" == "$timing_run" ]]; then
    printf '%s\n' "Choose an isolated experiment directory" >&2
    exit 2
fi
mkdir -p "$host"
exec 9>"/tmp/$container.lock"
flock -n 9
stage() {
    printf '%s\n' "$1" > "$host/stage.status.tmp"
    mv "$host/stage.status.tmp" "$host/stage.status"
    printf '%s %s\n' "$(date -Iseconds)" "$1"
}
trap 'code=$?; if ((code != 0)); then stage "failed:$code:$current"; fi' EXIT

current=waiting-for-posttraining
stage "$current"
while true; do
    state="$(python3 - "$run/posttraining_status.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(json.loads(path.read_text())["status"] if path.exists() else "waiting")
PY
)"
    case "$state" in
        completed) break ;;
        failed) printf '%s\n' "Post-training failed. Inspect $run/posttraining_status.json" >&2; exit 1 ;;
        *) sleep 10 ;;
    esac
done

if [[ -s "$host/report/comparison_status.json" ]]; then
    current=completed
    stage "$current"
    exit 0
fi

current=staging
stage "$current"
checkpoint="$(python3 - "$run/unisrec" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
entry = json.loads((root/"status.json").read_text())
path = (root / entry["checkpoint"]).resolve()
with path.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
if entry["stage"] != "completed" or not path.is_relative_to(root) or checksum != entry["checkpoint_sha256"]:
    raise ValueError("UniSRec selected checkpoint is missing or changed")
print(path)
PY
)"

if ! docker container inspect "$container" >/dev/null 2>&1; then
    docker create --name "$container" --gpus all --shm-size 16g -w /work --entrypoint sleep "$image" infinity
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
    docker start "$container"
fi
docker exec "$container" mkdir -p /work/hybrid /work/scripts "$assets/data" "$assets/rl" \
    "$assets/unisrec" "$original/valid" "$original/test" "$original/report/predictions" \
    /run/rqkmeans-timing/report /run/rqkmeans-data /run/rqvae-rl-evidence/evaluation "$inference"
for file in data.py npu_support.py LogitProcessor.py; do
    docker cp "$project/$file" "$container:/work/$file"
done
docker cp "$project/scripts/sft_eval_utils.py" "$container:/work/scripts/"
docker cp "$project/hybrid/." "$container:/work/hybrid/"
docker cp "$run/catalog/$dataset.index.json" "$container:$assets/sid.json"
docker cp "$run/data/." "$container:$assets/data/"
docker cp "$run/rl/final_checkpoint/." "$container:$assets/rl/"
docker cp "$run/rl/evaluation/predictions.jsonl" "$container:$assets/reference_test_beam.jsonl"
for name in run_config training_summary; do
    docker cp "$run/rl/$name.json" "$container:/run/rqvae-rl-evidence/$name.json"
done
docker cp "$run/rl/evaluation/metrics.json" "$container:/run/rqvae-rl-evidence/evaluation/"
docker cp "$run/unisrec/export/." "$container:$assets/unisrec/"
docker cp "$checkpoint" "$container:/run/selected-unisrec.pt"
for split in train valid test; do
    docker cp "$legacy_data/$split.csv" "$container:/run/rqkmeans-data/"
done
for name in manifest.json catalog.json reference_test_beam.jsonl; do
    docker cp "$legacy_run/$name" "$container:$original/"
done
for split in valid test; do
    docker cp "$legacy_run/$split/." "$container:$original/$split/"
done
for name in selection.json test.csv batch_audit.json; do
    docker cp "$legacy_run/report/$name" "$container:$original/report/"
done
docker cp "$legacy_run/report/predictions/validation_winners.jsonl.gz" "$container:$original/report/predictions/"
for name in benchmark_protocol.json benchmarks.json; do
    docker cp "$timing_run/report/$name" "$container:/run/rqkmeans-timing/report/"
done

execute() {
    docker exec -w /work -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false \
        -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e MPLCONFIGDIR=/tmp/rqvae-inference-mpl \
        "$container" "$@"
}
current=validating-assets
stage "$current"
# Duplicate the two named paths as links only inside the verifier's isolated
# container. The sealed input hashes therefore refer to the exact staged files.
execute sh -c 'ln -sfn /run/rqvae-assets/rl /run/rqvae-rl-evidence/final_checkpoint
ln -sfn /run/rqvae-assets/reference_test_beam.jsonl /run/rqvae-rl-evidence/evaluation/predictions.jsonl'
execute python -m hybrid.rqvae_assets --assets "$assets" --legacy-data /run/rqkmeans-data \
    --legacy-run "$original" --rl-run /run/rqvae-rl-evidence --unisrec-checkpoint /run/selected-unisrec.pt
docker cp "$container:$assets/rqvae_manifest.json" "$host/asset_manifest.json"

current=waiting-for-gpus
stage "$current"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 5
done

current=quality-generation
stage "$current"
for split in valid test; do
    execute python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.rqvae_quality \
        generate --splits "$split" --assets "$assets" --run-dir "$inference" --legacy-run "$original"
    if [[ "$split" == valid ]]; then
        execute python -m hybrid.rqvae_quality select --assets "$assets" --run-dir "$inference" --legacy-run "$original"
    fi
done
current=paired-quality
stage "$current"
execute python -m hybrid.rqvae_quality quality --assets "$assets" --run-dir "$inference" --legacy-run "$original"
mkdir -p "$host/report"
docker cp "$container:$inference/report/quality_comparison.json" "$host/report/"
docker cp "$container:$inference/report/paired_comparison.json" "$host/report/"
docker cp "$container:$inference/report/quality_status.json" "$host/report/"

current=performance
stage "$current"
execute python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.benchmark \
    --rqvae --assets "$assets" --run-dir "$inference" --mode both
current=batch-audit
stage "$current"
execute python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.batch_audit \
    --rqvae --assets "$assets" --run-dir "$inference"
execute python -m hybrid.rqvae_report --assets "$assets" --run-dir "$inference" \
    --legacy-timing /run/rqkmeans-timing --legacy-audit "$original"

current=exporting
stage "$current"
docker cp "$container:$inference/." "$host/"
docker stop --time 10 "$container"
current=completed
stage "$current"
