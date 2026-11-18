#!/usr/bin/env bash
# Run a separate instrumented beam50 test after the uninstrumented benchmarks.
set -euo pipefail
run="$(realpath "${1:?usage: launch_industrial_rqvae_tpot.sh RUN_DIR}")"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd -- "$repo/../.." && pwd)"
old=industrial-hybrid-beta001-infer-v1
new=industrial-rqvae-2k-infer-20261118
result="$run/inference/report"

status() {
    printf '%s\n' "$1" > "$run/tpot-stage.status.next"
    mv "$run/tpot-stage.status.next" "$run/tpot-stage.status"
    printf '%s %s\n' "$(date -Iseconds)" "$1"
}

trap 'code=$?; if ((code != 0)); then status "failed:$code"; fi' EXIT
status waiting-for-benchmarks
while true; do
    state="$(cat "$run/inference/stage.status")"
    case "$state" in
        completed) break ;;
        failed:*) printf '%s\n' "Inference failed before TPOT: $state" >&2; exit 1 ;;
        *) sleep 10 ;;
    esac
done

if [[ "$(docker inspect --format '{{.Image}}' "$old")" != "$(docker inspect --format '{{.Image}}' "$new")" ]]; then
    printf '%s\n' "The two benchmark containers use different images" >&2
    exit 1
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$old")" != true ]]; then
    docker start "$old"
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$new")" != true ]]; then
    docker start "$new"
fi
status waiting-for-gpu
while [[ -n "$(docker exec "$old" nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 5
done

measure() {
    local container="$1" assets="$2" protocol="$3" output="$4"
    shift 4
    docker cp "$repo/hybrid/rqvae_tpot.py" "$container:/work/hybrid/"
    docker exec -w /work -e CUDA_VISIBLE_DEVICES=0 -e TOKENIZERS_PARALLELISM=false \
        -e OMP_NUM_THREADS=2 "$container" python -B -m hybrid.rqvae_tpot \
        --assets "$assets" --protocol "$protocol" --output "/tmp/$output" "$@"
    docker cp "$container:/tmp/$output" "$result/$output"
}

status legacy-beam-tpot
measure "$old" /assets /run/hybrid-rho1/report/benchmark_protocol.json tpot_rqkmeans.json
status rqvae-beam-tpot
measure "$new" /run/rqvae-assets /run/rqvae-inference/report/benchmark_protocol.json tpot_rqvae.json --rqvae
docker stop --time 10 "$new"
status publishing
cd "$repo"
env MPLCONFIGDIR=/tmp/industrial-rqvae-report-mpl \
    /root/miniconda3/envs/vpp2/bin/python -B -m scripts.report_industrial_rqvae_rerun \
    --workspace "$workspace" --run-dir "$run" --inference-dir "$run/inference" \
    --destination "$workspace/MiniOneRec/report/Industrial-2018-RQVAE-vs-RQKMeans-2K.zip"
status completed
