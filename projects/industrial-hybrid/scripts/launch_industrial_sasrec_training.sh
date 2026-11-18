#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -ne 2 ]]; then
    printf '%s\n' 'Usage: launch_industrial_sasrec_training.sh DATA_DIR RUN_DIR' \
        'Run in a RecBole 1.2.1 environment with four idle GPUs.' \
        'CUDA_VISIBLE_DEVICES defaults to 0,1,2,3. Completed cases are reused.'
    if [[ ${1:-} == --help ]]; then exit 0; fi
    exit 2
fi

data="$(realpath -- "$1")"
run="$(realpath -m -- "$2")"
if [[ "$run" == "$data" || "$run" == "$data/"* ]]; then
    printf '%s\n' 'Use an output directory outside the source data.' >&2
    exit 2
fi
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#devices[@]} != 4 )); then
    printf '%s\n' 'Provide four GPU IDs in CUDA_VISIBLE_DEVICES.' >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p -- "$run"
read -r run_hash _ < <(printf '%s' "$run" | sha256sum)
exec 9>"/tmp/sasrec-training-$run_hash.lock"
flock -n 9
if [[ -e "$run/selection.json" ]]; then
    printf 'SASRec artifacts are already published in %s.\n' "$run" >&2
    exit 2
fi
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
pids=()
for case_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${devices[case_index]}" python -u -B -m sasrec_baseline.train \
        --data "$data" --run "$run" --case "$case_index" >> "$run/case-$case_index.log" 2>&1 &
    pids+=("$!")
done
exit_code=0
for process in "${pids[@]}"; do
    wait "$process" || exit_code=$?
done
if (( exit_code != 0 )); then
    printf 'SASRec training failed. Check the case logs in %s.\n' "$run" >&2
    exit "$exit_code"
fi
CUDA_VISIBLE_DEVICES="${devices[0]}" python -u -B -m sasrec_baseline.train \
    --data "$data" --run "$run" --select 2>&1 | tee -a "$run/select.log"
