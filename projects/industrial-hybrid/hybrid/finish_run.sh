#!/usr/bin/env bash
# Finish and verify an inference run after launching hybrid.experiment with four ranks.
set -euo pipefail
run_dir="${1:-/run/hybrid}"
assets="${2:-/assets}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
trap 'echo $? > "$run_dir/pipeline.exit"' EXIT
have_valid() {
    local rank
    for rank in 0 1 2 3; do
        [[ -f "$run_dir/valid/rank$rank.jsonl" ]] || return 1
    done
}
while ! have_valid; do
    if [[ -f "$run_dir/generation.exit" ]]; then
        echo 'Incomplete validation shards after inference exited' >&2
        exit 1
    fi
    sleep 5
done
python -m hybrid.sweep --run-dir "$run_dir" select
while [[ ! -f "$run_dir/generation.exit" ]]; do sleep 5; done
[[ $(cat "$run_dir/generation.exit") == 0 ]]
python -m hybrid.sweep --run-dir "$run_dir" quality
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.benchmark --assets "$assets" --run-dir "$run_dir" --mode both > "$run_dir/performance.log" 2>&1
python -m hybrid.sweep --run-dir "$run_dir" report
