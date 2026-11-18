#!/usr/bin/env bash
set -euo pipefail
QWEN_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_WORKSPACE_DIR="$(cd -- "$QWEN_PROJECT_DIR/../../.." && pwd)"
if [[ -f "$QWEN_WORKSPACE_DIR/env.sh" ]]; then
    source "$QWEN_WORKSPACE_DIR/env.sh"
fi
cd "$QWEN_PROJECT_DIR"
QWEN_MODE="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
QWEN_PYTHON_BIN="${QWEN_PYTHON:-python3}"
case "$QWEN_MODE" in
    preflight)
        exec "$QWEN_PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
            qwen_ddp_preflight.py --config props/qwen_industrial_4gpu.yaml "$@"
        ;;
    train)
        exec "$QWEN_PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
            qwen_train.py --config props/qwen_industrial_4gpu.yaml --start-training "$@"
        ;;
    *)
        echo 'Usage: bash scripts/run_4gpu.sh [preflight|train] [additional arguments]' >&2
        exit 2
        ;;
esac
