#!/usr/bin/env bash
# Run from any directory; all persistent files remain in the mounted workspace.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="${ZIWEIZHAO_WORKDIR:-$(cd -- "$ROOT_DIR/.." && pwd)}"
IMAGE="${MINIONEREC_IMAGE:-minionerec:h20-cu124}"
DOCKER_BIN="${DOCKER:-docker}"
action="${1:-dry-run}"
if (( $# )); then shift; fi

case "$action" in
    sft-run)
        exec "${PYTHON:-python}" "$ROOT_DIR/docker/sft_experiment.py" start "$@"
        ;;
    sft-status)
        exec "${PYTHON:-python}" "$ROOT_DIR/docker/sft_experiment.py" status "$@"
        ;;
esac

if [[ "$action" == build ]]; then
    exec "$DOCKER_BIN" build -f "$ROOT_DIR/docker/Dockerfile.h20" -t "$IMAGE" "$@" "$ROOT_DIR"
fi

mode="${H20_DOCKER_MODE:-auto}"
if [[ "$mode" == auto ]]; then
    mode=bind
    if [[ "${DOCKER_HOST:-}" == *docker-proxy.sock ]]; then mode=copy; fi
fi
if [[ "$mode" == copy ]]; then
    exec "${PYTHON:-python}" "$ROOT_DIR/docker/run_copy.py" "$action" "$@"
elif [[ "$mode" != bind ]]; then
    echo 'H20_DOCKER_MODE must be auto, bind, or copy' >&2
    exit 2
fi

options=(run --rm --shm-size "${H20_SHM_SIZE:-16g}"
    --mount "type=bind,source=$ROOT_DIR,target=/opt/minionerec"
    --mount "type=bind,source=$WORKSPACE_DIR,target=/mnt/ossdata/ziweizhao"
    -e CUDA_VISIBLE_DEVICES=0,1,2,3)
case "$action" in
    sft|rl|preflight|gpu-smoke|sft-monitor-smoke|evaluate-sft|shell)
        # Quote the entire device CSV for Docker's --gpus parser.
        options+=(--gpus "\"device=${H20_GPUS:-0,1,2,3}\"")
        ;;
esac
if [[ -t 0 && -t 1 ]]; then options+=(-it); fi
# Pass names only, so tokens never appear in command-line output.
for name in HF_TOKEN HF_ENDPOINT MODEL_REVISION BASE_MODEL SFT_MODEL H20_OUTPUT_ROOT \
    NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_DEBUG OMP_NUM_THREADS; do
    if [[ -v "$name" ]]; then options+=(-e "$name"); fi
done
exec "$DOCKER_BIN" "${options[@]}" "$IMAGE" "$action" "$@"
