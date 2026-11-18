#!/usr/bin/env bash
set -euo pipefail
cd /opt/minionerec

if [[ -f "${ZIWEIZHAO_WORKDIR}/env.sh" ]]; then
    source "${ZIWEIZHAO_WORKDIR}/env.sh"
fi
export H20_OUTPUT_ROOT="${H20_OUTPUT_ROOT:-${ZIWEIZHAO_WORKDIR}/weights/checkpoints/MiniOneRec/Video_Games_sid2900}"
export BASE_MODEL="${BASE_MODEL:-${ZIWEIZHAO_WORKDIR}/weights/Qwen2.5-1.5B}"
if [[ ! -e data && -d "${ZIWEIZHAO_WORKDIR}/MiniOneRec/data" ]]; then
    ln -s "${ZIWEIZHAO_WORKDIR}/MiniOneRec/data" data
fi

action="${1:-dry-run}"
if (( $# )); then shift; fi
case "$action" in
    sft|rl)
        exec python scripts/launch_h20.py "$action" "$@"
        ;;
    dry-run|preflight)
        stage="${1:-sft}"
        if (( $# )); then shift; fi
        flag=--dry-run
        if [[ "$action" == preflight ]]; then flag=--preflight-only; fi
        exec python scripts/launch_h20.py "$stage" "$flag" "$@"
        ;;
    prepare-data)
        exec python scripts/prepare_video_games_posttraining.py "$@"
        ;;
    download-model)
        exec python docker/download_model.py "$@"
        ;;
    check-env)
        exec python docker/check_env.py "$@"
        ;;
    smoke-test)
        exec python docker/smoke_training.py "$@"
        ;;
    gpu-smoke)
        exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 docker/smoke_distributed.py "$@"
        ;;
    sft-monitor-smoke)
        exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 docker/smoke_sft_monitor.py "$@"
        ;;
    evaluate-sft)
        exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 scripts/evaluate_sft.py "$@"
        ;;
    report-sft)
        exec python scripts/report_sft.py "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        exec "$action" "$@"
        ;;
esac
