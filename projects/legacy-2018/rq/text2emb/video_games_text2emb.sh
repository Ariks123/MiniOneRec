#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-G:/Qwen}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_DIR="${PROJECT_ROOT}/data/Amazon18/Video_Games"

for input in "${DATA_DIR}/Video_Games.item.json" "${MODEL_DIR}/config.json" "${MODEL_DIR}/model.safetensors.index.json" "${MODEL_DIR}/model-00001-of-00002.safetensors" "${MODEL_DIR}/model-00002-of-00002.safetensors" "${MODEL_DIR}/tokenizer.json" "${MODEL_DIR}/tokenizer_config.json"; do
    if [[ ! -s "$input" ]]; then
        printf 'Missing or empty input: %s\n' "$input" >&2
        exit 1
    fi
done

export PYTHONUTF8=1
exec "$PYTHON" -X utf8 "${SCRIPT_DIR}/amazon_text2emb.py" \
    --dataset Video_Games \
    --root "${DATA_DIR}" \
    --plm_name qwen \
    --plm_checkpoint "${MODEL_DIR}" \
    --max_sent_len 2048 \
    --batch_size "${BATCH_SIZE}" \
    --local_files_only
