#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/Amazon18/raw"
PYTHON="${PYTHON:-python}"

for input in "${RAW_DIR}/Video_Games_5.json" "${RAW_DIR}/meta_Video_Games.json"; do
    if [[ ! -s "$input" ]]; then
        printf 'Missing or empty input: %s\nDownload and decompress the official JSONL file before running this script.\n' "$input" >&2
        exit 1
    fi
done

# Keep the filtering and time window from amazon18_data_process.sh.
# Explicit UTF-8 mode also supports these JSON files on Windows.
export PYTHONUTF8=1
exec "$PYTHON" "${SCRIPT_DIR}/amazon18_data_process.py" \
    --dataset Video_Games \
    --user_k 5 \
    --item_k 5 \
    --max_history_len 20 \
    --st_year 1996 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 10 \
    --metadata_file "${RAW_DIR}/meta_Video_Games.json" \
    --reviews_file "${RAW_DIR}/Video_Games_5.json" \
    --output_path "${SCRIPT_DIR}/Amazon18"
