#!/usr/bin/env bash
# Continue the RQ-VAE experiment after its new catalog has been exported.
set -euo pipefail

run="$(realpath "${1:?usage: launch_industrial_rqvae_unisrec.sh RUN_DIR}")"
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
container=industrial-rqvae-unisrec-20261118
manifest="$run/data/preparation_manifest.json"
dataset=Industrial_and_Scientific
sid="$run/catalog/$dataset.index.json"
metadata="$run/catalog/$dataset.item.json"

publish_status() {
    printf '%s\n' "$1" > "$run/unisrec-stage.status.tmp"
    mv "$run/unisrec-stage.status.tmp" "$run/unisrec-stage.status"
}

trap 'code=$?; if ((code != 0)); then publish_status "failed:$code"; fi' EXIT
publish_status waiting-for-new-sid
until [[ -s "$manifest" && -s "$sid" && -s "$metadata" ]]; do
    sleep 5
done
python3 - "$manifest" "$sid" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest, sid = map(Path, sys.argv[1:])
prepared = json.loads(manifest.read_text())
if prepared["original_sid"] or hashlib.sha256(sid.read_bytes()).hexdigest() != prepared["expected_sid_sha256"]:
    raise ValueError("UniSRec must see the selected, regenerated SID catalog")
PY

publish_status waiting-for-gpus
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 5
done
publish_status running
docker cp "$sid" "$container:/opt/industrial-input/catalog/"
docker cp "$metadata" "$container:/opt/industrial-input/catalog/"
docker cp "$project/hybrid/drafter_export.py" "$container:/opt/industrial-input/"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$container" python -u /opt/industrial-input/retrain_industrial_unisrec.py

publish_status exporting
mkdir -p "$run/unisrec"
docker cp "$container:/opt/industrial-run/." "$run/unisrec/"
python3 - "$run" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
data = json.loads((run/"data/preparation_manifest.json").read_text())
model = json.loads((run/"unisrec/status.json").read_text())
export = json.loads((run/"unisrec/export/metadata.json").read_text())
if not (model["stage"] == "completed" and model["sid_sha256"] == data["expected_sid_sha256"]
        and model["checkpoint_sha256"] == export["checkpoint_sha256"]):
    raise ValueError("UniSRec stage and export identity did not match")
PY
publish_status completed

# Require the regenerated SID vocabulary and dataset checks before LLM training.
until [[ -s "$run/posttraining_preflight.json" ]]; do
    sleep 5
done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 5
done
printf '%s\n' "UniSRec export validated" > "$run/posttraining.gpu-ready"
