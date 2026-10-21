#!/usr/bin/env bash
# Offline-capable, public-weight model validation. No live service is modified.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}"
: "${HDD_ROOT:?Set HDD_ROOT}"
: "${SSD_ROOT:?Set SSD_ROOT}"
: "${MODEL_CACHE:?Set MODEL_CACHE to the Hugging Face cache containing the pinned model}"
: "${RESULT_DIR:=results/consumer-cuda}"
revision=7ae557604adf67be50417f59c2c2f167def9a775
weight="$MODEL_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$revision/model.safetensors"
# A detached run may wait for an in-flight rsync's atomic final rename.
ready=false
for ((attempt = 0; attempt < 180; attempt++)); do
  if [[ -f "$weight" ]] && [[ "$(stat -c %s "$weight")" == 988097824 ]]; then
    ready=true
    break
  fi
  sleep 10
done
if [[ "$ready" != true ]]; then
  echo 'Pinned model weights have not arrived after 30 minutes.' >&2
  exit 1
fi
printf '%s  %s\n' fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe \
  "$weight" | sha256sum --check
mkdir -p "$RESULT_DIR"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
common=(--model Qwen/Qwen2.5-0.5B-Instruct --revision "$revision" \
  --model-cache-dir "$MODEL_CACHE" --local-files-only --device cuda:0 --direct-io)
"$PYTHON" examples/hf_resume.py "${common[@]}" --root "$SSD_ROOT" --tokens 8192 \
  --repeats 5 --output "$RESULT_DIR/qwen-half-b-8192.json"
for tier in hdd ssd; do
  if [[ "$tier" == hdd ]]; then root="$HDD_ROOT"; else root="$SSD_ROOT"; fi
  "$PYTHON" examples/queued_resume.py "${common[@]}" --root "$root" --tokens 4096 \
    --sessions 4 --decode-tokens 64 --output "$RESULT_DIR/qwen-queued-$tier.json"
done
touch private/pretrained.complete
printf '\nPretrained validation complete.\n'
