#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}"
: "${HDD_ROOT:?Set HDD_ROOT}"
: "${SSD_ROOT:?Set SSD_ROOT}"
: "${MODEL_CACHE:?Set MODEL_CACHE}"
: "${RESULT_DIR:=results/consumer-cuda}"
export PYTHONPATH="$(pwd)/private/test-deps:$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$RESULT_DIR"
rm -f private/pretrained-small.complete
"$PYTHON" -m pytest -q --tb=short --junitxml=private/pytest-v02.xml
if [[ "${SKIP_TRANSPORT:-0}" != 1 ]]; then
rm -f private/pipeline-bench.complete
for mode in serial pipelined; do
  flags=()
  if [[ "$mode" == serial ]]; then flags+=(--no-pipeline-reads); fi
  "$PYTHON" -m kvpark.benchmark --root "$SSD_ROOT" --label "ssd-v02-$mode" \
    --sizes-mib 256 1024 --repeats 5 --direct-io "${flags[@]}" \
    --output "$RESULT_DIR/ssd-v02-$mode.json"
done
"$PYTHON" -m kvpark.benchmark --root "$HDD_ROOT" --label hdd-v02-pipelined \
  --sizes-mib 1024 --repeats 3 --direct-io --output "$RESULT_DIR/hdd-v02-pipelined.json"
touch private/pipeline-bench.complete
fi

ready=false
for ((attempt = 0; attempt < 180; attempt++)); do
  if [[ -f private/tiny-model.complete ]]; then ready=true; break; fi
  sleep 5
done
if [[ "$ready" != true ]]; then echo 'Verified pretrained model has not arrived.' >&2; exit 1; fi
revision=77f1b168e219585646439073245fe87e56b3023e
common=(--model roneneldan/TinyStories-1M --revision "$revision" \
  --model-cache-dir "$MODEL_CACHE" --local-files-only --device cuda:0 \
  --attn-implementation eager --direct-io)
"$PYTHON" examples/hf_resume.py "${common[@]}" --root "$SSD_ROOT" --tokens 1024 \
  --repeats 5 --output "$RESULT_DIR/tinystories-cuda.json"
for tier in hdd ssd; do
  if [[ "$tier" == hdd ]]; then root="$HDD_ROOT"; else root="$SSD_ROOT"; fi
  "$PYTHON" examples/queued_resume.py "${common[@]}" --root "$root" --tokens 1024 \
    --sessions 4 --decode-tokens 64 --output "$RESULT_DIR/tinystories-queued-$tier.json"
done
touch private/pretrained-small.complete
printf '\nPipeline and pretrained validation complete.\n'
