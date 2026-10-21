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
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$RESULT_DIR"
rm -f private/writeback-validation.complete
"$PYTHON" -m pytest -q --tb=short --junitxml=private/pytest-v03.xml
revision=77f1b168e219585646439073245fe87e56b3023e
for tier in hdd ssd; do
  if [[ "$tier" == hdd ]]; then root="$HDD_ROOT"; else root="$SSD_ROOT"; fi
  "$PYTHON" examples/mixed_writeback.py --model roneneldan/TinyStories-1M \
    --revision "$revision" --model-cache-dir "$MODEL_CACHE" --local-files-only \
    --device cuda:0 --direct-io --root "$root" --archive-mib "${ARCHIVE_MIB:-1024}" \
    --tokens 1024 --repeats "${REPEATS:-3}" --requests 3 --decode-tokens 8 \
    --output "$RESULT_DIR/tinystories-mixed-$tier.json"
done
if [[ -n "${PROVENANCE_TESTS:-}" ]]; then
  "$PYTHON" scripts/validation_summary.py --junit private/pytest-v03.xml \
    --expected-tests "$PROVENANCE_TESTS" --local-passed "${LOCAL_PASSED:?Set LOCAL_PASSED}" \
    --local-skipped "${LOCAL_SKIPPED:?Set LOCAL_SKIPPED}" \
    --output "$RESULT_DIR/validation-summary-v03.json"
fi
touch private/writeback-validation.complete
