#!/usr/bin/env bash
# Run only inside an isolated experiment checkout; never stop other GPU services.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}"
: "${HDD_ROOT:?Set HDD_ROOT to a writable directory on the cold storage device}"
: "${SSD_ROOT:?Set SSD_ROOT to a writable directory on the SSD device}"
: "${RESULT_DIR:=results/consumer-cuda}"
mkdir -p "$RESULT_DIR" private/test-deps
rm -f private/matrix.complete
export PYTHONPATH="$(pwd)/private/test-deps:$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=15

if [[ -n "${KV_WHEEL_DIR:-}" ]]; then
  "$PYTHON" -m pip install --target private/test-deps --no-index \
    --find-links "$KV_WHEEL_DIR" 'pytest>=8,<10' >private/test-dependencies.log 2>&1
else
  "$PYTHON" -m pip install --target private/test-deps --timeout 20 --retries 2 \
    'pytest>=8,<10' >private/test-dependencies.log 2>&1
fi
"$PYTHON" -m pytest -q --tb=short --junitxml=private/pytest.xml

for tier in hdd ssd; do
  if [[ "$tier" == hdd ]]; then root="$HDD_ROOT"; else root="$SSD_ROOT"; fi
  "$PYTHON" -m kv_tier.benchmark --root "$root" --label "$tier" \
    --device cuda:0 --sizes-mib 16 256 1024 --repeats 5 --direct-io \
    --output "$RESULT_DIR/$tier-verified.json"
done

"$PYTHON" examples/hf_resume.py --root "$SSD_ROOT" --device cuda:0 \
  --tokens 512 --repeats 5 --direct-io --output "$RESULT_DIR/tiny-llama-cuda.json"

for chunk in 1 8 32; do
  for slots in 1 2; do
    "$PYTHON" -m kv_tier.benchmark --root "$SSD_ROOT" --label ssd-tuning \
      --device cuda:0 --sizes-mib 256 --repeats 3 --direct-io \
      --chunk-mib "$chunk" --slots "$slots" \
      --output "$RESULT_DIR/ssd-chunk${chunk}-slots${slots}.json"
  done
done
"$PYTHON" -m kv_tier.benchmark --root "$SSD_ROOT" --label ssd-trusted-no-read-checksum \
  --device cuda:0 --sizes-mib 256 1024 --repeats 5 --direct-io --no-verify \
  --output "$RESULT_DIR/ssd-no-read-checksum.json"
printf '\nMatrix complete. Results: %s\n' "$RESULT_DIR"
touch private/matrix.complete
