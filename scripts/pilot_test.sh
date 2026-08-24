#!/usr/bin/env bash
# Fast smoke test: exercises data prep, detector training, every corrector
# variant and evaluation on a few hundred rows. Finishes in minutes.
#   usage: scripts/pilot_test.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-runs/pilot}"
PY="${PY:-python}"
BACKBONE="${BACKBONE:-pkot5}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

echo "=== pilot [1] data ==="
"$PY" scripts/prepare_data.py

echo "=== pilot [2] detector (1 epoch, 2k rows) ==="
"$PY" scripts/train_detector.py --seed 42 --run_root "$RUN_ROOT" \
  --epochs 1 --limit_train 2000 --limit_val 300

echo "=== pilot [3] zero-rule reference ==="
"$PY" scripts/evaluate.py --variant zero_rule --run_root "$RUN_ROOT" --limit_dialogues 5

for VARIANT in u s con_s full; do
  echo "=== pilot [4:${VARIANT}] corrector + eval ==="
  EXTRA=()
  [ "$VARIANT" = "full" ] && EXTRA=(--detector_dir "$RUN_ROOT/detector_seed42/best")
  "$PY" scripts/train_corrector.py --variant "$VARIANT" --backbone "$BACKBONE" --seed 42 \
    --run_root "$RUN_ROOT" --epochs 1 --limit_train 400 --limit_val 100 "${EXTRA[@]}"
  "$PY" scripts/evaluate.py --variant "$VARIANT" --backbone "$BACKBONE" --seed 42 \
    --corrector_dir "$RUN_ROOT/corrector_${BACKBONE}_${VARIANT}_seed42/best" \
    --run_root "$RUN_ROOT" --limit_dialogues 5 "${EXTRA[@]}"
done
echo "=== pilot complete ==="
