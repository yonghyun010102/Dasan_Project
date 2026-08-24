#!/usr/bin/env bash
# Progressive ablation (paper Tab. 6): u -> s -> con_s -> full.
#   usage: scripts/run_ablation.sh [SEED] [BACKBONE] [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-42}"
BACKBONE="${2:-pkot5}"
GPU="${3:-0}"
RUN_ROOT="${RUN_ROOT:-runs}"
PY="${PY:-python}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

DET_DIR="$RUN_ROOT/detector_seed${SEED}"
[ -f data/splits/train.csv ] || "$PY" scripts/prepare_data.py
[ -d "$DET_DIR/best" ] || "$PY" scripts/train_detector.py --seed "$SEED" --run_root "$RUN_ROOT"

"$PY" scripts/evaluate.py --variant zero_rule --run_root "$RUN_ROOT" || true

for VARIANT in u s con_s full; do
  COR_DIR="$RUN_ROOT/corrector_${BACKBONE}_${VARIANT}_seed${SEED}"
  echo "=== variant ${VARIANT} ==="
  EXTRA=()
  [ "$VARIANT" = "full" ] && EXTRA=(--detector_dir "$DET_DIR/best")
  [ -d "$COR_DIR/best" ] || "$PY" scripts/train_corrector.py \
    --variant "$VARIANT" --backbone "$BACKBONE" --seed "$SEED" \
    --run_root "$RUN_ROOT" "${EXTRA[@]}"
  "$PY" scripts/evaluate.py --variant "$VARIANT" --backbone "$BACKBONE" --seed "$SEED" \
    --corrector_dir "$COR_DIR/best" --run_root "$RUN_ROOT" "${EXTRA[@]}"
done
