#!/usr/bin/env bash
# End-to-end DCSC (the paper's main pipeline) for one seed.
#   usage: scripts/run_dcsc.sh [SEED] [BACKBONE] [GPU]
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
COR_DIR="$RUN_ROOT/corrector_${BACKBONE}_full_seed${SEED}"

echo "=== [1/4] data preparation ==="
[ -f data/splits/train.csv ] || "$PY" scripts/prepare_data.py

echo "=== [2/4] detector (KoELECTRA, token-level) ==="
[ -d "$DET_DIR/best" ] || "$PY" scripts/train_detector.py --seed "$SEED" --run_root "$RUN_ROOT"

echo "=== [3/4] corrector (${BACKBONE}, span-level, detector-gated) ==="
[ -d "$COR_DIR/best" ] || "$PY" scripts/train_corrector.py \
  --variant full --backbone "$BACKBONE" --seed "$SEED" \
  --detector_dir "$DET_DIR/best" --run_root "$RUN_ROOT"

echo "=== [4/4] evaluation on test ==="
"$PY" scripts/evaluate.py \
  --variant full --backbone "$BACKBONE" --seed "$SEED" \
  --detector_dir "$DET_DIR/best" --corrector_dir "$COR_DIR/best" \
  --run_root "$RUN_ROOT"
