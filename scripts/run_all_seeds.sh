#!/usr/bin/env bash
# Train and evaluate DCSC for every seed and both primary backbones.
# Detector is shared per seed; pkoT5 runs on GPU 0, Llama on GPU 1.
#   usage: scripts/run_all_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT="${RUN_ROOT:-runs}"
PY="${PY:-python}"
SEEDS="${SEEDS:-42 43 44}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
export TOKENIZERS_PARALLELISM=false

[ -f data/splits/train.csv ] || "$PY" scripts/prepare_data.py

run_backbone () {          # $1 backbone, $2 gpu
  local backbone="$1" gpu="$2"
  for seed in $SEEDS; do
    local det="$RUN_ROOT/detector_seed${seed}"
    local cor="$RUN_ROOT/corrector_${backbone}_full_seed${seed}"
    echo "[$backbone] seed $seed : corrector"
    if [ ! -d "$cor/best" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_corrector.py \
        --variant full --backbone "$backbone" --seed "$seed" \
        --detector_dir "$det/best" --run_root "$RUN_ROOT" \
        > "$LOG_DIR/corrector_${backbone}_seed${seed}.log" 2>&1
    fi
    echo "[$backbone] seed $seed : evaluate"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/evaluate.py \
      --variant full --backbone "$backbone" --seed "$seed" \
      --detector_dir "$det/best" --corrector_dir "$cor/best" --run_root "$RUN_ROOT" \
      > "$LOG_DIR/eval_${backbone}_seed${seed}.log" 2>&1
  done
}

# --- detectors first (shared by both backbones), sequentially on GPU 0 ---
for seed in $SEEDS; do
  det="$RUN_ROOT/detector_seed${seed}"
  if [ ! -d "$det/best" ]; then
    echo "[detector] seed $seed"
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_detector.py --seed "$seed" --run_root "$RUN_ROOT" \
      > "$LOG_DIR/detector_seed${seed}.log" 2>&1
  fi
done

# --- correctors in parallel across GPUs ---
run_backbone pkot5 0 &
PID_T5=$!
run_backbone llama 1 &
PID_LLM=$!
wait $PID_T5; wait $PID_LLM

"$PY" scripts/evaluate.py --variant zero_rule --run_root "$RUN_ROOT" \
  > "$LOG_DIR/eval_zero_rule.log" 2>&1 || true
"$PY" scripts/aggregate_seeds.py --run_root "$RUN_ROOT" --out "$RUN_ROOT/summary.json"
touch "$RUN_ROOT/ALL_SEEDS_DONE"
