#!/usr/bin/env bash
set -euo pipefail

# Expected counts
EXPECTED_FFT=5
EXPECTED_LOW_LR=5
EXPECTED_HIGH_LR=9

# Models and ranks
MODELS=("roberta-base" "roberta-large")
LOW_RANKS=(1 4 8 16 32)
METHODS=("lora" "rslora")

# Per-model high ranks
declare -A HIGH_RANKS_MAP
HIGH_RANKS_MAP["roberta-base"]="64 128 256 512 768"
HIGH_RANKS_MAP["roberta-large"]="64 128 256 512 1024"

# ---- helpers ---------------------------------------------------
count_matches () { # dir pattern -> prints count
  local dir="$1" pat="$2"
  find "$dir" -maxdepth 1 -type d -name "$pat" 2>/dev/null | wc -l | tr -d ' '
}

report () { # label count expected
  local label="$1" count="$2" expected="$3"
  if [[ "$count" -eq "$expected" ]]; then
    echo "FOUND     | $label ($count/$expected)"
  else
    echo "NOT FOUND | $label ($count/$expected)"
  fi
}

# ---- input -----------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_name>   e.g., $0 rte"
  exit 1
fi

TASK="$1"
TASK_DIR="./$TASK"
[[ -d "$TASK_DIR" ]] || { echo "❗ Task dir '$TASK_DIR' not found."; exit 1; }

echo "=== Verifying task: $TASK ==="
echo

for model in "${MODELS[@]}"; do
  MODEL_DIR="$TASK_DIR/$model"
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "NOT FOUND | model dir: $MODEL_DIR"
    echo
    continue
  fi

  echo ">> $model"

  # 1) Full fine-tuning
  fft_count=$(count_matches "$MODEL_DIR" "full_finetuning_*")
  report "full_finetuning_*" "$fft_count" "$EXPECTED_FFT"

  # 2) LoRA / RSLoRA
  for method in "${METHODS[@]}"; do
    # low ranks
    for r in "${LOW_RANKS[@]}"; do
      c=$(count_matches "$MODEL_DIR" "${method}_r${r}_*")
      report "${method} r=${r}" "$c" "$EXPECTED_LOW_LR"
    done
    # high ranks (per model)
    read -r -a HIGH_RANKS <<< "${HIGH_RANKS_MAP[$model]}"
    for r in "${HIGH_RANKS[@]}"; do
      c=$(count_matches "$MODEL_DIR" "${method}_r${r}_*")
      report "${method} r=${r}" "$c" "$EXPECTED_HIGH_LR"
    done
  done

  echo
done

echo "=== Verification complete ==="
