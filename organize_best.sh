#!/usr/bin/env bash
set -euo pipefail

# Tasks and models to scan
TASKS=("rte" )
MODELS=("roberta-base" "roberta-large")
METHODS=("full_finetuning" "lora" "rslora")

DRY_RUN=0
if [[ "${1-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "(dry run) No files will be copied."
fi

mkdir -p glue_results_final

copy_or_echo () { # src dest
  local src="$1" dest="$2"
  mkdir -p "$dest"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  would copy: $src  ->  $dest/"
  else
    echo "  copying:    $src  ->  $dest/"
    cp -r "$src" "$dest/"
  fi
}

for TASK in "${TASKS[@]}"; do
  echo "Processing task: $TASK"
  for MODEL in "${MODELS[@]}"; do
    SRC="./$TASK/$MODEL"
    if [[ ! -d "$SRC" ]]; then
      echo "  Warning: missing source: $SRC (skipping)"
      continue
    fi

    DEST="./glue_results_final/$TASK/$MODEL"
    for M in "${METHODS[@]}"; do mkdir -p "$DEST/$M"; done

    echo "  Model: $MODEL"

    # full_finetuning bests
    while IFS= read -r -d '' d; do
      copy_or_echo "$d" "$DEST/full_finetuning"
    done < <(find "$SRC" -maxdepth 1 -type d -name 'full_finetuning_*_best_accuracy' -print0)

    # lora bests
    while IFS= read -r -d '' d; do
      copy_or_echo "$d" "$DEST/lora"
    done < <(find "$SRC" -maxdepth 1 -type d -name 'lora_*_best_accuracy' -print0)

    # rslora bests
    while IFS= read -r -d '' d; do
      copy_or_echo "$d" "$DEST/rslora"
    done < <(find "$SRC" -maxdepth 1 -type d -name 'rslora_*_best_accuracy' -print0)

  done
done

echo "Done."
