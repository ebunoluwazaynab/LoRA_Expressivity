#!/bin/bash

# Define the list of tasks to process
TASKS=("rte" "mrpc" "cola")

# Define the model types and fine-tuning methods
MODELS=("roberta-base" "roberta-large")
METHODS=("full_finetuning" "lora" "rslora")

# Create the main 'glue' directory if it doesn't exist
mkdir -p glue

# Loop through each task
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"

    # Loop through each model type
    for MODEL in "${MODELS[@]}"; do
        SOURCE_DIR="./$TASK/$MODEL"
        DEST_DIR="./glue/$TASK/$MODEL"

        # Check if the source directory exists
        if [ -d "$SOURCE_DIR" ]; then
            echo "  Processing model: $MODEL"

            # Create the destination directory and all subdirectories
            for METHOD in "${METHODS[@]}"; do
                mkdir -p "$DEST_DIR/$METHOD"
            done

            cp -r "$SOURCE_DIR"/*full_finetuning* "$DEST_DIR/full_finetuning/" 2>/dev/null
            cp -r "$SOURCE_DIR"/*lora* "$DEST_DIR/lora/" 2>/dev/null
            cp -r "$SOURCE_DIR"/*rslora* "$DEST_DIR/rslora/" 2>/dev/null
            


        else
            echo "  Warning: Source directory not found: $SOURCE_DIR. Skipping."
        fi
    done
done

echo "Script complete. The new directory structure is in the 'glue' folder."