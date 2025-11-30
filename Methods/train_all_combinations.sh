#!/bin/bash
# Script to train all 6 feature combinations for both XGBoost and ANN

MODEL_TYPE=$1  # "xgboost" or "ann"

if [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 [xgboost|ann|both]"
    exit 1
fi

# Define the 6 feature combinations
FEATURE_MODES=("text_only" "features_only" "inst_only" "prompt_only" "text_prompt" "all")

echo "======================================================================="
echo "TRAINING ALL FEATURE COMBINATIONS"
echo "======================================================================="
echo "Model Type: $MODEL_TYPE"
echo "Feature Combinations: ${FEATURE_MODES[@]}"
echo "======================================================================="
echo ""

# Function to train a single combination
train_combination() {
    local mode=$1
    local script=$2

    echo "-----------------------------------------------------------------------"
    echo "Training: $mode"
    echo "-----------------------------------------------------------------------"

    # Set environment variable
    export FEATURE_MODE=$mode

    # Run the training script
    python $script

    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained $mode"
    else
        echo "✗ Failed to train $mode"
    fi

    echo ""
}

# Train XGBoost models
if [ "$MODEL_TYPE" == "xgboost" ] || [ "$MODEL_TYPE" == "both" ]; then
    echo "======================================================================="
    echo "TRAINING XGBOOST MODELS"
    echo "======================================================================="
    echo ""

    for mode in "${FEATURE_MODES[@]}"; do
        train_combination $mode "xgboost_unified.py"
    done
fi

# Train ANN models
if [ "$MODEL_TYPE" == "ann" ] || [ "$MODEL_TYPE" == "both" ]; then
    echo "======================================================================="
    echo "TRAINING ANN MODELS"
    echo "======================================================================="
    echo ""

    for mode in "${FEATURE_MODES[@]}"; do
        train_combination $mode "ann_unified.py"
    done
fi

echo "======================================================================="
echo "TRAINING COMPLETE"
echo "======================================================================="
echo "Trained ${#FEATURE_MODES[@]} combinations"
echo ""
