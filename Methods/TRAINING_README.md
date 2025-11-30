# Unified Training Framework for XGBoost and ANN Models

This directory contains unified training scripts that can train models with different feature combinations through environment variable configuration.

## Feature Combinations

Six different feature combinations are supported:

1. **text_only** - Only BERT embeddings of the prompt text
2. **features_only** - All instance + prompt features (no text)
3. **inst_only** - Only instance features (mathematical properties)
4. **prompt_only** - Only prompt features (prompt engineering techniques)
5. **text_prompt** - BERT embeddings + prompt features
6. **all** - BERT embeddings + instance features + prompt features

## Files

### Training Scripts
- **xgboost_unified.py** - Unified XGBoost training script
- **ann_unified.py** - Unified ANN training script

### Batch Training
- **train_all_combinations.sh** - Shell script to train all 6 combinations
- **run_train_all_xgboost.sh** - SLURM script for XGBoost batch training
- **run_train_all_ann.sh** - SLURM script for ANN batch training

### Evaluation
- **evaluate_all_models.py** - Evaluate all trained models and compare results

## Quick Start

### Training a Single Combination

Set the feature mode in `.env`:
```bash
FEATURE_MODE=text_only  # or features_only, inst_only, prompt_only, text_prompt, all
USE_BERT_EMBEDDINGS=true
```

Then run:
```bash
# XGBoost
python xgboost_unified.py

# ANN
python ann_unified.py
```

### Training All Combinations

**Option 1: Using bash script**
```bash
cd Methods
bash train_all_combinations.sh xgboost  # Train all XGBoost models
bash train_all_combinations.sh ann      # Train all ANN models
bash train_all_combinations.sh both     # Train both
```

**Option 2: Using SLURM (for cluster)**
```bash
# Train all XGBoost combinations
sbatch run_train_all_xgboost.sh

# Train all ANN combinations
sbatch run_train_all_ann.sh
```

### Evaluating All Models

After training, evaluate all models:
```bash
cd Methods
python evaluate_all_models.py
```

This will:
- Load all trained models
- Evaluate on the full dataset
- Generate comparison tables
- Save results to `model_comparison_results.csv`
- Identify best performing models

## Configuration

All configuration is done through environment variables in `.env`:

```bash
# Data file
DATA_FILE=../data_gen/countdown_results_with_prompt_gemini.csv

# Feature configuration
FEATURE_MODE=text_only  # Which features to use
USE_BERT_EMBEDDINGS=true  # Whether to extract BERT embeddings

# Model parameters
MODEL_NAME=bert-base-uncased
MAX_LENGTH=512
TEST_SIZE=0.2
RANDOM_STATE=42

# XGBoost parameters
XGB_ETA=0.1
XGB_MAX_DEPTH=5
XGB_N_ESTIMATORS=100

# ANN parameters
ANN_HIDDEN_SIZES=256,128,64
ANN_DROPOUT=0.3
ANN_LEARNING_RATE=0.001
ANN_BATCH_SIZE=32
ANN_NUM_EPOCHS=20

# Weights & Biases
WANDB_PROJECT=countdown-models
```

## Output Files

After training, each combination produces:

### XGBoost
- `xgboost_model_{feature_mode}.json` - Trained model
- `xgboost_summary_{feature_mode}.json` - Training summary and metrics

### ANN
- `ann_model_{feature_mode}_best.pth` - Best model weights
- `ann_summary_{feature_mode}.json` - Training summary and metrics

### Evaluation
- `model_comparison_results.csv` - Comparison table of all models
- `evaluation_results_detailed.json` - Detailed metrics for all models

## Feature Details

### Text Features (BERT Embeddings)
- Dimension: 768 (from bert-base-uncased)
- Extracted from the full prompt with numbers/target filled in
- Mean pooling of last hidden state

### Instance Features (24 features)
Mathematical properties of the countdown instance:
- Number of numbers, range, std
- Counts of small/large/duplicate/even/odd numbers
- Divisibility counts (div by 2, 3, 5, 7)
- Prime number count
- Distance metrics to target
- Solution complexity (depth, operator counts)

### Prompt Features (10 features)
Prompt engineering techniques used:
- Paraphrasing, role-specification, reasoning-trigger
- Chain-of-thought, self-check
- Conciseness, verbosity
- Context-expansion, few-shot-count
- Prompt length

## Example Workflow

```bash
# 1. Update .env with desired configuration
echo "FEATURE_MODE=all" >> .env

# 2. Train a single model
python xgboost_unified.py

# 3. Or train all combinations
bash train_all_combinations.sh xgboost

# 4. Evaluate all models
python evaluate_all_models.py

# 5. Check results
cat model_comparison_results.csv
```

## Tips

- **text_only** - Best for general performance, captures semantic meaning
- **features_only** - Fast training, good for understanding feature importance
- **all** - Usually best performance but slower to train
- **inst_only** - Good for understanding problem difficulty
- **prompt_only** - Good for analyzing prompt engineering impact
- **text_prompt** - Balance between semantic and prompt features

## Troubleshooting

**Out of memory during BERT embedding extraction:**
- Reduce batch size in the embedding extraction (default is 16)
- Use CPU instead of GPU (slower but more memory)

**Models not found during evaluation:**
- Ensure training completed successfully
- Check that model files exist in Methods directory
- Verify file naming matches feature mode

**Different results from manual training:**
- Check FEATURE_MODE environment variable is set correctly
- Ensure USE_BERT_EMBEDDINGS matches training configuration
- Verify same random seed is used
