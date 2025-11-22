# ML Models for Prompt Correctness Prediction

This directory contains three different approaches for predicting whether a prompt will result in a correct response:

1. **BERT Fine-tuning** ([bert.py](bert.py)) - End-to-end BERT training
2. **XGBoost** ([xgboost_model.py](xgboost_model.py)) - XGBoost with BERT embeddings
3. **Simple ANN** ([ann_model.py](ann_model.py)) - Artificial Neural Network with BERT embeddings

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers datasets wandb scikit-learn xgboost pandas numpy tqdm python-dotenv
```

### 2. Configure Environment Variables

All models read configuration from a `.env` file in the parent directory. This makes it easy to change paths and hyperparameters without modifying code.

**Copy the example file:**
```bash
cd /path/to/15.099\ Homework/Project
cp .env.example .env
```

**Edit `.env` to customize:**
```bash
nano .env  # or use your preferred editor
```

### 3. Key Configuration Options

#### Data Paths
```env
PROMPTS_FILE=../prompts.csv
RESULTS_FILE=../results.csv
```

#### Model Selection
```env
MODEL_NAME=bert-base-uncased  # or bert-large-uncased, roberta-base, etc.
```

#### BERT Training
```env
BERT_BATCH_SIZE=16
BERT_LEARNING_RATE=2e-5
BERT_NUM_EPOCHS=3
```

#### XGBoost
```env
XGB_ETA=0.1
XGB_MAX_DEPTH=5
XGB_N_ESTIMATORS=100
```

#### ANN
```env
ANN_HIDDEN_SIZES=256,128,64  # Comma-separated layer sizes
ANN_DROPOUT=0.3
ANN_LEARNING_RATE=0.001
ANN_NUM_EPOCHS=20
```

## Usage

### Running Locally

```bash
cd Methods

# BERT fine-tuning
python bert.py

# XGBoost
python xgboost_model.py

# ANN
python ann_model.py
```

### Running on ORCD/Engaging Cluster

See the parent directory's `run_script.sh` for SLURM batch job examples.

**Example for BERT:**
```bash
#!/bin/bash
#SBATCH -J bert_finetune
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate promptenv

cd Methods
python bert.py
```

## Output

Each model produces:
- **Trained model file** (path specified in `.env`)
- **JSON summary** with training/validation/test metrics
- **Wandb logs** (if configured)

### Example JSON Output

```json
{
  "model_configuration": {
    "model_name": "bert-base-uncased",
    "num_labels": 2,
    "max_length": 512
  },
  "test_results": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1": 0.85
  }
}
```

## Model Comparison

| Model | Training Time | Memory | Accuracy (Expected) |
|-------|--------------|--------|---------------------|
| BERT Fine-tuning | ~30-60 min | High (GPU) | Highest |
| XGBoost | ~10-20 min | Medium | Good |
| Simple ANN | ~15-30 min | Medium | Moderate |

## Wandb Integration

All models log to Wandb. Configure your project name in `.env`:

```env
WANDB_PROJECT=your-project-name
```

First time setup:
```bash
wandb login
```

## Troubleshooting

### Import Error: `dotenv`
```bash
pip install python-dotenv
```

### File Not Found
Check that paths in `.env` are relative to the script location:
```env
# If running from Methods/ directory
PROMPTS_FILE=../prompts.csv
```

### CUDA Out of Memory
Reduce batch size in `.env`:
```env
BERT_BATCH_SIZE=8  # Instead of 16
ANN_BATCH_SIZE=16  # Instead of 32
```

## Advanced Configuration

### Using Different Embedding Models

Change the base model:
```env
MODEL_NAME=roberta-base
# or
MODEL_NAME=bert-large-uncased
# or
MODEL_NAME=distilbert-base-uncased
```

### Adjusting Train/Test Split

```env
TEST_SIZE=0.2  # 80/20 split
# or
TEST_SIZE=0.1  # 90/10 split
```

### Customizing Output Locations

```env
BERT_MODEL_PATH=/scratch/username/models/bert_model
BERT_SUMMARY_FILE=/scratch/username/results/summary.json
```

## Contact

For issues or questions about the models, refer to the main project documentation.
