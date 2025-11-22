# Setting Up Conda Environment for This Project

This guide shows you how to set up a conda environment from the command line for running the BERT, XGBoost, and ANN models.

## Quick Setup (Copy-Paste Commands)

```bash
# 1. Create a new conda environment with Python 3.10
conda create -n promptenv python=3.10 -y

# 2. Activate the environment
conda activate promptenv

# 3. Install PyTorch with CUDA support (for GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install machine learning libraries
pip install transformers datasets wandb scikit-learn xgboost

# 5. Install additional utilities
pip install pandas numpy tqdm python-dotenv

# 6. Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step-by-Step Explanation

### 1. Create the Conda Environment

```bash
conda create -n promptenv python=3.10 -y
```

**What this does:**
- `-n promptenv`: Names your environment "promptenv"
- `python=3.10`: Installs Python 3.10 (compatible with all our dependencies)
- `-y`: Automatically answers "yes" to all prompts

### 2. Activate the Environment

```bash
conda activate promptenv
```

**What this does:**
- Switches your shell to use the `promptenv` environment
- All packages you install will be isolated to this environment
- Your prompt should change to show `(promptenv)` at the beginning

### 3. Install PyTorch with CUDA Support

```bash
# For CUDA 12.1 (most common on modern GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR for CUDA 11.8 (if your cluster uses older CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OR for CPU-only (if no GPU available)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**What this does:**
- Installs PyTorch (deep learning framework)
- Installs CUDA-enabled version for GPU acceleration
- `-c pytorch -c nvidia`: Uses official PyTorch and NVIDIA channels

### 4. Install HuggingFace and ML Libraries

```bash
pip install transformers datasets wandb scikit-learn xgboost
```

**What each package does:**
- `transformers`: HuggingFace library for BERT models
- `datasets`: HuggingFace datasets library
- `wandb`: Weights & Biases for experiment tracking
- `scikit-learn`: Machine learning utilities (metrics, train/test split)
- `xgboost`: XGBoost gradient boosting library

### 5. Install Data Processing Utilities

```bash
pip install pandas numpy tqdm python-dotenv
```

**What each package does:**
- `pandas`: Data manipulation (reading CSV files)
- `numpy`: Numerical operations
- `tqdm`: Progress bars for training loops
- `python-dotenv`: Load environment variables from .env file

### 6. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check all packages
python -c "import torch, transformers, xgboost, pandas, sklearn, wandb; print('All packages imported successfully!')"
```

## For MIT Engaging Cluster

On the MIT Engaging cluster, you'll need to load modules first:

```bash
# Load required modules
module purge
module load miniforge/24.3.0-0
module load cuda/12.4.0

# Create conda environment
conda create -n promptenv python=3.10 -y

# Activate environment
conda activate promptenv

# Install PyTorch (match CUDA version to loaded module)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install other packages
pip install transformers datasets wandb scikit-learn xgboost pandas numpy tqdm python-dotenv

# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Managing Your Environment

### List all conda environments
```bash
conda env list
```

### Activate the environment
```bash
conda activate promptenv
```

### Deactivate the environment
```bash
conda deactivate
```

### Export environment to file (for reproducibility)
```bash
conda env export > environment.yml
```

### Create environment from file
```bash
conda env create -f environment.yml
```

### Delete the environment (if you need to start over)
```bash
conda deactivate  # Make sure you're not in the environment
conda env remove -n promptenv
```

## Troubleshooting

### "conda: command not found"
Install Miniconda or Anaconda first:
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install it
bash Miniconda3-latest-Linux-x86_64.sh

# Restart your shell or run
source ~/.bashrc
```

### CUDA not available after installation
1. Check if you have CUDA installed: `nvidia-smi`
2. Verify CUDA version: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   # Find your CUDA version from nvidia-smi
   # Then install matching PyTorch version
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

### Package conflicts
Start fresh:
```bash
conda deactivate
conda env remove -n promptenv
conda create -n promptenv python=3.10 -y
conda activate promptenv
# Follow installation steps again
```

### Out of disk space in conda cache
```bash
conda clean --all -y
```

## Testing Your Setup

Create a test script to verify everything works:

```bash
cat > test_setup.py << 'EOF'
import torch
import transformers
import xgboost as xgb
import pandas as pd
import wandb
from dotenv import load_dotenv

print("=== Package Versions ===")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"Pandas: {pd.__version__}")

print("\n=== CUDA Info ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

print("\n=== Test BERT Model Loading ===")
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
print("✓ BERT model loaded successfully!")

print("\n=== All tests passed! ===")
EOF

python test_setup.py
```

## Next Steps

After setting up your environment:

1. **Configure wandb** (first time only):
   ```bash
   wandb login
   # Enter your API key when prompted
   ```

2. **Navigate to project directory**:
   ```bash
   cd "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project"
   ```

3. **Verify .env file exists**:
   ```bash
   ls -la .env
   ```

4. **Run a model locally to test**:
   ```bash
   cd Methods
   python bert.py  # or xgboost_model.py or ann_model.py
   ```

5. **Submit to cluster** (on MIT Engaging):
   ```bash
   sbatch run_bert.sh
   # or run_xgboost.sh, run_ann.sh, run_all.sh
   ```
