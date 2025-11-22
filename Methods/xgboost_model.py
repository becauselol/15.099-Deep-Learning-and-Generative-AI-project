import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import wandb
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
config = {
    'model_name': os.getenv('MODEL_NAME', 'bert-base-uncased'),
    'max_length': int(os.getenv('MAX_LENGTH', 512)),
    'test_size': float(os.getenv('TEST_SIZE', 0.2)),
    'random_state': int(os.getenv('RANDOM_STATE', 42)),
    'xgb_params': {
        'objective': 'binary:logistic',
        'eta': float(os.getenv('XGB_ETA', 0.1)),
        'max_depth': int(os.getenv('XGB_MAX_DEPTH', 5)),
        'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', 100)),
        'eval_metric': 'logloss'
    }
}

# File paths from .env
prompts_file = os.getenv('PROMPTS_FILE', '../prompts.csv')
results_file = os.getenv('RESULTS_FILE', '../results.csv')
model_path = os.getenv('XGBOOST_MODEL_PATH', './xgboost_model.json')
summary_file = os.getenv('XGBOOST_SUMMARY_FILE', './xgboost_training_summary.json')
wandb_project = os.getenv('WANDB_PROJECT', 'bert-prompt-correctness')

# Initialize wandb
wandb.init(
    project=wandb_project,
    config=config,
    name="xgboost-bert-embeddings"
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and merge data
print("Loading data...")
print(f"Prompts file: {prompts_file}")
print(f"Results file: {results_file}")
prompts_df = pd.read_csv(prompts_file)
results_df = pd.read_csv(results_file)

print("Merging datasets...")
data = results_df.merge(prompts_df, left_on='prompt_id', right_on='id', suffixes=('_result', '_prompt'))

# Prepare data
X = data['prompt'].values
y = data['correct'].values.astype(int)

print(f"Total samples: {len(X)}")
print(f"Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")

# Log dataset statistics
wandb.log({
    'total_samples': len(X),
    'positive_samples': int(sum(y)),
    'negative_samples': int(len(y) - sum(y)),
    'class_balance': sum(y) / len(y)
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['test_size'],
    random_state=config['random_state'],
    stratify=y
)

# Further split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=config['random_state'],
    stratify=y_train
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Load BERT model for embeddings
print("\nLoading BERT model for embeddings...")
tokenizer = BertTokenizer.from_pretrained(config['model_name'])
bert_model = BertModel.from_pretrained(config['model_name'])
bert_model.to(device)
bert_model.eval()

# Function to extract BERT embeddings
def get_bert_embeddings(texts, tokenizer, bert_model, batch_size=16):
    """
    Extract BERT embeddings for a list of texts.
    Returns the mean of the last hidden state as embeddings.
    """
    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts.tolist() if hasattr(batch_texts, 'tolist') else list(batch_texts),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=config['max_length']
        )

        # Move to device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Use mean of last hidden state
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# Extract embeddings for all splits
print("\nExtracting BERT embeddings for training set...")
X_train_emb = get_bert_embeddings(X_train, tokenizer, bert_model)

print("Extracting BERT embeddings for validation set...")
X_val_emb = get_bert_embeddings(X_val, tokenizer, bert_model)

print("Extracting BERT embeddings for test set...")
X_test_emb = get_bert_embeddings(X_test, tokenizer, bert_model)

print(f"\nEmbedding shape: {X_train_emb.shape}")

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(**config['xgb_params'], random_state=config['random_state'])

# Fit with evaluation set
xgb_model.fit(
    X_train_emb, y_train,
    eval_set=[(X_val_emb, y_val)],
    verbose=True
)

# Get training predictions
print("\nEvaluating on training set...")
y_train_pred = xgb_model.predict(X_train_emb)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
    y_train, y_train_pred, average='binary'
)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1: {train_f1:.4f}")

# Get validation predictions
print("\nEvaluating on validation set...")
y_val_pred = xgb_model.predict(X_val_emb)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val, y_val_pred, average='binary'
)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1: {val_f1:.4f}")

# Get test predictions
print("\nEvaluating on test set...")
y_test_pred = xgb_model.predict(X_test_emb)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='binary'
)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")

# Log metrics to wandb
wandb.log({
    'train/accuracy': train_accuracy,
    'train/precision': train_precision,
    'train/recall': train_recall,
    'train/f1': train_f1,
    'val/accuracy': val_accuracy,
    'val/precision': val_precision,
    'val/recall': val_recall,
    'val/f1': val_f1,
    'test/accuracy': test_accuracy,
    'test/precision': test_precision,
    'test/recall': test_recall,
    'test/f1': test_f1
})

# Save model
print("\nSaving model...")
xgb_model.save_model(model_path)

# Create comprehensive JSON summary
summary = {
    'model_type': 'XGBoost with BERT Embeddings',
    'model_configuration': {
        'bert_model': config['model_name'],
        'embedding_dim': 768,
        'max_length': config['max_length'],
        'xgb_params': config['xgb_params']
    },
    'dataset_statistics': {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'positive_samples': int(sum(y)),
        'negative_samples': int(len(y) - sum(y)),
        'class_balance': float(sum(y) / len(y))
    },
    'training_results': {
        'train_metrics': {
            'accuracy': float(train_accuracy),
            'precision': float(train_precision),
            'recall': float(train_recall),
            'f1': float(train_f1)
        },
        'validation_metrics': {
            'accuracy': float(val_accuracy),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1)
        }
    },
    'test_results': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1': float(test_f1)
    },
    'model_path': model_path
}

# Print JSON summary
print("\n" + "="*70)
print("TRAINING SUMMARY (JSON)")
print("="*70)
print(json.dumps(summary, indent=2))

# Save JSON to file
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to '{summary_file}'")

# Log summary to wandb
wandb.log({"final_summary": summary})

# Save model as wandb artifact
artifact = wandb.Artifact('xgboost-model', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

# Finish wandb run
wandb.finish()

print("\nTraining complete!")
