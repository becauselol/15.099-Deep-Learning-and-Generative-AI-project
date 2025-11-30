import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from tqdm import tqdm
import wandb
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
FEATURE_MODE = os.getenv('FEATURE_MODE', 'text_only')  # text_only, features_only, inst_only, prompt_only, text_prompt, all
USE_BERT_EMBEDDINGS = os.getenv('USE_BERT_EMBEDDINGS', 'true').lower() == 'true'

config = {
    'feature_mode': FEATURE_MODE,
    'use_bert_embeddings': USE_BERT_EMBEDDINGS,
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
train_data_file = os.getenv('TRAIN_DATA_FILE', '../data_gen/train_countdown_results_with_prompt_gemini.csv')
test_data_file = os.getenv('TEST_DATA_FILE', '../data_gen/test_countdown_results_with_prompt_gemini.csv')
model_path = os.getenv('XGBOOST_MODEL_PATH', f'./xgboost_model_{FEATURE_MODE}.json')
summary_file = os.getenv('XGBOOST_SUMMARY_FILE', f'./xgboost_summary_{FEATURE_MODE}.json')
wandb_project = os.getenv('WANDB_PROJECT', 'countdown-xgboost')

print("=" * 70)
print("XGBOOST UNIFIED TRAINING")
print("=" * 70)
print(f"Feature Mode: {FEATURE_MODE}")
print(f"Use BERT Embeddings: {USE_BERT_EMBEDDINGS}")
print(f"Train Data File: {train_data_file}")
print(f"Test Data File: {test_data_file}")
print("=" * 70)
print()

# Initialize wandb
wandb.init(
    project=wandb_project,
    config=config,
    name=f"xgboost-{FEATURE_MODE}"
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load train and test data
print("Loading train data...")
train_df = pd.read_csv(train_data_file)
print(f"Train samples: {len(train_df)}")

print("Loading test data...")
test_df = pd.read_csv(test_data_file)
print(f"Test samples: {len(test_df)}")

# Validate that both datasets have the same columns
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
if train_cols != test_cols:
    raise ValueError(f"Column mismatch between train and test datasets!\n"
                     f"Train only: {train_cols - test_cols}\n"
                     f"Test only: {test_cols - train_cols}")
print(f"✓ Train and test datasets have matching columns")
print(f"Total samples: {len(train_df) + len(test_df)}")

# Prepare features based on mode
def extract_features(df, dataset_name="dataset"):
    """Extract features from a dataframe based on the feature mode."""
    X_features_list = []
    feature_names = []

    if FEATURE_MODE in ['text_only', 'text_prompt', 'all']:
        # Extract BERT embeddings for text
        if USE_BERT_EMBEDDINGS:
            if dataset_name == "train":
                print(f"\nExtracting BERT embeddings for {dataset_name} data...")
            text_embeddings = get_bert_embeddings(df['prompt'].values, desc=f"{dataset_name} BERT embeddings")
            X_features_list.append(text_embeddings)
            if dataset_name == "train":  # Only set feature names once
                feature_names.extend([f'bert_emb_{i}' for i in range(text_embeddings.shape[1])])
            print(f"{dataset_name} text embeddings shape: {text_embeddings.shape}")

    if FEATURE_MODE in ['features_only', 'inst_only', 'all']:
        # Add instance features
        inst_cols = [col for col in df.columns if col.startswith('inst_')]
        if inst_cols:
            inst_features = df[inst_cols].fillna(0).values
            X_features_list.append(inst_features)
            if dataset_name == "train":  # Only set feature names once
                feature_names.extend(inst_cols)
            print(f"{dataset_name} instance features shape: {inst_features.shape}")

    if FEATURE_MODE in ['features_only', 'prompt_only', 'text_prompt', 'all']:
        # Add prompt features
        prompt_cols = [col for col in df.columns if col.startswith('prompt_')]
        if prompt_cols:
            prompt_features = df[prompt_cols].fillna(0).values
            X_features_list.append(prompt_features)
            if dataset_name == "train":  # Only set feature names once
                feature_names.extend(prompt_cols)
            print(f"{dataset_name} prompt features shape: {prompt_features.shape}")

    # Combine all features
    if len(X_features_list) == 0:
        raise ValueError(f"No features selected for mode: {FEATURE_MODE}")

    X = np.hstack(X_features_list) if len(X_features_list) > 1 else X_features_list[0]
    y = df['correct'].values.astype(int)

    return X, y, feature_names

# Initialize BERT model if needed
if FEATURE_MODE in ['text_only', 'text_prompt', 'all'] and USE_BERT_EMBEDDINGS:
    print("\nLoading BERT model for embeddings...")
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    bert_model = BertModel.from_pretrained(config['model_name'])
    bert_model.to(device)
    bert_model.eval()

    def get_bert_embeddings(texts, batch_size=16, desc="Extracting BERT embeddings"):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts.tolist() if hasattr(batch_texts, 'tolist') else list(batch_texts),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=config['max_length']
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

# Extract features from train data
X_train_val, y_train_val, feature_names = extract_features(train_df, "train")

# Extract features from test data
X_test, y_test, _ = extract_features(test_df, "test")

print(f"\nTrain feature matrix shape: {X_train_val.shape}")
print(f"Test feature matrix shape: {X_test.shape}")
print(f"Total features: {X_train_val.shape[1]}")
print(f"Train - Positive: {sum(y_train_val)}, Negative: {len(y_train_val) - sum(y_train_val)}")
print(f"Test - Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")

# Log dataset statistics
wandb.log({
    'total_samples': len(X_train_val) + len(X_test),
    'num_features': X_train_val.shape[1],
    'positive_samples': int(sum(y_train_val) + sum(y_test)),
    'negative_samples': int((len(y_train_val) - sum(y_train_val)) + (len(y_test) - sum(y_test))),
    'class_balance': (sum(y_train_val) + sum(y_test)) / (len(y_train_val) + len(y_test))
})

# Further split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1,
    random_state=config['random_state'],
    stratify=y_train_val
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(**config['xgb_params'], random_state=config['random_state'])

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Evaluate
print("\nEvaluating on training set...")
y_train_pred = xgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
    y_train, y_train_pred, average='binary'
)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training F1: {train_f1:.4f}")

print("\nEvaluating on validation set...")
y_val_pred = xgb_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val, y_val_pred, average='binary'
)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1: {val_f1:.4f}")

print("\nEvaluating on test set...")
y_test_pred = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='binary'
)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1: {test_f1:.4f}")

# Log metrics
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

# Create summary
summary = {
    'model_type': 'XGBoost with configurable features',
    'feature_mode': FEATURE_MODE,
    'use_bert_embeddings': USE_BERT_EMBEDDINGS,
    'model_configuration': {
        'bert_model': config['model_name'] if USE_BERT_EMBEDDINGS else None,
        'num_features': X_train_val.shape[1],
        'xgb_params': config['xgb_params']
    },
    'dataset_statistics': {
        'total_samples': len(X_train_val) + len(X_test),
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'positive_samples': int(sum(y_train_val) + sum(y_test)),
        'negative_samples': int((len(y_train_val) - sum(y_train_val)) + (len(y_test) - sum(y_test))),
        'class_balance': float((sum(y_train_val) + sum(y_test)) / (len(y_train_val) + len(y_test)))
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

# Save summary
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to '{summary_file}'")

# Log summary
wandb.log({"final_summary": summary})

# Save model as artifact
artifact = wandb.Artifact(f'xgboost-model-{FEATURE_MODE}', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()

print("\nTraining complete!")
