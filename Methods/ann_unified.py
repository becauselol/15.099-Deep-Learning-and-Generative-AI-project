import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wandb
import json
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
FEATURE_MODE = os.getenv('FEATURE_MODE', 'text_only')  # text_only, features_only, inst_only, prompt_only, text_prompt, all
USE_BERT_EMBEDDINGS = os.getenv('USE_BERT_EMBEDDINGS', 'true').lower() == 'true'

# Parse hidden sizes from .env
hidden_sizes_str = os.getenv('ANN_HIDDEN_SIZES', '256,128,64')
hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(',')]

config = {
    'feature_mode': FEATURE_MODE,
    'use_bert_embeddings': USE_BERT_EMBEDDINGS,
    'model_name': os.getenv('MODEL_NAME', 'bert-base-uncased'),
    'max_length': int(os.getenv('MAX_LENGTH', 512)),
    'test_size': float(os.getenv('TEST_SIZE', 0.2)),
    'random_state': int(os.getenv('RANDOM_STATE', 42)),
    'ann_params': {
        'hidden_sizes': hidden_sizes,
        'dropout': float(os.getenv('ANN_DROPOUT', 0.3)),
        'learning_rate': float(os.getenv('ANN_LEARNING_RATE', 0.001)),
        'batch_size': int(os.getenv('ANN_BATCH_SIZE', 32)),
        'num_epochs': int(os.getenv('ANN_NUM_EPOCHS', 20))
    }
}

# File paths from .env
train_data_file = os.getenv('TRAIN_DATA_FILE', '../data_gen/train_countdown_results_gemini_done.csv')
test_data_file = os.getenv('TEST_DATA_FILE', '../data_gen/test_countdown_results_gemini_done.csv')
model_path = os.getenv('ANN_MODEL_PATH', f'./ann_model_{FEATURE_MODE}_best.pth')
summary_file = os.getenv('ANN_SUMMARY_FILE', f'./ann_summary_{FEATURE_MODE}.json')
wandb_project = os.getenv('WANDB_PROJECT', 'countdown-ann')

print("=" * 70)
print("ANN UNIFIED TRAINING")
print("=" * 70)
print(f"Feature Mode: {FEATURE_MODE}")
print(f"Use BERT Embeddings: {USE_BERT_EMBEDDINGS}")
print(f"Train Data File: {train_data_file}")
print(f"Test Data File: {test_data_file}")
print("=" * 70)
print()

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print()

# Initialize wandb
wandb.init(
    project=wandb_project,
    config=config,
    name=f"ann-{FEATURE_MODE}"
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

# Identify which features to normalize
# Binary/categorical features that should NOT be normalized
binary_categorical_features = [
    'inst_count_small', 'inst_count_large', 'inst_count_duplicates',
    'inst_count_even', 'inst_count_odd', 'inst_count_div_2', 'inst_count_div_3',
    'inst_count_div_5', 'inst_count_div_7', 'inst_count_primes',
    'inst_easy_pairs', 'inst_count_add', 'inst_count_sub', 'inst_count_mul',
    'inst_count_div', 'inst_noncomm_ops',
    'prompt_paraphrasing', 'prompt_role-specification', 'prompt_reasoning-trigger',
    'prompt_chain-of-thought', 'prompt_self-check', 'prompt_conciseness',
    'prompt_verbosity', 'prompt_context-expansion', 'prompt_few-shot-count'
]

# Continuous features that SHOULD be normalized
continuous_features = [
    'inst_n_numbers', 'inst_range', 'inst_std', 'inst_distance_simple',
    'inst_distance_max', 'inst_distance_avg', 'inst_log_target',
    'inst_expr_depth', 'prompt_length'
]

# Identify which columns correspond to which feature types
normalize_mask = np.zeros(len(feature_names), dtype=bool)
for i, fname in enumerate(feature_names):
    # BERT embeddings should be normalized
    if fname.startswith('bert_emb_'):
        normalize_mask[i] = True
    # Continuous features should be normalized
    elif any(fname == cf for cf in continuous_features):
        normalize_mask[i] = True
    # Binary/categorical features should NOT be normalized
    # (keep normalize_mask[i] = False)

print(f"\nNormalizing features...")
print(f"  Features to normalize: {normalize_mask.sum()} / {len(feature_names)}")
print(f"  Features to keep as-is: {(~normalize_mask).sum()} / {len(feature_names)}")

# Apply selective normalization
scaler = StandardScaler()
if normalize_mask.sum() > 0:
    X_train_val_normalized = X_train_val.copy()
    X_test_normalized = X_test.copy()

    # Normalize only selected features
    X_train_val_normalized[:, normalize_mask] = scaler.fit_transform(X_train_val[:, normalize_mask])
    X_test_normalized[:, normalize_mask] = scaler.transform(X_test[:, normalize_mask])

    X_train_val = X_train_val_normalized
    X_test = X_test_normalized
    print("✓ Continuous features normalized (mean=0, std=1)")
    print("✓ Binary/categorical features kept as-is")
else:
    scaler = None
    print("✓ No features to normalize (all binary/categorical)")

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

# Define ANN model
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(SimpleANN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Create model
print("\nCreating ANN model...")
input_size = X_train_val.shape[1]
ann_model = SimpleANN(
    input_size=input_size,
    hidden_sizes=config['ann_params']['hidden_sizes'],
    dropout=config['ann_params']['dropout']
)
ann_model.to(device)
print(ann_model)

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.LongTensor(y_val)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=config['ann_params']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['ann_params']['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config['ann_params']['batch_size'], shuffle=False)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ann_model.parameters(), lr=config['ann_params']['learning_rate'])

# Training loop
print("\nStarting training...")
best_val_f1 = -1
global_step = 0

for epoch in range(config['ann_params']['num_epochs']):
    print(f"\nEpoch {epoch + 1}/{config['ann_params']['num_epochs']}")

    # Training phase
    ann_model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for batch_features, batch_labels in tqdm(train_loader, desc='Training'):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = ann_model(batch_features)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(batch_labels.cpu().numpy())

        wandb.log({'batch_loss': loss.item(), 'global_step': global_step, 'epoch': epoch + 1})
        global_step += 1

    # Calculate training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average='binary'
    )

    print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

    # Validation phase
    ann_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = ann_model(batch_features)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(batch_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='binary'
    )

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

    # Log epoch metrics
    wandb.log({
        'epoch': epoch + 1,
        'train/loss': avg_train_loss,
        'train/accuracy': train_accuracy,
        'train/f1': train_f1,
        'val/loss': avg_val_loss,
        'val/accuracy': val_accuracy,
        'val/f1': val_f1
    })

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(ann_model.state_dict(), model_path)
        # Save scaler and normalization info alongside model
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'normalize_mask': normalize_mask,
                'feature_names': feature_names
            }, f)
        print(f"Best model saved with F1: {best_val_f1:.4f}")

# Load best model for test evaluation
print("\nLoading best model for test evaluation...")
ann_model.load_state_dict(torch.load(model_path))

# Test evaluation
print("\nEvaluating on test set...")
ann_model.eval()
test_loss = 0
test_preds = []
test_labels = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = ann_model(batch_features)
        loss = criterion(outputs, batch_labels)

        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(batch_labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(test_labels, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average='binary'
)

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")

# Log test metrics to wandb
wandb.log({
    'test/loss': avg_test_loss,
    'test/accuracy': test_accuracy,
    'test/precision': test_precision,
    'test/recall': test_recall,
    'test/f1': test_f1
})

# Create summary
summary = {
    'model_type': 'ANN with configurable features',
    'feature_mode': FEATURE_MODE,
    'use_bert_embeddings': USE_BERT_EMBEDDINGS,
    'model_configuration': {
        'bert_model': config['model_name'] if USE_BERT_EMBEDDINGS else None,
        'input_size': input_size,
        'hidden_sizes': config['ann_params']['hidden_sizes'],
        'dropout': config['ann_params']['dropout'],
        'learning_rate': config['ann_params']['learning_rate'],
        'batch_size': config['ann_params']['batch_size'],
        'num_epochs': config['ann_params']['num_epochs'],
        'feature_normalization': 'StandardScaler (selective)',
        'normalized_features_count': int(normalize_mask.sum()),
        'unnormalized_features_count': int((~normalize_mask).sum())
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
        'best_validation_f1': float(best_val_f1)
    },
    'test_results': {
        'loss': float(avg_test_loss),
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
artifact = wandb.Artifact(f'ann-model-{FEATURE_MODE}', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()

print("\nTraining complete!")
