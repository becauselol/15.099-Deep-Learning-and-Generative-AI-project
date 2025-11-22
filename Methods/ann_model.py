import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse hidden sizes from .env
hidden_sizes_str = os.getenv('ANN_HIDDEN_SIZES', '256,128,64')
hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(',')]

# Configuration from .env
config = {
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
prompts_file = os.getenv('PROMPTS_FILE', '../prompts.csv')
results_file = os.getenv('RESULTS_FILE', '../results.csv')
model_path = os.getenv('ANN_MODEL_PATH', './ann_model_best.pth')
summary_file = os.getenv('ANN_SUMMARY_FILE', './ann_training_summary.json')
wandb_project = os.getenv('WANDB_PROJECT', 'bert-prompt-correctness')

# Initialize wandb
wandb.init(
    project=wandb_project,
    config=config,
    name="ann-bert-embeddings"
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

# Define simple ANN model
class SimpleANN(nn.Module):
    def __init__(self, input_size=768, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(SimpleANN, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Create model
print("\nCreating ANN model...")
ann_model = SimpleANN(
    input_size=768,
    hidden_sizes=config['ann_params']['hidden_sizes'],
    dropout=config['ann_params']['dropout']
)
ann_model.to(device)

print(ann_model)

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_emb),
    torch.LongTensor(y_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_emb),
    torch.LongTensor(y_val)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_emb),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['ann_params']['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['ann_params']['batch_size'],
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['ann_params']['batch_size'],
    shuffle=False
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ann_model.parameters(), lr=config['ann_params']['learning_rate'])

# Training loop
print("\nStarting training...")
best_val_f1 = 0
global_step = 0

for epoch in range(config['ann_params']['num_epochs']):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{config['ann_params']['num_epochs']}")
    print(f"{'='*50}")

    # Training phase
    ann_model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch_emb, batch_labels in progress_bar:
        batch_emb = batch_emb.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = ann_model(batch_emb)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(batch_labels.cpu().numpy())

        # Log batch metrics
        wandb.log({
            'batch_loss': loss.item(),
            'global_step': global_step,
            'epoch': epoch + 1
        })
        global_step += 1

        progress_bar.set_postfix({'loss': loss.item()})

    # Calculate training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average='binary'
    )

    print(f"\nTraining Loss: {avg_train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training F1: {train_f1:.4f}")

    # Validation phase
    ann_model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch_emb, batch_labels in val_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            outputs = ann_model(batch_emb)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(batch_labels.cpu().numpy())

    # Calculate validation metrics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='binary'
    )

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1: {val_f1:.4f}")

    # Log epoch metrics
    wandb.log({
        'epoch': epoch + 1,
        'train/loss': avg_train_loss,
        'train/accuracy': train_accuracy,
        'train/precision': train_precision,
        'train/recall': train_recall,
        'train/f1': train_f1,
        'val/loss': avg_val_loss,
        'val/accuracy': val_accuracy,
        'val/precision': val_precision,
        'val/recall': val_recall,
        'val/f1': val_f1
    })

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(ann_model.state_dict(), model_path)
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
    for batch_emb, batch_labels in test_loader:
        batch_emb = batch_emb.to(device)
        batch_labels = batch_labels.to(device)

        outputs = ann_model(batch_emb)
        loss = criterion(outputs, batch_labels)

        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(batch_labels.cpu().numpy())

# Calculate test metrics
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

# Create comprehensive JSON summary
summary = {
    'model_type': 'Simple ANN with BERT Embeddings',
    'model_configuration': {
        'bert_model': config['model_name'],
        'embedding_dim': 768,
        'max_length': config['max_length'],
        'hidden_sizes': config['ann_params']['hidden_sizes'],
        'dropout': config['ann_params']['dropout'],
        'learning_rate': config['ann_params']['learning_rate'],
        'batch_size': config['ann_params']['batch_size'],
        'num_epochs': config['ann_params']['num_epochs']
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
artifact = wandb.Artifact('ann-model', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

# Finish wandb run
wandb.finish()

print("\nTraining complete!")
