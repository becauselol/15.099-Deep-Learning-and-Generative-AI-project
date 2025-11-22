import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import wandb

# Configuration
config = {
    'model_name': 'bert-base-uncased',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'max_length': 512,
    'test_size': 0.2,
    'random_state': 42
}

# Initialize wandb
wandb.init(
    project="bert-prompt-correctness",
    config=config,
    name="bert-base-finetuning"
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
wandb.config.update({'device': str(device)})

# Load data
print("Loading data...")
prompts_df = pd.read_csv('../prompts.csv')
results_df = pd.read_csv('../results.csv')

# Merge datasets on prompt_id and id
print("Merging datasets...")
data = results_df.merge(prompts_df, left_on='prompt_id', right_on='id', suffixes=('_result', '_prompt'))

# Create features and labels
# Use the prompt text as input and correct (True/False) as label
X = data['prompt'].values
y = data['correct'].values.astype(int)  # Convert True/False to 1/0

print(f"Total samples: {len(X)}")
print(f"Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")

# Log dataset statistics to wandb
wandb.log({
    'total_samples': len(X),
    'positive_samples': int(sum(y)),
    'negative_samples': int(len(y) - sum(y)),
    'class_balance': sum(y) / len(y)
})

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=config['test_size'],
    random_state=config['random_state'],
    stratify=y
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Initialize BERT tokenizer and model
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(config['model_name'])
model = BertForSequenceClassification.from_pretrained(config['model_name'], num_labels=2)
model.to(device)

# Watch model with wandb
wandb.watch(model, log='all', log_freq=10)

# Create custom dataset class
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=None):
        if max_length is None:
            max_length = config['max_length']
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = PromptDataset(X_train, y_train, tokenizer)
val_dataset = PromptDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# Training setup
optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
num_epochs = config['num_epochs']

print(f"\nStarting training for {num_epochs} epochs...")

# Training loop
global_step = 0
for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"{'='*50}")

    # Training phase
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

        # Log batch metrics to wandb
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

    # Validation phase
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')

    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    # Log epoch metrics to wandb
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

# Save the model
print("\nSaving model...")
model.save_pretrained('./bert_finetuned_model')
tokenizer.save_pretrained('./bert_finetuned_model')
print("Model saved to './bert_finetuned_model'")

# Save model as wandb artifact
artifact = wandb.Artifact('bert-finetuned-model', type='model')
artifact.add_dir('./bert_finetuned_model')
wandb.log_artifact(artifact)

# Function to predict on new prompts
def predict(text, model, tokenizer, device):
    """
    Predict whether a response will be correct given a prompt.

    Args:
        text: The prompt text
        model: Trained BERT model
        tokenizer: BERT tokenizer
        device: torch device

    Returns:
        prediction: 0 (incorrect) or 1 (correct)
        probability: confidence score
    """
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

# Example usage
print("\n" + "="*50)
print("Example predictions:")
print("="*50)

sample_prompts = X_val[:3]
example_table = wandb.Table(columns=["Prompt (truncated)", "Predicted", "Confidence", "Actual", "Match"])
for i, prompt in enumerate(sample_prompts):
    pred, conf = predict(prompt, model, tokenizer, device)
    actual = y_val[i]
    print(f"\nPrompt {i+1} (truncated): {prompt[:100]}...")
    print(f"Predicted: {'Correct' if pred == 1 else 'Incorrect'} (confidence: {conf:.4f})")
    print(f"Actual: {'Correct' if actual == 1 else 'Incorrect'}")

    example_table.add_data(
        prompt[:100] + "...",
        'Correct' if pred == 1 else 'Incorrect',
        f"{conf:.4f}",
        'Correct' if actual == 1 else 'Incorrect',
        pred == actual
    )

wandb.log({"example_predictions": example_table})

# Finish wandb run
wandb.finish()
