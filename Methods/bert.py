import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
config = {
    'model_name': os.getenv('MODEL_NAME', 'bert-base-uncased'),
    'batch_size': int(os.getenv('BERT_BATCH_SIZE', 16)),
    'learning_rate': float(os.getenv('BERT_LEARNING_RATE', 2e-5)),
    'num_epochs': int(os.getenv('BERT_NUM_EPOCHS', 3)),
    'max_length': int(os.getenv('MAX_LENGTH', 512)),
    'test_size': float(os.getenv('TEST_SIZE', 0.2)),
    'random_state': int(os.getenv('RANDOM_STATE', 42)),
    'weight_decay': float(os.getenv('BERT_WEIGHT_DECAY', 0.01))
}

# File paths from .env
prompts_file = os.getenv('PROMPTS_FILE', '../prompts.csv')
results_file = os.getenv('RESULTS_FILE', '../results.csv')
model_path = os.getenv('BERT_MODEL_PATH', './bert_finetuned_model')
results_dir = os.getenv('BERT_RESULTS_DIR', './results')
logs_dir = os.getenv('BERT_LOGS_DIR', './logs')
summary_file = os.getenv('BERT_SUMMARY_FILE', './training_summary.json')
wandb_project = os.getenv('WANDB_PROJECT', 'bert-prompt-correctness')

# Initialize wandb
wandb.init(
    project=wandb_project,
    config=config,
    name="bert-base-finetuning"
)

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

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
val_dataset = Dataset.from_dict({'text': X_val, 'label': y_val})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

# Load tokenizer and model
print("\nLoading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=config['max_length'])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=results_dir,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    num_train_epochs=config['num_epochs'],
    weight_decay=config['weight_decay'],
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir=logs_dir,
    logging_steps=10,
    report_to='wandb',
    run_name='bert-base-finetuning'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
print("\nStarting training...")
train_result = trainer.train()

# Save model
print("\nSaving model...")
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)

# Get training history
train_history = trainer.state.log_history

# Extract final metrics
final_train_metrics = {}
final_val_metrics = {}

for log in reversed(train_history):
    if 'eval_loss' in log and not final_val_metrics:
        final_val_metrics = {
            'loss': log.get('eval_loss'),
            'accuracy': log.get('eval_accuracy'),
            'precision': log.get('eval_precision'),
            'recall': log.get('eval_recall'),
            'f1': log.get('eval_f1')
        }
    if 'loss' in log and 'eval_loss' not in log and not final_train_metrics:
        final_train_metrics = {
            'loss': log.get('loss')
        }

# Create comprehensive JSON summary
summary = {
    'model_configuration': {
        'model_name': config['model_name'],
        'num_labels': 2,
        'max_length': config['max_length']
    },
    'training_configuration': {
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'optimizer': 'AdamW',
        'weight_decay': config['weight_decay']
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
        'final_train_loss': float(final_train_metrics.get('loss', 0)) if final_train_metrics else None,
        'final_validation_metrics': {
            'loss': float(final_val_metrics.get('loss', 0)) if final_val_metrics else None,
            'accuracy': float(final_val_metrics.get('accuracy', 0)) if final_val_metrics else None,
            'precision': float(final_val_metrics.get('precision', 0)) if final_val_metrics else None,
            'recall': float(final_val_metrics.get('recall', 0)) if final_val_metrics else None,
            'f1': float(final_val_metrics.get('f1', 0)) if final_val_metrics else None
        }
    },
    'test_results': {
        'loss': float(test_results.get('eval_loss', 0)),
        'accuracy': float(test_results.get('eval_accuracy', 0)),
        'precision': float(test_results.get('eval_precision', 0)),
        'recall': float(test_results.get('eval_recall', 0)),
        'f1': float(test_results.get('eval_f1', 0))
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
artifact = wandb.Artifact('bert-finetuned-model', type='model')
artifact.add_dir(model_path)
wandb.log_artifact(artifact)

# Finish wandb run
wandb.finish()

print("\nTraining complete!")
