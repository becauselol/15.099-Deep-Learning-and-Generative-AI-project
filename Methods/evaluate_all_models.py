#!/usr/bin/env python3
"""
Script to evaluate all trained models (all 6 feature combinations for both XGBoost and ANN)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import xgboost as xgb
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
FEATURE_MODES = ['text_only', 'features_only', 'inst_only', 'prompt_only', 'text_prompt', 'all']
DATA_FILE = os.getenv('DATA_FILE', '../data_gen/countdown_results_with_prompt_gemini.csv')
MODEL_NAME = os.getenv('MODEL_NAME', 'bert-base-uncased')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("EVALUATING ALL MODELS")
print("=" * 70)
print(f"Data file: {DATA_FILE}")
print(f"Device: {device}")
print("=" * 70)
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_FILE)
y_true = df['correct'].values.astype(int)
print(f"Total samples: {len(df)}")
print()

# ANN model definition
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

# Function to extract features
def extract_features(df, feature_mode, use_bert_embeddings=True):
    X_features_list = []

    if feature_mode in ['text_only', 'text_prompt', 'all']:
        if use_bert_embeddings:
            print(f"  Extracting BERT embeddings...")
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            bert_model = BertModel.from_pretrained(MODEL_NAME)
            bert_model.to(device)
            bert_model.eval()

            all_embeddings = []
            for i in tqdm(range(0, len(df), 16), desc="  BERT embeddings"):
                batch_texts = df['prompt'].values[i:i + 16]
                inputs = tokenizer(
                    batch_texts.tolist() if hasattr(batch_texts, 'tolist') else list(batch_texts),
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                    all_embeddings.append(embeddings)

            text_embeddings = np.vstack(all_embeddings)
            X_features_list.append(text_embeddings)

    if feature_mode in ['features_only', 'inst_only', 'all']:
        inst_cols = [col for col in df.columns if col.startswith('inst_')]
        if inst_cols:
            inst_features = df[inst_cols].fillna(0).values
            X_features_list.append(inst_features)

    if feature_mode in ['features_only', 'prompt_only', 'text_prompt', 'all']:
        prompt_cols = [col for col in df.columns if col.startswith('prompt_')]
        if prompt_cols:
            prompt_features = df[prompt_cols].fillna(0).values
            X_features_list.append(prompt_features)

    X = np.hstack(X_features_list) if len(X_features_list) > 1 else X_features_list[0]
    return X

# Evaluate a single model
def evaluate_model(model, X, y_true, model_type='xgboost'):
    if model_type == 'xgboost':
        y_pred = model.predict(X)
    else:  # ANN
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'predictions': y_pred
    }

# Results storage
results = {
    'xgboost': {},
    'ann': {}
}

# Evaluate XGBoost models
print("=" * 70)
print("EVALUATING XGBOOST MODELS")
print("=" * 70)
for mode in FEATURE_MODES:
    model_path = f'./xgboost_model_{mode}.json'

    if not os.path.exists(model_path):
        print(f"⚠ Skipping {mode}: Model not found at {model_path}")
        continue

    print(f"\nEvaluating XGBoost - {mode}")
    print("-" * 70)

    try:
        # Load model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Extract features
        X = extract_features(df, mode, use_bert_embeddings=True)
        print(f"  Feature shape: {X.shape}")

        # Evaluate
        metrics = evaluate_model(model, X, y_true, model_type='xgboost')
        results['xgboost'][mode] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['xgboost'][mode] = {'error': str(e)}

# Evaluate ANN models
print("\n" + "=" * 70)
print("EVALUATING ANN MODELS")
print("=" * 70)
for mode in FEATURE_MODES:
    model_path = f'./ann_model_{mode}_best.pth'

    if not os.path.exists(model_path):
        print(f"⚠ Skipping {mode}: Model not found at {model_path}")
        continue

    print(f"\nEvaluating ANN - {mode}")
    print("-" * 70)

    try:
        # Extract features
        X = extract_features(df, mode, use_bert_embeddings=True)
        print(f"  Feature shape: {X.shape}")

        # Load model
        input_size = X.shape[1]
        model = SimpleANN(input_size=input_size)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        # Evaluate
        metrics = evaluate_model(model, X, y_true, model_type='ann')
        results['ann'][mode] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['ann'][mode] = {'error': str(e)}

# Save results
print("\n" + "=" * 70)
print("SUMMARY OF ALL MODELS")
print("=" * 70)

# Create comparison table
comparison_data = []
for model_type in ['xgboost', 'ann']:
    for mode in FEATURE_MODES:
        if mode in results[model_type] and 'error' not in results[model_type][mode]:
            metrics = results[model_type][mode]
            comparison_data.append({
                'Model': model_type.upper(),
                'Feature Mode': mode,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1']
            })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # Save to CSV
    comparison_df.to_csv('./model_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to './model_comparison_results.csv'")

    # Find best models
    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)

    best_xgb = comparison_df[comparison_df['Model'] == 'XGBOOST'].sort_values('F1', ascending=False).iloc[0]
    best_ann = comparison_df[comparison_df['Model'] == 'ANN'].sort_values('F1', ascending=False).iloc[0]

    print(f"\nBest XGBoost Model:")
    print(f"  Feature Mode: {best_xgb['Feature Mode']}")
    print(f"  F1 Score: {best_xgb['F1']:.4f}")
    print(f"  Accuracy: {best_xgb['Accuracy']:.4f}")

    print(f"\nBest ANN Model:")
    print(f"  Feature Mode: {best_ann['Feature Mode']}")
    print(f"  F1 Score: {best_ann['F1']:.4f}")
    print(f"  Accuracy: {best_ann['Accuracy']:.4f}")

# Save detailed results to JSON
output_file = './evaluation_results_detailed.json'
# Remove predictions from JSON (too large)
results_for_json = {}
for model_type in results:
    results_for_json[model_type] = {}
    for mode in results[model_type]:
        if 'predictions' in results[model_type][mode]:
            results_for_json[model_type][mode] = {k: v for k, v in results[model_type][mode].items() if k != 'predictions'}
        else:
            results_for_json[model_type][mode] = results[model_type][mode]

with open(output_file, 'w') as f:
    json.dump(results_for_json, f, indent=2)

print(f"\n✓ Detailed results saved to '{output_file}'")
print("\nEvaluation complete!")
