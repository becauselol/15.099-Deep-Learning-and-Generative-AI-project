import pandas as pd
import os

# Path to the results export folder (current directory)
results_dir = os.path.dirname(os.path.abspath(__file__))

# Read all metric files
accuracy_df = pd.read_csv(os.path.join(results_dir, "accuracy.csv"))
precision_df = pd.read_csv(os.path.join(results_dir, "precision.csv"))
recall_df = pd.read_csv(os.path.join(results_dir, "recall.csv"))
f1_df = pd.read_csv(os.path.join(results_dir, "f1.csv"))

print("="*80)
print("COMPILING RESULTS FROM EXPORT FOLDER")
print("="*80)
print(f"\nLoaded files:")
print(f"  - accuracy.csv: {len(accuracy_df)} rows")
print(f"  - precision.csv: {len(precision_df)} rows")
print(f"  - recall.csv: {len(recall_df)} rows")
print(f"  - f1.csv: {len(f1_df)} rows")

# Extract model names from column names (remove MIN/MAX columns and just keep the base metric)
def extract_model_names(df, metric_name):
    """Extract unique model names from column headers, excluding Step, MIN, and MAX columns"""
    model_columns = [col for col in df.columns if col != 'Step' and not col.endswith('__MIN') and not col.endswith('__MAX')]
    return model_columns

# Get model names from accuracy (they should be the same across all files)
model_names = extract_model_names(accuracy_df, 'accuracy')
print(f"\nFound {len(model_names)} models:")
for model in model_names:
    print(f"  - {model}")

# Create compiled results
compiled_results = []

# For each row (step)
for idx in range(len(accuracy_df)):
    step = accuracy_df.loc[idx, 'Step']

    # For each model
    for model_col in model_names:
        # Extract the model name (remove the " - test/accuracy" part)
        model_name = model_col.replace(' - test/accuracy', '')

        # Get corresponding column names for other metrics
        precision_col = model_col.replace('accuracy', 'precision')
        recall_col = model_col.replace('accuracy', 'recall')
        f1_col = model_col.replace('accuracy', 'f1')

        # Get values (skip if empty)
        acc_val = accuracy_df.loc[idx, model_col]
        if pd.isna(acc_val) or acc_val == '':
            continue

        prec_val = precision_df.loc[idx, precision_col] if precision_col in precision_df.columns else None
        recall_val = recall_df.loc[idx, recall_col] if recall_col in recall_df.columns else None
        f1_val = f1_df.loc[idx, f1_col] if f1_col in f1_df.columns else None

        # Add to compiled results (round to 4 decimal places)
        compiled_results.append({
            'model': model_name,
            'accuracy': round(float(acc_val), 4) if not pd.isna(acc_val) else None,
            'precision': round(float(prec_val), 4) if not pd.isna(prec_val) else None,
            'recall': round(float(recall_val), 4) if not pd.isna(recall_val) else None,
            'f1': round(float(f1_val), 4) if not pd.isna(f1_val) else None
        })

# Create DataFrame
compiled_df = pd.DataFrame(compiled_results)

# Pivot to have models as rows and metrics as columns (one row per model, ignoring step)
# Since we don't need step, let's group by model and take the last non-null value
final_df = compiled_df.groupby('model').last().reset_index()

# Reorder columns
final_df = final_df[['model', 'accuracy', 'precision', 'recall', 'f1']]

# Sort by accuracy descending
final_df = final_df.sort_values('accuracy', ascending=False)

print("\n" + "="*80)
print("COMPILED RESULTS")
print("="*80)
print("\n" + final_df.to_string(index=False))

# Save to CSV
output_path = os.path.join(results_dir, "compiled_results.csv")
final_df.to_csv(output_path, index=False)
print(f"\n✓ Compiled results saved to: {output_path}")

# Also create a detailed version with all steps
detailed_output_path = os.path.join(results_dir, "compiled_results_detailed.csv")
compiled_df.to_csv(detailed_output_path, index=False)
print(f"✓ Detailed results (with steps) saved to: {detailed_output_path}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal unique models: {final_df['model'].nunique()}")
print(f"\nTop 5 models by accuracy:")
print(final_df.head(5)[['model', 'accuracy']].to_string(index=False))

print(f"\nTop 5 models by F1 score:")
top_f1 = final_df.sort_values('f1', ascending=False).head(5)
print(top_f1[['model', 'f1']].to_string(index=False))

print("\n" + "="*80)
print("COMPILATION COMPLETE!")
print("="*80)
