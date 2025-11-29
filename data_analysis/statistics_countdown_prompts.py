import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
csv_path = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_gen/countdown_prompts_gemini_transformed.csv"
df = pd.read_csv(csv_path)

# Define the feature columns (excluding id, length, and prompt)
feature_columns = [
    'paraphrasing',
    'role-specification',
    'reasoning-trigger',
    'chain-of-thought',
    'self-check',
    'conciseness',
    'verbosity',
    'context-expansion',
    'few-shot-count'
]

# Calculate statistics for each feature
statistics = {}

for feature in feature_columns:
    if feature == 'few-shot-count':
        # For few-shot-count, we want to count occurrences of each value
        value_counts = df[feature].value_counts().sort_index()
        statistics[feature] = {
            'count': len(df),
            'mean': df[feature].mean(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            'max': df[feature].max(),
            'value_counts': value_counts.to_dict()
        }
    else:
        # For binary features (0 or 1)
        count_ones = (df[feature] == 1).sum()
        count_zeros = (df[feature] == 0).sum()
        percentage = (count_ones / len(df)) * 100

        statistics[feature] = {
            'count_used (1)': count_ones,
            'count_not_used (0)': count_zeros,
            'percentage_used': percentage
        }

# Create a DataFrame for the statistics (without mean, std, min, max columns)
stats_rows = []
for feature in feature_columns:
    if feature == 'few-shot-count':
        count_used = statistics[feature]['value_counts'].get(1, 0)
        count_not_used = statistics[feature]['value_counts'].get(0, 0)
        percentage = (count_used / statistics[feature]['count']) * 100
        stats_rows.append({
            'feature': feature,
            'count_used (1)': count_used,
            'count_not_used (0)': count_not_used,
            'percentage_used': percentage
        })
    else:
        stats_rows.append({
            'feature': feature,
            'count_used (1)': statistics[feature]['count_used (1)'],
            'count_not_used (0)': statistics[feature]['count_not_used (0)'],
            'percentage_used': statistics[feature]['percentage_used']
        })

stats_df = pd.DataFrame(stats_rows)

# Create nested output directories
base_output_dir = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_analysis"
output_dir = os.path.join(base_output_dir, "prompt_analysis")
csv_dir = os.path.join(output_dir, "csv")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Save high-level dataset statistics to JSON
import json
dataset_stats = {
    "dataset_name": "Countdown Prompts",
    "total_prompts": int(len(df)),
    "unique_prompt_ids": int(df['id'].nunique()),
    "features_tracked": len(feature_columns),
    "feature_names": feature_columns,
    "prompt_length": {
        "mean": float(df['length'].mean()),
        "median": float(df['length'].median()),
        "min": int(df['length'].min()),
        "max": int(df['length'].max()),
        "std": float(df['length'].std())
    }
}
stats_json_path = os.path.join(output_dir, "dataset_stats.json")
with open(stats_json_path, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"Dataset statistics saved to: {stats_json_path}\n")

# Save statistics to CSV
stats_output_path = os.path.join(csv_dir, "feature_statistics.csv")
stats_df.to_csv(stats_output_path, index=False)
print(f"Feature statistics saved to: {stats_output_path}")
print("\nFeature Statistics:")
print(stats_df.to_string(index=False))

# Save few-shot prompts to a separate CSV for inspection
few_shot_prompts = df[df['few-shot-count'] > 0]
if len(few_shot_prompts) > 0:
    few_shot_output_path = os.path.join(csv_dir, "few_shot_prompts.csv")
    few_shot_prompts.to_csv(few_shot_output_path, index=False)
    print(f"\nFew-shot prompts saved to: {few_shot_output_path}")
    print(f"Number of prompts with few-shot examples: {len(few_shot_prompts)}")

# Create histogram for prompt length
plt.figure(figsize=(12, 6))
plt.hist(df['length'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prompt Length (characters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prompt Lengths', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add statistics to the plot
mean_length = df['length'].mean()
median_length = df['length'].median()
plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
plt.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
plt.legend()

# Save histogram
histogram_path = os.path.join(plots_dir, "prompt_length_histogram.png")
plt.tight_layout()
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {histogram_path}")
plt.close()

# Print length statistics
print(f"\nPrompt Length Statistics:")
print(f"  Count: {len(df)}")
print(f"  Mean: {mean_length:.2f}")
print(f"  Median: {median_length:.2f}")
print(f"  Std Dev: {df['length'].std():.2f}")
print(f"  Min: {df['length'].min()}")
print(f"  Max: {df['length'].max()}")
print(f"  25th percentile: {df['length'].quantile(0.25):.2f}")
print(f"  75th percentile: {df['length'].quantile(0.75):.2f}")

# Additional analysis: feature combinations
print("\n" + "="*60)
print("Additional Analysis: Most Common Feature Combinations")
print("="*60)

# Create a feature combination column
df['feature_combo'] = df[feature_columns].apply(lambda row: tuple(row), axis=1)
top_combos = df['feature_combo'].value_counts().head(10)

print("\nTop 10 Most Common Feature Combinations:")
for i, (combo, count) in enumerate(top_combos.items(), 1):
    print(f"\n{i}. Count: {count} ({count/len(df)*100:.1f}%)")
    for feature, value in zip(feature_columns, combo):
        if value != 0:
            print(f"   - {feature}: {value}")

print("\n" + "="*60)
print("Analysis complete!")
