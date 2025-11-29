import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read the results CSV file
results_csv_path = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_gen/countdown_results_gemini_done.csv"
features_csv_path = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_gen/countdown_features.csv"
prompts_csv_path = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_gen/transformed_countdown_prompts_gemini.csv"

print("="*80)
print("COUNTDOWN RESULTS ANALYSIS - MODEL PERFORMANCE")
print("="*80)
print("\nLoading data...")

df_results = pd.read_csv(results_csv_path)
df_features = pd.read_csv(features_csv_path)
df_prompts = pd.read_csv(prompts_csv_path)

print(f"Total results: {len(df_results):,}")
print(f"Total instances with features: {len(df_features):,}")
print(f"Total prompts with features: {len(df_prompts):,}")

# Create nested output directories
base_output_dir = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_analysis"
output_dir = os.path.join(base_output_dir, "results_analysis")
csv_dir = os.path.join(output_dir, "csv")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Save high-level dataset statistics to JSON (will be updated after accuracy calculation)
import json

# ============================================================================
# 1. OVERALL ACCURACY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. OVERALL ACCURACY ANALYSIS")
print("="*80)

# Check unique values in correct column
print("\nUnique values in 'correct' column:")
print(df_results['correct'].unique())

print("\nValue counts for 'correct' column:")
correct_counts = df_results['correct'].value_counts().sort_index()
print(correct_counts)

# Calculate accuracy
total_attempts = len(df_results)
correct_attempts = (df_results['correct'] == 1.0).sum()
incorrect_attempts = (df_results['correct'] == 0.0).sum()
accuracy = (correct_attempts / total_attempts) * 100

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {accuracy:.2f}%")
print(f"{'='*60}")
print(f"Correct answers: {correct_attempts:,} / {total_attempts:,}")
print(f"Incorrect answers: {incorrect_attempts:,} / {total_attempts:,}")

# Save overall accuracy stats
accuracy_stats = pd.DataFrame({
    'Metric': ['Total Attempts', 'Correct', 'Incorrect', 'Accuracy (%)'],
    'Value': [total_attempts, correct_attempts, incorrect_attempts, f"{accuracy:.2f}"]
})
accuracy_csv = os.path.join(csv_dir, "overall_accuracy.csv")
accuracy_stats.to_csv(accuracy_csv, index=False)
print(f"\n✓ Saved to: {accuracy_csv}")

# Save high-level dataset statistics to JSON
dataset_stats = {
    "dataset_name": "Countdown Results",
    "total_results": int(len(df_results)),
    "total_attempts": int(total_attempts),
    "correct_attempts": int(correct_attempts),
    "incorrect_attempts": int(incorrect_attempts),
    "overall_accuracy_pct": float(accuracy),
    "unique_prompts": int(df_results['prompt_id'].nunique()),
    "unique_instances": int(df_results['instance_id'].nunique()),
    "instances_with_features": int(len(df_features)),
    "prompts_with_features": int(len(df_prompts))
}
stats_json_path = os.path.join(output_dir, "dataset_stats.json")
with open(stats_json_path, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"✓ Dataset statistics saved to: {stats_json_path}")

# ============================================================================
# 2. ACCURACY BY PROMPT
# ============================================================================
print("\n" + "="*80)
print("2. ACCURACY BY PROMPT")
print("="*80)

# Group by prompt_id and calculate accuracy
prompt_accuracy = df_results.groupby('prompt_id').agg({
    'correct': ['count', 'sum', 'mean']
}).reset_index()
prompt_accuracy.columns = ['prompt_id', 'total_attempts', 'correct_count', 'accuracy']
prompt_accuracy['accuracy_pct'] = prompt_accuracy['accuracy'] * 100

print(f"\nNumber of unique prompts: {len(prompt_accuracy)}")
print(f"\nPrompt accuracy statistics:")
print(f"  Mean accuracy: {prompt_accuracy['accuracy_pct'].mean():.2f}%")
print(f"  Median accuracy: {prompt_accuracy['accuracy_pct'].median():.2f}%")
print(f"  Min accuracy: {prompt_accuracy['accuracy_pct'].min():.2f}%")
print(f"  Max accuracy: {prompt_accuracy['accuracy_pct'].max():.2f}%")
print(f"  Std dev: {prompt_accuracy['accuracy_pct'].std():.2f}%")

# Best and worst prompts
print("\nTop 10 Best Performing Prompts:")
best_prompts = prompt_accuracy.nlargest(10, 'accuracy_pct')[['prompt_id', 'total_attempts', 'correct_count', 'accuracy_pct']]
print(best_prompts.to_string(index=False))

print("\nTop 10 Worst Performing Prompts:")
worst_prompts = prompt_accuracy.nsmallest(10, 'accuracy_pct')[['prompt_id', 'total_attempts', 'correct_count', 'accuracy_pct']]
print(worst_prompts.to_string(index=False))

# Save prompt accuracy
prompt_accuracy_csv = os.path.join(csv_dir, "prompt_accuracy.csv")
prompt_accuracy.to_csv(prompt_accuracy_csv, index=False)
print(f"\n✓ Saved to: {prompt_accuracy_csv}")

# ============================================================================
# 3. ACCURACY BY INSTANCE
# ============================================================================
print("\n" + "="*80)
print("3. ACCURACY BY INSTANCE")
print("="*80)

# Group by instance_id and calculate accuracy
instance_accuracy = df_results.groupby('instance_id').agg({
    'correct': ['count', 'sum', 'mean']
}).reset_index()
instance_accuracy.columns = ['instance_id', 'total_attempts', 'correct_count', 'accuracy']
instance_accuracy['accuracy_pct'] = instance_accuracy['accuracy'] * 100

print(f"\nNumber of unique instances: {len(instance_accuracy)}")
print(f"\nInstance accuracy statistics:")
print(f"  Mean accuracy: {instance_accuracy['accuracy_pct'].mean():.2f}%")
print(f"  Median accuracy: {instance_accuracy['accuracy_pct'].median():.2f}%")
print(f"  Min accuracy: {instance_accuracy['accuracy_pct'].min():.2f}%")
print(f"  Max accuracy: {instance_accuracy['accuracy_pct'].max():.2f}%")
print(f"  Std dev: {instance_accuracy['accuracy_pct'].std():.2f}%")

# Easiest and hardest instances
print("\nTop 10 Easiest Instances (highest accuracy):")
easiest_instances = instance_accuracy.nlargest(10, 'accuracy_pct')[['instance_id', 'total_attempts', 'correct_count', 'accuracy_pct']]
print(easiest_instances.to_string(index=False))

print("\nTop 10 Hardest Instances (lowest accuracy):")
hardest_instances = instance_accuracy.nsmallest(10, 'accuracy_pct')[['instance_id', 'total_attempts', 'correct_count', 'accuracy_pct']]
print(hardest_instances.to_string(index=False))

# Save instance accuracy
instance_accuracy_csv = os.path.join(csv_dir, "instance_accuracy.csv")
instance_accuracy.to_csv(instance_accuracy_csv, index=False)
print(f"\n✓ Saved to: {instance_accuracy_csv}")

# ============================================================================
# 4. MERGE WITH FEATURES FOR DEEPER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. ACCURACY BY INSTANCE FEATURES")
print("="*80)

# Merge instance accuracy with features
instance_accuracy_merged = instance_accuracy.merge(
    df_features,
    left_on='instance_id',
    right_on='id',
    how='inner'
)

# Calculate complexity score (like in the features analysis)
instance_accuracy_merged['total_ops'] = (instance_accuracy_merged['count_add'] +
                                          instance_accuracy_merged['count_sub'] +
                                          instance_accuracy_merged['count_mul'] +
                                          instance_accuracy_merged['count_div'])
instance_accuracy_merged['complexity_score'] = (instance_accuracy_merged['expr_depth'] * 0.5 +
                                                 instance_accuracy_merged['total_ops'] * 0.3 +
                                                 instance_accuracy_merged['noncomm_ops'] * 0.2)

print(f"\nSuccessfully merged {len(instance_accuracy_merged)} instances with features")

# Analyze accuracy by number of numbers
print("\nAccuracy by Number Set Size:")
accuracy_by_n_numbers = instance_accuracy_merged.groupby('n_numbers')['accuracy_pct'].agg(['count', 'mean', 'std', 'min', 'max'])
print(accuracy_by_n_numbers)

# Analyze accuracy by expression depth
print("\nAccuracy by Expression Depth:")
accuracy_by_depth = instance_accuracy_merged.groupby('expr_depth')['accuracy_pct'].agg(['count', 'mean', 'std', 'min', 'max'])
print(accuracy_by_depth)

# Analyze accuracy by operator types
print("\nAccuracy by presence of operations:")
for op, op_name in [('count_add', 'Addition'), ('count_sub', 'Subtraction'),
                     ('count_mul', 'Multiplication'), ('count_div', 'Division')]:
    has_op = instance_accuracy_merged[instance_accuracy_merged[op] > 0]['accuracy_pct'].mean()
    no_op = instance_accuracy_merged[instance_accuracy_merged[op] == 0]['accuracy_pct'].mean()
    print(f"  {op_name:15s} - Has: {has_op:.2f}%, No: {no_op:.2f}%, Diff: {has_op - no_op:+.2f}%")

# Save merged data
merged_csv = os.path.join(csv_dir, "instance_accuracy_with_features.csv")
instance_accuracy_merged.to_csv(merged_csv, index=False)
print(f"\n✓ Saved to: {merged_csv}")

# ============================================================================
# 4B. ACCURACY BY PROMPT FEATURES
# ============================================================================
print("\n" + "="*80)
print("4B. ACCURACY BY PROMPT FEATURES")
print("="*80)

# Merge prompt accuracy with prompt features
prompt_accuracy_merged = prompt_accuracy.merge(
    df_prompts,
    left_on='prompt_id',
    right_on='id',
    how='inner'
)

print(f"\nSuccessfully merged {len(prompt_accuracy_merged)} prompts with features")

# Define prompt feature columns
prompt_feature_columns = [
    'paraphrasing', 'role-specification', 'reasoning-trigger',
    'chain-of-thought', 'self-check', 'conciseness',
    'verbosity', 'context-expansion', 'few-shot-count'
]

print("\nAccuracy by Prompt Feature Presence:")
prompt_feature_stats = []
for feature in prompt_feature_columns:
    if feature == 'few-shot-count':
        has_feature = prompt_accuracy_merged[prompt_accuracy_merged[feature] > 0]['accuracy_pct'].mean()
        no_feature = prompt_accuracy_merged[prompt_accuracy_merged[feature] == 0]['accuracy_pct'].mean()
        count_with = (prompt_accuracy_merged[feature] > 0).sum()
        count_without = (prompt_accuracy_merged[feature] == 0).sum()
    else:
        has_feature = prompt_accuracy_merged[prompt_accuracy_merged[feature] == 1]['accuracy_pct'].mean()
        no_feature = prompt_accuracy_merged[prompt_accuracy_merged[feature] == 0]['accuracy_pct'].mean()
        count_with = (prompt_accuracy_merged[feature] == 1).sum()
        count_without = (prompt_accuracy_merged[feature] == 0).sum()

    diff = has_feature - no_feature
    prompt_feature_stats.append({
        'Feature': feature,
        'Has_Feature_Acc': has_feature,
        'No_Feature_Acc': no_feature,
        'Difference': diff,
        'Count_With': count_with,
        'Count_Without': count_without
    })
    print(f"  {feature:20s} - Has: {has_feature:.2f}%, No: {no_feature:.2f}%, Diff: {diff:+.2f}% (n_with={count_with}, n_without={count_without})")

# Save prompt feature stats
prompt_feature_stats_df = pd.DataFrame(prompt_feature_stats)
prompt_feature_csv = os.path.join(csv_dir, "prompt_feature_accuracy.csv")
prompt_feature_stats_df.to_csv(prompt_feature_csv, index=False)
print(f"\n✓ Saved to: {prompt_feature_csv}")

# Save merged prompt data
prompt_merged_csv = os.path.join(csv_dir, "prompt_accuracy_with_features.csv")
prompt_accuracy_merged.to_csv(prompt_merged_csv, index=False)
print(f"✓ Saved to: {prompt_merged_csv}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 5.1 Overall accuracy pie chart
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
wedges, texts, autotexts = ax.pie(
    [correct_attempts, incorrect_attempts],
    labels=['Correct', 'Incorrect'],
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
ax.set_title(f'Overall Model Accuracy\n{accuracy:.2f}% Correct',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "overall_accuracy_pie.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: overall_accuracy_pie.png")

# 5.2 Prompt accuracy distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(prompt_accuracy['accuracy_pct'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Prompts', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Accuracy Across Different Prompts', fontsize=14, fontweight='bold')
ax.axvline(prompt_accuracy['accuracy_pct'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {prompt_accuracy["accuracy_pct"].mean():.2f}%')
ax.axvline(prompt_accuracy['accuracy_pct'].median(), color='green', linestyle='--',
           linewidth=2, label=f'Median: {prompt_accuracy["accuracy_pct"].median():.2f}%')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plot_path = os.path.join(plots_dir, "prompt_accuracy_distribution.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: prompt_accuracy_distribution.png")

# 5.3 Instance accuracy distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(instance_accuracy['accuracy_pct'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Accuracy Across Different Instances', fontsize=14, fontweight='bold')
ax.axvline(instance_accuracy['accuracy_pct'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {instance_accuracy["accuracy_pct"].mean():.2f}%')
ax.axvline(instance_accuracy['accuracy_pct'].median(), color='green', linestyle='--',
           linewidth=2, label=f'Median: {instance_accuracy["accuracy_pct"].median():.2f}%')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plot_path = os.path.join(plots_dir, "instance_accuracy_distribution.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: instance_accuracy_distribution.png")

# 5.4 Accuracy by number set size
fig, ax = plt.subplots(figsize=(10, 6))
n_numbers_acc = instance_accuracy_merged.groupby('n_numbers')['accuracy_pct'].mean()
bars = ax.bar(n_numbers_acc.index, n_numbers_acc.values, color='#3498db', edgecolor='black')
ax.set_xlabel('Number of Numbers in Instance', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Number Set Size', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_by_n_numbers.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_by_n_numbers.png")

# 5.5 Accuracy by expression depth
fig, ax = plt.subplots(figsize=(10, 6))
depth_acc = instance_accuracy_merged.groupby('expr_depth')['accuracy_pct'].mean()
bars = ax.bar(depth_acc.index, depth_acc.values, color='#e67e22', edgecolor='black')
ax.set_xlabel('Expression Tree Depth', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Expression Depth', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_by_depth.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_by_depth.png")

# 5.6 Accuracy by operator presence
fig, ax = plt.subplots(figsize=(12, 6))
operator_data = []
for op, op_name in [('count_add', 'Addition'), ('count_sub', 'Subtraction'),
                     ('count_mul', 'Multiplication'), ('count_div', 'Division')]:
    has_op = instance_accuracy_merged[instance_accuracy_merged[op] > 0]['accuracy_pct'].mean()
    no_op = instance_accuracy_merged[instance_accuracy_merged[op] == 0]['accuracy_pct'].mean()
    operator_data.append({'Operator': op_name, 'Has Op': has_op, 'No Op': no_op})

op_df = pd.DataFrame(operator_data)
x = np.arange(len(op_df))
width = 0.35
bars1 = ax.bar(x - width/2, op_df['Has Op'], width, label='Has Operator', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, op_df['No Op'], width, label='No Operator', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Operator Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Operator Presence', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(op_df['Operator'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_by_operator.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_by_operator.png")

# 5.7 Accuracy vs Complexity Score
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(instance_accuracy_merged['complexity_score'],
                     instance_accuracy_merged['accuracy_pct'],
                     alpha=0.5, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Complexity Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Instance Accuracy vs Complexity Score', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(instance_accuracy_merged['complexity_score'], instance_accuracy_merged['accuracy_pct'], 1)
p = np.poly1d(z)
x_trend = np.linspace(instance_accuracy_merged['complexity_score'].min(),
                      instance_accuracy_merged['complexity_score'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_vs_complexity.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_vs_complexity.png")

# 5.8 Accuracy vs Easy Pairs
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(instance_accuracy_merged['easy_pairs'],
                     instance_accuracy_merged['accuracy_pct'],
                     alpha=0.5, s=50, c='coral', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Easy Pairs', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Instance Accuracy vs Number of Easy Pairs', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_vs_easy_pairs.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_vs_easy_pairs.png")

# 5.9 Accuracy by Prompt Features
fig, ax = plt.subplots(figsize=(14, 6))
sorted_prompt_features = prompt_feature_stats_df.sort_values('Difference', ascending=False)
x = np.arange(len(sorted_prompt_features))
width = 0.35
bars1 = ax.bar(x - width/2, sorted_prompt_features['Has_Feature_Acc'], width,
               label='Has Feature', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, sorted_prompt_features['No_Feature_Acc'], width,
               label='No Feature', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Prompt Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy by Prompt Feature Presence (sorted by difference)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sorted_prompt_features['Feature'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_by_prompt_features.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_by_prompt_features.png")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations between accuracy and features
correlation_features = [
    'n_numbers', 'range', 'std', 'count_small', 'count_large',
    'count_primes', 'easy_pairs', 'expr_depth',
    'count_add', 'count_sub', 'count_mul', 'count_div',
    'noncomm_ops', 'log_target', 'distance_simple', 'complexity_score'
]

correlations = []
for feature in correlation_features:
    corr = instance_accuracy_merged['accuracy_pct'].corr(instance_accuracy_merged[feature])
    correlations.append({'Feature': feature, 'Correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
print("\nFeature Correlations with Accuracy (sorted by absolute value):")
print(corr_df.to_string(index=False))

# Save correlations
corr_csv = os.path.join(csv_dir, "accuracy_feature_correlations.csv")
corr_df.to_csv(corr_csv, index=False)
print(f"\n✓ Saved to: {corr_csv}")

# Correlation bar chart
fig, ax = plt.subplots(figsize=(12, 8))
colors_corr = ['#2ecc71' if x >= 0 else '#e74c3c' for x in corr_df['Correlation']]
bars = ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors_corr, edgecolor='black')
ax.set_xlabel('Correlation with Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Correlations with Model Accuracy', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "accuracy_correlations_chart.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: accuracy_correlations_chart.png")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("7. SUMMARY")
print("="*80)

summary_data = {
    'Category': [],
    'Metric': [],
    'Value': []
}

# Overall stats
summary_data['Category'].extend(['Overall'] * 3)
summary_data['Metric'].extend(['Total attempts', 'Correct answers', 'Overall accuracy'])
summary_data['Value'].extend([f"{total_attempts:,}", f"{correct_attempts:,}", f"{accuracy:.2f}%"])

# Prompt stats
summary_data['Category'].extend(['By Prompt'] * 3)
summary_data['Metric'].extend(['Unique prompts', 'Avg prompt accuracy', 'Std dev prompt accuracy'])
summary_data['Value'].extend([
    f"{len(prompt_accuracy):,}",
    f"{prompt_accuracy['accuracy_pct'].mean():.2f}%",
    f"{prompt_accuracy['accuracy_pct'].std():.2f}%"
])

# Instance stats
summary_data['Category'].extend(['By Instance'] * 3)
summary_data['Metric'].extend(['Unique instances', 'Avg instance accuracy', 'Std dev instance accuracy'])
summary_data['Value'].extend([
    f"{len(instance_accuracy):,}",
    f"{instance_accuracy['accuracy_pct'].mean():.2f}%",
    f"{instance_accuracy['accuracy_pct'].std():.2f}%"
])

# Top correlations
top_pos_corr = corr_df.iloc[0]
top_neg_corr = corr_df.iloc[-1]
summary_data['Category'].extend(['Correlations'] * 2)
summary_data['Metric'].extend(['Strongest positive correlation', 'Strongest negative correlation'])
summary_data['Value'].extend([
    f"{top_pos_corr['Feature']}: {top_pos_corr['Correlation']:.4f}",
    f"{top_neg_corr['Feature']}: {top_neg_corr['Correlation']:.4f}"
])

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

summary_csv = os.path.join(csv_dir, "results_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n✓ Saved summary to: {summary_csv}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  CSV files:")
print("    - overall_accuracy.csv")
print("    - prompt_accuracy.csv")
print("    - instance_accuracy.csv")
print("    - instance_accuracy_with_features.csv")
print("    - accuracy_feature_correlations.csv")
print("    - results_summary.csv")
print("\n  Visualizations:")
print("    - overall_accuracy_pie.png")
print("    - prompt_accuracy_distribution.png")
print("    - instance_accuracy_distribution.png")
print("    - accuracy_by_n_numbers.png")
print("    - accuracy_by_depth.png")
print("    - accuracy_by_operator.png")
print("    - accuracy_vs_complexity.png")
print("    - accuracy_vs_easy_pairs.png")
print("    - accuracy_correlations_chart.png")
