import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read the CSV file
csv_path = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_gen/countdown_features.csv"
df = pd.read_csv(csv_path)

# Create nested output directories
base_output_dir = "/home/becauselol/MIT-2025-Fall-Homework/15.099 Homework/Project/data_analysis"
output_dir = os.path.join(base_output_dir, "instance_features_analysis")
csv_dir = os.path.join(output_dir, "csv")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print("="*80)
print("COUNTDOWN INSTANCES FEATURE ANALYSIS")
print("="*80)
print(f"\nTotal instances: {len(df)}")

# Save high-level dataset statistics to JSON
import json
dataset_stats = {
    "dataset_name": "Countdown Instances Features",
    "total_instances": int(len(df)),
    "unique_instance_ids": int(df['id'].nunique()),
    "number_set_sizes": {
        "3_numbers": int((df['n_numbers'] == 3).sum()),
        "4_numbers": int((df['n_numbers'] == 4).sum())
    },
    "expression_depths": {
        f"depth_{int(d)}": int(count)
        for d, count in df['expr_depth'].value_counts().sort_index().items()
    },
    "target_values": {
        "mean": float(df['target'].mean()),
        "median": float(df['target'].median()),
        "min": int(df['target'].min()),
        "max": int(df['target'].max())
    }
}
stats_json_path = os.path.join(output_dir, "dataset_stats.json")
with open(stats_json_path, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"\n✓ Dataset statistics saved to: {stats_json_path}")

# ============================================================================
# 1. STRUCTURAL FEATURES (Input-level)
# ============================================================================
print("\n" + "="*80)
print("1. STRUCTURAL FEATURES (Input-level)")
print("="*80)

structural_features = {
    'n_numbers': 'Number set size',
    'range': 'Range (max - min)',
    'std': 'Standard deviation',
    'count_small': 'Count small numbers (<10)',
    'count_large': 'Count large numbers (>50)',
    'count_duplicates': 'Count duplicates',
    'count_even': 'Count even numbers',
    'count_odd': 'Count odd numbers',
    'count_div_2': 'Divisible by 2',
    'count_div_3': 'Divisible by 3',
    'count_div_5': 'Divisible by 5',
    'count_div_7': 'Divisible by 7',
    'count_primes': 'Prime numbers'
}

structural_stats = []
for feature, description in structural_features.items():
    stats = {
        'Feature': description,
        'Mean': df[feature].mean(),
        'Median': df[feature].median(),
        'Std': df[feature].std(),
        'Min': df[feature].min(),
        'Max': df[feature].max()
    }
    structural_stats.append(stats)

structural_df = pd.DataFrame(structural_stats)
print("\nStructural Feature Statistics:")
print(structural_df.to_string(index=False))

# Save to CSV
structural_csv = os.path.join(csv_dir, "structural_features_stats.csv")
structural_df.to_csv(structural_csv, index=False)
print(f"\n Saved to: {structural_csv}")

# ============================================================================
# 2. TARGET-RELATED FEATURES
# ============================================================================
print("\n" + "="*80)
print("2. TARGET-RELATED FEATURES")
print("="*80)

target_features = {
    'target': 'Target value',
    'log_target': 'Log10(target)',
    'distance_simple': 'Distance |sum - target|',
    'distance_max': 'Distance |max - target|',
    'distance_avg': 'Distance |avg - target|'
}

target_stats = []
for feature, description in target_features.items():
    stats = {
        'Feature': description,
        'Mean': df[feature].mean(),
        'Median': df[feature].median(),
        'Std': df[feature].std(),
        'Min': df[feature].min(),
        'Max': df[feature].max()
    }
    target_stats.append(stats)

target_df = pd.DataFrame(target_stats)
print("\nTarget Feature Statistics:")
print(target_df.to_string(index=False))

# Save to CSV
target_csv = os.path.join(csv_dir, "target_features_stats.csv")
target_df.to_csv(target_csv, index=False)
print(f"\n Saved to: {target_csv}")

# ============================================================================
# 3. COMBINATORIAL DIFFICULTY FEATURES
# ============================================================================
print("\n" + "="*80)
print("3. COMBINATORIAL DIFFICULTY FEATURES")
print("="*80)

combinatorial_features = {
    'easy_pairs': 'Number of easy pairs'
}

combinatorial_stats = []
for feature, description in combinatorial_features.items():
    stats = {
        'Feature': description,
        'Mean': df[feature].mean(),
        'Median': df[feature].median(),
        'Std': df[feature].std(),
        'Min': df[feature].min(),
        'Max': df[feature].max()
    }
    combinatorial_stats.append(stats)

combinatorial_df = pd.DataFrame(combinatorial_stats)
print("\nCombinatorial Feature Statistics:")
print(combinatorial_df.to_string(index=False))

# Save to CSV
combinatorial_csv = os.path.join(csv_dir, "combinatorial_features_stats.csv")
combinatorial_df.to_csv(combinatorial_csv, index=False)
print(f"\n Saved to: {combinatorial_csv}")

# ============================================================================
# 4. SOLUTION-BASED FEATURES
# ============================================================================
print("\n" + "="*80)
print("4. SOLUTION-BASED FEATURES")
print("="*80)

solution_features = {
    'expr_depth': 'Expression tree depth',
    'count_add': 'Addition operations',
    'count_sub': 'Subtraction operations',
    'count_mul': 'Multiplication operations',
    'count_div': 'Division operations',
    'noncomm_ops': 'Non-commutative ops (- and /)'
}

solution_stats = []
for feature, description in solution_features.items():
    stats = {
        'Feature': description,
        'Mean': df[feature].mean(),
        'Median': df[feature].median(),
        'Std': df[feature].std(),
        'Min': df[feature].min(),
        'Max': df[feature].max()
    }
    solution_stats.append(stats)

solution_df = pd.DataFrame(solution_stats)
print("\nSolution Feature Statistics:")
print(solution_df.to_string(index=False))

# Save to CSV
solution_csv = os.path.join(csv_dir, "solution_features_stats.csv")
solution_df.to_csv(solution_csv, index=False)
print(f"\n Saved to: {solution_csv}")

# ============================================================================
# 5. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. DISTRIBUTION ANALYSIS")
print("="*80)

# Number set size distribution
print("\nNumber Set Size Distribution:")
n_numbers_dist = df['n_numbers'].value_counts().sort_index()
for n, count in n_numbers_dist.items():
    print(f"  {n} numbers: {count} instances ({count/len(df)*100:.1f}%)")

# Expression depth distribution
print("\nExpression Depth Distribution:")
depth_dist = df['expr_depth'].value_counts().sort_index()
for depth, count in depth_dist.items():
    print(f"  Depth {depth}: {count} instances ({count/len(df)*100:.1f}%)")

# Operator usage
print("\nOperator Usage Statistics:")
total_instances = len(df)
print(f"  Instances with addition: {(df['count_add'] > 0).sum()} ({(df['count_add'] > 0).sum()/total_instances*100:.1f}%)")
print(f"  Instances with subtraction: {(df['count_sub'] > 0).sum()} ({(df['count_sub'] > 0).sum()/total_instances*100:.1f}%)")
print(f"  Instances with multiplication: {(df['count_mul'] > 0).sum()} ({(df['count_mul'] > 0).sum()/total_instances*100:.1f}%)")
print(f"  Instances with division: {(df['count_div'] > 0).sum()} ({(df['count_div'] > 0).sum()/total_instances*100:.1f}%)")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 6.1 Number set size distribution
fig, ax = plt.subplots(figsize=(10, 6))
n_numbers_dist.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Number of Numbers', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Number Set Sizes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "number_set_size_distribution.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: number_set_size_distribution.png")

# 6.2 Expression depth distribution
fig, ax = plt.subplots(figsize=(10, 6))
depth_dist.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Expression Tree Depth', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Expression Tree Depths', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "expression_depth_distribution.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: expression_depth_distribution.png")

# 6.3 Operator usage comparison
fig, ax = plt.subplots(figsize=(10, 6))
operator_usage = {
    'Addition': (df['count_add'] > 0).sum(),
    'Subtraction': (df['count_sub'] > 0).sum(),
    'Multiplication': (df['count_mul'] > 0).sum(),
    'Division': (df['count_div'] > 0).sum()
}
bars = ax.bar(operator_usage.keys(), operator_usage.values(), color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'], edgecolor='black')
ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
ax.set_title('Operator Usage Across All Instances', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/total_instances*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "operator_usage.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: operator_usage.png")

# 6.4 Target value distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['target'], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
ax.set_xlabel('Target Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Target Values', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(df['target'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["target"].mean():.1f}')
ax.axvline(df['target'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["target"].median():.1f}')
ax.legend()
plt.tight_layout()
plot_path = os.path.join(plots_dir, "target_value_distribution.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: target_value_distribution.png")

# 6.5 Range vs Standard Deviation
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['range'], df['std'], c=df['n_numbers'], cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Range (max - min)', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax.set_title('Range vs Standard Deviation (colored by number set size)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Numbers', fontsize=11, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "range_vs_std.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: range_vs_std.png")

# 6.6 Easy pairs vs Expression depth
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['easy_pairs'], df['expr_depth'], alpha=0.5, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Easy Pairs', fontsize=12, fontweight='bold')
ax.set_ylabel('Expression Tree Depth', fontsize=12, fontweight='bold')
ax.set_title('Easy Pairs vs Expression Depth\n(More easy pairs might correlate with shallower expressions)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "easy_pairs_vs_depth.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: easy_pairs_vs_depth.png")

# 6.7 Distance features comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
distance_features_plot = ['distance_simple', 'distance_max', 'distance_avg']
titles = ['Distance: |sum - target|', 'Distance: |max - target|', 'Distance: |mean - target|']
colors = ['#e74c3c', '#3498db', '#2ecc71']

for ax, feature, title, color in zip(axes, distance_features_plot, titles, colors):
    ax.hist(df[feature], bins=30, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Distance', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(df[feature].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {df[feature].mean():.1f}')
    ax.legend()

plt.tight_layout()
plot_path = os.path.join(plots_dir, "distance_features.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: distance_features.png")

# 6.8 Divisibility features heatmap
fig, ax = plt.subplots(figsize=(10, 6))
divisibility_data = df[['count_div_2', 'count_div_3', 'count_div_5', 'count_div_7']].mean()
divisibility_data.index = ['Div by 2', 'Div by 3', 'Div by 5', 'Div by 7']
bars = ax.bar(divisibility_data.index, divisibility_data.values, color=['#e67e22', '#9b59b6', '#1abc9c', '#34495e'], edgecolor='black')
ax.set_ylabel('Average Count per Instance', fontsize=12, fontweight='bold')
ax.set_title('Average Number of Numbers Divisible by 2, 3, 5, 7', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "divisibility_features.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: divisibility_features.png")

# 6.9 Correlation heatmap for key features
print("\nGenerating correlation heatmap...")
correlation_features = [
    'n_numbers', 'range', 'std', 'count_small', 'count_large',
    'count_primes', 'easy_pairs', 'expr_depth',
    'count_add', 'count_sub', 'count_mul', 'count_div',
    'noncomm_ops', 'log_target', 'distance_simple'
]
correlation_matrix = df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plot_path = os.path.join(plots_dir, "feature_correlation_heatmap.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved: feature_correlation_heatmap.png")

# ============================================================================
# 7. COMPLEXITY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("7. COMPLEXITY INSIGHTS")
print("="*80)

# Identify "hard" instances (high depth + many operations)
df['total_ops'] = df['count_add'] + df['count_sub'] + df['count_mul'] + df['count_div']
df['complexity_score'] = df['expr_depth'] * 0.5 + df['total_ops'] * 0.3 + df['noncomm_ops'] * 0.2

print("\nMost Complex Instances (top 10 by complexity score):")
complex_instances = df.nlargest(10, 'complexity_score')[['id', 'n_numbers', 'expr_depth', 'total_ops', 'noncomm_ops', 'complexity_score', 'solution']]
print(complex_instances.to_string(index=False))

print("\nSimplest Instances (bottom 10 by complexity score):")
simple_instances = df.nsmallest(10, 'complexity_score')[['id', 'n_numbers', 'expr_depth', 'total_ops', 'noncomm_ops', 'complexity_score', 'solution']]
print(simple_instances.to_string(index=False))

# Save complexity analysis
complexity_csv = os.path.join(csv_dir, "complexity_analysis.csv")
df[['id', 'n_numbers', 'expr_depth', 'total_ops', 'noncomm_ops', 'complexity_score', 'solution']].to_csv(complexity_csv, index=False)
print(f"\n Saved complexity analysis to: {complexity_csv}")

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("8. OVERALL SUMMARY")
print("="*80)

summary_data = {
    'Category': [],
    'Metric': [],
    'Value': []
}

# Structural summary
summary_data['Category'].extend(['Structural'] * 3)
summary_data['Metric'].extend(['Avg numbers per instance', 'Avg range', 'Avg primes per instance'])
summary_data['Value'].extend([f"{df['n_numbers'].mean():.2f}", f"{df['range'].mean():.2f}", f"{df['count_primes'].mean():.2f}"])

# Target summary
summary_data['Category'].extend(['Target'] * 2)
summary_data['Metric'].extend(['Avg target value', 'Avg log(target)'])
summary_data['Value'].extend([f"{df['target'].mean():.2f}", f"{df['log_target'].mean():.2f}"])

# Solution summary
summary_data['Category'].extend(['Solution'] * 3)
summary_data['Metric'].extend(['Avg expression depth', 'Avg total operations', 'Avg non-commutative ops'])
summary_data['Value'].extend([f"{df['expr_depth'].mean():.2f}", f"{df['total_ops'].mean():.2f}", f"{df['noncomm_ops'].mean():.2f}"])

# Difficulty summary
summary_data['Category'].extend(['Difficulty'] * 2)
summary_data['Metric'].extend(['Avg easy pairs', 'Avg complexity score'])
summary_data['Value'].extend([f"{df['easy_pairs'].mean():.2f}", f"{df['complexity_score'].mean():.2f}"])

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

summary_csv = os.path.join(csv_dir, "overall_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n Saved summary to: {summary_csv}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  CSV files:")
print("    - structural_features_stats.csv")
print("    - target_features_stats.csv")
print("    - combinatorial_features_stats.csv")
print("    - solution_features_stats.csv")
print("    - complexity_analysis.csv")
print("    - overall_summary.csv")
print("\n  Visualizations:")
print("    - number_set_size_distribution.png")
print("    - expression_depth_distribution.png")
print("    - operator_usage.png")
print("    - target_value_distribution.png")
print("    - range_vs_std.png")
print("    - easy_pairs_vs_depth.png")
print("    - distance_features.png")
print("    - divisibility_features.png")
print("    - feature_correlation_heatmap.png")
