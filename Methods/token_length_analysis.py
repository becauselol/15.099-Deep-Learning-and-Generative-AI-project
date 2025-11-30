import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib not available. Skipping visualizations.")

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'bert-base-uncased')
PROMPTS_FILE = os.getenv('DATA_FILE', '../data_gen/countdown_results_with_prompt_gemini.csv')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))

print("=" * 70)
print("TOKEN LENGTH ANALYSIS")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Prompts file: {PROMPTS_FILE}")
print(f"Current MAX_LENGTH setting: {MAX_LENGTH}")
print("=" * 70)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load prompts
print(f"Loading prompts from {PROMPTS_FILE}...")
prompts_df = pd.read_csv(PROMPTS_FILE)
print(f"Total prompts: {len(prompts_df)}")
print()

# Get prompts
prompts = prompts_df['prompt'].values

# Filter out any NaN prompts
valid_prompts = [p for p in prompts if pd.notna(p)]
print(f"Valid prompts (non-NaN): {len(valid_prompts)}")
print()

# Tokenize and get lengths
print("Tokenizing prompts and calculating lengths...")
token_lengths = []

for prompt in valid_prompts:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    token_lengths.append(len(tokens))

token_lengths = np.array(token_lengths)

# Calculate statistics
print("=" * 70)
print("TOKEN LENGTH STATISTICS")
print("=" * 70)
print(f"Mean length: {np.mean(token_lengths):.2f} tokens")
print(f"Median length: {np.median(token_lengths):.2f} tokens")
print(f"Std deviation: {np.std(token_lengths):.2f} tokens")
print(f"Min length: {np.min(token_lengths)} tokens")
print(f"Max length: {np.max(token_lengths)} tokens")
print()

# Percentiles
percentiles = [50, 75, 90, 95, 99, 99.5, 100]
print("PERCENTILES:")
for p in percentiles:
    value = np.percentile(token_lengths, p)
    print(f"  {p}th percentile: {value:.0f} tokens")
print()

# Check against MAX_LENGTH
prompts_exceeding = np.sum(token_lengths > MAX_LENGTH)
percentage_exceeding = (prompts_exceeding / len(token_lengths)) * 100

print("=" * 70)
print(f"PROMPTS EXCEEDING MAX_LENGTH ({MAX_LENGTH})")
print("=" * 70)
print(f"Number of prompts: {prompts_exceeding}")
print(f"Percentage: {percentage_exceeding:.2f}%")
print()

if prompts_exceeding > 0:
    print(f"WARNING: {prompts_exceeding} prompts ({percentage_exceeding:.2f}%) exceed the current MAX_LENGTH of {MAX_LENGTH}!")
    print("These prompts will be truncated during training.")
    print()

    # Suggest new MAX_LENGTH
    suggested_max_length = int(np.percentile(token_lengths, 99.5))
    print(f"SUGGESTION: Consider increasing MAX_LENGTH to {suggested_max_length} to cover 99.5% of prompts.")
else:
    print(f"✓ All prompts fit within MAX_LENGTH of {MAX_LENGTH}")

    # Suggest optimization
    optimal_max_length = int(np.percentile(token_lengths, 99))
    if optimal_max_length < MAX_LENGTH:
        print(f"OPTIMIZATION: You could reduce MAX_LENGTH to {optimal_max_length} and still cover 99% of prompts.")

print()

# Distribution by bins
print("=" * 70)
print("TOKEN LENGTH DISTRIBUTION")
print("=" * 70)
bins = [0, 64, 128, 256, 512, 1024, 2048, float('inf')]
bin_labels = ['0-64', '65-128', '129-256', '257-512', '513-1024', '1025-2048', '2048+']

for i in range(len(bins) - 1):
    count = np.sum((token_lengths > bins[i]) & (token_lengths <= bins[i+1]))
    percentage = (count / len(token_lengths)) * 100
    print(f"  {bin_labels[i]:>12} tokens: {count:5d} prompts ({percentage:5.2f}%)")

print("=" * 70)
print()

# Create visualization if matplotlib is available
if PLOTTING_AVAILABLE:
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(token_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'MAX_LENGTH ({MAX_LENGTH})')
    axes[0, 0].axvline(np.mean(token_lengths), color='green', linestyle='--', linewidth=2, label=f'Mean ({np.mean(token_lengths):.0f})')
    axes[0, 0].set_xlabel('Token Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Token Lengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot(token_lengths, vert=True)
    axes[0, 1].axhline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'MAX_LENGTH ({MAX_LENGTH})')
    axes[0, 1].set_ylabel('Token Length')
    axes[0, 1].set_title('Token Length Distribution (Box Plot)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_lengths = np.sort(token_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    axes[1, 0].plot(sorted_lengths, cumulative, linewidth=2)
    axes[1, 0].axvline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'MAX_LENGTH ({MAX_LENGTH})')
    axes[1, 0].axhline(99, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='99th percentile')
    axes[1, 0].set_xlabel('Token Length')
    axes[1, 0].set_ylabel('Cumulative Percentage (%)')
    axes[1, 0].set_title('Cumulative Distribution of Token Lengths')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Violin plot
    parts = axes[1, 1].violinplot([token_lengths], vert=True, showmeans=True, showmedians=True)
    axes[1, 1].axhline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'MAX_LENGTH ({MAX_LENGTH})')
    axes[1, 1].set_ylabel('Token Length')
    axes[1, 1].set_title('Token Length Distribution (Violin Plot)')
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels(['All Prompts'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = './token_length_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    print()
else:
    print("Skipping visualization (matplotlib not available)")
    print()

# Show some examples of longest prompts
print("=" * 70)
print("EXAMPLES OF LONGEST PROMPTS")
print("=" * 70)
longest_indices = np.argsort(token_lengths)[-5:][::-1]

for i, idx in enumerate(longest_indices, 1):
    print(f"\n{i}. Prompt (length: {token_lengths[idx]} tokens):")
    print("-" * 70)
    print(valid_prompts[idx][:300] + "..." if len(valid_prompts[idx]) > 300 else valid_prompts[idx])
    print("-" * 70)

print("\nAnalysis complete!")
