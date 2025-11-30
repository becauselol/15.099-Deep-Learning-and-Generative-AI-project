#!/usr/bin/env python3
"""
Script to create countdown_results_with_prompt_gemini.csv

This script combines:
- countdown_results_gemini_done.csv (prompt_id, instance_id, response, correct, message)
- countdown_prompts_gemini_transformed.csv (id, prompt template + prompt features)
- countdown_features.csv (id/instance_id, numbers, target, solution + instance features)

Output: countdown_results_with_prompt_gemini.csv with columns:
- prompt_id, instance_id
- prompt (with placeholders filled in)
- correct, message
- Instance features: n_numbers, range, std, count_small, count_large, count_duplicates,
   count_even, count_odd, count_div_2, count_div_3, count_div_5, count_div_7, count_primes,
   distance_simple, distance_max, distance_avg, easy_pairs, log_target, expr_depth,
   count_add, count_sub, count_mul, count_div, noncomm_ops, numbers, target, solution
- Prompt features: paraphrasing, role-specification, reasoning-trigger, chain-of-thought,
   self-check, conciseness, verbosity, context-expansion, few-shot-count, length
"""

import pandas as pd
import ast
import re
import os


def format_numbers_for_prompt(numbers):
    """
    Format numbers to inject into the prompt.
    Format: [44 19 35]
    """
    return "[" + " ".join(str(x) for x in numbers) + "]"


def main():
    print("=" * 70)
    print("CREATE COUNTDOWN RESULTS WITH PROMPT")
    print("=" * 70)
    print()

    # File paths
    results_file = "countdown_results_gemini_done.csv"
    prompts_file = "countdown_prompts_gemini_transformed.csv"
    features_file = "countdown_features.csv"
    output_file = "countdown_results_with_prompt_gemini.csv"

    # Check if files exist
    for file in [results_file, prompts_file, features_file]:
        if not os.path.exists(file):
            print(f"ERROR: File '{file}' not found!")
            print(f"Current directory: {os.getcwd()}")
            return

    print("Loading data files...")
    print(f"  - {results_file}")
    print(f"  - {prompts_file}")
    print(f"  - {features_file}")
    print()

    # Load data
    results_df = pd.read_csv(results_file)
    prompts_df = pd.read_csv(prompts_file)
    features_df = pd.read_csv(features_file)

    print(f"Loaded {len(results_df)} results")
    print(f"Loaded {len(prompts_df)} prompt templates")
    print(f"Loaded {len(features_df)} feature instances")
    print()

    # Filter out not-evaluated results (where correct is NaN)
    original_count = len(results_df)
    results_df = results_df[results_df['correct'].notna()].copy()
    filtered_count = original_count - len(results_df)

    print(f"Filtered out {filtered_count} not-evaluated results")
    print(f"Remaining results: {len(results_df)} (evaluated only)")
    print()

    # Create dictionaries for fast lookup
    prompts_by_id = {int(row['id']): row for _, row in prompts_df.iterrows()}
    features_by_id = {int(row['id']): row for _, row in features_df.iterrows()}

    # Define feature columns to include
    instance_feature_cols = [
        'n_numbers', 'range', 'std', 'count_small', 'count_large', 'count_duplicates',
        'count_even', 'count_odd', 'count_div_2', 'count_div_3', 'count_div_5',
        'count_div_7', 'count_primes', 'distance_simple', 'distance_max', 'distance_avg',
        'easy_pairs', 'log_target', 'expr_depth', 'count_add', 'count_sub',
        'count_mul', 'count_div', 'noncomm_ops'
    ]

    prompt_feature_cols = [
        'paraphrasing', 'role-specification', 'reasoning-trigger', 'chain-of-thought',
        'self-check', 'conciseness', 'verbosity', 'context-expansion', 'few-shot-count', 'length'
    ]

    # Process each result
    print("Processing results and filling prompt templates...")
    output_rows = []
    skipped = 0

    for idx, row in results_df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(results_df)} results...")

        prompt_id = int(row["prompt_id"])
        instance_id = int(row["instance_id"])

        # Check if we have the required data
        if prompt_id not in prompts_by_id:
            print(f"WARNING: Skipping row {idx}: prompt_id={prompt_id} not found in prompts")
            skipped += 1
            continue

        if instance_id not in features_by_id:
            print(f"WARNING: Skipping row {idx}: instance_id={instance_id} not found in features")
            skipped += 1
            continue

        # Get prompt template and instance data
        prompt_row = prompts_by_id[prompt_id]
        inst_row = features_by_id[instance_id]

        # Parse instance fields
        try:
            numbers = ast.literal_eval(inst_row["numbers"])
        except Exception:
            numbers = inst_row["numbers"]
            if isinstance(numbers, str):
                numbers = list(map(int, re.findall(r"\d+", numbers)))

        target = inst_row["target"]
        solution = inst_row.get("solution", "")

        # Fill placeholders in prompt template
        numbers_str = format_numbers_for_prompt(numbers)
        prompt_template = prompt_row['prompt']
        full_prompt = (
            prompt_template
            .replace("<numbers_placeholder>", numbers_str)
            .replace("<target_placeholder>", str(target))
        )

        # Create output row with basic fields
        output_row = {
            'prompt_id': prompt_id,
            'instance_id': instance_id,
            'prompt': full_prompt,
            'correct': row.get('correct', ''),
            'message': row.get('message', ''),
        }

        # Add instance features
        for col in instance_feature_cols:
            if col in inst_row:
                output_row[f'inst_{col}'] = inst_row[col]
            else:
                output_row[f'inst_{col}'] = None

        # Add numbers, target, solution separately (not prefixed)
        output_row['numbers'] = str(numbers)  # Store as string for CSV compatibility
        output_row['target'] = target
        output_row['solution'] = solution

        # Add prompt features
        for col in prompt_feature_cols:
            if col in prompt_row:
                output_row[f'prompt_{col}'] = prompt_row[col]
            else:
                output_row[f'prompt_{col}'] = None

        output_rows.append(output_row)

    print(f"Processed {len(results_df)} results")
    print(f"Successfully created {len(output_rows)} combined rows")
    print(f"Skipped {skipped} rows due to missing data")
    print()

    # Create output dataframe
    output_df = pd.DataFrame(output_rows)

    # Save to CSV
    print(f"Saving to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print(f"✓ Successfully saved {len(output_df)} rows to {output_file}")
    print()

    # Show summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total rows: {len(output_df)}")
    print(f"Total columns: {len(output_df.columns)}")
    print(f"Unique prompt_ids: {output_df['prompt_id'].nunique()}")
    print(f"Unique instance_ids: {output_df['instance_id'].nunique()}")

    if 'correct' in output_df.columns:
        correct_counts = output_df['correct'].value_counts()
        print(f"\nCorrectness distribution:")
        for value, count in correct_counts.items():
            percentage = (count / len(output_df)) * 100
            print(f"  {value}: {count} ({percentage:.2f}%)")

    print()
    print("=" * 70)
    print("COLUMN SUMMARY")
    print("=" * 70)
    print("Columns in output file:")
    for i, col in enumerate(output_df.columns, 1):
        print(f"  {i:2d}. {col}")

    print()
    print("=" * 70)
    print("SAMPLE ROWS")
    print("=" * 70)
    sample_cols = ['prompt_id', 'instance_id', 'correct', 'numbers', 'target',
                   'inst_n_numbers', 'inst_range', 'prompt_paraphrasing', 'prompt_length']
    available_cols = [col for col in sample_cols if col in output_df.columns]
    print(output_df[available_cols].head(10))
    print()

    # Show a sample prompt
    print("=" * 70)
    print("SAMPLE FULL PROMPT")
    print("=" * 70)
    sample_prompt = output_df['prompt'].iloc[0]
    print(sample_prompt[:500] + "..." if len(sample_prompt) > 500 else sample_prompt)
    print()

    print("✓ Complete!")


if __name__ == "__main__":
    main()
