import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
RANDOM_SEED = int(os.getenv('RANDOM_STATE', 42))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
INPUT_FILE = 'countdown_results_with_prompt_gemini.csv'
OUTPUT_TRAIN_FILE = 'train_countdown_results_with_prompt_gemini.csv'
OUTPUT_TEST_FILE = 'test_countdown_results_with_prompt_gemini.csv'

print("=" * 70)
print("TRAIN-TEST SPLIT GENERATION")
print("=" * 70)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Test Size: {TEST_SIZE}")
print(f"Input File: {INPUT_FILE}")
print("=" * 70)
print()

# Load data
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Total samples (before filtering): {len(df)}")

# Filter out rows with NaN in 'correct' column
df = df.dropna(subset=['correct'])
print(f"Total samples (after filtering NaN): {len(df)}")

# Get labels for stratification
y = df['correct'].values.astype(int)
print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
print(f"Negative samples: {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.2f}%)")

# Perform train-test split
print(f"\nPerforming stratified train-test split...")
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# Verify stratification
train_pos = sum(train_df['correct'])
train_total = len(train_df)
test_pos = sum(test_df['correct'])
test_total = len(test_df)

print(f"\nTrain set: {train_total} samples")
print(f"  Positive: {train_pos} ({train_pos/train_total*100:.2f}%)")
print(f"  Negative: {train_total - train_pos} ({(train_total - train_pos)/train_total*100:.2f}%)")

print(f"\nTest set: {test_total} samples")
print(f"  Positive: {test_pos} ({test_pos/test_total*100:.2f}%)")
print(f"  Negative: {test_total - test_pos} ({(test_total - test_pos)/test_total*100:.2f}%)")

# Save splits
print(f"\nSaving train set to '{OUTPUT_TRAIN_FILE}'...")
train_df.to_csv(OUTPUT_TRAIN_FILE, index=False)

print(f"Saving test set to '{OUTPUT_TEST_FILE}'...")
test_df.to_csv(OUTPUT_TEST_FILE, index=False)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT COMPLETE")
print("=" * 70)
print(f"Train file: {OUTPUT_TRAIN_FILE}")
print(f"Test file: {OUTPUT_TEST_FILE}")
print("=" * 70)
