# Countdown Results with Prompt - Data Dictionary

## File: `countdown_results_with_prompt_gemini.csv`

This file combines data from three sources to create a comprehensive dataset for training and analysis:
- `countdown_results_gemini_done.csv` - Model results and correctness labels (filtered to evaluated only)
- `countdown_prompts_gemini_transformed.csv` - Prompt templates and prompt features
- `countdown_features.csv` - Instance features (numbers, target, solution, etc.)

**Total Rows:** 7,213 (evaluated results only)
**Total Columns:** 42
**File Size:** ~5.6 MB

**Note:** This file contains only evaluated results (where `correct` is not NaN). Not-evaluated results (4,464 rows) are automatically filtered out during generation.

## How to Regenerate This File

Run the following command from the `data_gen` directory:

```bash
# Using conda environment
module load miniforge/24.3.0-0
conda activate ~/.conda/envs/promptenv
python create_results_with_prompt.py
```

Or use the shell script:

```bash
./run_create_results_with_prompt.sh
```

## Column Descriptions

### Basic Identification (5 columns)
1. **prompt_id** - ID of the prompt template used
2. **instance_id** - ID of the instance (numbers + target combination)
3. **prompt** - Full prompt text with placeholders filled in
4. **correct** - Whether the model's response was correct (1.0 = correct, 0.0 = incorrect, NaN = not evaluated)
5. **message** - Any error or status message from evaluation

### Instance Features (24 columns, prefixed with `inst_`)
Features describing the mathematical instance (numbers and target):

6. **inst_n_numbers** - Number of numbers in the instance (e.g., 3 or 4)
7. **inst_range** - Range of the numbers (max - min)
8. **inst_std** - Standard deviation of the numbers
9. **inst_count_small** - Count of small numbers (<10)
10. **inst_count_large** - Count of large numbers (>90)
11. **inst_count_duplicates** - Count of duplicate numbers
12. **inst_count_even** - Count of even numbers
13. **inst_count_odd** - Count of odd numbers
14. **inst_count_div_2** - Count of numbers divisible by 2
15. **inst_count_div_3** - Count of numbers divisible by 3
16. **inst_count_div_5** - Count of numbers divisible by 5
17. **inst_count_div_7** - Count of numbers divisible by 7
18. **inst_count_primes** - Count of prime numbers
19. **inst_distance_simple** - Distance to target using simple operations
20. **inst_distance_max** - Maximum distance to target
21. **inst_distance_avg** - Average distance to target
22. **inst_easy_pairs** - Count of easy number pairs
23. **inst_log_target** - Logarithm of target value
24. **inst_expr_depth** - Depth of the solution expression tree
25. **inst_count_add** - Count of addition operations in solution
26. **inst_count_sub** - Count of subtraction operations in solution
27. **inst_count_mul** - Count of multiplication operations in solution
28. **inst_count_div** - Count of division operations in solution
29. **inst_noncomm_ops** - Count of non-commutative operations in solution

### Instance Core Data (3 columns)
30. **numbers** - The list of numbers (as string representation)
31. **target** - The target value to achieve
32. **solution** - The correct solution expression

### Prompt Features (10 columns, prefixed with `prompt_`)
Features describing the prompt engineering techniques used:

33. **prompt_paraphrasing** - Whether the prompt uses paraphrasing (0/1)
34. **prompt_role-specification** - Whether a role is specified (0/1)
35. **prompt_reasoning-trigger** - Whether reasoning is triggered (0/1)
36. **prompt_chain-of-thought** - Whether chain-of-thought prompting is used (0/1)
37. **prompt_self-check** - Whether self-checking is requested (0/1)
38. **prompt_conciseness** - Whether concise output is requested (0/1)
39. **prompt_verbosity** - Whether verbose output is requested (0/1)
40. **prompt_context-expansion** - Whether context is expanded (0/1)
41. **prompt_few-shot-count** - Number of few-shot examples (0 or more)
42. **prompt_length** - Length of the prompt in characters

## Data Statistics

- **Unique Prompts:** 3,869
- **Unique Instances:** 1,926
- **Correctness Distribution:**
  - Correct (1.0): 3,018 (41.84%)
  - Incorrect (0.0): 4,195 (58.16%)
  - Not evaluated (NaN): 0 (filtered out)

## Usage Examples

### Load the data in Python

```python
import pandas as pd

# Load the full dataset (already filtered to evaluated results only)
df = pd.read_csv('countdown_results_with_prompt_gemini.csv')

# Access prompt text and labels
X = df['prompt'].values
y = df['correct'].values

# Access features
instance_features = df[[col for col in df.columns if col.startswith('inst_')]]
prompt_features = df[[col for col in df.columns if col.startswith('prompt_')]]
```

### Use in ML model training

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data (already filtered to evaluated results only)
df = pd.read_csv('countdown_results_with_prompt_gemini.csv')

# Split features and labels
X = df['prompt'].values
y = df['correct'].values.astype(int)

# Optional: Include additional features
feature_cols = [col for col in df.columns if col.startswith('inst_') or col.startswith('prompt_')]
additional_features = df[feature_cols].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Notes

- **This file contains only evaluated results** - not-evaluated results (where `correct` is NaN) are automatically filtered out during generation
- The `prompt` column contains the full prompt with placeholders (`<numbers_placeholder>` and `<target_placeholder>`) filled in with actual values
- Instance features are prefixed with `inst_` to distinguish them from prompt features
- Prompt features are prefixed with `prompt_` and are binary (0/1) except for `prompt_few-shot-count` and `prompt_length`
- All rows have valid `correct` values (either 0.0 or 1.0) - no NaN values
- The `numbers` column is stored as a string representation of a list (e.g., "[44, 19, 35]")
- If you need the not-evaluated results, see `countdown_results_not_evaluated.csv` (4,464 rows with NaN `correct` values)
