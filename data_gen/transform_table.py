import pandas as pd
import ast
import re

# --- config ---
INPUT_CSV = "countdown_prompts_gemini.csv"       # change to your actual filename
OUTPUT_CSV = "transformed_countdown_prompts_gemini.csv"     # output filename
# ---------------

# all possible binary feature columns
FEATURE_COLS = [
    "paraphrasing",
    "role-specification",
    "reasoning-trigger",
    "chain-of-thought",
    "self-check",
    "conciseness",
    "verbosity",
    "context-expansion",
]

# regex to extract "<k>-shot" style features
SHOT_PATTERN = re.compile(r"(\d+)-shot", flags=re.IGNORECASE)


def parse_features_cell(cell):
    """
    Safely parse the 'features' cell which looks like a Python list literal,
    e.g. "["paraphrasing", "conciseness"]"
    Returns a *list of strings* (possibly empty).
    """
    if pd.isna(cell):
        return []

    # If already a list (e.g. if you pre-processed), just return
    if isinstance(cell, list):
        return cell

    # Try literal_eval first
    try:
        feats = ast.literal_eval(cell)
        # Ensure it's a list of strings
        if isinstance(feats, (list, tuple)):
            return [str(f) for f in feats]
    except Exception:
        pass

    # Fallback: naive split
    cell = str(cell).strip()
    cell = cell.strip("[]")
    if not cell:
        return []
    parts = [p.strip() for p in cell.split(",")]
    # remove quotes if present
    parts = [p.strip('"').strip("'") for p in parts if p]
    return parts


def extract_few_shot_count(features_list):
    """
    Given a list of features, look for something like '2-shot', '5-shot', etc.
    Returns the integer k if found, else 0.
    """
    for feat in features_list:
        match = SHOT_PATTERN.search(str(feat))
        if match:
            return int(match.group(1))
    return 0


# 1. Read input CSV
df = pd.read_csv(INPUT_CSV)

# 2. Parse the features column into a proper list
df["features_list"] = df["features"].apply(parse_features_cell)

# Normalize to lowercase to match FEATURE_COLS robustly
df["features_list_lower"] = df["features_list"].apply(lambda lst: [f.lower() for f in lst])

# 3. Add binary columns for each feature
for feat in FEATURE_COLS:
    df[feat] = df["features_list_lower"].apply(lambda lst, f=feat: int(f in lst))

# 4. few-shot-count from any "<k>-shot" feature
df["few-shot-count"] = df["features_list"].apply(extract_few_shot_count)

# 5. length = number of characters in the prompt
df["length"] = df["prompt"].astype(str).str.len()

# 6. id column (0-based index; change to start=1 if you want 1-based IDs)
df["id"] = range(len(df))

# 7. Keep only the requested columns, in order
output_cols = (
    ["id"]
    + FEATURE_COLS
    + ["few-shot-count", "length", "prompt"]
)
df_out = df[output_cols]

# 8. Write to new CSV
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote transformed CSV to {OUTPUT_CSV}")