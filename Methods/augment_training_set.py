#!/usr/bin/env python3
"""
Augment training set by adding pseudo-labeled synthetic pairs.

Inputs (defaults can be overridden by environment variables):

- TRAIN_DATA_FILE
    ../data_gen/train_countdown_results_with_prompt_gemini.csv
    (real evaluated data, contains inst_* and prompt_* columns)

- ALL_PAIRS_FILE
    ../data_gen/countdown_results.csv
    (all (prompt_id, instance_id) pairs, with many correct = NaN)

- FEATURES_FILE
    ../data_gen/countdown_features.csv
    (canonical instance features: id, n_numbers, range, ..., numbers, target, solution)

- PROMPTS_FILE
    ../data_gen/countdown_prompts_gemini_transformed.csv
    (canonical prompt features: id, paraphrasing, ..., length, prompt)

- XGBOOST_MODEL_PATH
    path to a trained XGBoost model (from xgboost_unified.py with FEATURE_MODE=features_only)

Output:

- train_countdown_results_with_prompt_augmented_xgb.csv
    Real rows + synthetic rows (with recovered inst_* and prompt_* and predicted `correct`).
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv


def main():
    load_dotenv()

    # ------------------------------------------------------------------
    # Paths and config
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent

    train_file = "data_gen/train_countdown_results_with_prompt_gemini.csv"
    all_pairs_file = "data_gen/countdown_results.csv"
    features_file = "data_gen/countdown_features.csv"
    prompts_file = "data_gen/countdown_prompts_gemini_transformed.csv"
    model_path = "Methods/xgboost_model_features_only.json"
    output_file = "data_gen/train_countdown_results_with_prompt_augmented_xgb.csv"
    pred_threshold = 0.5
    def resolve(path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = (project_root / path_str).resolve()
        return p

    train_path = resolve(train_file)
    all_pairs_path = resolve(all_pairs_file)
    features_path = resolve(features_file)
    prompts_path = resolve(prompts_file)
    model_path = resolve(model_path)
    output_path = resolve(output_file)

    print("===================================================================")
    print("AUGMENT TRAINING SET WITH XGBOOST PSEUDO-LABELS")
    print("===================================================================")
    print(f"Train file (real)   : {train_path}")
    print(f"All pairs file      : {all_pairs_path}")
    print(f"Features file       : {features_path}")
    print(f"Prompts file        : {prompts_path}")
    print(f"XGBoost model path  : {model_path}")
    print(f"Output augmented CSV: {output_path}")
    print(f"Prediction threshold: {pred_threshold}")
    print("===================================================================")

    # ------------------------------------------------------------------
    # Load base data
    # ------------------------------------------------------------------
    print("\n[1] Loading real training data...")
    train_df = pd.read_csv(train_path)
    print(f"Real training rows: {len(train_df)}")

    print("\n[2] Loading all (prompt_id, instance_id) pairs...")
    pairs_df = pd.read_csv(all_pairs_path)
    print(f"Total pairs in countdown_results.csv: {len(pairs_df)}")

    print("\n[3] Loading canonical instance & prompt features...")
    features_df = pd.read_csv(features_path)
    prompts_df = pd.read_csv(prompts_path)
    print(f"Instance feature rows: {len(features_df)}")
    print(f"Prompt template rows : {len(prompts_df)}")

    # Build lookup dictionaries
    features_by_id = {
        int(row["id"]): row for _, row in features_df.iterrows()
    }
    prompts_by_id = {
        int(row["id"]): row for _, row in prompts_df.iterrows()
    }

    # ------------------------------------------------------------------
    # Identify feature columns consistent with XGBoost training
    # (features_only mode: inst_* and prompt_* only)
    # ------------------------------------------------------------------
    inst_cols = [c for c in train_df.columns if c.startswith("inst_")]
    prompt_cols = [c for c in train_df.columns if c.startswith("prompt_")]
    feature_cols = inst_cols + prompt_cols

    print("\nFeature columns used for XGBoost predictions:")
    print(feature_cols)
    print(f"Total feature columns: {len(feature_cols)}")

    # Sanity: label column must exist
    label_col = "correct"
    assert label_col in train_df.columns, f"Label column {label_col} not found in train data."

    # ------------------------------------------------------------------
    # Load XGBoost model (already trained with xgboost_unified.py)
    # ------------------------------------------------------------------
    print("\n[4] Loading XGBoost model...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_path))
    print("✓ Model loaded.")

    # ------------------------------------------------------------------
    # Determine which pairs are already in the real training set
    # ------------------------------------------------------------------
    print("\n[5] Computing new pairs (not in real training)...")

    # Keys in real train
    real_keys = train_df[["prompt_id", "instance_id"]].drop_duplicates()
    real_keys["in_train"] = True

    # Merge to find which pairs are new
    pairs_with_flag = pairs_df.merge(
        real_keys,
        on=["prompt_id", "instance_id"],
        how="left",
        indicator=False,
    )
    pairs_with_flag["in_train"].fillna(False, inplace=True)

    new_pairs = pairs_with_flag[~pairs_with_flag["in_train"]].copy()
    print(f"New (prompt_id, instance_id) pairs: {len(new_pairs)}")

    # ------------------------------------------------------------------
    # Reconstruct full feature rows for synthetic pairs
    # using countdown_features.csv and countdown_prompts_gemini_transformed.csv
    # ------------------------------------------------------------------
    print("\n[6] Reconstructing features for new pairs from base feature files...")

    # columns present in train_df; we will mirror its schema as much as possible
    train_columns = list(train_df.columns)
    synthetic_rows = []
    skipped_missing = 0

    # instance feature columns in base file
    instance_feature_cols = [
        "n_numbers",
        "range",
        "std",
        "count_small",
        "count_large",
        "count_duplicates",
        "count_even",
        "count_odd",
        "count_div_2",
        "count_div_3",
        "count_div_5",
        "count_div_7",
        "count_primes",
        "distance_simple",
        "distance_max",
        "distance_avg",
        "easy_pairs",
        "log_target",
        "expr_depth",
        "count_add",
        "count_sub",
        "count_mul",
        "count_div",
        "noncomm_ops",
    ]

    prompt_feature_cols = [
        "paraphrasing",
        "role-specification",
        "reasoning-trigger",
        "chain-of-thought",
        "self-check",
        "conciseness",
        "verbosity",
        "context-expansion",
        "few-shot-count",
        "length",
    ]

    for _, row in new_pairs.iterrows():
        pid = int(row["prompt_id"])
        iid = int(row["instance_id"])

        feat_row = features_by_id.get(iid)
        prompt_row = prompts_by_id.get(pid)

        if feat_row is None or prompt_row is None:
            skipped_missing += 1
            continue

        # Start from a template with all columns as NaN
        out = {col: np.nan for col in train_columns}

        # Basic identifiers and metadata
        if "prompt_id" in out:
            out["prompt_id"] = pid
        if "instance_id" in out:
            out["instance_id"] = iid

        # Response / message from pairs_df (may be NaN; that's fine)
        if "response" in out and "response" in row:
            out["response"] = row["response"]
        if "message" in out and "message" in row:
            out["message"] = row["message"]

        # Mark synthetic source if column exists or create later
        if "source" in out:
            out["source"] = "synthetic"

        # Numbers / target / solution if present in train schema
        for col in ["numbers", "target", "solution"]:
            if col in out and col in feat_row.index:
                out[col] = feat_row[col]

        # Instance features -> inst_* columns
        for col in instance_feature_cols:
            inst_col = f"inst_{col}"
            if inst_col in out and col in feat_row.index:
                out[inst_col] = feat_row[col]

        # Prompt numerical features -> prompt_* columns
        for col in prompt_feature_cols:
            prompt_col = f"prompt_{col}"
            if prompt_col in out and col in prompt_row.index:
                out[prompt_col] = prompt_row[col]

        # Prompt text itself, if schema has a 'prompt' column
        if "prompt" in out and "prompt" in prompt_row.index:
            out["prompt"] = prompt_row["prompt"]

        synthetic_rows.append(out)

    print(f"Synthetic rows reconstructed: {len(synthetic_rows)}")
    print(f"Synthetic rows skipped (missing base features): {skipped_missing}")

    if not synthetic_rows:
        print("No synthetic rows could be constructed with complete features. Nothing to do.")
        return

    synthetic_df = pd.DataFrame(synthetic_rows)
    print(f"Synthetic dataframe shape: {synthetic_df.shape}")

    # ------------------------------------------------------------------
    # Predict pseudo-labels for synthetic rows with XGBoost
    # ------------------------------------------------------------------
    print("\n[7] Predicting pseudo-labels for synthetic rows...")

    # Ensure no missing in feature columns (they should all be filled now)
    if synthetic_df[feature_cols].isna().any().any():
        # As a safety net, fill any remaining NaNs with 0.0
        print("[!] Warning: some feature values were still NaN; filling with 0.0")
        X_new = synthetic_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        X_new = synthetic_df[feature_cols].to_numpy(dtype=np.float32)

    prob_correct = xgb_model.predict_proba(X_new)[:, 1]
    pred_correct = (prob_correct >= pred_threshold).astype(float)

    synthetic_df["pred_prob_correct"] = prob_correct
    synthetic_df["correct"] = pred_correct

    print("Example synthetic rows with predictions:")
    print(synthetic_df[["prompt_id", "instance_id", "pred_prob_correct", "correct"]].head())

    # ------------------------------------------------------------------
    # Merge real + synthetic into augmented training set
    # ------------------------------------------------------------------
    print("\n[8] Building final augmented training set...")

    # Ensure 'source' column exists in real data too
    if "source" not in train_df.columns:
        train_df["source"] = "real"
    else:
        train_df["source"] = train_df["source"].fillna("real")

    augmented_df = pd.concat(
        [train_df, synthetic_df],
        axis=0,
        ignore_index=True,
    )

    print(f"Real rows     : {len(train_df)}")
    print(f"Synthetic rows: {len(synthetic_df)}")
    print(f"Total rows    : {len(augmented_df)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"\n✓ Augmented training set saved to: {output_path}")


if __name__ == "__main__":
    main()