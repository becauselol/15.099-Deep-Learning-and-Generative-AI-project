#!/usr/bin/env python3
"""
Build reward matrix using XGBoost model predictions, in a memory-friendly way.

For every (instance, prompt-template) combination, predict the
probability of correctness P(correct = 1 | inst_*, prompt_*)
using the trained XGBoost model (features_only mode), but WITHOUT
materializing the full 7.6M-row feature matrix at once.

Outputs:

- data_gen/reward_matrix_xgb.csv
    Rows: instance_id
    Cols: template labels (paraphrasing_role-spec_..._few-shot-bin_lenbin)
    Values: predicted probability of correctness

- data_gen/reward_long_xgb.csv
    Long form: (instance_id, template, reward)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv


def resolve(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (project_root / path_str).resolve()
    return p


def main():
    load_dotenv()

    project_root = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    train_file     = os.getenv("TRAIN_DATA_FILE",  "data_gen/train_countdown_results_with_prompt_gemini.csv")
    features_file  = os.getenv("FEATURES_FILE",    "data_gen/countdown_features.csv")
    prompts_file   = os.getenv("PROMPTS_FILE",     "data_gen/countdown_prompts_gemini_transformed.csv")
    model_path_str = os.getenv("XGBOOST_MODEL_PATH", "Methods/xgboost_model_features_only.json")

    reward_matrix_out = "data_gen/reward_matrix_xgb.csv"
    reward_long_out   = "data_gen/reward_long_xgb.csv"

    train_path    = resolve(train_file, project_root)
    features_path = resolve(features_file, project_root)
    prompts_path  = resolve(prompts_file, project_root)
    model_path    = resolve(model_path_str, project_root)
    reward_matrix_path = resolve(reward_matrix_out, project_root)
    reward_long_path   = resolve(reward_long_out, project_root)

    print("===================================================================")
    print("BUILD REWARD MATRIX WITH XGBOOST PREDICTIONS (CHUNKED)")
    print("===================================================================")
    print(f"Train file        : {train_path}")
    print(f"Features file     : {features_path}")
    print(f"Prompts file      : {prompts_path}")
    print(f"XGBoost model path: {model_path}")
    print(f"Reward matrix out : {reward_matrix_path}")
    print(f"Reward long out   : {reward_long_path}")
    print("===================================================================\n")

    # ------------------------------------------------------------------
    # 1. Load training data to infer feature columns (inst_* + prompt_*)
    # ------------------------------------------------------------------
    print("[1] Loading training data to infer feature schema...")
    train_df = pd.read_csv(train_path)
    print(f"Train rows: {len(train_df)}")

    inst_cols   = [c for c in train_df.columns if c.startswith("inst_")]
    prompt_cols = [c for c in train_df.columns if c.startswith("prompt_")]
    feature_cols = inst_cols + prompt_cols

    print("Instance feature columns (inst_*):")
    print(inst_cols)
    print("Prompt feature columns (prompt_*):")
    print(prompt_cols)
    print(f"Total feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 2. Load canonical instance & prompt features
    # ------------------------------------------------------------------
    print("\n[2] Loading canonical instance & prompt features...")
    features_df = pd.read_csv(features_path)
    prompts_df  = pd.read_csv(prompts_path)

    # Instance features (by id)
    if "id" not in features_df.columns:
        raise ValueError("features_df must contain an 'id' column (instance_id).")
    instances = features_df.copy()
    instances.rename(columns={"id": "instance_id"}, inplace=True)
    instance_ids = instances["instance_id"].astype(int).tolist()
    print(f"Number of unique instances: {len(instance_ids)}")

    # Prompt features (by id)
    if "id" not in prompts_df.columns:
        raise ValueError("prompts_df must contain an 'id' column (prompt_id).")
    prompts = prompts_df.copy()
    prompts.rename(columns={"id": "prompt_id"}, inplace=True)

    prompt_feature_cols_base = [
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
    missing_prompt_cols = [c for c in prompt_feature_cols_base if c not in prompts.columns]
    if missing_prompt_cols:
        raise ValueError(f"Missing prompt feature columns in prompts_df: {missing_prompt_cols}")

    # ------------------------------------------------------------------
    # 3. Build unique templates from prompt features
    # ------------------------------------------------------------------
    print("\n[3] Building unique prompt templates...")
    tmpl_df = (
        prompts[prompt_feature_cols_base]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    def make_template_label(row):
        paraphrasing    = row["paraphrasing"]
        role_spec       = row["role-specification"]
        reasoning_trig  = row["reasoning-trigger"]
        chain_thought   = row["chain-of-thought"]
        self_check      = row["self-check"]
        conc            = row["conciseness"]
        verb            = row["verbosity"]
        context_exp     = row["context-expansion"]
        few_shot        = row["few-shot-count"]
        length          = row["length"]
        len_bin = "S" if length < 789 else "L"
        return f"{paraphrasing}_{role_spec}_{reasoning_trig}_{chain_thought}_{self_check}_{conc}_{verb}_{context_exp}_{few_shot}_{len_bin}"

    tmpl_df["template"] = tmpl_df.apply(make_template_label, axis=1)

    # ensure one row per template (handles multiple lengths mapping to same bin)
    tmpl_df = tmpl_df.drop_duplicates(subset=["template"]).reset_index(drop=True)

    templates = tmpl_df["template"].tolist()
    print(f"Number of unique templates: {len(templates)}")

    tmpl_rows = tmpl_df.set_index("template")

    # ------------------------------------------------------------------
    # 4. Map canonical instance features to inst_* columns
    # ------------------------------------------------------------------
    instance_feature_cols_base = [
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
    missing_inst_cols = [c for c in instance_feature_cols_base if c not in instances.columns]
    if missing_inst_cols:
        raise ValueError(f"Missing instance feature columns in features_df: {missing_inst_cols}")

    inst_feat = pd.DataFrame(index=instance_ids)
    for base_col in instance_feature_cols_base:
        inst_col = f"inst_{base_col}"
        if inst_col in inst_cols:
            inst_feat[inst_col] = (
                instances.set_index("instance_id")
                .loc[instance_ids, base_col]
                .to_numpy()
            )

    inst_feat = inst_feat.astype(float)
    inst_matrix = inst_feat.to_numpy(dtype=np.float32)
    print("\nInstance feature matrix shape:", inst_matrix.shape)

    # ------------------------------------------------------------------
    # 5. Load XGBoost model
    # ------------------------------------------------------------------
    print("\n[4] Loading XGBoost model...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_path))
    print("✓ Model loaded.")

    # ------------------------------------------------------------------
    # 6. For each template, build (inst_* + prompt_*) block and predict
    # ------------------------------------------------------------------
    print("\n[5] Predicting rewards template-by-template (chunked)...")
    n_instances = len(instance_ids)
    n_templates = len(templates)
    print(f"Total pairs: {n_instances} instances × {n_templates} templates")

    rewards_long_rows = []

    n_inst_feat   = inst_matrix.shape[1]
    n_prompt_feat = len(prompt_cols)
    print("n_inst_feat   =", n_inst_feat)
    print("n_prompt_feat =", n_prompt_feat)

    for idx, tmpl_label in enumerate(templates, start=1):
        if idx % 100 == 0 or idx == 1:
            print(f"  Template {idx}/{n_templates}: {tmpl_label}")

        prompt_feat_row = tmpl_rows.loc[tmpl_label]
        if isinstance(prompt_feat_row, pd.DataFrame):
            prompt_feat_row = prompt_feat_row.iloc[0]

        prompt_vec = []
        for col in prompt_cols:
            base_name = col.replace("prompt_", "")
            if base_name not in prompt_feature_cols_base:
                prompt_vec.append(0.0)
            else:
                val = prompt_feat_row[base_name]
                # handle scalar or length-1 Series robustly
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                prompt_vec.append(float(val))

        prompt_vec = np.array(prompt_vec, dtype=np.float32)

        prompt_block = np.tile(prompt_vec, (n_instances, 1))

        X_block = np.concatenate([inst_matrix, prompt_block], axis=1)
        assert X_block.shape[1] == len(feature_cols), (
            f"Feature dimension mismatch: X_block has {X_block.shape[1]}, "
            f"expected {len(feature_cols)}"
        )

        prob_correct = xgb_model.predict_proba(X_block)[:, 1]

        rewards_long_rows.append(
            pd.DataFrame(
                {
                    "instance_id": instance_ids,
                    "template": tmpl_label,
                    "reward": prob_correct,
                }
            )
        )

    reward_long = pd.concat(rewards_long_rows, axis=0, ignore_index=True)
    print("\nLong-form rewards shape:", reward_long.shape)

    reward_long.to_csv(reward_long_path, index=False)
    print(f"✓ Long-form rewards saved to: {reward_long_path}")

    # ------------------------------------------------------------------
    # 7. Pivot to reward matrix
    # ------------------------------------------------------------------
    print("\n[6] Pivoting to reward matrix...")
    reward_matrix = reward_long.pivot(
        index="instance_id",
        columns="template",
        values="reward",
    ).sort_index(axis=0).sort_index(axis=1)

    reward_matrix.to_csv(reward_matrix_path)
    print(f"✓ Reward matrix saved to: {reward_matrix_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()