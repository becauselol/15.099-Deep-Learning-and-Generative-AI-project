import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def compute_covariance_stats(X: np.ndarray) -> dict:
    """
    Compute simple covariance-based summary statistics for a feature matrix X.
    Uses only (inst_* + prompt_*) feature columns (already standardized).
    """
    if X.shape[0] < 2:
        return {
            "trace": float("nan"),
            "frobenius_norm": float("nan"),
            "logdet": float("nan"),
        }
    cov = np.cov(X, rowvar=False)
    trace = float(np.trace(cov))
    fro = float(np.linalg.norm(cov, ord="fro"))

    # Robust logdet with diagonal jitter & eigenvalue threshold
    try:
        jitter = 1e-6 * np.eye(cov.shape[0], dtype=cov.dtype)
        eigvals = np.linalg.eigvalsh(cov + jitter)
        eigvals = eigvals[eigvals > 1e-12]
        if eigvals.size == 0:
            logdet = float("nan")
        else:
            logdet = float(np.sum(np.log(eigvals)))
    except Exception:
        logdet = float("nan")

    return {"trace": trace, "frobenius_norm": fro, "logdet": logdet}


def diverse_sample_indices(
    X: np.ndarray, sample_size: int, rng: np.random.RandomState
) -> np.ndarray:
    """
    Greedy k-center / farthest-first sampling in feature space.
    X is assumed to be standardized.
    """
    n = X.shape[0]
    if sample_size >= n:
        return np.arange(n, dtype=int)

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    # Pick first point at random
    first = rng.randint(0, n)
    selected = [first]

    # min_dists[i] = min_{j in selected} ||x_i - x_j||^2
    dist2 = np.sum((X - X[first]) ** 2, axis=1)
    min_dists = dist2.copy()
    min_dists[first] = -1.0  # mark selected

    for _ in range(1, sample_size):
        # Pick the point farthest from the current selected set
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        # Update distances with the new center
        dist2 = np.sum((X - X[next_idx]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dist2)
        min_dists[next_idx] = -1.0  # keep it from being reselected

    return np.array(selected, dtype=int)


def run_xgboost(
    methods_dir: Path,
    output_dir: Path,
    sample_tag: str,
    train_path: Path,
    test_path: Path,
    feature_mode: str,
    cov_stats: dict,
) -> dict:
    """
    Run xgboost_unified.py on a specific sampled train set and feature mode.
    """
    env = os.environ.copy()
    env["FEATURE_MODE"] = feature_mode
    env["TRAIN_DATA_FILE"] = str(train_path)
    env["TEST_DATA_FILE"] = str(test_path)

    xgb_summary_path = output_dir / f"xgb_summary_{sample_tag}_{feature_mode}.json"
    xgb_model_path = output_dir / f"xgb_model_{sample_tag}_{feature_mode}.json"
    env["XGBOOST_SUMMARY_FILE"] = str(xgb_summary_path)
    env["XGBOOST_MODEL_PATH"] = str(xgb_model_path)

    print(f"[*] Training XGBoost (mode={feature_mode}) on sample {sample_tag}")
    subprocess.run(
        [sys.executable, "xgboost_unified.py"],
        cwd=str(methods_dir),
        env=env,
        check=True,
    )

    with open(xgb_summary_path, "r") as f:
        summary = json.load(f)
    test_results = summary.get("test_results", {})

    return {
        "sample_tag": sample_tag,
        "sampling_method": sample_tag.split("_")[0],
        "model_family": "xgboost",
        "feature_mode": feature_mode,
        "cov_trace": cov_stats["trace"],
        "cov_frobenius": cov_stats["frobenius_norm"],
        "cov_logdet": cov_stats["logdet"],
        "test_accuracy": test_results.get("accuracy"),
        "test_precision": test_results.get("precision"),
        "test_recall": test_results.get("recall"),
        "test_f1": test_results.get("f1"),
        "summary_file": str(xgb_summary_path),
        "model_path": summary.get("model_path"),
    }


def run_ann(
    methods_dir: Path,
    output_dir: Path,
    sample_tag: str,
    train_path: Path,
    test_path: Path,
    feature_mode: str,
    cov_stats: dict,
) -> dict:
    """
    Run ann_unified.py on a specific sampled train set and feature mode.
    """
    env = os.environ.copy()
    env["FEATURE_MODE"] = feature_mode
    env["TRAIN_DATA_FILE"] = str(train_path)
    env["TEST_DATA_FILE"] = str(test_path)

    ann_summary_path = output_dir / f"ann_summary_{sample_tag}_{feature_mode}.json"
    ann_model_path = output_dir / f"ann_model_{sample_tag}_{feature_mode}_best.pth"
    env["ANN_SUMMARY_FILE"] = str(ann_summary_path)
    env["ANN_MODEL_PATH"] = str(ann_model_path)

    print(f"[*] Training ANN (mode={feature_mode}) on sample {sample_tag}")
    subprocess.run(
        [sys.executable, "ann_unified.py"],
        cwd=str(methods_dir),
        env=env,
        check=True,
    )

    with open(ann_summary_path, "r") as f:
        summary = json.load(f)
    test_results = summary.get("test_results", {})

    return {
        "sample_tag": sample_tag,
        "sampling_method": sample_tag.split("_")[0],
        "model_family": "ann",
        "feature_mode": feature_mode,
        "cov_trace": cov_stats["trace"],
        "cov_frobenius": cov_stats["frobenius_norm"],
        "cov_logdet": cov_stats["logdet"],
        "test_accuracy": test_results.get("accuracy"),
        "test_precision": test_results.get("precision"),
        "test_recall": test_results.get("recall"),
        "test_f1": test_results.get("f1"),
        "summary_file": str(ann_summary_path),
        "model_path": summary.get("model_path"),
    }


def run_bert(
    methods_dir: Path,
    output_dir: Path,
    sample_tag: str,
    train_path: Path,
    test_path: Path,
    cov_stats: dict,
) -> dict:
    """
    Run bert_model.py on a specific sampled train set.
    (BERT always uses the text prompt; we tag feature_mode as 'text_only' here.)
    """
    env = os.environ.copy()
    env["TRAIN_DATA_FILE"] = str(train_path)
    env["TEST_DATA_FILE"] = str(test_path)

    bert_summary_path = output_dir / f"bert_summary_{sample_tag}.json"
    bert_model_path = output_dir / f"bert_model_{sample_tag}"
    bert_results_dir = output_dir / f"bert_results_{sample_tag}"
    bert_logs_dir = output_dir / f"bert_logs_{sample_tag}"
    env["BERT_SUMMARY_FILE"] = str(bert_summary_path)
    env["BERT_MODEL_PATH"] = str(bert_model_path)
    env["BERT_RESULTS_DIR"] = str(bert_results_dir)
    env["BERT_LOGS_DIR"] = str(bert_logs_dir)

    print(f"[*] Training BERT on sample {sample_tag}")
    subprocess.run(
        [sys.executable, "bert_model.py"],
        cwd=str(methods_dir),
        env=env,
        check=True,
    )

    with open(bert_summary_path, "r") as f:
        summary = json.load(f)
    test_results = summary.get("test_results", {})

    return {
        "sample_tag": sample_tag,
        "sampling_method": sample_tag.split("_")[0],
        "model_family": "bert",
        "feature_mode": "text_only",
        "cov_trace": cov_stats["trace"],
        "cov_frobenius": cov_stats["frobenius_norm"],
        "cov_logdet": cov_stats["logdet"],
        "test_accuracy": test_results.get("accuracy"),
        "test_precision": test_results.get("precision"),
        "test_recall": test_results.get("recall"),
        "test_f1": test_results.get("f1"),
        "summary_file": str(bert_summary_path),
        "model_path": summary.get("model_path"),
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run covariance-based train-set sampling and train all models."
    )
    parser.add_argument(
        "--sampling-method",
        choices=["random", "diverse"],
        default="diverse",
        help="Sampling heuristic for selecting training subsets.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of different training subsets to generate.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.5,
        help="Fraction of the original training data to use in each sample (0,1].",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=[
            "text_only",
            "features_only",
            "inst_only",
            "prompt_only",
            "text_prompt",
            "all",
        ],
        help="Feature modes to train for XGBoost and ANN.",
    )
    parser.add_argument(
        "--include-bert",
        action="store_true",
        help="Also fine-tune the BERT sequence classifier for each sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="sampling_experiments",
        help="Directory (relative to project root) where samples and results will be stored.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()

    methods_dir = Path(__file__).resolve().parent
    project_root = methods_dir.parent

    # Resolve train/test paths from .env, keeping them compatible with existing scripts
    base_train_file = os.getenv(
        "TRAIN_DATA_FILE", "../data_gen/train_countdown_results_with_prompt_gemini.csv"
    )
    base_test_file = os.getenv(
        "TEST_DATA_FILE", "../data_gen/test_countdown_results_with_prompt_gemini.csv"
    )

    if os.path.isabs(base_train_file):
        base_train_path = Path(base_train_file)
    else:
        base_train_path = (methods_dir / base_train_file).resolve()

    if os.path.isabs(base_test_file):
        base_test_path = Path(base_test_file)
    else:
        base_test_path = (methods_dir / base_test_file).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print("COVARIANCE-BASED TRAINING SET SAMPLING PIPELINE")
    print("===================================================================")
    print(f"Base train file: {base_train_path}")
    print(f"Base test file:  {base_test_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Sampling method: {args.sampling_method}")
    print(f"Num samples:     {args.num_samples}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Feature modes:   {args.feature_modes}")
    print(f"Include BERT:    {args.include_bert}")
    print("===================================================================")

    # Load base train/test
    train_df = pd.read_csv(base_train_path)
    test_df = pd.read_csv(base_test_path)

    # Column sanity check (same condition as in existing scripts)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    if train_cols != test_cols:
        raise ValueError(
            "Column mismatch between train and test datasets!\n"
            f"Train only: {train_cols - test_cols}\n"
            f"Test only: {test_cols - train_cols}"
        )

    n_train = len(train_df)
    if not (0 < args.sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be in (0, 1].")
    sample_size = max(2, int(round(args.sample_fraction * n_train)))
    sample_size = min(sample_size, n_train)

    print(f"Full train size: {n_train}")
    print(f"Per-sample size: {sample_size}")
    print()

    # Build feature matrix for covariance (instance + prompt template features only)
    feature_cols = [
        c for c in train_df.columns if c.startswith("inst_") or c.startswith("prompt_")
    ]
    if not feature_cols:
        raise ValueError(
            "No instance/prompt feature columns found (inst_* or prompt_*). "
            "Cannot compute covariance-based sampling."
        )

    X_full = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    # Standardize to equalize scales
    mean = X_full.mean(axis=0, keepdims=True)
    std = X_full.std(axis=0, keepdims=True) + 1e-8
    X_std = (X_full - mean) / std

    rng = np.random.RandomState(args.random_state)

    all_results = []

    for sample_idx in range(args.num_samples):
        method_tag = args.sampling_method
        sample_tag = f"{method_tag}_s{sample_idx}"

        if args.sampling_method == "random":
            indices = rng.choice(n_train, size=sample_size, replace=False)
        else:  # "diverse"
            indices = diverse_sample_indices(X_std, sample_size, rng)

        indices = np.sort(indices)
        X_sample = X_std[indices]

        cov_stats = compute_covariance_stats(X_sample)
        print(
            f"[+] Sample {sample_tag}: n={len(indices)}, "
            f"trace={cov_stats['trace']:.4f}, "
            f"fro={cov_stats['frobenius_norm']:.4f}, "
            f"logdet={cov_stats['logdet']:.4f}"
        )

        # Save sampled train set and index list
        sample_train_df = train_df.iloc[indices].reset_index(drop=True)
        sample_train_path = output_dir / f"train_{sample_tag}.csv"
        sample_indices_path = output_dir / f"indices_{sample_tag}.npy"
        sample_train_df.to_csv(sample_train_path, index=False)
        np.save(sample_indices_path, indices)

        # Train XGBoost & ANN for all requested feature modes
        for mode in args.feature_modes:
            xgb_res = run_xgboost(
                methods_dir=methods_dir,
                output_dir=output_dir,
                sample_tag=sample_tag,
                train_path=sample_train_path,
                test_path=base_test_path,
                feature_mode=mode,
                cov_stats=cov_stats,
            )
            all_results.append(xgb_res)

            ann_res = run_ann(
                methods_dir=methods_dir,
                output_dir=output_dir,
                sample_tag=sample_tag,
                train_path=sample_train_path,
                test_path=base_test_path,
                feature_mode=mode,
                cov_stats=cov_stats,
            )
            all_results.append(ann_res)

        # Optional: train BERT
        if args.include_bert:
            bert_res = run_bert(
                methods_dir=methods_dir,
                output_dir=output_dir,
                sample_tag=sample_tag,
                train_path=sample_train_path,
                test_path=base_test_path,
                cov_stats=cov_stats,
            )
            all_results.append(bert_res)

    # Save aggregated results
    results_path_csv = output_dir / "covariance_sampling_results.csv"
    results_path_json = output_dir / "covariance_sampling_results.json"

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path_csv, index=False)
    with open(results_path_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print("===================================================================")
    print("EXPERIMENTS COMPLETE")
    print(f"Results CSV:  {results_path_csv}")
    print(f"Results JSON: {results_path_json}")
    print("===================================================================")


if __name__ == "__main__":
    main()