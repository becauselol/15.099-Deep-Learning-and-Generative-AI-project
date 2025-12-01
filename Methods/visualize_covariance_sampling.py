#!/usr/bin/env python
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    # Make sample index numeric (s0 -> 0, etc.) for nicer plotting
    if "sample_tag" in df.columns:
        df["sample_idx"] = (
            df["sample_tag"]
            .astype(str)
            .str.extract(r"s(\d+)", expand=False)
            .astype(int)
        )
    return df


def scatter_covariance_vs_f1(df: pd.DataFrame, outdir: str):
    """Scatter: covariance summary vs test_f1."""
    sns.set(style="whitegrid")

    for cov_col in ["cov_trace", "cov_logdet", "cov_frobenius"]:
        if cov_col not in df.columns:
            continue

        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df,
            x=cov_col,
            y="test_f1",
            hue="model_family",
            style="feature_mode",
            s=60,
        )
        ax.set_title(f"{cov_col} vs test_f1")
        ax.set_xlabel(cov_col)
        ax.set_ylabel("test_f1")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fname = os.path.join(outdir, f"scatter_{cov_col}_vs_test_f1.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved {fname}")


def scatter_by_model_family(df: pd.DataFrame, outdir: str):
    """One scatter per model_family: cov_trace vs test_f1."""
    sns.set(style="whitegrid")
    cov_col = "cov_trace"
    if cov_col not in df.columns:
        return

    model_fams = sorted(df["model_family"].unique())
    for mf in model_fams:
        sub = df[df["model_family"] == mf]

        plt.figure(figsize=(7, 5))
        ax = sns.scatterplot(
            data=sub,
            x=cov_col,
            y="test_f1",
            hue="feature_mode",
            style="sample_tag",
            s=60,
        )
        ax.set_title(f"{mf}: {cov_col} vs test_f1")
        ax.set_xlabel(cov_col)
        ax.set_ylabel("test_f1")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fname = os.path.join(outdir, f"scatter_{cov_col}_vs_test_f1_{mf}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved {fname}")


def line_over_samples(df: pd.DataFrame, outdir: str):
    """Lineplot of test_f1 over samples (sample_idx) per model + feature_mode."""
    if "sample_idx" not in df.columns:
        print("sample_idx not found, skipping line_over_samples")
        return

    sns.set(style="whitegrid")

    # One plot per model_family
    for mf in sorted(df["model_family"].unique()):
        sub = df[df["model_family"] == mf].copy()

        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=sub,
            x="sample_idx",
            y="test_f1",
            hue="feature_mode",
            marker="o",
        )
        ax.set_title(f"{mf}: test_f1 across samples")
        ax.set_xlabel("sample_idx (e.g., s0..s4)")
        ax.set_ylabel("test_f1")
        ax.set_xticks(sorted(sub["sample_idx"].unique()))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        fname = os.path.join(outdir, f"line_test_f1_over_samples_{mf}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved {fname}")


def heatmap_mean_f1(df: pd.DataFrame, outdir: str):
    """Heatmap of mean test_f1 by (model_family, feature_mode)."""
    sns.set(style="white")

    pivot = (
        df.groupby(["model_family", "feature_mode"])["test_f1"]
        .mean()
        .unstack("feature_mode")
        .sort_index()
    )

    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "mean test_f1"},
    )
    ax.set_title("Mean test_f1 by model_family and feature_mode")
    ax.set_xlabel("feature_mode")
    ax.set_ylabel("model_family")
    plt.tight_layout()

    fname = os.path.join(outdir, "heatmap_mean_test_f1.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to covariance_sampling_results.json",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots_covariance_sampling",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.results_file)
    print(df.head())

    scatter_covariance_vs_f1(df, args.outdir)
    scatter_by_model_family(df, args.outdir)
    line_over_samples(df, args.outdir)
    heatmap_mean_f1(df, args.outdir)


if __name__ == "__main__":
    main()