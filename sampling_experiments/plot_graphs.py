import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load results CSV
df = pd.read_csv("covariance_sampling_results.csv")

# Extract sample_fraction from sampling_method like "frac20" -> 0.20
def extract_fraction(tag):
    if isinstance(tag, str) and tag.startswith("frac"):
        try:
            return int(tag.replace("frac", "")) / 100
        except Exception:
            return None
    return None

df["sample_fraction"] = df["sampling_method"].apply(extract_fraction)

# Drop rows where we couldn't parse the fraction
df = df.dropna(subset=["sample_fraction"]).copy()

# -------------------------------------------------------------------
# Keep:
#   - For XGBoost / ANN: feature_mode == "all"
#   - For BERT: whatever feature_mode it has (typically "text_only")
# -------------------------------------------------------------------
non_bert = df["model_family"] != "bert"
bert_mask = df["model_family"] == "bert"

df_non_bert = df[non_bert & (df["feature_mode"] == "all")]
df_bert = df[bert_mask]  # no feature_mode filter

df_plot = pd.concat([df_non_bert, df_bert], ignore_index=True)

# Compute average accuracy per (model_family, sample_fraction)
avg = (
    df_plot.groupby(["model_family", "sample_fraction"])["test_accuracy"]
    .mean()
    .reset_index()
)

# Plot one curve per model family
for model in avg["model_family"].unique():
    sub = avg[avg["model_family"] == model].sort_values("sample_fraction")

    plt.figure(figsize=(7, 5))
    plt.plot(sub["sample_fraction"], sub["test_accuracy"], marker="o", linewidth=2)
    plt.title(f"Accuracy vs Sample Fraction ({model})")
    plt.xlabel("Sample Fraction")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/accuracy_vs_sample_fraction_{model}.png")
    plt.close()