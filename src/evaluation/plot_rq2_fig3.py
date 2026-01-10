"""
RQ2_Fig3: Top-N Match Quality (Precision@K)

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv (manually labeled)

Output:
- figures/RQ2_Fig3.pdf
"""

import os
import argparse
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def precision_at_k(df: pd.DataFrame, score_col: str, k: int) -> float:
    topk = df.sort_values(score_col, ascending=False).head(k)
    if len(topk) == 0:
        return 0.0
    return float(topk["is_match"].sum()) / float(len(topk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ2_Fig3 (Precision@K)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")

    scores_df = read_csv_robust(scores_path)
    gt_df = read_csv_robust(gt_path)

    if "is_match" not in gt_df.columns:
        raise ValueError(f"'is_match' not found in ground truth. Columns: {list(gt_df.columns)}")

    gt_df["is_match"] = gt_df["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    score_cols = ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]

    # Merge to labeled subset
    df = pd.merge(
        scores_df[key_cols + score_cols],
        gt_df[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    if len(df) == 0:
        raise ValueError("Merged labeled dataset is empty. Check key columns alignment.")

    ks = [10, 20, 50, 100]

    methods = [
        ("lexical_jaccard", "Lexical (Jaccard)"),
        ("embedding_header_only_cosine", "Embedding (Header-only)"),
        ("embedding_header_context_cosine", "Embedding (Header+Context)"),
    ]

    # Compute Precision@K
    data = {label: [precision_at_k(df, col, k) for k in ks] for col, label in methods}

    # Plot grouped bars
    x = np.arange(len(ks))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, data["Lexical (Jaccard)"], width, label="Lexical (Jaccard)")
    plt.bar(x, data["Embedding (Header-only)"], width, label="Embedding (Header-only)")
    plt.bar(x + width, data["Embedding (Header+Context)"], width, label="Embedding (Header+Context)")

    plt.xticks(x, [str(k) for k in ks])
    plt.ylim(0, 1.05)
    plt.xlabel("K (Top-N predictions)")
    plt.ylabel("Precision@K")
    plt.title("RQ2_Fig3 — Top-N Match Quality (Precision@K) on Labeled Subset")
    plt.legend()

    # Value labels
    for i, k in enumerate(ks):
        plt.text(i - width, data["Lexical (Jaccard)"][i], f"{data['Lexical (Jaccard)'][i]:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(i, data["Embedding (Header-only)"][i], f"{data['Embedding (Header-only)'][i]:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(i + width, data["Embedding (Header+Context)"][i], f"{data['Embedding (Header+Context)'][i]:.2f}", ha="center", va="bottom", fontsize=9)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ2_Fig3.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print("[INFO] Precision@K values:")
    for label, vals in data.items():
        print(f"  - {label}: " + ", ".join([f"K={k}:{v:.3f}" for k, v in zip(ks, vals)]))


if __name__ == "__main__":
    main()