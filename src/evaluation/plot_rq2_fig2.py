"""
RQ2_Fig2: Similarity Score Distributions (True vs False)

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv (manually labeled)

Output:
- figures/RQ2_Fig2.pdf

Notes:
- Uses only labeled subset (ground truth template).
- Plots distributions for lexical and embedding scores.
"""

import os
import argparse
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ2_Fig2 (score distributions)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")

    scores_df = read_csv_robust(scores_path)
    gt_df = read_csv_robust(gt_path)

    # Ensure label column
    if "is_match" not in gt_df.columns:
        raise ValueError(f"'is_match' not found in ground truth. Columns: {list(gt_df.columns)}")

    gt_df["is_match"] = gt_df["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    needed_scores = set(key_cols + ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"])
    missing = needed_scores - set(scores_df.columns)
    if missing:
        raise ValueError(f"Missing columns in similarity_scores.csv: {missing}")

    # Merge to get labeled subset only
    df = pd.merge(
        scores_df[list(needed_scores)],
        gt_df[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    true_df = df[df["is_match"] == 1]
    false_df = df[df["is_match"] == 0]

    if len(df) == 0:
        raise ValueError("Merged labeled dataset is empty. Check key columns alignment.")

    # Plot: 3 stacked axes in one PDF figure
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)

    plots = [
        ("lexical_jaccard", "Lexical Jaccard Score"),
        ("embedding_header_only_cosine", "Embedding Cosine (Header-only)"),
        ("embedding_header_context_cosine", "Embedding Cosine (Header+Context)"),
    ]

    bins = 20

    for ax, (col, title) in zip(axes, plots):
        ax.hist(false_df[col].astype(float), bins=bins, alpha=0.7, density=True, label="False matches (0)")
        ax.hist(true_df[col].astype(float), bins=bins, alpha=0.7, density=True, label="True matches (1)")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("RQ2_Fig2 — Score Distributions for True vs False Matches (Labeled Subset)", y=0.98)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ2_Fig2.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print(f"[INFO] Labeled rows used: {len(df)} | True matches: {len(true_df)} | False matches: {len(false_df)}")


if __name__ == "__main__":
    main()