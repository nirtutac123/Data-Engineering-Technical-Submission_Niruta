"""
RQ4_Fig2: F1-score by Column Type

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv (labeled)

Output:
- figures/RQ4_Fig2.pdf

Notes:
- Uses fixed thresholds (same as evaluation step):
  - lexical_jaccard >= 0.5
  - embedding_header_only_cosine >= 0.6
  - embedding_header_context_cosine >= 0.6
- Column type is derived from column names (schema-level, explainable).
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


def classify_column(name: str) -> str:
    n = str(name).lower()

    if any(k in n for k in ["id", "code", "number", "no"]):
        return "code_or_id"
    if any(k in n for k in ["date", "time", "year", "month"]):
        return "date_or_time"
    if any(k in n for k in ["lat", "lon", "location", "address", "area"]):
        return "location"
    if any(k in n for k in ["desc", "description", "text", "type"]):
        return "text_or_description"
    return "other"


def compute_f1(truth: pd.Series, preds: pd.Series) -> float:
    truth = truth.astype(int)
    preds = preds.astype(int)

    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ4_Fig2 (F1 by column type)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")

    scores_df = read_csv_robust(scores_path)
    gt_df = read_csv_robust(gt_path)

    gt_df["is_match"] = gt_df["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    need_scores = key_cols + ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]
    for c in need_scores:
        if c not in scores_df.columns:
            raise ValueError(f"Missing column in similarity_scores.csv: {c}")

    df = pd.merge(
        scores_df[need_scores],
        gt_df[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    if len(df) == 0:
        raise ValueError("Merged labeled subset is empty. Check keys / file separators.")

    # Determine column type for the pair
    left_types = df["left_column_final"].apply(classify_column)
    right_types = df["right_column_final"].apply(classify_column)
    df["column_type"] = np.where(left_types == right_types, left_types, "mixed")

    methods = [
        ("lexical_jaccard", 0.5, "Lexical (Jaccard)"),
        ("embedding_header_only_cosine", 0.6, "Embedding (Header-only)"),
        ("embedding_header_context_cosine", 0.6, "Embedding (Header+Context)"),
    ]

    # Compute F1 by type for each method
    types_order = ["code_or_id", "text_or_description", "date_or_time", "location", "other", "mixed"]
    results = []

    for t in types_order:
        sub = df[df["column_type"] == t]
        if len(sub) == 0:
            continue

        for score_col, thr, label in methods:
            preds = (sub[score_col].astype(float) >= thr).astype(int)
            f1 = compute_f1(sub["is_match"], preds)
            results.append({"column_type": t, "method": label, "f1": f1, "n": len(sub)})

    res_df = pd.DataFrame(results)
    if res_df.empty:
        raise ValueError("No results computed for RQ4_Fig2 (unexpected).")

    # Plot grouped bars: types on x, methods as groups
    plt.figure(figsize=(11, 5))

    plotted_types = [t for t in types_order if t in set(res_df["column_type"])]
    x = np.arange(len(plotted_types))
    width = 0.25

    method_labels = [m[2] for m in methods]

    for i, mlabel in enumerate(method_labels):
        vals = []
        for t in plotted_types:
            v = res_df[(res_df["column_type"] == t) & (res_df["method"] == mlabel)]["f1"]
            vals.append(float(v.iloc[0]) if len(v) else 0.0)
        plt.bar(x + (i - 1) * width, vals, width, label=mlabel)

        # value labels
        for j, v in enumerate(vals):
            plt.text(x[j] + (i - 1) * width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, plotted_types, rotation=0)
    plt.ylim(0, 1.05)
    plt.xlabel("Column Type (pair)")
    plt.ylabel("F1-score")
    plt.title("RQ4_Fig2 — F1-score by Column Type (Labeled Subset)")
    plt.legend()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ4_Fig2.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print("[INFO] Sample sizes by type:")
    print(res_df.groupby("column_type")["n"].max().sort_values(ascending=False))


if __name__ == "__main__":
    main()