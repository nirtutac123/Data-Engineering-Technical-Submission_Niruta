"""
RQ5_Fig3: Error Type Distribution

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv (labeled)

Output:
- figures/RQ5_Fig3.pdf

Approach:
- Focus on false positives (is_match=0) where model score is high
- Classify errors using simple explainable rules
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


def tokenize(s: str) -> set:
    return set(str(s).lower().replace("_", " ").split())


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


def classify_error(row: pd.Series) -> str:
    left = str(row["left_column_final"])
    right = str(row["right_column_final"])

    left_type = classify_column(left)
    right_type = classify_column(right)

    tok_overlap = len(tokenize(left).intersection(tokenize(right))) > 0

    # Rule 1: structural mismatch (types disagree)
    if left_type != right_type:
        return "structural_mismatch"

    # Rule 2: token overlap / abbreviation-like
    if tok_overlap and row["lexical_jaccard"] > 0:
        return "abbreviation_or_token_overlap"

    # Rule 3: semantic near miss (embedding high but not actually match)
    if row["embedding_header_context_cosine"] >= 0.6:
        return "semantic_near_miss"

    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ5_Fig3 (Error Type Distribution)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top_n", type=int, default=200, help="Use top-N highest-score false positives")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    scores = read_csv_robust(os.path.join(tables_dir, "similarity_scores.csv"))
    gt = read_csv_robust(os.path.join(tables_dir, "ground_truth_template.csv"))

    gt["is_match"] = gt["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    needed = key_cols + ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]

    df = pd.merge(
        scores[needed],
        gt[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    false_df = df[df["is_match"] == 0].copy()
    if len(false_df) == 0:
        raise ValueError("No false matches found in labeled subset (unexpected).")

    # Focus on high-score false positives (more informative)
    false_df = false_df.sort_values("embedding_header_context_cosine", ascending=False).head(args.top_n)

    false_df["error_type"] = false_df.apply(classify_error, axis=1)
    counts = false_df["error_type"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")

    plt.xlabel("Error Type")
    plt.ylabel("Count (Top high-score false positives)")
    plt.title("RQ5_Fig3 — Error Type Distribution (Rule-based, Explainable)")

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(int(v)), ha="center", va="bottom")

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ5_Fig3.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print(counts)


if __name__ == "__main__":
    main()
