"""
Ground Truth Template Creation

Purpose:
- Create a labeling template for manual ground-truth matching
- Add 'is_match' column (empty) for human annotation
- Sort by embedding_header_context_cosine (most informative) to speed up labeling
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E1: Create Ground Truth Template")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top_n", type=int, default=300, help="Top-N rows per dataset pair to preselect (default: 300)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    df = pd.read_csv(scores_path)

    # Keep only essential columns for labeling
    keep_cols = [
        "left_city",
        "right_city",
        "left_column_final",
        "right_column_final",
        "lexical_jaccard",
        "exact_match",
        "embedding_header_only_cosine",
        "embedding_header_context_cosine",
    ]
    df = df[keep_cols].copy()

    # Preselect top-N per city-pair by context-embedding score
    df["pair_key"] = df["left_city"].astype(str) + "__" + df["right_city"].astype(str)
    df = df.sort_values(["pair_key", "embedding_header_context_cosine"], ascending=[True, False])

    top_df = df.groupby("pair_key", as_index=False).head(args.top_n).copy()

    # Add manual label column
    top_df["is_match"] = ""  # fill with 1/0 manually
    top_df["match_notes"] = ""  # optional notes

    # Output
    output_path = os.path.join(tables_dir, "ground_truth_template.csv")
    top_df.drop(columns=["pair_key"]).to_csv(output_path, index=False)

    print(f"[OK] Ground truth template saved: {output_path}")
    print(f"[INFO] Rows in template: {len(top_df)}")
    print("[INFO] Fill 'is_match' with 1 (match) or 0 (non-match). Leave notes if needed.")


if __name__ == "__main__":
    main()