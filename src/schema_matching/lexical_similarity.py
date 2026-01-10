"""
Lexical Similarity Computation

Purpose:
- Compute baseline lexical similarity scores
- No embeddings, no ML
- Supports RQ1 (lexical vs embedding comparison)
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return set(text.lower().replace("_", " ").split())


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    return len(intersection) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase D1: Lexical Similarity")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    input_path = os.path.join(tables_dir, "schema_pairs_repr.csv")
    df = pd.read_csv(input_path)

    scores = []

    for _, row in df.iterrows():
        left = row["left_header_only"]
        right = row["right_header_only"]

        scores.append(
            {
                "left_city": row["left_city"],
                "right_city": row["right_city"],
                "left_column": row["left_column_final"],
                "right_column": row["right_column_final"],
                "lexical_jaccard": jaccard_similarity(left, right),
                "exact_match": int(left == right),
            }
        )

    score_df = pd.DataFrame(scores)

    output_path = os.path.join(tables_dir, "similarity_scores_lexical.csv")
    score_df.to_csv(output_path, index=False)

    print(f"[OK] Lexical similarity scores saved: {output_path}")
    print(f"[INFO] Total rows: {len(score_df)}")


if __name__ == "__main__":
    main()