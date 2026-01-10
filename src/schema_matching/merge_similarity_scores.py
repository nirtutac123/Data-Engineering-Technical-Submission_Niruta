"""
Merge Similarity Scores (Robust)

Purpose:
- Merge lexical and embedding similarity scores into a single master table
- Handles column-name differences between intermediate outputs
- Output: tables/similarity_scores.csv
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def standardize_key_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Ensure both lexical and embedding tables share the same key columns:
    left_city, right_city, left_column_final, right_column_final
    """
    df = df.copy()

    # city columns are consistent
    required_city_cols = {"left_city", "right_city"}
    if not required_city_cols.issubset(set(df.columns)):
        raise ValueError(f"[{kind}] Missing city key columns. Found: {list(df.columns)}")

    # lexical file uses left_column/right_column (from our earlier script)
    if "left_column_final" not in df.columns and "left_column" in df.columns:
        df = df.rename(columns={"left_column": "left_column_final"})
    if "right_column_final" not in df.columns and "right_column" in df.columns:
        df = df.rename(columns={"right_column": "right_column_final"})

    required = {"left_city", "right_city", "left_column_final", "right_column_final"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{kind}] Missing required key columns: {missing}. Found: {list(df.columns)}")

    # Normalize types to avoid merge issues
    for c in ["left_city", "right_city", "left_column_final", "right_column_final"]:
        df[c] = df[c].astype(str)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase D3: Merge Similarity Scores")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    lexical_path = os.path.join(tables_dir, "similarity_scores_lexical.csv")
    embed_path = os.path.join(tables_dir, "similarity_scores_embedding.csv")

    lex_df = pd.read_csv(lexical_path)
    emb_df = pd.read_csv(embed_path)

    lex_df = standardize_key_columns(lex_df, kind="lexical")
    emb_df = standardize_key_columns(emb_df, kind="embedding")

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]

    merged = pd.merge(lex_df, emb_df, on=key_cols, how="inner")

    # Quick sanity check
    if len(merged) == 0:
        raise ValueError(
            "Merge produced 0 rows. Key columns may not align between files."
        )

    output_path = os.path.join(tables_dir, "similarity_scores.csv")
    merged.to_csv(output_path, index=False)

    print(f"[OK] Master similarity table saved: {output_path}")
    print(f"[INFO] Rows: {len(merged)}")
    print(f"[INFO] Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()