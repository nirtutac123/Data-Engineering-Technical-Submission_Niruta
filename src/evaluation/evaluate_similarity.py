"""
Phase E2 - Evaluation Metrics (Delimiter-Aware)

Purpose:
- Compute precision, recall, F1-score using manual ground truth labels
- Robustly reads CSVs that may be comma- or semicolon-separated
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Try comma first, then semicolon.
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            # likely wrong delimiter
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=";")


def compute_metrics(df: pd.DataFrame, score_col: str, threshold: float) -> dict:
    preds = (df[score_col] >= threshold).astype(int)
    truth = df["is_match"].astype(int)

    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E2: Evaluation Metrics")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")

    scores_df = read_csv_robust(scores_path)
    gt_df = read_csv_robust(gt_path)

    # Ensure numeric labels (empty -> 0)
    if "is_match" not in gt_df.columns:
        raise ValueError(f"'is_match' column not found in ground truth file. Columns: {list(gt_df.columns)}")

    gt_df["is_match"] = gt_df["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]

    # Validate columns exist
    missing_scores = set(key_cols) - set(scores_df.columns)
    missing_gt = set(key_cols) - set(gt_df.columns)
    if missing_scores:
        raise ValueError(f"Missing key columns in similarity_scores.csv: {missing_scores}")
    if missing_gt:
        raise ValueError(f"Missing key columns in ground_truth_template.csv: {missing_gt}")

    df = pd.merge(scores_df, gt_df[key_cols + ["is_match"]], on=key_cols, how="inner")

    results = []

    results.append({"method": "lexical_jaccard", "threshold": 0.5, **compute_metrics(df, "lexical_jaccard", 0.5)})
    results.append({"method": "embedding_header_only", "threshold": 0.6, **compute_metrics(df, "embedding_header_only_cosine", 0.6)})
    results.append({"method": "embedding_header_context", "threshold": 0.6, **compute_metrics(df, "embedding_header_context_cosine", 0.6)})

    out_df = pd.DataFrame(results)
    output_path = os.path.join(tables_dir, "evaluation_metrics.csv")
    out_df.to_csv(output_path, index=False)

    print(f"[OK] Evaluation metrics saved: {output_path}")
    print(out_df)


if __name__ == "__main__":
    main()