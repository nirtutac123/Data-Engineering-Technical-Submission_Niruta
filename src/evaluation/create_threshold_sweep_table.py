"""
RQ3: Create Threshold Sweep Metrics Table

Purpose:
- Compute precision/recall/F1 across multiple thresholds
- Output a long-format table for plotting RQ3 figures

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv

Output:
- tables/threshold_sweep_metrics.csv
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


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


def prettify(method: str) -> str:
    return {
        "lexical_jaccard": "Lexical (Jaccard)",
        "embedding_header_only_cosine": "Embedding (Header-only)",
        "embedding_header_context_cosine": "Embedding (Header+Context)",
    }.get(method, method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create threshold sweep metrics table (RQ3)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--end", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    scores_path = os.path.join(tables_dir, "similarity_scores.csv")
    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")

    scores_df = read_csv_robust(scores_path)
    gt_df = read_csv_robust(gt_path)

    gt_df["is_match"] = gt_df["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    score_cols = ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]

    df = pd.merge(
        scores_df[key_cols + score_cols],
        gt_df[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    thresholds = []
    t = args.start
    while t <= args.end + 1e-9:
        thresholds.append(round(t, 2))
        t += args.step

    rows = []
    for th in thresholds:
        for col in score_cols:
            m = compute_metrics(df, col, th)
            rows.append(
                {
                    "method": prettify(col),
                    "score_column": col,
                    "threshold": th,
                    **m,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(tables_dir, "threshold_sweep_metrics.csv")
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] Rows: {len(out_df)} (methods={len(score_cols)} x thresholds={len(thresholds)})")


if __name__ == "__main__":
    main()