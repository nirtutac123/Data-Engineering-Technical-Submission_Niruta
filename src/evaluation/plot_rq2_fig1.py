"""
RQ2_Fig1: Lexical vs Best Embedding (F1 Gap)

Input:
- tables/evaluation_metrics.csv

Output:
- figures/RQ2_Fig1.pdf

Logic:
- Compare Lexical (Jaccard) F1 against the best embedding variant (higher F1).
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
    parser = argparse.ArgumentParser(description="Plot RQ2_Fig1 (Lexical vs Best Embedding)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    metrics_path = os.path.join(tables_dir, "evaluation_metrics.csv")
    df = read_csv_robust(metrics_path)

    required = {"method", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in evaluation_metrics.csv: {missing}. Found: {list(df.columns)}")

    df["method"] = df["method"].astype(str)

    # Lexical row
    lex = df[df["method"] == "lexical_jaccard"]
    if lex.empty:
        raise ValueError("Method 'lexical_jaccard' not found in evaluation_metrics.csv")
    lexical_f1 = float(lex["f1"].iloc[0])

    # Best embedding row
    emb = df[df["method"].isin(["embedding_header_only", "embedding_header_context"])].copy()
    if emb.empty:
        raise ValueError("No embedding methods found (embedding_header_only / embedding_header_context).")

    best_emb_row = emb.sort_values("f1", ascending=False).iloc[0]
    best_emb_method = str(best_emb_row["method"])
    best_emb_f1 = float(best_emb_row["f1"])

    labels = ["Lexical (Jaccard)", f"Best Embedding\n({best_emb_method})"]
    values = [lexical_f1, best_emb_f1]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("RQ2_Fig1 — Lexical vs Best Embedding (F1-score)")
    plt.ylabel("F1-score")
    plt.xlabel("Approach")

    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    # Optional gap annotation
    gap = values[0] - values[1]
    plt.figtext(0.5, 0.01, f"F1 Gap (Lexical - Embedding) = {gap:.3f}", ha="center")

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ2_Fig1.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print(f"[INFO] Lexical F1: {lexical_f1:.4f}")
    print(f"[INFO] Best embedding: {best_emb_method} (F1={best_emb_f1:.4f})")


if __name__ == "__main__":
    main()