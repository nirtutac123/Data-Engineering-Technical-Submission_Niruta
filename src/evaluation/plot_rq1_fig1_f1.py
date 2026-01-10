"""
RQ1_Fig1: Overall F1-score Comparison

Input:
- tables/evaluation_metrics.csv

Output (PDF):
- figures/RQ1_Fig1_f1_comparison.pdf

Rules:
- Use config.yaml for all paths (no hardcoding).
- Save as PDF only.
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
    # handle comma or semicolon separated CSVs
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def prettify_method_name(method: str) -> str:
    mapping = {
        "lexical_jaccard": "Lexical (Jaccard)",
        "embedding_header_only": "Embedding (Header-only)",
        "embedding_header_context": "Embedding (Header+Context)",
    }
    return mapping.get(method, method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ1_Fig1 (F1 comparison)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    input_path = os.path.join(tables_dir, "evaluation_metrics.csv")
    df = read_csv_robust(input_path)

    required_cols = {"method", "f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in evaluation_metrics.csv: {missing}. Found: {list(df.columns)}")

    # Keep the three methods we care about (stable order for presentation)
    order = ["lexical_jaccard", "embedding_header_only", "embedding_header_context"]
    df = df[df["method"].astype(str).isin(order)].copy()
    df["method"] = df["method"].astype(str)
    df["method_label"] = df["method"].map(prettify_method_name)

    # sort in the predefined order
    df["method_rank"] = df["method"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("method_rank")

    # Plot
    plt.figure(figsize=(9, 5))
    plt.bar(df["method_label"], df["f1"])
    plt.title("RQ1_Fig1 — Overall F1-score Comparison (Ground Truth Evaluation)")
    plt.xlabel("Method")
    plt.ylabel("F1-score")

    # add value labels
    for i, v in enumerate(df["f1"].tolist()):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "RQ1_Fig1_f1_comparison.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {output_path}")


if __name__ == "__main__":
    main()