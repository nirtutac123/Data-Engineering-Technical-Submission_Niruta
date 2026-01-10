"""
RQ1_Fig2: Precision vs Recall by Method

Input:
- tables/evaluation_metrics.csv

Output:
- figures/RQ1_Fig2.pdf
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


def prettify(method: str) -> str:
    return {
        "lexical_jaccard": "Lexical (Jaccard)",
        "embedding_header_only": "Embedding (Header-only)",
        "embedding_header_context": "Embedding (Header+Context)",
    }.get(method, method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ1_Fig2 (Precision vs Recall)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    df = read_csv_robust(os.path.join(tables_dir, "evaluation_metrics.csv"))

    order = ["lexical_jaccard", "embedding_header_only", "embedding_header_context"]
    df = df[df["method"].isin(order)].copy()
    df["label"] = df["method"].map(prettify)

    df["rank"] = df["method"].apply(lambda x: order.index(x))
    df = df.sort_values("rank")

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, df["precision"], width, label="Precision")
    plt.bar(x + width/2, df["recall"], width, label="Recall")

    plt.xticks(x, df["label"])
    plt.ylabel("Score")
    plt.xlabel("Method")
    plt.title("RQ1_Fig2 — Precision vs Recall by Method")
    plt.legend()

    # value labels
    for i, (p, r) in enumerate(zip(df["precision"], df["recall"])):
        plt.text(i - width/2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(i + width/2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "RQ1_Fig2.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {output_path}")


if __name__ == "__main__":
    main()