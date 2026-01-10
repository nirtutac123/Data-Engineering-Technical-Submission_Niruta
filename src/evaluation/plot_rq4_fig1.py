"""
RQ4_Fig1: Match Count by Column Type

Input:
- tables/true_match_column_types.csv

Output:
- figures/RQ4_Fig1.pdf
"""

import os
import argparse
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ4_Fig1 (Match Count by Column Type)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    input_path = os.path.join(tables_dir, "true_match_column_types.csv")
    df = pd.read_csv(input_path)

    if "column_type" not in df.columns:
        raise ValueError(f"'column_type' column not found. Columns: {list(df.columns)}")

    counts = df["column_type"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")

    plt.xlabel("Column Type")
    plt.ylabel("Number of Columns in True Matches")
    plt.title("RQ4_Fig1 — Match Count by Column Type")

    # value labels
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ4_Fig1.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print(counts)


if __name__ == "__main__":
    main()