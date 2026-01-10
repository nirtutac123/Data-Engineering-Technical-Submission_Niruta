"""
RQ4_Fig3: City-to-City Match Distribution

Input:
- tables/ground_truth_template.csv (labeled)

Output:
- figures/RQ4_Fig3.pdf

Shows:
- True vs False labeled counts per city-pair
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ4_Fig3 (city-pair distribution)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")
    df = read_csv_robust(gt_path)

    df["is_match"] = df["is_match"].fillna(0).replace("", 0).astype(int)
    df["pair"] = df["left_city"].astype(str) + " — " + df["right_city"].astype(str)

    counts = df.groupby(["pair", "is_match"]).size().unstack(fill_value=0)
    # ensure both columns exist
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0

    counts = counts[[0, 1]].copy()
    counts.columns = ["False (0)", "True (1)"]
    counts = counts.sort_index()

    x = np.arange(len(counts.index))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, counts["False (0)"], width, label="False (0)")
    plt.bar(x + width/2, counts["True (1)"], width, label="True (1)")

    plt.xticks(x, counts.index)
    plt.xlabel("City Pair")
    plt.ylabel("Count (labeled rows)")
    plt.title("RQ4_Fig3 — Labeled Match Distribution by City Pair")
    plt.legend()

    # labels
    for i, (f, t) in enumerate(zip(counts["False (0)"].values, counts["True (1)"].values)):
        plt.text(i - width/2, f, str(int(f)), ha="center", va="bottom", fontsize=9)
        plt.text(i + width/2, t, str(int(t)), ha="center", va="bottom", fontsize=9)

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ4_Fig3.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")
    print(counts)


if __name__ == "__main__":
    main()