"""
RQ3_Fig1: Threshold vs Precision

Input:
- tables/threshold_sweep_metrics.csv

Output:
- figures/RQ3_Fig1.pdf
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
    parser = argparse.ArgumentParser(description="Plot RQ3_Fig1 (Threshold vs Precision)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    df = read_csv_robust(os.path.join(tables_dir, "threshold_sweep_metrics.csv"))

    plt.figure(figsize=(9, 5))

    for method, sub in df.groupby("method"):
        plt.plot(
            sub["threshold"],
            sub["precision"],
            marker="o",
            label=method,
        )

    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("RQ3_Fig1 — Threshold vs Precision")
    plt.ylim(0, 1.05)
    plt.legend()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ3_Fig1.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()