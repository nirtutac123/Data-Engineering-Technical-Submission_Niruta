"""
RQ5_Fig2: False Negative Examples (Annotated)

Inputs:
- tables/similarity_scores.csv
- tables/ground_truth_template.csv (labeled)

Output:
- figures/RQ5_Fig2.pdf

Definition:
- False Negative: is_match=1 but model score is low (likely to be missed)

Selection:
- Among true matches, take Top 10 with LOWEST embedding_header_context_cosine
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


def make_table_figure(df: pd.DataFrame, title: str, out_path: str) -> None:
    plt.figure(figsize=(12, 4 + 0.35 * len(df)))
    plt.axis("off")
    tbl = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    plt.title(title, pad=12)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RQ5_Fig2 (False Negative Examples)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    scores = read_csv_robust(os.path.join(tables_dir, "similarity_scores.csv"))
    gt = read_csv_robust(os.path.join(tables_dir, "ground_truth_template.csv"))

    gt["is_match"] = gt["is_match"].fillna(0).replace("", 0).astype(int)

    key_cols = ["left_city", "right_city", "left_column_final", "right_column_final"]
    needed = key_cols + ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]

    df = pd.merge(
        scores[needed],
        gt[key_cols + ["is_match"]],
        on=key_cols,
        how="inner",
    )

    true_df = df[df["is_match"] == 1].copy()
    if len(true_df) == 0:
        raise ValueError("No true matches found in labeled subset (unexpected).")

    # Pick true matches that are likely missed by thresholding (lowest scores)
    true_df = true_df.sort_values("embedding_header_context_cosine", ascending=True).head(args.top_n)

    view = true_df[
        ["left_city", "right_city", "left_column_final", "right_column_final",
         "lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]
    ].copy()

    for c in ["lexical_jaccard", "embedding_header_only_cosine", "embedding_header_context_cosine"]:
        view[c] = view[c].astype(float).round(3)

    view["why_missed_note"] = ""

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "RQ5_Fig2.pdf")
    make_table_figure(
        view,
        title="RQ5_Fig2 — False Negative Examples (True matches with lowest model scores)",
        out_path=out_path,
    )

    print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()