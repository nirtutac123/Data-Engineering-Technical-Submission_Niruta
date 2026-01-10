"""
RQ4: Create Column Type Table for True Matches

Purpose:
- Classify true matched columns into semantic types
- Support RQ4 figures

Input:
- tables/ground_truth_template.csv

Output:
- tables/true_match_column_types.csv
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_column(name: str) -> str:
    n = name.lower()

    if any(k in n for k in ["id", "code", "number", "no"]):
        return "code_or_id"
    if any(k in n for k in ["date", "time", "year", "month"]):
        return "date_or_time"
    if any(k in n for k in ["lat", "lon", "location", "address", "area"]):
        return "location"
    if any(k in n for k in ["desc", "description", "text", "type"]):
        return "text_or_description"

    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create true match column type table (RQ4)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = cfg["paths"]["tables_dir"]

    gt_path = os.path.join(tables_dir, "ground_truth_template.csv")
    df = pd.read_csv(gt_path, sep=None, engine="python")

    df["is_match"] = df["is_match"].fillna(0).replace("", 0).astype(int)
    true_df = df[df["is_match"] == 1].copy()

    rows = []
    for _, r in true_df.iterrows():
        rows.append(
            {
                "column_name": r["left_column_final"],
                "column_type": classify_column(str(r["left_column_final"])),
            }
        )
        rows.append(
            {
                "column_name": r["right_column_final"],
                "column_type": classify_column(str(r["right_column_final"])),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(tables_dir, "true_match_column_types.csv")
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Saved: {out_path}")
    if len(out_df) > 0 and "column_type" in out_df.columns:
        print(out_df["column_type"].value_counts())
    else:
        print("No true matches found - empty output file created")


if __name__ == "__main__":
    main()