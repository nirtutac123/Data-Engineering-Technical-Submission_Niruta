"""
Final Cleaned Schema Table

Purpose:
- Merge normalized + expanded schema information
- Produce one final cleaned schema table per city
- This table is the single schema source for Phase C+
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B3: Final Cleaned Schema Table")
    parser.add_argument("--city", required=True, choices=["chicago", "sf", "la"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_dir = cfg["paths"]["sample_dir"]

    input_file = cfg["output"][f"{args.city}_schema"].replace("_sample", "_expanded")
    input_path = os.path.join(sample_dir, input_file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Expanded schema not found: {input_path}")

    df = pd.read_csv(input_path)

    final_cols = [
        "dataset_key",
        "city",
        "original_column_name",
        "normalized_column_name",
        "expanded_column_name",
        "inferred_dtype",
        "has_metadata_description",
        "metadata_description",
        "flag_ambiguous_or_abbrev",
        "abbreviation_rules_applied",
    ]

    final_df = df[final_cols].copy()

    output_name = input_file.replace("_expanded.csv", "_final.csv")
    output_path = os.path.join(sample_dir, output_name)

    final_df.to_csv(output_path, index=False)

    print(f"[OK] Final cleaned schema saved: {output_path}")
    print(f"[INFO] Columns in final schema: {len(final_df)}")


if __name__ == "__main__":
    main()