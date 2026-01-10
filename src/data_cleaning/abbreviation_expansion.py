"""
Abbreviation & Code Expansion

Purpose:
- Expand known abbreviations using an explicit mapping table.
- Do NOT guess meanings.
- Preserve original and normalized column names.
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_abbreviations(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    expanded_names = []
    applied_rules = []

    for name in df["normalized_column_name"]:
        expanded = name
        rules = []

        for _, row in mapping_df.iterrows():
            pattern = row["source_pattern"]
            replacement = row["expanded_form"]

            if pattern in expanded:
                expanded = expanded.replace(pattern, replacement)
                rules.append(pattern)

        expanded_names.append(expanded)
        applied_rules.append(" | ".join(rules) if rules else "no_change")

    df["expanded_column_name"] = expanded_names
    df["abbreviation_rules_applied"] = applied_rules
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B2: Abbreviation Expansion")
    parser.add_argument("--city", required=True, choices=["chicago", "sf", "la"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_dir = cfg["paths"]["sample_dir"]

    input_map = {
        "chicago": cfg["output"]["chicago_schema"].replace("_sample", "_normalized"),
        "sf": cfg["output"]["sf_schema"].replace("_sample", "_normalized"),
        "la": cfg["output"]["la_schema"].replace("_sample", "_normalized"),
    }

    input_path = os.path.join(sample_dir, input_map[args.city])
    mapping_path = os.path.join("tables", "abbreviation_mapping.csv")

    df = pd.read_csv(input_path)
    mapping_df = pd.read_csv(mapping_path)

    out_df = expand_abbreviations(df, mapping_df)

    output_path = input_path.replace("_normalized.csv", "_expanded.csv")
    out_df.to_csv(output_path, index=False)

    print(f"[OK] Abbreviation-expanded schema saved: {output_path}")
    print(f"[INFO] Columns expanded: {(out_df['abbreviation_rules_applied'] != 'no_change').sum()}")


if __name__ == "__main__":
    main()