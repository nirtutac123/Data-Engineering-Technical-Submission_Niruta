"""
Schema Pair Generation

Purpose:
- Generate all pairwise column combinations between city datasets
- This defines the candidate space for schema matching
- No similarity computation is done here
"""

import os
import itertools
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(sample_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(sample_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema file not found: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C1: Schema Pair Generation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    sample_dir = cfg["paths"]["sample_dir"]
    tables_dir = cfg["paths"]["tables_dir"]

    schema_files = {
        "chicago": cfg["output"]["chicago_schema"].replace("_sample", "_final"),
        "sf": cfg["output"]["sf_schema"].replace("_sample", "_final"),
        "la": cfg["output"]["la_schema"].replace("_sample", "_final"),
    }

    schemas = {
        city: load_schema(sample_dir, fname)
        for city, fname in schema_files.items()
    }

    rows = []

    # Generate all dataset pairs (no self-pairs)
    for (city_a, df_a), (city_b, df_b) in itertools.combinations(schemas.items(), 2):
        for _, row_a in df_a.iterrows():
            for _, row_b in df_b.iterrows():
                rows.append(
                    {
                        "left_city": city_a,
                        "right_city": city_b,
                        "left_column_original": row_a["original_column_name"],
                        "left_column_final": row_a["expanded_column_name"],
                        "right_column_original": row_b["original_column_name"],
                        "right_column_final": row_b["expanded_column_name"],
                    }
                )

    pairs_df = pd.DataFrame(rows)

    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, "schema_pairs.csv")
    pairs_df.to_csv(output_path, index=False)

    print(f"[OK] Schema pairs generated: {output_path}")
    print(f"[INFO] Total candidate pairs: {len(pairs_df)}")


if __name__ == "__main__":
    main()