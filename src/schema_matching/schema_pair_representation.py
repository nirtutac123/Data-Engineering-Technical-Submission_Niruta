"""
Matching Candidate Representation

Purpose:
- Create header-only and header+context representations
- Prepare inputs for similarity computation (Phase D)
"""

import os
import argparse
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema_lookup(sample_dir: str, filename: str) -> dict:
    """
    Build lookup:
    expanded_column_name -> metadata_description
    """
    path = os.path.join(sample_dir, filename)
    df = pd.read_csv(path)

    lookup = {}
    for _, row in df.iterrows():
        key = row["expanded_column_name"]
        desc = row.get("metadata_description", "")
        lookup[key] = desc if isinstance(desc, str) else ""

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C2: Schema Pair Representation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    sample_dir = cfg["paths"]["sample_dir"]
    tables_dir = cfg["paths"]["tables_dir"]

    # Load schema pairs
    pairs_path = os.path.join(tables_dir, "schema_pairs.csv")
    pairs_df = pd.read_csv(pairs_path)

    # Load schema lookups
    schema_files = {
        "chicago": cfg["output"]["chicago_schema"].replace("_sample", "_final"),
        "sf": cfg["output"]["sf_schema"].replace("_sample", "_final"),
        "la": cfg["output"]["la_schema"].replace("_sample", "_final"),
    }

    lookups = {
        city: load_schema_lookup(sample_dir, fname)
        for city, fname in schema_files.items()
    }

    # Build representations
    left_headers = []
    right_headers = []
    left_with_ctx = []
    right_with_ctx = []

    for _, row in pairs_df.iterrows():
        lc = row["left_city"]
        rc = row["right_city"]

        l_name = row["left_column_final"]
        r_name = row["right_column_final"]

        l_desc = lookups.get(lc, {}).get(l_name, "")
        r_desc = lookups.get(rc, {}).get(r_name, "")

        left_headers.append(l_name)
        right_headers.append(r_name)

        left_with_ctx.append(
            f"{l_name} | {l_desc}" if l_desc else l_name
        )
        right_with_ctx.append(
            f"{r_name} | {r_desc}" if r_desc else r_name
        )

    pairs_df["left_header_only"] = left_headers
    pairs_df["right_header_only"] = right_headers
    pairs_df["left_header_with_context"] = left_with_ctx
    pairs_df["right_header_with_context"] = right_with_ctx

    # Save
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, "schema_pairs_repr.csv")
    pairs_df.to_csv(output_path, index=False)

    print(f"[OK] Schema pair representations saved: {output_path}")
    print(f"[INFO] Total rows: {len(pairs_df)}")


if __name__ == "__main__":
    main()