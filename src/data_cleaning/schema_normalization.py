"""
Schema Cleaning & Normalization

Input:
- data/sample/*_schema_sample.csv (produced in Phase A)

Output:
- data/sample/*_schema_normalized.csv (produced in Phase B)

Rules:
- Use config.yaml for all paths and filenames (no hardcoding).
- Schema-level only (do not process full raw datasets).
"""

from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yaml


# -----------------------------
# Config models
# -----------------------------
@dataclass
class PathsConfig:
    sample_dir: str


@dataclass
class OutputConfig:
    chicago_schema: str
    sf_schema: str
    la_schema: str


@dataclass
class ProjectConfig:
    root: str
    paths: PathsConfig
    output: OutputConfig


def load_config(config_path: str) -> ProjectConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ProjectConfig(
        root=raw["project"]["root"],
        paths=PathsConfig(sample_dir=raw["paths"]["sample_dir"]),
        output=OutputConfig(
            chicago_schema=raw["output"]["chicago_schema"],
            sf_schema=raw["output"]["sf_schema"],
            la_schema=raw["output"]["la_schema"],
        ),
    )


def resolve_path(project_root: str, *parts: str) -> str:
    return os.path.normpath(os.path.join(project_root, *parts))


# -----------------------------
# Normalization logic
# -----------------------------
def normalize_column_name(col: str) -> Dict[str, str]:
    """
    Standard normalization:
    - strip leading/trailing whitespace
    - lowercase
    - replace punctuation with underscores
    - collapse multiple underscores
    - remove leading/trailing underscores

    Returns a dict with:
    - normalized
    - steps (human-readable transformation trace)
    """
    original = col
    steps = []

    s = col

    s2 = s.strip()
    if s2 != s:
        steps.append("trim_whitespace")
    s = s2

    s2 = s.lower()
    if s2 != s:
        steps.append("lowercase")
    s = s2

    # Replace non-alphanumeric characters with underscore
    s2 = re.sub(r"[^a-z0-9]+", "_", s)
    if s2 != s:
        steps.append("replace_special_chars")
    s = s2

    # Collapse multiple underscores
    s2 = re.sub(r"_+", "_", s)
    if s2 != s:
        steps.append("collapse_underscores")
    s = s2

    # Remove leading/trailing underscores
    s2 = s.strip("_")
    if s2 != s:
        steps.append("strip_edge_underscores")
    s = s2

    return {
        "original": original,
        "normalized": s,
        "steps": " | ".join(steps) if steps else "no_change",
    }


def make_normalized_output_name(input_name: str) -> str:
    """
    Convert:
      chicago_schema_sample.csv -> chicago_schema_normalized.csv
    """
    if input_name.endswith("_sample.csv"):
        return input_name.replace("_sample.csv", "_normalized.csv")
    if input_name.endswith(".csv"):
        return input_name.replace(".csv", "_normalized.csv")
    return f"{input_name}_normalized.csv"


def process_one(schema_path: str) -> pd.DataFrame:
    df = pd.read_csv(schema_path)

    # Expected Phase A column name
    if "original_column_name" not in df.columns:
        raise ValueError(f"Missing 'original_column_name' in: {schema_path}")

    normalized_records = []
    for c in df["original_column_name"].astype(str).tolist():
        out = normalize_column_name(c)
        normalized_records.append(out)

    norm_df = pd.DataFrame(normalized_records)

    # Attach to original schema table (row-aligned by original_column_name order)
    out_df = df.copy()
    out_df["normalized_column_name"] = norm_df["normalized"]
    out_df["normalization_steps"] = norm_df["steps"]

    # Basic quality checks
    out_df["normalized_is_empty"] = out_df["normalized_column_name"].astype(str).str.len() == 0
    out_df["normalized_is_duplicate"] = out_df["normalized_column_name"].duplicated(keep=False)

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: Schema Cleaning & Normalization (B1)")
    parser.add_argument("--city", required=True, choices=["chicago", "sf", "la"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_dir = resolve_path(cfg.root, cfg.paths.sample_dir)

    key_to_input = {
        "chicago": cfg.output.chicago_schema,
        "sf": cfg.output.sf_schema,
        "la": cfg.output.la_schema,
    }

    input_name = key_to_input[args.city]
    input_path = resolve_path(sample_dir, input_name)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input schema sample not found: {input_path}")

    out_df = process_one(input_path)

    output_name = make_normalized_output_name(input_name)
    output_path = resolve_path(sample_dir, output_name)

    os.makedirs(sample_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] Normalized schema saved: {output_path}")
    print(f"[INFO] Rows: {len(out_df)}")
    print(f"[INFO] Empty normalized names: {int(out_df['normalized_is_empty'].sum())}")
    print(f"[INFO] Duplicate normalized names: {int(out_df['normalized_is_duplicate'].sum())}")


if __name__ == "__main__":
    main()