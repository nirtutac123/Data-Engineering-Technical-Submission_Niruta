"""
Dataset Registry & Ingestion Validation

Purpose:
- Validate that all selected datasets defined in config.yaml exist.
- Collect lightweight, schema-level ingestion metadata.
- DO NOT load full datasets into memory.
- Output a dataset registry table for documentation and reproducibility.

This script supports Phase A only.
"""

import os
import yaml
import pandas as pd


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_csv_basic_info(csv_path: str) -> dict:
    """
    Extract lightweight file-level information without full ingestion.
    """
    file_size_mb = round(os.path.getsize(csv_path) / (1024 * 1024), 2)

    # Read only header
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    return {
        "file_exists": True,
        "file_size_mb": file_size_mb,
        "num_columns": len(columns),
        "column_names": " | ".join(columns),
    }


def main() -> None:
    config = load_config("config.yaml")

    raw_dir = config["paths"]["raw_external_dir"]
    sample_dir = config["paths"]["sample_dir"]

    registry_rows = []

    for key, ds in config["datasets"].items():
        csv_path = os.path.join(raw_dir, ds["csv"])

        if not os.path.exists(csv_path):
            registry_rows.append(
                {
                    "dataset_key": key,
                    "city": ds["city"],
                    "csv_file": ds["csv"],
                    "file_exists": False,
                    "file_size_mb": None,
                    "num_columns": None,
                    "has_metadata_json": bool(ds.get("metadata_json")),
                }
            )
            continue

        info = get_csv_basic_info(csv_path)

        registry_rows.append(
            {
                "dataset_key": key,
                "city": ds["city"],
                "csv_file": ds["csv"],
                "file_exists": info["file_exists"],
                "file_size_mb": info["file_size_mb"],
                "num_columns": info["num_columns"],
                "has_metadata_json": bool(ds.get("metadata_json")),
            }
        )

    registry_df = pd.DataFrame(registry_rows)

    os.makedirs(sample_dir, exist_ok=True)
    output_path = os.path.join(sample_dir, "dataset_registry.csv")
    registry_df.to_csv(output_path, index=False)

    print(f"[OK] Dataset registry created: {output_path}")


if __name__ == "__main__":
    main()
