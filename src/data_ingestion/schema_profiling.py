"""
Phase A - Data Ingestion & Schema Profiling

Purpose:
- Read config.yaml (single source of truth).
- Ingest large CSVs using chunk-based reading (no full load).
- Generate schema profiling tables:
  - column names
  - inferred dtype
  - estimated non-null ratio
  - representative sample values
  - metadata descriptions (if available)
  - ambiguous/abbrev column flag (heuristic)
- Save outputs to data/sample/ using config.yaml output filenames.

Notes:
- This is schema-level profiling, not row-level analytics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import yaml


# -----------------------------
# Config structures
# -----------------------------
@dataclass
class SamplingConfig:
    max_unique_values_per_column: int
    chunksize: int
    max_chunks: int
    min_values_to_stop_early: int


@dataclass
class CityDatasetConfig:
    city: str
    csv: str
    metadata_json: Optional[str]


@dataclass
class PathsConfig:
    raw_external_dir: str
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
    datasets: Dict[str, CityDatasetConfig]
    sampling: SamplingConfig
    output: OutputConfig


def load_config(config_path: str) -> ProjectConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = raw["project"]["root"]
    paths = PathsConfig(
        raw_external_dir=raw["paths"]["raw_external_dir"],
        sample_dir=raw["paths"]["sample_dir"],
    )

    datasets: Dict[str, CityDatasetConfig] = {}
    for key, ds in raw["datasets"].items():
        datasets[key] = CityDatasetConfig(
            city=ds["city"],
            csv=ds["csv"],
            metadata_json=ds.get("metadata_json", None),
        )

    sampling = SamplingConfig(
        max_unique_values_per_column=int(raw["sampling"]["max_unique_values_per_column"]),
        chunksize=int(raw["sampling"]["chunksize"]),
        max_chunks=int(raw["sampling"]["max_chunks"]),
        min_values_to_stop_early=int(raw["sampling"]["min_values_to_stop_early"]),
    )

    output = OutputConfig(
        chicago_schema=raw["output"]["chicago_schema"],
        sf_schema=raw["output"]["sf_schema"],
        la_schema=raw["output"]["la_schema"],
    )

    return ProjectConfig(
        root=project_root,
        paths=paths,
        datasets=datasets,
        sampling=sampling,
        output=output,
    )


def resolve_path(project_root: str, *parts: str) -> str:
    return os.path.normpath(os.path.join(project_root, *parts))


# -----------------------------
# Metadata parsing (Socrata-like JSON)
# -----------------------------
def _norm_col(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum())


def load_metadata_descriptions(metadata_path: str) -> Dict[str, str]:
    """
    Load column descriptions from metadata JSON (if present).
    Returns: dict of normalized_column_name -> description
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    candidates: List[Any] = []

    if isinstance(obj, dict):
        if isinstance(obj.get("columns"), list):
            candidates = obj["columns"]
        elif isinstance(obj.get("meta"), dict) and isinstance(obj["meta"].get("view"), dict):
            view = obj["meta"]["view"]
            if isinstance(view.get("columns"), list):
                candidates = view["columns"]

    desc_map: Dict[str, str] = {}
    for col in candidates:
        if not isinstance(col, dict):
            continue
        name = col.get("fieldName") or col.get("name") or col.get("field_name") or col.get("columnName")
        desc = col.get("description") or col.get("fieldDescription") or col.get("desc")
        if isinstance(name, str) and isinstance(desc, str) and desc.strip():
            desc_map[_norm_col(name)] = desc.strip()

    return desc_map


# -----------------------------
# Profiling utilities
# -----------------------------
def is_ambiguous_or_abbrev(col_name: str) -> bool:
    """
    Conservative heuristic to flag ambiguous/abbreviated headers.
    Phase A only flags; later phases may normalize/expand.
    """
    s = col_name.strip()
    if not s:
        return True

    compact = "".join(ch for ch in s if ch.isalnum())
    if len(compact) <= 3:
        return True

    if compact.isupper() and len(compact) <= 6:
        return True

    if any(ch.isdigit() for ch in s):
        return True

    if "_" in s:
        return True

    return False


def collect_schema_profile(
    csv_path: str,
    sampling: SamplingConfig,
) -> Tuple[List[str], Dict[str, str], Dict[str, float], Dict[str, List[str]]]:
    """
    Chunk-based profiling:
    - columns
    - dtype (per column)
    - estimated non-null ratio
    - representative sample values
    """
    chunk_iter = pd.read_csv(csv_path, chunksize=sampling.chunksize, low_memory=False)

    total_rows_scanned = 0
    non_null_counts: Dict[str, int] = {}
    sample_values: Dict[str, List[str]] = {}
    dtype_seen: Dict[str, str] = {}
    columns: Optional[List[str]] = None

    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        if columns is None:
            columns = list(chunk.columns)
            non_null_counts = {c: 0 for c in columns}
            sample_values = {c: [] for c in columns}

        for c in chunk.columns:
            dtype_seen[c] = str(chunk[c].dtype)

        nn = chunk.notna().sum().to_dict()
        for c, v in nn.items():
            non_null_counts[c] += int(v)

        for c in chunk.columns:
            if len(sample_values[c]) >= sampling.max_unique_values_per_column:
                continue

            series = chunk[c].dropna()
            if series.empty:
                continue

            vals = series.astype(str).map(lambda x: x.strip()).tolist()
            for v in vals:
                if not v:
                    continue
                if v not in sample_values[c]:
                    sample_values[c].append(v)
                if len(sample_values[c]) >= sampling.max_unique_values_per_column:
                    break

        total_rows_scanned += len(chunk)

        enough = sum(1 for c in columns if len(sample_values[c]) >= sampling.min_values_to_stop_early)
        if enough == len(columns):
            break

        if chunk_idx >= sampling.max_chunks:
            break

    if columns is None:
        raise ValueError(f"No data read from CSV: {csv_path}")

    non_null_ratio = {c: (non_null_counts[c] / total_rows_scanned if total_rows_scanned else 0.0) for c in columns}
    return columns, dtype_seen, non_null_ratio, sample_values


def build_schema_table(
    city_key: str,
    city_conf: CityDatasetConfig,
    metadata_desc_map: Dict[str, str],
    columns: List[str],
    dtype_seen: Dict[str, str],
    non_null_ratio: Dict[str, float],
    sample_values: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []
    for c in columns:
        norm = _norm_col(c)
        desc = metadata_desc_map.get(norm, "")
        rows.append(
            {
                "dataset_key": city_key,
                "city": city_conf.city,
                "original_column_name": c,
                "inferred_dtype": dtype_seen.get(c, ""),
                "non_null_ratio_est": round(float(non_null_ratio.get(c, 0.0)), 6),
                "sample_values": " | ".join(sample_values.get(c, [])[:20]),
                "has_metadata_description": bool(desc),
                "metadata_description": desc,
                "flag_ambiguous_or_abbrev": is_ambiguous_or_abbrev(c),
            }
        )
    return pd.DataFrame(rows)


def get_output_filename(cfg: ProjectConfig, city_key: str) -> str:
    key_map = {
        "chicago": cfg.output.chicago_schema,
        "sf": cfg.output.sf_schema,
        "la": cfg.output.la_schema,
    }
    if city_key not in key_map:
        raise ValueError(f"Unknown city key: {city_key}")
    return key_map[city_key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: Data Ingestion & Schema Profiling")
    parser.add_argument("--city", required=True, choices=["chicago", "sf", "la"])
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = resolve_path(cfg.root, cfg.paths.raw_external_dir)
    sample_dir = resolve_path(cfg.root, cfg.paths.sample_dir)

    ds = cfg.datasets[args.city]
    csv_path = resolve_path(raw_dir, ds.csv)

    metadata_desc_map: Dict[str, str] = {}
    if ds.metadata_json:
        metadata_path = resolve_path(raw_dir, ds.metadata_json)
        metadata_desc_map = load_metadata_descriptions(metadata_path)

    columns, dtype_seen, non_null_ratio, sample_values = collect_schema_profile(
        csv_path=csv_path,
        sampling=cfg.sampling,
    )

    schema_df = build_schema_table(
        city_key=args.city,
        city_conf=ds,
        metadata_desc_map=metadata_desc_map,
        columns=columns,
        dtype_seen=dtype_seen,
        non_null_ratio=non_null_ratio,
        sample_values=sample_values,
    )

    os.makedirs(sample_dir, exist_ok=True)
    out_name = get_output_filename(cfg, args.city)
    out_path = resolve_path(sample_dir, out_name)

    schema_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Saved schema profile: {out_path}")
    print(f"[INFO] Columns profiled: {len(schema_df)}")
    print(f"[INFO] Metadata attached: {int(schema_df['has_metadata_description'].sum())} columns")


if __name__ == "__main__":
    main()