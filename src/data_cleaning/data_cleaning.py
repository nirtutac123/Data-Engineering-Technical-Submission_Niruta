
#!/usr/bin/env python3
"""
Project 22 – Schema Matching Automation using Embeddings for Tabular Data Integration
-----------------------------------------------------------------------------------
File: data_cleaning.py
Version: 2026-01-02T13:20+01:00

Cities supported:
- chicago: CPD incident files (CSV)
- la: auto-detect crimes vs arrests (CSV)
- la_crimes: LAPD crime-data-from-2010-to-present.csv (CSV)
- la_arrests: LAPD arrest-data-from-2010-to-present.csv (CSV)
- sf_radio: San Francisco Radio Codes 2016.xlsx (Excel → structured codebook)

Outputs:
- Cleaned CSV (+ optional Parquet if pyarrow/fastparquet installed)
- Compact JSON profile (null %, dtypes)
- For sf_radio: parsed Excel (multi-sheet .xlsx) + JSON codebook
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__version__ = "2026-01-02T13:20+01:00"

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(_handler)

# --------------------------------------------------------------------------------------
# Canonical schema (column intents)
# --------------------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "incident_id", "case_number", "incident_datetime", "block_address", "iucr_code",
    "crime_category", "crime_subtype", "location_type", "arrest_made", "domestic_flag",
    "beat", "district", "ward", "community_area", "fbi_code", "x_coord", "y_coord",
    "year", "updated_on", "latitude", "longitude", "location_point"
]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def robust_read_csv(input_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Robust CSV reader (handles messy quoting, backslashes; skips malformed lines)."""
    logger.info(f"Reading CSV (robust, skipping malformed lines): {input_path} [nrows={nrows}]")
    return pd.read_csv(
        input_path,
        sep=",",
        engine="python",
        dtype=str,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        quotechar='"',
        on_bad_lines="skip",
        nrows=nrows
    )

def drop_index_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if (str(c).strip() == "" or str(c).startswith("Unnamed"))]
    if to_drop:
        logger.info(f"Dropping index artifact columns: {to_drop}")
        df = df.drop(columns=to_drop)
    return df

def standardize_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True, "false": False, "t": True, "f": False, "1": True, "0": False,
        "y": True, "n": False, "yes": True, "no": False
    }
    return s.map(mapping)

def to_int64(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def parse_datetime(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    # If format known, pass fmt for faster consistent parsing; else let pandas infer.
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def clean_text(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("\\\\", "", regex=True)   # drop backslashes
    s = s.str.replace('"', '', regex=False)
    s = s.str.replace("\u2013", "-", regex=False).str.replace("\u2014", "-", regex=False)
    s = s.str.strip()
    s = s.replace({"nan": pd.NA, "": pd.NA})
    return s

def profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    profile = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": {}}
    for c in df.columns:
        profile["columns"][c] = {
            "dtype": str(df[c].dtype),
            "null_pct": round(float(df[c].isna().mean() * 100.0), 2)
        }
    return profile

def ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def finalize_types(df: pd.DataFrame) -> pd.DataFrame:
    # booleans
    df["arrest_made"] = standardize_bool(df["arrest_made"])
    df["domestic_flag"] = standardize_bool(df["domestic_flag"])

    # integer admin fields
    for c in ["beat", "district", "ward", "community_area", "year"]:
        df[c] = to_int64(df[c])

    # numeric coords
    for c in ["x_coord", "y_coord", "latitude", "longitude"]:
        df[c] = to_float(df[c])

    # datetimes
    df["incident_datetime"] = parse_datetime(df["incident_datetime"])
    df["updated_on"] = parse_datetime(df["updated_on"])

    # clean text
    for c in ["block_address", "iucr_code", "crime_category", "crime_subtype",
              "location_type", "fbi_code", "location_point"]:
        df[c] = clean_text(df[c])
    return df

def deduplicate(df: pd.DataFrame, key_priority: Optional[List[List[str]]] = None) -> pd.DataFrame:
    if key_priority is None:
        key_priority = [
            ["incident_id", "case_number", "incident_datetime"],
            ["case_number", "incident_datetime"],
            ["incident_id"],
        ]
    before = len(df)
    for keys in key_priority:
        present = [k for k in keys if k in df.columns]
        if present and len(present) == len(keys):
            df = df.drop_duplicates(subset=keys)
            break
    after = len(df)
    if before != after:
        logger.info(f"Dropped {before - after} duplicates")
    return df

def build_datetime_from_date_hhmm(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """Combine date string (varied) + HHMM numeric/str into a timestamp string."""
    t = time_s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(4)
    hhmm = t.str.slice(0, 2) + ":" + t.str.slice(2, 4)
    return (date_s.astype(str) + " " + hhmm).str.strip()

# --------------------------------------------------------------------------------------
# CHICAGO cleaner (already working)
# --------------------------------------------------------------------------------------
CHI_COLS_MAP = {
    'ID': 'incident_id',
    'Case Number': 'case_number',
    'Date': 'incident_datetime',
    'Block': 'block_address',
    'IUCR': 'iucr_code',
    'Primary Type': 'crime_category',
    'Description': 'crime_subtype',
    'Location Description': 'location_type',
    'Arrest': 'arrest_made',
    'Domestic': 'domestic_flag',
    'Beat': 'beat',
    'District': 'district',
    'Ward': 'ward',
    'Community Area': 'community_area',
    'FBI Code': 'fbi_code',
    'X Coordinate': 'x_coord',
    'Y Coordinate': 'y_coord',
    'Year': 'year',
    'Updated On': 'updated_on',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Location': 'location_point'
}

def clean_chicago(df: pd.DataFrame) -> pd.DataFrame:
    keep = {k: v for k, v in CHI_COLS_MAP.items() if k in df.columns}
    df = df.rename(columns=keep)
    df = ensure_canonical(df)
    # If year missing but incident_datetime available, fill it
    if df["year"].isna().all() and "incident_datetime" in df:
        y = pd.to_datetime(df["incident_datetime"], errors="coerce").dt.year
        df.loc[~y.isna(), "year"] = y.astype("Int64")
    return df

# --------------------------------------------------------------------------------------
# LOS ANGELES cleaners (crimes + arrests)
# --------------------------------------------------------------------------------------
def detect_la_dataset(df: pd.DataFrame) -> str:
    """Return 'crimes' or 'arrests' based on header hints."""
    cols = set(df.columns)
    if {"DR_NO", "DATE OCC", "TIME OCC"}.intersection(cols) and ("Crm Cd Desc" in cols or "CrmCdDesc" in cols):
        return "crimes"
    if {"Report ID", "Arrest Date"}.issubset(cols) or "Charge Description" in cols:
        return "arrests"
    # fallback: filename/unknown
    return "unknown"

def clean_la_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    LAPD crime-data-from-2010-to-present.csv → canonical schema
    Common columns: DR_NO, Date Rptd, DATE OCC, TIME OCC, AREA, AREA NAME,
                    Crm Cd Desc, Premis Desc, Status Desc, LOCATION, LAT, LON
    """
    out = pd.DataFrame(index=df.index)

    # IDs / case
    if "DR_NO" in df.columns:
        out["incident_id"] = df["DR_NO"]
        out["case_number"] = df["DR_NO"]

    # Datetime (DATE OCC + TIME OCC → HH:MM)
    date_col = None
    for c in ["DATE OCC", "Date Occurred", "DATE_OCC", "DateOccured", "DATE"]:
        if c in df.columns:
            date_col = c; break
    time_col = None
    for c in ["TIME OCC", "Time Occurred", "TIME_OCC", "TIME"]:
        if c in df.columns:
            time_col = c; break
    if date_col and time_col:
        out["incident_datetime"] = build_datetime_from_date_hhmm(df[date_col], df[time_col])
    elif date_col:
        out["incident_datetime"] = df[date_col]

    # Updated on (Date Rptd)
    for c in ["Date Rptd", "DATE RPTD", "Report Date"]:
        if c in df.columns:
            out["updated_on"] = df[c]; break

    # Address / location text
    for c in ["LOCATION", "Address", "Location"]:
        if c in df.columns:
            out["block_address"] = df[c]; break

    # Category / subtype
    for c in ["Crm Cd Desc", "CrmCdDesc", "Category"]:
        if c in df.columns:
            out["crime_category"] = df[c]; break
    # Subtype: additional detail if present
    for c in ["MO Codes", "Weapon Desc", "Desc", "Status"]:
        if c in df.columns:
            out["crime_subtype"] = df[c]; break

    # Location type
    for c in ["Premis Desc", "PremisDesc", "Premise Description"]:
        if c in df.columns:
            out["location_type"] = df[c]; break

    # Arrest flag from Status Desc
    for c in ["Status Desc", "STATUS", "STATUS DESC"]:
        if c in df.columns:
            s = df[c].astype(str).str.upper()
            out["arrest_made"] = np.where(s.str.contains("ARREST|BOOKED"), True, np.nan)
            break

    # District/Area
    if "AREA" in df.columns:
        out["district"] = df["AREA"]
    elif "AREA NAME" in df.columns:
        out["district"] = df["AREA NAME"]

    # Coords
    if "LAT" in df.columns: out["latitude"] = df["LAT"]
    if "LON" in df.columns: out["longitude"] = df["LON"]

    # Year (from DATE OCC if not present)
    # We set later from incident_datetime in finalize_types if missing.

    # Fill remaining canonical columns
    out = ensure_canonical(out)

    # Set year if possible
    y = pd.to_datetime(out["incident_datetime"], errors="coerce").dt.year
    out.loc[~y.isna(), "year"] = y.astype("Int64")

    # Dedup keys typical for LAPD crimes
    out = deduplicate(out, key_priority=[["incident_id"], ["case_number", "incident_datetime"]])
    return out

def clean_la_arrests(df: pd.DataFrame) -> pd.DataFrame:
    """
    LAPD arrest-data-from-2010-to-present.csv → canonical schema
    Common columns: Report ID, Arrest Date, Time, Area ID, Area Name, Charge Description,
                    Charge Group Description, Address/Location, LAT, LON, Arrest Type/Code
    """
    out = pd.DataFrame(index=df.index)

    # IDs / case
    if "Report ID" in df.columns:
        out["incident_id"] = df["Report ID"]
        out["case_number"] = df["Report ID"]

    # Datetime (Arrest Date + Time)
    dcol = "Arrest Date" if "Arrest Date" in df.columns else None
    tcol = None
    for c in ["Time", "Arrest Time", "TIME"]:
        if c in df.columns:
            tcol = c; break
    if dcol and tcol:
        # 'Time' can be HH:MM or HHMM or H:MM; normalize
        t = df[tcol].astype(str).str.replace(r"\D", "", regex=True)
        t = t.str.zfill(4)
        hhmm = t.str.slice(0, 2) + ":" + t.str.slice(2, 4)
        out["incident_datetime"] = (df[dcol].astype(str) + " " + hhmm).str.strip()
    elif dcol:
        out["incident_datetime"] = df[dcol]

    # Updated on: default to Arrest Date if available
    if dcol: out["updated_on"] = df[dcol]

    # Address
    for c in ["Address", "Location", "LOCATION"]:
        if c in df.columns:
            out["block_address"] = df[c]; break

    # Category/subtype from charges
    if "Charge Group Description" in df.columns:
        out["crime_category"] = df["Charge Group Description"]
    elif "Charge" in df.columns:
        out["crime_category"] = df["Charge"]

    for c in ["Charge Description", "Charge", "Arrest Type"]:
        if c in df.columns:
            out["crime_subtype"] = df[c]; break

    # Arrest flag: arrests dataset → True
    out["arrest_made"] = True

    # District / Area
    if "Area ID" in df.columns:
        out["district"] = df["Area ID"]
    elif "Area Name" in df.columns:
        out["district"] = df["Area Name"]

    # Coords
    if "LAT" in df.columns: out["latitude"] = df["LAT"]
    if "LON" in df.columns: out["longitude"] = df["LON"]

    out = ensure_canonical(out)
    # Year from arrest date/time
    y = pd.to_datetime(out["incident_datetime"], errors="coerce").dt.year
    out.loc[~y.isna(), "year"] = y.astype("Int64")

    # Dedup keys typical for LAPD arrests
    out = deduplicate(out, key_priority=[["incident_id"], ["case_number", "incident_datetime"]])
    return out

# --------------------------------------------------------------------------------------
# SAN FRANCISCO radio codebook parser (Excel)
# --------------------------------------------------------------------------------------
def parse_sf_radio_codes(input_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse 'Radio Codes 2016.xlsx' into structured tables.
    We scan all cells on Sheet1 row-wise, recognize sections and pair code->description.
    Outputs: radio_codes, dispositions, modifiers, priorities, sheriff_codes
    """
    logger.info(f"Reading Excel: {input_path}")
    # Read all non-empty cells as strings
    s1 = pd.read_excel(input_path, sheet_name=0, header=None, dtype=str, engine="openpyxl")
    values: List[str] = []
    for _, row in s1.iterrows():
        for val in row.tolist():
            if isinstance(val, str):
                v = val.strip()
                if v and v.lower() != "nan":
                    values.append(v.replace("\u2013", "-").replace("\u2014", "-"))

    section_headers = {
        "RADIO CODES": "radio_codes",
        "Dispositions:": "dispositions",
        "Suffixes and Radio Code Modifiers:": "modifiers",
        "Priorities": "priorities",
        "RADIO CODES- S.F. Sheriff Dept.": "sheriff_codes",
        "RADIO CODES-S.F. Sheriff Dept.": "sheriff_codes",
    }

    def is_code_token(t: str) -> bool:
        return bool(re.fullmatch(r"(10-\d{1,3}[A-Z]?|\d{2,4}[A-Z]?|CODE\s*\d+)", t))

    radio_rows, dispo_rows, mod_rows, prio_rows, sheriff_rows = [], [], [], [], []
    section = "radio_codes"
    i = 0
    while i < len(values):
        t = values[i]
        if t in section_headers:
            section = section_headers[t]; i += 1; continue

        if section == "radio_codes":
            if is_code_token(t) and i + 1 < len(values):
                desc = values[i + 1]
                if not is_code_token(desc) and desc not in section_headers:
                    radio_rows.append({"code": t, "description": desc})
                    i += 2; continue

        if section == "dispositions":
            if re.fullmatch(r"[A-Z]{2,4}", t) and i + 1 < len(values):
                dispo_rows.append({"code": t, "description": values[i + 1]})
                i += 2; continue

        if section == "modifiers":
            m = re.match(r"^([A-Z]{1,3}|RED)\s*-\s*(.+)$", t)
            if m:
                mod_rows.append({"modifier": m.group(1), "meaning": m.group(2)})
                i += 1; continue

        if section == "priorities":
            if re.fullmatch(r"[A-Z]", t) and i + 1 < len(values):
                prio_rows.append({"priority": t, "meaning": values[i + 1]})
                i += 2; continue

        if section == "sheriff_codes":
            if re.fullmatch(r"10-\d{1,3}[A-Z]?", t) and i + 1 < len(values):
                sheriff_rows.append({"code": t, "description": values[i + 1]})
                i += 2; continue

        i += 1

    frames = {
        "radio_codes": pd.DataFrame(radio_rows).drop_duplicates(),
        "dispositions": pd.DataFrame(dispo_rows).drop_duplicates(),
        "modifiers": pd.DataFrame(mod_rows).drop_duplicates(),
        "priorities": pd.DataFrame(prio_rows).drop_duplicates(),
        "sheriff_codes": pd.DataFrame(sheriff_rows).drop_duplicates(),
    }
    return frames

def write_sf_radio_outputs(input_path: Path, outdir: Path, frames: Dict[str, pd.DataFrame]) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.lower().replace(' ', '_')
    xlsx_out = outdir / f"sf_{stem}_parsed.xlsx"
    json_out = outdir / f"sf_{stem}_codebook.json"

    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        for name, df in frames.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)

    codebook = {name: df.to_dict(orient="records") for name, df in frames.items()}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    logger.info(f"Parsed Excel written: {xlsx_out}")
    logger.info(f"JSON codebook written: {json_out}")
    return xlsx_out, json_out

# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------
def write_canonical_outputs(city_tag: str, input_path: Path, outdir: Path, df: pd.DataFrame) -> Tuple[Path, Optional[Path], Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.lower().replace(' ', '_')
    csv_out = outdir / f"{city_tag}_{stem}_clean.csv"
    parquet_out = outdir / f"{city_tag}_{stem}_clean.parquet"
    profile_out = outdir / f"{city_tag}_{stem}_profile.json"

    df.to_csv(csv_out, index=False)
    parquet_path = None
    try:
        df.to_parquet(parquet_out, index=False)
        parquet_path = parquet_out
    except Exception as e:
        logger.warning(f"Parquet write failed (install pyarrow or fastparquet to enable): {e}")

    with open(profile_out, "w", encoding="utf-8") as f:
        json.dump(profile_df(df), f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Cleaned CSV: {csv_out}")
    if parquet_path:
        logger.info(f"Cleaned Parquet: {parquet_path}")
    logger.info(f"Profile JSON: {profile_out}")

    return csv_out, parquet_path, profile_out

# --------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------
def interactive_pick_file(root: Path, patterns: List[str]) -> Path:
    candidates: List[Path] = []
    for rel in ["data/sample", "data/raw", "data", "."]:
        base = root / rel
        if not base.exists():
            continue
        for pat in patterns:
            candidates.extend(base.rglob(pat))
    if not candidates:
        raise SystemExit("No matching files found. Please provide --input.")
    print("\nDetected candidate files:")
    for i, p in enumerate(candidates[:20], 1):
        print(f"  [{i}] {p}")
    sel = input(f"Select number [1-{min(len(candidates),20)}] or paste full path: ").strip()
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= min(len(candidates), 20):
            return candidates[idx - 1]
    return Path(sel)

def main():
    logger.info(f"data_cleaning.py version: {__version__}")

    p = argparse.ArgumentParser(description="Multi-city data cleaner to canonical schema + SF radio parser")
    p.add_argument('--city', required=False, choices=['chicago','la','la_crimes','la_arrests','sf_radio'],
                   help='Dataset to clean/parse.')
    p.add_argument('--input', required=False, help='Path to raw CSV/Parquet/Excel file')
    p.add_argument('--outdir', default='./data/clean', help='Output directory')
    p.add_argument('--profile-only', action='store_true', help='Only print profile (no files written)')
    p.add_argument('--fast', action='store_true', help='Load only first 200k rows (trial run for large CSVs)')
    args = p.parse_args()

    city = args.city
    input_path = Path(args.input) if args.input else None
    outdir = Path(args.outdir)
    fast_n = 200_000 if args.fast else None

    # Interactive if not provided
    if not city:
        print("Choose dataset:")
        print("  1) chicago")
        print("  2) la (auto-detect crimes vs arrests)")
        print("  3) la_crimes")
        print("  4) la_arrests")
        print("  5) sf_radio (Excel)")
        city_sel = input("Enter 1-5: ").strip()
        city_map = {"1": "chicago", "2": "la", "3": "la_crimes", "4": "la_arrests", "5": "sf_radio"}
        city = city_map.get(city_sel, "chicago")

    if not input_path:
        root = Path.cwd()
        if city == "chicago":
            input_path = interactive_pick_file(root, ["Chicago*.csv"])
        elif city in ("la", "la_crimes", "la_arrests"):
            input_path = interactive_pick_file(root, ["*crime-data-from-2010-to-present*.csv",
                                                      "*arrest-data-from-2010-to-present*.csv"])
        elif city == "sf_radio":
            input_path = interactive_pick_file(root, ["Radio Codes 2016.xlsx", "*Radio*Codes*.xlsx"])
        else:
            raise SystemExit("Unsupported city selection.")

    if city == "sf_radio":
        frames = parse_sf_radio_codes(input_path)
        write_sf_radio_outputs(input_path, outdir, frames)
        return

    # CSV paths (Chicago / LA) → read
    df_raw = robust_read_csv(input_path, nrows=fast_n)
    df_raw = drop_index_artifacts(df_raw)

    if city == "chicago":
        df = clean_chicago(df_raw)
        df = finalize_types(df)
        df = deduplicate(df)
        if args.profile_only:
            print(json.dumps(profile_df(df), indent=2, default=str)); return
        write_canonical_outputs("chicago", input_path, outdir, df)
        return

    if city in ("la", "la_crimes", "la_arrests"):
        # Detect arrests vs crimes for --city la
        which = None
        if city == "la":
            which = detect_la_dataset(df_raw)
            logger.info(f"Auto-detected LA dataset: {which}")
            if which == "unknown":
                logger.warning("Could not auto-detect LA dataset by headers; defaulting to 'crimes'.")
                which = "crimes"
        else:
            which = "crimes" if city == "la_crimes" else "arrests"

        if which == "crimes":
            df = clean_la_crimes(df_raw)
            df = finalize_types(df)
            df = deduplicate(df, key_priority=[["incident_id"], ["case_number", "incident_datetime"]])
            if args.profile_only:
                print(json.dumps(profile_df(df), indent=2, default=str)); return
            write_canonical_outputs("la_crimes", input_path, outdir, df)
            return

        if which == "arrests":
            df = clean_la_arrests(df_raw)
            df = finalize_types(df)
            df = deduplicate(df, key_priority=[["incident_id"], ["case_number", "incident_datetime"]])
            if args.profile_only:
                print(json.dumps(profile_df(df), indent=2, default=str)); return
            write_canonical_outputs("la_arrests", input_path, outdir, df)
            return

    raise SystemExit("Unexpected control flow; please check --city and input file.")

if __name__ == '__main__':
    main()
