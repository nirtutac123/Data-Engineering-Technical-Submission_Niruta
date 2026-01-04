#!/usr/bin/env python3
"""
Data Ingestion Module
---------------------
Loads sample data, validates it, and integrates with the cleaning pipeline.
"""

import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3
import logging

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / '..' / 'data_cleaning'))
try:
    from data_cleaning import (
        robust_read_csv, drop_index_artifacts, profile_df,
        clean_chicago, clean_la_crimes, clean_la_arrests,
        finalize_types, deduplicate, write_canonical_outputs
    )
except ImportError as e:
    logger.error(f"Failed to import from data_cleaning: {e}")
    # Fallback: define minimal functions if needed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected schemas for validation
EXPECTED_SCHEMAS = {
    'chicago': ['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
                'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
                'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year',
                'Updated On', 'Latitude', 'Longitude', 'Location'],
    'la_crimes': ['DR_NO', 'Date Rptd', 'DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME',
                  'Crm Cd Desc', 'Premis Desc', 'Status Desc', 'LOCATION', 'LAT', 'LON'],
    'la_arrests': ['Report ID', 'Arrest Date', 'Time', 'Area ID', 'Area Name',
                   'Charge Group Description', 'Charge Description', 'Address',
                   'LAT', 'LON', 'Arrest Type Code'],
    'sf_radio': ['Radio Code', 'Description', 'Disposition', 'Priority']
}

def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect dataset type based on column presence."""
    cols = set(df.columns)
    if {'ID', 'Case Number', 'Date'}.issubset(cols):
        return 'chicago'
    elif {'DR_NO', 'DATE OCC'}.issubset(cols):
        return 'la_crimes'
    elif {'Report ID', 'Arrest Date'}.issubset(cols):
        return 'la_arrests'
    elif {'Radio Code'}.issubset(cols):
        return 'sf_radio'
    else:
        return 'unknown'

def load_sample_data(sample_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all sample data files."""
    data = {}
    for file_path in sample_dir.rglob('*'):
        if file_path.suffix.lower() in ['.csv', '.xlsx']:
            try:
                if file_path.suffix.lower() == '.csv':
                    # Use pandas directly for speed
                    df = pd.read_csv(file_path, nrows=1000, encoding='utf-8', low_memory=False)
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')
                df = drop_index_artifacts(df)
                dataset_type = detect_dataset_type(df)
                if dataset_type != 'unknown':
                    data[file_path.stem] = {
                        'data': df,
                        'type': dataset_type,
                        'path': file_path
                    }
                    logger.info(f"Loaded {file_path.name} as {dataset_type}")
                else:
                    logger.warning(f"Could not detect type for {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    return data

def validate_data(data: Dict) -> Dict[str, Dict]:
    """Validate loaded data against expected schemas."""
    validation_results = {}
    for name, info in data.items():
        df = info['data']
        expected_cols = EXPECTED_SCHEMAS.get(info['type'], [])
        actual_cols = list(df.columns)

        # Check column presence
        missing_cols = set(expected_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(expected_cols)

        # Basic data quality
        null_pct = df.isnull().mean().mean() * 100
        row_count = len(df)

        validation_results[name] = {
            'type': info['type'],
            'row_count': row_count,
            'col_count': len(actual_cols),
            'missing_cols': list(missing_cols),
            'extra_cols': list(extra_cols),
            'null_percentage': round(null_pct, 2),
            'is_valid': len(missing_cols) == 0 and null_pct < 50
        }

        if validation_results[name]['is_valid']:
            logger.info(f"{name}: Validation passed")
        else:
            logger.warning(f"{name}: Validation issues - Missing: {missing_cols}, Null%: {null_pct:.1f}")

    return validation_results

def process_data(data: Dict, validation_results: Dict, output_dir: Path) -> Dict[str, Path]:
    """Process validated data through cleaning pipeline."""
    processed_files = {}

    for name, info in data.items():
        if not validation_results[name]['is_valid']:
            logger.warning(f"Skipping {name} due to validation failure")
            continue

        df = info['data']
        dataset_type = info['type']
        input_path = info['path']

        try:
            # Apply cleaning based on type
            if dataset_type == 'chicago':
                df_clean = clean_chicago(df)
            elif dataset_type == 'la_crimes':
                df_clean = clean_la_crimes(df)
            elif dataset_type == 'la_arrests':
                df_clean = clean_la_arrests(df)
            else:
                logger.warning(f"No cleaning function for {dataset_type}")
                continue

            df_clean = finalize_types(df_clean)
            df_clean = deduplicate(df_clean)

            # Save cleaned data
            csv_path, parquet_path, profile_path = write_canonical_outputs(
                dataset_type, input_path, output_dir, df_clean
            )

            processed_files[name] = {
                'csv': csv_path,
                'parquet': parquet_path,
                'profile': profile_path
            }

            logger.info(f"Processed {name} successfully")

        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    return processed_files

def create_database_schema(db_path: Path):
    """Create SQLite database schema for crime data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create main incidents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT,
            case_number TEXT,
            incident_datetime TEXT,
            block_address TEXT,
            iucr_code TEXT,
            crime_category TEXT,
            crime_subtype TEXT,
            location_type TEXT,
            arrest_made INTEGER,
            domestic_flag INTEGER,
            beat TEXT,
            district TEXT,
            ward TEXT,
            community_area TEXT,
            fbi_code TEXT,
            x_coord REAL,
            y_coord REAL,
            year INTEGER,
            updated_on TEXT,
            latitude REAL,
            longitude REAL,
            location_point TEXT,
            dataset_source TEXT,
            PRIMARY KEY (incident_id, dataset_source)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"Database schema created at {db_path}")

def load_to_database(processed_files: Dict, db_path: Path):
    """Load processed data into SQLite database."""
    create_database_schema(db_path)
    conn = sqlite3.connect(db_path)

    for name, paths in processed_files.items():
        try:
            df = pd.read_csv(paths['csv'])
            df['dataset_source'] = name

            # Handle potential duplicates
            df.to_sql('incidents', conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(df)} rows from {name} to database")

        except Exception as e:
            logger.error(f"Failed to load {name} to database: {e}")

    conn.close()

def run_ingestion_pipeline(sample_dir: Path = None, output_dir: Path = None, db_path: Path = None):
    """Run the complete ingestion pipeline."""
    if sample_dir is None:
        sample_dir = Path('data/sample')
    if output_dir is None:
        output_dir = Path('data/clean')
    if db_path is None:
        db_path = Path('data/crime_data.db')

    logger.info("Starting data ingestion pipeline...")

    # Load data
    data = load_sample_data(sample_dir)
    if not data:
        logger.error("No data loaded. Check sample directory.")
        return

    # Validate
    validation_results = validate_data(data)

    # Process
    processed_files = process_data(data, validation_results, output_dir)

    # Load to database
    if processed_files:
        load_to_database(processed_files, db_path)

    logger.info("Ingestion pipeline completed.")
    return processed_files, validation_results

if __name__ == '__main__':
    # Run pipeline
    run_ingestion_pipeline()