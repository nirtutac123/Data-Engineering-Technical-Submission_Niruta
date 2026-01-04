#!/usr/bin/env python3
"""
Airflow DAG for Crime Data Integration Pipeline
-----------------------------------------------
Orchestrates the complete data engineering pipeline for schema matching evaluation.

DAG Tasks:
- extract_data: Load raw crime datasets from sample directory
- clean_data: Clean and standardize data to canonical schema
- transform_features: Create ML features (temporal, spatial, categorical)
- train_model: Train ML models for arrest prediction
- generate_figures: Generate research evaluation figures and tables
- save_outputs: Save all outputs to appropriate directories
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def extract_data():
    """Extract raw crime data from sample datasets."""
    from data_ingestion.data_ingestion import load_sample_data
    from pathlib import Path
    print("Extracting raw crime data from sample directory...")
    sample_dir = Path('/opt/airflow/data/sample')
    data = load_sample_data(sample_dir)
    print(f"Extracted {len(data)} datasets")
    # Return only the count, not the full data (data is too large for XCom)
    return len(data)

def clean_data():
    """Clean and standardize crime data to canonical schema."""
    from data_ingestion.data_ingestion import run_ingestion_pipeline
    print("Cleaning and standardizing crime data...")
    processed_files, validation_results = run_ingestion_pipeline()
    print(f"Cleaned {len(processed_files)} datasets")
    # Return only the count, not the file paths
    return len(processed_files)

def transform_features():
    """Transform data and create ML features."""
    from feature_engineering.feature_engineering import run_feature_engineering
    print("Creating ML features (temporal, spatial, categorical)...")
    run_feature_engineering()
    print("Feature engineering completed")
    return "success"

def train_model():
    """Train ML models for arrest prediction."""
    from modeling.modeling import run_modeling
    print("Training ML models for arrest prediction...")
    run_modeling()
    print("Model training completed")
    return "success"

def generate_figures():
    """Generate research evaluation figures and tables."""
    from evaluation.evaluation import run_evaluation
    print("Generating research evaluation figures and tables...")
    run_evaluation()
    print("Figure and table generation completed")
    return "success"

def save_outputs():
    """Save all outputs to appropriate directories."""
    print("Saving outputs to figures/ and tables/ directories...")
    # Outputs are automatically saved by the previous tasks
    # This task ensures all outputs are properly organized
    figures_dir = Path('figures')
    tables_dir = Path('tables')
    if figures_dir.exists():
        pdf_count = len(list(figures_dir.glob('*.pdf')))
        print(f"Saved {pdf_count} figures")
    if tables_dir.exists():
        xlsx_count = len(list(tables_dir.glob('*.xlsx')))
        print(f"Saved {xlsx_count} tables")

    # Check for additional outputs
    outputs = ['feature_importance.png', 'model_results.txt', 'data/crime_data.db']
    for output in outputs:
        if Path(output).exists():
            print(f"Output saved: {output}")

# Default DAG arguments
default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'crime_data_integration_pipeline',
    default_args=default_args,
    description='Complete pipeline for crime data integration and schema matching evaluation',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['data_engineering', 'crime_data', 'schema_matching'],
)

# Define tasks with meaningful names
extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag,
)

transform_features_task = PythonOperator(
    task_id='transform_features',
    python_callable=transform_features,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

generate_figures_task = PythonOperator(
    task_id='generate_figures',
    python_callable=generate_figures,
    dag=dag,
)

save_outputs_task = PythonOperator(
    task_id='save_outputs',
    python_callable=save_outputs,
    dag=dag,
)

# Set logical task dependencies
extract_data_task >> clean_data_task >> transform_features_task >> train_model_task >> generate_figures_task >> save_outputs_task