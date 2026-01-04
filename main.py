#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
--------------------------
Runs the complete data engineering pipeline for crime data integration.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_ingestion.data_ingestion import run_ingestion_pipeline
from modeling.modeling import run_modeling
from feature_engineering.feature_engineering import run_feature_engineering
from evaluation.evaluation import run_evaluation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete data engineering pipeline."""
    logger.info("Starting full data engineering pipeline...")

    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Running data ingestion...")
        processed_files, validation_results = run_ingestion_pipeline()
        logger.info("Data ingestion completed.")

        # Step 2: Feature Engineering
        logger.info("Step 2: Running feature engineering...")
        run_feature_engineering()
        logger.info("Feature engineering completed.")

        # Step 3: Modeling
        logger.info("Step 3: Running modeling...")
        run_modeling()
        logger.info("Modeling completed.")

        # Step 4: Evaluation
        logger.info("Step 4: Running evaluation...")
        run_evaluation()
        logger.info("Evaluation completed.")

        logger.info("Full pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    run_full_pipeline()