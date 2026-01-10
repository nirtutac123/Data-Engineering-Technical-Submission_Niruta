"""
Schema Matching and Evaluation DAG

This DAG orchestrates the complete schema matching pipeline:
1. Data Ingestion and Schema Profiling
2. Schema Normalization
3. Schema Matching
4. Ground Truth Construction
5. Evaluation
6. Figures and Analysis

Author: Data Engineering Part 2 Project
Date: 2026-01-10
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2026, 1, 1),
}

# Instantiate the DAG
dag = DAG(
    'schema_matching_pipeline',
    default_args=default_args,
    description='Complete schema matching and evaluation pipeline',
    schedule_interval=None,  # Trigger manually or via external events
    catchup=False,
    tags=['schema-matching', 'data-engineering', 'evaluation'],
)

# ========================================================================
# PHASE 1: DATA INGESTION AND SCHEMA PROFILING
# ========================================================================

def create_dataset_registry():
    """
    Task 1.1: Create Dataset Registry
    Creates a registry of all datasets with metadata including city, 
    source file paths, and metadata JSON files.
    """
    from src.data_ingestion.dataset_registry import main as registry_main
    registry_main()
    print("✓ Dataset registry created successfully")


def profile_schemas():
    """
    Task 1.2: Schema Profiling
    Extracts schema information from raw datasets including:
    - Column names
    - Data types
    - Sample values
    - Unique value counts
    """
    import subprocess
    
    cities = ['chicago', 'sf', 'la']
    
    for city in cities:
        print(f"Profiling schemas for {city}...")
        result = subprocess.run(
            ['python', '-m', 'src.data_ingestion.schema_profiling', '--city', city, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error profiling {city}:")
            print(result.stderr)
            raise RuntimeError(f"Schema profiling failed for {city}")
        
        print(result.stdout)
    
    print("✓ Schema profiling completed for all datasets")


# Data Ingestion Tasks
task_create_dataset_registry = PythonOperator(
    task_id='create_dataset_registry',
    python_callable=create_dataset_registry,
    dag=dag,
)

task_profile_schemas = PythonOperator(
    task_id='profile_schemas',
    python_callable=profile_schemas,
    dag=dag,
)

# ========================================================================
# PHASE 2: SCHEMA NORMALIZATION
# ========================================================================

def expand_abbreviations():
    """
    Task 2.1: Abbreviation Expansion
    Expands common abbreviations in column names to full forms:
    - 'desc' → 'description'
    - 'num' → 'number'
    - 'addr' → 'address'
    This improves matching accuracy across datasets with different naming conventions.
    """
    import subprocess
    
    cities = ['chicago', 'sf', 'la']
    
    for city in cities:
        print(f"Expanding abbreviations for {city}...")
        result = subprocess.run(
            ['python', '-m', 'src.data_cleaning.abbreviation_expansion', '--city', city, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error expanding abbreviations for {city}:")
            print(result.stderr)
            raise RuntimeError(f"Abbreviation expansion failed for {city}")
        
        print(result.stdout)
    
    print("✓ Abbreviations expanded in all schemas")


def normalize_schemas():
    """
    Task 2.2: Schema Normalization
    Normalizes column names by:
    - Converting to lowercase
    - Replacing special characters with underscores
    - Removing extra whitespace
    - Standardizing data types
    """
    import subprocess
    
    cities = ['chicago', 'sf', 'la']
    
    for city in cities:
        print(f"Normalizing schemas for {city}...")
        result = subprocess.run(
            ['python', '-m', 'src.data_cleaning.schema_normalization', '--city', city, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error normalizing schemas for {city}:")
            print(result.stderr)
            raise RuntimeError(f"Schema normalization failed for {city}")
        
        print(result.stdout)
    
    print("✓ Schemas normalized for all datasets")


def create_final_schema_table():
    """
    Task 2.3: Create Final Schema Table
    Consolidates all normalized schemas into a single table
    with standardized structure for downstream matching tasks.
    """
    import subprocess
    
    cities = ['chicago', 'sf', 'la']
    
    for city in cities:
        print(f"Creating final schema table for {city}...")
        result = subprocess.run(
            ['python', '-m', 'src.data_cleaning.final_schema_table', '--city', city, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error creating final schema table for {city}:")
            print(result.stderr)
            raise RuntimeError(f"Final schema table creation failed for {city}")
        
        print(result.stdout)
    
    print("✓ Final schema table created")


# Schema Normalization Tasks
task_expand_abbreviations = PythonOperator(
    task_id='expand_abbreviations',
    python_callable=expand_abbreviations,
    dag=dag,
)

task_normalize_schemas = PythonOperator(
    task_id='normalize_schemas',
    python_callable=normalize_schemas,
    dag=dag,
)

task_create_final_schema_table = PythonOperator(
    task_id='create_final_schema_table',
    python_callable=create_final_schema_table,
    dag=dag,
)

# ========================================================================
# PHASE 3: SCHEMA MATCHING
# ========================================================================

def generate_schema_pairs():
    """
    Task 3.1: Generate Schema Pairs
    Creates all possible column pairs across different datasets.
    Uses Cartesian product approach to generate candidates for matching.
    Excludes within-dataset pairs to focus on cross-dataset matching.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.schema_matching.schema_pairs', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error generating schema pairs:")
        print(result.stderr)
        raise RuntimeError("Schema pair generation failed")
    print(result.stdout)
    print("✓ Schema pairs generated")


def create_pair_representations():
    """
    Task 3.2: Create Pair Representations
    Prepares column pair data for similarity computation:
    - Concatenates column names and descriptions
    - Creates header-only representations
    - Creates header + context representations
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.schema_matching.schema_pair_representation', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error creating pair representations:")
        print(result.stderr)
        raise RuntimeError("Pair representation creation failed")
    print(result.stdout)
    print("✓ Schema pair representations created")


def compute_lexical_similarity():
    """
    Task 3.3: Compute Lexical Similarity
    Calculates Jaccard similarity between column pairs based on:
    - Character n-grams
    - Token-level comparison
    Provides baseline similarity scores for comparison.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.schema_matching.lexical_similarity', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error computing lexical similarity:")
        print(result.stderr)
        raise RuntimeError("Lexical similarity computation failed")
    print(result.stdout)
    print("✓ Lexical similarity scores computed")


def compute_embedding_similarity():
    """
    Task 3.4: Compute Embedding Similarity
    Calculates semantic similarity using pre-trained embeddings:
    - Uses sentence transformers for encoding
    - Computes cosine similarity
    - Generates both header-only and header+context scores
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.schema_matching.embedding_similarity', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error computing embedding similarity:")
        print(result.stderr)
        raise RuntimeError("Embedding similarity computation failed")
    print(result.stdout)
    print("✓ Embedding similarity scores computed")


def merge_similarity_scores():
    """
    Task 3.5: Merge Similarity Scores
    Consolidates all similarity scores into a unified table:
    - Lexical (Jaccard) scores
    - Embedding (header-only) scores
    - Embedding (header+context) scores
    Creates the main matching results table for evaluation.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.schema_matching.merge_similarity_scores', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error merging similarity scores:")
        print(result.stderr)
        raise RuntimeError("Similarity score merging failed")
    print(result.stdout)
    print("✓ All similarity scores merged")


# Schema Matching Tasks
task_generate_schema_pairs = PythonOperator(
    task_id='generate_schema_pairs',
    python_callable=generate_schema_pairs,
    dag=dag,
)

task_create_pair_representations = PythonOperator(
    task_id='create_pair_representations',
    python_callable=create_pair_representations,
    dag=dag,
)

task_compute_lexical_similarity = PythonOperator(
    task_id='compute_lexical_similarity',
    python_callable=compute_lexical_similarity,
    dag=dag,
)

task_compute_embedding_similarity = PythonOperator(
    task_id='compute_embedding_similarity',
    python_callable=compute_embedding_similarity,
    dag=dag,
)

task_merge_similarity_scores = PythonOperator(
    task_id='merge_similarity_scores',
    python_callable=merge_similarity_scores,
    dag=dag,
)

# ========================================================================
# PHASE 4: GROUND TRUTH CONSTRUCTION
# ========================================================================

def create_ground_truth_template():
    """
    Task 4.1: Create Ground Truth Template
    Generates template for manual labeling of true column matches.
    Includes column pairs with metadata for expert review.
    Ground truth is essential for evaluating matching accuracy.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.evaluation.create_ground_truth_template', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error creating ground truth template:")
        print(result.stderr)
        raise RuntimeError("Ground truth template creation failed")
    print(result.stdout)
    print("✓ Ground truth template created")


def create_true_match_column_types():
    """
    Task 4.2: Analyze True Match Column Types
    Analyzes the types of columns that are successfully matched:
    - Temporal columns (dates, times)
    - Location columns (addresses, coordinates)
    - Categorical columns (types, codes)
    - Numeric columns (IDs, counts)
    Supports RQ4 analysis.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.evaluation.create_true_match_column_types', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error analyzing true match column types:")
        print(result.stderr)
        raise RuntimeError("True match column types analysis failed")
    print(result.stdout)
    print("✓ True match column types analyzed")


# Ground Truth Tasks
task_create_ground_truth_template = PythonOperator(
    task_id='create_ground_truth_template',
    python_callable=create_ground_truth_template,
    dag=dag,
)

task_create_true_match_column_types = PythonOperator(
    task_id='create_true_match_column_types',
    python_callable=create_true_match_column_types,
    dag=dag,
)

# ========================================================================
# PHASE 5: EVALUATION
# ========================================================================

def evaluate_similarity_methods():
    """
    Task 5.1: Evaluate Similarity Methods
    Computes evaluation metrics for all matching methods:
    - Precision: Ratio of correct matches among predicted matches
    - Recall: Ratio of correct matches found among all true matches
    - F1-Score: Harmonic mean of precision and recall
    - True Positives, False Positives, False Negatives, True Negatives
    Supports RQ1 and RQ2 analysis.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.evaluation.evaluate_similarity', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error evaluating similarity methods:")
        print(result.stderr)
        raise RuntimeError("Similarity evaluation failed")
    print(result.stdout)
    print("✓ Similarity methods evaluated")


def create_threshold_sweep():
    """
    Task 5.2: Perform Threshold Sweep
    Evaluates matching performance across different similarity thresholds:
    - Tests thresholds from 0.0 to 1.0
    - Computes metrics at each threshold
    - Identifies optimal threshold for each method
    Supports RQ3 analysis on threshold selection.
    """
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'src.evaluation.create_threshold_sweep_table', '--config', 'config.yaml'],
        cwd='/opt/airflow',
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error creating threshold sweep:")
        print(result.stderr)
        raise RuntimeError("Threshold sweep creation failed")
    print(result.stdout)
    print("✓ Threshold sweep analysis completed")


# Evaluation Tasks
task_evaluate_similarity_methods = PythonOperator(
    task_id='evaluate_similarity_methods',
    python_callable=evaluate_similarity_methods,
    dag=dag,
)

task_create_threshold_sweep = PythonOperator(
    task_id='create_threshold_sweep',
    python_callable=create_threshold_sweep,
    dag=dag,
)

# ========================================================================
# PHASE 6: FIGURES AND ANALYSIS
# ========================================================================

def generate_rq1_figures():
    """
    Task 6.1: Generate RQ1 Figures
    Creates visualizations for Research Question 1:
    "Which schema matching method performs best overall?"
    - Figure 1: F1-score comparison across methods
    - Figure 2: Precision vs Recall tradeoff
    - Figure 3: Confusion matrix components
    """
    import subprocess
    scripts = [
        'src.evaluation.plot_rq1_fig1_f1',
        'src.evaluation.plot_rq1_fig2',
        'src.evaluation.plot_rq1_fig3'
    ]
    for script in scripts:
        result = subprocess.run(
            ['python', '-m', script, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Figure generation failed for {script}")
        print(result.stdout)
    print("✓ RQ1 figures generated")


def generate_rq2_figures():
    """
    Task 6.2: Generate RQ2 Figures
    Creates visualizations for Research Question 2:
    "Do embedding-based methods improve matching quality?"
    - Figure 1: Lexical vs Embedding comparison
    - Figure 2: Score distribution analysis
    - Figure 3: Top-N precision analysis
    """
    import subprocess
    scripts = [
        'src.evaluation.plot_rq2_fig1',
        'src.evaluation.plot_rq2_fig2',
        'src.evaluation.plot_rq2_fig3'
    ]
    for script in scripts:
        result = subprocess.run(
            ['python', '-m', script, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Figure generation failed for {script}")
        print(result.stdout)
    print("✓ RQ2 figures generated")


def generate_rq3_figures():
    """
    Task 6.3: Generate RQ3 Figures
    Creates visualizations for Research Question 3:
    "How does threshold selection affect matching quality?"
    - Figure 1: Threshold vs Precision
    - Figure 2: Threshold vs Recall
    - Figure 3: Threshold vs F1-score
    """
    import subprocess
    scripts = [
        'src.evaluation.plot_rq3_fig1',
        'src.evaluation.plot_rq3_fig2',
        'src.evaluation.plot_rq3_fig3'
    ]
    for script in scripts:
        result = subprocess.run(
            ['python', '-m', script, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Figure generation failed for {script}")
        print(result.stdout)
    print("✓ RQ3 figures generated")


def generate_rq4_figures():
    """
    Task 6.4: Generate RQ4 Figures
    Creates visualizations for Research Question 4:
    "Which column types are matched more reliably?"
    - Figure 1: Match accuracy by column type
    - Figure 2: Column type distribution
    - Figure 3: Error patterns by type
    """
    import subprocess
    scripts = [
        'src.evaluation.plot_rq4_fig1',
        'src.evaluation.plot_rq4_fig2',
        'src.evaluation.plot_rq4_fig3'
    ]
    for script in scripts:
        result = subprocess.run(
            ['python', '-m', script, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Figure generation failed for {script}")
        print(result.stdout)
    print("✓ RQ4 figures generated")


def generate_rq5_figures():
    """
    Task 6.5: Generate RQ5 Figures
    Creates visualizations for Research Question 5:
    "What kinds of matching errors occur and why?"
    - Figure 1: Error type distribution
    - Figure 2: False positive analysis
    - Figure 3: False negative analysis
    """
    import subprocess
    scripts = [
        'src.evaluation.plot_rq5_fig1',
        'src.evaluation.plot_rq5_fig2',
        'src.evaluation.plot_rq5_fig3'
    ]
    for script in scripts:
        result = subprocess.run(
            ['python', '-m', script, '--config', 'config.yaml'],
            cwd='/opt/airflow',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr)
            raise RuntimeError(f"Figure generation failed for {script}")
        print(result.stdout)
    print("✓ RQ5 figures generated")


# Figure Generation Tasks
task_generate_rq1_figures = PythonOperator(
    task_id='generate_rq1_figures',
    python_callable=generate_rq1_figures,
    dag=dag,
)

task_generate_rq2_figures = PythonOperator(
    task_id='generate_rq2_figures',
    python_callable=generate_rq2_figures,
    dag=dag,
)

task_generate_rq3_figures = PythonOperator(
    task_id='generate_rq3_figures',
    python_callable=generate_rq3_figures,
    dag=dag,
)

task_generate_rq4_figures = PythonOperator(
    task_id='generate_rq4_figures',
    python_callable=generate_rq4_figures,
    dag=dag,
)

task_generate_rq5_figures = PythonOperator(
    task_id='generate_rq5_figures',
    python_callable=generate_rq5_figures,
    dag=dag,
)

# ========================================================================
# PIPELINE COMPLETION
# ========================================================================

def pipeline_complete():
    """
    Final Task: Pipeline Completion
    Marks the successful completion of the entire pipeline.
    Generates summary report of all outputs created.
    """
    print("=" * 70)
    print("SCHEMA MATCHING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nOutputs Generated:")
    print("- Dataset Registry: tables/dataset_registry.csv")
    print("- Schema Profiles: data/sample/*_schema_*.csv")
    print("- Schema Pairs: tables/schema_pairs.csv")
    print("- Similarity Scores: tables/similarity_scores.csv")
    print("- Evaluation Metrics: tables/evaluation_metrics.csv")
    print("- Threshold Sweep: tables/threshold_sweep_metrics.csv")
    print("- Figures: figures/*.pdf (15 figures total)")
    print("\nAll research questions (RQ1-RQ5) can now be analyzed.")
    print("=" * 70)


task_pipeline_complete = PythonOperator(
    task_id='pipeline_complete',
    python_callable=pipeline_complete,
    dag=dag,
)

# ========================================================================
# DEFINE TASK DEPENDENCIES
# ========================================================================

# Phase 1: Data Ingestion and Schema Profiling
# Registry must be created before profiling schemas
task_create_dataset_registry >> task_profile_schemas

# Phase 2: Schema Normalization
# Profiling must complete before normalization starts
task_profile_schemas >> task_expand_abbreviations
task_expand_abbreviations >> task_normalize_schemas
task_normalize_schemas >> task_create_final_schema_table

# Phase 3: Schema Matching
# Final schema table is required for all matching tasks
task_create_final_schema_table >> task_generate_schema_pairs
task_generate_schema_pairs >> task_create_pair_representations

# Similarity computations can run in parallel after pair representations
task_create_pair_representations >> [task_compute_lexical_similarity, task_compute_embedding_similarity]

# Merge requires both similarity computations to complete
[task_compute_lexical_similarity, task_compute_embedding_similarity] >> task_merge_similarity_scores

# Phase 4: Ground Truth Construction
# Ground truth template depends on merged similarity scores
task_merge_similarity_scores >> task_create_ground_truth_template
task_create_ground_truth_template >> task_create_true_match_column_types

# Phase 5: Evaluation
# Evaluation requires both similarity scores and ground truth
task_create_true_match_column_types >> task_evaluate_similarity_methods
task_evaluate_similarity_methods >> task_create_threshold_sweep

# Phase 6: Figures and Analysis
# All figures depend on evaluation metrics and threshold sweep
task_create_threshold_sweep >> [
    task_generate_rq1_figures,
    task_generate_rq2_figures,
    task_generate_rq3_figures,
    task_generate_rq4_figures,
    task_generate_rq5_figures
]

# Pipeline completion depends on all figures being generated
[
    task_generate_rq1_figures,
    task_generate_rq2_figures,
    task_generate_rq3_figures,
    task_generate_rq4_figures,
    task_generate_rq5_figures
] >> task_pipeline_complete
