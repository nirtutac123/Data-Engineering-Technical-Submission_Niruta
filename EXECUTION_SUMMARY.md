# Airflow Schema Matching Pipeline - Execution Summary

## ✓ COMPLETED TASKS

### 1. Airflow DAG Created ✓
**File**: `dags/schema_matching_dag.py`

The DAG includes:
- ✅ 20 tasks organized in 6 phases
- ✅ Meaningful task names (e.g., `create_dataset_registry`, `compute_lexical_similarity`)
- ✅ Logical dependencies with proper task sequencing
- ✅ Detailed comments explaining each task's purpose
- ✅ Complete pipeline from data ingestion to figure generation

#### Pipeline Phases:
1. **Data Ingestion and Schema Profiling** (2 tasks)
2. **Schema Normalization** (3 tasks)
3. **Schema Matching** (5 tasks)
4. **Ground Truth Construction** (2 tasks)
5. **Evaluation** (2 tasks)
6. **Figures and Analysis** (6 tasks)

### 2. Docker Configuration Created ✓
**Files Created**:
- ✅ `docker-compose.yaml` - Complete Airflow stack configuration
- ✅ `.env` - Environment variables
- ✅ `.dockerignore` - Docker build optimization
- ✅ `airflow.ps1` - PowerShell management script
- ✅ `airflow.sh` - Bash management script
- ✅ `AIRFLOW_SETUP.md` - Complete documentation

### 3. Airflow Started ✓
**Status**: Running
- ✅ PostgreSQL database: Healthy
- ✅ Airflow Scheduler: Starting (installing dependencies)
- ✅ Airflow Webserver: Starting (installing dependencies)

**Access Information**:
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

## DOCKER SERVICES

### Running Containers:
```
NAME                             STATUS                         PORTS
de_project-postgres-1            Up (healthy)                   5432/tcp
de_project-airflow-scheduler-1   Up (health: starting)          8080/tcp
de_project-airflow-webserver-1   Up (health: starting)          0.0.0.0:8080->8080/tcp
```

### Installation Progress:
Currently installing Python packages:
- pandas ✓
- numpy ✓
- pyyaml ✓
- sentence-transformers (in progress)
- scikit-learn (in progress)
- matplotlib (in progress)
- seaborn (pending)

## NEXT STEPS

### 1. Wait for Services to Be Healthy (~2-3 minutes)
The services are currently installing dependencies. You can monitor progress:

**PowerShell**:
```powershell
cd "c:\Project files\Data Engineering\2nd attempt project files\DE_Project"
docker-compose logs -f
```

**Check Status**:
```powershell
docker-compose ps
```

### 2. Access the Web UI
Once healthy, open your browser to: **http://localhost:8080**

Login with:
- Username: `airflow`
- Password: `airflow`

### 3. Trigger the DAG
You can trigger the pipeline in three ways:

#### Option A: Web UI (Recommended)
1. Find `schema_matching_pipeline` in the DAG list
2. Toggle the switch to "ON" (unpause)
3. Click the "Play" button → "Trigger DAG"
4. Click the DAG name to monitor progress

#### Option B: PowerShell Script
```powershell
.\airflow.ps1 trigger
```

#### Option C: Docker Exec
```powershell
docker-compose exec airflow-scheduler airflow dags trigger schema_matching_pipeline
```

### 4. Monitor Execution
- **Graph View**: Shows task dependencies and execution status
- **Grid View**: Shows historical runs
- **Task Logs**: Click on any task to view detailed logs

### 5. View Results
After successful execution, check:
- `tables/` - All CSV output files
- `figures/` - All PDF visualizations (15 figures)

## USEFUL COMMANDS

### Service Management
```powershell
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Complete cleanup
docker-compose down -v
```

### Using Helper Scripts
```powershell
# PowerShell commands
.\airflow.ps1 status   # Check status
.\airflow.ps1 logs     # View logs
.\airflow.ps1 trigger  # Trigger DAG
.\airflow.ps1 stop     # Stop services
.\airflow.ps1 clean    # Clean everything
```

## DAG STRUCTURE OVERVIEW

```
Phase 1: Data Ingestion
  └─ create_dataset_registry
     └─ profile_schemas

Phase 2: Schema Normalization
  └─ expand_abbreviations
     └─ normalize_schemas
        └─ create_final_schema_table

Phase 3: Schema Matching
  └─ generate_schema_pairs
     └─ create_pair_representations
        ├─ compute_lexical_similarity
        └─ compute_embedding_similarity
           └─ merge_similarity_scores

Phase 4: Ground Truth
  └─ create_ground_truth_template
     └─ create_true_match_column_types

Phase 5: Evaluation
  └─ evaluate_similarity_methods
     └─ create_threshold_sweep

Phase 6: Figures (all run in parallel)
  ├─ generate_rq1_figures (3 PDFs)
  ├─ generate_rq2_figures (3 PDFs)
  ├─ generate_rq3_figures (3 PDFs)
  ├─ generate_rq4_figures (3 PDFs)
  └─ generate_rq5_figures (3 PDFs)
     └─ pipeline_complete
```

## EXPECTED OUTPUTS

### Tables (CSV files in `tables/`):
- `dataset_registry.csv`
- `schema_pairs.csv`
- `schema_pairs_repr.csv`
- `similarity_scores_lexical.csv`
- `similarity_scores_embedding.csv`
- `similarity_scores.csv`
- `evaluation_metrics.csv`
- `threshold_sweep_metrics.csv`
- `true_match_column_types.csv`
- `abbreviation_mapping.csv`
- `ground_truth_template.csv`

### Figures (PDF files in `figures/`):
- **RQ1**: rq1_fig1_f1.pdf, rq1_fig2.pdf, rq1_fig3.pdf
- **RQ2**: rq2_fig1.pdf, rq2_fig2.pdf, rq2_fig3.pdf
- **RQ3**: rq3_fig1.pdf, rq3_fig2.pdf, rq3_fig3.pdf
- **RQ4**: rq4_fig1.pdf, rq4_fig2.pdf, rq4_fig3.pdf
- **RQ5**: rq5_fig1.pdf, rq5_fig2.pdf, rq5_fig3.pdf

## TROUBLESHOOTING

### If services don't start:
```powershell
# View detailed logs
docker-compose logs airflow-scheduler
docker-compose logs airflow-webserver

# Restart services
docker-compose restart
```

### If DAG doesn't appear:
```powershell
# Check for DAG errors
docker-compose exec airflow-scheduler airflow dags list-import-errors

# Verify DAG syntax
docker-compose exec airflow-scheduler airflow dags list
```

### If tasks fail:
1. Click on the failed task in the Web UI
2. Click "Log" to view error details
3. Check if required data files exist in `data/raw_external/`
4. Verify Python dependencies are installed

## NOTES

- First run may take 10-15 minutes due to model downloads
- Ensure raw data files are in `data/raw_external/`
- All paths in the DAG are configurable via `config.yaml`
- The pipeline is fully reproducible and self-contained

## CURRENT STATUS: ✅ READY

Airflow is currently starting up and installing dependencies. 
Wait 2-3 minutes, then access the Web UI at: **http://localhost:8080**
