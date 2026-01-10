# Dataset Selection and Exclusion Rationale

## Goal
This project is a **schema-level** data engineering pipeline for automated schema matching.
The objective is **not** record-level crime analytics, but learning robust schema alignment across heterogeneous datasets (RQ1–RQ5).

## Key Constraint: Large Data Handling
Part 2 discourages storing or committing large raw datasets into the repository.
Our datasets contain very large numbers of rows across multiple files.
Therefore, the pipeline is designed to:
- read large CSV files **locally**,
- extract **schema-level** information only (column names, representative sample values, metadata),
- store only small generated outputs (schema samples, tables, figures) inside the repository.

## Research Question Coverage (RQ1–RQ5)
The five research questions focus on schema matching behavior, not row-level predictions.
Each selected dataset supports the research questions as follows:
- **RQ1 (Embedding vs. Lexical Matching):** requires heterogeneous column naming across datasets.
- **RQ2 (Header vs. Header + Value):** requires ambiguous column names where sample values provide semantic context.
- **RQ3 (Cross-dataset Generalization):** requires multiple cities within the same domain but different schemas.
- **RQ4 (Bias from Abbreviations and Coded Fields):** requires datasets with abbreviations and numeric/categorical codes.
- **RQ5 (Explainability):** benefits from available metadata describing columns.

## Selected Datasets (Final)
We selected **one representative dataset per city**, which is sufficient to address all five research questions without redundancy:

### Chicago
- Domain: Crime incidents
- Selected as a high-variance schema with abbreviations/coded fields.
- Metadata JSON: not available (kept realistic).

### San Francisco
- Domain: Police department incidents
- Selected for schema diversity and availability of metadata JSON.

### Los Angeles
- Domain: Crime data (2010 to present)
- Selected for schema diversity and availability of metadata JSON.

## Metadata Inclusion (RQ2, RQ5)
Where metadata JSON files are natively available (SF and LA), we include them to:
- resolve ambiguous headers more reliably (RQ2),
- provide explainability evidence (RQ5).
We do not artificially recreate metadata for sources where it does not exist (e.g., Chicago).

## Excluded Datasets (and Why)
Datasets were excluded not due to low quality, but because their cost/complexity outweighed their contribution to RQ1–RQ5:
- Additional Chicago yearly CSV files: same schema/city, minimal added insight.
- SF “Calls for Service”: different operational process; increases integration complexity.
- LA Arrest dataset: introduces a second major schema branch; not required for matching demonstration.
- PDF/XLSX codebooks and GIS shapefiles: useful as references, but not pipeline inputs for schema matching.

## One-Sentence Summary
We selected one representative dataset per city and included metadata where available to maximize schema diversity, fully support RQ1–RQ5, and remain compliant with Part 2 constraints on large raw data while maintaining a reproducible schema-level pipeline.