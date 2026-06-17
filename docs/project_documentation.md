# Project Documentation

## Overview

Rural Health Care Anomaly Detection System is a multi-part project for detecting unusual clinical patterns in patient records and presenting them in a clinician-friendly workflow.

The repository combines:

- A Python-based anomaly detection stack for preprocessing, training, scoring, evaluation, and export
- A web dashboard for guided patient entry, lab upload, patient-care insight visualizations, comparative model analysis, and decision support
- Supporting documentation for feature provenance, configuration examples, and normalized healthcare schema design
- Test datasets and generated lab reports for exercising the upload and analysis flows

The project is intentionally built as a full workflow instead of a single model script. The user experience starts with patient intake, continues through lab review, then moves into insight generation, model comparison, and final recommendations.

## What The System Is For

The system is designed to help with:

- Capturing structured patient information in a consistent format
- Ingesting lab values from manual entry or uploaded report files
- Detecting anomalies and clinically important deviations
- Comparing multiple detectors and model families
- Summarizing the current case in plain language
- Supporting decisions with actionable recommendations
- Exporting models for offline or edge use

It is not just a scoring engine. It is a complete decision-support pipeline that preserves the patient record, the model outputs, and the human-readable explanation path.

## High-Level Architecture

The system has five major layers:

1. Data layer
2. Preprocessing layer
3. Training and scoring layer
4. Evaluation and export layer
5. Web experience and decision-support layer

### Data Layer

The data layer includes:

- Structured CSV or Parquet inputs
- Synthetic demo data for local testing
- Generated lab report samples for upload testing
- Schema definitions and configuration examples

### Preprocessing Layer

The preprocessing stack handles:

- Schema-aware feature handling
- Missing value handling
- Encoding and feature expansion
- Time-series or aggregated feature generation where applicable
- Optional dimensionality reduction and feature map export

### Training And Scoring Layer

The training layer supports:

- Ensemble anomaly detection
- Model-specific tuning
- Threshold calibration
- Stacking and weighted fusion
- Local explanations and risk scores

### Evaluation And Export Layer

This layer supports:

- Precision, recall, F1, AUC-ROC, and AUPRC reporting
- Dashboard-style evaluation output
- Edge bundle export
- Offline scoring bundle usage
- Runtime and deployment cost reporting

### Web Experience And Decision Support Layer

The web app provides:

- Patient details entry
- Lab report upload
- Patient care insights
- Comparative analysis of detectors and costs
- Decision support and clinician feedback
- Model hub / final review views

## Repository Layout

The repository is organized around both the Python backend and the web dashboard.

### Top-Level Files

- `README.md` - project entry point and usage summary
- `pyproject.toml` - Python package and tool configuration
- `LICENSE` - project license
- `Dockerfile` - container build for deployment
- `dashboard_server.py` - local dashboard server for scored outputs
- `backend_server.py` - backend integration helper
- `streamlit_dashboard.py` - Streamlit dashboard entry point
- `train_pipeline.py` - training orchestration entry point
- `preprocessing_pipeline.py` - preprocessing pipeline helper
- `anomaly_cli.py` - command-line shim for the anomaly CLI
- `edge_inference.py` - offline edge inference entry point
- `example_training_inference.py` - example end-to-end pipeline

### Python Package

The main package is `rural_health_anomaly/`, which contains the reusable model, preprocessing, training, evaluation, and support code.

Typical responsibilities include:

- `cli.py` - user-facing CLI commands
- `training.py` - training orchestration and ensemble setup
- `evaluation.py` - metrics and report generation
- `pipeline.py` - integrated preprocessing and model flow
- `preprocessing.py` - feature transformations and schema handling
- `ensemble.py` - score fusion and detector orchestration
- `autoencoder.py`, `deep_svdd.py`, `ganomaly.py`, `anomaly_transformer.py`, `vital_signs` related modules - model implementations and helpers
- `feedback.py` - feedback ingestion and retraining support
- `model_store.py`, `storage.py` - persistence and artifact helpers
- `feature_extract.py`, `assessment.py`, `predict.py` - inference and explanation helpers

### Web Application

The web app lives in `web/` and is implemented with React and Vite.

Important files:

- `web/src/main.jsx` - main application logic, routing, and screen content
- `web/src/styles.css` - the dashboard styling system
- `web/index.html` - Vite entry HTML
- `web/script.js` - supporting front-end script

### Data And Docs

The repository also includes:

- `data/` - generated and sample datasets
- `docs/` - documentation pages and configuration references
- `tests/` - CLI and pipeline tests
- schema and collection design documents in the root

## Local Development Setup

The project has two main development loops: the Python model pipeline and the web dashboard.

### Python Environment

Typical setup:

1. Create and activate a virtual environment.
2. Install the package with `pip install .`.
3. Run the CLI or evaluation helpers from the installed entry points.
4. Use the test suite to verify the workflow after code changes.

### Web Environment

For the dashboard:

1. Install the web dependencies in `web/`.
2. Run the Vite development server.
3. Open the local dashboard in a browser.
4. Use the browser console only for debugging, not as a user flow.

### Why Two Loops Exist

The Python side owns the anomaly detection logic and artifacts. The web side owns the guided user experience and visual summaries. Keeping them separate makes the backend easier to test and the dashboard easier to iterate on.

## Patient Workflow In The Web Dashboard

The web dashboard is structured as a guided clinical workflow.

| Step | Screen | Key behavior |
| --- | --- | --- |
| 1 | Patient Details | Collects patient context and mandatory bedside inputs before the user can continue. |
| 2 | Lab Investigation | Accepts manual or uploaded labs and stays available even when only partial data is entered. |
| 3 | Patient Care Insights | Shows compact, responsive cards for severity, anomaly position, and bubble summaries. |
| 4 | Comparative Analysis | Compares models, latency, memory, and score spread in a mobile-style stacked layout. |
| 5 | Decision Support | Turns model output into action-oriented recommendations and risk guidance. |
| 6 | Model Analytical Hub | Shows model inventory, latent manifold, and reconstruction residuals for final review. |

### Step 1: Patient Details

This step captures the minimum required patient context.

The form is designed around these required inputs:

- Age
- Symptom duration
- Comorbidities
- Heart rate
- Blood pressure
- SpO2
- Body temperature
- Respiratory rate

The screen also includes optional identity and encounter details such as patient ID, full name, sex, location type, complaint text, and visit date. Those fields support documentation quality but are not the primary gating fields.

The page shows:

- Form readiness status
- Completion percentage
- BMI summary
- Triage and location context

The BMI value is shown as `0.00` until weight and height are both supplied.

The screen is intentionally simple and supports quick bedside intake. The optional identity fields help with record keeping, while the required clinical fields drive the actual workflow progress.

The Continue button remains disabled until the mandatory intake and bedside labs are complete.

Required bedside labs on Step 1:

- Hemoglobin
- Blood glucose

### Step 2: Lab Investigation

This step supports:

- Manual entry of lab values
- Upload of CSV reports
- Upload of PDF reports

The lab uploader parses the report and auto-fills matching fields.

There are no blocking required fields on Step 2. The step is designed to work with partial lab data, uploaded reports, or manually entered values.

The remaining lab fields are optional and can be provided when available:

- HbA1c
- WBC count
- Platelet count
- LDL
- HDL
- Triglycerides
- AST
- ALT
- Bilirubin
- Albumin
- Creatinine
- Urea
- eGFR
- Sodium
- Potassium
- Chloride
- Bicarbonate

The page also shows validation guidance so the user knows what is optional and what the uploader can auto-fill.

The upload parser is designed to recognize the same field names that the dashboard renders, so a generated report or a manually prepared test file can auto-fill the form without extra mapping.

### Step 3: Patient Care Insights

This is the clinical interpretation layer for the current patient record.

It combines:

- Vitals and intake information
- Lab results
- Severity calculations
- Visual summaries
- Trend and distribution charts

The screen is compact and visually grouped for quick reading.

Important behaviors:

- If no meaningful page 1 and page 2 entries exist, the charts remain zeroed
- Optional missing fields are not treated as anomalies
- Only the important required clinical inputs contribute missing-field warnings
- Intake-related labels are hidden from the visual summaries when they do not add value

The insights page is meant to answer the question: "What matters right now?" It does that by collapsing the signal into severity, area, and trend views rather than repeating the raw values.

The patient-care view includes:

- Anomaly rate gauge
- Anomaly level by area
- Clinical anomaly radar
- Anomaly score trend
- Heatmap of feature strength
- Anomaly position map
- Bubble and ranking summaries

| Card | Purpose | Responsive note |
| --- | --- | --- |
| Anomaly position map | Places the current record across body systems and severity zones. | Uses a compact stacked layout so labels do not overlap. |
| Anomaly bubbles | Shows the strongest feature-level signals. | Expands flexibly as more bubbles are added. |
| Residual-style summaries | Highlights how the strongest signals compare. | Text wraps to avoid collisions on mobile and desktop. |

### Step 4: Comparative Analysis

This step compares multiple detector families and model behavior.

It is intended to answer questions like:

- Which model is strongest?
- How wide is the score spread?
- Which model is fastest?
- Which model has the smallest memory footprint?
- Which model offers the best tradeoff between quality and runtime cost?
- Did the latest run improve or worsen the overall score?

The page includes four major information blocks:

1. Comparison matrix
2. Operational cost
3. Before / after
4. Narrative

These cards are deliberately designed to be explanation-rich, not just chart-heavy.

The comparison matrix shows the model ranking and metric balance. Operational cost shows the runtime burden. Before / after shows run-to-run change. Narrative translates the numbers into a practical reading.

| Comparison area | What changed |
| --- | --- |
| Active area | The primary workspace and reference area are separated into clear rows rather than crowded grids. |
| Supporting context | The supporting cards are stacked and resized to prevent overlap. |
| Layout behavior | Mobile-style spacing is used more consistently across screen sizes. |

### Step 5: Decision Support

This page turns the comparative output into guidance.

It includes:

- Risk summary
- Top contributing signals
- Immediate recommendations
- Follow-up plan
- Reference guidance
- Consensus display
- Consensus risk map

The language is intended to be concise and usable by a human reviewer, not only by a technical operator.

The decision support screen is where the system stops being descriptive and becomes actionable. It does not just show the score; it shows what to do next, and the current UI automatically fills the conformal verdict and drift-related fields when backend scoring data is available.

| Decision card | Purpose |
| --- | --- |
| Recommendation summary | Shows the highest-level suggestion for the current case. |
| Consensus strip | Summarizes the model agreement. |
| Risk map | Places the decision in a safety-oriented view for quick review. |

### Step 6: Model Analytical Hub

The final stage acts as a model review hub.

It is used to inspect:

- Model families
- Performance summaries
- Banding and ranking
- Comparison outputs
- Final review notes
- Latent manifold projection
- Reconstruction residual heatmaps

It serves as the wrap-up view for comparing trained model families and reviewing the strongest candidate before finalizing the flow.

| Hub card | What it shows | Current behavior |
| --- | --- | --- |
| Model inventory | ML and DL families with score, accuracy, latency, memory, and AUC. | The top score badge now matches the displayed accuracy metric. |
| Latent manifold | VAE latents projected to 2D with a Deep SVDD boundary overlay. | The card is compacted and keeps the current record visible. |
| Residuals | Per-feature reconstruction errors. | The matrix and the explanation panel are constrained so they do not overflow the card. |

## Lab Report Dataset Support

The repository includes generated lab report samples for testing the uploader and the parsing logic.

### Sample Directories

- `data/lab_report_samples/`
- `data/lab_report_mixed/`

### Purpose Of These Datasets

These files let you:

- Test CSV upload parsing
- Test PDF upload parsing
- Exercise both normal and abnormal clinical patterns
- Verify that required fields auto-fill correctly
- Check anomaly and anomaly-free cases

### Core Fields In Every Sample

Every report includes the same lab field set:

- Fasting glucose
- Postprandial glucose
- HbA1c
- Hemoglobin
- WBC count
- Platelet count
- LDL
- HDL
- Triglycerides
- AST
- ALT
- Bilirubin
- Albumin
- Creatinine
- Urea
- eGFR
- Sodium
- Potassium
- Chloride
- Bicarbonate

### Mixed Dataset Content

The mixed dataset includes:

- Normal reports
- Prediabetes and glycemic shift reports
- Anemia and CBC variation reports
- Kidney function reports
- Liver stress reports
- Electrolyte imbalance reports
- Combined multi-system abnormality reports

The idea is to provide a realistic mix for test upload, visual inspection, and training experimentation.

The datasets are useful for checking:

- Normal parsing
- Mild deviation behavior
- Strong anomaly behavior
- Mixed-field reports
- CSV and PDF upload handling
- Auto-fill robustness

They also support repeatable demonstrations when you want to show the dashboard to someone without relying on live patient data.

## Python CLI And Pipeline

The Python CLI is the backbone of the model workflow.

### Main CLI Tasks

- Train a model
- Split datasets
- Predict scores
- Evaluate predictions
- Export edge bundles
- Retrain from feedback
- Serve API-backed scoring

### Common Commands

Typical CLI commands include:

- `anomaly-cli train`
- `anomaly-cli split-data`
- `anomaly-cli predict`
- `anomaly-cli evaluate`
- `anomaly-cli export-edge`
- `anomaly-cli retrain-feedback`
- `anomaly-edge-infer`
- `anomaly-dashboard`
- `anomaly-api`

### Training Flow

The training process usually consists of:

1. Loading structured data
2. Applying preprocessing
3. Building the detector ensemble
4. Optional threshold calibration
5. Saving the trained model artifact
6. Exporting a feature map
7. Optional evaluation on a held-out set

### Prediction Flow

Prediction typically:

1. Loads a trained artifact
2. Applies the same preprocessing pipeline
3. Produces per-detector scores
4. Computes the fused anomaly score
5. Maps the result into a risk band
6. Emits structured outputs for the dashboard or reports

### Evaluation Flow

Evaluation can produce:

- Metrics
- Markdown report
- HTML report
- Dashboard HTML
- Runtime and deployment summaries
- Consensus analysis

## Model Families And Scoring

The project uses multiple detectors and presents them as a cohesive ensemble.

The model family list includes:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder
- Variational Autoencoder
- GANomaly
- CNN Autoencoder
- Anomaly Transformer
- Deep SVDD
- Ensemble fusion

The dashboard and reports show:

- Comparative score
- Precision
- Recall
- F1
- Latency
- Memory
- Model band

## Risk Bands And Interpretation

The final anomaly score is mapped into clinical-style risk bands:

- Low
- Medium
- High

These bands support concise communication in the UI, report output, and decision support layer.

The decision support logic then uses those outputs to produce:

- Immediate recommendations
- Follow-up planning
- Source guidance

## Configuration

Configuration is intentionally flexible.

### Examples Included In The Repo

- `docs/config_examples.md`
- `docs/risk_scoring_config.example.json`

### Configurable Areas

- Autoencoder latent size
- Autoencoder threshold percentile
- Autoencoder dropout and learning rate
- VAE hidden size, latent size, beta, and learning rate
- Deep SVDD architecture and pretraining options
- Ensemble fusion strategy
- Ensemble weights
- Risk scoring weights
- Threshold calibration settings

### When To Use Config Files

Use config files when you want to:

- Reproduce a training setup
- Tune the ensemble for precision or recall
- Change runtime cost tradeoffs
- Switch between fusion modes
- Adapt the model to a new dataset profile

## Feature Provenance

The preprocessing pipeline exports feature provenance so that transformed features can be traced back to raw fields.

This is important because:

- The system works with clinical data
- The dashboard and reports need explainability
- Model debugging is easier when each derived feature can be traced
- Auditing becomes possible when feature origin is visible

See:

- `docs/feature_provenance.md`

## Deployment Options

The project supports several deployment modes.

### Local Python API

Use the API server when you want:

- Real-time scoring
- Request/response scoring
- Batch scoring
- Explainability responses

### Docker

The repo includes a Dockerfile for containerized deployment.

### Edge Bundle

The edge export flow is meant for offline or low-power environments.

It packages the necessary model and preprocessing artifacts so the scoring logic can run without the full training environment.

### Streamlit

The Streamlit dashboard is useful when you want a lighter, Python-first visual interface.

### React Web Dashboard

The React dashboard is the most guided user experience in the repository.

It is best when you want:

- A step-by-step patient workflow
- Rich visual summaries
- Interactive report uploads
- Side-by-side model analysis
- A more polished clinical narrative

## Testing

Tests are included for the CLI and training artifacts.

The test suite is meant to catch:

- CLI entry point issues
- Training path regressions
- Artifact generation problems
- Evaluation output stability

When changing model logic or dashboard data contracts, it is a good idea to run the tests and a web build.

## Practical Usage Flow

The usual end-to-end path looks like this:

1. Prepare patient intake details
2. Upload or enter lab values
3. Review patient care insights
4. Run comparative analysis
5. Review decision support
6. Optionally inspect the backend/model hub stages
7. Export or deploy if needed

For development work, the most useful loop is:

1. Update data or config
2. Train or rerun the pipeline
3. Check evaluation output
4. Inspect the dashboard
5. Refine the copy, visuals, or thresholds

## Detailed Screen Guide

### Patient Details Screen

Use this screen to collect the minimum clinical anchor for the record.

Key behaviors:

- The workflow should not advance until the required clinical fields are filled
- BMI should stay at `0.00` when the height/weight pair is incomplete
- Optional demographics should remain available, but not mandatory

### Lab Investigation Screen

Use this screen to capture the analytical inputs.

Key behaviors:

- Required lab fields gate the next step
- Uploaded reports can fill the form automatically
- Empty optional lab fields should not block the workflow

### Patient Care Insights Screen

Use this screen to summarize the case after both intake and labs are available.

Key behaviors:

- Charts should remain zeroed on a fresh empty run
- Intake-only noise should not dominate the charts
- The view should stay compact and easy to scan

### Comparative Analysis Screen

Use this screen to compare detectors and runtime cost.

Key behaviors:

- The model matrix should explain the ranking
- The cost view should make runtime tradeoffs obvious
- Before / after should show the direction of change
- Narrative should stay clinician-friendly

### Decision Support Screen

Use this screen to turn the analysis into recommendations.

Key behaviors:

- Recommendations should appear automatically after analysis
- Consensus should be readable at a glance
- Conformal and drift summaries should stay visible when backend values exist

## Operational Notes

- Keep the same preprocessing path between training and scoring
- Treat the generated lab samples as testing fixtures, not production data
- Use the documentation index as the entry point to supporting references
- Use the feature provenance docs when debugging transformed columns
- Use the config examples when tuning model behavior

## Suggested Reading Order

If you are new to the project, read in this order:

1. `README.md`
2. `docs/index.md`
3. `docs/feature_provenance.md`
4. `docs/config_examples.md`
5. `normalized_healthcare_schema.md`
6. `vital_signs_data_collection_parameter_design.md`
7. This document

## Summary

This repository is a full anomaly detection workflow for rural healthcare use cases. It combines:

- Structured clinical data handling
- Ensemble anomaly detection
- Explainability and evaluation
- A guided web dashboard
- Generated report datasets for testing
- Deployment support for API, container, and edge use

The goal of the system is not only to detect anomalies, but to present them clearly enough that a clinician or reviewer can trust the result and act on it.
