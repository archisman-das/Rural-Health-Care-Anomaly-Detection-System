# System Architecture

This document describes how the rural health anomaly detection system is organized, how data moves through the application, and how the major backend and frontend pieces fit together.

## Goals

The architecture is designed to:

- Keep clinical intake simple
- Support both manual entry and uploaded lab reports
- Run multiple anomaly detectors on the same patient record
- Present the output in a clinician-friendly workflow
- Separate training, scoring, explanation, and reporting concerns
- Allow hidden backend stages to remain part of the system without exposing them in the main user flow

## High-Level View

The system has four main layers:

1. Data ingestion and validation
2. Feature processing and model scoring
3. Explanation and decision support
4. Web presentation and documentation

The Python backend owns preprocessing, training, prediction, calibration, and export. The React dashboard owns the user journey, visual summaries, and workflow navigation.

## Main Components

### 1. Data Layer

The data layer includes:

- Patient intake fields
- Lab test values
- CSV and PDF report uploads
- Generated sample datasets for testing
- Train, validation, and test splits
- Configuration and provenance files

The repository supports both small interactive records and larger batch-style datasets.

### 2. Preprocessing Layer

The preprocessing layer normalizes raw records into model-ready features.

It handles:

- Missing value handling
- Type conversion
- Encoding
- Scaling
- Feature expansion
- Optional longitudinal and interaction features

The preprocessing path is shared across training and inference so the model sees the same feature space in both places.

### 3. Model Layer

The model layer contains the anomaly detectors and fusion logic.

Typical model families include:

- Classical detectors such as Isolation Forest, One-Class SVM, and LOF
- Reconstruction models such as Autoencoder, VAE, GANomaly, and CNN Autoencoder
- Attention-based detectors such as Anomaly Transformer
- One-class hypersphere models such as Deep SVDD
- Ensemble fusion methods such as weighted average, stacking, and MoE gating

The model layer can emit:

- Per-detector anomaly scores
- Fused anomaly score
- Risk category
- Thresholded alert output
- Explainability signals

### 4. Explanation Layer

The explanation layer converts raw model output into visuals and human-readable summaries.

It includes:

- Patient care insight charts
- Comparative analysis cards
- Decision-support recommendations
- Conformal verdicts
- Sequence drift summaries
- Latent manifold views
- Reconstruction residual heatmaps

These views are presentation layers built on top of model outputs, not separate primary detectors.

## Frontend Architecture

The frontend is a React application under `web/`.

It presents the workflow in ordered steps:

1. Patient details
2. Lab investigation
3. Patient care insights
4. Comparative analysis
5. Decision support
6. Hidden backend processing stage
7. Model analytical hub

The main dashboard is intentionally workflow-driven. Each step exposes only the information needed for that part of the review.

### Frontend Responsibilities

The UI handles:

- Guided data entry
- Upload parsing feedback
- Compact visual summaries
- Model comparison cards
- Decision support panels
- Hidden or internal-only views when needed

The frontend does not train models. It consumes backend prediction and explanation payloads.

## Backend Architecture

The Python backend owns the core computation.

### Core Responsibilities

- Parse and validate input data
- Build training and test splits
- Train detectors and ensembles
- Score new records
- Calibrate thresholds
- Export reports and edge bundles
- Produce backend explanation payloads

### Service Boundaries

The backend is organized around reusable modules rather than one large script.

Common responsibilities are split across:

- Preprocessing modules
- Training orchestration
- Detector implementations
- Ensemble fusion
- Evaluation and reporting
- Storage and artifact helpers

This keeps the training path, scoring path, and explanation path easier to test independently.

## Workflow by Stage

### Stage 1: Patient Details

The record starts with required clinical context.

This stage captures:

- Vital signs
- Symptom timing
- Comorbidities
- Patient identity context

BMI stays at `0.00` until the needed inputs are available.

### Stage 2: Lab Investigation

This stage accepts manual lab entry or uploaded report files.

It supports:

- Required lab fields
- Optional lab fields
- CSV parsing
- PDF extraction

The upload parser maps report labels into the same field structure used by the form.

### Stage 3: Patient Care Insights

This stage turns the current record into compact clinical summaries.

It emphasizes:

- Anomaly strength
- Area-level contribution
- Distribution views
- Trend views
- Feature heatmaps

It is designed to stay readable even when some optional data is missing.

### Stage 4: Comparative Analysis

This stage compares model families and cost characteristics.

It focuses on:

- Ranking
- Runtime cost
- Before and after comparison
- Narrative interpretation

### Stage 5: Decision Support

This stage converts the analysis into action-oriented output.

It can show:

- Recommendations
- Follow-up guidance
- Consensus summaries
- Conformal verdicts
- Sequence and drift signals

### Stage 6: Hidden Backend Processing

This stage remains part of the architecture, but it is hidden from the main user flow.

It exists to document:

- Preprocessing
- Feature engineering
- Validation
- Model-ready transformation

### Stage 7: Model Analytical Hub

This stage is the final internal review hub.

It can surface:

- Model family summaries
- Latent manifold projection
- Reconstruction residual heatmaps
- Comparison output
- Final notes

## Data Flow

The normal request flow is:

1. User enters or uploads clinical data.
2. The frontend validates the visible fields.
3. The backend normalizes the record.
4. The detector stack scores the record.
5. Fusion logic combines the detector outputs.
6. Explanation helpers generate summaries and visuals.
7. The dashboard renders the result in the relevant step.

This flow is the same whether the input comes from a manual form or from a parsed lab report.

## Storage And Artifacts

The system can produce and consume:

- Trained model bundles
- Prediction outputs
- Evaluation reports
- Edge deployment exports
- Generated sample datasets
- Documentation artifacts

These artifacts make the project easier to test, reproduce, and deploy.

## Extensibility

The architecture is built so that new detectors or new visualizations can be added without rewriting the whole app.

Common extension points include:

- Adding another detector to the ensemble
- Adding another explanation chart
- Extending lab parsing rules
- Changing the fusion strategy
- Updating risk mapping rules

## Operational Notes

- Keep the preprocessing path consistent between training and inference
- Treat generated sample reports as test fixtures
- Keep hidden internal workflow stages out of the main user path unless they are needed for debugging
- Update the documentation whenever the dashboard workflow changes

## Summary

The project is structured as a workflow-driven clinical anomaly system:

- The Python side computes models and explanations
- The React side presents the workflow
- The docs describe how the pieces fit together

That separation keeps the system easier to understand, easier to test, and easier to extend.
