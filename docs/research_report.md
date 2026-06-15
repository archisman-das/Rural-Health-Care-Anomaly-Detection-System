# Research Report

This document is a research-facing summary of the Rural Health Care Anomaly Detection System. It is written to help with project writeups, internal reporting, and paper-style documentation.

## Working Title

Rural Health Care Anomaly Detection with Multi-Model Fusion, Compact Clinical Visualization, and Workflow-Guided Decision Support

## Abstract

This project presents a workflow-driven anomaly detection system for rural healthcare records. The system combines structured patient intake, lab report ingestion, multiple anomaly detectors, and a clinician-friendly dashboard. Instead of exposing raw scores alone, the platform presents compact visual summaries, comparative model analysis, decision support, conformal-style anomaly verdicts, sequence drift signals, latent-space projections, and reconstruction residual views.

The system is designed for tabular clinical records and mixed CSV or PDF lab report inputs. It supports both classical and neural detectors, ensemble fusion, and hidden internal review stages. The goal is to make unusual patient patterns easier to detect, compare, and explain in a practical review workflow.

## Problem Statement

Clinical anomaly detection on rural healthcare data has several challenges:

- Records are tabular and heterogeneous
- Missing values are common
- Labs may arrive as uploaded reports rather than clean tables
- A single detector is often not enough
- Clinicians need summaries, not just scores
- Drift and score changes over time can matter even when individual values look mild

This project addresses those issues by combining detection, fusion, explanation, and decision support in one system.

## Research Objectives

The system is designed to:

1. Detect abnormal clinical patterns in structured patient records.
2. Accept both manual entry and uploaded lab reports.
3. Compare several anomaly detectors on the same patient record.
4. Fuse detector outputs into a more stable final signal.
5. Present the output in a compact and clinically readable way.
6. Surface conformal, sequence, drift, and latent-space explanations.
7. Support research-style documentation and reproducibility.

## System Overview

The platform includes:

- A Python anomaly-detection pipeline
- A React clinical dashboard
- Mixed anomaly and anomaly-free sample datasets
- Documentation for architecture, scope, model details, and references

The workflow starts with patient details, moves through lab investigation, then shows patient care insights, comparative analysis, decision support, and a final model hub.

## Method Summary

The model stack includes:

- Isolation-based methods
- Boundary-based methods
- Local density methods
- Reconstruction-based neural methods
- One-class hypersphere methods
- Attention-based anomaly detection
- Ensemble fusion strategies
- Mixture-of-experts gating
- Stacking meta-models
- Conformal-style anomaly reporting
- Drift-aware sequence summaries

### Fusion Strategy

The project does not depend on a single detector. It combines multiple detectors and can route or blend their outputs using:

- Weighted averaging
- Max-score voting
- Stacking
- Mixture-of-experts gating

This makes the system more robust to detector disagreement and varied clinical patterns.

### Explanation Strategy

The dashboard is not just a score viewer. It also exposes:

- Compact clinical heatmaps
- Area-level anomaly summaries
- Model comparison cards
- Score drift charts
- Conformal verdicts
- Latent manifold visualizations
- Reconstruction residual heatmaps

These views help explain why a record was marked unusual.

## Data And Dataset Design

The project uses a stable feature set that covers:

- Vital signs
- Symptom duration
- Comorbidities
- Hemoglobin
- Glucose measures
- CBC measures
- Kidney and liver function measures
- Electrolytes

The repository includes generated mixed datasets that contain both anomaly-free and anomalous records. These samples are intended for upload testing, research demonstrations, and controlled experimentation.

## Experimental Workflow

A typical experimental workflow is:

1. Prepare train, validation, and test splits.
2. Train or calibrate model components.
3. Score the validation and test records.
4. Compare detector outputs and fusion behavior.
5. Review anomaly explanations in the dashboard.
6. Record findings in a research summary.

## Research Themes Supported By The Codebase

### 1. Multi-Detector Clinical Fusion

The system can compare how classical and neural detectors behave on the same patient record.

### 2. Compact Clinical Visualization

The patient-care screen is designed to keep the output compact, side-by-side, and easy to read.

### 3. Conformal Anomaly Reporting

The decision-support page can surface significance-style anomaly verdicts derived from calibration data.

### 4. Sequence Drift Awareness

The workflow can highlight score stream change points and slow drift patterns.

### 5. Latent Geometry And Residual Analysis

The analytical hub can show a latent projection and a reconstruction residual heatmap to reveal structure hidden inside the raw scores.

## Evaluation Angle

The project is suited for research questions such as:

- Which detector family is strongest on this feature set?
- Does ensemble fusion improve stability?
- Can latent-space views reveal clusters that raw scores miss?
- Does drift-aware scoring help identify gradual deterioration?
- Do compact explanations improve interpretability for clinical review?

## Limitations

Current limitations include:

- The project is still a software research platform, not a clinical deployment
- The sample datasets are synthetic or generated for testing
- The dashboard is optimized for structured records, not free-text notes
- Ground-truth anomaly labels may be limited or unavailable in some flows

## Expected Contributions

The repository supports several reportable contributions:

- Workflow-driven anomaly detection for rural healthcare data
- Multi-model fusion for heterogeneous clinical records
- Compact and readable dashboard explanations
- Conformal and drift-aware decision support
- Latent-space and residual visual diagnostics

## Related Files

Use these files alongside this report:

- `README.md`
- `docs/architecture.md`
- `docs/project_scope.md`
- `docs/project_documentation.md`
- `docs/model_details.md`
- `docs/reference.md`
- `docs/feature_provenance.md`
- `docs/index.md`

## Suggested Research Writing Structure

If you are turning this into a formal report, use this structure:

1. Abstract
2. Introduction
3. Problem statement
4. Related work
5. System architecture
6. Methods
7. Data and preprocessing
8. Experimental setup
9. Results
10. Discussion
11. Limitations
12. Conclusion

## Conclusion

The project is best described as a clinical anomaly detection workflow with strong emphasis on explainability, comparison, and practical decision support. It is suitable for research reporting because it combines model diversity, structured data handling, and an explainable dashboard in one repository.
