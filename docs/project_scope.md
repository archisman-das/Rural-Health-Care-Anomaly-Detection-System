# Project Scope

This document defines what the Rural Health Care Anomaly Detection System is intended to cover, what it does not try to solve, and how to interpret its current implementation.

## In Scope

The project is focused on:

- Structured rural healthcare anomaly detection
- Patient intake and lab result capture
- CSV and PDF lab report upload handling
- Multi-model anomaly scoring and comparison
- Compact clinical visualizations
- Decision support and workflow guidance
- Backend scoring, calibration, and explanation output
- Documentation for models, architecture, and provenance

The system is intentionally built around a full clinical review flow rather than a single scoring endpoint.

## Primary Use Cases

The repository is meant to support:

1. Manual patient intake in the dashboard
2. Uploading or entering lab investigations
3. Reviewing patient-care insights in a compact visual format
4. Comparing model families and runtime cost
5. Producing decision-support summaries
6. Inspecting latent and residual explanation views in the model hub
7. Training, evaluating, and exporting models from the Python side

## Data Scope

The project works with:

- Structured tabular clinical records
- Generated sample datasets
- Mixed anomaly and anomaly-free lab reports
- Train, validation, and test splits

The dataset examples are meant for testing, development, and demonstration. They are not a substitute for a real clinical data pipeline.

## Feature Scope

The dashboard and model pipeline are centered on:

- Vital signs
- Patient context
- Lab values
- Derived severity signals
- Model comparison metrics
- Explanation outputs

The current forms and generated reports use a stable clinical feature set so the same fields can move across upload, scoring, and documentation workflows.

## Model Scope

The project includes a broad anomaly-detection stack:

- Classical detectors
- Reconstruction-based neural models
- Hypersphere-style one-class models
- Attention-based models
- Ensemble fusion methods
- Mixture-of-experts gating
- Stacking meta-models
- Conformal-style anomaly reporting
- Sequence and drift-aware summaries

The scope is to compare and fuse detector behavior for tabular healthcare use cases, not to build a single monolithic model.

## UI Scope

The React dashboard is responsible for:

- Step-by-step patient workflow
- Compact visuals for anomaly review
- Decision support output
- Hidden internal review stages
- Model analytical summaries

The UI is not intended to expose every backend detail directly. Some stages remain hidden or internal-only by design.

## Out Of Scope

The project does not aim to be:

- A substitute for a licensed medical diagnosis system
- A real-time hospital EHR replacement
- A production clinical decision authority
- A public health surveillance platform
- A full longitudinal patient registry
- A regulatory submission package

It is also not trying to replace clinician judgment. The system is meant to support review, not to make final medical decisions on its own.

## Current Implementation Boundaries

The current codebase emphasizes:

- Local development and browser-based review
- Upload-based testing
- Modular Python scoring
- React dashboard presentation
- Documentation-driven interpretation

It does not require live hospital integrations to exercise the main workflow.

## Assumptions

The project assumes:

- A structured input schema
- A consistent preprocessing path for training and inference
- The presence of enough clinical context to compute meaningful summaries
- That missing optional fields should not break the workflow

## Success Criteria

The project is successful when it can:

- Accept or parse the expected patient and lab fields
- Produce stable anomaly scores and explanations
- Show clear visual summaries in the dashboard
- Support model comparison and decision support
- Keep the workflow understandable for a non-technical reviewer

## Maintenance Notes

When the workflow changes, this document should be updated alongside:

- `README.md`
- `docs/architecture.md`
- `docs/project_documentation.md`
- `docs/model_details.md`

Keeping scope explicit helps prevent the docs from drifting away from the actual app behavior.
