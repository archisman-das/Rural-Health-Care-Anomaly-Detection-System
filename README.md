# Rural Health Anomaly Detection

Rural Health Anomaly Detection is a full workflow for identifying unusual patterns in rural healthcare data and presenting them in a form that is easier to review, explain, and act on.

The project combines:

- A Python anomaly-detection and evaluation pipeline
- A React-based clinical dashboard
- Sample lab report datasets for CSV and PDF upload testing
- Documentation for the workflow, schema, configuration, and feature provenance

If you want the deeper design and workflow reference, start here:

- [Project documentation](docs/project_documentation.md)
- [System architecture](docs/architecture.md)
- [Project scope](docs/project_scope.md)
- [Research report](docs/research_report.md)
- [Research report outline](docs/research_report_outline.md)
- [Model details documentation](docs/model_details.md)
- [Reference papers](docs/reference.md)
- [Documentation index](docs/index.md)

## At A Glance

The system follows a simple clinical flow:

| Step | Screen | What it does |
| --- | --- | --- |
| 1 | Patient Details | Collects patient context and the bedside inputs needed to unlock the workflow. |
| 2 | Lab Investigation | Accepts manual or uploaded lab values, with responsive reference/help cards. |
| 3 | Patient Care Insights | Summarizes the current record with compact, mobile-friendly insight cards. |
| 4 | Comparative Analysis | Compares models, runtime cost, and risk behavior in a stacked layout. |
| 5 | Decision Support | Converts the analysis into practical next steps and consensus guidance. |
| 6 | Model Analytical Hub | Surfaces latent manifold, residuals, and model inventory for final review. |

Internal backend processing still exists in the architecture, but it is no longer presented as a separate public step.

The goal is not only to detect anomalies, but to present them clearly enough that a clinician or reviewer can understand what changed and what to do next.

## What The Repository Contains

### Python Pipeline

The Python side handles:

- Preprocessing
- Training
- Scoring
- Evaluation
- Model export
- Edge inference
- Feedback-based retraining

### Web Dashboard

The web app provides a guided, step-by-step front end for:

- Patient intake
- Lab report upload
- Insight summaries
- Comparative analysis
- Decision support

### Dataset Assets

The repository includes generated sample reports and mixed datasets that can be uploaded directly into the lab investigation step.

### Documentation

The docs explain:

- Feature provenance
- Configuration examples
- Normalized schema
- Data collection design
- Full project workflow

## Documentation Links

- [Project documentation](docs/project_documentation.md)
- [System architecture](docs/architecture.md)
- [Project scope](docs/project_scope.md)
- [Research report](docs/research_report.md)
- [Research report outline](docs/research_report_outline.md)
- [Model details documentation](docs/model_details.md)
- [Reference papers](docs/reference.md)
- [Feature provenance](docs/feature_provenance.md)
- [Configuration examples](docs/config_examples.md)
- [Risk scoring config example](docs/risk_scoring_config.example.json)
- [Normalized healthcare schema](normalized_healthcare_schema.md)
- [Vital signs data collection and parameter design](vital_signs_data_collection_parameter_design.md)

## Requirements

You will generally need:

- Python 3.11 or newer
- Pip
- Node.js and npm for the web dashboard

If you are only using the Python CLI, the Node.js part is optional. If you are working with the dashboard, install both the Python and web dependencies.

## Installation

### Python Package

```bash
pip install .
```

### Web Dashboard

```bash
cd web
npm install
```

## Quick Start

### 1. Train A Model

```bash
anomaly-cli train --input train.csv --output artifacts/model.joblib --config-json config.json
```

### 2. Score New Data

```bash
anomaly-cli predict --model artifacts/model.joblib --input test.csv --output artifacts/predictions.csv
```

### 3. Evaluate The Results

```bash
anomaly-cli evaluate --input artifacts/predictions.csv --labels-file labels.csv --labels-column label --report-prefix artifacts/report
```

### 4. Export An Edge Bundle

```bash
anomaly-cli export-edge --model artifacts/model.joblib --output-dir artifacts/edge_bundle --opset 13
```

### 5. Run The Web Dashboard

```bash
cd web
npm run dev
```

## The Clinical Workflow

### Step 1. Patient Details

The first screen collects the minimum information needed to start the case and includes the bedside checks that gate progress.

Required inputs:

- Age
- Symptom duration
- Comorbidities
- Heart rate
- Blood pressure
- SpO2
- Body temperature
- Respiratory rate

Required bedside labs:

- Hemoglobin
- Blood glucose

Optional context:

- Patient ID
- Full name
- Sex
- Location type
- Chief complaint
- Visit date

The page shows intake readiness, BMI, triage, and location. BMI remains `0.00` until weight and height are both entered. The Continue button stays disabled until the mandatory fields are complete.

### Step 2. Lab Investigation

This screen supports both manual lab entry and uploaded reports.

Required lab fields:

None. Step 2 is now intentionally optional so users can continue even when only partial lab data is available.

Optional lab fields:

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

The uploader accepts CSV and PDF reports and auto-fills recognized values. Missing lab values do not block progress.

### Step 3. Patient Care Insights

This screen summarizes the current case with compact visualizations.

It is designed to answer:

- How strong is the overall anomaly signal?
- Which area needs attention?
- How does the current pattern look across vitals and labs?
- What is the severity distribution?

The page keeps the visuals compact and avoids noise from empty optional fields. It includes severity, radar, trend, heatmap, position, bubble, and ranking views.

### Step 4. Comparative Analysis

This screen compares model behavior and runtime cost.

It focuses on:

- Comparison matrix
- Operational cost
- Before / after change
- Narrative summary

The goal is to make the ranking, tradeoffs, and score movement easy to interpret at a glance.

### Step 5. Decision Support

This screen converts the analysis into practical next steps.

It includes:

- Immediate recommendations
- Follow-up planning
- Reference guidance
- Model consensus
- Consensus risk map

### Step 6. Model Analytical Hub

This is the final review stage.

It groups:

- Model families
- Model rankings
- Comparison outputs
- Final review notes
- Latent manifold and reconstruction residual views

| Hub Card | Purpose | Notable behavior |
| --- | --- | --- |
| Latent manifold | Shows VAE latents projected to 2D with a Deep SVDD boundary overlay. | Keeps the current record highlighted and uses a compact chart frame across screen sizes. |
| Residuals | Shows per-feature reconstruction errors for the current record. | Uses a readable summary panel and a constrained matrix so content does not overflow on mobile or desktop. |
| Model inventory | Lists trained ML and DL models by family. | The top score badge now matches the displayed accuracy metric. |

## Sample Data And Test Reports

The repository includes upload-ready sample reports in:

- `data/lab_report_samples/`
- `data/lab_report_mixed/`

These datasets are useful for:

- CSV upload testing
- PDF upload testing
- Normal cases
- Anomaly cases
- Borderline cases
- Mixed multi-system abnormality cases

Each report contains the same core lab field set used by the dashboard.

## Core Model Families

The project compares multiple detectors, including:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder
- Variational Autoencoder
- GANomaly
- CNN Autoencoder
- Anomaly Transformer
- Deep SVDD
- Mixture-of-Experts gate
- Nonlinear stacking meta-model

The model comparison views show:

- Score
- Precision
- Recall
- F1
- Gate weights for MoE runs
- Stacking meta-model probability
- Latency
- Memory
- Family and band

## Evaluation Output

The evaluation pipeline can generate:

- JSON reports
- Markdown reports
- HTML reports
- Dashboard-style HTML summaries
- Runtime and deployment summaries

The reports can include accuracy-style metrics, score distribution analysis, consensus information, and edge-readiness details.

## Edge And Deployment Support

The project supports:

- A local API server
- Docker-based deployment
- Edge bundle export
- Offline scoring from the exported bundle
- Feedback-led retraining

Typical API usage:

```bash
anomaly-api --model artifacts/model.joblib --host 0.0.0.0 --port 8001
```

Typical edge export usage:

```bash
anomaly-cli export-edge --model artifacts/model.joblib --output-dir artifacts/edge_bundle --opset 13
```

Typical offline scoring usage:

```bash
anomaly-edge-infer --bundle-dir artifacts/edge_bundle --input test.csv --output artifacts/edge_predictions.csv
```

## Helpful Development Commands

### Inspect CLI Help

```bash
anomaly-cli train --help
anomaly-cli evaluate --help
anomaly-cli export-edge --help
```

### Run The Web Build

```bash
cd web
npm run build
```

### Run The Python Tests

```bash
pytest
```

## Configuration And Tuning

If you want to tune the system behavior, use:

- `docs/config_examples.md`
- `docs/risk_scoring_config.example.json`

These pages show examples for:

- Autoencoder tuning
- Variational autoencoder tuning
- Deep SVDD architecture choices
- Stacking meta-model selection and hidden-layer sizing
- Ensemble fusion weights
- Risk scoring weights
- Stacking and max-score-voting modes

## Feature Provenance

If you need to understand how raw clinical fields turn into engineered model inputs, see:

- [Feature provenance](docs/feature_provenance.md)

This is especially useful when debugging transformed columns or explaining model behavior.

## Suggested Reading Order

If you are new to the repository, read these in order:

1. `README.md`
2. `docs/index.md`
3. `docs/project_documentation.md`
4. `docs/model_details.md`
5. `docs/reference.md`
6. `docs/feature_provenance.md`
7. `docs/config_examples.md`
8. `normalized_healthcare_schema.md`
9. `vital_signs_data_collection_parameter_design.md`

## Troubleshooting

### The Web App Does Not Start

- Make sure you ran `npm install` inside `web/`
- Make sure you are starting the dev server from the `web/` folder
- If the page looks stale, rebuild with `npm run build`

### The Lab Upload Does Not Fill Fields

- Confirm the CSV or PDF includes the expected lab field names
- Check that the file uses recognized labels like fasting glucose, postprandial glucose, and hemoglobin
- Use the generated sample reports in `data/lab_report_mixed/` to verify the parser

### The Pipeline Feels Incomplete

- Re-run the web build
- Re-run the Python tests
- Check the documentation index for the workflow stage you are modifying

## License

See [LICENSE](LICENSE).
