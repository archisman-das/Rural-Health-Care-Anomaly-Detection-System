# Rural Health Anomaly Detection

Schema-aware preprocessing, training, and inference utilities for rural health
anomaly detection workflows.

Docs index: [docs/index.md](docs/index.md)

## Install

```bash
pip install .
```

## CLI

Use the installed entry points to train a model and score new data:

```bash
anomaly-cli train --input train.csv --output artifacts/model.joblib --feature-map artifacts/feature_map.csv --config-json config.json --ensemble-fusion-strategy max_score_voting --ensemble-max-score-threshold 0.8
anomaly-cli train --synthetic-demo-data --synthetic-demo-rows 9600 --synthetic-demo-seed 42 --no-calibrate-threshold --output artifacts/model.joblib --feature-map artifacts/feature_map.csv --config-json config.json
anomaly-cli train --input train.csv --output artifacts/model.joblib --config-json config.json --calibrate-threshold --calibration-min-samples 25
anomaly-cli split-data --input dataset.csv --output-dir data/splits --train-fraction 0.7 --validation-fraction 0.15 --test-fraction 0.15 --group-column patient_id --random-state 42
anomaly-cli predict --model artifacts/model.joblib --input test.csv --output artifacts/predictions.csv
anomaly-cli evaluate --input artifacts/predictions.csv --labels-file labels.csv --labels-column label --report-prefix artifacts/report
anomaly-cli evaluate --input artifacts/predictions.csv --report-prefix artifacts/report
anomaly-cli evaluate --input artifacts/predictions.csv --dashboard-html artifacts/dashboard.html
anomaly-cli export-edge --model artifacts/model.joblib --output-dir artifacts/edge_bundle --opset 13
anomaly-cli retrain-feedback --input train.csv --feedback-file artifacts/feedback_ledger.jsonl --output artifacts/model_refreshed.joblib
anomaly-edge-infer --bundle-dir artifacts/edge_bundle --input test.csv --output artifacts/edge_predictions.csv
python dashboard_server.py --input artifacts/predictions.csv --labels-file labels.csv --labels-column label
anomaly-dashboard --input artifacts/predictions.csv --labels-file labels.csv --labels-column label
```

`anomaly-cli train` now describes the full default ensemble in its help text,
including the Anomaly Transformer, GANomaly, VAE, and CNN autoencoder detectors.

`anomaly-cli export-edge` now advertises the autoencoder, Anomaly Transformer,
GANomaly, VAE, CNN autoencoder, and Deep SVDD artifacts in its help text, and
`anomaly-edge-infer` does the same for the offline scoring bundle.

Tip: start from [docs/risk_scoring_config.example.json](docs/risk_scoring_config.example.json) when you want to tune the blended risk score weights.

For an interactive Streamlit view with a patient risk map, score trends, alert
explanations, and model agreement, run:

The comparison tables include Isolation Forest, One-Class SVM, Local Outlier
Factor, Autoencoder, Variational Autoencoder, CNN Autoencoder, and Deep SVDD.

```bash
streamlit run streamlit_dashboard.py
```

Dedicated commands are also available:

```bash
anomaly-train --input train.csv --output artifacts/model.joblib --config-json config.json
anomaly-train --synthetic-demo-data --synthetic-demo-rows 9600 --synthetic-demo-seed 42 --no-calibrate-threshold --output artifacts/model.joblib --config-json config.json
anomaly-train --input train.csv --output artifacts/model.joblib --config-json config.json --calibrate-threshold --calibration-min-samples 25
anomaly-cli split-data --input dataset.csv --output-dir data/splits --train-fraction 0.7 --validation-fraction 0.15 --test-fraction 0.15 --group-column patient_id --random-state 42
anomaly-predict --model artifacts/model.joblib --input test.csv --output artifacts/predictions.csv
anomaly-evaluate --input artifacts/predictions.csv --labels-file labels.csv --labels-column label --report-prefix artifacts/report
anomaly-evaluate --input artifacts/predictions.csv --report-prefix artifacts/report
anomaly-evaluate --input artifacts/predictions.csv --dashboard-html artifacts/dashboard.html
anomaly-edge-export --model artifacts/model.joblib --output-dir artifacts/edge_bundle --opset 13
anomaly-cli retrain-feedback --input train.csv --feedback-file artifacts/feedback_ledger.jsonl --output artifacts/model_refreshed.joblib
anomaly-edge-infer --bundle-dir artifacts/edge_bundle --input test.csv --output artifacts/edge_predictions.csv
python dashboard_server.py --input artifacts/predictions.csv --labels-file labels.csv --labels-column label
anomaly-dashboard --input artifacts/predictions.csv --labels-file labels.csv --labels-column label
```

Use `--synthetic-demo-data` when you want a larger local demo cohort without preparing an input file first. The default synthetic size is 9,600 rows, and you can adjust it with `--synthetic-demo-rows` and `--synthetic-demo-seed`.

Use real `--input` data when you have a CSV or Parquet file from the field. Use synthetic demo mode when you want to stress-test the pipeline, explore the UI, or show a larger training flow without waiting on a real dataset.

Use `--calibrate-threshold` when you want labeled training runs to tune the final decision cutoff for better precision and F1. The calibration step only runs after the labeled set reaches `calibration_min_samples` rows, which keeps tiny label sets from overfitting the threshold. Pass `--no-calibrate-threshold` if you want to keep the model's native threshold behavior instead. If you need to override that floor from the command line, use `--calibration-min-samples`.

## Deployment

For a small FastAPI backend that serves real-time patient scoring:

The API backend uses the same ensemble comparison set, including Local Outlier
Factor alongside Isolation Forest, One-Class SVM, Autoencoder, Variational
Autoencoder, CNN Autoencoder, and Deep SVDD.

```bash
anomaly-api --model artifacts/model.joblib --host 0.0.0.0 --port 8001
```

It exposes three endpoints:

- `GET /health`
- `POST /predict`
- `POST /batch-predict`
- `POST /predict_file`
- `POST /batch`
- `POST /explain`
- `POST /explain_file`

`/predict` accepts one patient record as JSON and returns the scored result in
real time. `/batch-predict` accepts a list of patient records for small batch
jobs. `/predict_file` accepts a CSV upload for quick file-based scoring.
`/batch` accepts the same CSV upload flow for batch scoring.
`/explain` accepts a single patient record and returns the scored result plus
feature importance for the flagged anomaly. The API returns SHAP values when
the `shap` package is available and falls back to deterministic local feature
attributions otherwise.
`/explain_file` uploads a CSV and explains the most anomalous row in the file.
Set `--auth-token` on `anomaly-api`, or define `API_AUTH_TOKEN`, to require the
`X-API-Token` header on every request.

For offline edge deployment on low-power devices, export the fitted ensemble
and ONNX artifacts with:

The exported bundle matches the same detector set used in the dashboard and
backend comparisons, including Local Outlier Factor.

```bash
anomaly-cli export-edge --model artifacts/model.joblib --output-dir artifacts/edge_bundle --opset 13
```

The export command help now calls out the bundled autoencoder, Anomaly
Transformer, GANomaly, variational autoencoder, CNN autoencoder, and Deep SVDD
artifacts explicitly.

The bundle includes the fitted preprocessor, feature map exports, an ONNX file
for Isolation Forest, an ONNX file for the autoencoder, an
`anomaly_transformer.joblib` artifact for the Anomaly Transformer, a
`ganomaly.joblib` artifact for GANomaly, an ONNX file for the variational
autoencoder, and an ONNX file for the Deep SVDD model.

To score data from the exported bundle on an offline device, run:

```bash
anomaly-edge-infer --bundle-dir artifacts/edge_bundle --input test.csv --output artifacts/edge_predictions.csv
```

The edge inference help now also calls out the bundled autoencoder, GANomaly,
VAE, CNN autoencoder, and Deep SVDD artifacts.

Clinicians can submit feedback to the running API with `POST /feedback` for a
single alert review or `POST /feedback_batch` for a batch of reviews. The API
appends these records to a JSONL ledger that the retraining command can use
later.

To run the API in Docker:

```bash
docker build -t rural-health-anomaly .
docker run --rm -p 8001:8001 -v "%cd%/artifacts:/models" rural-health-anomaly
```

Mount a trained `model.joblib` at `/models/model.joblib` before starting the
container.

`anomaly-evaluate` writes the executive summary by default for `--report-prefix`,
`--report-md`, and `--report-html`. Use `--no-executive-summary` if you want the
full report sections instead. Use `--dashboard-html` for the full dashboard view
with metrics, score distributions, agreement analysis, and runtime comparison.

If you want to see every training override in one place, use the CLI help:

```bash
anomaly-cli train --help
```

You can also inspect the evaluation command with:

```bash
anomaly-cli evaluate --help
```

The evaluation command accepts:

- `--input`
- `--score-column`
- `--score-columns`
- `--threshold`
- `--output`
- `--report-prefix`
- `--report-md`
- `--report-html`
- `--dashboard-html`
- `--top-fraction`
- `--executive-summary`
- `--labels-file`
- `--labels-column`
- `--label-column`

Use `--executive-summary` when you only want the side-by-side model comparison,
the best single model, and the ensemble score in the output report.

When you use `--report-prefix`, `--report-md`, or `--report-html`, the Markdown
and HTML reports default to this executive summary. Pass
`--no-executive-summary` if you want the full report sections instead. Use
`--dashboard-html` when you want the full dashboard with metrics, score
distribution, reconstruction histograms, agreement, and runtime metrics.

That help output includes direct flags for the autoencoder, Anomaly Transformer,
GANomaly, variational autoencoder, and Deep SVDD settings, including:

- `--autoencoder-latent-dim`
- `--autoencoder-threshold-percentile`
- `--autoencoder-dropout`
- `--autoencoder-learning-rate`
- `--anomaly-transformer-hidden-dim`
- `--anomaly-transformer-latent-dim`
- `--anomaly-transformer-attention-weight`
- `--anomaly-transformer-weight`
- `--ganomaly-hidden-dim`
- `--ganomaly-latent-dim`
- `--ganomaly-consistency-weight`
- `--ganomaly-weight`
- `--vae-hidden-dim`
- `--vae-latent-dim`
- `--vae-beta`
- `--vae-learning-rate`
- `--deep-svdd-architecture`
- `--deep-svdd-nu`
- `--deep-svdd-pretrain-autoencoder` / `--no-deep-svdd-pretrain-autoencoder`

For a compact code example that fits and scores every detector through the
shared `fit()` / `score()` interface, see
[rural_health_anomaly/common_interface_demo.py](rural_health_anomaly/common_interface_demo.py).

Common autoencoder settings you can pass through `config.json`:

- `autoencoder_latent_dim`: latent space size, usually `8` to `16`
- `autoencoder_threshold_percentile`: reconstruction cutoff percentile, usually `95.0` to `99.0`
- `autoencoder_dropout`: dropout rate for hidden layers, usually `0.1` to `0.3`
- `autoencoder_learning_rate`: optimizer step size, usually `1e-3` to `1e-4`
- `vae_hidden_dim`: encoder and decoder hidden width, usually `32` to `128`
- `vae_latent_dim`: probabilistic latent size, usually `8` to `16`
- `vae_beta`: KL regularization strength, usually `0.5` to `2.0`
- `vae_learning_rate`: optimizer step size, usually `1e-3` to `1e-4`
- `vae_batch_size`: minibatch size used during training
- `vae_threshold_percentile`: reconstruction cutoff percentile, usually `95.0` to `99.0`

Common ensemble calibration settings you can pass through `config.json`:

- `calibrate_threshold`: enable or disable label-aware threshold tuning
- `calibration_min_samples`: minimum labeled rows required before threshold tuning runs, usually `25` or higher

## Parallel Ensemble

The anomaly model stage now runs nine detectors in parallel on the same
preprocessed feature matrix, and the evaluation/dashboard comparison views
show each component side by side:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor with `novelty=True`
- Deep autoencoder with a mirrored 128-64-32-8-32-64-128 reconstruction path
- Anomaly Transformer with feature-attention discrepancy scoring
- GANomaly with latent-consistency scoring
- Variational autoencoder with a probabilistic latent bottleneck
- CNN autoencoder with a 1D convolutional encoder over the tabular feature axis
- Deep SVDD with a hypersphere around the learned normal center

Their scores are min-max normalized to `[0, 1]` before fusion.
By default, the Anomaly Transformer, GANomaly, and CNN autoencoder contribute
small nonzero weights to the fused anomaly score alongside the main
reconstruction-based detectors.
The CLI report, Streamlit view, and web dashboard now also surface Local
Outlier Factor in the component comparison tables, and the transformer-based
detectors appear there as well.
You can choose between:

- `weighted_average`: combine normalized scores with explicit weights
- `max_score_voting`: flag a record if any model exceeds the configured threshold
- `stacking`: train a logistic regression meta-classifier on the normalized IF, AE, and SVDD scores using labeled examples

The scored output includes:

- `isolation_forest_anomaly_score`
- `one_class_svm_anomaly_score`
- `local_outlier_factor_anomaly_score`
- `autoencoder_anomaly_score`
- `anomaly_transformer_anomaly_score`
- `variational_autoencoder_anomaly_score`
- `ganomaly_anomaly_score`
- `autoencoder_reconstruction_error`
- `autoencoder_reconstruction_mae`
- `anomaly_transformer_reconstruction_error`
- `anomaly_transformer_attention_discrepancy`
- `variational_autoencoder_reconstruction_error`
- `variational_autoencoder_reconstruction_mae`
- `ganomaly_reconstruction_error`
- `ganomaly_latent_consistency_error`
- `raw_anomaly_score`
- `anomaly_score`
- `risk_level`
- `risk_score`
- `alert_triggered`
- `anomaly_flag`
- `is_anomaly`

Risk bands are mapped from the final `anomaly_score` as:

- `Low`: `0.0` to `0.4` (exclusive of `0.4`)
- `Medium`: `0.4` to `0.7` (exclusive of `0.7`)
- `High`: `0.7` to `1.0`

When labels are available, the evaluation helpers can compare scored outputs
with precision, recall, F1-score, AUC-ROC, and AUPRC.

When the scored output came from the built-in `predict` command, the report
also includes runtime comparison metrics for rural edge deployment:

- inference latency per patient
- batch latency
- training time
- model size
- estimated RAM usage
- an edge-readiness verdict that checks latency, model size, and RAM against simple deployment thresholds

Isolation Forest is still the default tree-based detector, and its key
hyperparameters are exposed through the preprocessing config:

- `isolation_forest_n_estimators`
- `isolation_forest_contamination`
- `isolation_forest_max_samples`
- `isolation_forest_max_features`
- `isolation_forest_bootstrap`
- `autoencoder_latent_dim`
- `autoencoder_dropout`
- `autoencoder_learning_rate`
- `autoencoder_batch_size`
- `autoencoder_threshold_percentile`
- `deep_svdd_nu`
- `deep_svdd_center_fixed`
- `deep_svdd_architecture`
- `deep_svdd_pretrain_autoencoder`

The autoencoder threshold is set from the 95th-99th percentile of validation
reconstruction errors, which keeps the cutoff tied to normal-profile
reconstruction quality rather than the training batch itself. See
[docs/config_examples.md](docs/config_examples.md) for a compact JSON tuning
example, including
[docs/risk_scoring_config.example.json](docs/risk_scoring_config.example.json).

You can also set ensemble fusion weights in `config.json`. For example:

```json
{
  "ensemble_fusion_weights": {
  "isolation_forest": 0.3,
  "autoencoder": 0.4,
  "anomaly_transformer": 0.1,
  "variational_autoencoder": 0.1,
  "ganomaly": 0.1,
  "cnn_autoencoder": 0.1,
  "deep_svdd": 0.3
  }
}
```

You can also tune the CNN autoencoder directly from the CLI with `--cnn-autoencoder-weight 0.2`.
You can tune the Anomaly Transformer directly with `--anomaly-transformer-weight 0.1`.
You can tune GANomaly directly with `--ganomaly-weight 0.1`.
You can tune the variational autoencoder directly with `--vae-weight 0.1`.

If you omit a detector from the map, its weight defaults to `0.0` and the
remaining weights are normalized before fusion.

For a simpler config-only override, you can set the top-level key directly:

```json
{
  "cnn_autoencoder_weight": 0.2,
  "anomaly_transformer_weight": 0.1,
  "ganomaly_weight": 0.1,
  "vae_weight": 0.1
}
```

## Examples

- [docs/config_examples.md](docs/config_examples.md)
- [docs/risk_scoring_config.example.json](docs/risk_scoring_config.example.json)

Start from `docs/risk_scoring_config.example.json` when you want to tune the blended risk score.

For stacking, pass a labeled target array to training and set:

```json
{
  "ensemble_fusion_strategy": "stacking"
}
```

The meta-classifier is a lightweight logistic regression trained on the
min-max scaled Isolation Forest, autoencoder, and Deep SVDD scores.

If your labels are in the same training file, pass `--label-column label` on
the train command and the pipeline will use that column as the stacking target.
If the labels are in a separate file, pass `--labels-file labels.csv` and, if
needed, `--labels-column label`.

Example:

```bash
anomaly-cli train --input train.csv --output artifacts/model.joblib \
  --ensemble-fusion-strategy stacking \
  --labels-file labels.csv \
  --labels-column label
```

## Feature Provenance

Export the feature map from a fitted preprocessor to see how raw schema fields
become model inputs. For a fuller example, see [docs/feature_provenance.md](docs/feature_provenance.md).

```python
feature_map = pipeline.named_steps["preprocessor"].export_feature_map()
feature_map.to_csv("artifacts/feature_map.csv", index=False)
```

The exported map includes:

- `source_columns`: raw schema columns used to build the feature
- `transformation_path`: the transformation steps applied
- `provenance_depth`: number of stages in the path

Common transformation rules:

- `raw`
- `multi_value_expand`
- `one_hot_encode`
- `time_series_engineer`
- `pca`

### Example Feature Map

| final_feature | source_columns | transformation_path | provenance_depth |
| --- | --- | --- | --- |
| `comorbidities__diabetes` | `["comorbidities"]` | `["raw", "multi_value_expand"]` | `2` |
| `heart_rate_bpm_mean_7d` | `["patient_id", "recorded_at", "heart_rate_bpm"]` | `["raw", "time_series_engineer"]` | `2` |
| `gender_female` | `["gender"]` | `["raw", "one_hot_encode"]` | `2` |
| `pca_1` | `[...]` | `["raw", "scaling", "pca"]` | `3` |
