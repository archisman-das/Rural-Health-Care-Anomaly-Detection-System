# Model Details Documentation

This document explains how the anomaly-detection models work in this project, how each algorithm scores records, and how the ensemble turns those scores into a single clinical output.

## What This Document Covers

- Shared preprocessing and feature engineering
- Classical anomaly detectors
- Neural reconstruction-based detectors
- Sequence-style and attention-based detectors
- Deep SVDD hypersphere scoring
- Ensemble fusion and threshold calibration
- Final scoring, risk mapping, and evaluation outputs

## Shared Input Pipeline

All models consume the transformed output of the preprocessing pipeline in [`rural_health_anomaly/preprocessing.py`](../rural_health_anomaly/preprocessing.py).

The preprocessing stage does four important jobs:

1. Normalizes raw CSV or Parquet data.
2. Expands multi-value fields and list-like numeric fields.
3. Builds leakage-safe longitudinal features such as rolling means, lag values, and interactions.
4. Applies numeric scaling, categorical encoding, and optional PCA.

Feature provenance is tracked so transformed columns can be traced back to their raw source fields. This is useful when the dashboard or reports need to explain where a score came from.

## Score Meaning

Across the project, a larger anomaly score means the record looks more unusual.

- Classical detectors expose a model decision score or decision margin.
- Reconstruction models use reconstruction error.
- Deep SVDD uses distance from a learned center.
- The ensemble combines all component scores into one fused anomaly score.

The final pipeline then maps that anomaly score into:

- `anomaly_score`
- `risk_score`
- `risk_category`
- `alert_triggered`

## 1. Isolation Forest

Implementation: [`rural_health_anomaly/detectors.py`](../rural_health_anomaly/detectors.py)

### Working

Isolation Forest builds many random trees and isolates points by random splits. Normal records usually take more splits to isolate because they sit in dense regions. Outliers are isolated faster.

In this code:

- The model is created with `sklearn.ensemble.IsolationForest`.
- The raw detector score is taken from the negative decision function.
- Scores are normalized using the training mean and standard deviation.

### Why it helps

This detector is strong for tabular clinical data because it can notice records that sit far away from the common patterns without needing labeled anomalies.

### Parameters that matter

- `n_estimators`
- `contamination`
- `max_samples`
- `max_features`
- `bootstrap`

## 2. One-Class SVM

Implementation: [`rural_health_anomaly/detectors.py`](../rural_health_anomaly/detectors.py)

### Working

One-Class SVM learns a boundary around the normal data distribution. Points inside the learned boundary are treated as regular, while points outside are flagged as unusual.

In this code:

- `nu` controls the expected fraction of anomalies and the margin softness.
- `kernel` is set to RBF by default.
- The raw score comes from the negative decision function, then gets normalized.

### Why it helps

This detector is useful when the normal class has a smooth boundary but the anomalous class is sparse and irregular.

## 3. Local Outlier Factor

Implementation: [`rural_health_anomaly/detectors.py`](../rural_health_anomaly/detectors.py)

### Working

Local Outlier Factor compares the local density around a point to the density around its neighbors. If a point sits in a much lower-density area than its neighbors, it gets a high anomaly signal.

In this code:

- The detector runs in novelty mode so it can score new samples after training.
- `n_neighbors` controls the neighborhood size.
- The raw anomaly score is the negative decision function.

### Why it helps

LOF is good when anomalies are local rather than global, for example when a patient record is not wildly extreme but still inconsistent with nearby records.

## 4. Deep Autoencoder

Implementation: [`rural_health_anomaly/autoencoder.py`](../rural_health_anomaly/autoencoder.py)

### Working

The autoencoder compresses the input through a bottleneck and then reconstructs the original features.

The architecture is symmetric:

- Input layer
- Hidden layers
- Latent bottleneck
- Mirror decoder layers
- Output layer

Training minimizes mean squared reconstruction error. If the model can reconstruct a record well, the record is considered normal. If reconstruction is poor, the record is more suspicious.

### Scoring

- `reconstruction_error(X)` returns per-row mean squared error.
- `score(X)` returns a standardized version of reconstruction error.
- `predict(X)` flags records whose reconstruction error is above the learned threshold.

### Training behavior

- Uses mini-batch Adam updates.
- Uses dropout and L2 regularization.
- Keeps the best validation weights by early stopping.
- Learns a threshold from a validation percentile.

### Why it helps

This model captures nonlinear relationships among lab values, vitals, and derived features.

## 5. Variational Autoencoder

Implementation: [`rural_health_anomaly/variational_autoencoder.py`](../rural_health_anomaly/variational_autoencoder.py)

### Working

The VAE is like an autoencoder, but instead of encoding directly into one latent vector, it learns:

- `mu` for the latent mean
- `logvar` for latent uncertainty

During training, it samples from the latent distribution using the reparameterization trick. The loss combines:

- Reconstruction error
- KL divergence between the learned latent distribution and a standard normal prior

The `beta` parameter controls the strength of the KL term.

### Scoring

The anomaly score is still based on reconstruction error after training.

### Why it helps

The VAE is useful when you want the latent space to stay smooth and well-regularized, which can improve robustness on noisy healthcare data.

## 6. GANomaly

Implementation: [`rural_health_anomaly/ganomaly.py`](../rural_health_anomaly/ganomaly.py)

### Working

This model follows the GANomaly style of detection without building a full adversarial training loop.

It uses:

- An encoder to create a latent code
- A decoder to reconstruct the record
- A re-encoder to map the reconstruction back to latent space

The anomaly score combines:

- Reconstruction error
- Latent consistency error

### Why it helps

If a record reconstructs reasonably well but the latent representation changes a lot after the round trip, the model treats it as suspicious.

## 7. CNN Autoencoder

Implementation: [`rural_health_anomaly/cnn_autoencoder.py`](../rural_health_anomaly/cnn_autoencoder.py)

### Working

This model treats tabular features like a 1D sequence. It applies:

- 1D convolution
- Nonlinearity
- Bottleneck encoding
- Dense decoding

The reconstruction error becomes the anomaly score.

### Why it helps

It can capture local feature patterns that may matter when neighboring fields tend to move together, such as related vitals or lab clusters.

## 8. Anomaly Transformer

Implementation: [`rural_health_anomaly/anomaly_transformer.py`](../rural_health_anomaly/anomaly_transformer.py)

### Working

This detector applies a lightweight attention mechanism over the input features before reconstruction.

The score combines:

- Reconstruction error
- Attention discrepancy

Attention discrepancy measures how far the learned attention pattern is from a uniform reference pattern. If the attention becomes sharply concentrated or unusual, the score increases.

### Why it helps

This model adds a feature-importance style signal on top of reconstruction, which can highlight unusual combinations more clearly than reconstruction alone.

## 9. Deep SVDD

Implementation: [`rural_health_anomaly/deep_svdd.py`](../rural_health_anomaly/deep_svdd.py)

### Working

Deep SVDD learns a compact hypersphere in latent space that contains normal data.

The process is:

1. Map inputs into a latent representation.
2. Compute a center vector from the training latent space.
3. Minimize the distance between each point and the center.
4. Set a radius from the validation distribution.

If a record lies outside the learned hypersphere, it is considered anomalous.

### Optional pretraining

For the MLP version, the model can preload encoder weights from a deep autoencoder. This gives it a better starting point.

### Why it helps

Deep SVDD is useful when the goal is to learn a compact normal region rather than reconstruct the input.

## Ensemble Fusion

Implementation: [`rural_health_anomaly/ensemble.py`](../rural_health_anomaly/ensemble.py)

### Working

The project does not rely on one detector. It fits nine detectors in parallel and then fuses their outputs.

The component scores are first normalized into a comparable 0 to 1 range. Then the ensemble can combine them in one of three ways:

- `weighted_average`
- `max_score_voting`
- `stacking`
- `moe`

### Weighted average

This is the default mode. Each model gets a configurable weight. The fused score is the weighted sum of all component scores.

### Max score voting

The fused score becomes the maximum component score. This mode is more conservative because a single strong detector can drive the alert.

### Stacking

If labeled data is available, the ensemble can train a nonlinear meta-model on top of the component scores.

In the current implementation, stacking uses a nonlinear meta-model by default:

- A shallow MLP when the project dependencies are used as-is
- XGBoost when it is available and selected through configuration

The meta-model sees the detector score vector as features and learns how to combine them into a better final anomaly probability.

### Threshold calibration

When labels are available and enough labeled samples exist, the ensemble can search for the threshold that maximizes F1, then precision, then threshold value.

This makes the final boundary more data-driven instead of relying only on a fixed cutoff.

### Mixture of Experts routing

The MoE gate is a lightweight neural network that takes the transformed feature vector as input and outputs a soft assignment over detectors.

- If labeled anomalies are available, the gate learns to route toward detectors that align best with the label.
- If labels are not available, the gate uses detector disagreement as a self-supervised routing signal.

This makes the ensemble behave more like a conditional router than a fixed weighted average.

## Final Scoring Layer

Implementation: [`rural_health_anomaly/training.py`](../rural_health_anomaly/training.py)

After the model produces an anomaly score, the training/scoring layer adds the clinical view:

- `clinical_risk_score` blends anomaly, vital-sign, lab, and access components.
- `risk_score` is the same score expressed on a 0 to 100 scale.
- `risk_category` maps the score into Normal, Moderate, High, or Critical.
- `alert_triggered` becomes true for High and Critical cases.

This means the output is not just a raw model score. It is turned into something that is easier to use in a workflow.

## Training, Validation, and Splits

The repository supports train, validation, and test splits in separate folders. The helper functions in `training.py` can:

- Load a split directory
- Train on the train set
- Calibrate on the validation set
- Evaluate on the test set

This keeps training, threshold tuning, and evaluation separate.

## Evaluation Outputs

Implementation: [`rural_health_anomaly/evaluation.py`](../rural_health_anomaly/evaluation.py)

The evaluation layer can summarize:

- Precision
- Recall
- F1
- ROC-AUC
- Average precision
- Score distributions
- Agreement between models
- Runtime and deployment cost

It also highlights the rows where detectors disagree most, which is useful for manual review.

## How To Read The Models Together

The models are designed to complement each other:

- Classical detectors catch simple distribution shifts.
- Autoencoders catch nonlinear reconstruction failures.
- VAE adds latent regularization.
- GANomaly adds latent consistency.
- CNN autoencoder captures local feature structure.
- Anomaly Transformer adds attention-based discrepancy.
- Deep SVDD learns a compact normal hypersphere.

The ensemble uses all of these perspectives so the final signal is more stable than any single detector alone.

## Practical Summary

If you want the shortest description:

- Isolation Forest isolates rare points quickly.
- One-Class SVM draws a boundary around normal cases.
- LOF compares local density.
- Autoencoder checks reconstruction quality.
- VAE reconstructs while regularizing latent space.
- GANomaly checks reconstruction plus latent consistency.
- CNN Autoencoder uses 1D convolution before reconstruction.
- Anomaly Transformer combines attention and reconstruction.
- Deep SVDD measures distance from a learned normal center.
- The ensemble fuses all scores and calibrates the final threshold.
- The MoE gate learns which detector to trust more for each record.
