"""Parallel anomaly ensemble estimators."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.parallel import Parallel, delayed

from .autoencoder import DeepAutoencoder
from .anomaly_transformer import AnomalyTransformer
from .cnn_autoencoder import CNNAutoencoder
from .deep_svdd import DeepSVDD
from .detectors import (
    IsolationForestAnomalyModel,
    LocalOutlierFactorAnomalyModel,
    OneClassSVMAnomalyModel,
)
from .ganomaly import GANomaly
from .variational_autoencoder import VariationalAutoencoder


def _ensure_2d_array(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array-like input.")
    return array


def _fit_estimator(estimator, X: np.ndarray):
    return clone(estimator).fit(X)


def _normalized_anomaly_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "score"):
        return np.asarray(estimator.score(X), dtype=float)
    if hasattr(estimator, "reconstruction_error"):
        raw = np.asarray(estimator.reconstruction_error(X), dtype=float)
        if hasattr(estimator, "_training_raw_score_mean_") and hasattr(estimator, "_training_raw_score_std_"):
            return (raw - estimator._training_raw_score_mean_) / estimator._training_raw_score_std_
        return raw
    if hasattr(estimator, "decision_function"):
        raw = -np.asarray(estimator.decision_function(X), dtype=float)
        return raw
    if hasattr(estimator, "score_samples"):
        raw = -np.asarray(estimator.score_samples(X), dtype=float)
        return raw
    raise AttributeError(
        f"{type(estimator).__name__} does not expose score, reconstruction_error, decision_function, or score_samples."
    )


def _minmax_scale(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    scale = maximum - minimum
    if scale == 0.0 or np.isnan(scale):
        scale = 1.0
    scaled = (scores - minimum) / scale
    return np.clip(scaled, 0.0, 1.0), minimum, maximum


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums = np.where(~np.isfinite(row_sums) | (row_sums <= 0.0), 1.0, row_sums)
    return matrix / row_sums


def _clone_meta_model(model: Any) -> Any:
    if hasattr(model, "get_params"):
        return clone(model)
    return model


def _coerce_binary_labels(y: Any) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.size == 0:
        raise ValueError("Stacking requires at least one labeled sample.")
    if labels.dtype == bool:
        return labels.astype(int)
    unique_values = set(np.unique(labels).tolist())
    if unique_values.issubset({0, 1}):
        return labels.astype(int)
    if unique_values.issubset({-1, 1}):
        return np.where(labels == -1, 1, 0).astype(int)
    raise ValueError("Stacking labels must be binary, using 1/-1 or 0/1 conventions.")


def _binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return precision, recall, and F1 for binary predictions."""

    true_positive = float(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = float(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _calibrate_threshold_from_scores(scores: np.ndarray, y_true: np.ndarray, *, candidate_count: int = 201) -> tuple[float, dict[str, float]]:
    """Find the score cutoff that maximizes F1, then precision, then threshold."""

    if scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must have the same length.")

    if scores.size == 0:
        raise ValueError("scores must not be empty.")

    thresholds = np.linspace(0.0, 1.0, max(3, int(candidate_count)), dtype=float)
    best_threshold = float(thresholds[0])
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    best_key = (-1.0, -1.0, -1.0)

    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        metrics = _binary_classification_metrics(y_true, predicted)
        key = (metrics["f1"], metrics["precision"], float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


class _MoEGatingNetwork(BaseEstimator):
    """Lightweight neural gate that routes each sample to detector experts."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 80,
        patience: int = 10,
        l2: float = 1e-5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_parameters(self, n_features: int, n_experts: int) -> None:
        rng = np.random.default_rng(self.random_state)
        hidden_scale = np.sqrt(2.0 / max(n_features, 1))
        output_scale = np.sqrt(2.0 / max(self.hidden_dim, 1))

        self.weights_input_ = rng.normal(0.0, hidden_scale, size=(n_features, self.hidden_dim))
        self.bias_input_ = np.zeros(self.hidden_dim, dtype=float)
        self.weights_output_ = rng.normal(0.0, output_scale, size=(self.hidden_dim, n_experts))
        self.bias_output_ = np.zeros(n_experts, dtype=float)

        self._adam_state_ = {
            "step": 0,
            "m_input": np.zeros_like(self.weights_input_),
            "v_input": np.zeros_like(self.weights_input_),
            "m_bias_input": np.zeros_like(self.bias_input_),
            "v_bias_input": np.zeros_like(self.bias_input_),
            "m_output": np.zeros_like(self.weights_output_),
            "v_output": np.zeros_like(self.weights_output_),
            "m_bias_output": np.zeros_like(self.bias_output_),
            "v_bias_output": np.zeros_like(self.bias_output_),
        }

    def _forward(
        self,
        X: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        hidden_linear = X @ self.weights_input_ + self.bias_input_
        hidden = np.maximum(hidden_linear, 0.0)
        dropout_mask = None
        if training and self.dropout > 0.0:
            keep_prob = 1.0 - float(self.dropout)
            if keep_prob <= 0.0:
                raise ValueError("dropout must be less than 1.0")
            generator = rng or np.random.default_rng(self.random_state)
            dropout_mask = (generator.random(hidden.shape) < keep_prob).astype(float) / keep_prob
            hidden = hidden * dropout_mask
        logits = hidden @ self.weights_output_ + self.bias_output_
        probs = _softmax(logits)
        return hidden_linear, hidden, dropout_mask, probs

    def _loss(self, targets: np.ndarray, probs: np.ndarray) -> float:
        clipped = np.clip(probs, 1e-12, 1.0)
        return float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))

    def _apply_adam(self, grad_input: np.ndarray, grad_bias_input: np.ndarray, grad_output: np.ndarray, grad_bias_output: np.ndarray) -> None:
        state = self._adam_state_
        state["step"] += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = state["step"]

        def update(param: np.ndarray, grad: np.ndarray, m_key: str, v_key: str) -> np.ndarray:
            state[m_key] = beta1 * state[m_key] + (1.0 - beta1) * grad
            state[v_key] = beta2 * state[v_key] + (1.0 - beta2) * (grad**2)
            m_hat = state[m_key] / (1.0 - beta1**step)
            v_hat = state[v_key] / (1.0 - beta2**step)
            return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.weights_input_ = update(self.weights_input_, grad_input, "m_input", "v_input")
        self.bias_input_ = update(self.bias_input_, grad_bias_input, "m_bias_input", "v_bias_input")
        self.weights_output_ = update(self.weights_output_, grad_output, "m_output", "v_output")
        self.bias_output_ = update(self.bias_output_, grad_bias_output, "m_bias_output", "v_bias_output")

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 2:
            raise ValueError("MoE gating targets must be a 2D array.")
        if targets.shape[0] != X.shape[0]:
            raise ValueError("MoE gating targets must align with X.")
        if targets.shape[1] < 2:
            raise ValueError("MoE gating requires at least two experts.")

        targets = np.clip(targets, 0.0, None)
        targets = _normalize_rows(targets)
        self.n_features_in_ = X.shape[1]
        self.n_experts_ = targets.shape[1]
        self._initialize_parameters(self.n_features_in_, self.n_experts_)

        if 0.0 < self.batch_size:
            batch_size = max(1, min(int(self.batch_size), X.shape[0]))
        else:
            batch_size = max(1, X.shape[0])

        if X.shape[0] >= 5:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                targets,
                test_size=min(max(0.2, 1.0 / max(X.shape[0], 5)), 0.5),
                random_state=self.random_state,
                shuffle=True,
            )
        else:
            X_train, X_val, y_train, y_val = X, X, targets, targets

        rng = np.random.default_rng(self.random_state)
        best_state: dict[str, np.ndarray] | None = None
        best_val_loss = float("inf")
        stalled_epochs = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(int(self.max_epochs)):
            indices = rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                hidden_linear, hidden, dropout_mask, probs = self._forward(batch_X, training=True, rng=rng)

                grad_logits = (probs - batch_y) / float(max(batch_X.shape[0], 1))
                grad_output = hidden.T @ grad_logits + self.l2 * self.weights_output_
                grad_bias_output = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ self.weights_output_.T
                if dropout_mask is not None:
                    grad_hidden = grad_hidden * dropout_mask
                grad_hidden = grad_hidden * (hidden_linear > 0.0).astype(float)
                grad_input = batch_X.T @ grad_hidden + self.l2 * self.weights_input_
                grad_bias_input = grad_hidden.sum(axis=0)

                self._apply_adam(grad_input, grad_bias_input, grad_output, grad_bias_output)

            _, _, _, train_probs = self._forward(X_train, training=False)
            _, _, _, val_probs = self._forward(X_val, training=False)
            train_loss = self._loss(y_train, train_probs)
            val_loss = self._loss(y_val, val_probs)
            self.history_.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if self.verbose:
                print(f"[MoE Gate] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    "weights_input": self.weights_input_.copy(),
                    "bias_input": self.bias_input_.copy(),
                    "weights_output": self.weights_output_.copy(),
                    "bias_output": self.bias_output_.copy(),
                }
                stalled_epochs = 0
            else:
                stalled_epochs += 1
                if stalled_epochs >= int(self.patience):
                    break

        if best_state is not None:
            self.weights_input_ = best_state["weights_input"].copy()
            self.bias_input_ = best_state["bias_input"].copy()
            self.weights_output_ = best_state["weights_output"].copy()
            self.bias_output_ = best_state["bias_output"].copy()

        _, _, _, train_probs = self._forward(X, training=False)
        self.training_routing_entropy_ = float(-np.mean(np.sum(train_probs * np.log(np.clip(train_probs, 1e-12, 1.0)), axis=1)))
        self.fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "weights_input_"):
            raise RuntimeError("MoE gate must be fit before scoring.")
        X = _ensure_2d_array(X)
        _, _, _, probs = self._forward(X, training=False)
        return probs

    def gate_weights(self, X) -> np.ndarray:
        return self.predict_proba(X)


class ParallelAnomalyEnsemble(BaseEstimator, OutlierMixin):
    """Fit nine anomaly detectors in parallel and fuse their scores."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_jobs: int = -1,
        fusion_strategy: str = "weighted_average",
        max_score_threshold: float = 0.8,
        calibrate_threshold: bool = True,
        calibration_min_samples: int = 25,
        fusion_weights: dict[str, float] | None = None,
        stacking_meta_model_type: str = "mlp",
        stacking_hidden_layer_sizes: tuple[int, ...] = (32, 16),
        stacking_alpha: float = 1e-4,
        stacking_learning_rate_init: float = 1e-3,
        stacking_max_iter: int = 500,
        stacking_random_state: int = 42,
        stacking_verbose: bool = False,
        moe_gate_hidden_dim: int = 32,
        moe_gate_dropout: float = 0.1,
        moe_gate_learning_rate: float = 1e-3,
        moe_gate_batch_size: int = 32,
        moe_gate_max_epochs: int = 80,
        moe_gate_patience: int = 10,
        moe_gate_l2: float = 1e-5,
        moe_gate_random_state: int = 42,
        moe_gate_verbose: bool = False,
        isolation_forest_n_estimators: int = 300,
        isolation_forest_max_samples: int | str = "auto",
        isolation_forest_max_features: float = 1.0,
        isolation_forest_bootstrap: bool = False,
        isolation_forest_random_state: int = 42,
        isolation_forest_n_jobs: int = -1,
        one_class_svm_nu: float | None = None,
        one_class_svm_kernel: str = "rbf",
        one_class_svm_gamma: str | float = "scale",
        local_outlier_factor_n_neighbors: int = 20,
        local_outlier_factor_contamination: float | None = None,
        local_outlier_factor_n_jobs: int = -1,
        autoencoder_latent_dim: int = 8,
        autoencoder_dropout: float = 0.2,
        autoencoder_learning_rate: float = 1e-3,
        autoencoder_batch_size: int = 32,
        autoencoder_threshold_percentile: float = 97.5,
        autoencoder_validation_fraction: float = 0.2,
        autoencoder_max_epochs: int = 80,
        autoencoder_patience: int = 10,
        autoencoder_l2: float = 1e-5,
        autoencoder_random_state: int = 42,
        autoencoder_verbose: bool = False,
        ganomaly_hidden_dim: int = 64,
        ganomaly_latent_dim: int = 8,
        ganomaly_dropout: float = 0.2,
        ganomaly_learning_rate: float = 1e-3,
        ganomaly_batch_size: int = 32,
        ganomaly_consistency_weight: float = 1.0,
        ganomaly_threshold_percentile: float = 97.5,
        ganomaly_validation_fraction: float = 0.2,
        ganomaly_max_epochs: int = 80,
        ganomaly_patience: int = 10,
        ganomaly_l2: float = 1e-5,
        ganomaly_random_state: int = 42,
        ganomaly_verbose: bool = False,
        anomaly_transformer_hidden_dim: int = 64,
        anomaly_transformer_latent_dim: int = 8,
        anomaly_transformer_dropout: float = 0.2,
        anomaly_transformer_learning_rate: float = 1e-3,
        anomaly_transformer_batch_size: int = 32,
        anomaly_transformer_attention_weight: float = 0.5,
        anomaly_transformer_attention_temperature: float = 1.0,
        anomaly_transformer_threshold_percentile: float = 97.5,
        anomaly_transformer_validation_fraction: float = 0.2,
        anomaly_transformer_max_epochs: int = 80,
        anomaly_transformer_patience: int = 10,
        anomaly_transformer_l2: float = 1e-5,
        anomaly_transformer_random_state: int = 42,
        anomaly_transformer_verbose: bool = False,
        vae_hidden_dim: int = 64,
        vae_latent_dim: int = 8,
        vae_dropout: float = 0.2,
        vae_learning_rate: float = 1e-3,
        vae_batch_size: int = 32,
        vae_beta: float = 1.0,
        vae_threshold_percentile: float = 97.5,
        vae_validation_fraction: float = 0.2,
        vae_max_epochs: int = 80,
        vae_patience: int = 10,
        vae_l2: float = 1e-5,
        vae_random_state: int = 42,
        vae_verbose: bool = False,
        cnn_autoencoder_filters: int = 8,
        cnn_autoencoder_kernel_size: int = 3,
        cnn_autoencoder_latent_dim: int = 8,
        cnn_autoencoder_dropout: float = 0.2,
        cnn_autoencoder_learning_rate: float = 1e-3,
        cnn_autoencoder_batch_size: int = 32,
        cnn_autoencoder_threshold_percentile: float = 97.5,
        cnn_autoencoder_validation_fraction: float = 0.2,
        cnn_autoencoder_max_epochs: int = 80,
        cnn_autoencoder_patience: int = 10,
        cnn_autoencoder_l2: float = 1e-5,
        cnn_autoencoder_random_state: int = 42,
        cnn_autoencoder_verbose: bool = False,
        deep_svdd_nu: float = 0.05,
        deep_svdd_center_fixed: bool = True,
        deep_svdd_architecture: str = "mlp",
        deep_svdd_latent_dim: int = 8,
        deep_svdd_learning_rate: float = 1e-3,
        deep_svdd_batch_size: int = 32,
        deep_svdd_max_epochs: int = 60,
        deep_svdd_validation_fraction: float = 0.2,
        deep_svdd_pretrain_autoencoder: bool = True,
        deep_svdd_pretrain_epochs: int = 25,
        deep_svdd_pretrain_dropout: float = 0.2,
        deep_svdd_pretrain_learning_rate: float = 1e-3,
        deep_svdd_pretrain_batch_size: int = 32,
        deep_svdd_random_state: int = 42,
        deep_svdd_verbose: bool = False,
    ):
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.fusion_strategy = fusion_strategy
        self.max_score_threshold = max_score_threshold
        self.calibrate_threshold = calibrate_threshold
        self.calibration_min_samples = calibration_min_samples
        self.fusion_weights = fusion_weights
        self.stacking_meta_model_type = stacking_meta_model_type
        self.stacking_hidden_layer_sizes = stacking_hidden_layer_sizes
        self.stacking_alpha = stacking_alpha
        self.stacking_learning_rate_init = stacking_learning_rate_init
        self.stacking_max_iter = stacking_max_iter
        self.stacking_random_state = stacking_random_state
        self.stacking_verbose = stacking_verbose
        self.moe_gate_hidden_dim = moe_gate_hidden_dim
        self.moe_gate_dropout = moe_gate_dropout
        self.moe_gate_learning_rate = moe_gate_learning_rate
        self.moe_gate_batch_size = moe_gate_batch_size
        self.moe_gate_max_epochs = moe_gate_max_epochs
        self.moe_gate_patience = moe_gate_patience
        self.moe_gate_l2 = moe_gate_l2
        self.moe_gate_random_state = moe_gate_random_state
        self.moe_gate_verbose = moe_gate_verbose
        self.isolation_forest_n_estimators = isolation_forest_n_estimators
        self.isolation_forest_max_samples = isolation_forest_max_samples
        self.isolation_forest_max_features = isolation_forest_max_features
        self.isolation_forest_bootstrap = isolation_forest_bootstrap
        self.isolation_forest_random_state = isolation_forest_random_state
        self.isolation_forest_n_jobs = isolation_forest_n_jobs
        self.one_class_svm_nu = one_class_svm_nu
        self.one_class_svm_kernel = one_class_svm_kernel
        self.one_class_svm_gamma = one_class_svm_gamma
        self.local_outlier_factor_n_neighbors = local_outlier_factor_n_neighbors
        self.local_outlier_factor_contamination = local_outlier_factor_contamination
        self.local_outlier_factor_n_jobs = local_outlier_factor_n_jobs
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.autoencoder_dropout = autoencoder_dropout
        self.autoencoder_learning_rate = autoencoder_learning_rate
        self.autoencoder_batch_size = autoencoder_batch_size
        self.autoencoder_threshold_percentile = autoencoder_threshold_percentile
        self.autoencoder_validation_fraction = autoencoder_validation_fraction
        self.autoencoder_max_epochs = autoencoder_max_epochs
        self.autoencoder_patience = autoencoder_patience
        self.autoencoder_l2 = autoencoder_l2
        self.autoencoder_random_state = autoencoder_random_state
        self.autoencoder_verbose = autoencoder_verbose
        self.ganomaly_hidden_dim = ganomaly_hidden_dim
        self.ganomaly_latent_dim = ganomaly_latent_dim
        self.ganomaly_dropout = ganomaly_dropout
        self.ganomaly_learning_rate = ganomaly_learning_rate
        self.ganomaly_batch_size = ganomaly_batch_size
        self.ganomaly_consistency_weight = ganomaly_consistency_weight
        self.ganomaly_threshold_percentile = ganomaly_threshold_percentile
        self.ganomaly_validation_fraction = ganomaly_validation_fraction
        self.ganomaly_max_epochs = ganomaly_max_epochs
        self.ganomaly_patience = ganomaly_patience
        self.ganomaly_l2 = ganomaly_l2
        self.ganomaly_random_state = ganomaly_random_state
        self.ganomaly_verbose = ganomaly_verbose
        self.anomaly_transformer_hidden_dim = anomaly_transformer_hidden_dim
        self.anomaly_transformer_latent_dim = anomaly_transformer_latent_dim
        self.anomaly_transformer_dropout = anomaly_transformer_dropout
        self.anomaly_transformer_learning_rate = anomaly_transformer_learning_rate
        self.anomaly_transformer_batch_size = anomaly_transformer_batch_size
        self.anomaly_transformer_attention_weight = anomaly_transformer_attention_weight
        self.anomaly_transformer_attention_temperature = anomaly_transformer_attention_temperature
        self.anomaly_transformer_threshold_percentile = anomaly_transformer_threshold_percentile
        self.anomaly_transformer_validation_fraction = anomaly_transformer_validation_fraction
        self.anomaly_transformer_max_epochs = anomaly_transformer_max_epochs
        self.anomaly_transformer_patience = anomaly_transformer_patience
        self.anomaly_transformer_l2 = anomaly_transformer_l2
        self.anomaly_transformer_random_state = anomaly_transformer_random_state
        self.anomaly_transformer_verbose = anomaly_transformer_verbose
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_dropout = vae_dropout
        self.vae_learning_rate = vae_learning_rate
        self.vae_batch_size = vae_batch_size
        self.vae_beta = vae_beta
        self.vae_threshold_percentile = vae_threshold_percentile
        self.vae_validation_fraction = vae_validation_fraction
        self.vae_max_epochs = vae_max_epochs
        self.vae_patience = vae_patience
        self.vae_l2 = vae_l2
        self.vae_random_state = vae_random_state
        self.vae_verbose = vae_verbose
        self.cnn_autoencoder_filters = cnn_autoencoder_filters
        self.cnn_autoencoder_kernel_size = cnn_autoencoder_kernel_size
        self.cnn_autoencoder_latent_dim = cnn_autoencoder_latent_dim
        self.cnn_autoencoder_dropout = cnn_autoencoder_dropout
        self.cnn_autoencoder_learning_rate = cnn_autoencoder_learning_rate
        self.cnn_autoencoder_batch_size = cnn_autoencoder_batch_size
        self.cnn_autoencoder_threshold_percentile = cnn_autoencoder_threshold_percentile
        self.cnn_autoencoder_validation_fraction = cnn_autoencoder_validation_fraction
        self.cnn_autoencoder_max_epochs = cnn_autoencoder_max_epochs
        self.cnn_autoencoder_patience = cnn_autoencoder_patience
        self.cnn_autoencoder_l2 = cnn_autoencoder_l2
        self.cnn_autoencoder_random_state = cnn_autoencoder_random_state
        self.cnn_autoencoder_verbose = cnn_autoencoder_verbose
        self.deep_svdd_nu = deep_svdd_nu
        self.deep_svdd_center_fixed = deep_svdd_center_fixed
        self.deep_svdd_architecture = deep_svdd_architecture
        self.deep_svdd_latent_dim = deep_svdd_latent_dim
        self.deep_svdd_learning_rate = deep_svdd_learning_rate
        self.deep_svdd_batch_size = deep_svdd_batch_size
        self.deep_svdd_max_epochs = deep_svdd_max_epochs
        self.deep_svdd_validation_fraction = deep_svdd_validation_fraction
        self.deep_svdd_pretrain_autoencoder = deep_svdd_pretrain_autoencoder
        self.deep_svdd_pretrain_epochs = deep_svdd_pretrain_epochs
        self.deep_svdd_pretrain_dropout = deep_svdd_pretrain_dropout
        self.deep_svdd_pretrain_learning_rate = deep_svdd_pretrain_learning_rate
        self.deep_svdd_pretrain_batch_size = deep_svdd_pretrain_batch_size
        self.deep_svdd_random_state = deep_svdd_random_state
        self.deep_svdd_verbose = deep_svdd_verbose

    def _build_estimators(self):
        nu = self.one_class_svm_nu if self.one_class_svm_nu is not None else self.contamination
        return OrderedDict(
            [
                (
                    "isolation_forest",
                    IsolationForestAnomalyModel(
                        n_estimators=self.isolation_forest_n_estimators,
                        contamination=self.contamination,
                        max_samples=self.isolation_forest_max_samples,
                        max_features=self.isolation_forest_max_features,
                        bootstrap=self.isolation_forest_bootstrap,
                        random_state=self.isolation_forest_random_state,
                        n_jobs=self.isolation_forest_n_jobs,
                    ),
                ),
                (
                    "one_class_svm",
                    OneClassSVMAnomalyModel(
                        nu=nu,
                        kernel=self.one_class_svm_kernel,
                        gamma=self.one_class_svm_gamma,
                    ),
                ),
                (
                    "local_outlier_factor",
                    LocalOutlierFactorAnomalyModel(
                        n_neighbors=self.local_outlier_factor_n_neighbors,
                        contamination=(
                            self.local_outlier_factor_contamination
                            if self.local_outlier_factor_contamination is not None
                            else self.contamination
                        ),
                        n_jobs=self.local_outlier_factor_n_jobs,
                    ),
                ),
                (
                    "autoencoder",
                    DeepAutoencoder(
                        latent_dim=self.autoencoder_latent_dim,
                        dropout=self.autoencoder_dropout,
                        learning_rate=self.autoencoder_learning_rate,
                        batch_size=self.autoencoder_batch_size,
                        threshold_percentile=self.autoencoder_threshold_percentile,
                        validation_fraction=self.autoencoder_validation_fraction,
                        max_epochs=self.autoencoder_max_epochs,
                        patience=self.autoencoder_patience,
                        l2=self.autoencoder_l2,
                        random_state=self.autoencoder_random_state,
                        verbose=self.autoencoder_verbose,
                    ),
                ),
                (
                    "variational_autoencoder",
                    VariationalAutoencoder(
                        hidden_dim=self.vae_hidden_dim,
                        latent_dim=self.vae_latent_dim,
                        dropout=self.vae_dropout,
                        learning_rate=self.vae_learning_rate,
                        batch_size=self.vae_batch_size,
                        beta=self.vae_beta,
                        threshold_percentile=self.vae_threshold_percentile,
                        validation_fraction=self.vae_validation_fraction,
                        max_epochs=self.vae_max_epochs,
                        patience=self.vae_patience,
                        l2=self.vae_l2,
                        random_state=self.vae_random_state,
                        verbose=self.vae_verbose,
                    ),
                ),
                (
                    "ganomaly",
                    GANomaly(
                        hidden_dim=self.ganomaly_hidden_dim,
                        latent_dim=self.ganomaly_latent_dim,
                        dropout=self.ganomaly_dropout,
                        learning_rate=self.ganomaly_learning_rate,
                        batch_size=self.ganomaly_batch_size,
                        consistency_weight=self.ganomaly_consistency_weight,
                        threshold_percentile=self.ganomaly_threshold_percentile,
                        validation_fraction=self.ganomaly_validation_fraction,
                        max_epochs=self.ganomaly_max_epochs,
                        patience=self.ganomaly_patience,
                        l2=self.ganomaly_l2,
                        random_state=self.ganomaly_random_state,
                        verbose=self.ganomaly_verbose,
                    ),
                ),
                (
                    "anomaly_transformer",
                    AnomalyTransformer(
                        hidden_dim=self.anomaly_transformer_hidden_dim,
                        latent_dim=self.anomaly_transformer_latent_dim,
                        dropout=self.anomaly_transformer_dropout,
                        learning_rate=self.anomaly_transformer_learning_rate,
                        batch_size=self.anomaly_transformer_batch_size,
                        attention_weight=self.anomaly_transformer_attention_weight,
                        attention_temperature=self.anomaly_transformer_attention_temperature,
                        threshold_percentile=self.anomaly_transformer_threshold_percentile,
                        validation_fraction=self.anomaly_transformer_validation_fraction,
                        max_epochs=self.anomaly_transformer_max_epochs,
                        patience=self.anomaly_transformer_patience,
                        l2=self.anomaly_transformer_l2,
                        random_state=self.anomaly_transformer_random_state,
                        verbose=self.anomaly_transformer_verbose,
                    ),
                ),
                (
                    "cnn_autoencoder",
                    CNNAutoencoder(
                        filters=self.cnn_autoencoder_filters,
                        kernel_size=self.cnn_autoencoder_kernel_size,
                        latent_dim=self.cnn_autoencoder_latent_dim,
                        dropout=self.cnn_autoencoder_dropout,
                        learning_rate=self.cnn_autoencoder_learning_rate,
                        batch_size=self.cnn_autoencoder_batch_size,
                        threshold_percentile=self.cnn_autoencoder_threshold_percentile,
                        validation_fraction=self.cnn_autoencoder_validation_fraction,
                        max_epochs=self.cnn_autoencoder_max_epochs,
                        patience=self.cnn_autoencoder_patience,
                        l2=self.cnn_autoencoder_l2,
                        random_state=self.cnn_autoencoder_random_state,
                        verbose=self.cnn_autoencoder_verbose,
                    ),
                ),
                (
                    "deep_svdd",
                    DeepSVDD(
                        nu=self.deep_svdd_nu,
                        center_fixed=self.deep_svdd_center_fixed,
                        architecture=self.deep_svdd_architecture,
                        latent_dim=self.deep_svdd_latent_dim,
                        learning_rate=self.deep_svdd_learning_rate,
                        batch_size=self.deep_svdd_batch_size,
                        max_epochs=self.deep_svdd_max_epochs,
                        validation_fraction=self.deep_svdd_validation_fraction,
                        pretrain_autoencoder=self.deep_svdd_pretrain_autoencoder,
                        pretrain_epochs=self.deep_svdd_pretrain_epochs,
                        pretrain_dropout=self.deep_svdd_pretrain_dropout,
                        pretrain_learning_rate=self.deep_svdd_pretrain_learning_rate,
                        pretrain_batch_size=self.deep_svdd_pretrain_batch_size,
                        random_state=self.deep_svdd_random_state,
                        verbose=self.deep_svdd_verbose,
                    ),
                ),
            ]
        )

    def _fit_single(self, name: str, estimator, X: np.ndarray):
        fitted = _fit_estimator(estimator, X)
        return name, fitted

    def _build_stacking_meta_model(self, *, n_samples: int | None = None):
        meta_model_type = str(self.stacking_meta_model_type or "mlp").lower()
        if meta_model_type not in {"mlp", "xgboost", "auto"}:
            raise ValueError("stacking_meta_model_type must be 'mlp', 'xgboost', or 'auto'")

        if meta_model_type in {"xgboost", "auto"}:
            try:
                import xgboost as xgb  # type: ignore

                return xgb.XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=self.stacking_random_state,
                    n_jobs=self.n_jobs,
                    eval_metric="logloss",
                    tree_method="hist",
                )
            except Exception:
                if meta_model_type == "xgboost":
                    raise RuntimeError(
                        "xgboost is not installed, so stacking_meta_model_type='xgboost' cannot be used."
                    )

        hidden_layers = tuple(int(value) for value in self.stacking_hidden_layer_sizes) or (32, 16)
        use_early_stopping = bool(n_samples is not None and n_samples >= 20)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=float(self.stacking_alpha),
            batch_size="auto",
            learning_rate_init=float(self.stacking_learning_rate_init),
            max_iter=int(self.stacking_max_iter),
            random_state=self.stacking_random_state,
            verbose=bool(self.stacking_verbose),
            early_stopping=use_early_stopping,
            validation_fraction=0.2,
            n_iter_no_change=max(10, int(self.stacking_max_iter // 10)),
        )

    def _build_moe_targets(self, component_matrix: np.ndarray, labels: np.ndarray | None) -> tuple[np.ndarray, str]:
        if labels is not None:
            labels_binary = _coerce_binary_labels(labels)
            if labels_binary.shape[0] != component_matrix.shape[0]:
                raise ValueError("MoE labels must have the same number of rows as X.")
            positive_signal = component_matrix
            negative_signal = 1.0 - component_matrix
            targets = np.where(labels_binary[:, None] == 1, positive_signal, negative_signal)
            signal_mode = "label_alignment"
        else:
            consensus = np.mean(component_matrix, axis=1, keepdims=True)
            targets = np.abs(component_matrix - consensus)
            signal_mode = "disagreement_routing"

        targets = np.asarray(targets, dtype=float)
        zero_rows = np.sum(targets, axis=1, keepdims=True) <= 0.0
        if np.any(zero_rows):
            targets = targets.copy()
            targets[zero_rows[:, 0]] = 1.0
        return _normalize_rows(targets), signal_mode

    def _fit_moe_gate(self, X: np.ndarray, component_matrix: np.ndarray, labels: np.ndarray | None):
        gate_targets, signal_mode = self._build_moe_targets(component_matrix, labels)
        gate = _MoEGatingNetwork(
            hidden_dim=self.moe_gate_hidden_dim,
            dropout=self.moe_gate_dropout,
            learning_rate=self.moe_gate_learning_rate,
            batch_size=self.moe_gate_batch_size,
            max_epochs=self.moe_gate_max_epochs,
            patience=self.moe_gate_patience,
            l2=self.moe_gate_l2,
            random_state=self.moe_gate_random_state,
            verbose=self.moe_gate_verbose,
        )
        gate.fit(X, gate_targets)
        gate.training_signal_ = signal_mode
        gate.component_names_ = list(self.component_names_)
        return gate, signal_mode

    def _moe_gate_weights(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "moe_gate_"):
            raise RuntimeError("MoE gate must be fit before scoring.")
        weights = np.asarray(self.moe_gate_.predict_proba(X), dtype=float)
        if weights.ndim != 2 or weights.shape[1] != len(self.component_names_):
            raise RuntimeError("MoE gate produced incompatible routing weights.")
        return weights

    def gate_weights(self, X) -> pd.DataFrame:
        if not hasattr(self, "moe_gate_"):
            raise RuntimeError("MoE gate is only available when fusion_strategy='moe'.")
        matrix = self._moe_gate_weights(_ensure_2d_array(X))
        return pd.DataFrame(matrix, columns=[f"{name}_gate_weight" for name in self.component_names_])

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        base_estimators = self._build_estimators()
        labels = _coerce_binary_labels(y) if y is not None else None

        fitted_pairs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single)(name, estimator, X) for name, estimator in base_estimators.items()
        )
        self.estimators_ = OrderedDict(fitted_pairs)
        self.component_names_ = list(self.estimators_)

        self.component_stats_: dict[str, dict[str, float]] = {}
        component_anomaly_scores: list[np.ndarray] = []
        for name, estimator in self.estimators_.items():
            scores = _normalized_anomaly_scores(estimator, X)
            scaled, minimum, maximum = _minmax_scale(scores)
            self.component_stats_[name] = {"min": minimum, "max": maximum}
            component_anomaly_scores.append(scaled)

        component_matrix = np.column_stack(component_anomaly_scores)
        self.fusion_strategy_ = self.fusion_strategy
        if self.fusion_strategy_ not in {"weighted_average", "max_score_voting", "stacking", "moe"}:
            raise ValueError(
                "fusion_strategy must be 'weighted_average', 'max_score_voting', 'stacking', or 'moe'"
            )

        self.fusion_weights_ = self._resolve_fusion_weights()
        if self.fusion_strategy_ == "weighted_average":
            fused = component_matrix @ np.array([self.fusion_weights_[name] for name in self.component_names_], dtype=float)
            self.offset_ = float(np.quantile(fused, 1 - self.contamination))
        elif self.fusion_strategy_ == "max_score_voting":
            fused = np.max(component_matrix, axis=1)
            self.offset_ = float(self.max_score_threshold)
        elif self.fusion_strategy_ == "moe":
            self.moe_gate_, self.moe_gate_training_signal_ = self._fit_moe_gate(X, component_matrix, labels)
            fused = np.sum(component_matrix * self._moe_gate_weights(X), axis=1)
            self.offset_ = float(np.quantile(fused, 1 - self.contamination))
        else:
            if labels is None:
                raise ValueError("Stacking fusion requires labeled training targets passed to fit(X, y).")
            if labels.shape[0] != X.shape[0]:
                raise ValueError("Stacking labels must have the same number of rows as X.")

            stacking_features = self._stacking_feature_matrix(component_matrix)
            self.stacking_meta_model_ = self._build_stacking_meta_model(n_samples=stacking_features.shape[0])
            meta_labels = np.asarray(labels, dtype=int).reshape(-1)
            meta_model = _clone_meta_model(self.stacking_meta_model_)
            meta_model.fit(stacking_features, meta_labels)
            self.stacking_meta_model_ = meta_model
            fused = np.asarray(self.stacking_meta_model_.predict_proba(stacking_features), dtype=float)[:, 1]
            self.offset_ = float(np.quantile(fused, 1 - self.contamination))
            self.stacking_meta_model_type_ = type(self.stacking_meta_model_).__name__

        should_calibrate = (
            self.calibrate_threshold
            and labels is not None
            and self.fusion_strategy_ in {"weighted_average", "stacking", "moe"}
            and labels.shape[0] >= int(self.calibration_min_samples)
            and np.unique(labels).size > 1
        )
        self.calibration_applied_ = bool(should_calibrate)
        if should_calibrate:
            calibrated_threshold, calibration_metrics = _calibrate_threshold_from_scores(fused, labels)
            self.offset_ = float(calibrated_threshold)
            self.calibrated_threshold_ = float(calibrated_threshold)
            self.calibration_metrics_ = calibration_metrics

        self.training_raw_anomaly_score_ = fused
        return self

    def _stacking_feature_matrix(self, component_matrix: np.ndarray) -> np.ndarray:
        if component_matrix.shape[1] < 2:
            raise ValueError("Stacking requires at least two detector score columns.")
        return component_matrix

    def _resolve_fusion_weights(self) -> dict[str, float]:
        default_weights = {
            "isolation_forest": 0.3,
            "one_class_svm": 0.0,
            "local_outlier_factor": 0.0,
            "autoencoder": 0.4,
            "variational_autoencoder": 0.1,
            "ganomaly": 0.1,
            "anomaly_transformer": 0.1,
            "cnn_autoencoder": 0.1,
            "deep_svdd": 0.3,
        }
        if self.fusion_weights is None:
            return {name: default_weights.get(name, 0.0) for name in self.component_names_}

        resolved = {name: float(self.fusion_weights.get(name, 0.0)) for name in self.component_names_}
        weight_sum = float(sum(resolved.values()))
        if weight_sum <= 0.0 or np.isnan(weight_sum):
            return {name: 1.0 / len(self.component_names_) for name in self.component_names_}
        return {name: weight / weight_sum for name, weight in resolved.items()}

    def _component_anomaly_matrix(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Ensemble must be fit before scoring.")

        X = _ensure_2d_array(X)
        columns: list[np.ndarray] = []
        for name, estimator in self.estimators_.items():
            scores = _normalized_anomaly_scores(estimator, X)
            scaled, _, _ = _minmax_scale(scores)
            columns.append(scaled)
        return np.column_stack(columns)

    def score_components(self, X) -> pd.DataFrame:
        matrix = self._component_anomaly_matrix(X)
        return pd.DataFrame(
            matrix,
            columns=[f"{name}_anomaly_score" for name in self.component_names_],
        )

    def raw_anomaly_score(self, X) -> np.ndarray:
        matrix = self._component_anomaly_matrix(X)
        if self.fusion_strategy_ == "stacking":
            if not hasattr(self, "stacking_meta_model_"):
                raise RuntimeError("Stacking fusion requires a fitted meta-classifier.")
            features = self._stacking_feature_matrix(matrix)
            return np.asarray(self.stacking_meta_model_.predict_proba(features), dtype=float)[:, 1]
        if self.fusion_strategy_ == "max_score_voting":
            return np.max(matrix, axis=1)
        if self.fusion_strategy_ == "moe":
            gate_weights = self._moe_gate_weights(_ensure_2d_array(X))
            return np.sum(matrix * gate_weights, axis=1)
        weights = np.array([self.fusion_weights_[name] for name in self.component_names_], dtype=float)
        return matrix @ weights

    def score(self, X) -> np.ndarray:
        return self.raw_anomaly_score(X)

    def score_samples(self, X) -> np.ndarray:
        return -self.score(X)

    def decision_function(self, X) -> np.ndarray:
        return self.offset_ - self.raw_anomaly_score(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)
