"""Deep autoencoder anomaly detector."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.model_selection import train_test_split


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


@dataclass
class _ForwardCache:
    activations: list[np.ndarray]
    pre_activations: list[np.ndarray]
    dropout_masks: list[np.ndarray | None]


class DeepAutoencoder(BaseEstimator, OutlierMixin):
    """Mirror-architecture autoencoder with reconstruction-threshold scoring."""

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        threshold_percentile: float = 97.5,
        validation_fraction: float = 0.2,
        max_epochs: int = 80,
        patience: int = 10,
        l2: float = 1e-5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.validation_fraction = validation_fraction
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    def _layer_sizes(self, n_features: int) -> list[int]:
        return [n_features, 128, 64, 32, self.latent_dim, 32, 64, 128, n_features]

    def _initialize_parameters(self, n_features: int) -> None:
        layer_sizes = self._layer_sizes(n_features)
        rng = np.random.default_rng(self.random_state)

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self._adam_m_w_: list[np.ndarray] = []
        self._adam_v_w_: list[np.ndarray] = []
        self._adam_m_b_: list[np.ndarray] = []
        self._adam_v_b_: list[np.ndarray] = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / in_dim)
            weight = rng.normal(0.0, scale, size=(in_dim, out_dim))
            bias = np.zeros(out_dim, dtype=float)
            self.weights_.append(weight)
            self.biases_.append(bias)
            self._adam_m_w_.append(np.zeros_like(weight))
            self._adam_v_w_.append(np.zeros_like(weight))
            self._adam_m_b_.append(np.zeros_like(bias))
            self._adam_v_b_.append(np.zeros_like(bias))

        self._adam_step_ = 0

    def _forward(self, X: np.ndarray, *, training: bool) -> tuple[np.ndarray, _ForwardCache]:
        activations: list[np.ndarray] = [X]
        pre_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray | None] = []
        a = X

        for layer_idx in range(len(self.weights_) - 1):
            z = a @ self.weights_[layer_idx] + self.biases_[layer_idx]
            pre_activations.append(z)
            a = _relu(z)
            mask = None
            if training and self.dropout > 0:
                keep_prob = 1.0 - self.dropout
                rng = np.random.default_rng(self.random_state + self._adam_step_ + layer_idx)
                mask = (rng.random(a.shape) < keep_prob).astype(float)
                a = a * mask / keep_prob
            dropout_masks.append(mask)
            activations.append(a)

        output = a @ self.weights_[-1] + self.biases_[-1]
        pre_activations.append(output)
        return output, _ForwardCache(activations, pre_activations, dropout_masks)

    def _backward(
        self,
        X: np.ndarray,
        y_hat: np.ndarray,
        cache: _ForwardCache,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        m = X.shape[0]
        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights_]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases_]

        delta = (2.0 / m) * (y_hat - X)
        grads_w[-1] = cache.activations[-1].T @ delta + self.l2 * self.weights_[-1]
        grads_b[-1] = delta.sum(axis=0)
        delta = delta @ self.weights_[-1].T

        for layer_idx in range(len(self.weights_) - 2, -1, -1):
            delta = delta * _relu_grad(cache.pre_activations[layer_idx])
            mask = cache.dropout_masks[layer_idx]
            if mask is not None:
                delta = delta * mask / (1.0 - self.dropout)

            grads_w[layer_idx] = cache.activations[layer_idx].T @ delta + self.l2 * self.weights_[layer_idx]
            grads_b[layer_idx] = delta.sum(axis=0)

            if layer_idx > 0:
                delta = delta @ self.weights_[layer_idx].T

        return grads_w, grads_b

    def _apply_adam(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        self._adam_step_ += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for idx, (grad_w, grad_b) in enumerate(zip(grads_w, grads_b)):
            self._adam_m_w_[idx] = beta1 * self._adam_m_w_[idx] + (1.0 - beta1) * grad_w
            self._adam_v_w_[idx] = beta2 * self._adam_v_w_[idx] + (1.0 - beta2) * (grad_w**2)
            self._adam_m_b_[idx] = beta1 * self._adam_m_b_[idx] + (1.0 - beta1) * grad_b
            self._adam_v_b_[idx] = beta2 * self._adam_v_b_[idx] + (1.0 - beta2) * (grad_b**2)

            m_w_hat = self._adam_m_w_[idx] / (1.0 - beta1**self._adam_step_)
            v_w_hat = self._adam_v_w_[idx] / (1.0 - beta2**self._adam_step_)
            m_b_hat = self._adam_m_b_[idx] / (1.0 - beta1**self._adam_step_)
            v_b_hat = self._adam_v_b_[idx] / (1.0 - beta2**self._adam_step_)

            self.weights_[idx] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases_[idx] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        recon, _ = self._forward(X, training=False)
        return np.mean((recon - X) ** 2, axis=1)

    def _update_score_stats(self, X: np.ndarray) -> None:
        raw_scores = self._reconstruction_error(X)
        self._training_raw_score_mean_ = float(np.mean(raw_scores))
        self._training_raw_score_std_ = float(np.std(raw_scores, ddof=0))
        if self._training_raw_score_std_ == 0.0 or np.isnan(self._training_raw_score_std_):
            self._training_raw_score_std_ = 1.0

    def reconstruction_error(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Autoencoder must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        return self._reconstruction_error(X)

    def raw_score(self, X: Any) -> np.ndarray:
        return self.reconstruction_error(X)

    def score(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_training_raw_score_mean_"):
            raise RuntimeError("Autoencoder must be fit before scoring.")
        raw_scores = self.raw_score(X)
        return (raw_scores - self._training_raw_score_mean_) / self._training_raw_score_std_

    def reconstruction_mae(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Autoencoder must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        recon, _ = self._forward(X, training=False)
        return np.mean(np.abs(recon - X), axis=1)

    def reconstruction_residuals(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Autoencoder must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        recon, _ = self._forward(X, training=False)
        return recon - X

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected a 2D array-like input.")
        if X.shape[0] < 2:
            raise ValueError("Autoencoder requires at least two samples.")

        self.n_features_in_ = X.shape[1]
        self._initialize_parameters(self.n_features_in_)

        if 0.0 < self.validation_fraction < 1.0 and X.shape[0] >= 5:
            X_train, X_val = train_test_split(
                X,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=True,
            )
        else:
            X_train = X
            X_val = X

        batch_size = max(1, min(self.batch_size, X_train.shape[0]))
        rng = np.random.default_rng(self.random_state)
        best_state: dict[str, Any] | None = None
        best_val_error = np.inf
        patience_counter = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            indices = rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = X_train[batch_idx]
                predictions, cache = self._forward(batch, training=True)
                grads_w, grads_b = self._backward(batch, predictions, cache)
                self._apply_adam(grads_w, grads_b)

            train_error = self._reconstruction_error(X_train)
            val_error = self._reconstruction_error(X_val)
            train_loss = float(np.mean(train_error))
            val_loss = float(np.mean(val_error))
            self.history_.append({"epoch": float(epoch), "train_mse": train_loss, "val_mse": val_loss})

            if self.verbose:
                print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

            if val_loss < best_val_error - 1e-6:
                best_val_error = val_loss
                best_state = {
                    "weights": [weight.copy() for weight in self.weights_],
                    "biases": [bias.copy() for bias in self.biases_],
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.weights_ = [weight.copy() for weight in best_state["weights"]]
            self.biases_ = [bias.copy() for bias in best_state["biases"]]

        validation_errors = self._reconstruction_error(X_val)
        self.threshold_ = float(np.percentile(validation_errors, self.threshold_percentile))
        self.validation_mse_ = float(np.mean(validation_errors))
        self._update_score_stats(X)
        self.fitted_ = True
        return self

    def score_samples(self, X):
        return -self.raw_score(X)

    def decision_function(self, X):
        return self.threshold_ - self.raw_score(X)

    def predict(self, X):
        errors = self.reconstruction_error(X)
        return np.where(errors <= self.threshold_, 1, -1)
