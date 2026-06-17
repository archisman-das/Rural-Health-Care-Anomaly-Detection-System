"""GANomaly-inspired detector for tabular anomaly scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.model_selection import train_test_split


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _relu_grad(values: np.ndarray) -> np.ndarray:
    return (values > 0.0).astype(float)


@dataclass
class _GANForwardCache:
    encoder_linear: np.ndarray
    encoder_activation: np.ndarray
    latent: np.ndarray
    decoder_linear: np.ndarray
    decoder_activation: np.ndarray
    recon: np.ndarray
    reencoder_linear: np.ndarray
    reencoder_activation: np.ndarray
    latent_hat: np.ndarray
    encoder_dropout_mask: np.ndarray | None
    decoder_dropout_mask: np.ndarray | None
    reencoder_dropout_mask: np.ndarray | None


class GANomaly(BaseEstimator, OutlierMixin):
    """A lightweight GANomaly-style autoencoder with latent-consistency scoring."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        consistency_weight: float = 1.0,
        threshold_percentile: float = 97.5,
        validation_fraction: float = 0.2,
        max_epochs: int = 80,
        patience: int = 10,
        l2: float = 1e-5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.consistency_weight = consistency_weight
        self.threshold_percentile = threshold_percentile
        self.validation_fraction = validation_fraction
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_parameters(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)

        def init_weight(in_dim: int, out_dim: int) -> np.ndarray:
            scale = np.sqrt(2.0 / max(in_dim, 1))
            return rng.normal(0.0, scale, size=(in_dim, out_dim))

        self.weights_: list[np.ndarray] = [
            init_weight(n_features, self.hidden_dim),
            init_weight(self.hidden_dim, self.latent_dim),
            init_weight(self.latent_dim, self.hidden_dim),
            init_weight(self.hidden_dim, n_features),
            init_weight(n_features, self.hidden_dim),
            init_weight(self.hidden_dim, self.latent_dim),
        ]
        self.biases_: list[np.ndarray] = [
            np.zeros(self.hidden_dim, dtype=float),
            np.zeros(self.latent_dim, dtype=float),
            np.zeros(self.hidden_dim, dtype=float),
            np.zeros(n_features, dtype=float),
            np.zeros(self.hidden_dim, dtype=float),
            np.zeros(self.latent_dim, dtype=float),
        ]
        self._adam_m_w_: list[np.ndarray] = [np.zeros_like(weight) for weight in self.weights_]
        self._adam_v_w_: list[np.ndarray] = [np.zeros_like(weight) for weight in self.weights_]
        self._adam_m_b_: list[np.ndarray] = [np.zeros_like(bias) for bias in self.biases_]
        self._adam_v_b_: list[np.ndarray] = [np.zeros_like(bias) for bias in self.biases_]
        self._adam_step_ = 0

    def _apply_dropout(self, values: np.ndarray, *, training: bool, layer_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        if not training or self.dropout <= 0.0:
            return values, None
        keep_prob = 1.0 - self.dropout
        rng = np.random.default_rng(self.random_state + self._adam_step_ + layer_idx)
        mask = (rng.random(values.shape) < keep_prob).astype(float)
        return values * mask / keep_prob, mask

    def _forward(self, X: np.ndarray, *, training: bool) -> tuple[np.ndarray, _GANForwardCache]:
        encoder_linear = X @ self.weights_[0] + self.biases_[0]
        encoder_activation = _relu(encoder_linear)
        encoder_activation, encoder_dropout_mask = self._apply_dropout(encoder_activation, training=training, layer_idx=0)
        latent = encoder_activation @ self.weights_[1] + self.biases_[1]

        decoder_linear = latent @ self.weights_[2] + self.biases_[2]
        decoder_activation = _relu(decoder_linear)
        decoder_activation, decoder_dropout_mask = self._apply_dropout(decoder_activation, training=training, layer_idx=1)
        recon = decoder_activation @ self.weights_[3] + self.biases_[3]

        reencoder_linear = recon @ self.weights_[4] + self.biases_[4]
        reencoder_activation = _relu(reencoder_linear)
        reencoder_activation, reencoder_dropout_mask = self._apply_dropout(reencoder_activation, training=training, layer_idx=2)
        latent_hat = reencoder_activation @ self.weights_[5] + self.biases_[5]

        cache = _GANForwardCache(
            encoder_linear=encoder_linear,
            encoder_activation=encoder_activation,
            latent=latent,
            decoder_linear=decoder_linear,
            decoder_activation=decoder_activation,
            recon=recon,
            reencoder_linear=reencoder_linear,
            reencoder_activation=reencoder_activation,
            latent_hat=latent_hat,
            encoder_dropout_mask=encoder_dropout_mask,
            decoder_dropout_mask=decoder_dropout_mask,
            reencoder_dropout_mask=reencoder_dropout_mask,
        )
        return latent_hat, cache

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        _, cache = self._forward(X, training=False)
        return np.mean((cache.recon - X) ** 2, axis=1)

    def _latent_consistency_error(self, X: np.ndarray) -> np.ndarray:
        _, cache = self._forward(X, training=False)
        return np.mean((cache.latent_hat - cache.latent) ** 2, axis=1)

    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        recon_error = self._reconstruction_error(X)
        latent_error = self._latent_consistency_error(X)
        return recon_error + self.consistency_weight * latent_error

    def _update_score_stats(self, X: np.ndarray) -> None:
        raw_scores = self._raw_score(X)
        self._training_raw_score_mean_ = float(np.mean(raw_scores))
        self._training_raw_score_std_ = float(np.std(raw_scores, ddof=0))
        if self._training_raw_score_std_ == 0.0 or np.isnan(self._training_raw_score_std_):
            self._training_raw_score_std_ = 1.0

    def _backward(
        self,
        X: np.ndarray,
        cache: _GANForwardCache,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_size = X.shape[0]
        grads_w: list[np.ndarray] = [np.zeros_like(weight) for weight in self.weights_]
        grads_b: list[np.ndarray] = [np.zeros_like(bias) for bias in self.biases_]

        delta_latent_hat = (2.0 * self.consistency_weight / batch_size) * (cache.latent_hat - cache.latent)
        grads_w[5] = cache.reencoder_activation.T @ delta_latent_hat + self.l2 * self.weights_[5]
        grads_b[5] = delta_latent_hat.sum(axis=0)

        delta_reencoder = delta_latent_hat @ self.weights_[5].T
        if cache.reencoder_dropout_mask is not None:
            delta_reencoder = delta_reencoder * cache.reencoder_dropout_mask / (1.0 - self.dropout)
        delta_reencoder = delta_reencoder * _relu_grad(cache.reencoder_linear)

        grads_w[4] = cache.recon.T @ delta_reencoder + self.l2 * self.weights_[4]
        grads_b[4] = delta_reencoder.sum(axis=0)

        delta_recon = (2.0 / batch_size) * (cache.recon - X)
        delta_recon_total = delta_recon + delta_reencoder @ self.weights_[4].T

        grads_w[3] = cache.decoder_activation.T @ delta_recon_total + self.l2 * self.weights_[3]
        grads_b[3] = delta_recon_total.sum(axis=0)

        delta_decoder = delta_recon_total @ self.weights_[3].T
        if cache.decoder_dropout_mask is not None:
            delta_decoder = delta_decoder * cache.decoder_dropout_mask / (1.0 - self.dropout)
        delta_decoder = delta_decoder * _relu_grad(cache.decoder_linear)

        grads_w[2] = cache.latent.T @ delta_decoder + self.l2 * self.weights_[2]
        grads_b[2] = delta_decoder.sum(axis=0)

        delta_latent = delta_decoder @ self.weights_[2].T - delta_latent_hat

        grads_w[1] = cache.encoder_activation.T @ delta_latent + self.l2 * self.weights_[1]
        grads_b[1] = delta_latent.sum(axis=0)

        delta_encoder = delta_latent @ self.weights_[1].T
        if cache.encoder_dropout_mask is not None:
            delta_encoder = delta_encoder * cache.encoder_dropout_mask / (1.0 - self.dropout)
        delta_encoder = delta_encoder * _relu_grad(cache.encoder_linear)

        grads_w[0] = X.T @ delta_encoder + self.l2 * self.weights_[0]
        grads_b[0] = delta_encoder.sum(axis=0)
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

    def reconstruction_error(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("GANomaly must be fit before scoring.")
        return self._reconstruction_error(np.asarray(X, dtype=float))

    def latent_consistency_error(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("GANomaly must be fit before scoring.")
        return self._latent_consistency_error(np.asarray(X, dtype=float))

    def reconstruction_residuals(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("GANomaly must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        _, cache = self._forward(X, training=False)
        return cache.recon - X

    def raw_score(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("GANomaly must be fit before scoring.")
        return self._raw_score(np.asarray(X, dtype=float))

    def score(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_training_raw_score_mean_"):
            raise RuntimeError("GANomaly must be fit before scoring.")
        raw_scores = self.raw_score(X)
        return (raw_scores - self._training_raw_score_mean_) / self._training_raw_score_std_

    def decision_function(self, X: Any) -> np.ndarray:
        return self.threshold_ - self.raw_score(X)

    def score_samples(self, X: Any) -> np.ndarray:
        return -self.raw_score(X)

    def predict(self, X: Any) -> np.ndarray:
        return np.where(self.raw_score(X) <= self.threshold_, 1, -1)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected a 2D array-like input.")
        if X.shape[0] < 2:
            raise ValueError("GANomaly requires at least two samples.")

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
        best_val_loss = np.inf
        patience_counter = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            indices = rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = X_train[batch_idx]
                _, cache = self._forward(batch, training=True)
                grads_w, grads_b = self._backward(batch, cache)
                self._apply_adam(grads_w, grads_b)

            train_recon = self._reconstruction_error(X_train)
            train_latent = self._latent_consistency_error(X_train)
            val_recon = self._reconstruction_error(X_val)
            val_latent = self._latent_consistency_error(X_val)
            train_loss = float(np.mean(train_recon + self.consistency_weight * train_latent))
            val_loss = float(np.mean(val_recon + self.consistency_weight * val_latent))
            self.history_.append({"epoch": float(epoch), "train_mse": train_loss, "val_mse": val_loss})

            if self.verbose:
                print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
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

        validation_scores = self._raw_score(X_val)
        self.threshold_ = float(np.percentile(validation_scores, self.threshold_percentile))
        self.validation_mse_ = float(np.mean(validation_scores))
        self._update_score_stats(X)
        self.fitted_ = True
        return self
