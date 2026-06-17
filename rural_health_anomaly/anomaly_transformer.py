"""Attention-based anomaly transformer for tabular screening data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin


def _ensure_2d_array(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array-like input.")
    return array


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)


def _dropout(x: np.ndarray, *, dropout: float, rng: np.random.Generator, training: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if not training or dropout <= 0.0:
        return x, None
    keep_prob = float(np.clip(1.0 - dropout, 1e-6, 1.0))
    mask = rng.binomial(1, keep_prob, size=x.shape).astype(float) / keep_prob
    return x * mask, mask


@dataclass
class _ForwardCache:
    attention_logits: np.ndarray
    attention_weights: np.ndarray
    context: np.ndarray
    hidden_linear: np.ndarray
    hidden_activation: np.ndarray
    hidden_dropout_mask: np.ndarray | None
    latent_linear: np.ndarray
    latent_activation: np.ndarray
    latent_dropout_mask: np.ndarray | None
    decoder_linear: np.ndarray
    decoder_activation: np.ndarray
    decoder_dropout_mask: np.ndarray | None
    recon: np.ndarray


class AnomalyTransformer(BaseEstimator, OutlierMixin):
    """A lightweight transformer-style anomaly detector for tabular data."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        attention_weight: float = 0.5,
        attention_temperature: float = 1.0,
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
        self.attention_weight = attention_weight
        self.attention_temperature = attention_temperature
        self.threshold_percentile = threshold_percentile
        self.validation_fraction = validation_fraction
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_parameters(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.random_state)
        scale = 1.0 / np.sqrt(max(input_dim, 1))
        latent_scale = 1.0 / np.sqrt(max(self.hidden_dim, 1))
        output_scale = 1.0 / np.sqrt(max(self.latent_dim, 1))

        self.attention_weights_ = rng.normal(0.0, scale, size=(input_dim, input_dim))
        self.attention_bias_ = np.zeros(input_dim, dtype=float)
        self.encoder_weights_ = rng.normal(0.0, scale, size=(input_dim, self.hidden_dim))
        self.encoder_bias_ = np.zeros(self.hidden_dim, dtype=float)
        self.latent_weights_ = rng.normal(0.0, latent_scale, size=(self.hidden_dim, self.latent_dim))
        self.latent_bias_ = np.zeros(self.latent_dim, dtype=float)
        self.decoder_weights_ = rng.normal(0.0, output_scale, size=(self.latent_dim, self.hidden_dim))
        self.decoder_bias_ = np.zeros(self.hidden_dim, dtype=float)
        self.output_weights_ = rng.normal(0.0, output_scale, size=(self.hidden_dim, input_dim))
        self.output_bias_ = np.zeros(input_dim, dtype=float)

    def _forward(self, X: np.ndarray, *, training: bool, rng: np.random.Generator | None = None) -> _ForwardCache:
        if training and rng is None:
            raise ValueError("rng is required when training=True.")

        temperature = max(float(self.attention_temperature), 1e-6)
        attention_logits = (X @ self.attention_weights_ + self.attention_bias_) / temperature
        attention_weights = _softmax(attention_logits)
        context = X * attention_weights

        hidden_linear = context @ self.encoder_weights_ + self.encoder_bias_
        hidden_activation = _relu(hidden_linear)
        hidden_activation, hidden_dropout_mask = _dropout(hidden_activation, dropout=self.dropout, rng=rng or np.random.default_rng(), training=training)

        latent_linear = hidden_activation @ self.latent_weights_ + self.latent_bias_
        latent_activation = _relu(latent_linear)
        latent_activation, latent_dropout_mask = _dropout(latent_activation, dropout=self.dropout, rng=rng or np.random.default_rng(), training=training)

        decoder_linear = latent_activation @ self.decoder_weights_ + self.decoder_bias_
        decoder_activation = _relu(decoder_linear)
        decoder_activation, decoder_dropout_mask = _dropout(decoder_activation, dropout=self.dropout, rng=rng or np.random.default_rng(), training=training)

        recon = decoder_activation @ self.output_weights_ + self.output_bias_

        return _ForwardCache(
            attention_logits=attention_logits,
            attention_weights=attention_weights,
            context=context,
            hidden_linear=hidden_linear,
            hidden_activation=hidden_activation,
            hidden_dropout_mask=hidden_dropout_mask,
            latent_linear=latent_linear,
            latent_activation=latent_activation,
            latent_dropout_mask=latent_dropout_mask,
            decoder_linear=decoder_linear,
            decoder_activation=decoder_activation,
            decoder_dropout_mask=decoder_dropout_mask,
            recon=recon,
        )

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        cache = self._forward(X, training=False)
        return np.mean((cache.recon - X) ** 2, axis=1)

    def _attention_discrepancy(self, X: np.ndarray) -> np.ndarray:
        cache = self._forward(X, training=False)
        attention = np.clip(cache.attention_weights, 1e-12, 1.0)
        uniform_mass = 1.0 / max(attention.shape[1], 1)
        kl_to_uniform = np.sum(attention * np.log(attention / uniform_mass), axis=1)
        normalizer = np.log(max(attention.shape[1], 2))
        return np.clip(kl_to_uniform / max(normalizer, 1e-6), 0.0, 1.0)

    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        reconstruction = self._reconstruction_error(X)
        discrepancy = self._attention_discrepancy(X)
        return reconstruction + self.attention_weight * discrepancy

    def _train_batch(self, X_batch: np.ndarray, rng: np.random.Generator) -> float:
        cache = self._forward(X_batch, training=True, rng=rng)
        batch_size, feature_dim = X_batch.shape

        recon_error = cache.recon - X_batch
        grad_recon = 2.0 * recon_error / float(max(batch_size * feature_dim, 1))

        grad_output_weights = cache.decoder_activation.T @ grad_recon + self.l2 * self.output_weights_
        grad_output_bias = np.sum(grad_recon, axis=0)
        grad_decoder = grad_recon @ self.output_weights_.T

        if cache.decoder_dropout_mask is not None:
            grad_decoder = grad_decoder * cache.decoder_dropout_mask
        grad_decoder_linear = grad_decoder * _relu_grad(cache.decoder_linear)
        grad_decoder_weights = cache.latent_activation.T @ grad_decoder_linear + self.l2 * self.decoder_weights_
        grad_decoder_bias = np.sum(grad_decoder_linear, axis=0)
        grad_latent = grad_decoder_linear @ self.decoder_weights_.T

        if cache.latent_dropout_mask is not None:
            grad_latent = grad_latent * cache.latent_dropout_mask
        grad_latent_linear = grad_latent * _relu_grad(cache.latent_linear)
        grad_latent_weights = cache.hidden_activation.T @ grad_latent_linear + self.l2 * self.latent_weights_
        grad_latent_bias = np.sum(grad_latent_linear, axis=0)
        grad_hidden = grad_latent_linear @ self.latent_weights_.T

        if cache.hidden_dropout_mask is not None:
            grad_hidden = grad_hidden * cache.hidden_dropout_mask
        grad_hidden_linear = grad_hidden * _relu_grad(cache.hidden_linear)
        grad_encoder_weights = cache.context.T @ grad_hidden_linear + self.l2 * self.encoder_weights_
        grad_encoder_bias = np.sum(grad_hidden_linear, axis=0)
        grad_context = grad_hidden_linear @ self.encoder_weights_.T

        grad_attention = grad_context * X_batch
        softmax_correction = np.sum(grad_attention * cache.attention_weights, axis=1, keepdims=True)
        grad_attention_logits = cache.attention_weights * (grad_attention - softmax_correction)
        grad_attention_weights = X_batch.T @ grad_attention_logits + self.l2 * self.attention_weights_
        grad_attention_bias = np.sum(grad_attention_logits, axis=0)

        self._adam_step("output_weights_", grad_output_weights)
        self._adam_step("output_bias_", grad_output_bias)
        self._adam_step("decoder_weights_", grad_decoder_weights)
        self._adam_step("decoder_bias_", grad_decoder_bias)
        self._adam_step("latent_weights_", grad_latent_weights)
        self._adam_step("latent_bias_", grad_latent_bias)
        self._adam_step("encoder_weights_", grad_encoder_weights)
        self._adam_step("encoder_bias_", grad_encoder_bias)
        self._adam_step("attention_weights_", grad_attention_weights)
        self._adam_step("attention_bias_", grad_attention_bias)

        raw_score = self._raw_score(X_batch)
        return float(np.mean(raw_score))

    def _adam_step(self, name: str, grad: np.ndarray) -> None:
        if not hasattr(self, "_adam_state_"):
            self._adam_state_ = {}
        state = self._adam_state_.setdefault(
            name,
            {
                "m": np.zeros_like(getattr(self, name)),
                "v": np.zeros_like(getattr(self, name)),
                "t": 0,
            },
        )
        state["t"] += 1
        beta1, beta2 = 0.9, 0.999
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad * grad)
        m_hat = state["m"] / (1 - beta1**state["t"])
        v_hat = state["v"] / (1 - beta2**state["t"])
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        setattr(self, name, getattr(self, name) - update)

    def reconstruction_error(self, X: Any) -> np.ndarray:
        return self._reconstruction_error(_ensure_2d_array(X))

    def reconstruction_residuals(self, X: Any) -> np.ndarray:
        X = _ensure_2d_array(X)
        if not hasattr(self, "attention_weights_"):
            raise RuntimeError("AnomalyTransformer must be fit before scoring.")
        cache = self._forward(X, training=False)
        return cache.recon - X

    def attention_discrepancy(self, X: Any) -> np.ndarray:
        return self._attention_discrepancy(_ensure_2d_array(X))

    def raw_score(self, X: Any) -> np.ndarray:
        return self.score(X)

    def score(self, X: Any) -> np.ndarray:
        X = _ensure_2d_array(X)
        return self._raw_score(X)

    def score_samples(self, X: Any) -> np.ndarray:
        return -self.score(X)

    def decision_function(self, X: Any) -> np.ndarray:
        return self.offset_ - self.score(X)

    def predict(self, X: Any) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        if X.shape[0] == 0:
            raise ValueError("AnomalyTransformer requires at least one sample.")

        self._initialize_parameters(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        validation_fraction = float(np.clip(self.validation_fraction, 0.0, 0.95))
        if X.shape[0] >= 5 and validation_fraction > 0.0:
            permutation = rng.permutation(X.shape[0])
            validation_size = max(1, int(np.floor(X.shape[0] * validation_fraction)))
            validation_size = min(validation_size, X.shape[0] - 1)
            val_idx = permutation[:validation_size]
            train_idx = permutation[validation_size:]
            if train_idx.size == 0:
                train_idx = permutation
                val_idx = permutation[-validation_size:]
            X_train = X[train_idx]
            X_val = X[val_idx]
        else:
            X_train = X
            X_val = X

        batch_size = max(1, min(int(self.batch_size), X_train.shape[0]))
        patience = max(1, int(self.patience))
        best_state: dict[str, np.ndarray] | None = None
        best_loss = float("inf")
        best_epoch = -1
        stalled_epochs = 0

        self._adam_state_ = {}
        for epoch in range(int(self.max_epochs)):
            order = rng.permutation(X_train.shape[0])
            epoch_losses: list[float] = []
            for start in range(0, X_train.shape[0], batch_size):
                batch_indices = order[start : start + batch_size]
                batch = X_train[batch_indices]
                epoch_losses.append(self._train_batch(batch, rng))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            val_loss = float(np.mean(self.score(X_val)))

            if self.verbose:
                print(
                    f"[AnomalyTransformer] epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                    flush=True,
                )

            if val_loss + 1e-10 < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                stalled_epochs = 0
                best_state = {
                    "attention_weights_": self.attention_weights_.copy(),
                    "attention_bias_": self.attention_bias_.copy(),
                    "encoder_weights_": self.encoder_weights_.copy(),
                    "encoder_bias_": self.encoder_bias_.copy(),
                    "latent_weights_": self.latent_weights_.copy(),
                    "latent_bias_": self.latent_bias_.copy(),
                    "decoder_weights_": self.decoder_weights_.copy(),
                    "decoder_bias_": self.decoder_bias_.copy(),
                    "output_weights_": self.output_weights_.copy(),
                    "output_bias_": self.output_bias_.copy(),
                }
            else:
                stalled_epochs += 1
                if stalled_epochs >= patience:
                    break

        if best_state is not None:
            for name, value in best_state.items():
                setattr(self, name, value)

        train_raw = self.score(X)
        self._training_raw_score_mean_ = float(np.mean(train_raw))
        self._training_raw_score_std_ = float(np.std(train_raw, ddof=0) or 1.0)
        self.threshold_ = float(np.percentile(train_raw, self.threshold_percentile))
        self.offset_ = float(self.threshold_)
        self.n_features_in_ = X.shape[1]
        self.best_epoch_ = int(best_epoch if best_epoch >= 0 else 0)
        return self
