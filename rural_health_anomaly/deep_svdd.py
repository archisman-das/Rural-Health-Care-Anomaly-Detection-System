"""Deep SVDD anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.model_selection import train_test_split

from .autoencoder import DeepAutoencoder


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


@dataclass
class _ForwardCache:
    activations: list[np.ndarray]
    pre_activations: list[np.ndarray]


@dataclass
class _CNNCache:
    inputs: np.ndarray
    conv1_pre: np.ndarray
    conv1_out: np.ndarray
    conv1_padded: np.ndarray
    conv2_pre: np.ndarray
    conv2_out: np.ndarray
    conv2_padded: np.ndarray
    pooled: np.ndarray


def _same_padding(kernel_size: int) -> int:
    return kernel_size // 2


def _conv1d_same_forward(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch, seq_len, in_channels = inputs.shape
    out_channels, kernel_size, weight_in_channels = weights.shape
    if weight_in_channels != in_channels:
        raise ValueError("Input channel count does not match convolution weights.")

    pad = _same_padding(kernel_size)
    padded = np.pad(inputs, ((0, 0), (pad, pad), (0, 0)), mode="constant")
    output = np.zeros((batch, seq_len, out_channels), dtype=float)

    for t in range(seq_len):
        window = padded[:, t : t + kernel_size, :]
        for oc in range(out_channels):
            output[:, t, oc] = np.sum(window * weights[oc][None, :, :], axis=(1, 2)) + bias[oc]

    return output, padded


def _conv1d_same_backward(
    grad_output: np.ndarray,
    padded_inputs: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, seq_len, out_channels = grad_output.shape
    _, padded_len, in_channels = padded_inputs.shape
    kernel_size = weights.shape[1]
    pad = _same_padding(kernel_size)

    grad_weights = np.zeros_like(weights)
    grad_bias = grad_output.sum(axis=(0, 1))
    grad_padded_inputs = np.zeros_like(padded_inputs)

    for t in range(seq_len):
        window = padded_inputs[:, t : t + kernel_size, :]
        for oc in range(out_channels):
            delta = grad_output[:, t, oc][:, None, None]
            grad_weights[oc] += np.sum(window * delta, axis=0)
            grad_padded_inputs[:, t : t + kernel_size, :] += weights[oc][None, :, :] * delta

    if pad > 0:
        grad_inputs = grad_padded_inputs[:, pad:-pad, :]
    else:
        grad_inputs = grad_padded_inputs

    return grad_inputs, grad_weights, grad_bias


class DeepSVDD(BaseEstimator, OutlierMixin):
    """Hypersphere-based deep one-class model with optional autoencoder pretraining."""

    def __init__(
        self,
        *,
        nu: float = 0.05,
        center_fixed: bool = True,
        architecture: str = "mlp",
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 60,
        validation_fraction: float = 0.2,
        pretrain_autoencoder: bool = True,
        pretrain_epochs: int = 25,
        pretrain_dropout: float = 0.2,
        pretrain_learning_rate: float = 1e-3,
        pretrain_batch_size: int = 32,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.nu = nu
        self.center_fixed = center_fixed
        self.architecture = architecture
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.pretrain_autoencoder = pretrain_autoencoder
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_dropout = pretrain_dropout
        self.pretrain_learning_rate = pretrain_learning_rate
        self.pretrain_batch_size = pretrain_batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _hidden_sizes(self) -> list[int]:
        if self.architecture == "mlp":
            return [128, 64, 32]
        if self.architecture == "1d_cnn":
            return [16, 32]
        raise ValueError("architecture must be 'mlp' or '1d_cnn'")

    def _layer_sizes(self, n_features: int) -> list[int]:
        return [n_features, *self._hidden_sizes(), self.latent_dim]

    def _initialize_parameters(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)

        if self.architecture == "mlp":
            layer_sizes = self._layer_sizes(n_features)
            self.weights_: list[np.ndarray] = []
            self.biases_: list[np.ndarray] = []
            self._adam_m_w_: list[np.ndarray] = []
            self._adam_v_w_: list[np.ndarray] = []
            self._adam_m_b_: list[np.ndarray] = []
            self._adam_v_b_: list[np.ndarray] = []

            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
                scale = np.sqrt(2.0 / max(in_dim, 1))
                weight = rng.normal(0.0, scale, size=(in_dim, out_dim))
                bias = np.zeros(out_dim, dtype=float)
                self.weights_.append(weight)
                self.biases_.append(bias)
                self._adam_m_w_.append(np.zeros_like(weight))
                self._adam_v_w_.append(np.zeros_like(weight))
                self._adam_m_b_.append(np.zeros_like(bias))
                self._adam_v_b_.append(np.zeros_like(bias))
        else:
            conv1_filters = 16
            conv1_kernel_size = 5
            conv2_filters = 32
            conv2_kernel_size = 3

            conv1_scale = np.sqrt(2.0 / max(conv1_kernel_size, 1))
            conv2_scale = np.sqrt(2.0 / max(conv1_filters * conv2_kernel_size, 1))
            dense_scale = np.sqrt(2.0 / max(conv2_filters, 1))

            self.conv1_weights_ = rng.normal(
                0.0,
                conv1_scale,
                size=(conv1_filters, conv1_kernel_size, 1),
            )
            self.conv1_biases_ = np.zeros(conv1_filters, dtype=float)
            self.conv2_weights_ = rng.normal(
                0.0,
                conv2_scale,
                size=(conv2_filters, conv2_kernel_size, conv1_filters),
            )
            self.conv2_biases_ = np.zeros(conv2_filters, dtype=float)
            self.dense_weights_ = rng.normal(0.0, dense_scale, size=(conv2_filters, self.latent_dim))
            self.dense_biases_ = np.zeros(self.latent_dim, dtype=float)

            self._adam_m_conv1_w_ = np.zeros_like(self.conv1_weights_)
            self._adam_v_conv1_w_ = np.zeros_like(self.conv1_weights_)
            self._adam_m_conv1_b_ = np.zeros_like(self.conv1_biases_)
            self._adam_v_conv1_b_ = np.zeros_like(self.conv1_biases_)
            self._adam_m_conv2_w_ = np.zeros_like(self.conv2_weights_)
            self._adam_v_conv2_w_ = np.zeros_like(self.conv2_weights_)
            self._adam_m_conv2_b_ = np.zeros_like(self.conv2_biases_)
            self._adam_v_conv2_b_ = np.zeros_like(self.conv2_biases_)
            self._adam_m_dense_w_ = np.zeros_like(self.dense_weights_)
            self._adam_v_dense_w_ = np.zeros_like(self.dense_weights_)
            self._adam_m_dense_b_ = np.zeros_like(self.dense_biases_)
            self._adam_v_dense_b_ = np.zeros_like(self.dense_biases_)

        self._adam_step_ = 0

    def _load_pretrained_encoder(self, pretrained: DeepAutoencoder) -> None:
        if self.architecture != "mlp":
            return
        encoder_weight_count = len(self.weights_)
        if len(pretrained.weights_) < encoder_weight_count:
            return

        for idx in range(encoder_weight_count):
            if pretrained.weights_[idx].shape == self.weights_[idx].shape:
                self.weights_[idx] = pretrained.weights_[idx].copy()
                self.biases_[idx] = pretrained.biases_[idx].copy()

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, _ForwardCache]:
        if self.architecture == "1d_cnn":
            inputs = X[:, :, None]
            conv1_pre, conv1_padded = _conv1d_same_forward(inputs, self.conv1_weights_, self.conv1_biases_)
            conv1_out = _relu(conv1_pre)
            conv2_pre, conv2_padded = _conv1d_same_forward(conv1_out, self.conv2_weights_, self.conv2_biases_)
            conv2_out = _relu(conv2_pre)
            pooled = conv2_out.mean(axis=1)
            latent = pooled @ self.dense_weights_ + self.dense_biases_
            return latent, _CNNCache(
                inputs=inputs,
                conv1_pre=conv1_pre,
                conv1_out=conv1_out,
                conv1_padded=conv1_padded,
                conv2_pre=conv2_pre,
                conv2_out=conv2_out,
                conv2_padded=conv2_padded,
                pooled=pooled,
            )

        activations: list[np.ndarray] = [X]
        pre_activations: list[np.ndarray] = []
        a = X

        for layer_idx in range(len(self.weights_) - 1):
            z = a @ self.weights_[layer_idx] + self.biases_[layer_idx]
            pre_activations.append(z)
            a = _relu(z)
            activations.append(a)

        latent = a @ self.weights_[-1] + self.biases_[-1]
        pre_activations.append(latent)
        return latent, _ForwardCache(activations, pre_activations)

    def _backward(self, latent: np.ndarray, cache: _ForwardCache) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.architecture == "1d_cnn":
            assert isinstance(cache, _CNNCache)
            m = latent.shape[0]
            grads_w: list[np.ndarray] = [
                np.zeros_like(self.conv1_weights_),
                np.zeros_like(self.conv2_weights_),
                np.zeros_like(self.dense_weights_),
            ]
            grads_b: list[np.ndarray] = [
                np.zeros_like(self.conv1_biases_),
                np.zeros_like(self.conv2_biases_),
                np.zeros_like(self.dense_biases_),
            ]

            delta = (2.0 / m) * (latent - self.center_)
            grads_w[2] = cache.pooled.T @ delta + self.weight_decay_ * self.dense_weights_
            grads_b[2] = delta.sum(axis=0)
            delta = delta @ self.dense_weights_.T
            delta = np.broadcast_to(delta[:, None, :], cache.conv2_out.shape) / cache.conv2_out.shape[1]
            delta = delta * _relu_grad(cache.conv2_pre)

            delta, grad_conv2_w, grad_conv2_b = _conv1d_same_backward(delta, cache.conv2_padded, self.conv2_weights_)
            grads_w[1] = grad_conv2_w + self.weight_decay_ * self.conv2_weights_
            grads_b[1] = grad_conv2_b

            delta = delta * _relu_grad(cache.conv1_pre)
            _, grad_conv1_w, grad_conv1_b = _conv1d_same_backward(delta, cache.conv1_padded, self.conv1_weights_)
            grads_w[0] = grad_conv1_w + self.weight_decay_ * self.conv1_weights_
            grads_b[0] = grad_conv1_b
            return grads_w, grads_b

        m = latent.shape[0]
        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights_]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases_]

        delta = (2.0 / m) * (latent - self.center_)
        grads_w[-1] = cache.activations[-1].T @ delta + self.weight_decay_ * self.weights_[-1]
        grads_b[-1] = delta.sum(axis=0)
        delta = delta @ self.weights_[-1].T

        for layer_idx in range(len(self.weights_) - 2, -1, -1):
            delta = delta * _relu_grad(cache.pre_activations[layer_idx])
            grads_w[layer_idx] = cache.activations[layer_idx].T @ delta + self.weight_decay_ * self.weights_[layer_idx]
            grads_b[layer_idx] = delta.sum(axis=0)
            if layer_idx > 0:
                delta = delta @ self.weights_[layer_idx].T

        return grads_w, grads_b

    def _apply_adam(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        if self.architecture == "1d_cnn":
            self._adam_step_ += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            def update(param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad**2)
                m_hat = m / (1.0 - beta1**self._adam_step_)
                v_hat = v / (1.0 - beta2**self._adam_step_)
                param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                return param, m, v

            self.conv1_weights_, self._adam_m_conv1_w_, self._adam_v_conv1_w_ = update(
                self.conv1_weights_, grads_w[0], self._adam_m_conv1_w_, self._adam_v_conv1_w_
            )
            self.conv1_biases_, self._adam_m_conv1_b_, self._adam_v_conv1_b_ = update(
                self.conv1_biases_, grads_b[0], self._adam_m_conv1_b_, self._adam_v_conv1_b_
            )
            self.conv2_weights_, self._adam_m_conv2_w_, self._adam_v_conv2_w_ = update(
                self.conv2_weights_, grads_w[1], self._adam_m_conv2_w_, self._adam_v_conv2_w_
            )
            self.conv2_biases_, self._adam_m_conv2_b_, self._adam_v_conv2_b_ = update(
                self.conv2_biases_, grads_b[1], self._adam_m_conv2_b_, self._adam_v_conv2_b_
            )
            self.dense_weights_, self._adam_m_dense_w_, self._adam_v_dense_w_ = update(
                self.dense_weights_, grads_w[2], self._adam_m_dense_w_, self._adam_v_dense_w_
            )
            self.dense_biases_, self._adam_m_dense_b_, self._adam_v_dense_b_ = update(
                self.dense_biases_, grads_b[2], self._adam_m_dense_b_, self._adam_v_dense_b_
            )
            return

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

    def _latent_distances(self, X: np.ndarray) -> np.ndarray:
        latent, _ = self._forward(X)
        return np.sum((latent - self.center_) ** 2, axis=1)

    def _update_score_stats(self, X: np.ndarray) -> None:
        raw_scores = self._latent_distances(X)
        self._training_raw_score_mean_ = float(np.mean(raw_scores))
        self._training_raw_score_std_ = float(np.std(raw_scores, ddof=0))
        if self._training_raw_score_std_ == 0.0 or np.isnan(self._training_raw_score_std_):
            self._training_raw_score_std_ = 1.0

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected a 2D array-like input.")
        if X.shape[0] < 2:
            raise ValueError("DeepSVDD requires at least two samples.")

        self.n_features_in_ = X.shape[1]
        self.weight_decay_ = 1e-6

        if self.pretrain_autoencoder:
            autoencoder = DeepAutoencoder(
                latent_dim=self.latent_dim,
                dropout=self.pretrain_dropout,
                learning_rate=self.pretrain_learning_rate,
                batch_size=self.pretrain_batch_size,
                threshold_percentile=97.5,
                validation_fraction=self.validation_fraction,
                max_epochs=self.pretrain_epochs,
                patience=max(3, self.pretrain_epochs // 4),
                l2=1e-5,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            autoencoder.fit(X)
            self.pretrained_autoencoder_ = autoencoder
        else:
            self.pretrained_autoencoder_ = None

        self._initialize_parameters(self.n_features_in_)
        if self.pretrained_autoencoder_ is not None and self.architecture == "mlp":
            self._load_pretrained_encoder(self.pretrained_autoencoder_)

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

        latent_train, _ = self._forward(X_train)
        self.center_ = latent_train.mean(axis=0)
        self.center_ = np.where(np.abs(self.center_) < 1e-6, np.sign(self.center_) * 1e-6 + 1e-6, self.center_)
        if self.center_fixed:
            self.fixed_center_ = self.center_.copy()

        batch_size = max(1, min(self.batch_size, X_train.shape[0]))
        rng = np.random.default_rng(self.random_state)
        best_state: dict[str, Any] | None = None
        best_val = np.inf
        patience_counter = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            indices = rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                batch = X_train[indices[start : start + batch_size]]
                latent, cache = self._forward(batch)
                grads_w, grads_b = self._backward(latent, cache)
                self._apply_adam(grads_w, grads_b)

            if not self.center_fixed:
                latent_train, _ = self._forward(X_train)
                self.center_ = latent_train.mean(axis=0)

            train_dist = self._latent_distances(X_train)
            val_dist = self._latent_distances(X_val)
            train_loss = float(np.mean(train_dist))
            val_loss = float(np.mean(val_dist))
            self.history_.append({"epoch": float(epoch), "train_distance": train_loss, "val_distance": val_loss})

            if self.verbose:
                print(f"epoch={epoch} train_distance={train_loss:.6f} val_distance={val_loss:.6f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                if self.architecture == "1d_cnn":
                    best_state = {
                        "conv1_weights": self.conv1_weights_.copy(),
                        "conv1_biases": self.conv1_biases_.copy(),
                        "conv2_weights": self.conv2_weights_.copy(),
                        "conv2_biases": self.conv2_biases_.copy(),
                        "dense_weights": self.dense_weights_.copy(),
                        "dense_biases": self.dense_biases_.copy(),
                        "center": self.center_.copy(),
                    }
                else:
                    best_state = {
                        "weights": [weight.copy() for weight in self.weights_],
                        "biases": [bias.copy() for bias in self.biases_],
                        "center": self.center_.copy(),
                    }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max(3, self.max_epochs // 6):
                    break

        if best_state is not None:
            if self.architecture == "1d_cnn":
                self.conv1_weights_ = best_state["conv1_weights"].copy()
                self.conv1_biases_ = best_state["conv1_biases"].copy()
                self.conv2_weights_ = best_state["conv2_weights"].copy()
                self.conv2_biases_ = best_state["conv2_biases"].copy()
                self.dense_weights_ = best_state["dense_weights"].copy()
                self.dense_biases_ = best_state["dense_biases"].copy()
            else:
                self.weights_ = [weight.copy() for weight in best_state["weights"]]
                self.biases_ = [bias.copy() for bias in best_state["biases"]]
            self.center_ = best_state["center"].copy()

        train_dist = self._latent_distances(X_train)
        val_dist = self._latent_distances(X_val)
        self.radius_ = float(np.percentile(val_dist, 100.0 * (1.0 - self.nu)))
        self.training_distance_ = float(np.mean(train_dist))
        self.validation_distance_ = float(np.mean(val_dist))
        self._update_score_stats(X)
        self.fitted_ = True
        return self

    def latent_distance(self, X: Any) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("DeepSVDD must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        return self._latent_distances(X)

    def latent_embedding(self, X: Any) -> np.ndarray:
        """Return the latent representation used by the hypersphere head."""

        if not getattr(self, "fitted_", False):
            raise RuntimeError("DeepSVDD must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        latent, _ = self._forward(X)
        return latent

    def raw_score(self, X: Any) -> np.ndarray:
        return self.latent_distance(X)

    def score(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_training_raw_score_mean_"):
            raise RuntimeError("DeepSVDD must be fit before scoring.")
        raw_scores = self.raw_score(X)
        return (raw_scores - self._training_raw_score_mean_) / self._training_raw_score_std_

    def score_samples(self, X):
        return -self.raw_score(X)

    def decision_function(self, X):
        return self.radius_ - self.raw_score(X)

    def predict(self, X):
        distances = self.latent_distance(X)
        return np.where(distances <= self.radius_, 1, -1)
