"""Temporal sequence anomaly detector for visit windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin


def _ensure_3d_array(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 3:
        raise ValueError("Expected a 3D array-like input with shape (samples, time, features).")
    return array


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


@dataclass
class _ForwardCache:
    x_padded: np.ndarray
    conv_pre: np.ndarray
    conv_post: np.ndarray
    conv_dropout_mask: np.ndarray | None
    flat: np.ndarray
    latent_pre: np.ndarray
    latent_post: np.ndarray
    decoder_pre: np.ndarray
    decoder_post: np.ndarray
    decoder_dropout_mask: np.ndarray | None
    recon: np.ndarray


class TemporalConvolutionalSequenceDetector(BaseEstimator, OutlierMixin):
    """A compact TCN-style autoencoder over sliding visit windows."""

    def __init__(
        self,
        *,
        window_size: int = 4,
        filters: int = 16,
        kernel_size: int = 3,
        latent_dim: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 80,
        patience: int = 10,
        l2: float = 1e-5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.window_size = window_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_window_size(self, time_steps: int) -> int:
        return max(1, min(int(self.window_size), int(time_steps)))

    def _resolve_kernel_size(self, time_steps: int) -> int:
        return max(1, min(int(self.kernel_size), int(time_steps)))

    def _resolve_filters(self, time_steps: int) -> int:
        return max(1, min(int(self.filters), max(1, int(time_steps) * 2)))

    def _same_padding(self, kernel_size: int) -> tuple[int, int]:
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        return pad_left, pad_right

    def _initialize_parameters(self, time_steps: int, feature_dim: int) -> None:
        self.window_size_ = self._resolve_window_size(time_steps)
        self.kernel_size_ = self._resolve_kernel_size(self.window_size_)
        self.filters_ = self._resolve_filters(self.window_size_)
        self.pad_left_, self.pad_right_ = self._same_padding(self.kernel_size_)

        flat_dim = self.window_size_ * self.filters_
        self.decoder_hidden_dim_ = max(32, min(128, max(self.latent_dim * 4, flat_dim // 2)))

        rng = np.random.default_rng(self.random_state)
        conv_scale = np.sqrt(2.0 / max(1, self.kernel_size_ * feature_dim))
        dense_scale = {
            "enc": np.sqrt(2.0 / max(1, flat_dim)),
            "dec": np.sqrt(2.0 / max(1, self.latent_dim)),
            "out": np.sqrt(2.0 / max(1, self.decoder_hidden_dim_)),
        }

        self.conv_filters_ = rng.normal(0.0, conv_scale, size=(self.filters_, self.kernel_size_, feature_dim))
        self.conv_bias_ = np.zeros(self.filters_, dtype=float)
        self.encoder_weight_ = rng.normal(0.0, dense_scale["enc"], size=(flat_dim, self.latent_dim))
        self.encoder_bias_ = np.zeros(self.latent_dim, dtype=float)
        self.decoder_weight_ = rng.normal(0.0, dense_scale["dec"], size=(self.latent_dim, self.decoder_hidden_dim_))
        self.decoder_bias_ = np.zeros(self.decoder_hidden_dim_, dtype=float)
        self.output_weight_ = rng.normal(0.0, dense_scale["out"], size=(self.decoder_hidden_dim_, self.window_size_ * feature_dim))
        self.output_bias_ = np.zeros(self.window_size_ * feature_dim, dtype=float)

        self._adam_state_ = {
            "step": 0,
            "m": {
                "conv_filters": np.zeros_like(self.conv_filters_),
                "conv_bias": np.zeros_like(self.conv_bias_),
                "encoder_weight": np.zeros_like(self.encoder_weight_),
                "encoder_bias": np.zeros_like(self.encoder_bias_),
                "decoder_weight": np.zeros_like(self.decoder_weight_),
                "decoder_bias": np.zeros_like(self.decoder_bias_),
                "output_weight": np.zeros_like(self.output_weight_),
                "output_bias": np.zeros_like(self.output_bias_),
            },
            "v": {
                "conv_filters": np.zeros_like(self.conv_filters_),
                "conv_bias": np.zeros_like(self.conv_bias_),
                "encoder_weight": np.zeros_like(self.encoder_weight_),
                "encoder_bias": np.zeros_like(self.encoder_bias_),
                "decoder_weight": np.zeros_like(self.decoder_weight_),
                "decoder_bias": np.zeros_like(self.decoder_bias_),
                "output_weight": np.zeros_like(self.output_weight_),
                "output_bias": np.zeros_like(self.output_bias_),
            },
        }

    def _apply_dropout(
        self,
        activations: np.ndarray,
        *,
        training: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not training or self.dropout <= 0.0:
            return activations, None
        keep_prob = 1.0 - float(self.dropout)
        if keep_prob <= 0.0:
            raise ValueError("dropout must be less than 1.0")
        mask = (rng.random(activations.shape) < keep_prob).astype(float) / keep_prob
        return activations * mask, mask

    def _temporal_convolution(self, x_padded: np.ndarray) -> np.ndarray:
        batch_size, padded_time_steps, feature_dim = x_padded.shape
        time_steps = padded_time_steps - self.kernel_size_ + 1
        conv = np.zeros((batch_size, time_steps, self.filters_), dtype=float)
        for kernel_index in range(self.kernel_size_):
            window = x_padded[:, kernel_index : kernel_index + time_steps, :]
            conv += np.einsum("btd,fd->btf", window, self.conv_filters_[:, kernel_index, :])
        conv += self.conv_bias_
        return conv

    def _forward(self, X: np.ndarray, *, training: bool, rng: np.random.Generator | None = None) -> tuple[np.ndarray, _ForwardCache]:
        x = np.asarray(X, dtype=float)
        if x.ndim != 3:
            raise ValueError("TemporalConvolutionalSequenceDetector expects a 3D tensor.")

        if training and rng is None:
            raise ValueError("rng is required when training=True.")

        x_padded = np.pad(x, ((0, 0), (self.pad_left_, self.pad_right_), (0, 0)), mode="constant")
        conv_pre = self._temporal_convolution(x_padded)
        conv_post = _relu(conv_pre)

        rng = rng or np.random.default_rng(self.random_state)
        conv_post, conv_dropout_mask = self._apply_dropout(conv_post, training=training, rng=rng)

        flat = conv_post.reshape(x.shape[0], -1)
        latent_pre = flat @ self.encoder_weight_ + self.encoder_bias_
        latent_post = np.tanh(latent_pre)

        decoder_pre = latent_post @ self.decoder_weight_ + self.decoder_bias_
        decoder_post = _relu(decoder_pre)
        decoder_post, decoder_dropout_mask = self._apply_dropout(decoder_post, training=training, rng=rng)

        recon_flat = decoder_post @ self.output_weight_ + self.output_bias_
        recon = recon_flat.reshape(x.shape[0], self.window_size_, x.shape[2])

        cache = _ForwardCache(
            x_padded=x_padded,
            conv_pre=conv_pre,
            conv_post=conv_post,
            conv_dropout_mask=conv_dropout_mask,
            flat=flat,
            latent_pre=latent_pre,
            latent_post=latent_post,
            decoder_pre=decoder_pre,
            decoder_post=decoder_post,
            decoder_dropout_mask=decoder_dropout_mask,
            recon=recon,
        )
        return recon, cache

    def _backward(self, X: np.ndarray, cache: _ForwardCache) -> dict[str, np.ndarray]:
        x = np.asarray(X, dtype=float)
        batch_size, window_size, feature_dim = x.shape
        scale = 2.0 / max(1, batch_size * window_size * feature_dim)

        d_recon = scale * (cache.recon - x)
        d_recon_flat = d_recon.reshape(batch_size, -1)

        grad_output_weight = cache.decoder_post.T @ d_recon_flat + self.l2 * self.output_weight_
        grad_output_bias = d_recon_flat.sum(axis=0)
        d_decoder_post = d_recon_flat @ self.output_weight_.T
        if cache.decoder_dropout_mask is not None:
            d_decoder_post *= cache.decoder_dropout_mask
        d_decoder_pre = d_decoder_post * _relu_grad(cache.decoder_pre)

        grad_decoder_weight = cache.latent_post.T @ d_decoder_pre + self.l2 * self.decoder_weight_
        grad_decoder_bias = d_decoder_pre.sum(axis=0)
        d_latent_post = d_decoder_pre @ self.decoder_weight_.T
        d_latent_pre = d_latent_post * (1.0 - np.square(cache.latent_post))

        grad_encoder_weight = cache.flat.T @ d_latent_pre + self.l2 * self.encoder_weight_
        grad_encoder_bias = d_latent_pre.sum(axis=0)
        d_flat = d_latent_pre @ self.encoder_weight_.T
        d_conv_post = d_flat.reshape(cache.conv_post.shape)
        if cache.conv_dropout_mask is not None:
            d_conv_post *= cache.conv_dropout_mask
        d_conv_pre = d_conv_post * _relu_grad(cache.conv_pre)

        grad_conv_bias = d_conv_pre.sum(axis=(0, 1))
        grad_conv_filters = np.zeros_like(self.conv_filters_)
        time_steps = x.shape[1]
        for kernel_index in range(self.kernel_size_):
            window = cache.x_padded[:, kernel_index : kernel_index + time_steps, :]
            grad_conv_filters[:, kernel_index, :] = np.einsum("btf,btd->fd", d_conv_pre, window) + self.l2 * self.conv_filters_[:, kernel_index, :]

        return {
            "conv_filters": grad_conv_filters,
            "conv_bias": grad_conv_bias,
            "encoder_weight": grad_encoder_weight,
            "encoder_bias": grad_encoder_bias,
            "decoder_weight": grad_decoder_weight,
            "decoder_bias": grad_decoder_bias,
            "output_weight": grad_output_weight,
            "output_bias": grad_output_bias,
        }

    def _apply_adam(self, grads: dict[str, np.ndarray]) -> None:
        state = self._adam_state_
        state["step"] += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = state["step"]

        for name, grad in grads.items():
            m = state["m"][name]
            v = state["v"][name]
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * np.square(grad)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            getattr(self, f"{name}_")[...] -= update

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        recon, _ = self._forward(X, training=False)
        return np.mean(np.square(recon - np.asarray(X, dtype=float)), axis=(1, 2))

    def _update_score_stats(self, X: np.ndarray) -> None:
        raw_scores = self._reconstruction_error(X)
        self._training_raw_score_mean_ = float(np.mean(raw_scores))
        self._training_raw_score_std_ = float(np.std(raw_scores, ddof=0))
        if self._training_raw_score_std_ == 0.0 or np.isnan(self._training_raw_score_std_):
            self._training_raw_score_std_ = 1.0

    def fit(self, X: Any, y=None):
        X = _ensure_3d_array(X)
        if X.shape[0] == 0:
            raise ValueError("TemporalConvolutionalSequenceDetector requires at least one window.")

        self._initialize_parameters(X.shape[1], X.shape[2])

        if 0.0 < self.batch_size:
            batch_size = max(1, min(int(self.batch_size), X.shape[0]))
        else:
            batch_size = max(1, X.shape[0])

        if X.shape[0] >= 8:
            split_index = max(1, int(round(X.shape[0] * (1.0 - min(max(0.2, 1.0 / max(X.shape[0], 5)), 0.5)))))
            split_index = min(split_index, X.shape[0] - 1)
            X_train = X[:split_index]
            X_val = X[split_index:]
            if X_val.shape[0] == 0:
                X_train = X
                X_val = X
        else:
            X_train = X
            X_val = X

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
                recon, cache = self._forward(batch_X, training=True, rng=rng)
                grads = self._backward(batch_X, cache)
                self._apply_adam(grads)

            train_loss = float(np.mean(self._reconstruction_error(X_train)))
            val_loss = float(np.mean(self._reconstruction_error(X_val)))
            self.history_.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if self.verbose:
                print(f"[TemporalSequenceDetector] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    "conv_filters": self.conv_filters_.copy(),
                    "conv_bias": self.conv_bias_.copy(),
                    "encoder_weight": self.encoder_weight_.copy(),
                    "encoder_bias": self.encoder_bias_.copy(),
                    "decoder_weight": self.decoder_weight_.copy(),
                    "decoder_bias": self.decoder_bias_.copy(),
                    "output_weight": self.output_weight_.copy(),
                    "output_bias": self.output_bias_.copy(),
                }
                stalled_epochs = 0
            else:
                stalled_epochs += 1
                if stalled_epochs >= int(self.patience):
                    break

        if best_state is not None:
            self.conv_filters_ = best_state["conv_filters"].copy()
            self.conv_bias_ = best_state["conv_bias"].copy()
            self.encoder_weight_ = best_state["encoder_weight"].copy()
            self.encoder_bias_ = best_state["encoder_bias"].copy()
            self.decoder_weight_ = best_state["decoder_weight"].copy()
            self.decoder_bias_ = best_state["decoder_bias"].copy()
            self.output_weight_ = best_state["output_weight"].copy()
            self.output_bias_ = best_state["output_bias"].copy()

        self._update_score_stats(X)
        self.threshold_ = float(np.quantile(self._reconstruction_error(X), 0.975))
        self.fitted_ = True
        return self

    def reconstruction_error(self, X: Any) -> np.ndarray:
        if not hasattr(self, "conv_filters_"):
            raise RuntimeError("TemporalConvolutionalSequenceDetector must be fit before scoring.")
        return self._reconstruction_error(_ensure_3d_array(X))

    def raw_score(self, X: Any) -> np.ndarray:
        return self.score(X)

    def score(self, X: Any) -> np.ndarray:
        return self.reconstruction_error(X)

    def score_samples(self, X: Any) -> np.ndarray:
        return -self.score(X)

    def decision_function(self, X: Any) -> np.ndarray:
        return self.threshold_ - self.score(X)

    def predict(self, X: Any) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)
