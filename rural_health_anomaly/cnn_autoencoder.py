"""1D CNN autoencoder anomaly detector for tabular health features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, OutlierMixin


@dataclass
class _ForwardCache:
    x_padded: np.ndarray
    windows: np.ndarray
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


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


class CNNAutoencoder(BaseEstimator, OutlierMixin):
    """A compact 1D CNN autoencoder with reconstruction-threshold scoring."""

    def __init__(
        self,
        *,
        filters: int = 8,
        kernel_size: int = 3,
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
        self.filters = filters
        self.kernel_size = kernel_size
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

    def _resolve_kernel_size(self, n_features: int) -> int:
        return max(1, min(int(self.kernel_size), int(n_features)))

    def _resolve_filters(self, n_features: int) -> int:
        return max(1, min(int(self.filters), max(1, int(n_features))))

    def _same_padding(self, kernel_size: int) -> tuple[int, int]:
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        return pad_left, pad_right

    def _initialize_parameters(self, n_features: int) -> None:
        self.kernel_size_ = self._resolve_kernel_size(n_features)
        self.filters_ = self._resolve_filters(n_features)
        self.pad_left_, self.pad_right_ = self._same_padding(self.kernel_size_)

        flat_dim = n_features * self.filters_
        self.decoder_hidden_dim_ = max(32, min(128, max(self.latent_dim * 4, flat_dim // 2)))

        rng = np.random.default_rng(self.random_state)
        conv_scale = np.sqrt(2.0 / max(1, self.kernel_size_))
        dense_scale = {
            "enc": np.sqrt(2.0 / max(1, flat_dim)),
            "dec": np.sqrt(2.0 / max(1, self.latent_dim)),
            "out": np.sqrt(2.0 / max(1, self.decoder_hidden_dim_)),
        }

        self.conv_filters_ = rng.normal(0.0, conv_scale, size=(self.filters_, self.kernel_size_))
        self.conv_bias_ = np.zeros(self.filters_, dtype=float)
        self.encoder_weight_ = rng.normal(0.0, dense_scale["enc"], size=(flat_dim, self.latent_dim))
        self.encoder_bias_ = np.zeros(self.latent_dim, dtype=float)
        self.decoder_weight_ = rng.normal(0.0, dense_scale["dec"], size=(self.latent_dim, self.decoder_hidden_dim_))
        self.decoder_bias_ = np.zeros(self.decoder_hidden_dim_, dtype=float)
        self.output_weight_ = rng.normal(0.0, dense_scale["out"], size=(self.decoder_hidden_dim_, n_features))
        self.output_bias_ = np.zeros(n_features, dtype=float)

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

    def _apply_dropout(self, activations: np.ndarray, *, training: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        if not training or self.dropout <= 0.0:
            return activations, None
        keep_prob = 1.0 - float(self.dropout)
        if keep_prob <= 0.0:
            raise ValueError("dropout must be less than 1.0")
        mask = (rng.random(activations.shape) < keep_prob).astype(float) / keep_prob
        return activations * mask, mask

    def _conv_forward(self, x_padded: np.ndarray) -> np.ndarray:
        windows = sliding_window_view(x_padded[:, :, 0], window_shape=self.kernel_size_, axis=1)
        # windows shape: (n_samples, n_features, kernel_size)
        return np.einsum("bnk,fk->bnf", windows, self.conv_filters_) + self.conv_bias_

    def _forward(self, X: np.ndarray, *, training: bool) -> tuple[np.ndarray, _ForwardCache]:
        x = np.asarray(X, dtype=float)
        x_padded = np.pad(x[:, :, None], ((0, 0), (self.pad_left_, self.pad_right_), (0, 0)), mode="constant")
        windows = sliding_window_view(x_padded[:, :, 0], window_shape=self.kernel_size_, axis=1)
        conv_pre = np.einsum("bnk,fk->bnf", windows, self.conv_filters_) + self.conv_bias_
        conv_post = _relu(conv_pre)

        rng = getattr(self, "_rng", np.random.default_rng(self.random_state))
        conv_post, conv_dropout_mask = self._apply_dropout(conv_post, training=training, rng=rng)

        flat = conv_post.reshape(x.shape[0], -1)
        latent_pre = flat @ self.encoder_weight_ + self.encoder_bias_
        latent_post = np.tanh(latent_pre)

        decoder_pre = latent_post @ self.decoder_weight_ + self.decoder_bias_
        decoder_post = _relu(decoder_pre)
        decoder_post, decoder_dropout_mask = self._apply_dropout(decoder_post, training=training, rng=rng)

        recon = decoder_post @ self.output_weight_ + self.output_bias_
        cache = _ForwardCache(
            x_padded=x_padded,
            windows=windows,
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
        batch_size = x.shape[0]
        n_features = x.shape[1]
        scale = 2.0 / max(1, batch_size * n_features)

        d_recon = scale * (cache.recon - x)

        grad_output_weight = cache.decoder_post.T @ d_recon + self.l2 * self.output_weight_
        grad_output_bias = d_recon.sum(axis=0)
        d_decoder_post = d_recon @ self.output_weight_.T
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
        grad_conv_filters = np.einsum("bnf,bnk->fk", d_conv_pre, cache.windows) + self.l2 * self.conv_filters_

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
        return np.mean(np.square(recon - np.asarray(X, dtype=float)), axis=1)

    def _update_score_stats(self, X: np.ndarray) -> None:
        raw_scores = self._reconstruction_error(X)
        self._training_raw_score_mean_ = float(np.mean(raw_scores))
        self._training_raw_score_std_ = float(np.std(raw_scores, ddof=0))
        if self._training_raw_score_std_ == 0.0 or np.isnan(self._training_raw_score_std_):
            self._training_raw_score_std_ = 1.0

    def reconstruction_error(self, X: Any) -> np.ndarray:
        if not hasattr(self, "conv_filters_"):
            raise RuntimeError("CNN autoencoder must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        return self._reconstruction_error(X)

    def reconstruction_residuals(self, X: Any) -> np.ndarray:
        if not hasattr(self, "conv_filters_"):
            raise RuntimeError("CNN autoencoder must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        recon, _ = self._forward(X, training=False)
        return recon - X

    def raw_score(self, X: Any) -> np.ndarray:
        return self.reconstruction_error(X)

    def score(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_training_raw_score_mean_"):
            raise RuntimeError("CNN autoencoder must be fit before scoring.")
        raw_scores = self.raw_score(X)
        return (raw_scores - self._training_raw_score_mean_) / self._training_raw_score_std_

    def decision_function(self, X):
        return self.threshold_ - self.raw_score(X)

    def score_samples(self, X):
        return -self.raw_score(X)

    def predict(self, X):
        errors = self.reconstruction_error(X)
        return np.where(errors <= self.threshold_, 1, -1)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected a 2D array-like input.")
        if X.shape[0] < 2:
            raise ValueError("CNN autoencoder requires at least two samples.")

        self.n_features_in_ = X.shape[1]
        self._initialize_parameters(self.n_features_in_)
        self._rng = np.random.default_rng(self.random_state)

        validation_fraction = float(self.validation_fraction)
        if not 0.0 < validation_fraction < 1.0:
            validation_fraction = 0.2

        indices = self._rng.permutation(X.shape[0])
        validation_size = int(round(X.shape[0] * validation_fraction))
        validation_size = min(max(validation_size, 1), X.shape[0] - 1)
        train_indices = indices[validation_size:]
        val_indices = indices[:validation_size]
        X_train = X[train_indices]
        X_val = X[val_indices]

        batch_size = max(1, min(int(self.batch_size), X_train.shape[0]))
        best_state: dict[str, np.ndarray] | None = None
        best_val_loss = float("inf")
        best_epoch = -1
        no_improve = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(1, int(self.max_epochs) + 1):
            shuffled = self._rng.permutation(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                batch_idx = shuffled[start : start + batch_size]
                batch = X_train[batch_idx]
                recon, cache = self._forward(batch, training=True)
                grads = self._backward(batch, cache)
                self._apply_adam(grads)

            train_loss = float(np.mean(self._reconstruction_error(X_train)))
            val_loss = float(np.mean(self._reconstruction_error(X_val)))
            self.history_.append({"epoch": float(epoch), "train_mse": train_loss, "val_mse": val_loss})

            if self.verbose:
                print(f"[CNN-AE] epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
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
            else:
                no_improve += 1
                if no_improve >= int(self.patience):
                    break

        if best_state is not None:
            self.conv_filters_ = best_state["conv_filters"]
            self.conv_bias_ = best_state["conv_bias"]
            self.encoder_weight_ = best_state["encoder_weight"]
            self.encoder_bias_ = best_state["encoder_bias"]
            self.decoder_weight_ = best_state["decoder_weight"]
            self.decoder_bias_ = best_state["decoder_bias"]
            self.output_weight_ = best_state["output_weight"]
            self.output_bias_ = best_state["output_bias"]

        validation_errors = self._reconstruction_error(X_val)
        self.threshold_ = float(np.percentile(validation_errors, self.threshold_percentile))
        self.validation_mse_ = float(np.mean(validation_errors))
        self.best_epoch_ = best_epoch
        self._update_score_stats(X)
        self.fitted_ = True
        return self
