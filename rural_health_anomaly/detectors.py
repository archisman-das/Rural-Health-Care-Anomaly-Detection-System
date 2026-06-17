"""Standalone anomaly detector classes with a shared fit/score interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _ensure_2d_array(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array-like input.")
    return array


def _normalize_scores(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=0))
    if std == 0.0 or np.isnan(std):
        std = 1.0
    return (scores - mean) / std, mean, std


class _BaseNormalizedAnomalyModel(BaseEstimator):
    """Base class for anomaly models that expose normalized `score()` values."""

    def _set_score_stats(self, raw_scores: np.ndarray) -> None:
        normalized, mean, std = _normalize_scores(raw_scores)
        self._training_raw_score_mean_ = mean
        self._training_raw_score_std_ = std
        self._training_normalized_score_ = normalized

    def _require_score_stats(self) -> None:
        if not hasattr(self, "_training_raw_score_mean_"):
            raise RuntimeError("Model must be fit before scoring.")

    def score(self, X) -> np.ndarray:
        self._require_score_stats()
        raw_scores = self.raw_score(X)
        return (raw_scores - self._training_raw_score_mean_) / self._training_raw_score_std_

    def score_samples(self, X) -> np.ndarray:
        return -self.raw_score(X)

    def decision_function(self, X) -> np.ndarray:
        return -self.raw_score(X)

    def predict(self, X) -> np.ndarray:
        return np.where(self.score(X) >= 0.0, -1, 1)


class IsolationForestAnomalyModel(_BaseNormalizedAnomalyModel):
    def __init__(
        self,
        *,
        n_estimators: int = 300,
        contamination: float = 0.05,
        max_samples: int | str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self._set_score_stats(self.raw_score(X))
        return self

    def raw_score(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model must be fit before scoring.")
        X = _ensure_2d_array(X)
        return -np.asarray(self.model_.decision_function(X), dtype=float)


class OneClassSVMAnomalyModel(_BaseNormalizedAnomalyModel):
    def __init__(
        self,
        *,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str | float = "scale",
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        self.model_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model_.fit(X)
        self._set_score_stats(self.raw_score(X))
        return self

    def raw_score(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model must be fit before scoring.")
        X = _ensure_2d_array(X)
        return -np.asarray(self.model_.decision_function(X), dtype=float)


class LocalOutlierFactorAnomalyModel(_BaseNormalizedAnomalyModel):
    def __init__(
        self,
        *,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = _ensure_2d_array(X)
        max_neighbors = max(1, X.shape[0] - 1)
        self.model_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, max_neighbors),
            contamination=self.contamination,
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self._set_score_stats(self.raw_score(X))
        return self

    def raw_score(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model must be fit before scoring.")
        X = _ensure_2d_array(X)
        return -np.asarray(self.model_.decision_function(X), dtype=float)
