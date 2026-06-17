"""Training helpers for the rural health anomaly pipeline."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import PreprocessingConfig
from .pipeline import build_anomaly_pipeline
from .temporal_sequence_detector import TemporalConvolutionalSequenceDetector
from .schema import (
    SCHEMA_LIST_NUMERIC_FEATURES,
    SCHEMA_MULTI_VALUE_FEATURES,
)

_DEFAULT_DATETIME_COLUMNS = ("recorded_at", "specimen_time")
_LIST_COLUMNS = tuple(SCHEMA_MULTI_VALUE_FEATURES + SCHEMA_LIST_NUMERIC_FEATURES)
_DEFAULT_RISK_SCORING_WEIGHTS = {
    "anomaly": 0.20,
    "vitals": 0.35,
    "labs": 0.30,
    "access": 0.15,
}
_CONFORMAL_ALPHA = 0.05
_CONFORMAL_RECONSTRUCTION_COMPONENTS = (
    "autoencoder",
    "variational_autoencoder",
    "ganomaly",
    "anomaly_transformer",
    "cnn_autoencoder",
)
_POINT_ANOMALY_ZSCORE_NORMALIZER = 3.0
_SPLIT_FILENAMES = ("data.csv", "data.parquet")


def _risk_category_from_score(score: float) -> str:
    """Map a normalized anomaly score to a clinical risk category."""

    if score < 0.3:
        return "Normal"
    if score < 0.6:
        return "Moderate"
    if score < 0.85:
        return "High"
    return "Critical"


def _generate_risk_score(score: float) -> float:
    """Convert a normalized anomaly score into a 0-100 risk score."""

    return float(round(max(0.0, min(1.0, score)) * 100.0, 1))


def _resolve_risk_scoring_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Normalize a configurable risk scoring weight mapping."""

    if not weights:
        return dict(_DEFAULT_RISK_SCORING_WEIGHTS)

    resolved = {name: float(weights.get(name, default)) for name, default in _DEFAULT_RISK_SCORING_WEIGHTS.items()}
    total = float(sum(value for value in resolved.values() if pd.notna(value)))
    if total <= 0.0:
        return dict(_DEFAULT_RISK_SCORING_WEIGHTS)
    return {name: float(value / total) for name, value in resolved.items()}


def _clamp_unit_interval(value: Any, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return float(max(0.0, min(1.0, numeric)))


def _scale_to_unit_interval(value: Any, *, anchor: float, span: float, invert: bool = False) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    if span <= 0:
        return None
    if invert:
        return _clamp_unit_interval((anchor - numeric) / span, default=None)
    return _clamp_unit_interval((numeric - anchor) / span, default=None)


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not present:
        return None
    return float(sum(present) / len(present))


def _parse_trend_days(value: Any) -> float | None:
    series = _coerce_list_cell(value)
    if not isinstance(series, list):
        return _safe_float(series)
    numbers = [_safe_float(item) for item in series]
    return _mean_present(numbers)


def _clinical_risk_component(
    row: pd.Series | dict[str, Any],
    *,
    anomaly_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Blend anomaly output with vitals, labs, and access barriers."""

    resolved_weights = _resolve_risk_scoring_weights(weights)
    anomaly_weight = resolved_weights["anomaly"]
    vitals_weight = resolved_weights["vitals"]
    labs_weight = resolved_weights["labs"]
    access_weight = resolved_weights["access"]

    def pick(*keys: str) -> Any:
        for key in keys:
            if isinstance(row, pd.Series):
                if key in row.index:
                    value = row[key]
                    if value is None:
                        continue
                    if isinstance(value, float) and pd.isna(value):
                        continue
                    return value
            else:
                value = row.get(key)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    return value
        return None

    vitals = _mean_present([
        _scale_to_unit_interval(pick("heart_rate_bpm", "heart_rate"), anchor=75.0, span=45.0),
        _scale_to_unit_interval(pick("systolic_bp_mmhg", "systolic_blood_pressure"), anchor=120.0, span=50.0),
        _scale_to_unit_interval(pick("diastolic_bp_mmhg", "diastolic_blood_pressure"), anchor=80.0, span=30.0),
        _scale_to_unit_interval(pick("spo2_percent", "spo2"), anchor=97.0, span=15.0, invert=True),
        _scale_to_unit_interval(pick("body_temperature_c", "body_temperature"), anchor=37.0, span=2.0),
        _scale_to_unit_interval(pick("respiratory_rate_bpm", "respiratory_rate"), anchor=16.0, span=10.0),
        _scale_to_unit_interval(pick("bmi_kg_m2", "bmi"), anchor=22.0, span=18.0),
    ])

    labs = _mean_present([
        _scale_to_unit_interval(pick("glucose_fasting_mg_dl", "fasting_glucose"), anchor=100.0, span=120.0),
        _scale_to_unit_interval(pick("glucose_postprandial_mg_dl", "postprandial_glucose"), anchor=140.0, span=160.0),
        _scale_to_unit_interval(pick("hba1c_percent", "hba1c"), anchor=5.7, span=4.3),
        _scale_to_unit_interval(pick("hemoglobin_g_dl", "hb_g_dl"), anchor=13.5, span=5.5, invert=True),
        _scale_to_unit_interval(pick("wbc_count_10e9_l", "wbc_count"), anchor=7.0, span=8.0),
        _scale_to_unit_interval(pick("platelets_10e9_l", "platelet_count"), anchor=250.0, span=180.0),
        _scale_to_unit_interval(pick("ldl_mg_dl", "ldl"), anchor=100.0, span=100.0),
        _scale_to_unit_interval(pick("hdl_mg_dl", "hdl"), anchor=50.0, span=40.0, invert=True),
        _scale_to_unit_interval(pick("triglycerides_mg_dl", "triglycerides"), anchor=150.0, span=250.0),
        _scale_to_unit_interval(pick("alt_u_l", "alt"), anchor=35.0, span=80.0),
        _scale_to_unit_interval(pick("ast_u_l", "ast"), anchor=35.0, span=80.0),
        _scale_to_unit_interval(pick("bilirubin_mg_dl", "bilirubin"), anchor=1.0, span=2.5),
        _scale_to_unit_interval(pick("creatinine_mg_dl", "creatinine"), anchor=1.0, span=1.8),
        _scale_to_unit_interval(pick("bun_mg_dl", "bun"), anchor=15.0, span=25.0),
        _scale_to_unit_interval(pick("egfr_ml_min_1_73m2", "egfr"), anchor=90.0, span=75.0, invert=True),
        _scale_to_unit_interval(pick("sodium_mmol_l", "sodium"), anchor=140.0, span=12.0),
        _scale_to_unit_interval(pick("potassium_mmol_l", "potassium"), anchor=4.2, span=2.0),
        _scale_to_unit_interval(pick("calcium_mg_dl", "calcium"), anchor=9.3, span=2.0),
    ])

    access = _mean_present([
        _scale_to_unit_interval(pick("visits_last_90_days", "visits_in_last_90_days"), anchor=0.0, span=8.0),
        _scale_to_unit_interval(pick("symptom_duration_days", "symptom_duration"), anchor=0.0, span=14.0),
        _scale_to_unit_interval(pick("distance_to_nearest_facility_km", "distance_to_facility_km"), anchor=0.0, span=20.0),
        _scale_to_unit_interval(pick("readmission_frequency", "readmission_count"), anchor=0.0, span=4.0),
        _scale_to_unit_interval(_parse_trend_days(pick("days_between_visits_trend")), anchor=30.0, span=30.0, invert=True),
        _scale_to_unit_interval(pick("sanitation_index"), anchor=1.0, span=1.0, invert=True),
        _scale_to_unit_interval(pick("drug_adherence_rate"), anchor=1.0, span=1.0, invert=True),
        _scale_to_unit_interval(pick("treatment_response_score"), anchor=1.0, span=1.0, invert=True),
    ])

    weighted_components: list[tuple[float, float]] = [(anomaly_weight, _clamp_unit_interval(anomaly_score, default=0.0) or 0.0)]
    if vitals is not None:
        weighted_components.append((vitals_weight, vitals))
    if labs is not None:
        weighted_components.append((labs_weight, labs))
    if access is not None:
        weighted_components.append((access_weight, access))

    total_weight = sum(weight for weight, _ in weighted_components)
    blended_score = sum(weight * value for weight, value in weighted_components) / total_weight if total_weight > 0 else 0.0
    return float(_generate_risk_score(blended_score))


def _estimate_object_size_bytes(obj: Any) -> int:
    """Estimate an object's memory footprint recursively."""

    seen: set[int] = set()

    def _walk(value: Any) -> int:
        object_id = id(value)
        if object_id in seen:
            return 0
        seen.add(object_id)

        size = sys.getsizeof(value)
        if hasattr(value, "nbytes"):
            try:
                size = max(size, int(value.nbytes))
            except Exception:
                pass

        if isinstance(value, dict):
            for key, item in value.items():
                size += _walk(key)
                size += _walk(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += _walk(item)
        elif hasattr(value, "__dict__"):
            size += _walk(vars(value))
        elif hasattr(value, "__slots__"):
            for slot in value.__slots__:  # type: ignore[attr-defined]
                if hasattr(value, slot):
                    size += _walk(getattr(value, slot))

        return int(size)

    return _walk(obj)


def _binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return simple binary classification metrics for threshold calibration."""

    true_positive = float(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = float(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = float(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positive + true_negative) / max(1.0, true_positive + true_negative + false_positive + false_negative)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _coerce_binary_labels(labels: Any) -> np.ndarray:
    """Normalize common binary label encodings into 0/1 integers."""

    array = np.asarray(labels)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.dtype == bool:
        return array.astype(int)

    unique_values = set(pd.unique(array).tolist())
    if unique_values.issubset({0, 1}):
        return array.astype(int)
    if unique_values.issubset({-1, 1}):
        return np.where(array == -1, 1, 0).astype(int)
    if unique_values.issubset({"0", "1"}):
        return array.astype(int)
    raise ValueError("Binary labels must use 0/1, -1/1, or boolean encoding.")


def _calibrate_threshold_from_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    candidate_count: int = 201,
) -> tuple[float, dict[str, float]]:
    """Find the score cutoff that best matches labeled validation data."""

    if scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must have the same length.")
    if scores.size == 0:
        raise ValueError("scores must not be empty.")

    thresholds = np.linspace(0.0, 1.0, max(3, int(candidate_count)), dtype=float)
    best_threshold = float(thresholds[0])
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "accuracy": 0.0}
    best_key = (-1.0, -1.0, -1.0, -1.0)

    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        metrics = _binary_classification_metrics(y_true, predicted)
        key = (metrics["f1"], metrics["accuracy"], metrics["precision"], float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def _collect_conformal_nonconformity_scores(model, transformed: Any) -> tuple[np.ndarray, str, list[str]]:
    """Return calibration-friendly nonconformity scores for conformal inference."""

    candidate_scores: list[np.ndarray] = []
    candidate_names: list[str] = []
    estimators = getattr(model, "estimators_", None)
    if isinstance(estimators, dict):
        for name in _CONFORMAL_RECONSTRUCTION_COMPONENTS:
            estimator = estimators.get(name)
            if estimator is None or not hasattr(estimator, "reconstruction_error"):
                continue
            try:
                scores = np.asarray(estimator.reconstruction_error(transformed), dtype=float).reshape(-1)
            except Exception:
                continue
            if scores.size:
                candidate_scores.append(scores)
                candidate_names.append(name)

    if candidate_scores:
        matrix = np.column_stack(candidate_scores)
        return np.mean(matrix, axis=1), "reconstruction_error", candidate_names

    if hasattr(model, "reconstruction_error"):
        try:
            scores = np.asarray(model.reconstruction_error(transformed), dtype=float).reshape(-1)
            if scores.size:
                return scores, "reconstruction_error", [type(model).__name__]
        except Exception:
            pass

    scores = np.asarray(model.raw_anomaly_score(transformed), dtype=float).reshape(-1)
    return scores, "raw_anomaly_score", [type(model).__name__]


def _project_latent_vectors(latent_vectors: np.ndarray, *, random_state: int = 42) -> tuple[np.ndarray, str]:
    latent_vectors = np.asarray(latent_vectors, dtype=float)
    if latent_vectors.ndim != 2 or latent_vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=float), "unavailable"
    if latent_vectors.shape[0] == 1:
        return np.zeros((1, 2), dtype=float), "single_point"
    if latent_vectors.shape[1] == 1:
        x = latent_vectors[:, 0]
        centered = x - np.mean(x)
        return np.column_stack([centered, np.zeros_like(centered)]), "1d_projection"

    try:  # pragma: no cover - optional dependency
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, latent_vectors.shape[0] - 1)),
            min_dist=0.15,
            metric="euclidean",
            random_state=random_state,
        )
        return np.asarray(reducer.fit_transform(latent_vectors), dtype=float), "umap"
    except Exception:
        pass

    if latent_vectors.shape[0] >= 3:
        perplexity = min(30, max(2, latent_vectors.shape[0] - 1))
        perplexity = min(perplexity, max(2, latent_vectors.shape[0] - 1))
        if perplexity < latent_vectors.shape[0]:
            try:
                reducer = TSNE(
                    n_components=2,
                    random_state=random_state,
                    init="pca",
                    learning_rate="auto",
                    perplexity=float(perplexity),
                )
                return np.asarray(reducer.fit_transform(latent_vectors), dtype=float), "tsne"
            except Exception:
                pass

    reducer = PCA(n_components=2, random_state=random_state)
    return np.asarray(reducer.fit_transform(latent_vectors), dtype=float), "pca"


def _compute_latent_manifold(
    pipeline,
    patient: dict[str, Any],
    *,
    max_background_rows: int = 48,
) -> dict[str, Any]:
    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    background_raw = getattr(pipeline, "explain_background_", frame)
    if isinstance(background_raw, pd.DataFrame) and not background_raw.empty:
        background_raw = background_raw.head(max_background_rows).copy()
    else:
        background_raw = frame.copy()

    manifold_frame = pd.concat([background_raw, frame], ignore_index=True)
    transformed = preprocessor.transform(manifold_frame)
    scored = score_records(pipeline, manifold_frame)
    features = list(manifold_frame.columns)
    estimators = getattr(model, "estimators_", {})
    vae = estimators.get("variational_autoencoder") if isinstance(estimators, dict) else None
    deep_svdd = estimators.get("deep_svdd") if isinstance(estimators, dict) else None

    if vae is not None and hasattr(vae, "latent_embedding"):
        latent_vectors = np.asarray(vae.latent_embedding(transformed), dtype=float)
    else:
        latent_vectors = np.asarray(transformed, dtype=float)

    projected, projection_method = _project_latent_vectors(latent_vectors, random_state=int(getattr(model, "stacking_random_state", 42)))
    if projected.shape[0] != len(manifold_frame):
        projected = np.zeros((len(manifold_frame), 2), dtype=float)

    deep_svdd_distances = np.asarray(scored.get("deep_svdd_distance", np.zeros(len(manifold_frame), dtype=float)), dtype=float).reshape(-1)
    deep_svdd_radius = float(getattr(deep_svdd, "radius_", np.nan)) if deep_svdd is not None else float("nan")
    center = projected.mean(axis=0) if projected.size else np.zeros(2, dtype=float)
    projected_distances = np.linalg.norm(projected - center, axis=1) if projected.size else np.zeros(len(manifold_frame), dtype=float)
    boundary_radius = float(np.quantile(projected_distances[:-1] if len(projected_distances) > 1 else projected_distances, 0.8)) if projected_distances.size else 0.0
    boundary_points = np.where(
        np.isfinite(deep_svdd_distances)
        & np.isfinite(deep_svdd_radius)
        & (np.abs(deep_svdd_distances - deep_svdd_radius) <= max(0.05 * abs(deep_svdd_radius), 1e-6))
    )[0]
    if boundary_points.size >= 3:
        boundary_center = projected[boundary_points].mean(axis=0)
        boundary_radius = float(np.mean(np.linalg.norm(projected[boundary_points] - boundary_center, axis=1)))
    else:
        boundary_center = center

    points: list[dict[str, Any]] = []
    for index, row in manifold_frame.reset_index(drop=True).iterrows():
        score_row = scored.iloc[index]
        role = "current" if index == len(manifold_frame) - 1 else "background"
        point = {
            "index": int(index),
            "role": role,
            "label": str(row.get("patient_id") or row.get("record_id") or f"record-{index + 1}"),
            "x": float(projected[index, 0]) if projected.shape[0] > index else 0.0,
            "y": float(projected[index, 1]) if projected.shape[0] > index else 0.0,
            "anomaly_score": float(score_row.get("anomaly_score", np.nan)),
            "deep_svdd_distance": float(score_row.get("deep_svdd_distance", np.nan)),
            "is_current": role == "current",
            "is_anomalous": bool(score_row.get("is_anomaly", False)),
        }
        points.append(point)

    current_point = points[-1] if points else None
    current_distance = None
    if current_point is not None and np.isfinite(current_point["deep_svdd_distance"]) and np.isfinite(deep_svdd_radius):
        current_distance = float(current_point["deep_svdd_distance"] - deep_svdd_radius)

    return {
        "projection_method": projection_method,
        "source_model": "vae_latent",
        "point_count": len(points),
        "points": points,
        "current_point": current_point,
        "latent_dim": int(latent_vectors.shape[1]) if latent_vectors.ndim == 2 else 0,
        "deep_svdd": {
            "radius": None if not np.isfinite(deep_svdd_radius) else float(deep_svdd_radius),
            "boundary_center": [float(boundary_center[0]), float(boundary_center[1])],
            "boundary_radius": float(boundary_radius),
            "current_distance_from_boundary": current_distance,
            "approximation": "Projected 2D boundary drawn from near-threshold Deep SVDD points.",
        },
    }


def _compute_reconstruction_residual_heatmap(
    pipeline,
    patient: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.DataFrame([patient])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(frame)
    feature_names = list(preprocessor.get_feature_names_out())
    estimators = getattr(model, "estimators_", {})

    model_specs = [
        ("autoencoder", "Autoencoder"),
        ("variational_autoencoder", "VAE"),
        ("ganomaly", "GANomaly"),
        ("anomaly_transformer", "Anomaly Transformer"),
        ("cnn_autoencoder", "CNN Autoencoder"),
    ]

    rows: list[dict[str, Any]] = []
    matrix_rows: list[np.ndarray] = []
    selected_model_names: list[str] = []

    for key, label in model_specs:
        estimator = estimators.get(key) if isinstance(estimators, dict) else None
        if estimator is None:
            continue

        residuals = None
        if hasattr(estimator, "reconstruction_residuals"):
            try:
                residuals = np.asarray(estimator.reconstruction_residuals(transformed), dtype=float)
            except Exception:
                residuals = None
        if residuals is None:
            try:
                if hasattr(estimator, "_forward"):
                    recon = estimator._forward(np.asarray(transformed, dtype=float), training=False)[0]
                    residuals = np.asarray(recon - transformed, dtype=float)
            except Exception:
                residuals = None
        if residuals is None or residuals.ndim != 2 or residuals.shape[0] == 0:
            continue

        abs_residuals = np.abs(residuals[0])
        rows.append(
            {
                "model": label,
                "model_key": key,
                "mean_abs_residual": float(np.mean(abs_residuals)),
                "max_abs_residual": float(np.max(abs_residuals)),
                "top_feature": feature_names[int(np.argmax(abs_residuals))] if feature_names else f"feature_{int(np.argmax(abs_residuals)) + 1}",
                "top_feature_residual": float(np.max(abs_residuals)),
            }
        )
        matrix_rows.append(abs_residuals)
        selected_model_names.append(label)

    if not matrix_rows:
        return {
            "status": "unavailable",
            "models": [],
            "feature_names": [],
            "matrix": [],
            "highlight_feature": None,
            "current_record_label": str(patient.get("patient_id") or patient.get("record_id") or "current-record"),
        }

    heatmap_models = []
    matrix = np.vstack(matrix_rows)
    for model_row, abs_row in zip(rows, matrix, strict=False):
        heatmap_models.append(
            {
                **model_row,
                "selected_feature_count": len(feature_names),
                "heatmap_row": abs_row.tolist(),
            }
        )

    return {
        "status": "ready",
        "models": heatmap_models,
        "feature_names": feature_names,
        "matrix": matrix.tolist(),
        "current_record_label": str(patient.get("patient_id") or patient.get("record_id") or "current-record"),
        "highlight_feature": feature_names[int(np.argmax(np.mean(matrix, axis=0)))] if feature_names else None,
        "max_abs_residual": float(np.max(matrix)),
    }


def _point_anomaly_feature_names(preprocessor) -> list[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
    if names:
        return names
    feature_count = int(getattr(preprocessor, "n_features_in_", 0) or 0)
    return [f"feature_{index}" for index in range(feature_count)]


def _normalize_point_anomaly_scores(max_abs_zscores: np.ndarray) -> np.ndarray:
    max_abs_zscores = np.asarray(max_abs_zscores, dtype=float).reshape(-1)
    return np.clip(max_abs_zscores / (max_abs_zscores + _POINT_ANOMALY_ZSCORE_NORMALIZER), 0.0, 1.0)


def _fit_point_anomaly_head(pipeline, data: pd.DataFrame) -> dict[str, Any]:
    """Fit a z-score point anomaly head over the transformed population baseline."""

    preprocessor = pipeline.named_steps["preprocessor"]
    if len(data) == 0:
        pipeline.point_anomaly_feature_names_ = []
        pipeline.point_anomaly_baseline_mean_ = np.asarray([], dtype=float)
        pipeline.point_anomaly_baseline_std_ = np.asarray([], dtype=float)
        pipeline.point_anomaly_summary_ = {"status": "skipped", "reason": "empty_data"}
        return pipeline.point_anomaly_summary_

    transformed = np.asarray(preprocessor.transform(data), dtype=float)
    if transformed.ndim != 2 or transformed.shape[1] == 0:
        pipeline.point_anomaly_feature_names_ = []
        pipeline.point_anomaly_baseline_mean_ = np.asarray([], dtype=float)
        pipeline.point_anomaly_baseline_std_ = np.asarray([], dtype=float)
        pipeline.point_anomaly_summary_ = {"status": "skipped", "reason": "no_features"}
        return pipeline.point_anomaly_summary_

    feature_names = _point_anomaly_feature_names(preprocessor)
    if len(feature_names) != transformed.shape[1]:
        feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

    baseline_mean = np.nanmean(transformed, axis=0)
    baseline_std = np.nanstd(transformed, axis=0, ddof=0)
    baseline_std = np.where(~np.isfinite(baseline_std) | (baseline_std <= 1e-12), 1.0, baseline_std)
    zscores = (transformed - baseline_mean) / baseline_std
    abs_zscores = np.abs(zscores)
    max_abs_zscores = np.max(abs_zscores, axis=1)
    normalized_scores = _normalize_point_anomaly_scores(max_abs_zscores)
    top_indices = np.argmax(abs_zscores, axis=1)
    top_features = np.asarray(feature_names, dtype=object)[top_indices]
    top_zscores = zscores[np.arange(zscores.shape[0]), top_indices]

    pipeline.point_anomaly_feature_names_ = list(feature_names)
    pipeline.point_anomaly_baseline_mean_ = np.asarray(baseline_mean, dtype=float)
    pipeline.point_anomaly_baseline_std_ = np.asarray(baseline_std, dtype=float)
    pipeline.point_anomaly_summary_ = {
        "status": "trained",
        "feature_count": int(transformed.shape[1]),
        "training_row_count": int(transformed.shape[0]),
        "score_mean": float(np.mean(normalized_scores)),
        "score_max": float(np.max(normalized_scores)),
        "score_p95": float(np.quantile(normalized_scores, 0.95)),
    }

    return {
        **pipeline.point_anomaly_summary_,
        "top_feature": str(top_features[0]) if len(top_features) else None,
        "top_feature_zscore": float(top_zscores[0]) if len(top_zscores) else float("nan"),
    }


def _score_point_anomaly_head(pipeline, transformed: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.Series, np.ndarray]:
    """Score per-feature z-score deviation against the fitted population baseline."""

    feature_names = list(getattr(pipeline, "point_anomaly_feature_names_", []))
    baseline_mean = np.asarray(getattr(pipeline, "point_anomaly_baseline_mean_", []), dtype=float).reshape(-1)
    baseline_std = np.asarray(getattr(pipeline, "point_anomaly_baseline_std_", []), dtype=float).reshape(-1)

    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim != 2 or transformed.shape[1] == 0:
        row_count = int(transformed.shape[0]) if transformed.ndim == 2 else 0
        empty = pd.DataFrame(index=range(row_count))
        return np.zeros(row_count, dtype=float), empty, pd.Series([""] * row_count, dtype=object), np.zeros(row_count, dtype=float)

    if baseline_mean.shape[0] != transformed.shape[1] or baseline_std.shape[0] != transformed.shape[1]:
        baseline_mean = np.nanmean(transformed, axis=0)
        baseline_std = np.nanstd(transformed, axis=0, ddof=0)

    baseline_std = np.where(~np.isfinite(baseline_std) | (baseline_std <= 1e-12), 1.0, baseline_std)
    if len(feature_names) != transformed.shape[1]:
        feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

    zscores = (transformed - baseline_mean) / baseline_std
    abs_zscores = np.abs(zscores)
    max_abs_zscores = np.max(abs_zscores, axis=1)
    normalized_scores = _normalize_point_anomaly_scores(max_abs_zscores)
    zscore_frame = pd.DataFrame(
        zscores,
        columns=[f"point_zscore__{feature_name}" for feature_name in feature_names],
    )
    top_indices = np.argmax(abs_zscores, axis=1)
    top_features = pd.Series(np.asarray(feature_names, dtype=object)[top_indices], dtype=object)
    top_feature_zscores = zscores[np.arange(zscores.shape[0]), top_indices]
    return normalized_scores, zscore_frame, top_features, top_feature_zscores


def _score_contextual_anomaly_head(
    pipeline,
    data: pd.DataFrame,
    transformed: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Score deviation from each patient's own prior-history baseline."""

    preprocessor = pipeline.named_steps["preprocessor"]
    config = getattr(preprocessor, "config", PreprocessingConfig())
    feature_names = list(getattr(pipeline, "point_anomaly_feature_names_", []))
    if len(feature_names) != np.asarray(transformed).shape[1]:
        feature_names = _point_anomaly_feature_names(preprocessor)
    if len(feature_names) != np.asarray(transformed).shape[1]:
        feature_names = [f"feature_{index}" for index in range(np.asarray(transformed).shape[1])]

    transformed = np.asarray(transformed, dtype=float)
    row_count = int(transformed.shape[0]) if transformed.ndim == 2 else 0
    if transformed.ndim != 2 or transformed.shape[1] == 0:
        empty = pd.DataFrame(index=range(row_count))
        return (
            np.zeros(row_count, dtype=float),
            empty,
            pd.Series([""] * row_count, dtype=object),
            np.zeros(row_count, dtype=float),
            np.zeros(row_count, dtype=int),
        )

    if data is None or len(data) == 0:
        contextual_frame = pd.DataFrame(
            np.zeros((row_count, transformed.shape[1]), dtype=float),
            columns=[f"contextual_zscore__{feature_name}" for feature_name in feature_names],
        )
        return (
            np.zeros(row_count, dtype=float),
            contextual_frame,
            pd.Series([""] * row_count, dtype=object),
            np.zeros(row_count, dtype=float),
            np.zeros(row_count, dtype=int),
        )

    if config.patient_id_col not in data.columns or config.encounter_time_col not in data.columns:
        contextual_frame = pd.DataFrame(
            np.zeros((row_count, transformed.shape[1]), dtype=float),
            columns=[f"contextual_zscore__{feature_name}" for feature_name in feature_names],
        )
        return (
            np.zeros(row_count, dtype=float),
            contextual_frame,
            pd.Series([""] * row_count, dtype=object),
            np.zeros(row_count, dtype=float),
            np.zeros(row_count, dtype=int),
        )

    ordered = data.copy().reset_index(drop=True)
    ordered["__contextual_row_index__"] = np.arange(len(ordered), dtype=int)
    ordered[config.encounter_time_col] = pd.to_datetime(ordered[config.encounter_time_col], errors="coerce")
    ordered = ordered.sort_values(
        [config.patient_id_col, config.encounter_time_col, "__contextual_row_index__"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered_features = transformed[ordered["__contextual_row_index__"].to_numpy(dtype=int)]

    contextual_scores = np.zeros(row_count, dtype=float)
    contextual_history_lengths = np.zeros(row_count, dtype=int)
    contextual_top_features = np.full(row_count, "", dtype=object)
    contextual_top_feature_zscores = np.zeros(row_count, dtype=float)
    contextual_frame = pd.DataFrame(
        np.zeros((row_count, transformed.shape[1]), dtype=float),
        columns=[f"contextual_zscore__{feature_name}" for feature_name in feature_names],
    )

    for _, group in ordered.groupby(config.patient_id_col, sort=False):
        group_indices = group.index.to_numpy(dtype=int)
        group_original_indices = ordered.loc[group_indices, "__contextual_row_index__"].to_numpy(dtype=int)
        group_features = ordered_features[group_indices]
        history: list[np.ndarray] = []

        for position, original_index in enumerate(group_original_indices):
            current_vector = group_features[position]
            if history:
                history_matrix = np.asarray(history, dtype=float)
                baseline_mean = np.nanmean(history_matrix, axis=0)
                baseline_std = np.nanstd(history_matrix, axis=0, ddof=0)
                baseline_std = np.where(~np.isfinite(baseline_std) | (baseline_std <= 1e-12), 1.0, baseline_std)
                zscores = (current_vector - baseline_mean) / baseline_std
                abs_zscores = np.abs(zscores)
                max_abs_zscore = float(np.max(abs_zscores))
                contextual_scores[original_index] = _normalize_point_anomaly_scores(np.asarray([max_abs_zscore], dtype=float))[0]
                contextual_history_lengths[original_index] = len(history)
                contextual_top_feature_index = int(np.argmax(abs_zscores))
                contextual_top_features[original_index] = feature_names[contextual_top_feature_index]
                contextual_top_feature_zscores[original_index] = float(zscores[contextual_top_feature_index])
                contextual_frame.iloc[original_index] = zscores
            history.append(current_vector)

    return (
        contextual_scores,
        contextual_frame,
        pd.Series(contextual_top_features, dtype=object),
        contextual_top_feature_zscores,
        contextual_history_lengths,
    )


def _collective_group_slug(group_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(group_name).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "group"


def _collective_group_definitions(pipeline) -> list[tuple[str, list[int]]]:
    """Build combination groups from the fitted feature map."""

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_map = preprocessor.export_feature_map()
    except Exception:
        feature_map = pd.DataFrame()

    if feature_map.empty or "final_feature" not in feature_map.columns:
        feature_names = _point_anomaly_feature_names(preprocessor)
        return [("all_features", list(range(len(feature_names))))]

    group_map: dict[str, list[int]] = {}
    fallback_direct_indices: list[int] = []
    for index, row in feature_map.reset_index(drop=True).iterrows():
        feature_type = str(row.get("feature_type", "direct"))
        source_columns = row.get("source_columns", [])
        if isinstance(source_columns, str):
            source_columns = [source_columns]
        if not isinstance(source_columns, (list, tuple)):
            source_columns = [source_columns] if source_columns is not None else []
        normalized_sources = [str(source) for source in source_columns if source is not None and str(source).strip()]
        group_name = None
        if len(normalized_sources) > 1 or feature_type != "direct":
            group_name = f"{feature_type}:{'|'.join(sorted(normalized_sources) or [str(row.get('source_features', row.get('final_feature', index)))])}"
        elif feature_type == "direct":
            fallback_direct_indices.append(int(index))

        if group_name is not None:
            group_map.setdefault(group_name, []).append(int(index))

    if not group_map and fallback_direct_indices:
        return [("all_features", fallback_direct_indices)]
    if not group_map:
        feature_names = _point_anomaly_feature_names(preprocessor)
        return [("all_features", list(range(len(feature_names))))]

    return [(group_name, indices) for group_name, indices in group_map.items() if len(indices) >= 1]


def _normalize_collective_reconstruction_scores(rms_errors: np.ndarray) -> np.ndarray:
    rms_errors = np.asarray(rms_errors, dtype=float).reshape(-1)
    return np.clip(rms_errors / (rms_errors + _POINT_ANOMALY_ZSCORE_NORMALIZER), 0.0, 1.0)


def _sample_rows_for_distribution_monitor(transformed: np.ndarray, *, max_rows: int, random_state: int = 42) -> np.ndarray:
    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim != 2 or transformed.shape[0] == 0:
        return np.asarray(transformed, dtype=float).reshape(0, transformed.shape[1] if transformed.ndim == 2 else 0)
    max_rows = max(1, min(int(max_rows), transformed.shape[0]))
    if transformed.shape[0] <= max_rows:
        return transformed.copy()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(transformed.shape[0], size=max_rows, replace=False)
    return transformed[indices]


def _estimate_rbf_gamma(reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[0] < 2:
        return 1.0 / max(float(reference.shape[1]) if reference.ndim == 2 else 1.0, 1.0)

    sample_size = min(reference.shape[0], 64)
    sample = reference[:sample_size]
    diffs = sample[:, None, :] - sample[None, :, :]
    squared_distances = np.sum(diffs**2, axis=2)
    upper = squared_distances[np.triu_indices(sample_size, k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0.0)]
    if finite.size == 0:
        return 1.0 / max(float(reference.shape[1]), 1.0)
    median_distance = float(np.median(finite))
    if median_distance <= 0.0 or not np.isfinite(median_distance):
        return 1.0 / max(float(reference.shape[1]), 1.0)
    return 1.0 / (2.0 * median_distance)


def _rbf_kernel_matrix(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("RBF kernel expects 2D matrices.")
    if left.shape[1] != right.shape[1]:
        raise ValueError("RBF kernel matrices must have the same feature width.")
    diffs = left[:, None, :] - right[None, :, :]
    squared_distances = np.sum(diffs**2, axis=2)
    return np.exp(-float(gamma) * squared_distances)


def _maximum_mean_discrepancy(left: np.ndarray, right: np.ndarray, *, gamma: float) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("MMD expects 2D matrices.")
    if left.shape[0] == 0 or right.shape[0] == 0:
        return 0.0
    if left.shape[1] != right.shape[1]:
        raise ValueError("MMD matrices must have the same feature width.")

    k_xx = _rbf_kernel_matrix(left, left, gamma)
    k_yy = _rbf_kernel_matrix(right, right, gamma)
    k_xy = _rbf_kernel_matrix(left, right, gamma)

    m = left.shape[0]
    n = right.shape[0]

    if m > 1:
        term_xx = (np.sum(k_xx) - np.trace(k_xx)) / float(m * (m - 1))
    else:
        term_xx = 0.0
    if n > 1:
        term_yy = (np.sum(k_yy) - np.trace(k_yy)) / float(n * (n - 1))
    else:
        term_yy = 0.0
    term_xy = float(np.mean(k_xy))
    return float(max(0.0, term_xx + term_yy - 2.0 * term_xy))


def _build_distribution_monitor(pipeline, data: pd.DataFrame, *, config: PreprocessingConfig) -> dict[str, Any]:
    """Fit a kernel two-sample baseline for staleness detection."""

    if data is None or len(data) == 0:
        pipeline.distribution_monitor_reference_ = np.asarray([], dtype=float)
        pipeline.distribution_monitor_gamma_ = float("nan")
        pipeline.distribution_monitor_threshold_ = float("nan")
        pipeline.distribution_monitor_summary_ = {"status": "skipped", "reason": "empty_data"}
        return pipeline.distribution_monitor_summary_

    transformed = np.asarray(pipeline.named_steps["preprocessor"].transform(data), dtype=float)
    reference = _sample_rows_for_distribution_monitor(
        transformed,
        max_rows=getattr(config, "distribution_monitor_reference_size", 256),
        random_state=42,
    )
    if reference.ndim != 2 or reference.shape[0] < 2:
        pipeline.distribution_monitor_reference_ = reference
        pipeline.distribution_monitor_gamma_ = float("nan")
        pipeline.distribution_monitor_threshold_ = float("nan")
        pipeline.distribution_monitor_summary_ = {"status": "skipped", "reason": "insufficient_reference"}
        return pipeline.distribution_monitor_summary_

    gamma = getattr(config, "distribution_monitor_kernel_gamma", None)
    gamma = float(gamma) if gamma is not None else _estimate_rbf_gamma(reference)
    calibration_size = max(1, min(int(getattr(config, "distribution_monitor_calibration_batch_size", 32)), reference.shape[0] // 2))
    calibration_trials = max(8, int(getattr(config, "distribution_monitor_bootstrap_trials", 64)))
    rng = np.random.default_rng(42)
    calibration_scores: list[float] = []
    if reference.shape[0] >= 2 * calibration_size:
        for _ in range(calibration_trials):
            indices = rng.choice(reference.shape[0], size=2 * calibration_size, replace=False)
            left = reference[indices[:calibration_size]]
            right = reference[indices[calibration_size:]]
            calibration_scores.append(_maximum_mean_discrepancy(left, right, gamma=gamma))
    elif reference.shape[0] >= 2:
        midpoint = reference.shape[0] // 2
        left = reference[:midpoint]
        right = reference[midpoint:]
        if len(left) and len(right):
            calibration_scores.append(_maximum_mean_discrepancy(left, right, gamma=gamma))

    if calibration_scores:
        threshold_quantile = float(getattr(config, "distribution_monitor_threshold_quantile", 0.95))
        threshold = float(np.quantile(calibration_scores, threshold_quantile))
    else:
        threshold = 0.0

    pipeline.distribution_monitor_reference_ = reference
    pipeline.distribution_monitor_gamma_ = float(gamma)
    pipeline.distribution_monitor_threshold_ = float(threshold)
    pipeline.distribution_monitor_summary_ = {
        "status": "trained",
        "reference_size": int(reference.shape[0]),
        "feature_count": int(reference.shape[1]),
        "gamma": float(gamma),
        "threshold": float(threshold),
        "threshold_quantile": float(getattr(config, "distribution_monitor_threshold_quantile", 0.95)),
        "calibration_trials": int(len(calibration_scores)),
    }
    return pipeline.distribution_monitor_summary_


def _score_distribution_monitor(pipeline, transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Compare an incoming batch against the training distribution baseline."""

    transformed = np.asarray(transformed, dtype=float)
    row_count = int(transformed.shape[0]) if transformed.ndim == 2 else 0
    if transformed.ndim != 2 or transformed.shape[1] == 0:
        return np.zeros(row_count, dtype=float), np.asarray([], dtype=float), False, float("nan")

    reference = np.asarray(getattr(pipeline, "distribution_monitor_reference_", []), dtype=float)
    gamma = float(getattr(pipeline, "distribution_monitor_gamma_", float("nan")))
    threshold = float(getattr(pipeline, "distribution_monitor_threshold_", float("nan")))
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != transformed.shape[1] or not np.isfinite(gamma):
        return np.zeros(row_count, dtype=float), np.asarray([], dtype=float), False, threshold

    batch_score = _maximum_mean_discrepancy(reference, transformed, gamma=gamma)
    score_vector = np.full(row_count, batch_score, dtype=float)
    alarm = bool(np.isfinite(threshold) and batch_score > threshold)
    return score_vector, np.asarray([batch_score], dtype=float), alarm, threshold


def _fit_collective_anomaly_head(pipeline, data: pd.DataFrame) -> dict[str, Any]:
    """Summarize the combination-level reconstruction baseline used by the collective head."""

    if data is None or len(data) == 0:
        pipeline.collective_anomaly_group_definitions_ = []
        pipeline.collective_anomaly_summary_ = {"status": "skipped", "reason": "empty_data"}
        return pipeline.collective_anomaly_summary_

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(data), dtype=float)
    residual_matrix = None
    estimators = getattr(pipeline.named_steps["model"], "estimators_", {})
    variational_autoencoder = estimators.get("variational_autoencoder") if isinstance(estimators, dict) else None
    if variational_autoencoder is not None and hasattr(variational_autoencoder, "reconstruction_residuals"):
        try:
            residual_matrix = np.asarray(variational_autoencoder.reconstruction_residuals(transformed), dtype=float)
        except Exception:
            residual_matrix = None
    if residual_matrix is None and hasattr(pipeline.named_steps["model"], "reconstruction_residuals"):
        try:
            residual_matrix = np.asarray(pipeline.named_steps["model"].reconstruction_residuals(transformed), dtype=float)
        except Exception:
            residual_matrix = None
    if residual_matrix is None:
        residual_matrix = transformed

    group_definitions = _collective_group_definitions(pipeline)
    if not group_definitions:
        group_definitions = [("all_features", list(range(transformed.shape[1])))]

    group_errors: list[np.ndarray] = []
    for _, indices in group_definitions:
        valid_indices = [index for index in indices if 0 <= int(index) < residual_matrix.shape[1]]
        if not valid_indices:
            continue
        group_matrix = residual_matrix[:, valid_indices]
        group_rms_error = np.sqrt(np.mean(group_matrix**2, axis=1))
        group_errors.append(group_rms_error)

    if not group_errors:
        group_errors = [np.sqrt(np.mean(transformed**2, axis=1))]

    stacked_errors = np.column_stack(group_errors)
    collective_scores = _normalize_collective_reconstruction_scores(np.max(stacked_errors, axis=1))
    pipeline.collective_anomaly_group_definitions_ = group_definitions
    pipeline.collective_anomaly_summary_ = {
        "status": "trained",
        "group_count": int(len(group_definitions)),
        "training_row_count": int(len(data)),
        "score_mean": float(np.mean(collective_scores)),
        "score_max": float(np.max(collective_scores)),
        "score_p95": float(np.quantile(collective_scores, 0.95)),
    }
    return pipeline.collective_anomaly_summary_


def _score_collective_anomaly_head(
    pipeline,
    transformed: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series, np.ndarray]:
    """Score combination-level reconstruction error across grouped features."""

    transformed = np.asarray(transformed, dtype=float)
    row_count = int(transformed.shape[0]) if transformed.ndim == 2 else 0
    if transformed.ndim != 2 or transformed.shape[1] == 0:
        empty = pd.DataFrame(index=range(row_count))
        return np.zeros(row_count, dtype=float), empty, pd.Series([""] * row_count, dtype=object), np.zeros(row_count, dtype=float)

    residual_matrix = None
    estimators = getattr(pipeline.named_steps["model"], "estimators_", {})
    variational_autoencoder = estimators.get("variational_autoencoder") if isinstance(estimators, dict) else None
    if variational_autoencoder is not None and hasattr(variational_autoencoder, "reconstruction_residuals"):
        try:
            residual_matrix = np.asarray(variational_autoencoder.reconstruction_residuals(transformed), dtype=float)
        except Exception:
            residual_matrix = None
    if residual_matrix is None and hasattr(pipeline.named_steps["model"], "reconstruction_residuals"):
        try:
            residual_matrix = np.asarray(pipeline.named_steps["model"].reconstruction_residuals(transformed), dtype=float)
        except Exception:
            residual_matrix = None
    if residual_matrix is None:
        residual_matrix = transformed

    group_definitions = list(getattr(pipeline, "collective_anomaly_group_definitions_", []))
    if not group_definitions:
        group_definitions = [("all_features", list(range(residual_matrix.shape[1])))]

    group_columns: dict[str, np.ndarray] = {}
    group_names: list[str] = []
    group_scores: list[np.ndarray] = []
    for group_name, indices in group_definitions:
        valid_indices = [index for index in indices if 0 <= int(index) < residual_matrix.shape[1]]
        if not valid_indices:
            continue
        valid_name = _collective_group_slug(group_name)
        group_matrix = residual_matrix[:, valid_indices]
        group_rms_error = np.sqrt(np.mean(group_matrix**2, axis=1))
        group_columns[f"collective_group_error__{valid_name}"] = group_rms_error
        group_names.append(valid_name)
        group_scores.append(group_rms_error)

    if not group_scores:
        group_scores = [np.sqrt(np.mean(residual_matrix**2, axis=1))]
        group_names = ["all_features"]
        group_columns["collective_group_error__all_features"] = group_scores[0]

    stacked_scores = np.column_stack(group_scores)
    top_indices = np.argmax(stacked_scores, axis=1)
    top_group_names = pd.Series(np.asarray(group_names, dtype=object)[top_indices], dtype=object)
    top_group_errors = stacked_scores[np.arange(stacked_scores.shape[0]), top_indices]
    collective_scores = _normalize_collective_reconstruction_scores(top_group_errors)
    collective_frame = pd.DataFrame(group_columns)
    return collective_scores, collective_frame, top_group_names, top_group_errors


def _conformal_p_values(calibration_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Compute finite-sample conformal p-values for larger-is-more-anomalous scores."""

    calibration_scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1)
    calibration_scores = calibration_scores[np.isfinite(calibration_scores)]
    p_values = np.full(test_scores.shape, np.nan, dtype=float)
    if calibration_scores.size == 0:
        return p_values

    finite_mask = np.isfinite(test_scores)
    if not finite_mask.any():
        return p_values

    comparisons = calibration_scores[:, None] >= test_scores[finite_mask][None, :]
    p_values[finite_mask] = (1.0 + np.sum(comparisons, axis=0)) / float(calibration_scores.size + 1)
    return p_values


def _configure_conformal_calibration(pipeline, calibration_frame: pd.DataFrame) -> dict[str, Any]:
    """Attach conformal calibration scores to a fitted pipeline."""

    if calibration_frame is None or len(calibration_frame) == 0:
        pipeline.conformal_alpha_ = float(_CONFORMAL_ALPHA)
        pipeline.conformal_calibration_scores_ = np.asarray([], dtype=float)
        pipeline.conformal_calibration_source_ = "unavailable"
        pipeline.conformal_calibration_nonconformity_ = "unavailable"
        pipeline.conformal_calibration_components_ = []
        return {
            "alpha": float(_CONFORMAL_ALPHA),
            "calibration_size": 0,
            "calibration_source": "unavailable",
            "nonconformity_source": "unavailable",
            "components": [],
        }

    transformed = pipeline.named_steps["preprocessor"].transform(calibration_frame)
    calibration_scores, nonconformity_source, components = _collect_conformal_nonconformity_scores(
        pipeline.named_steps["model"],
        transformed,
    )
    sequence_output = _score_sequence_detector(pipeline, calibration_frame)
    if sequence_output is not None:
        sequence_scores, _ = sequence_output
        calibration_scores, _, _ = _blend_tabular_and_sequence_scores(calibration_scores, sequence_scores, pipeline=pipeline)
        nonconformity_source = f"{nonconformity_source}+sequence"
    pipeline.conformal_alpha_ = float(_CONFORMAL_ALPHA)
    pipeline.conformal_calibration_scores_ = np.asarray(calibration_scores, dtype=float).reshape(-1)
    pipeline.conformal_calibration_source_ = "validation"
    pipeline.conformal_calibration_nonconformity_ = nonconformity_source
    pipeline.conformal_calibration_components_ = list(components)
    return {
        "alpha": float(_CONFORMAL_ALPHA),
        "calibration_size": int(len(pipeline.conformal_calibration_scores_)),
        "calibration_source": "validation",
        "nonconformity_source": nonconformity_source,
        "components": list(components),
    }


def _build_temporal_sequence_windows(
    data: pd.DataFrame,
    transformed: np.ndarray,
    *,
    patient_id_col: str,
    time_col: str,
    window_size: int,
    require_full_window: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding visit windows aligned back to the original row order."""

    if data is None or len(data) == 0:
        empty_windows = np.empty((0, max(1, int(window_size)), transformed.shape[1] if transformed.ndim == 2 else 0), dtype=float)
        return empty_windows, np.asarray([], dtype=int), np.asarray([], dtype=int)

    if patient_id_col not in data.columns or time_col not in data.columns:
        raise ValueError("Sequence detection requires patient and timestamp columns.")

    ordered = data.copy().reset_index(drop=True)
    ordered["__sequence_row_index__"] = np.arange(len(ordered), dtype=int)
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    ordered = ordered.sort_values([patient_id_col, time_col, "__sequence_row_index__"], kind="mergesort").reset_index(drop=True)

    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim != 2:
        raise ValueError("transformed sequence features must be a 2D array.")
    ordered_features = transformed[ordered["__sequence_row_index__"].to_numpy(dtype=int)]

    windows: list[np.ndarray] = []
    row_positions: list[int] = []
    history_lengths: list[int] = []
    resolved_window_size = max(1, int(window_size))

    for _, group in ordered.groupby(patient_id_col, sort=False):
        group_positions = group.index.to_numpy(dtype=int)
        group_features = ordered_features[group_positions]
        group_original_indices = ordered.loc[group_positions, "__sequence_row_index__"].to_numpy(dtype=int)

        for end_position in range(group_features.shape[0]):
            start_position = max(0, end_position - resolved_window_size + 1)
            window = group_features[start_position : end_position + 1]
            history_length = int(window.shape[0])
            if require_full_window and history_length < resolved_window_size:
                continue
            if history_length < resolved_window_size:
                pad = np.zeros((resolved_window_size - history_length, group_features.shape[1]), dtype=float)
                window = np.vstack([pad, window])
            windows.append(window)
            row_positions.append(int(group_original_indices[end_position]))
            history_lengths.append(history_length)

    if not windows:
        empty_windows = np.empty((0, resolved_window_size, transformed.shape[1]), dtype=float)
        return empty_windows, np.asarray([], dtype=int), np.asarray([], dtype=int)

    return np.stack(windows, axis=0), np.asarray(row_positions, dtype=int), np.asarray(history_lengths, dtype=int)


def _fit_sequence_detector(
    pipeline,
    data: pd.DataFrame,
    *,
    config: PreprocessingConfig,
) -> dict[str, Any] | None:
    """Train a temporal sequence detector from the raw visit chronology."""

    if data is None or len(data) == 0:
        pipeline.sequence_detector_ = None
        pipeline.sequence_detector_summary_ = {"status": "skipped", "reason": "empty_data"}
        return pipeline.sequence_detector_summary_

    if config.patient_id_col not in data.columns or config.encounter_time_col not in data.columns:
        pipeline.sequence_detector_ = None
        pipeline.sequence_detector_summary_ = {
            "status": "skipped",
            "reason": "missing_sequence_columns",
            "patient_id_col": config.patient_id_col,
            "encounter_time_col": config.encounter_time_col,
        }
        return pipeline.sequence_detector_summary_

    transformed = pipeline.named_steps["preprocessor"].transform(data)
    window_size = max(1, int(config.sequence_window_size))
    windows, row_positions, history_lengths = _build_temporal_sequence_windows(
        data,
        transformed,
        patient_id_col=config.patient_id_col,
        time_col=config.encounter_time_col,
        window_size=window_size,
        require_full_window=True,
    )

    if windows.shape[0] == 0:
        windows, row_positions, history_lengths = _build_temporal_sequence_windows(
            data,
            transformed,
            patient_id_col=config.patient_id_col,
            time_col=config.encounter_time_col,
            window_size=window_size,
            require_full_window=False,
        )

    if windows.shape[0] == 0:
        pipeline.sequence_detector_ = None
        pipeline.sequence_detector_summary_ = {"status": "skipped", "reason": "no_windows"}
        return pipeline.sequence_detector_summary_

    detector = TemporalConvolutionalSequenceDetector(
        window_size=window_size,
        filters=config.sequence_detector_filters,
        kernel_size=config.sequence_detector_kernel_size,
        latent_dim=config.sequence_detector_latent_dim,
        dropout=config.sequence_detector_dropout,
        learning_rate=config.sequence_detector_learning_rate,
        batch_size=config.sequence_detector_batch_size,
        max_epochs=config.sequence_detector_max_epochs,
        patience=config.sequence_detector_patience,
        l2=config.sequence_detector_l2,
        random_state=config.sequence_detector_random_state,
        verbose=config.sequence_detector_verbose,
    )
    detector.fit(windows)
    detector.row_positions_ = np.asarray(row_positions, dtype=int)
    detector.history_lengths_ = np.asarray(history_lengths, dtype=int)
    detector.patient_id_col_ = config.patient_id_col
    detector.encounter_time_col_ = config.encounter_time_col
    detector.feature_count_ = int(transformed.shape[1])
    detector.sequence_window_size_ = window_size
    detector.training_window_count_ = int(windows.shape[0])
    detector.training_row_count_ = int(len(data))

    pipeline.sequence_detector_ = detector
    pipeline.sequence_detector_summary_ = {
        "status": "trained",
        "window_size": window_size,
        "window_count": int(windows.shape[0]),
        "row_count": int(len(data)),
        "patient_id_col": config.patient_id_col,
        "encounter_time_col": config.encounter_time_col,
    }
    return pipeline.sequence_detector_summary_


def _score_sequence_detector(pipeline, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Return per-row sequence anomaly scores and context lengths if available."""

    detector = getattr(pipeline, "sequence_detector_", None)
    if detector is None:
        return None
    if data is None or len(data) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=int)

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(data)
        window_size = int(getattr(detector, "sequence_window_size_", getattr(detector, "window_size", 1)))
        windows, row_positions, history_lengths = _build_temporal_sequence_windows(
            data,
            transformed,
            patient_id_col=getattr(detector, "patient_id_col_", preprocessor.config.patient_id_col),
            time_col=getattr(detector, "encounter_time_col_", preprocessor.config.encounter_time_col),
            window_size=window_size,
            require_full_window=False,
        )
    except Exception:
        return None

    if windows.shape[0] == 0:
        return np.zeros(len(data), dtype=float), np.zeros(len(data), dtype=int)

    window_scores = np.asarray(detector.score(windows), dtype=float).reshape(-1)
    ordered_scores = np.zeros(len(data), dtype=float)
    ordered_history = np.zeros(len(data), dtype=int)
    ordered_scores[row_positions] = window_scores
    ordered_history[row_positions] = history_lengths
    return ordered_scores, ordered_history


def _resolve_sequence_detector_weight(pipeline) -> float:
    """Resolve the blend weight for sequence-based anomaly scoring."""

    weight = getattr(pipeline, "sequence_detector_weight_", None)
    if weight is None:
        weight = getattr(getattr(pipeline, "named_steps", {}).get("preprocessor", None), "config", None)
        weight = getattr(weight, "sequence_detector_weight", None) if weight is not None else None
    if weight is None:
        return 0.25
    try:
        numeric = float(weight)
    except (TypeError, ValueError):
        return 0.25
    if not np.isfinite(numeric):
        return 0.25
    return float(np.clip(numeric, 0.0, 1.0))


def _sequence_scores_to_unit_interval(scores: np.ndarray, detector: Any | None) -> np.ndarray:
    """Normalize raw sequence reconstruction errors onto a 0-1 scale."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return scores

    if detector is None:
        minimum = float(np.min(scores))
        maximum = float(np.max(scores))
        scale = maximum - minimum
        if scale <= 0.0 or not np.isfinite(scale):
            return np.zeros_like(scores, dtype=float)
        return np.clip((scores - minimum) / scale, 0.0, 1.0)

    mean = float(getattr(detector, "_training_raw_score_mean_", np.mean(scores)))
    std = float(getattr(detector, "_training_raw_score_std_", np.std(scores, ddof=0)))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    z_scores = (scores - mean) / std
    return np.clip(1.0 / (1.0 + np.exp(-z_scores)), 0.0, 1.0)


def _detect_score_stream_drift(
    scores: np.ndarray,
    *,
    baseline_window: int = 5,
    cusum_k: float = 0.25,
    cusum_h: float = 5.0,
    adwin_delta: float = 0.01,
    adwin_min_window: int = 5,
) -> dict[str, Any]:
    """Detect gradual or abrupt drift on a per-patient anomaly score stream."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    finite_scores = np.where(np.isfinite(scores), scores, np.nan)
    n = finite_scores.size
    if n == 0:
        return {
            "baseline_window": int(baseline_window),
            "cusum_alarm": False,
            "adwin_alarm": False,
            "drift_alarm": False,
            "cusum_stat": [],
            "adwin_stat": [],
            "cusum_change_index": None,
            "adwin_change_index": None,
        }

    baseline_window = max(1, min(int(baseline_window), n))
    baseline = finite_scores[:baseline_window]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0:
        baseline = finite_scores[np.isfinite(finite_scores)]
    if baseline.size == 0:
        baseline_mean = 0.0
        baseline_std = 1.0
    else:
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline, ddof=0))
        if not np.isfinite(baseline_std) or baseline_std <= 0.0:
            baseline_std = max(1e-6, abs(baseline_mean) * 0.1 + 1e-6)

    cusum_stats: list[float] = []
    cusum_alarm = False
    cusum_change_index: int | None = None
    cusum_value = 0.0
    for index, score in enumerate(finite_scores):
        if not np.isfinite(score):
            cusum_stats.append(cusum_value)
            continue
        standardized = (float(score) - baseline_mean) / baseline_std
        cusum_value = max(0.0, cusum_value + standardized - float(cusum_k))
        cusum_stats.append(float(cusum_value))
        if not cusum_alarm and cusum_value > float(cusum_h):
            cusum_alarm = True
            cusum_change_index = int(index)

    adwin_stats: list[float] = []
    adwin_alarm = False
    adwin_change_index: int | None = None
    drift_window: list[float] = []
    min_window = max(2, min(int(adwin_min_window), n))
    delta = float(np.clip(adwin_delta, 1e-6, 0.5))
    for index, score in enumerate(finite_scores):
        if np.isfinite(score):
            drift_window.append(float(score))
        adwin_stat = 0.0
        if len(drift_window) >= 2 * min_window:
            window = np.asarray(drift_window, dtype=float)
            best_stat = 0.0
            best_split: int | None = None
            for split in range(min_window, len(window) - min_window + 1):
                left = window[:split]
                right = window[split:]
                diff = float(abs(np.mean(right) - np.mean(left)))
                bound = float(np.sqrt(0.5 * np.log(4.0 / delta) * (1.0 / len(left) + 1.0 / len(right))))
                stat = diff - bound
                if stat > best_stat:
                    best_stat = stat
                    best_split = split
            adwin_stat = float(max(best_stat, 0.0))
            if best_stat > 0.0:
                adwin_alarm = True
                if adwin_change_index is None:
                    adwin_change_index = int(index if best_split is None else max(0, best_split - 1))
                drift_window = drift_window[-min_window:]
        adwin_stats.append(adwin_stat)

    return {
        "baseline_window": baseline_window,
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "cusum_alarm": bool(cusum_alarm),
        "adwin_alarm": bool(adwin_alarm),
        "drift_alarm": bool(cusum_alarm or adwin_alarm),
        "cusum_stat": cusum_stats,
        "adwin_stat": adwin_stats,
        "cusum_change_index": cusum_change_index,
        "adwin_change_index": adwin_change_index,
    }


def _blend_tabular_and_sequence_scores(
    raw_scores: np.ndarray,
    sequence_scores: np.ndarray | None,
    *,
    pipeline,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Blend tabular and sequence anomaly scores into a single final score."""

    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if sequence_scores is None:
        return raw_scores, np.zeros_like(raw_scores, dtype=float), 0.0

    detector = getattr(pipeline, "sequence_detector_", None)
    normalized_sequence_scores = _sequence_scores_to_unit_interval(sequence_scores, detector)
    sequence_weight = _resolve_sequence_detector_weight(pipeline)
    final_scores = (1.0 - sequence_weight) * raw_scores + sequence_weight * normalized_sequence_scores
    return final_scores, normalized_sequence_scores, sequence_weight


def _build_grouped_drift_detection(
    data: pd.DataFrame,
    final_scores: np.ndarray,
    *,
    patient_id_col: str,
    time_col: str,
    baseline_window: int,
    cusum_k: float,
    cusum_h: float,
    adwin_delta: float,
    adwin_min_window: int,
) -> dict[str, Any]:
    """Compute drift alarms per patient and align them back to rows."""

    if data is None or len(data) == 0:
        return {
            "alarm": np.asarray([], dtype=bool),
            "cusum_alarm": np.asarray([], dtype=bool),
            "adwin_alarm": np.asarray([], dtype=bool),
            "score_stream": np.asarray([], dtype=float),
            "cusum_stat": np.asarray([], dtype=float),
            "adwin_stat": np.asarray([], dtype=float),
            "change_index": np.asarray([], dtype=int),
            "method": [],
        }

    frame = data.copy().reset_index(drop=True)
    frame["__row_index__"] = np.arange(len(frame), dtype=int)
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    if patient_id_col not in frame.columns or time_col not in frame.columns:
        return {
            "alarm": np.zeros(len(frame), dtype=bool),
            "cusum_alarm": np.zeros(len(frame), dtype=bool),
            "adwin_alarm": np.zeros(len(frame), dtype=bool),
            "score_stream": np.asarray(final_scores, dtype=float).reshape(-1),
            "cusum_stat": np.zeros(len(frame), dtype=float),
            "adwin_stat": np.zeros(len(frame), dtype=float),
            "change_index": np.full(len(frame), -1, dtype=int),
            "method": ["unavailable"] * len(frame),
        }

    frame = frame.sort_values([patient_id_col, time_col, "__row_index__"], kind="mergesort").reset_index(drop=True)
    ordered_scores = np.asarray(final_scores, dtype=float).reshape(-1)[frame["__row_index__"].to_numpy(dtype=int)]
    alarm_sorted = np.zeros(len(frame), dtype=bool)
    cusum_alarm_sorted = np.zeros(len(frame), dtype=bool)
    adwin_alarm_sorted = np.zeros(len(frame), dtype=bool)
    cusum_stat_sorted = np.zeros(len(frame), dtype=float)
    adwin_stat_sorted = np.zeros(len(frame), dtype=float)
    change_index_sorted = np.full(len(frame), -1, dtype=int)
    method_sorted = ["none"] * len(frame)

    for _, group in frame.groupby(patient_id_col, sort=False):
        positions = group.index.to_numpy(dtype=int)
        original_indices = group["__row_index__"].to_numpy(dtype=int)
        group_scores = ordered_scores[positions]
        drift = _detect_score_stream_drift(
            group_scores,
            baseline_window=baseline_window,
            cusum_k=cusum_k,
            cusum_h=cusum_h,
            adwin_delta=adwin_delta,
            adwin_min_window=adwin_min_window,
        )
        group_cusum_alarm = np.zeros(len(positions), dtype=bool)
        group_adwin_alarm = np.zeros(len(positions), dtype=bool)
        group_alarm = np.zeros(len(positions), dtype=bool)
        group_cusum_stat = np.asarray(drift["cusum_stat"], dtype=float)
        group_adwin_stat = np.asarray(drift["adwin_stat"], dtype=float)
        change_idx = drift["cusum_change_index"] if drift["cusum_change_index"] is not None else drift["adwin_change_index"]
        if change_idx is not None and 0 <= int(change_idx) < len(positions):
            change_idx = int(change_idx)
            group_alarm[change_idx:] = True
            if drift["cusum_alarm"]:
                group_cusum_alarm[change_idx:] = True
            if drift["adwin_alarm"]:
                group_adwin_alarm[change_idx:] = True
            method_value = "hybrid" if drift["cusum_alarm"] and drift["adwin_alarm"] else ("cusum" if drift["cusum_alarm"] else "adwin")
            for idx in range(change_idx, len(positions)):
                change_index_sorted[positions[idx]] = change_idx
                method_sorted[positions[idx]] = method_value
        alarm_sorted[positions] = group_alarm
        cusum_alarm_sorted[positions] = group_cusum_alarm
        adwin_alarm_sorted[positions] = group_adwin_alarm
        cusum_stat_sorted[positions] = group_cusum_stat
        adwin_stat_sorted[positions] = group_adwin_stat
    original_indices = frame["__row_index__"].to_numpy(dtype=int)
    restored_alarm = np.zeros(len(frame), dtype=bool)
    restored_cusum_alarm = np.zeros(len(frame), dtype=bool)
    restored_adwin_alarm = np.zeros(len(frame), dtype=bool)
    restored_cusum_stat = np.zeros(len(frame), dtype=float)
    restored_adwin_stat = np.zeros(len(frame), dtype=float)
    restored_change_index = np.full(len(frame), -1, dtype=int)
    restored_method = ["none"] * len(frame)
    restored_alarm[original_indices] = alarm_sorted
    restored_cusum_alarm[original_indices] = cusum_alarm_sorted
    restored_adwin_alarm[original_indices] = adwin_alarm_sorted
    restored_cusum_stat[original_indices] = cusum_stat_sorted
    restored_adwin_stat[original_indices] = adwin_stat_sorted
    restored_change_index[original_indices] = change_index_sorted
    for original_index, value in zip(original_indices, method_sorted):
        restored_method[int(original_index)] = value
    return {
        "alarm": restored_alarm,
        "cusum_alarm": restored_cusum_alarm,
        "adwin_alarm": restored_adwin_alarm,
        "score_stream": np.asarray(final_scores, dtype=float).reshape(-1),
        "cusum_stat": restored_cusum_stat,
        "adwin_stat": restored_adwin_stat,
        "change_index": restored_change_index,
        "method": restored_method,
    }


def _load_split_file(split_dir: Path, split_name: str) -> pd.DataFrame | None:
    candidate_dir = split_dir / split_name
    if not candidate_dir.exists():
        return None
    if candidate_dir.is_file():
        return load_tabular_data(candidate_dir)
    for filename in _SPLIT_FILENAMES:
        candidate_file = candidate_dir / filename
        if candidate_file.exists():
            return load_tabular_data(candidate_file)
    csv_files = sorted(candidate_dir.glob("*.csv"))
    if csv_files:
        return load_tabular_data(csv_files[0])
    parquet_files = sorted(candidate_dir.glob("*.parquet"))
    if parquet_files:
        return load_tabular_data(parquet_files[0])
    return None


def load_dataset_split_directory(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load train/validation/test frames from a split directory."""

    split_dir = Path(path)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not split_dir.is_dir():
        raise ValueError(f"Split directory must be a folder: {split_dir}")

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        frame = _load_split_file(split_dir, split_name)
        if frame is not None:
            splits[split_name] = frame

    if "train" not in splits:
        raise ValueError(
            f"Split directory {split_dir} does not contain a train split. "
            "Expected a train folder with data.csv or data.parquet."
        )
    return splits


def _split_indices_by_groups(
    groups: pd.Series,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = pd.Index(groups.dropna().astype(str).unique())
    if unique_groups.size < 3:
        indices = np.arange(len(groups))
        if indices.size < 3:
            return indices, np.asarray([], dtype=int), np.asarray([], dtype=int)
        train_idx, remainder_idx = train_test_split(
            indices,
            test_size=2 / max(3, len(indices)),
            random_state=random_state,
            shuffle=True,
        )
        val_idx, test_idx = train_test_split(
            remainder_idx,
            test_size=0.5,
            random_state=random_state,
            shuffle=True,
        )
        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
    if unique_groups.empty:
        indices = np.arange(len(groups))
        train_idx, remainder_idx = train_test_split(
            indices,
            test_size=max(0.0, validation_fraction + test_fraction),
            random_state=random_state,
            shuffle=True,
        )
        if validation_fraction + test_fraction <= 0.0 or len(remainder_idx) < 2:
            return np.asarray(train_idx), np.asarray([], dtype=int), np.asarray([], dtype=int)

        remainder_ratio = test_fraction / max(validation_fraction + test_fraction, 1e-12)
        val_idx, test_idx = train_test_split(
            remainder_idx,
            test_size=remainder_ratio,
            random_state=random_state,
            shuffle=True,
        )
        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

    rng = np.random.default_rng(random_state)
    shuffled_groups = unique_groups.to_numpy(copy=True)
    rng.shuffle(shuffled_groups)

    total_groups = len(shuffled_groups)
    train_end = max(1, int(round(total_groups * train_fraction)))
    validation_end = max(train_end + 1, int(round(total_groups * (train_fraction + validation_fraction))))
    train_end = min(train_end, total_groups)
    validation_end = min(validation_end, total_groups)

    train_groups = set(shuffled_groups[:train_end].tolist())
    validation_groups = set(shuffled_groups[train_end:validation_end].tolist())
    test_groups = set(shuffled_groups[validation_end:].tolist())

    if not validation_groups and test_groups:
        validation_groups.add(test_groups.pop())
    if not test_groups and validation_groups:
        test_groups.add(validation_groups.pop())

    group_values = groups.astype(str).fillna("__missing__")
    train_idx = np.flatnonzero(group_values.isin(train_groups).to_numpy())
    validation_idx = np.flatnonzero(group_values.isin(validation_groups).to_numpy())
    test_idx = np.flatnonzero(group_values.isin(test_groups).to_numpy())

    assigned = set(train_idx.tolist()) | set(validation_idx.tolist()) | set(test_idx.tolist())
    missing_idx = np.array(sorted(set(range(len(groups))) - assigned), dtype=int)
    if missing_idx.size:
        train_idx = np.sort(np.concatenate([train_idx, missing_idx]))

    return np.sort(train_idx), np.sort(validation_idx), np.sort(test_idx)


def split_tabular_dataset(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    group_column: str = "patient_id",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split a tabular health dataset into train, validation, and test frames."""

    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("Split fractions must be finite and non-negative.")
    total = float(fractions.sum())
    if total <= 0.0:
        raise ValueError("At least one split fraction must be greater than zero.")
    train_fraction, validation_fraction, test_fraction = (fractions / total).tolist()

    if len(data) == 0:
        empty = data.iloc[0:0].copy()
        return {"train": empty, "validation": empty, "test": empty}
    if len(data) < 3:
        empty = data.iloc[0:0].copy()
        return {"train": data.copy().reset_index(drop=True), "validation": empty, "test": empty}

    if group_column in data.columns and data[group_column].notna().sum() > 0:
        train_idx, validation_idx, test_idx = _split_indices_by_groups(
            data[group_column],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_state=random_state,
        )
    else:
        indices = np.arange(len(data))
        train_idx, remainder_idx = train_test_split(
            indices,
            test_size=max(0.0, validation_fraction + test_fraction),
            random_state=random_state,
            shuffle=True,
        )
        if validation_fraction + test_fraction <= 0.0 or len(remainder_idx) < 2:
            validation_idx = np.asarray([], dtype=int)
            test_idx = np.asarray([], dtype=int)
        else:
            remainder_ratio = test_fraction / max(validation_fraction + test_fraction, 1e-12)
            validation_idx, test_idx = train_test_split(
                remainder_idx,
                test_size=remainder_ratio,
                random_state=random_state,
                shuffle=True,
            )

    return {
        "train": data.iloc[np.asarray(train_idx)].copy().reset_index(drop=True),
        "validation": data.iloc[np.asarray(validation_idx)].copy().reset_index(drop=True),
        "test": data.iloc[np.asarray(test_idx)].copy().reset_index(drop=True),
    }


def save_dataset_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    filename: str = "data.csv",
) -> dict[str, str]:
    """Save train/validation/test splits into separate folders."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            continue
        split_frame = splits[split_name]
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_file = split_dir / filename
        split_frame.to_csv(split_file, index=False)
        manifest[split_name] = str(split_file)

    metadata = {
        "splits": {name: int(len(frame)) for name, frame in splits.items()},
        "columns": list(next(iter(splits.values())).columns) if splits else [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (output_path / "split_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest


def prepare_dataset_splits(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    group_column: str = "patient_id",
    random_state: int = 42,
) -> dict[str, str]:
    """Load a raw dataset, split it, and write the three folders to disk."""

    data = load_tabular_data(input_path)
    splits = split_tabular_dataset(
        data,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        group_column=group_column,
        random_state=random_state,
    )
    return save_dataset_splits(splits, output_dir)


def _coerce_list_cell(value: Any) -> Any:
    """Normalize list-like strings into Python lists.

    CSV exports often serialize lists as JSON strings or comma-separated text.
    This helper converts the common representations into actual Python lists so
    downstream preprocessing can expand them consistently.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, (list, tuple, set)):
        return list(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.lower() in {"none", "null", "nan"}:
            return None

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass

        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]

    return value


def _normalize_scalar_cell(value: Any) -> Any:
    """Normalize common missing-value sentinels in scalar cells."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None
        return text

    return value


def _normalize_loaded_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV/Parquet ingest quirks into schema-friendly types."""

    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    for column in _DEFAULT_DATETIME_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    for column in _LIST_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].apply(_coerce_list_cell)

    for column in frame.columns:
        if column in _LIST_COLUMNS:
            continue
        if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].apply(_normalize_scalar_cell)

    return frame


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    """Load training or inference data from CSV or Parquet.

    The loader applies a small schema-aware normalization pass so the rest of
    the pipeline can consume list-like fields and timestamps consistently.
    """

    input_path = Path(path)
    if input_path.is_dir():
        for filename in _SPLIT_FILENAMES:
            candidate = input_path / filename
            if candidate.exists():
                return load_tabular_data(candidate)
        csv_files = sorted(input_path.glob("*.csv"))
        if csv_files:
            return load_tabular_data(csv_files[0])
        parquet_files = sorted(input_path.glob("*.parquet"))
        if parquet_files:
            return load_tabular_data(parquet_files[0])
        raise ValueError(f"Directory {input_path} does not contain a CSV or Parquet file.")

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(input_path, low_memory=False)
        return _normalize_loaded_frame(data)
    if suffix == ".parquet":
        data = pd.read_parquet(input_path)
        return _normalize_loaded_frame(data)

    raise ValueError("Unsupported input format. Use .csv or .parquet.")


def train_anomaly_pipeline(
    data: pd.DataFrame,
    *,
    y: Any | None = None,
    config: PreprocessingConfig | None = None,
):
    """Fit the end-to-end anomaly pipeline on a dataframe."""

    pipeline = build_anomaly_pipeline(config)
    start = time.perf_counter()
    pipeline.fit(data, y)
    elapsed = time.perf_counter() - start
    pipeline.training_time_seconds_ = float(elapsed)
    pipeline.training_time_ms_ = float(elapsed * 1000.0)
    pipeline.training_sample_count_ = int(len(data))
    point_summary = _fit_point_anomaly_head(pipeline, data)
    pipeline.point_anomaly_summary_ = point_summary
    pipeline.contextual_anomaly_summary_ = {
        "status": "ready",
        "feature_count": int(len(getattr(pipeline, "point_anomaly_feature_names_", []))),
        "row_count": int(len(data)),
        "mode": "patient_history_baseline",
    }
    collective_summary = _fit_collective_anomaly_head(pipeline, data)
    pipeline.collective_anomaly_summary_ = collective_summary
    distribution_summary = _build_distribution_monitor(pipeline, data, config=pipeline.named_steps["preprocessor"].config)
    pipeline.distribution_monitor_summary_ = distribution_summary
    pipeline.sequence_detector_weight_ = _resolve_sequence_detector_weight(pipeline)
    sequence_summary = _fit_sequence_detector(pipeline, data, config=pipeline.named_steps["preprocessor"].config)
    pipeline.sequence_detector_summary_ = sequence_summary
    if sequence_summary.get("status") == "trained":
        raw_scores = np.asarray(pipeline.named_steps["model"].raw_anomaly_score(pipeline.named_steps["preprocessor"].transform(data)), dtype=float)
        sequence_scores, _ = _score_sequence_detector(pipeline, data) or (None, None)
        final_scores, _, _ = _blend_tabular_and_sequence_scores(raw_scores, sequence_scores, pipeline=pipeline)
        pipeline.final_anomaly_threshold_ = float(np.quantile(final_scores, 1 - float(getattr(pipeline.named_steps["model"], "contamination", 0.05))))
    else:
        pipeline.final_anomaly_threshold_ = float(getattr(pipeline.named_steps["model"], "offset_", 0.5))
    calibration_summary = _configure_conformal_calibration(pipeline, data)
    pipeline.conformal_summary_ = calibration_summary
    if len(data) > 0:
        sample_size = min(25, len(data))
        pipeline.explain_background_ = data.sample(n=sample_size, random_state=42).copy() if len(data) > sample_size else data.copy()
    pipeline.model_serialized_size_bytes_ = int(len(pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)))
    pipeline.model_estimated_ram_usage_bytes_ = int(_estimate_object_size_bytes(pipeline))
    return pipeline


def train_anomaly_pipeline_from_split(
    split_dir: str | Path,
    *,
    label_column: str | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train on a split directory and optionally calibrate against validation labels."""

    splits = load_dataset_split_directory(split_dir)
    train_data = splits["train"].copy()
    validation_data = splits.get("validation")
    test_data = splits.get("test")

    train_labels = None
    validation_labels = None
    test_labels = None

    if label_column:
        if label_column not in train_data.columns:
            raise ValueError(f"Label column '{label_column}' was not found in the train split.")
        train_labels = train_data[label_column]
        train_data = train_data.drop(columns=[label_column])
        if validation_data is not None and label_column in validation_data.columns:
            validation_labels = validation_data[label_column]
            validation_data = validation_data.drop(columns=[label_column])
        if test_data is not None and label_column in test_data.columns:
            test_labels = test_data[label_column]
            test_data = test_data.drop(columns=[label_column])

    pipeline = train_anomaly_pipeline(train_data, y=train_labels, config=config)
    metrics: dict[str, Any] = {
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)) if validation_data is not None else 0,
        "test_rows": int(len(test_data)) if test_data is not None else 0,
    }

    if validation_data is not None and validation_labels is not None and len(validation_data) > 0:
        validation_scored = score_records(pipeline, validation_data)
        validation_scores = np.asarray(validation_scored["anomaly_score"], dtype=float)
        validation_labels_binary = _coerce_binary_labels(validation_labels)
        threshold, calibration_metrics = _calibrate_threshold_from_scores(validation_scores, validation_labels_binary)
        pipeline.final_anomaly_threshold_ = float(threshold)
        model = pipeline.named_steps["model"]
        model.calibrated_threshold_ = float(threshold)
        model.calibration_metrics_ = calibration_metrics
        model.calibration_source_ = "validation"
        metrics["validation_calibration"] = calibration_metrics

    conformal_frame = validation_data
    conformal_source = "validation"
    if conformal_frame is not None and validation_labels is not None and len(conformal_frame) > 0:
        try:
            validation_labels_binary = _coerce_binary_labels(validation_labels)
            normal_mask = validation_labels_binary == 0
            if np.any(normal_mask):
                conformal_frame = conformal_frame.iloc[np.flatnonzero(normal_mask)].copy().reset_index(drop=True)
                conformal_source = "validation_normal_only"
        except Exception:
            conformal_frame = validation_data
            conformal_source = "validation"

    if conformal_frame is None or len(conformal_frame) == 0:
        conformal_frame = train_data
        conformal_source = "train_fallback"

    conformal_summary = _configure_conformal_calibration(pipeline, conformal_frame)
    conformal_summary["calibration_source"] = conformal_source
    pipeline.conformal_summary_ = conformal_summary
    pipeline.conformal_calibration_source_ = conformal_source
    metrics["conformal_calibration"] = conformal_summary

    if test_data is not None and len(test_data) > 0 and test_labels is not None:
        test_scored = score_records(pipeline, test_data)
        test_scores = np.asarray(test_scored["anomaly_score"], dtype=float)
        test_labels_binary = _coerce_binary_labels(test_labels)
        predictions = (test_scores >= float(getattr(pipeline, "final_anomaly_threshold_", getattr(pipeline.named_steps["model"], "offset_", 0.5)))).astype(int)
        metrics["test_metrics"] = _binary_classification_metrics(test_labels_binary, predictions)

    return pipeline, metrics


def save_pipeline(pipeline, output_path: str | Path) -> None:
    """Persist a trained pipeline to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    artifact_bytes = output_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    metadata = {
        "model_name": output_path.stem,
        "model_version": f"artifact-{artifact_sha256[:12]}",
        "artifact_sha256": artifact_sha256,
        "model_path": str(output_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_class": type(pipeline).__name__,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_pipeline(path: str | Path):
    """Load a trained pipeline from disk."""

    return joblib.load(Path(path))


def score_records(pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Return anomaly scores and flags for a batch of records."""

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    config = getattr(preprocessor, "config", PreprocessingConfig())
    risk_scoring_weights = getattr(pipeline, "risk_scoring_weights_", None)
    batch_start = time.perf_counter()
    transformed = preprocessor.transform(data)
    raw_scores = model.raw_anomaly_score(transformed)
    distribution_scores, distribution_batch_scores, distribution_alarm, distribution_threshold = _score_distribution_monitor(
        pipeline,
        transformed,
    )
    point_scores, point_zscore_frame, point_top_features, point_top_feature_zscores = _score_point_anomaly_head(
        pipeline,
        transformed,
    )
    contextual_scores, contextual_zscore_frame, contextual_top_features, contextual_top_feature_zscores, contextual_history_lengths = _score_contextual_anomaly_head(
        pipeline,
        data,
        transformed,
    )
    collective_scores, collective_group_frame, collective_top_groups, collective_top_group_errors = _score_collective_anomaly_head(
        pipeline,
        transformed,
    )
    sequence_output = _score_sequence_detector(pipeline, data)
    if sequence_output is None:
        sequence_scores = None
        sequence_history_length = np.zeros(len(data), dtype=int)
    else:
        sequence_scores, sequence_history_length = sequence_output
    final_scores, normalized_sequence_scores, sequence_weight = _blend_tabular_and_sequence_scores(
        raw_scores,
        sequence_scores,
        pipeline=pipeline,
    )
    drift_output = _build_grouped_drift_detection(
        data,
        final_scores,
        patient_id_col=getattr(config, "patient_id_col", "patient_id"),
        time_col=getattr(config, "encounter_time_col", "recorded_at"),
        baseline_window=getattr(config, "drift_baseline_window", 5),
        cusum_k=getattr(config, "drift_cusum_k", 0.25),
        cusum_h=getattr(config, "drift_cusum_h", 5.0),
        adwin_delta=getattr(config, "drift_adwin_delta", 0.01),
        adwin_min_window=getattr(config, "drift_adwin_min_window", 5),
    )
    calibration_scores = np.asarray(
        getattr(pipeline, "conformal_calibration_scores_", getattr(model, "conformal_calibration_scores_", [])),
        dtype=float,
    ).reshape(-1)
    conformal_alpha = float(getattr(pipeline, "conformal_alpha_", getattr(model, "conformal_alpha_", _CONFORMAL_ALPHA)))
    conformal_p_values = _conformal_p_values(calibration_scores, final_scores)
    decision_threshold = float(getattr(pipeline, "final_anomaly_threshold_", getattr(model, "offset_", 0.5)))
    decision_margin = decision_threshold - final_scores
    flags = np.where(final_scores >= decision_threshold, -1, 1)
    batch_elapsed = time.perf_counter() - batch_start
    per_patient_latency_ms = float((batch_elapsed / max(len(data), 1)) * 1000.0)
    batch_latency_ms = float(batch_elapsed * 1000.0)

    output = data.copy()
    if not point_zscore_frame.empty:
        output = pd.concat([output, point_zscore_frame], axis=1)
    if not contextual_zscore_frame.empty:
        output = pd.concat([output, contextual_zscore_frame], axis=1)
    if not collective_group_frame.empty:
        output = pd.concat([output, collective_group_frame], axis=1)
    if hasattr(model, "score_components"):
        output = pd.concat([output, model.score_components(transformed)], axis=1)
    if hasattr(model, "gate_weights"):
        try:
            output = pd.concat([output, model.gate_weights(transformed)], axis=1)
        except Exception:
            pass
    if hasattr(model, "estimators_") and "autoencoder" in model.estimators_:
        autoencoder = model.estimators_["autoencoder"]
        output["autoencoder_reconstruction_error"] = autoencoder.reconstruction_error(transformed)
        output["autoencoder_reconstruction_mae"] = autoencoder.reconstruction_mae(transformed)
    if hasattr(model, "estimators_") and "variational_autoencoder" in model.estimators_:
        variational_autoencoder = model.estimators_["variational_autoencoder"]
        output["variational_autoencoder_reconstruction_error"] = variational_autoencoder.reconstruction_error(transformed)
        output["variational_autoencoder_reconstruction_mae"] = variational_autoencoder.reconstruction_mae(transformed)
    if hasattr(model, "estimators_") and "ganomaly" in model.estimators_:
        ganomaly = model.estimators_["ganomaly"]
        output["ganomaly_reconstruction_error"] = ganomaly.reconstruction_error(transformed)
        output["ganomaly_latent_consistency_error"] = ganomaly.latent_consistency_error(transformed)
    if hasattr(model, "estimators_") and "anomaly_transformer" in model.estimators_:
        anomaly_transformer = model.estimators_["anomaly_transformer"]
        output["anomaly_transformer_reconstruction_error"] = anomaly_transformer.reconstruction_error(transformed)
        output["anomaly_transformer_attention_discrepancy"] = anomaly_transformer.attention_discrepancy(transformed)
    if hasattr(model, "estimators_") and "deep_svdd" in model.estimators_:
        deep_svdd = model.estimators_["deep_svdd"]
        output["deep_svdd_distance"] = deep_svdd.latent_distance(transformed)
    output["tabular_anomaly_score"] = raw_scores
    output["raw_anomaly_score"] = raw_scores
    output["point_anomaly_score"] = point_scores
    output["point_anomaly_max_abs_zscore"] = np.max(np.abs(point_zscore_frame.to_numpy(dtype=float)), axis=1) if not point_zscore_frame.empty else np.zeros(len(data), dtype=float)
    output["point_anomaly_mean_abs_zscore"] = np.mean(np.abs(point_zscore_frame.to_numpy(dtype=float)), axis=1) if not point_zscore_frame.empty else np.zeros(len(data), dtype=float)
    output["point_anomaly_top_feature"] = point_top_features.to_numpy(dtype=object) if len(point_top_features) else np.asarray([""] * len(data), dtype=object)
    output["point_anomaly_top_feature_zscore"] = point_top_feature_zscores
    output["contextual_anomaly_score"] = contextual_scores
    output["contextual_anomaly_max_abs_zscore"] = np.max(np.abs(contextual_zscore_frame.to_numpy(dtype=float)), axis=1) if not contextual_zscore_frame.empty else np.zeros(len(data), dtype=float)
    output["contextual_anomaly_mean_abs_zscore"] = np.mean(np.abs(contextual_zscore_frame.to_numpy(dtype=float)), axis=1) if not contextual_zscore_frame.empty else np.zeros(len(data), dtype=float)
    output["contextual_anomaly_top_feature"] = contextual_top_features.to_numpy(dtype=object) if len(contextual_top_features) else np.asarray([""] * len(data), dtype=object)
    output["contextual_anomaly_top_feature_zscore"] = contextual_top_feature_zscores
    output["contextual_anomaly_history_length"] = contextual_history_lengths
    output["collective_anomaly_score"] = collective_scores
    output["collective_anomaly_top_group"] = collective_top_groups.to_numpy(dtype=object) if len(collective_top_groups) else np.asarray([""] * len(data), dtype=object)
    output["collective_anomaly_top_group_error"] = collective_top_group_errors
    output["distribution_mmd_score"] = distribution_scores
    output["distribution_mmd_batch_score"] = distribution_batch_scores[0] if distribution_batch_scores.size else float("nan")
    output["distribution_mmd_threshold"] = distribution_threshold
    output["distribution_staleness_alarm"] = distribution_alarm
    output["sequence_anomaly_score"] = np.zeros(len(data), dtype=float) if sequence_scores is None else sequence_scores
    output["sequence_anomaly_score_normalized"] = normalized_sequence_scores
    output["sequence_history_length"] = sequence_history_length
    output["sequence_detector_weight"] = sequence_weight
    output["blended_anomaly_score"] = final_scores
    output["anomaly_score"] = final_scores
    output["score_stream_drift_alarm"] = drift_output["alarm"]
    output["score_stream_cusum_alarm"] = drift_output["cusum_alarm"]
    output["score_stream_adwin_alarm"] = drift_output["adwin_alarm"]
    output["score_stream_cusum_stat"] = drift_output["cusum_stat"]
    output["score_stream_adwin_stat"] = drift_output["adwin_stat"]
    output["score_stream_drift_change_index"] = drift_output["change_index"]
    output["score_stream_drift_method"] = drift_output["method"]
    output["conformal_nonconformity_score"] = final_scores
    output["conformal_p_value"] = conformal_p_values
    output["conformal_alpha"] = conformal_alpha
    output["conformal_calibration_source"] = getattr(
        pipeline,
        "conformal_calibration_source_",
        getattr(model, "conformal_calibration_source_", "unavailable"),
    )
    output["conformal_calibration_size"] = int(calibration_scores.size)
    output["conformal_nonconformity_source"] = getattr(
        pipeline,
        "conformal_calibration_nonconformity_",
        getattr(model, "conformal_calibration_nonconformity_", "unavailable"),
    )
    output["conformal_is_anomalous"] = np.where(np.isfinite(conformal_p_values), conformal_p_values <= conformal_alpha, False)
    output["conformal_assessment"] = np.where(
        np.isfinite(conformal_p_values),
        np.where(
            output["conformal_is_anomalous"],
            f"This record is anomalous at α={conformal_alpha:.2f} significance level.",
            f"This record is not anomalous at α={conformal_alpha:.2f} significance level.",
        ),
        "Conformal calibration unavailable; falling back to raw anomaly score.",
    )
    output["clinical_risk_score"] = output.apply(
        lambda row: _clinical_risk_component(row, anomaly_score=float(row["anomaly_score"]), weights=risk_scoring_weights),
        axis=1,
    )
    output["risk_score"] = output["clinical_risk_score"]
    output["risk_category"] = output["risk_score"].apply(lambda value: _risk_category_from_score(float(value) / 100.0))
    output["risk_level"] = output["risk_category"]
    output["alert_triggered"] = output["risk_category"].isin(["High", "Critical"])
    output["drift_alert_triggered"] = output["score_stream_drift_alarm"]
    output["staleness_alert_triggered"] = output["distribution_staleness_alarm"]
    output["drift_assessment"] = np.where(
        output["score_stream_drift_alarm"],
        "Score stream shifted from the patient baseline; review the trend even if individual visits stay below threshold.",
        "No persistent score drift detected in the current visit stream.",
    )
    output["staleness_assessment"] = np.where(
        output["distribution_staleness_alarm"],
        "Incoming batch no longer matches the training distribution; consider retraining or review before deployment.",
        "Incoming batch remains close to the training distribution.",
    )
    output["decision_margin"] = decision_margin
    output["anomaly_flag"] = flags
    output["is_anomaly"] = output["anomaly_flag"].map({1: False, -1: True})
    output["training_time_seconds"] = getattr(pipeline, "training_time_seconds_", float("nan"))
    output["training_time_ms"] = getattr(pipeline, "training_time_ms_", float("nan"))
    output["model_size_bytes"] = getattr(pipeline, "model_serialized_size_bytes_", float("nan"))
    output["estimated_ram_usage_bytes"] = getattr(pipeline, "model_estimated_ram_usage_bytes_", float("nan"))
    output["inference_batch_latency_ms"] = batch_latency_ms
    output["inference_latency_ms_per_patient"] = per_patient_latency_ms
    output["inference_throughput_rows_per_second"] = float(len(data) / batch_elapsed) if batch_elapsed > 0 else float("inf")
    return output
