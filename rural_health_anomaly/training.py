"""Training helpers for the rural health anomaly pipeline."""

from __future__ import annotations

import ast
import hashlib
import json
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

from .config import PreprocessingConfig
from .pipeline import build_anomaly_pipeline
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
    pipeline.model_serialized_size_bytes_ = int(len(pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)))
    pipeline.model_estimated_ram_usage_bytes_ = int(_estimate_object_size_bytes(pipeline))
    if len(data) > 0:
        sample_size = min(25, len(data))
        pipeline.explain_background_ = data.sample(n=sample_size, random_state=42).copy() if len(data) > sample_size else data.copy()
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
        model = pipeline.named_steps["model"]
        transformed = pipeline.named_steps["preprocessor"].transform(validation_data)
        validation_scores = np.asarray(model.raw_anomaly_score(transformed), dtype=float)
        validation_labels_binary = _coerce_binary_labels(validation_labels)
        threshold, calibration_metrics = _calibrate_threshold_from_scores(validation_scores, validation_labels_binary)
        model.offset_ = float(threshold)
        model.calibrated_threshold_ = float(threshold)
        model.calibration_metrics_ = calibration_metrics
        model.calibration_source_ = "validation"
        metrics["validation_calibration"] = calibration_metrics

    if test_data is not None and len(test_data) > 0 and test_labels is not None:
        model = pipeline.named_steps["model"]
        transformed = pipeline.named_steps["preprocessor"].transform(test_data)
        test_scores = np.asarray(model.raw_anomaly_score(transformed), dtype=float)
        test_labels_binary = _coerce_binary_labels(test_labels)
        predictions = (test_scores >= float(getattr(model, "offset_", 0.5))).astype(int)
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
    risk_scoring_weights = getattr(pipeline, "risk_scoring_weights_", None)
    batch_start = time.perf_counter()
    transformed = pipeline.named_steps["preprocessor"].transform(data)
    raw_scores = model.raw_anomaly_score(transformed)
    decision_margin = pipeline.decision_function(data)
    flags = pipeline.predict(data)
    batch_elapsed = time.perf_counter() - batch_start
    per_patient_latency_ms = float((batch_elapsed / max(len(data), 1)) * 1000.0)
    batch_latency_ms = float(batch_elapsed * 1000.0)

    output = data.copy()
    if hasattr(model, "score_components"):
        output = pd.concat([output, model.score_components(transformed)], axis=1)
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
    output["raw_anomaly_score"] = raw_scores
    output["anomaly_score"] = raw_scores
    output["clinical_risk_score"] = output.apply(
        lambda row: _clinical_risk_component(row, anomaly_score=float(row["anomaly_score"]), weights=risk_scoring_weights),
        axis=1,
    )
    output["risk_score"] = output["clinical_risk_score"]
    output["risk_category"] = output["risk_score"].apply(lambda value: _risk_category_from_score(float(value) / 100.0))
    output["risk_level"] = output["risk_category"]
    output["alert_triggered"] = output["risk_category"].isin(["High", "Critical"])
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
