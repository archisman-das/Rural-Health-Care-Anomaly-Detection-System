"""Backward-compatible redirect to the package preprocessing API."""

from rural_health_anomaly import (
    HealthcarePreprocessor,
    PreprocessingConfig,
    SCHEMA_CATEGORICAL_FEATURES,
    SCHEMA_EXCLUDED_TEXT_FEATURES,
    SCHEMA_LIST_NUMERIC_FEATURES,
    SCHEMA_MULTI_VALUE_FEATURES,
    SCHEMA_NUMERIC_FEATURES,
    build_anomaly_pipeline,
    engineer_longitudinal_features,
)

__all__ = [
    "HealthcarePreprocessor",
    "PreprocessingConfig",
    "SCHEMA_NUMERIC_FEATURES",
    "SCHEMA_CATEGORICAL_FEATURES",
    "SCHEMA_MULTI_VALUE_FEATURES",
    "SCHEMA_LIST_NUMERIC_FEATURES",
    "SCHEMA_EXCLUDED_TEXT_FEATURES",
    "build_anomaly_pipeline",
    "engineer_longitudinal_features",
]
