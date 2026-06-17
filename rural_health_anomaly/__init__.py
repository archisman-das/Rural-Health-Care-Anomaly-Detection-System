"""Rural health anomaly detection package."""

from .config import PreprocessingConfig
from .autoencoder import DeepAutoencoder
from .cnn_autoencoder import CNNAutoencoder
from .anomaly_transformer import AnomalyTransformer
from .temporal_sequence_detector import TemporalConvolutionalSequenceDetector
from .ganomaly import GANomaly
from .variational_autoencoder import VariationalAutoencoder
from .deep_svdd import DeepSVDD
from .ensemble import ParallelAnomalyEnsemble
from .evaluation import compare_labeled_score_columns, evaluate_labeled_scores, evaluate_score_columns, summarize_anomaly_scores
from .detectors import (
    IsolationForestAnomalyModel,
    LocalOutlierFactorAnomalyModel,
    OneClassSVMAnomalyModel,
)
from .pipeline import build_anomaly_pipeline
from .preprocessing import HealthcarePreprocessor, engineer_longitudinal_features
from .schema import (
    SCHEMA_CATEGORICAL_FEATURES,
    SCHEMA_EXCLUDED_TEXT_FEATURES,
    SCHEMA_LIST_NUMERIC_FEATURES,
    SCHEMA_MULTI_VALUE_FEATURES,
    SCHEMA_NUMERIC_FEATURES,
)

__all__ = [
    "PreprocessingConfig",
    "DeepAutoencoder",
    "CNNAutoencoder",
    "AnomalyTransformer",
    "TemporalConvolutionalSequenceDetector",
    "GANomaly",
    "VariationalAutoencoder",
    "DeepSVDD",
    "IsolationForestAnomalyModel",
    "OneClassSVMAnomalyModel",
    "LocalOutlierFactorAnomalyModel",
    "HealthcarePreprocessor",
    "ParallelAnomalyEnsemble",
    "evaluate_labeled_scores",
    "compare_labeled_score_columns",
    "evaluate_score_columns",
    "summarize_anomaly_scores",
    "build_anomaly_pipeline",
    "engineer_longitudinal_features",
    "SCHEMA_NUMERIC_FEATURES",
    "SCHEMA_CATEGORICAL_FEATURES",
    "SCHEMA_MULTI_VALUE_FEATURES",
    "SCHEMA_LIST_NUMERIC_FEATURES",
    "SCHEMA_EXCLUDED_TEXT_FEATURES",
]
