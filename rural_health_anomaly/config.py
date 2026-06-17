"""Configuration objects for preprocessing and training."""

from dataclasses import dataclass, field

from .schema import (
    SCHEMA_CATEGORICAL_FEATURES,
    SCHEMA_LIST_NUMERIC_FEATURES,
    SCHEMA_MULTI_VALUE_FEATURES,
    SCHEMA_NUMERIC_FEATURES,
)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    patient_id_col: str = "patient_id"
    encounter_time_col: str = "recorded_at"
    numeric_features: list[str] = field(default_factory=lambda: SCHEMA_NUMERIC_FEATURES.copy())
    categorical_features: list[str] = field(default_factory=lambda: SCHEMA_CATEGORICAL_FEATURES.copy())
    multi_value_features: list[str] = field(default_factory=lambda: SCHEMA_MULTI_VALUE_FEATURES.copy())
    list_numeric_features: list[str] = field(default_factory=lambda: SCHEMA_LIST_NUMERIC_FEATURES.copy())
    rolling_windows_days: tuple[int, ...] = (7, 30)
    lag_steps: tuple[int, ...] = (1,)
    interaction_terms: tuple[tuple[str, str], ...] = ()
    sequence_window_size: int = 4
    sequence_stride: int = 1
    sequence_detector_filters: int = 16
    sequence_detector_kernel_size: int = 3
    sequence_detector_latent_dim: int = 8
    sequence_detector_dropout: float = 0.1
    sequence_detector_learning_rate: float = 1e-3
    sequence_detector_batch_size: int = 32
    sequence_detector_max_epochs: int = 80
    sequence_detector_patience: int = 10
    sequence_detector_l2: float = 1e-5
    sequence_detector_random_state: int = 42
    sequence_detector_verbose: bool = False
    sequence_detector_weight: float | None = None
    drift_baseline_window: int = 5
    drift_cusum_k: float = 0.25
    drift_cusum_h: float = 5.0
    drift_adwin_delta: float = 0.01
    drift_adwin_min_window: int = 5
    distribution_monitor_reference_size: int = 256
    distribution_monitor_calibration_batch_size: int = 32
    distribution_monitor_bootstrap_trials: int = 64
    distribution_monitor_threshold_quantile: float = 0.95
    distribution_monitor_kernel_gamma: float | None = None
    scaler: str = "standard"  # "standard" or "minmax"
    apply_pca: bool = True
    pca_feature_threshold: int = 50
    pca_variance_threshold: float = 0.95
    knn_neighbors: int = 5
    ensemble_n_jobs: int = -1
    ensemble_fusion_strategy: str = "weighted_average"
    ensemble_max_score_threshold: float = 0.8
    calibrate_threshold: bool = True
    calibration_min_samples: int = 25
    ensemble_fusion_weights: dict[str, float] | None = None
    stacking_meta_model_type: str = "mlp"
    stacking_hidden_layer_sizes: tuple[int, ...] = (32, 16)
    stacking_alpha: float = 1e-4
    stacking_learning_rate_init: float = 1e-3
    stacking_max_iter: int = 500
    stacking_random_state: int = 42
    stacking_verbose: bool = False
    moe_gate_hidden_dim: int = 32
    moe_gate_dropout: float = 0.1
    moe_gate_learning_rate: float = 1e-3
    moe_gate_batch_size: int = 32
    moe_gate_max_epochs: int = 80
    moe_gate_patience: int = 10
    moe_gate_l2: float = 1e-5
    moe_gate_random_state: int = 42
    moe_gate_verbose: bool = False
    cnn_autoencoder_weight: float | None = None
    anomaly_transformer_weight: float | None = None
    vae_weight: float | None = None
    risk_scoring_weights: dict[str, float] | None = None
    isolation_forest_n_estimators: int = 300
    isolation_forest_contamination: float = 0.05
    isolation_forest_max_samples: int | str = "auto"
    isolation_forest_max_features: float = 1.0
    isolation_forest_bootstrap: bool = False
    isolation_forest_random_state: int = 42
    isolation_forest_n_jobs: int = -1
    one_class_svm_nu: float | None = None
    one_class_svm_kernel: str = "rbf"
    one_class_svm_gamma: str | float = "scale"
    local_outlier_factor_n_neighbors: int = 20
    local_outlier_factor_contamination: float | None = None
    local_outlier_factor_n_jobs: int = -1
    autoencoder_latent_dim: int = 8
    autoencoder_dropout: float = 0.2
    autoencoder_learning_rate: float = 1e-3
    autoencoder_batch_size: int = 32
    autoencoder_threshold_percentile: float = 97.5
    autoencoder_validation_fraction: float = 0.2
    autoencoder_max_epochs: int = 80
    autoencoder_patience: int = 10
    autoencoder_l2: float = 1e-5
    autoencoder_random_state: int = 42
    autoencoder_verbose: bool = False
    ganomaly_hidden_dim: int = 64
    ganomaly_latent_dim: int = 8
    ganomaly_dropout: float = 0.2
    ganomaly_learning_rate: float = 1e-3
    ganomaly_batch_size: int = 32
    ganomaly_consistency_weight: float = 1.0
    ganomaly_threshold_percentile: float = 97.5
    ganomaly_validation_fraction: float = 0.2
    ganomaly_max_epochs: int = 80
    ganomaly_patience: int = 10
    ganomaly_l2: float = 1e-5
    ganomaly_random_state: int = 42
    ganomaly_verbose: bool = False
    anomaly_transformer_hidden_dim: int = 64
    anomaly_transformer_latent_dim: int = 8
    anomaly_transformer_dropout: float = 0.2
    anomaly_transformer_learning_rate: float = 1e-3
    anomaly_transformer_batch_size: int = 32
    anomaly_transformer_attention_weight: float = 0.5
    anomaly_transformer_attention_temperature: float = 1.0
    anomaly_transformer_threshold_percentile: float = 97.5
    anomaly_transformer_validation_fraction: float = 0.2
    anomaly_transformer_max_epochs: int = 80
    anomaly_transformer_patience: int = 10
    anomaly_transformer_l2: float = 1e-5
    anomaly_transformer_random_state: int = 42
    anomaly_transformer_verbose: bool = False
    vae_hidden_dim: int = 64
    vae_latent_dim: int = 8
    vae_dropout: float = 0.2
    vae_learning_rate: float = 1e-3
    vae_batch_size: int = 32
    vae_beta: float = 1.0
    vae_threshold_percentile: float = 97.5
    vae_validation_fraction: float = 0.2
    vae_max_epochs: int = 80
    vae_patience: int = 10
    vae_l2: float = 1e-5
    vae_random_state: int = 42
    vae_verbose: bool = False
    cnn_autoencoder_filters: int = 8
    cnn_autoencoder_kernel_size: int = 3
    cnn_autoencoder_latent_dim: int = 8
    cnn_autoencoder_dropout: float = 0.2
    cnn_autoencoder_learning_rate: float = 1e-3
    cnn_autoencoder_batch_size: int = 32
    cnn_autoencoder_threshold_percentile: float = 97.5
    cnn_autoencoder_validation_fraction: float = 0.2
    cnn_autoencoder_max_epochs: int = 80
    cnn_autoencoder_patience: int = 10
    cnn_autoencoder_l2: float = 1e-5
    cnn_autoencoder_random_state: int = 42
    cnn_autoencoder_verbose: bool = False
    ganomaly_weight: float | None = None
    deep_svdd_nu: float = 0.05
    deep_svdd_center_fixed: bool = True
    deep_svdd_architecture: str = "mlp"
    deep_svdd_latent_dim: int = 8
    deep_svdd_learning_rate: float = 1e-3
    deep_svdd_batch_size: int = 32
    deep_svdd_max_epochs: int = 60
    deep_svdd_validation_fraction: float = 0.2
    deep_svdd_pretrain_autoencoder: bool = True
    deep_svdd_pretrain_epochs: int = 25
    deep_svdd_pretrain_dropout: float = 0.2
    deep_svdd_pretrain_learning_rate: float = 1e-3
    deep_svdd_pretrain_batch_size: int = 32
    deep_svdd_random_state: int = 42
    deep_svdd_verbose: bool = False
