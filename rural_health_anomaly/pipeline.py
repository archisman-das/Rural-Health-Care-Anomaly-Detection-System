"""Model pipeline builders."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from .config import PreprocessingConfig
from .ensemble import ParallelAnomalyEnsemble
from .preprocessing import HealthcarePreprocessor


def _build_anomaly_ensemble(config: PreprocessingConfig) -> ParallelAnomalyEnsemble:
    """Build the parallel anomaly ensemble from config."""

    fusion_weights = dict(config.ensemble_fusion_weights or {})
    if config.cnn_autoencoder_weight is not None:
        fusion_weights["cnn_autoencoder"] = config.cnn_autoencoder_weight
    if config.anomaly_transformer_weight is not None:
        fusion_weights["anomaly_transformer"] = config.anomaly_transformer_weight
    if config.ganomaly_weight is not None:
        fusion_weights["ganomaly"] = config.ganomaly_weight
    if config.vae_weight is not None:
        fusion_weights["variational_autoencoder"] = config.vae_weight

    return ParallelAnomalyEnsemble(
        contamination=config.isolation_forest_contamination,
        n_jobs=config.ensemble_n_jobs,
        fusion_strategy=config.ensemble_fusion_strategy,
        max_score_threshold=config.ensemble_max_score_threshold,
        calibrate_threshold=config.calibrate_threshold,
        calibration_min_samples=config.calibration_min_samples,
        fusion_weights=fusion_weights or None,
        stacking_meta_model_type=config.stacking_meta_model_type,
        stacking_hidden_layer_sizes=config.stacking_hidden_layer_sizes,
        stacking_alpha=config.stacking_alpha,
        stacking_learning_rate_init=config.stacking_learning_rate_init,
        stacking_max_iter=config.stacking_max_iter,
        stacking_random_state=config.stacking_random_state,
        stacking_verbose=config.stacking_verbose,
        moe_gate_hidden_dim=config.moe_gate_hidden_dim,
        moe_gate_dropout=config.moe_gate_dropout,
        moe_gate_learning_rate=config.moe_gate_learning_rate,
        moe_gate_batch_size=config.moe_gate_batch_size,
        moe_gate_max_epochs=config.moe_gate_max_epochs,
        moe_gate_patience=config.moe_gate_patience,
        moe_gate_l2=config.moe_gate_l2,
        moe_gate_random_state=config.moe_gate_random_state,
        moe_gate_verbose=config.moe_gate_verbose,
        isolation_forest_n_estimators=config.isolation_forest_n_estimators,
        isolation_forest_max_samples=config.isolation_forest_max_samples,
        isolation_forest_max_features=config.isolation_forest_max_features,
        isolation_forest_bootstrap=config.isolation_forest_bootstrap,
        isolation_forest_random_state=config.isolation_forest_random_state,
        isolation_forest_n_jobs=config.isolation_forest_n_jobs,
        one_class_svm_nu=config.one_class_svm_nu,
        one_class_svm_kernel=config.one_class_svm_kernel,
        one_class_svm_gamma=config.one_class_svm_gamma,
        local_outlier_factor_n_neighbors=config.local_outlier_factor_n_neighbors,
        local_outlier_factor_contamination=config.local_outlier_factor_contamination,
        local_outlier_factor_n_jobs=config.local_outlier_factor_n_jobs,
        autoencoder_latent_dim=config.autoencoder_latent_dim,
        autoencoder_dropout=config.autoencoder_dropout,
        autoencoder_learning_rate=config.autoencoder_learning_rate,
        autoencoder_batch_size=config.autoencoder_batch_size,
        autoencoder_threshold_percentile=config.autoencoder_threshold_percentile,
        autoencoder_validation_fraction=config.autoencoder_validation_fraction,
        autoencoder_max_epochs=config.autoencoder_max_epochs,
        autoencoder_patience=config.autoencoder_patience,
        autoencoder_l2=config.autoencoder_l2,
        autoencoder_random_state=config.autoencoder_random_state,
        autoencoder_verbose=config.autoencoder_verbose,
        anomaly_transformer_hidden_dim=config.anomaly_transformer_hidden_dim,
        anomaly_transformer_latent_dim=config.anomaly_transformer_latent_dim,
        anomaly_transformer_dropout=config.anomaly_transformer_dropout,
        anomaly_transformer_learning_rate=config.anomaly_transformer_learning_rate,
        anomaly_transformer_batch_size=config.anomaly_transformer_batch_size,
        anomaly_transformer_attention_weight=config.anomaly_transformer_attention_weight,
        anomaly_transformer_attention_temperature=config.anomaly_transformer_attention_temperature,
        anomaly_transformer_threshold_percentile=config.anomaly_transformer_threshold_percentile,
        anomaly_transformer_validation_fraction=config.anomaly_transformer_validation_fraction,
        anomaly_transformer_max_epochs=config.anomaly_transformer_max_epochs,
        anomaly_transformer_patience=config.anomaly_transformer_patience,
        anomaly_transformer_l2=config.anomaly_transformer_l2,
        anomaly_transformer_random_state=config.anomaly_transformer_random_state,
        anomaly_transformer_verbose=config.anomaly_transformer_verbose,
        ganomaly_hidden_dim=config.ganomaly_hidden_dim,
        ganomaly_latent_dim=config.ganomaly_latent_dim,
        ganomaly_dropout=config.ganomaly_dropout,
        ganomaly_learning_rate=config.ganomaly_learning_rate,
        ganomaly_batch_size=config.ganomaly_batch_size,
        ganomaly_consistency_weight=config.ganomaly_consistency_weight,
        ganomaly_threshold_percentile=config.ganomaly_threshold_percentile,
        ganomaly_validation_fraction=config.ganomaly_validation_fraction,
        ganomaly_max_epochs=config.ganomaly_max_epochs,
        ganomaly_patience=config.ganomaly_patience,
        ganomaly_l2=config.ganomaly_l2,
        ganomaly_random_state=config.ganomaly_random_state,
        ganomaly_verbose=config.ganomaly_verbose,
        vae_hidden_dim=config.vae_hidden_dim,
        vae_latent_dim=config.vae_latent_dim,
        vae_dropout=config.vae_dropout,
        vae_learning_rate=config.vae_learning_rate,
        vae_batch_size=config.vae_batch_size,
        vae_beta=config.vae_beta,
        vae_threshold_percentile=config.vae_threshold_percentile,
        vae_validation_fraction=config.vae_validation_fraction,
        vae_max_epochs=config.vae_max_epochs,
        vae_patience=config.vae_patience,
        vae_l2=config.vae_l2,
        vae_random_state=config.vae_random_state,
        vae_verbose=config.vae_verbose,
        cnn_autoencoder_filters=config.cnn_autoencoder_filters,
        cnn_autoencoder_kernel_size=config.cnn_autoencoder_kernel_size,
        cnn_autoencoder_latent_dim=config.cnn_autoencoder_latent_dim,
        cnn_autoencoder_dropout=config.cnn_autoencoder_dropout,
        cnn_autoencoder_learning_rate=config.cnn_autoencoder_learning_rate,
        cnn_autoencoder_batch_size=config.cnn_autoencoder_batch_size,
        cnn_autoencoder_threshold_percentile=config.cnn_autoencoder_threshold_percentile,
        cnn_autoencoder_validation_fraction=config.cnn_autoencoder_validation_fraction,
        cnn_autoencoder_max_epochs=config.cnn_autoencoder_max_epochs,
        cnn_autoencoder_patience=config.cnn_autoencoder_patience,
        cnn_autoencoder_l2=config.cnn_autoencoder_l2,
        cnn_autoencoder_random_state=config.cnn_autoencoder_random_state,
        cnn_autoencoder_verbose=config.cnn_autoencoder_verbose,
        deep_svdd_nu=config.deep_svdd_nu,
        deep_svdd_center_fixed=config.deep_svdd_center_fixed,
        deep_svdd_architecture=config.deep_svdd_architecture,
        deep_svdd_latent_dim=config.deep_svdd_latent_dim,
        deep_svdd_learning_rate=config.deep_svdd_learning_rate,
        deep_svdd_batch_size=config.deep_svdd_batch_size,
        deep_svdd_max_epochs=config.deep_svdd_max_epochs,
        deep_svdd_validation_fraction=config.deep_svdd_validation_fraction,
        deep_svdd_pretrain_autoencoder=config.deep_svdd_pretrain_autoencoder,
        deep_svdd_pretrain_epochs=config.deep_svdd_pretrain_epochs,
        deep_svdd_pretrain_dropout=config.deep_svdd_pretrain_dropout,
        deep_svdd_pretrain_learning_rate=config.deep_svdd_pretrain_learning_rate,
        deep_svdd_pretrain_batch_size=config.deep_svdd_pretrain_batch_size,
        deep_svdd_random_state=config.deep_svdd_random_state,
        deep_svdd_verbose=config.deep_svdd_verbose,
    )


def build_anomaly_pipeline(
    config: PreprocessingConfig | None = None,
    estimator=None,
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing plus anomaly model."""

    config = config or PreprocessingConfig()
    estimator = estimator or _build_anomaly_ensemble(config)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", HealthcarePreprocessor(config)),
            ("model", estimator),
        ]
    )
    pipeline.risk_scoring_weights_ = dict(config.risk_scoring_weights or {})
    return pipeline
