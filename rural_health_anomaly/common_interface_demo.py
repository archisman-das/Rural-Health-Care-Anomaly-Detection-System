"""Small demo of the shared fit/score anomaly model interface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    DeepAutoencoder,
    DeepSVDD,
    HealthcarePreprocessor,
    IsolationForestAnomalyModel,
    LocalOutlierFactorAnomalyModel,
    OneClassSVMAnomalyModel,
    PreprocessingConfig,
    VariationalAutoencoder,
)


def build_demo_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "recorded_at": [
                "2026-06-01T09:00:00+05:30",
                "2026-06-08T09:00:00+05:30",
                "2026-06-03T10:15:00+05:30",
                "2026-06-10T10:15:00+05:30",
            ],
            "age_years": [54, 54, 61, 61],
            "gender": ["female", "female", "male", "male"],
            "heart_rate_bpm": [78, 81, 92, 95],
            "systolic_bp_mmhg": [118, 120, 136, 140],
            "diastolic_bp_mmhg": [76, 78, 88, 90],
            "glucose_fasting_mg_dl": [92, 110, 140, 138],
            "comorbidities": [["diabetes"], ["diabetes"], ["tb"], ["tb"]],
            "current_medications": [["metformin"], ["metformin"], ["isoniazid"], ["isoniazid"]],
            "visits_last_90_days": [3, 4, 2, 3],
            "symptom_duration_days": [12, 11, 8, 10],
        }
    )


def main() -> None:
    data = build_demo_data()
    config = PreprocessingConfig(apply_pca=False)
    preprocessor = HealthcarePreprocessor(config)
    X = preprocessor.fit_transform(data)

    models = [
        ("isolation_forest", IsolationForestAnomalyModel(n_estimators=50, random_state=7)),
        ("one_class_svm", OneClassSVMAnomalyModel(nu=0.05)),
        ("local_outlier_factor", LocalOutlierFactorAnomalyModel(n_neighbors=2, contamination=0.05)),
        ("autoencoder", DeepAutoencoder(latent_dim=4, max_epochs=5, patience=2, random_state=7)),
        ("variational_autoencoder", VariationalAutoencoder(hidden_dim=16, latent_dim=4, max_epochs=5, patience=2, random_state=7)),
        ("deep_svdd", DeepSVDD(latent_dim=4, max_epochs=5, pretrain_autoencoder=False, random_state=7)),
    ]

    summary = []
    for name, model in models:
        model.fit(X)
        scores = model.score(X)
        summary.append(
            {
                "model": name,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores, ddof=0)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
            }
        )

    results = pd.DataFrame(summary)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
