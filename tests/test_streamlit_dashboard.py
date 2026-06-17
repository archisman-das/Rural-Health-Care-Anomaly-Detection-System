from __future__ import annotations

import unittest

import pandas as pd

from rural_health_anomaly.streamlit_dashboard import (
    build_agreement_summary,
    build_risk_map_frame,
    build_top_alert_feature_views,
    normalize_explain_batch_payload,
    parse_feature_explanations,
)


class StreamlitDashboardTests(unittest.TestCase):
    def test_parse_feature_explanations_orders_by_absolute_value(self):
        blob = """
        [
          {"feature": "glucose", "shap_value": 0.4, "absolute_shap_value": 0.4, "source_columns": ["glucose_fasting_mg_dl"], "feature_type": "direct", "method": "ablation_fallback"},
          {"feature": "age", "shap_value": -0.8, "absolute_shap_value": 0.8, "source_columns": ["age_years"], "feature_type": "direct", "method": "ablation_fallback"}
        ]
        """

        frame = parse_feature_explanations(blob)

        self.assertEqual(frame.iloc[0]["feature"], "age")
        self.assertEqual(frame.iloc[1]["feature"], "glucose")
        self.assertEqual(frame.iloc[0]["source_columns"], ["age_years"])

    def test_normalize_explain_batch_payload_flattens_results(self):
        payload = {
            "count": 1,
            "results": [
                {
                    "patient_index": 2,
                    "prediction": {"patient_id": "P2", "anomaly_score": 0.91, "risk_level": "High"},
                    "explanation": {
                        "method": "ablation_fallback",
                        "top_k": 1,
                        "feature_explanations": [
                            {
                                "feature": "glucose",
                                "shap_value": 0.7,
                                "absolute_shap_value": 0.7,
                                "source_columns": ["glucose_fasting_mg_dl"],
                                "feature_type": "direct",
                                "method": "ablation_fallback",
                            }
                        ],
                    },
                }
            ],
        }

        frame = normalize_explain_batch_payload(payload)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["patient_index"], 2)
        self.assertEqual(frame.iloc[0]["patient_id"], "P2")
        self.assertEqual(frame.iloc[0]["anomaly_score"], 0.91)
        self.assertIn("feature_explanations", frame.columns)

    def test_build_risk_map_and_alert_views_use_expected_rows(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["P1", "P2"],
                "recorded_at": ["2026-06-01T09:00:00+05:30", "2026-06-02T09:00:00+05:30"],
                "anomaly_score": [0.2, 0.91],
                "risk_level": ["Low", "High"],
                "alert_triggered": [False, True],
                "feature_explanations": [
                    None,
                    [
                        {
                            "feature": "glucose",
                            "shap_value": 0.7,
                            "absolute_shap_value": 0.7,
                            "source_columns": ["glucose_fasting_mg_dl"],
                            "feature_type": "direct",
                            "method": "ablation_fallback",
                        }
                    ],
                ],
                "isolation_forest_anomaly_score": [0.1, 0.9],
                "autoencoder_anomaly_score": [0.15, 0.92],
                "deep_svdd_anomaly_score": [0.12, 0.88],
            }
        )

        risk_map = build_risk_map_frame(frame)
        agreement = build_agreement_summary(frame, threshold=0.7)
        views = build_top_alert_feature_views(frame, top_alerts=1, top_features=1)

        self.assertIn("patient_label", risk_map.columns)
        self.assertIn("risk_level", risk_map.columns)
        self.assertEqual(risk_map.iloc[1]["patient_label"], "P2")
        self.assertIsNotNone(agreement["agreement"])
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["patient_id"], "P2")
        self.assertEqual(views[0]["features"][0]["feature"], "glucose")


if __name__ == "__main__":
    unittest.main()
