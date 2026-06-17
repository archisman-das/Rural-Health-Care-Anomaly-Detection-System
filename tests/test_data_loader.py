import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rural_health_anomaly.training import load_tabular_data


class DataLoaderTests(unittest.TestCase):
    def test_csv_loader_normalizes_lists_and_datetimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            raw = pd.DataFrame(
                {
                    "patient_id": ["P1"],
                    "recorded_at": ["2026-06-01T09:00:00+05:30"],
                    "gender": [" none "],
                    "comorbidities": ['["diabetes", "hypertension"]'],
                    "current_medications": ["metformin, amlodipine"],
                    "days_between_visits_trend": ["[7, 14, 21]"],
                }
            )
            raw.to_csv(csv_path, index=False)

            loaded = load_tabular_data(csv_path)

            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["recorded_at"]))
            self.assertIsNone(loaded.loc[0, "gender"])
            self.assertEqual(loaded.loc[0, "comorbidities"], ["diabetes", "hypertension"])
            self.assertEqual(loaded.loc[0, "current_medications"], ["metformin", "amlodipine"])
            self.assertEqual(loaded.loc[0, "days_between_visits_trend"], [7, 14, 21])


if __name__ == "__main__":
    unittest.main()
