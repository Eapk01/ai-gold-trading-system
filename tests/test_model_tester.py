import unittest

import pandas as pd

from src.model_tester import ModelTester


class ModelTesterTests(unittest.TestCase):
    def test_evaluate_excludes_invalid_rows_and_reports_coverage(self):
        prepared_data = pd.DataFrame(
            {"Future_Direction_1": [1, 0, 1, 0]},
            index=pd.date_range("2025-01-01", periods=4, freq="5min"),
        )
        prediction_frame = pd.DataFrame(
            {
                "is_valid": [True, False, True, True],
                "prediction": [1, 0, 0, 0],
                "confidence": [0.80, 0.55, 0.65, 0.72],
            },
            index=prepared_data.index,
        )

        result = ModelTester().evaluate(prepared_data, prediction_frame, prepared_data["Future_Direction_1"])

        self.assertEqual(result.summary["total_rows"], 4)
        self.assertEqual(result.summary["valid_prediction_rows"], 3)
        self.assertEqual(result.summary["invalid_rows"], 1)
        self.assertEqual(result.summary["scored_rows"], 3)
        self.assertAlmostEqual(result.summary["coverage_rate"], 0.75)

    def test_evaluate_computes_metrics_thresholds_and_buckets(self):
        prepared_data = pd.DataFrame(
            {"Future_Direction_1": [1, 0, 1, 0]},
            index=pd.date_range("2025-01-01", periods=4, freq="5min"),
        )
        prediction_frame = pd.DataFrame(
            {
                "is_valid": [True, True, True, True],
                "prediction": [1, 1, 1, 0],
                "confidence": [0.52, 0.58, 0.67, 0.73],
            },
            index=prepared_data.index,
        )

        result = ModelTester().evaluate(prepared_data, prediction_frame, prepared_data["Future_Direction_1"])

        self.assertAlmostEqual(result.summary["accuracy"], 0.75)
        self.assertAlmostEqual(result.summary["precision"], 2 / 3)
        self.assertAlmostEqual(result.summary["recall"], 1.0)
        self.assertAlmostEqual(result.summary["f1"], 0.8)
        self.assertEqual(result.summary["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 0, "tp": 2})

        threshold_060 = result.threshold_performance.loc[result.threshold_performance["threshold"] == 0.60].iloc[0]
        self.assertEqual(int(threshold_060["rows_kept"]), 2)
        self.assertAlmostEqual(float(threshold_060["accuracy"]), 1.0)

        bucket_055_060 = result.confidence_buckets.loc[result.confidence_buckets["bucket"] == "0.55-0.60"].iloc[0]
        self.assertEqual(int(bucket_055_060["rows"]), 1)
        self.assertAlmostEqual(float(bucket_055_060["accuracy"]), 0.0)
