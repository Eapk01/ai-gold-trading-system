import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.app_service import ResearchAppService
from src.model_tester import ModelTester


class FakeAIModelManager:
    def __init__(self, prediction_frame):
        self.models = {"random_forest": object()}
        self.target_column = "Future_Direction_1"
        self.prediction_frame = prediction_frame

    def predict_ensemble_batch(self, feature_data, feature_columns=None, method="voting"):
        return self.prediction_frame.reindex(feature_data.index)


class ModelTestServiceTests(unittest.TestCase):
    def test_run_model_test_requires_loaded_models(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.ai_model_manager = type("FakeManager", (), {"models": {}})()

        result = ResearchAppService.run_model_test(service)

        self.assertFalse(result["success"])
        self.assertIn("train or load", result["message"].lower())

    def test_run_model_test_requires_prepared_data(self):
        service = ResearchAppService.__new__(ResearchAppService)
        service.ai_model_manager = type("FakeManager", (), {"models": {"model": object()}})()
        service.feature_data = None
        service.selected_features = []

        result = ResearchAppService.run_model_test(service)

        self.assertFalse(result["success"])
        self.assertIn("prepare the data", result["message"].lower())

    def test_run_model_test_returns_summary_and_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                Path("reports").mkdir(parents=True, exist_ok=True)

                index = pd.date_range("2025-01-01", periods=4, freq="5min")
                feature_data = pd.DataFrame(
                    {
                        "feat1": [1.0, 2.0, 3.0, 4.0],
                        "Future_Direction_1": [1, 0, 1, 0],
                    },
                    index=index,
                )
                predictions = pd.DataFrame(
                    {
                        "is_valid": [True, True, True, True],
                        "prediction": [1, 0, 1, 0],
                        "confidence": [0.80, 0.61, 0.73, 0.69],
                    },
                    index=index,
                )

                service = ResearchAppService.__new__(ResearchAppService)
                service.ai_model_manager = FakeAIModelManager(predictions)
                service.feature_data = feature_data
                service.selected_features = ["feat1"]
                service.model_tester = ModelTester()
                service.loaded_model_path = "models/Main.joblib"
                service.latest_model_test_summary = {}
                service.latest_model_test_artifacts = {}

                result = ResearchAppService.run_model_test(service)

                self.assertTrue(result["success"])
                self.assertEqual(result["data"]["summary"]["scored_rows"], 4)
                self.assertIn("threshold_performance", result["data"])
                self.assertIn("confidence_buckets", result["data"])
                self.assertTrue(Path(result["artifacts"]["report_file"]).exists())
                self.assertTrue(Path(result["artifacts"]["evaluation_rows_file"]).exists())
            finally:
                os.chdir(old_cwd)
