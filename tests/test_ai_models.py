import tempfile
import unittest
from pathlib import Path

from src.ai_models import AIModelManager


class AIModelPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest", "xgboost", "logistic_regression"],
            }
        }

    def test_save_and_load_models_preserves_metadata(self):
        manager = AIModelManager(self.config)
        manager.feature_columns = ["feat1", "feat2"]
        manager.target_column = "Future_Direction_1"
        manager.model_performance = {
            "logistic_regression": {"test_accuracy": 0.73}
        }
        manager.training_history = [{"timestamp": "2026-04-02T20:00:00"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "named_model.joblib"
            manager.save_models(str(model_path))

            reloaded_manager = AIModelManager(self.config)
            success = reloaded_manager.load_models(str(model_path))

        self.assertTrue(success)
        self.assertEqual(reloaded_manager.feature_columns, ["feat1", "feat2"])
        self.assertEqual(reloaded_manager.target_column, "Future_Direction_1")
        self.assertIn("logistic_regression", reloaded_manager.model_performance)
        self.assertEqual(len(reloaded_manager.training_history), 1)


if __name__ == "__main__":
    unittest.main()
