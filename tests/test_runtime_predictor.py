import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.ai_models import AIModelManager
from src.research.catalog.search_presets import resolve_stage5_preset_definitions
from src.research.trainers import LSTMTrainer, TrainerRegistry
from src.runtime_predictor import LSTMPredictor, load_runtime_predictor


class RuntimePredictorTests(unittest.TestCase):
    def _base_config(self) -> dict:
        return {
            "ai_model": {
                "type": "ensemble",
                "models": ["random_forest"],
                "target_column": "Future_Direction_1",
                "lookback_periods": 4,
                "models_directory": "models",
            }
        }

    def test_stage5_preset_resolution_supports_lstm(self):
        presets = resolve_stage5_preset_definitions("lstm", ["random_forest"], ["balanced"])
        self.assertIn("balanced", presets)
        self.assertIn("lookback_window", presets["balanced"])
        self.assertIn("epochs", presets["balanced"])

    def test_runtime_predictor_loads_legacy_ensemble_artifact(self):
        config = self._base_config()
        manager = AIModelManager(config)
        frame = pd.DataFrame(
            {
                "feat1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "feat2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "Future_Direction_1": [0, 1, 0, 1, 0, 1],
            }
        )
        manager.train_ensemble_models(frame, ["feat1", "feat2"], target_column="Future_Direction_1")

        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "legacy.joblib"
            manager.save_models(str(artifact_path))

            loaded_predictor = load_runtime_predictor(config, str(artifact_path))
            expected = manager.predict_ensemble_batch(frame, feature_columns=["feat1", "feat2"], method="voting")
            actual = loaded_predictor.predict_batch(frame)

            self.assertEqual(loaded_predictor.required_feature_columns, ["feat1", "feat2"])
            self.assertTrue(expected.equals(actual))

    def test_lstm_trainer_registers_and_round_trips_runtime_artifact(self):
        config = self._base_config()
        index = pd.date_range("2025-01-01", periods=24, freq="5min")
        features = pd.DataFrame(
            {
                "feat1": [float(value % 4) for value in range(24)],
                "feat2": [float((value // 2) % 3) for value in range(24)],
            },
            index=index,
        )
        target = pd.Series(
            [1.0 if ((value % 6) >= 3) else 0.0 for value in range(24)],
            index=index,
            name="Future_Direction_1",
        )

        trainer = LSTMTrainer(
            config,
            target_column="Future_Direction_1",
            model_params={
                "lookback_window": 4,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "learning_rate": 0.01,
                "batch_size": 4,
                "epochs": 3,
                "early_stopping_patience": 2,
                "seed": 7,
            },
        )
        registry = TrainerRegistry()
        registry.register("lstm", trainer)
        built = registry.build("lstm")
        self.assertIsInstance(built, LSTMTrainer)

        outputs = trainer.fit_predict_segments(
            features.iloc[:16],
            target.iloc[:16],
            {
                "validation": features.iloc[16:20],
                "test": features.iloc[20:24],
            },
        )
        self.assertTrue(outputs["validation"].prediction.notna().all())
        self.assertTrue(outputs["test"].prediction.notna().all())
        self.assertEqual(outputs["validation"].metadata["trainer_name"], "lstm")

        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "lstm.joblib"
            artifact = trainer.fit_candidate_artifact(
                features.iloc[:20],
                target.iloc[:20],
                str(artifact_path),
                metadata={"experiment_id": "lstm_smoke"},
            )

            self.assertEqual(artifact.metadata["runtime_loader"], "lstm")
            predictor = load_runtime_predictor(config, str(artifact_path))
            self.assertIsInstance(predictor, LSTMPredictor)
            self.assertEqual(predictor.min_history_rows, 4)

            prediction_frame = predictor.predict_batch(features.iloc[:20])
            self.assertFalse(bool(prediction_frame.iloc[0]["is_valid"]))
            self.assertTrue(bool(prediction_frame.iloc[3]["is_valid"]))
            self.assertEqual(list(predictor.required_feature_columns), ["feat1", "feat2"])


if __name__ == "__main__":
    unittest.main()
