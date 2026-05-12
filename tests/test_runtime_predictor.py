import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import pandas as pd

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.ai_models import AIModelManager
from src.research.catalog.search_presets import resolve_search_preset_definitions
from src.research.trainers import LSTMTrainer, TrainerRegistry
from src.runtime_predictor import LSTMPredictor, load_runtime_predictor, torch


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

    def test_search_preset_resolution_supports_lstm(self):
        presets = resolve_search_preset_definitions("lstm", ["random_forest"], ["balanced"])
        self.assertIn("balanced", presets)
        self.assertEqual(presets["balanced"]["architecture_name"], "lstm_v2_dense_head")
        self.assertEqual(presets["balanced"]["feature_mode"], "combined")
        self.assertEqual(len(presets["balanced"]["variants"]), 6)
        self.assertIn("lookback_window", presets["balanced"])
        self.assertIn("dense_hidden_size", presets["balanced"])
        self.assertIn("epochs", presets["balanced"])

    def test_lstm_device_policy_resolves_auto_cpu_when_cuda_unavailable(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=False):
            device_info = LSTMTrainer.resolve_training_device("auto")

        self.assertEqual(device_info["requested_device"], "auto")
        self.assertEqual(device_info["training_device"], "cpu")
        self.assertFalse(device_info["cuda_available"])
        self.assertFalse(device_info["cuda_usable"])
        self.assertEqual(device_info["cuda_device_name"], "")
        self.assertIn("torch_version", device_info)

    def test_lstm_device_policy_resolves_auto_cuda_when_available(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda,
            "get_device_name",
            return_value="Mock CUDA",
        ), patch.object(LSTMTrainer, "cuda_probe_error", return_value=""):
            device_info = LSTMTrainer.resolve_training_device("auto")

        self.assertEqual(device_info["requested_device"], "auto")
        self.assertEqual(device_info["training_device"], "cuda")
        self.assertTrue(device_info["cuda_available"])
        self.assertTrue(device_info["cuda_usable"])
        self.assertEqual(device_info["cuda_device_name"], "Mock CUDA")
        self.assertIn("torch_version", device_info)

    def test_lstm_device_policy_auto_falls_back_when_cuda_kernel_probe_fails(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda,
            "get_device_name",
            return_value="Mock CUDA",
        ), patch.object(
            LSTMTrainer,
            "cuda_probe_error",
            return_value="CUDA error: no kernel image is available for execution on the device",
        ):
            device_info = LSTMTrainer.resolve_training_device("auto")

        self.assertEqual(device_info["requested_device"], "auto")
        self.assertEqual(device_info["training_device"], "cpu")
        self.assertTrue(device_info["cuda_available"])
        self.assertFalse(device_info["cuda_usable"])
        self.assertIn("no kernel image", device_info["cuda_probe_error"])

    def test_lstm_device_policy_respects_cpu_override(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(LSTMTrainer, "cuda_probe_error", return_value=""):
            device_info = LSTMTrainer.resolve_training_device("cpu")

        self.assertEqual(device_info["requested_device"], "cpu")
        self.assertEqual(device_info["training_device"], "cpu")
        self.assertTrue(device_info["cuda_available"])

    def test_lstm_device_policy_rejects_cuda_when_kernel_probe_fails(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            LSTMTrainer,
            "cuda_probe_error",
            return_value="CUDA error: no kernel image is available for execution on the device",
        ):
            with self.assertRaisesRegex(ValueError, "cannot execute PyTorch kernels"):
                LSTMTrainer.resolve_training_device("cuda")

    def test_lstm_device_policy_rejects_unavailable_cuda_override(self):
        if torch is None:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is not available"):
                LSTMTrainer.resolve_training_device("cuda")

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
                "dense_hidden_size": 8,
                "dense_dropout": 0.1,
                "activation": "gelu",
                "learning_rate": 0.01,
                "batch_size": 4,
                "epochs": 3,
                "early_stopping_patience": 2,
                "decision_threshold": 0.42,
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
                metadata={"experiment_id": "lstm_smoke", "selected_threshold": 0.67},
            )

            self.assertEqual(artifact.metadata["runtime_loader"], "lstm")
            self.assertEqual(artifact.metadata["artifact_version"], 2)
            self.assertEqual(artifact.metadata["architecture_name"], "lstm_v2_dense_head")
            self.assertEqual(artifact.metadata["feature_mode"], "engineered")
            self.assertEqual(artifact.metadata["selected_threshold"], 0.67)
            self.assertEqual(artifact.metadata["decision_threshold"], 0.42)
            self.assertEqual(artifact.metadata["threshold_source"], "validation_selected")
            self.assertEqual(artifact.metadata["lookback_window"], 4)
            self.assertEqual(artifact.metadata["feature_count"], 2)
            self.assertEqual(artifact.metadata["sequence_feature_count"], 2)
            self.assertEqual(artifact.metadata["bidirectional"], False)
            self.assertEqual(artifact.metadata["training_device"], "cpu")
            self.assertFalse(artifact.metadata["cuda_usable"])
            self.assertIn("trainer_params", artifact.metadata)
            payload = joblib.load(artifact_path)
            self.assertTrue(all(str(tensor.device) == "cpu" for tensor in payload["state_dict"].values()))
            predictor = load_runtime_predictor(config, str(artifact_path))
            self.assertIsInstance(predictor, LSTMPredictor)
            self.assertEqual(predictor.min_history_rows, 4)
            self.assertEqual(predictor.decision_threshold, 0.67)
            self.assertEqual(predictor.threshold_source, "validation_selected")

            prediction_frame = predictor.predict_batch(features.iloc[:20])
            self.assertFalse(bool(prediction_frame.iloc[0]["is_valid"]))
            self.assertTrue(bool(prediction_frame.iloc[3]["is_valid"]))
            self.assertEqual(list(predictor.required_feature_columns), ["feat1", "feat2"])

    def test_lstm_runtime_predictor_rejects_legacy_artifact_without_v2_version(self):
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
                "learning_rate": 0.01,
                "batch_size": 4,
                "epochs": 2,
                "early_stopping_patience": 2,
                "decision_threshold": 0.73,
                "seed": 7,
            },
        )

        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "legacy_lstm.joblib"
            trainer.fit_candidate_artifact(
                features.iloc[:20],
                target.iloc[:20],
                str(artifact_path),
                metadata={"experiment_id": "legacy_lstm_smoke"},
            )
            payload = joblib.load(artifact_path)
            payload.pop("artifact_version", None)
            payload["metadata"].pop("artifact_version", None)
            joblib.dump(payload, artifact_path)

            with self.assertRaisesRegex(ValueError, "Retrain the LSTM candidate"):
                load_runtime_predictor(config, str(artifact_path))

    def test_lstm_v2_combined_feature_mode_builds_raw_sequence_channels(self):
        config = self._base_config()
        index = pd.date_range("2025-01-01", periods=24, freq="5min")
        close = pd.Series([100.0 + value * 0.1 for value in range(24)], index=index)
        features = pd.DataFrame(
            {
                "feat1": [float(value % 4) for value in range(24)],
                "Open": close - 0.02,
                "High": close + 0.05,
                "Low": close - 0.05,
                "Close": close,
                "Volume": [100.0 + value for value in range(24)],
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
                "feature_mode": "combined",
                "lookback_window": 4,
                "hidden_size": 8,
                "dense_hidden_size": 8,
                "learning_rate": 0.01,
                "batch_size": 4,
                "epochs": 2,
                "early_stopping_patience": 2,
                "seed": 7,
            },
        )

        with TemporaryDirectory() as temp_dir:
            artifact = trainer.fit_candidate_artifact(
                features.iloc[:20],
                target.iloc[:20],
                str(Path(temp_dir) / "combined_lstm.joblib"),
            )

            self.assertEqual(artifact.metadata["feature_mode"], "combined")
            self.assertEqual(artifact.metadata["raw_column_mapping"]["close"], "Close")
            self.assertIn("feat1", artifact.metadata["sequence_feature_columns"])
            self.assertIn("raw_return", artifact.metadata["sequence_feature_columns"])
            self.assertGreater(artifact.metadata["sequence_feature_count"], artifact.metadata["feature_count"])

    def test_lstm_v2_raw_market_mode_requires_raw_columns(self):
        trainer = LSTMTrainer(
            self._base_config(),
            target_column="Future_Direction_1",
            model_params={"feature_mode": "raw_market", "lookback_window": 4, "epochs": 1},
        )
        features = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0]})
        target = pd.Series([0.0, 1.0, 0.0, 1.0], name="Future_Direction_1")

        with self.assertRaisesRegex(ValueError, "requires raw market columns"):
            trainer.fit_predict(features, target, features)


if __name__ == "__main__":
    unittest.main()
