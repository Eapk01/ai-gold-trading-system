import sys
import types
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.modules.setdefault("talib", types.SimpleNamespace())
sys.modules.setdefault("pandas_ta", types.SimpleNamespace())

from src.app_service import ResearchAppService
from src.ai_models import AIModelManager
from src.config_utils import ensure_runtime_directories, get_default_config
from src.research import (
    CandidateArtifact,
    EvaluationPipeline,
    ExperimentRunner,
    ExperimentStore,
    FullSetFeatureSelector,
    FixedHorizonDirectionSpec,
    FoldMetrics,
    LegacyRuntimeDirectionSpec,
    NeutralBandLabelSpec,
    PromotionManifest,
    ResearchExperimentRequest,
    ReturnThresholdLabelSpec,
    SearchCandidateSummary,
    SearchRequest,
    SearchRunner,
    TrainerOutput,
    TrainerRegistry,
    TrainingPipeline,
    VarianceFeatureSelector,
    VolatilityAdjustedMoveSpec,
    WalkForwardSplitter,
    build_feature_inventory,
    build_named_feature_sets,
    build_binary_target_labels,
    build_feature_selector,
    build_experiment_integrity_proof,
    build_fixed_horizon_direction_labels,
    build_legacy_runtime_direction_labels,
    build_observed_persistence_labels,
    build_search_diagnostics,
    build_training_experiment_diagnostics,
    get_default_promoted_model_path,
    resolve_stage5_preset_definitions,
)
from src.research.feature_selection import CorrelationFeatureSelector
from src.research.trainers import CurrentEnsembleTrainer, ResearchTrainer


class ResearchStructureTests(unittest.TestCase):
    @staticmethod
    def _passing_integrity() -> dict:
        return {
            "proof_status": "passed",
            "integrity_contract_ok": True,
            "overview": {
                "invalid_fold_count": 0,
                "purge_required": True,
                "horizon_bars": 1,
                "total_purged_train_rows": 1,
                "total_purged_validation_rows": 1,
                "contract_failure_reasons": [],
            },
            "warnings": [],
            "fold_rows": [{"fold_name": "fold_1", "status": "passed"}],
        }

    def test_runtime_directories_include_research_paths(self):
        config = get_default_config()
        created = ensure_runtime_directories(config)

        self.assertIn("reports/experiments", created)
        self.assertIn("models/candidates", created)

    def test_walk_forward_splitter_is_deterministic_and_non_overlapping(self):
        splitter = WalkForwardSplitter(train_size=4, validation_size=2, test_size=2, step_size=2, expanding_window=True)

        first_run = splitter.split(12)
        second_run = splitter.split(12)

        self.assertEqual(first_run, second_run)
        self.assertEqual(len(first_run), 3)
        self.assertEqual(first_run[0].train_start, 0)
        self.assertEqual(first_run[0].train_end, 4)
        self.assertEqual(first_run[1].train_start, 0)
        self.assertEqual(first_run[1].train_end, 6)
        for split in first_run:
            self.assertLessEqual(split.train_end, split.validation_start)
            self.assertLessEqual(split.validation_end, split.test_start)
            self.assertLessEqual(split.train_start, split.train_end)
            self.assertLessEqual(split.validation_start, split.validation_end)
            self.assertLessEqual(split.test_start, split.test_end)

    def test_walk_forward_splitter_returns_empty_when_dataset_too_short(self):
        splitter = WalkForwardSplitter(train_size=10, validation_size=5, test_size=5)
        self.assertEqual(splitter.split(19), [])

    def test_fixed_horizon_direction_labels_are_deterministic(self):
        index = pd.date_range("2025-01-01", periods=5, freq="5min")
        frame = pd.DataFrame({"Close": [100.0, 101.0, 103.0, 102.0, 104.0]}, index=index)
        spec = FixedHorizonDirectionSpec(
            spec_id="direction_h1",
            display_name="Direction 1",
            horizon_bars=1,
        )

        first = build_fixed_horizon_direction_labels(frame, spec)
        second = build_fixed_horizon_direction_labels(frame, spec)

        self.assertTrue(first.equals(second))
        self.assertEqual(first.iloc[0], 1.0)
        self.assertEqual(first.iloc[2], 0.0)

    def test_legacy_runtime_direction_builder_matches_current_pipeline_formula(self):
        index = pd.date_range("2025-01-01", periods=5, freq="5min")
        frame = pd.DataFrame({"Close": [100.0, 101.0, 103.0, 102.0, 104.0]}, index=index)
        spec = LegacyRuntimeDirectionSpec()

        labels = build_legacy_runtime_direction_labels(frame, spec)
        future_return_pct = ((frame["Close"].shift(-1) / frame["Close"]) - 1.0) * 100.0
        expected = pd.Series(pd.NA, index=index, dtype="Float64")
        expected[future_return_pct > 0] = 1.0
        expected[future_return_pct <= 0] = 0.0
        expected[future_return_pct.isna()] = pd.NA

        self.assertTrue(labels.equals(expected))
        self.assertTrue(pd.isna(labels.iloc[-1]))

    def test_return_threshold_and_neutral_band_labels_produce_expected_neutral_rows(self):
        index = pd.date_range("2025-01-01", periods=5, freq="5min")
        frame = pd.DataFrame({"Close": [100.0, 100.02, 100.20, 100.18, 100.30]}, index=index)
        return_spec = ReturnThresholdLabelSpec(
            spec_id="threshold",
            display_name="Threshold",
            horizon_bars=1,
            return_threshold_pct=0.05,
        )
        neutral_spec = NeutralBandLabelSpec(
            spec_id="neutral",
            display_name="Neutral",
            horizon_bars=1,
            neutral_band_pct=0.05,
        )

        return_labels = build_binary_target_labels(frame, return_spec)
        neutral_labels = build_binary_target_labels(frame, neutral_spec)

        self.assertTrue(pd.isna(return_labels.iloc[0]))
        self.assertEqual(return_labels.iloc[1], 1.0)
        self.assertTrue(return_labels.equals(neutral_labels))

    def test_observed_persistence_labels_use_past_realized_moves_not_shifted_future_labels(self):
        index = pd.date_range("2025-01-01", periods=6, freq="5min")
        frame = pd.DataFrame({"Close": [100.0, 100.10, 100.20, 99.80, 99.70, 99.60]}, index=index)
        spec = ReturnThresholdLabelSpec(
            spec_id="threshold",
            display_name="Threshold",
            horizon_bars=2,
            return_threshold_pct=0.05,
        )

        future_labels = build_binary_target_labels(frame, spec)
        observed_persistence = build_observed_persistence_labels(frame, spec)

        self.assertTrue(pd.isna(observed_persistence.iloc[0]))
        self.assertTrue(pd.isna(observed_persistence.iloc[1]))
        self.assertEqual(observed_persistence.iloc[2], 1.0)
        self.assertEqual(observed_persistence.iloc[3], 0.0)
        self.assertFalse(observed_persistence.equals(future_labels.shift(1)))

    def test_volatility_adjusted_labels_require_price_column(self):
        frame = pd.DataFrame({"Other": [1.0, 2.0, 3.0]})
        spec = VolatilityAdjustedMoveSpec(
            spec_id="vol",
            display_name="Vol",
            horizon_bars=3,
            volatility_window=5,
            volatility_multiplier=1.0,
        )

        with self.assertRaises(ValueError):
            build_binary_target_labels(frame, spec)

    def test_feature_inventory_excludes_raw_and_future_columns_deterministically(self):
        inventory = build_feature_inventory(
            [
                "Close",
                "SMA_5",
                "Volume_Ratio",
                "Hour",
                "Close_Lag_1",
                "Returns_Mean_5",
                "Future_Direction_1",
            ]
        )

        self.assertEqual([row.column for row in inventory], sorted([row.column for row in inventory]))
        by_column = {row.column: row for row in inventory}
        self.assertFalse(by_column["Close"].eligible)
        self.assertEqual(by_column["Close"].exclusion_reason, "raw_market_column")
        self.assertFalse(by_column["Future_Direction_1"].eligible)
        self.assertEqual(by_column["Future_Direction_1"].exclusion_reason, "future_or_target_column")
        self.assertEqual(by_column["SMA_5"].group, "momentum_trend")
        self.assertEqual(by_column["Close_Lag_1"].group, "lag_features")

    def test_named_feature_sets_resolve_expected_columns(self):
        feature_sets = build_named_feature_sets(
            [
                "SMA_5",
                "EMA_10",
                "ATR_14",
                "Volume_Ratio",
                "Hour",
                "Pivot",
                "Close_Lag_1",
                "Returns_Mean_5",
                "Future_Direction_1",
                "Close",
            ]
        )

        self.assertIn("SMA_5", feature_sets["momentum"].columns)
        self.assertIn("ATR_14", feature_sets["volatility"].columns)
        self.assertIn("Hour", feature_sets["context"].columns)
        self.assertIn("Returns_Mean_5", feature_sets["lag_statistical"].columns)
        self.assertNotIn("Future_Direction_1", feature_sets["all_eligible"].columns)
        self.assertNotIn("Close", feature_sets["all_eligible"].columns)

    def test_feature_selectors_are_fold_local_and_return_ranking_metadata(self):
        frame_a = pd.DataFrame(
            {
                "feat1": [0, 1, 2, 3, 4, 5],
                "feat2": [0, 0, 0, 1, 1, 1],
                "feat3": [1, 1, 1, 1, 1, 1],
            }
        )
        frame_b = pd.DataFrame(
            {
                "feat1": [0, 0, 0, 1, 1, 1],
                "feat2": [0, 1, 2, 3, 4, 5],
                "feat3": [1, 1, 1, 1, 1, 1],
            }
        )
        target = pd.Series([0, 0, 0, 1, 1, 1])

        corr_selector = CorrelationFeatureSelector(max_features=1)
        corr_result_a = corr_selector.select(frame_a, target)
        corr_result_b = corr_selector.select(frame_b, target)

        self.assertEqual(corr_result_a.selector_name, "correlation")
        self.assertEqual(len(corr_result_a.ranking_rows), 1)
        self.assertNotEqual(corr_result_a.selected_columns, corr_result_b.selected_columns)

        variance_selector = VarianceFeatureSelector(max_features=2)
        variance_result = variance_selector.select(frame_a, target)
        self.assertEqual(variance_result.selector_name, "variance")
        self.assertGreaterEqual(len(variance_result.selected_columns), 1)

        full_selector = FullSetFeatureSelector()
        full_result = full_selector.select(frame_a, target)
        self.assertEqual(full_result.selector_name, "full_set")
        self.assertEqual(full_result.selected_columns, ["feat1", "feat2", "feat3"])
        self.assertEqual(build_feature_selector("full_set").select(frame_a, target).selector_name, "full_set")

    def test_evaluation_pipeline_generates_threshold_and_calibration_outputs(self):
        index = pd.date_range("2025-01-01", periods=6, freq="5min")
        pipeline = EvaluationPipeline(thresholds=[0.5, 0.7], calibration_bins=3)
        result = pipeline.evaluate_segment(
            timestamps=index,
            target=pd.Series([0, 1, 1, 0, 1, 0], index=index),
            prediction=pd.Series([0, 1, 1, 0, 0, 0], index=index),
            probability=pd.Series([0.20, 0.80, 0.72, 0.15, 0.45, 0.40], index=index),
            confidence=pd.Series([0.80, 0.80, 0.72, 0.85, 0.55, 0.60], index=index),
            fold_name="fold_01",
            split_segment="test",
            model_name="current_ensemble",
        )

        self.assertEqual(result.summary["rows"], 6)
        self.assertIn("roc_auc", result.summary)
        self.assertEqual(list(result.threshold_metrics["threshold"]), [0.5, 0.7])
        self.assertGreaterEqual(len(result.calibration_bins), 1)
        self.assertIn("probability", result.prediction_rows.columns)
        self.assertIn("confidence", result.prediction_rows.columns)

    def test_experiment_runner_produces_model_and_baseline_metrics(self):
        index = pd.date_range("2025-01-01", periods=12, freq="5min")
        frame = pd.DataFrame({"feat1": range(12)}, index=index)
        target = pd.Series([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], index=index, name="Future_Direction_1")
        splits = WalkForwardSplitter(train_size=4, validation_size=2, test_size=2, step_size=2).split(len(frame))
        request = ResearchExperimentRequest(
            experiment_name="smoke",
            target_column="Future_Direction_1",
            feature_columns=["feat1"],
            trainer_name="dummy",
            baseline_names=["majority_class", "persistence"],
            train_size=4,
            validation_size=2,
            test_size=2,
            step_size=2,
        )

        class DummyTrainer(ResearchTrainer):
            def fit_predict(self, train_features, train_target, test_features):
                return TrainerOutput(
                    prediction=pd.Series(1.0, index=test_features.index, dtype="float64"),
                    confidence=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    probabilities=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    selected_features=list(train_features.columns),
                    metadata={"model_family": "dummy"},
                )

            def fit_candidate_artifact(self, train_features, train_target, artifact_path, *, metadata=None):
                return CandidateArtifact(
                    artifact_path=str(artifact_path),
                    selected_features=list(train_features.columns),
                    trainer_name="dummy",
                    metadata=dict(metadata or {}),
                )

        runner = ExperimentRunner(
            training_pipeline=TrainingPipeline(DummyTrainer().fit_predict),
            evaluation_pipeline=EvaluationPipeline(),
        )
        result = runner.run(
            request=request,
            feature_frame=frame,
            target=target,
            splits=splits,
        )

        self.assertGreaterEqual(len(result.folds), len(splits) * 6)
        self.assertIn("mean_test_accuracy", result.aggregate_metrics)
        self.assertIn("baselines", result.baseline_comparison)
        self.assertTrue(result.prediction_artifacts)
        self.assertIn("prediction_rows", result.prediction_artifacts[0].metadata)

    def test_experiment_runner_purges_train_and_validation_tails_for_future_labels(self):
        index = pd.date_range("2025-01-01", periods=12, freq="5min")
        frame = pd.DataFrame(
            {
                "Close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8, 102.0, 102.2],
                "feat1": range(12),
            },
            index=index,
        )
        target_spec = ReturnThresholdLabelSpec(
            spec_id="threshold_h2",
            display_name="Threshold H2",
            horizon_bars=2,
            return_threshold_pct=0.0,
        )
        target = build_binary_target_labels(frame, target_spec)
        splits = WalkForwardSplitter(train_size=5, validation_size=4, test_size=3, step_size=3).split(len(frame))
        request = ResearchExperimentRequest(
            experiment_name="purge_smoke",
            target_column="threshold_h2",
            feature_columns=["feat1"],
            trainer_name="dummy",
            baseline_names=[],
            train_size=5,
            validation_size=4,
            test_size=3,
            step_size=3,
        )

        class RecordingSelector:
            def __init__(self):
                self.train_indices = []
                self.train_target_indices = []

            def select(self, feature_frame, target_series):
                self.train_indices.append(list(feature_frame.index))
                self.train_target_indices.append(list(target_series.index))
                return FullSetFeatureSelector().select(feature_frame, target_series)

        class DummyTrainer(ResearchTrainer):
            def fit_predict(self, train_features, train_target, test_features):
                return TrainerOutput(
                    prediction=pd.Series(1.0, index=test_features.index, dtype="float64"),
                    confidence=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    probabilities=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    selected_features=list(train_features.columns),
                )

            def fit_candidate_artifact(self, train_features, train_target, artifact_path, *, metadata=None):
                return CandidateArtifact(
                    artifact_path=str(artifact_path),
                    selected_features=list(train_features.columns),
                    trainer_name="dummy",
                    metadata=dict(metadata or {}),
                )

        selector = RecordingSelector()
        runner = ExperimentRunner(
            training_pipeline=TrainingPipeline(DummyTrainer().fit_predict),
            evaluation_pipeline=EvaluationPipeline(),
        )

        result = runner.run(
            request=request,
            feature_frame=frame,
            target=target,
            splits=splits,
            feature_selector=selector,
            target_spec=target_spec,
        )

        self.assertEqual(selector.train_indices[0], list(index[:3]))
        self.assertEqual(selector.train_target_indices[0], list(index[:3]))

        model_folds = [fold for fold in result.folds if fold.model_name == "dummy"]
        self.assertEqual(model_folds[0].train_rows, 3)
        self.assertEqual(model_folds[0].validation_rows, 2)
        self.assertEqual(model_folds[0].test_rows, 3)

    def test_experiment_integrity_builder_records_purged_fold_boundaries(self):
        index = pd.date_range("2025-01-01", periods=12, freq="5min")
        frame = pd.DataFrame(
            {
                "Close": [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8, 102.0, 102.2],
                "feat1": range(12),
            },
            index=index,
        )
        target_spec = ReturnThresholdLabelSpec(
            spec_id="threshold_h2",
            display_name="Threshold H2",
            horizon_bars=2,
            return_threshold_pct=0.0,
        )
        target = build_binary_target_labels(frame, target_spec)
        splits = WalkForwardSplitter(train_size=5, validation_size=4, test_size=3, step_size=3).split(len(frame))
        request = ResearchExperimentRequest(
            experiment_name="integrity_smoke",
            target_column="threshold_h2",
            feature_columns=["feat1"],
            trainer_name="dummy",
            baseline_names=[],
            train_size=5,
            validation_size=4,
            test_size=3,
            step_size=3,
        )

        class DummyTrainer(ResearchTrainer):
            def fit_predict(self, train_features, train_target, test_features):
                return TrainerOutput(
                    prediction=pd.Series(1.0, index=test_features.index, dtype="float64"),
                    confidence=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    probabilities=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    selected_features=list(train_features.columns),
                )

            def fit_candidate_artifact(self, train_features, train_target, artifact_path, *, metadata=None):
                return CandidateArtifact(
                    artifact_path=str(artifact_path),
                    selected_features=list(train_features.columns),
                    trainer_name="dummy",
                    metadata=dict(metadata or {}),
                )

        runner = ExperimentRunner(
            training_pipeline=TrainingPipeline(DummyTrainer().fit_predict),
            evaluation_pipeline=EvaluationPipeline(),
        )
        result = runner.run(
            request=request,
            feature_frame=frame,
            target=target,
            splits=splits,
            feature_selector=FullSetFeatureSelector(),
            target_spec=target_spec,
        )

        integrity = build_experiment_integrity_proof(
            experiment_result=result,
            expected_feature_selection_mode="fold_local_selector",
            expected_target_spec_id="threshold_h2",
        )

        self.assertTrue(integrity["integrity_contract_ok"])
        self.assertEqual(integrity["proof_status"], "passed")
        self.assertEqual(integrity["overview"]["horizon_bars"], 2)
        self.assertEqual(integrity["overview"]["total_purged_train_rows"], 2)
        self.assertEqual(integrity["overview"]["total_purged_validation_rows"], 2)
        self.assertEqual(len(integrity["fold_rows"]), 1)
        self.assertEqual(integrity["fold_rows"][0]["effective_train_end"], 3)
        self.assertEqual(integrity["fold_rows"][0]["effective_validation_end"], 7)

    def test_experiment_store_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            store = ExperimentStore(temp_dir)
            path = store.save_result(
                result=type(
                    "Result",
                    (),
                    {"to_dict": lambda self=None: {"experiment_name": "demo", "aggregate_metrics": {"mean_test_accuracy": 0.6}}},
                )(),
                filename="demo.json",
            )
            loaded = store.load_result(path.name)

            self.assertEqual(loaded["experiment_name"], "demo")
            self.assertEqual([item.name for item in store.list_results()], ["demo.json"])

    def test_trainer_registry_and_pipeline_accept_standardized_output(self):
        class DummyTrainer(ResearchTrainer):
            def fit_predict(self, train_features, train_target, test_features):
                return TrainerOutput(
                    prediction=pd.Series(1.0, index=test_features.index, dtype="float64"),
                    confidence=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    probabilities=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    selected_features=list(train_features.columns),
                    metadata={"model_family": "dummy"},
                )

            def fit_candidate_artifact(self, train_features, train_target, artifact_path, *, metadata=None):
                return CandidateArtifact(
                    artifact_path=str(artifact_path),
                    selected_features=list(train_features.columns),
                    trainer_name="dummy",
                    metadata=dict(metadata or {}),
                )

        trainer = DummyTrainer()
        registry = TrainerRegistry()
        registry.register("dummy", trainer)

        frame = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]}, index=pd.RangeIndex(3))
        pipeline = TrainingPipeline(registry.get("dummy").fit_predict)
        result = pipeline.run(frame.iloc[:2], pd.Series([1, 0]), frame.iloc[2:])

        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertEqual(result["selected_features"], ["feat1"])

    def test_current_ensemble_trainer_returns_probabilities_and_confidence(self):
        config = get_default_config()
        config["ai_model"]["models"] = ["random_forest"]
        trainer = CurrentEnsembleTrainer(config, target_column="Future_Direction_1")
        index = pd.date_range("2025-01-01", periods=12, freq="5min")
        train_features = pd.DataFrame(
            {
                "feat1": [0, 1, 2, 3, 4, 5, 6, 7],
                "feat2": [1, 1, 0, 0, 1, 1, 0, 0],
            },
            index=index[:8],
        )
        train_target = pd.Series([0, 0, 0, 1, 1, 1, 0, 1], index=index[:8])
        test_features = pd.DataFrame(
            {
                "feat1": [8, 9, 10, 11],
                "feat2": [1, 0, 1, 0],
            },
            index=index[8:],
        )

        result = trainer.fit_predict(train_features, train_target, test_features)

        self.assertEqual(list(result.prediction.index), list(test_features.index))
        self.assertEqual(list(result.probabilities.index), list(test_features.index))
        self.assertEqual(list(result.confidence.index), list(test_features.index))
        self.assertTrue(((result.probabilities >= 0.0) & (result.probabilities <= 1.0)).all())
        self.assertIn("trained_model_names", result.metadata)

    def test_current_ensemble_trainer_handles_nan_features_with_logistic_regression(self):
        config = get_default_config()
        config["ai_model"]["models"] = ["logistic_regression"]
        trainer = CurrentEnsembleTrainer(config, target_column="Future_Direction_1")
        index = pd.date_range("2025-01-01", periods=10, freq="5min")
        train_features = pd.DataFrame(
            {
                "feat1": [0.0, 1.0, None, 3.0, 4.0, 5.0],
                "feat2": [1.0, None, 0.0, 0.0, 1.0, 1.0],
            },
            index=index[:6],
        )
        train_target = pd.Series([0, 0, 1, 1, 0, 1], index=index[:6])
        test_features = pd.DataFrame(
            {
                "feat1": [6.0, None, 8.0, 9.0],
                "feat2": [0.0, 1.0, None, 1.0],
            },
            index=index[6:],
        )

        result = trainer.fit_predict(train_features, train_target, test_features)

        self.assertEqual(len(result.prediction), 4)
        self.assertFalse(result.prediction.isna().any())
        self.assertFalse(result.probabilities.isna().any())
        self.assertFalse(result.confidence.isna().any())
        self.assertEqual(result.metadata.get("preprocessing"), "median_imputation_then_standard_scaling")

    def test_current_ensemble_trainer_can_save_candidate_artifact_for_existing_model_loader(self):
        config = get_default_config()
        config["ai_model"]["models"] = ["random_forest"]
        trainer = CurrentEnsembleTrainer(config, target_column="Future_Direction_1")
        index = pd.date_range("2025-01-01", periods=10, freq="5min")
        train_features = pd.DataFrame(
            {
                "feat1": [0.0, 1.0, None, 3.0, 4.0, 5.0],
                "feat2": [1.0, None, 0.0, 0.0, 1.0, 1.0],
            },
            index=index[:6],
        )
        train_target = pd.Series([0, 0, 1, 1, 0, 1], index=index[:6])

        with TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "candidate.joblib"
            artifact = trainer.fit_candidate_artifact(train_features, train_target, str(artifact_path))

            self.assertIsInstance(artifact, CandidateArtifact)
            self.assertTrue(artifact_path.exists())

            manager = AIModelManager(config)
            self.assertTrue(manager.load_models(str(artifact_path)))
            prediction_frame = manager.predict_ensemble_batch(
                pd.DataFrame(
                    {
                        "feat1": [6.0, None, 8.0],
                        "feat2": [0.0, 1.0, None],
                    },
                    index=index[6:9],
                ),
                feature_columns=["feat1", "feat2"],
            )

            self.assertTrue(prediction_frame["is_valid"].all())
            self.assertFalse(prediction_frame["prediction"].isna().any())

    def test_current_ensemble_trainer_caps_internal_threads_for_parallel_research(self):
        config = get_default_config()
        config["ai_model"]["models"] = ["random_forest", "xgboost"]
        trainer = CurrentEnsembleTrainer(
            config,
            target_column="Future_Direction_1",
            model_names=["random_forest", "xgboost"],
            model_params={
                "random_forest": {"n_estimators": 10},
                "xgboost": {"n_estimators": 10},
            },
        )

        safe_params = trainer._research_safe_model_params()

        self.assertEqual(safe_params["random_forest"]["n_jobs"], 1)
        self.assertEqual(safe_params["xgboost"]["n_jobs"], 1)
        self.assertEqual(safe_params["xgboost"]["nthread"], 1)

    def test_promotion_manifest_and_default_promoted_path(self):
        manifest = PromotionManifest(
            experiment_name="demo run",
            source_model_path="models/candidates/demo.joblib",
            promoted_model_path="models/demo_run.joblib",
            target_column="Future_Direction_1",
            selected_threshold=0.67,
        )

        self.assertEqual(manifest.to_dict()["experiment_name"], "demo run")
        self.assertEqual(
            str(get_default_promoted_model_path("models", "demo run")),
            str(Path("models") / "demo_run.joblib"),
        )

    def test_stage5_preset_resolution_builds_expected_bounded_grid(self):
        request = SearchRequest(
            search_id="search_run_demo",
            search_name="stage5_demo",
            target_spec={"spec_id": "return_threshold_h3_0_05pct"},
            target_specs=[
                {"spec_id": "return_threshold_h3_0_05pct", "display_name": "Return Threshold"},
                {"spec_id": "legacy_future_direction_1", "display_name": "Legacy Runtime"},
                {"spec_id": "vol_adjusted_h3_x1_0", "display_name": "Volatility Adjusted"},
            ],
            feature_set_names=["volatility", "baseline_core"],
            trainer_name="current_ensemble",
            preset_names=["conservative", "balanced", "capacity"],
        )
        presets = resolve_stage5_preset_definitions(
            ["random_forest", "logistic_regression"],
            request.preset_names,
        )
        runner = SearchRunner()

        grid = runner.build_candidate_grid(request=request, preset_definitions=presets)

        self.assertEqual(len(grid), 18)
        self.assertEqual(
            {candidate.target_spec_id for candidate in grid},
            {"return_threshold_h3_0_05pct", "legacy_future_direction_1", "vol_adjusted_h3_x1_0"},
        )
        self.assertEqual({candidate.feature_set_name for candidate in grid}, {"volatility", "baseline_core"})
        self.assertEqual({candidate.preset_name for candidate in grid}, {"conservative", "balanced", "capacity"})
        self.assertTrue(all("random_forest" in candidate.trainer_params for candidate in grid))
        self.assertTrue(all("logistic_regression" in candidate.trainer_params for candidate in grid))

    def test_stage5_preset_resolution_fails_for_unsupported_model_names(self):
        with self.assertRaises(ValueError):
            resolve_stage5_preset_definitions(["lightgbm"], ["balanced"])

    def test_stage5_ranking_uses_validation_only_and_applies_test_guardrail(self):
        runner = SearchRunner()
        weaker_test_better_validation = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="volatility",
            preset_name="balanced",
            selected_threshold=0.6,
            report_file="a.json",
            validation_summary={
                "beat_rate": 1.0,
                "f1_std": 0.01,
                "mean_f1": 0.65,
                "mean_coverage": 0.40,
                "fold_count": 4.0,
            },
            test_summary={
                "mean_f1": 0.49,
                "mean_coverage": 0.30,
                "best_baseline_mean_f1": 0.50,
                "fold_count": 4.0,
            },
            passed_test_guardrail=False,
            overall_mean_test_accuracy=0.60,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )
        stronger_test_lower_validation = SearchCandidateSummary(
            candidate_id="candidate_b",
            experiment_id="exp_b",
            experiment_name="exp_b",
            trainer_name="current_ensemble",
            feature_set_name="baseline_core",
            preset_name="capacity",
            selected_threshold=0.55,
            report_file="b.json",
            validation_summary={
                "beat_rate": 0.75,
                "f1_std": 0.02,
                "mean_f1": 0.60,
                "mean_coverage": 0.50,
                "fold_count": 4.0,
            },
            test_summary={
                "mean_f1": 0.58,
                "mean_coverage": 0.30,
                "best_baseline_mean_f1": 0.52,
                "fold_count": 4.0,
            },
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.60,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([stronger_test_lower_validation, weaker_test_better_validation])

        self.assertEqual(ranked["leaderboard_rows"][0]["candidate_id"], "candidate_a")
        self.assertEqual(ranked["recommended_winner"]["experiment_id"], "exp_b")

    def test_stage51_truth_gate_blocks_low_coverage_candidate(self):
        runner = SearchRunner()
        low_coverage = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="volatility",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="a.json",
            validation_summary={"beat_rate": 0.8, "f1_std": 0.02, "mean_f1": 0.70, "mean_coverage": 0.35, "fold_count": 4.0},
            test_summary={"mean_f1": 0.68, "mean_coverage": 0.15, "best_baseline_mean_f1": 0.60, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.58,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )
        healthy = SearchCandidateSummary(
            candidate_id="candidate_b",
            experiment_id="exp_b",
            experiment_name="exp_b",
            trainer_name="current_ensemble",
            feature_set_name="baseline_core",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="b.json",
            validation_summary={"beat_rate": 0.7, "f1_std": 0.03, "mean_f1": 0.66, "mean_coverage": 0.40, "fold_count": 4.0},
            test_summary={"mean_f1": 0.61, "mean_coverage": 0.25, "best_baseline_mean_f1": 0.59, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.57,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([low_coverage, healthy])

        self.assertEqual(ranked["leaderboard_rows"][0]["candidate_id"], "candidate_a")
        self.assertIn("low_test_coverage", ranked["leaderboard_rows"][0]["truth_gate_failures"])
        self.assertEqual(ranked["recommended_winner"]["experiment_id"], "exp_b")

    def test_stage53_failed_candidates_are_excluded_from_ranking_but_kept_in_leaderboard(self):
        runner = SearchRunner()
        failed = SearchCandidateSummary(
            candidate_id="candidate_failed",
            experiment_id="exp_failed",
            experiment_name="exp_failed",
            trainer_name="current_ensemble",
            target_spec_id="return_threshold_h3_0_05pct",
            target_display_name="Return Threshold",
            feature_set_name="volatility",
            preset_name="balanced",
            execution_status="failed",
            error_message="boom",
        )
        healthy = SearchCandidateSummary(
            candidate_id="candidate_ok",
            experiment_id="exp_ok",
            experiment_name="exp_ok",
            trainer_name="current_ensemble",
            target_spec_id="return_threshold_h3_0_05pct",
            target_display_name="Return Threshold",
            feature_set_name="baseline_core",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="ok.json",
            validation_summary={"beat_rate": 0.7, "f1_std": 0.03, "mean_f1": 0.66, "mean_coverage": 0.40, "fold_count": 4.0},
            test_summary={"mean_f1": 0.61, "mean_coverage": 0.25, "best_baseline_mean_f1": 0.59, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.57,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([failed, healthy])

        self.assertEqual(ranked["recommended_winner"]["experiment_id"], "exp_ok")
        self.assertEqual(ranked["leaderboard_rows"][0]["candidate_id"], "candidate_ok")
        self.assertEqual(ranked["leaderboard_rows"][1]["candidate_id"], "candidate_failed")
        self.assertEqual(ranked["leaderboard_rows"][1]["execution_status"], "failed")

    def test_stage51_truth_gate_blocks_candidate_below_majority_baseline(self):
        runner = SearchRunner()
        under_majority = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="volatility",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="a.json",
            validation_summary={"beat_rate": 0.9, "f1_std": 0.01, "mean_f1": 0.72, "mean_coverage": 0.35, "fold_count": 4.0},
            test_summary={"mean_f1": 0.67, "mean_coverage": 0.30, "best_baseline_mean_f1": 0.60, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.49,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([under_majority])

        self.assertEqual(ranked["recommended_winner"]["status"], "no_winner")
        self.assertIn("broad_metric_under_majority", ranked["leaderboard_rows"][0]["truth_gate_failures"])

    def test_stage51_truth_gate_blocks_large_validation_test_drift(self):
        runner = SearchRunner()
        drifting = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="volatility",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="a.json",
            validation_summary={"beat_rate": 0.8, "f1_std": 0.01, "mean_f1": 0.80, "mean_coverage": 0.40, "fold_count": 4.0},
            test_summary={"mean_f1": 0.65, "mean_coverage": 0.30, "best_baseline_mean_f1": 0.60, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.60,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([drifting])

        self.assertEqual(ranked["recommended_winner"]["status"], "no_winner")
        self.assertIn("validation_test_drift", ranked["leaderboard_rows"][0]["truth_gate_failures"])

    def test_stage51_truth_gate_blocks_missing_selected_threshold_metrics(self):
        runner = SearchRunner()
        undefined = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="volatility",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="a.json",
            validation_summary={"beat_rate": 0.8, "f1_std": 0.01, "mean_coverage": 0.40, "fold_count": 4.0},
            test_summary={"mean_coverage": 0.30, "best_baseline_mean_f1": 0.60, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.60,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={"integrity": self._passing_integrity()},
        )

        ranked = runner.rank_candidates([undefined])

        self.assertEqual(ranked["recommended_winner"]["status"], "no_winner")
        self.assertIn("undefined_selected_threshold_metrics", ranked["leaderboard_rows"][0]["truth_gate_failures"])

    def test_stage51_truth_gate_blocks_candidate_with_failed_integrity_proof(self):
        runner = SearchRunner()
        integrity_failed = SearchCandidateSummary(
            candidate_id="candidate_a",
            experiment_id="exp_a",
            experiment_name="exp_a",
            trainer_name="current_ensemble",
            feature_set_name="baseline_core",
            preset_name="balanced",
            selected_threshold=0.55,
            report_file="a.json",
            validation_summary={"beat_rate": 0.8, "f1_std": 0.01, "mean_f1": 0.70, "mean_coverage": 0.40, "fold_count": 4.0},
            test_summary={"mean_f1": 0.66, "mean_coverage": 0.28, "best_baseline_mean_f1": 0.60, "fold_count": 4.0},
            passed_test_guardrail=True,
            overall_mean_test_accuracy=0.60,
            majority_baseline_mean_test_accuracy=0.55,
            expected_fold_count=4,
            diagnostics={
                "integrity": {
                    "proof_status": "failed",
                    "integrity_contract_ok": False,
                    "overview": {
                        "invalid_fold_count": 1,
                        "purge_required": True,
                        "total_purged_train_rows": 0,
                        "total_purged_validation_rows": 0,
                        "contract_failure_reasons": ["missing_required_train_purge"],
                    },
                    "warnings": [],
                    "fold_rows": [],
                }
            },
        )

        ranked = runner.rank_candidates([integrity_failed])

        self.assertEqual(ranked["recommended_winner"]["status"], "no_winner")
        self.assertIn("missing_required_train_purge", ranked["leaderboard_rows"][0]["truth_gate_failures"])
        self.assertFalse(ranked["leaderboard_rows"][0]["passed_truth_gate"])

    def test_training_diagnostics_flag_one_class_and_constant_features(self):
        index = pd.date_range("2025-01-01", periods=12, freq="5min")
        frame = pd.DataFrame(
            {
                "feat1": range(12),
                "feat_constant": [1.0] * 12,
            },
            index=index,
        )
        target = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], index=index, name="target")
        splits = WalkForwardSplitter(train_size=4, validation_size=2, test_size=2, step_size=2).split(len(frame))
        request = ResearchExperimentRequest(
            experiment_name="diag_smoke",
            target_column="target",
            feature_columns=["feat1", "feat_constant"],
            trainer_name="dummy",
            baseline_names=["majority_class", "persistence"],
            train_size=4,
            validation_size=2,
            test_size=2,
            step_size=2,
        )

        class DummyTrainer(ResearchTrainer):
            def fit_predict(self, train_features, train_target, test_features):
                return TrainerOutput(
                    prediction=pd.Series(1.0, index=test_features.index, dtype="float64"),
                    confidence=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    probabilities=pd.Series(0.8, index=test_features.index, dtype="float64"),
                    selected_features=list(train_features.columns),
                )

            def fit_candidate_artifact(self, train_features, train_target, artifact_path, *, metadata=None):
                return CandidateArtifact(
                    artifact_path=str(artifact_path),
                    selected_features=list(train_features.columns),
                    trainer_name="dummy",
                    metadata=dict(metadata or {}),
                )

        runner = ExperimentRunner(
            training_pipeline=TrainingPipeline(DummyTrainer().fit_predict),
            evaluation_pipeline=EvaluationPipeline(),
        )
        result = runner.run(
            request=request,
            feature_frame=frame,
            target=target,
            splits=splits,
            feature_selector=FullSetFeatureSelector(),
        )

        diagnostics = build_training_experiment_diagnostics(
            feature_frame=frame,
            target_series=target,
            splits=splits,
            experiment_result=result,
            trainer_name="dummy",
            selected_threshold=0.5,
            selected_threshold_summary={
                "validation": {"mean_f1": 0.6, "mean_coverage": 0.4},
                "test": {"mean_f1": 0.55, "mean_coverage": 0.35},
            },
        )

        self.assertGreater(diagnostics["highlights"]["one_class_fold_count"], 0)
        self.assertGreater(diagnostics["highlights"]["constant_feature_fold_count"], 0)
        self.assertTrue(diagnostics["feature_health_rows"])
        self.assertTrue(diagnostics["prediction_health_rows"])
        self.assertEqual(diagnostics["overview"]["selected_threshold_test_mean_f1"], 0.55)
        self.assertEqual(diagnostics["overview"]["selected_threshold_test_mean_coverage"], 0.35)

    def test_search_diagnostics_summarize_widespread_failures(self):
        candidates = [
            SearchCandidateSummary(
                candidate_id="a",
                experiment_id="a",
                experiment_name="a",
                trainer_name="current_ensemble",
                feature_set_name="volatility",
                preset_name="balanced",
                overall_mean_test_accuracy=0.48,
                majority_baseline_mean_test_accuracy=0.55,
                runtime_feature_contract_ok=False,
                diagnostics={
                    "integrity": self._passing_integrity(),
                    "highlights": {
                        "low_selected_threshold_test_coverage": True,
                        "broad_metric_under_majority": True,
                        "one_class_fold_count": 2,
                        "undefined_selected_threshold_metric_count": 1,
                        "constant_feature_fold_count": 1,
                        "near_constant_feature_fold_count": 0,
                    }
                },
            ),
            SearchCandidateSummary(
                candidate_id="b",
                experiment_id="b",
                experiment_name="b",
                trainer_name="current_ensemble",
                feature_set_name="baseline_core",
                preset_name="conservative",
                overall_mean_test_accuracy=0.49,
                majority_baseline_mean_test_accuracy=0.55,
                runtime_feature_contract_ok=True,
                diagnostics={
                    "integrity": self._passing_integrity(),
                    "highlights": {
                        "low_selected_threshold_test_coverage": True,
                        "broad_metric_under_majority": True,
                        "one_class_fold_count": 0,
                        "undefined_selected_threshold_metric_count": 0,
                        "constant_feature_fold_count": 0,
                        "near_constant_feature_fold_count": 1,
                    }
                },
            ),
        ]

        diagnostics = build_search_diagnostics(candidates)

        self.assertEqual(diagnostics["summary"]["candidate_count"], 2)
        self.assertEqual(diagnostics["summary"]["low_coverage_candidate_count"], 2)
        self.assertEqual(diagnostics["summary"]["majority_dominance_candidate_count"], 2)
        self.assertEqual(diagnostics["summary"]["runtime_feature_mismatch_candidate_count"], 1)
        self.assertTrue(diagnostics["warnings"])

    def test_search_diagnostics_count_integrity_failures(self):
        diagnostics = build_search_diagnostics(
            [
                SearchCandidateSummary(
                    candidate_id="failed_integrity",
                    experiment_id="failed_integrity",
                    experiment_name="failed_integrity",
                    trainer_name="current_ensemble",
                    feature_set_name="baseline_core",
                    preset_name="balanced",
                    diagnostics={
                        "integrity": {
                            "proof_status": "failed",
                            "integrity_contract_ok": False,
                            "overview": {
                                "contract_failure_reasons": ["missing_required_validation_purge"],
                            },
                        }
                    },
                )
            ]
        )

        self.assertEqual(diagnostics["summary"]["integrity_failure_candidate_count"], 1)
        self.assertTrue(any(warning.get("code") == "integrity_failures_present" for warning in diagnostics["warnings"]))

    def test_search_diagnostics_count_failed_candidates(self):
        diagnostics = build_search_diagnostics(
            [
                SearchCandidateSummary(
                    candidate_id="failed",
                    experiment_id="failed",
                    experiment_name="failed",
                    trainer_name="current_ensemble",
                    feature_set_name="volatility",
                    preset_name="balanced",
                    execution_status="failed",
                    error_message="boom",
                )
            ]
        )

        self.assertEqual(diagnostics["summary"]["failed_candidate_count"], 1)
        self.assertEqual(diagnostics["summary"]["successful_candidate_count"], 0)
        self.assertTrue(any(warning.get("code") == "candidate_execution_failures" for warning in diagnostics["warnings"]))


class ResearchExperimentServiceTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "trading:",
                    "  symbol: XAUUSDm",
                    "  timeframe: 5m",
                    "  position_size: 0.01",
                    "  stop_loss_pips: 50",
                    "  take_profit_pips: 100",
                    "data_sources:",
                    f"  dataset_directory: {str(root / 'imports').replace(chr(92), '/')}",
                    "  primary: local_csv",
                    "  min_rows: 10",
                    "ai_model:",
                    "  type: ensemble",
                    "  models: [random_forest]",
                    f"  models_directory: {str(root / 'models').replace(chr(92), '/')}",
                    "  lookback_periods: 20",
                    "  target_column: Future_Direction_1",
                    "research:",
                    f"  experiments_directory: {str(root / 'reports' / 'experiments').replace(chr(92), '/')}",
                    f"  candidate_models_directory: {str(root / 'models' / 'candidates').replace(chr(92), '/')}",
                    "risk_management:",
                    "  max_daily_loss: 30",
                    "  max_positions: 3",
                    "  risk_per_trade: 0.02",
                    "  drawdown_limit: 0.15",
                    "backtest:",
                    "  initial_capital: 10000",
                    "  commission: 0.0001",
                    "  slippage: 0.0002",
                    "database:",
                    f"  path: {str(root / 'data' / 'test.db').replace(chr(92), '/')}",
                    "logging:",
                    "  level: INFO",
                    f"  file_path: {str(root / 'logs' / 'test.log').replace(chr(92), '/')}",
                    "app:",
                    "  startup:",
                    "    autoload_latest_model: false",
                    "    autoconnect_broker: false",
                    "brokers:",
                    "  profiles: {}",
                    "  default_profile: ''",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def _build_stage12_feature_data(self, periods: int) -> pd.DataFrame:
        index = pd.date_range("2025-01-01", periods=periods, freq="5min")
        close = pd.Series(
            [100.0 + (value * 0.04) + ((-1) ** value) * 0.02 for value in range(periods)],
            index=index,
        )
        frame = pd.DataFrame(
            {
                "Open": close - 0.03,
                "High": close + 0.08,
                "Low": close - 0.08,
                "Close": close,
                "Volume": [100 + (value % 5) * 10 for value in range(periods)],
                "SMA_5": close.rolling(5, min_periods=1).mean(),
                "EMA_10": close.ewm(span=10, adjust=False).mean(),
                "MACD": close.diff().fillna(0.0),
                "ATR_14": close.diff().abs().rolling(14, min_periods=1).mean().fillna(0.0),
                "Volume_Ratio": pd.Series([1.0 + (value % 4) * 0.1 for value in range(periods)], index=index),
                "Pivot": close,
                "Hour": index.hour,
                "Future_Direction_1": [1.0 if (value % 4) >= 2 else 0.0 for value in range(periods)],
            },
            index=index,
        )
        frame["Future_Leak"] = list(range(periods))
        return frame

    def test_run_automated_search_saves_reports_and_recommended_candidate_can_be_promoted(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ResearchAppService(str(self._write_config(root)))
            index = pd.date_range("2025-01-01", periods=80, freq="5min")
            regime_signal = pd.Series([1.0 if ((value // 4) % 2 == 0) else 0.0 for value in range(80)], index=index)
            close = pd.Series(
                [
                    100.0 + sum(0.12 if regime_signal.iloc[offset] == 1.0 else -0.12 for offset in range(value + 1))
                    for value in range(80)
                ],
                index=index,
            )
            service.feature_data = pd.DataFrame(
                {
                    "Close": close,
                    "SMA_5": close.rolling(5, min_periods=1).mean(),
                    "EMA_10": close.ewm(span=10, adjust=False).mean(),
                    "MACD": close.diff().fillna(0.0),
                    "ATR_14": regime_signal.shift(-3).ffill().fillna(0.0),
                    "BB_Width": (1.0 - regime_signal.shift(-3).ffill()).fillna(0.0),
                    "Volume_Ratio": pd.Series([1.0 + (value % 4) * 0.1 for value in range(80)], index=index),
                    "Pivot": close,
                    "Hour": index.hour,
                    "Close_Lag_1": close.shift(1).bfill(),
                    "Returns_Mean_5": close.pct_change(fill_method=None).rolling(5, min_periods=1).mean().fillna(0.0),
                },
                index=index,
            )

            progress_snapshots = []

            result = service.run_automated_search(
                "stage5_smoke",
                progress_callback=lambda payload: progress_snapshots.append(dict(payload or {})),
                max_workers=2,
            )

            self.assertTrue(result["success"], msg=result)
            artifacts = result["artifacts"] or {}
            self.assertTrue(Path(artifacts["report_file"]).exists())
            self.assertTrue(Path(artifacts["leaderboard_file"]).exists())
            self.assertTrue(Path(artifacts["candidates_file"]).exists())

            listed = service.list_search_reports(limit=5)
            self.assertTrue(listed["success"])
            self.assertEqual(len(listed["data"]), 1)

            loaded = service.get_search_report(listed["data"][0]["path"])
            self.assertTrue(loaded["success"], msg=loaded)
            self.assertEqual(loaded["data"]["summary"]["search_name"], "stage5_smoke")
            self.assertEqual(loaded["data"]["summary"]["target_count"], 3)
            self.assertEqual(loaded["data"]["summary"]["candidate_count"], 12)
            self.assertEqual(loaded["data"]["summary"]["execution_mode"], "parallel_candidate_threads")
            self.assertEqual(loaded["data"]["summary"]["resolved_max_workers"], 2)
            self.assertEqual(loaded["data"]["summary"]["failed_candidate_count"], 0)
            self.assertEqual(loaded["data"]["summary"]["successful_candidate_count"], 12)
            self.assertEqual(len(loaded["data"]["leaderboard_rows"]), 12)
            self.assertIn("winner_reason", loaded["data"]["summary"])
            self.assertIn("gate_summary", loaded["data"])
            self.assertIn("diagnostics", loaded["data"])
            self.assertTrue(loaded["data"]["diagnostics"])
            self.assertIn("integrity", loaded["data"])
            self.assertEqual(loaded["data"]["integrity"]["proof_status"], "passed")
            self.assertEqual(len(loaded["data"]["target_specs"]), 3)
            self.assertTrue(all("passed_truth_gate" in row for row in loaded["data"]["leaderboard_rows"]))
            self.assertTrue(all("truth_gate_failures" in row for row in loaded["data"]["leaderboard_rows"]))
            self.assertTrue(all("diagnostics" in row for row in loaded["data"]["leaderboard_rows"]))
            self.assertTrue(all("proof_status" in row for row in loaded["data"]["leaderboard_rows"]))
            self.assertTrue(all("execution_status" in row for row in loaded["data"]["leaderboard_rows"]))
            self.assertTrue(Path(loaded["data"]["artifact_paths"]["fold_integrity_file"]).exists())
            self.assertTrue(progress_snapshots)
            self.assertEqual(progress_snapshots[-1]["phase"], "complete")
            self.assertEqual((progress_snapshots[-1].get("details") or {}).get("resolved_max_workers"), 2)

            recommended = loaded["data"]["recommended_winner"] or {}
            self.assertIn(recommended.get("status"), {"recommended", "no_winner"})
            if recommended.get("status") == "recommended":
                self.assertTrue(Path(recommended["report_file"]).exists())
                promotion = service.promote_training_experiment(recommended["report_file"])
                self.assertTrue(promotion["success"], msg=promotion)
                self.assertTrue(Path((promotion["artifacts"] or {})["model_path"]).exists())

    def test_run_automated_search_uses_configured_stage5_scope(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            original_config = Path(config_path).read_text(encoding="utf-8")
            scoped_config = original_config.replace(
                f"  candidate_models_directory: {str(root / 'models' / 'candidates').replace(chr(92), '/')}",
                "\n".join(
                    [
                        f"  candidate_models_directory: {str(root / 'models' / 'candidates').replace(chr(92), '/')}",
                        "  stage5_defaults:",
                        "    target_ids: [return_threshold_h3_0_05pct]",
                        "    feature_sets: [baseline_core]",
                        "    presets: [conservative]",
                    ]
                ),
            )
            Path(config_path).write_text(scoped_config, encoding="utf-8")

            service = ResearchAppService(str(config_path))
            service.feature_data = self._build_stage12_feature_data(80)

            result = service.run_automated_search("stage5_config_scope", max_workers=2)

            self.assertTrue(result["success"], msg=result)
            summary = (result.get("data") or {}).get("summary") or {}
            self.assertEqual(summary.get("candidate_count"), 1)
            self.assertEqual(summary.get("target_count"), 1)

            configuration = service.get_configuration_summary()
            self.assertTrue(configuration["success"], msg=configuration)
            self.assertEqual(configuration["data"]["research_primary_workflow"], "search")
            self.assertEqual(configuration["data"]["research_stage5_default_target_ids"], ["return_threshold_h3_0_05pct"])
            self.assertEqual(configuration["data"]["research_stage5_default_feature_sets"], ["baseline_core"])
            self.assertEqual(configuration["data"]["research_stage5_default_presets"], ["conservative"])
            self.assertEqual(
                ((configuration["data"].get("research_defaults") or {}).get("stage5") or {}).get("target_ids"),
                ["return_threshold_h3_0_05pct"],
            )

            report_file = (result.get("artifacts") or {}).get("report_file")
            loaded = service.get_search_report(str(report_file))
            self.assertTrue(loaded["success"], msg=loaded)
            self.assertEqual(
                ((loaded["data"].get("resolved_research_defaults") or {}).get("stage5") or {}).get("feature_set_names"),
                ["baseline_core"],
            )

    def test_legacy_training_report_loads_with_missing_integrity_and_promotion_is_blocked(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ResearchAppService(str(self._write_config(root)))
            report_path = service.experiment_store.resolve_path("training_experiment_legacy_missing_integrity.json")
            payload = {
                "experiment_id": "legacy_missing_integrity",
                "experiment_name": "legacy_missing_integrity",
                "target_spec": {"spec_id": "return_threshold_h3_0_05pct"},
                "feature_set_name": "baseline_core",
                "selector_name": "correlation",
                "trainer_name": "current_ensemble",
                "selected_threshold": 0.55,
                "aggregate_metrics": {"mean_test_accuracy": 0.60},
                "baseline_comparison": {"baselines": {"majority_class": {"mean_test_accuracy": 0.55}}},
                "folds": [],
                "diagnostics": {},
                "metadata": {},
                "candidate_artifact": {"artifact_path": str(root / "missing.joblib")},
            }
            report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            loaded = service.get_training_experiment_report(str(report_path))
            self.assertTrue(loaded["success"], msg=loaded)
            self.assertEqual(loaded["data"]["integrity"]["proof_status"], "missing")
            self.assertFalse(loaded["data"]["integrity"]["integrity_contract_ok"])

            promotion = service.promote_training_experiment(str(report_path))
            self.assertFalse(promotion["success"])
            self.assertIn("integrity proof", promotion["message"].lower())


if __name__ == "__main__":
    unittest.main()
