"""Repeatable experiment orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from ..baselines import MajorityClassBaseline, PersistenceBaseline
from ..labels import get_target_horizon_bars
from ..schemas import (
    ExperimentResult,
    FoldMetrics,
    PredictionArtifacts,
    ResearchExperimentRequest,
    ResearchSplit,
)
from .evaluation_pipeline import EvaluationPipeline
from .training_pipeline import TrainingPipeline


@dataclass
class ExperimentRunner:
    """Run a training pipeline plus baselines over predefined research splits."""

    training_pipeline: TrainingPipeline
    evaluation_pipeline: EvaluationPipeline

    def run(
        self,
        *,
        request: ResearchExperimentRequest,
        feature_frame: pd.DataFrame,
        target: pd.Series,
        splits: List[ResearchSplit],
        feature_selector: Any = None,
        target_spec: Any = None,
    ) -> ExperimentResult:
        """Execute fold-local training and evaluation across walk-forward splits."""
        fold_results: List[FoldMetrics] = []
        prediction_frames: List[pd.DataFrame] = []
        threshold_frames: List[pd.DataFrame] = []
        calibration_frames: List[pd.DataFrame] = []
        fold_feature_selections: List[Dict[str, Any]] = []
        fold_integrity_rows: List[Dict[str, Any]] = []
        baseline_metrics_by_name: Dict[str, List[float]] = {name: [] for name in request.baseline_names}
        model_test_metrics: List[float] = []
        horizon_bars = int(get_target_horizon_bars(target_spec)) if target_spec is not None else 0
        feature_selection_mode = "fold_local_selector" if feature_selector is not None else "fixed_feature_columns"

        majority_baseline = MajorityClassBaseline()
        persistence_baseline = PersistenceBaseline()

        for split in splits:
            train_start, train_end, validation_start, validation_end, test_start, test_end = self._effective_boundaries(
                split=split,
                target_spec=target_spec,
            )
            fold_integrity_rows.append(
                {
                    "fold_name": split.name,
                    "target_spec_id": getattr(target_spec, "spec_id", request.target_column),
                    "horizon_bars": horizon_bars,
                    "purge_required": bool(horizon_bars > 0),
                    "feature_selection_mode": feature_selection_mode,
                    "declared_train_start": int(split.train_start),
                    "declared_train_end": int(split.train_end),
                    "effective_train_start": int(train_start),
                    "effective_train_end": int(train_end),
                    "declared_validation_start": int(split.validation_start),
                    "declared_validation_end": int(split.validation_end),
                    "effective_validation_start": int(validation_start),
                    "effective_validation_end": int(validation_end),
                    "declared_test_start": int(split.test_start),
                    "declared_test_end": int(split.test_end),
                    "effective_test_start": int(test_start),
                    "effective_test_end": int(test_end),
                    "declared_train_rows": int(max(0, int(split.train_end) - int(split.train_start))),
                    "effective_train_rows": int(max(0, int(train_end) - int(train_start))),
                    "declared_validation_rows": int(max(0, int(split.validation_end) - int(split.validation_start))),
                    "effective_validation_rows": int(max(0, int(validation_end) - int(validation_start))),
                    "declared_test_rows": int(max(0, int(split.test_end) - int(split.test_start))),
                    "effective_test_rows": int(max(0, int(test_end) - int(test_start))),
                    "purged_train_rows": int(max(0, int(split.train_end) - int(train_end))),
                    "purged_validation_rows": int(max(0, int(split.validation_end) - int(validation_end))),
                    "status": "passed",
                    "failure_reason": "",
                }
            )
            train_features = feature_frame.iloc[train_start:train_end].loc[:, request.feature_columns]
            train_target = target.iloc[train_start:train_end]
            validation_features = feature_frame.iloc[validation_start:validation_end].loc[:, request.feature_columns]
            validation_target = target.iloc[validation_start:validation_end]
            test_features = feature_frame.iloc[test_start:test_end].loc[:, request.feature_columns]
            test_target = target.iloc[test_start:test_end]

            selection_result = None
            selected_columns = list(train_features.columns)
            if feature_selector is not None:
                selection_result = feature_selector.select(train_features, train_target)
                selected_columns = list(selection_result.selected_columns) or list(train_features.columns)
                train_features = train_features.loc[:, selected_columns]
                validation_features = validation_features.loc[:, selected_columns]
                test_features = test_features.loc[:, selected_columns]
                fold_feature_selections.append(
                    {
                        "fold_name": split.name,
                        "selector_name": selection_result.selector_name,
                        "candidate_count": len(request.feature_columns),
                        "selected_count": len(selected_columns),
                        "selected_columns": list(selected_columns),
                        "ranking_rows": list(selection_result.ranking_rows),
                    }
                )
            else:
                fold_feature_selections.append(
                    {
                        "fold_name": split.name,
                        "selector_name": "fixed_feature_columns",
                        "candidate_count": len(request.feature_columns),
                        "selected_count": len(selected_columns),
                        "selected_columns": list(selected_columns),
                        "ranking_rows": [],
                    }
                )

            segment_results = self.training_pipeline.run_segments(
                train_features,
                train_target,
                {"validation": validation_features, "test": test_features},
            )
            validation_result = segment_results["validation"]
            test_result = segment_results["test"]
            validation_probability = self._resolve_series(validation_result, "probabilities", validation_features.index, 0.5)
            validation_confidence = self._resolve_series(validation_result, "confidence", validation_features.index, 0.5)
            validation_prediction = self._resolve_series(validation_result, "prediction", validation_features.index, 0.0)

            validation_eval = self.evaluation_pipeline.evaluate_segment(
                timestamps=validation_features.index,
                target=validation_target,
                prediction=validation_prediction,
                probability=validation_probability,
                confidence=validation_confidence,
                fold_name=split.name,
                split_segment="validation",
                model_name=request.trainer_name,
            )
            fold_results.append(
                FoldMetrics(
                    fold_name=split.name,
                    model_name=request.trainer_name,
                    split_segment="validation",
                    train_rows=len(train_features),
                    validation_rows=len(validation_features),
                    test_rows=len(test_features),
                    metrics=validation_eval.summary,
                )
            )
            prediction_frames.append(validation_eval.prediction_rows)
            threshold_frames.append(validation_eval.threshold_metrics.assign(fold_name=split.name, split_segment="validation", model_name=request.trainer_name))
            calibration_frames.append(validation_eval.calibration_bins.assign(fold_name=split.name, split_segment="validation", model_name=request.trainer_name))

            test_probability = self._resolve_series(test_result, "probabilities", test_features.index, 0.5)
            test_confidence = self._resolve_series(test_result, "confidence", test_features.index, 0.5)
            test_prediction = self._resolve_series(test_result, "prediction", test_features.index, 0.0)

            test_eval = self.evaluation_pipeline.evaluate_segment(
                timestamps=test_features.index,
                target=test_target,
                prediction=test_prediction,
                probability=test_probability,
                confidence=test_confidence,
                fold_name=split.name,
                split_segment="test",
                model_name=request.trainer_name,
            )
            fold_results.append(
                FoldMetrics(
                    fold_name=split.name,
                    model_name=request.trainer_name,
                    split_segment="test",
                    train_rows=len(train_features),
                    validation_rows=len(validation_features),
                    test_rows=len(test_features),
                    metrics=test_eval.summary,
                )
            )
            model_test_metrics.append(float(test_eval.summary.get("accuracy") or 0.0))
            prediction_frames.append(test_eval.prediction_rows)
            threshold_frames.append(test_eval.threshold_metrics.assign(fold_name=split.name, split_segment="test", model_name=request.trainer_name))
            calibration_frames.append(test_eval.calibration_bins.assign(fold_name=split.name, split_segment="test", model_name=request.trainer_name))

            for baseline_name in request.baseline_names:
                baseline_eval = self._evaluate_baseline(
                    baseline_name=baseline_name,
                    full_feature_frame=feature_frame,
                    full_target=target,
                    train_target=train_target,
                    validation_target=validation_target,
                    test_target=test_target,
                    validation_index=validation_features.index,
                    test_index=test_features.index,
                    fold_name=split.name,
                    target_spec=target_spec,
                )
                fold_results.extend(baseline_eval["folds"])
                prediction_frames.extend(baseline_eval["prediction_frames"])
                threshold_frames.extend(baseline_eval["threshold_frames"])
                calibration_frames.extend(baseline_eval["calibration_frames"])
                baseline_metrics_by_name[baseline_name].append(float(baseline_eval["test_accuracy"]))

        predictions_frame = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        threshold_frame = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
        calibration_frame = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()

        aggregate_accuracy = sum(model_test_metrics) / len(model_test_metrics) if model_test_metrics else 0.0
        baseline_comparison = {
            "model_name": request.trainer_name,
            "model_mean_test_accuracy": float(aggregate_accuracy),
            "baselines": {
                name: {
                    "mean_test_accuracy": float(sum(values) / len(values)) if values else 0.0,
                    "fold_count": len(values),
                }
                for name, values in baseline_metrics_by_name.items()
            },
        }

        prediction_artifacts = [
            PredictionArtifacts(
                metadata={
                    "prediction_rows": predictions_frame.to_dict(orient="records") if not predictions_frame.empty else [],
                    "threshold_rows": threshold_frame.to_dict(orient="records") if not threshold_frame.empty else [],
                    "calibration_rows": calibration_frame.to_dict(orient="records") if not calibration_frame.empty else [],
                }
            )
        ]

        return ExperimentResult(
            experiment_name=request.experiment_name,
            target_column=request.target_column,
            feature_columns=list(request.feature_columns),
            request={
                "trainer_name": request.trainer_name,
                "baseline_names": list(request.baseline_names),
                "train_size": request.train_size,
                "validation_size": request.validation_size,
                "test_size": request.test_size,
                "step_size": request.step_size,
                "threshold_list": list(request.threshold_list),
                "expanding_window": request.expanding_window,
            },
            fold_boundaries=[split.to_dict() for split in splits],
            folds=fold_results,
            aggregate_metrics={
                "mean_test_accuracy": float(aggregate_accuracy),
                "fold_count": float(len(splits)),
            },
            baseline_comparison=baseline_comparison,
            calibration_summary=self._summarize_frame(calibration_frame, group_columns=["model_name", "split_segment"]),
            threshold_summary=self._summarize_frame(threshold_frame, group_columns=["model_name", "split_segment", "threshold"]),
            prediction_artifacts=prediction_artifacts,
            metadata={
                "target_spec_id": getattr(target_spec, "spec_id", request.target_column),
                "horizon_bars": horizon_bars,
                "purge_required": bool(horizon_bars > 0),
                "feature_selection_mode": feature_selection_mode,
                "feature_selection_note": (
                    "Stage 3 applied fold-local feature selection on train-fold data only."
                    if feature_selector is not None
                    else "This experiment used a fixed research feature-column list without full-dataset target tuning."
                ),
                "fold_feature_selections": fold_feature_selections,
                "integrity_fold_rows": fold_integrity_rows,
            },
        )

    def _effective_boundaries(
        self,
        *,
        split: ResearchSplit,
        target_spec: Any = None,
    ) -> tuple[int, int, int, int, int, int]:
        horizon_bars = int(get_target_horizon_bars(target_spec)) if target_spec is not None else 0
        train_start = int(split.train_start)
        train_end = int(split.train_end) - horizon_bars
        validation_start = int(split.validation_start)
        validation_end = int(split.validation_end) - horizon_bars
        test_start = int(split.test_start)
        test_end = int(split.test_end)

        if train_end <= train_start:
            raise ValueError(
                f"Fold {split.name} has no train rows after purging {horizon_bars} horizon bars from the train segment"
            )
        if validation_end <= validation_start:
            raise ValueError(
                f"Fold {split.name} has no validation rows after purging {horizon_bars} horizon bars from the validation segment"
            )
        if test_end <= test_start:
            raise ValueError(f"Fold {split.name} has no test rows available for evaluation")

        return train_start, train_end, validation_start, validation_end, test_start, test_end

    def _evaluate_baseline(
        self,
        *,
        baseline_name: str,
        full_feature_frame: pd.DataFrame,
        full_target: pd.Series,
        train_target: pd.Series,
        validation_target: pd.Series,
        test_target: pd.Series,
        validation_index: pd.Index,
        test_index: pd.Index,
        fold_name: str,
        target_spec: Any = None,
    ) -> Dict[str, Any]:
        if baseline_name == "majority_class":
            baseline = MajorityClassBaseline()
            validation_output = baseline.predict(train_target, validation_index)
            test_output = baseline.predict(train_target, test_index)
        elif baseline_name == "persistence":
            baseline = PersistenceBaseline()
            validation_output = baseline.predict(
                full_feature_frame,
                validation_index,
                target_spec=target_spec,
                fallback_target=full_target,
            )
            test_output = baseline.predict(
                full_feature_frame,
                test_index,
                target_spec=target_spec,
                fallback_target=full_target,
            )
        else:
            raise ValueError(f"Unsupported Stage 1 baseline: {baseline_name}")

        validation_eval = self.evaluation_pipeline.evaluate_segment(
            timestamps=validation_index,
            target=validation_target,
            prediction=validation_output.prediction,
            probability=validation_output.probability,
            confidence=validation_output.confidence,
            fold_name=fold_name,
            split_segment="validation",
            model_name=baseline_name,
        )
        test_eval = self.evaluation_pipeline.evaluate_segment(
            timestamps=test_index,
            target=test_target,
            prediction=test_output.prediction,
            probability=test_output.probability,
            confidence=test_output.confidence,
            fold_name=fold_name,
            split_segment="test",
            model_name=baseline_name,
        )

        return {
            "folds": [
                FoldMetrics(
                    fold_name=fold_name,
                    model_name=baseline_name,
                    split_segment="validation",
                    train_rows=len(train_target),
                    validation_rows=len(validation_index),
                    test_rows=len(test_index),
                    metrics=validation_eval.summary,
                ),
                FoldMetrics(
                    fold_name=fold_name,
                    model_name=baseline_name,
                    split_segment="test",
                    train_rows=len(train_target),
                    validation_rows=len(validation_index),
                    test_rows=len(test_index),
                    metrics=test_eval.summary,
                ),
            ],
            "prediction_frames": [validation_eval.prediction_rows, test_eval.prediction_rows],
            "threshold_frames": [
                validation_eval.threshold_metrics.assign(fold_name=fold_name, split_segment="validation", model_name=baseline_name),
                test_eval.threshold_metrics.assign(fold_name=fold_name, split_segment="test", model_name=baseline_name),
            ],
            "calibration_frames": [
                validation_eval.calibration_bins.assign(fold_name=fold_name, split_segment="validation", model_name=baseline_name),
                test_eval.calibration_bins.assign(fold_name=fold_name, split_segment="test", model_name=baseline_name),
            ],
            "test_accuracy": test_eval.summary.get("accuracy") or 0.0,
        }

    def _resolve_series(
        self,
        result: Dict[str, Any],
        key: str,
        index: pd.Index,
        fallback_value: float,
    ) -> pd.Series:
        raw_value = result.get(key)
        if isinstance(raw_value, pd.Series):
            return pd.to_numeric(raw_value.reindex(index), errors="coerce").fillna(fallback_value)
        if raw_value is None:
            return pd.Series(fallback_value, index=index, dtype="float64")
        return pd.Series(raw_value, index=index, dtype="float64")

    def _summarize_frame(self, frame: pd.DataFrame, *, group_columns: Iterable[str]) -> Dict[str, Any]:
        if frame.empty:
            return {}
        group_column_list = list(group_columns)
        numeric_columns = [
            column
            for column in frame.columns
            if column not in group_column_list and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric_columns:
            return {}
        grouped = frame.groupby(group_column_list, dropna=False)[numeric_columns].mean(numeric_only=True).reset_index()
        return {"rows": grouped.to_dict(orient="records")}
