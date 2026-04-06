"""Research/model workflow orchestration."""

from __future__ import annotations

import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, TYPE_CHECKING

import pandas as pd
from loguru import logger
from src.research import (
    CandidateArtifact,
    EvaluationPipeline,
    ExperimentRunner,
    FeatureStudyRequest,
    FeatureStudyResult,
    FeatureStudyRunner,
    FixedHorizonDirectionSpec,
    LegacyRuntimeDirectionSpec,
    NeutralBandLabelSpec,
    PromotionManifest,
    ResearchExperimentRequest,
    ReturnThresholdLabelSpec,
    SearchCandidateSummary,
    SearchRequest,
    SearchResult,
    SearchRunner,
    TrainingExperimentRequest,
    TrainingExperimentResult,
    TrainerRegistry,
    TargetStudyRequest,
    TargetStudyResult,
    TargetStudyRunner,
    TrainingPipeline,
    VolatilityAdjustedMoveSpec,
    WalkForwardSplitter,
    build_binary_target_labels,
    build_experiment_integrity_proof,
    build_target_result_stub,
    build_feature_selector,
    resolve_integrity_payload,
    build_search_diagnostics,
    build_training_experiment_diagnostics,
    get_default_promoted_model_path,
    resolve_stage5_preset_definitions,
    resolve_feature_sets,
    resolve_research_defaults,
    serialize_target_spec,
    summarize_selected_threshold_metrics,
)
from src.research.training_experiment import build_threshold_summary_rows, select_best_threshold
from src.research.trainers import CurrentEnsembleTrainer

if TYPE_CHECKING:
    from src.app_service import ResearchAppService


class ResearchWorkflowService:
    """Research/model orchestration delegated from the app facade."""

    def __init__(self, service: "ResearchAppService") -> None:
        self.service = service
        self.research_defaults = resolve_research_defaults(self.service.config)

    def _resolved_research_defaults_snapshot(self) -> Dict[str, Any]:
        return self.research_defaults.to_dict()

    def _compute_split_sizes(self, total_rows: int) -> tuple[int, int, int]:
        common_defaults = self.research_defaults.common
        train_size = max(int(total_rows * common_defaults.train_fraction), 1)
        validation_size = max(int(total_rows * common_defaults.validation_fraction), 1)
        test_size = max(int(total_rows * common_defaults.test_fraction), 1)

        while train_size + validation_size + test_size > total_rows and train_size > 1:
            train_size -= 1
        while train_size + validation_size + test_size > total_rows and validation_size > 1:
            validation_size -= 1
        while train_size + validation_size + test_size > total_rows and test_size > 1:
            test_size -= 1

        return train_size, validation_size, test_size

    def _resolve_target_specs_by_ids(
        self,
        serialized_target_specs: list[Dict[str, Any]],
        target_ids: list[str],
        *,
        config_key: str,
    ) -> list[Dict[str, Any]]:
        available_target_ids = [str(spec.get("spec_id") or "").strip() for spec in serialized_target_specs]
        unknown_target_ids = [target_id for target_id in target_ids if target_id not in available_target_ids]
        if unknown_target_ids:
            raise ValueError(f"Unsupported {config_key} values: " + ", ".join(sorted(unknown_target_ids)))

        selected_specs = [
            dict(spec)
            for spec in serialized_target_specs
            if str(spec.get("spec_id") or "").strip() in target_ids
        ]
        if not selected_specs:
            raise ValueError(f"{config_key} resolved zero targets")
        return selected_specs

    def _resolve_feature_set_names(
        self,
        available_feature_columns: list[str],
        requested_feature_set_names: list[str],
        *,
        config_key: str,
    ) -> list[str]:
        resolved_feature_sets = resolve_feature_sets(available_feature_columns)
        available_feature_set_names = [feature_set.name for feature_set in resolved_feature_sets if feature_set.columns]
        unknown_feature_sets = [
            feature_set_name
            for feature_set_name in requested_feature_set_names
            if feature_set_name not in available_feature_set_names
        ]
        if unknown_feature_sets:
            raise ValueError(f"Unsupported {config_key} values: " + ", ".join(sorted(unknown_feature_sets)))
        return list(requested_feature_set_names)

    def import_and_prepare_data(self) -> Dict[str, Any]:
        logger.info("Starting local dataset import and preparation...")
        try:
            csv_path, historical_data = self.service.data_collector.import_default_dataset()
            is_valid, issues = self.service.data_collector.validate_data_quality(historical_data)

            if historical_data.empty:
                logger.error("Imported dataset is empty after normalization")
                return self.service._response(False, "Imported dataset is empty after normalization")

            self.service.data_collector.save_data_to_db(historical_data, "raw_data")
            self.service.feature_data = self.service.feature_engineer.create_feature_matrix(
                historical_data,
                include_targets=True,
            )

            if self.service.feature_data.empty:
                logger.error("Failed to create features")
                return self.service._response(False, "Failed to create features")

            target_column = self.service.get_target_column()
            self.service.selected_features = self.service.feature_engineer.select_features(
                self.service.feature_data,
                target_column=target_column,
                method="correlation",
                max_features=30,
            )

            if not self.service.selected_features:
                logger.error("Feature selection failed")
                return self.service._response(False, "Feature selection failed")

            self.service.last_import_summary = {
                "path": str(csv_path),
                "rows": len(historical_data),
                "selected_features": len(self.service.selected_features),
                "data_valid": is_valid,
                "issues": issues,
                "feature_rows": len(self.service.feature_data),
                "target_column": target_column,
            }
            preview_frame = historical_data.reset_index().head(20)
            self.service.latest_data_preview = preview_frame.to_dict(orient="records")

            logger.info(
                f"Local dataset preparation complete - {len(historical_data)} raw rows, "
                f"{len(self.service.selected_features)} selected features"
            )
            return self.service._response(
                True,
                f"Imported {len(historical_data)} rows and selected {len(self.service.selected_features)} features",
                data={
                    "summary": self.service.last_import_summary,
                    "selected_features": list(self.service.selected_features),
                    "preview": self.service.latest_data_preview,
                },
                errors=list(issues),
            )
        except Exception as exc:
            logger.error(f"Failed to import local dataset: {exc}")
            return self.service._response(False, f"Local dataset import failed: {exc}", errors=[str(exc)])

    def train_models(self, model_name: str = "default") -> Dict[str, Any]:
        if self.service.feature_data is None or not self.service.selected_features:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        logger.info("Starting AI model training...")
        training_results = self.service.ai_model_manager.train_ensemble_models(
            self.service.feature_data,
            self.service.selected_features,
            target_column=self.service.get_target_column(),
        )

        if not training_results:
            logger.error("Model training failed")
            return self.service._response(False, "Model training failed")

        safe_name = self.service._sanitize_model_name(model_name or "default")
        model_path = Path("models") / f"{safe_name}.joblib"
        self.service.ai_model_manager.save_models(model_path)
        self.service.loaded_model_path = str(model_path)
        self.service.selected_features = list(self.service.ai_model_manager.feature_columns)
        self.service.latest_training_results = training_results

        return self.service._response(
            True,
            f"Training complete. Saved model file: {model_path.name}",
            data={
                "training_results": training_results,
                "saved_model_name": model_path.stem,
                "saved_model_path": str(model_path),
                "selected_features": list(self.service.selected_features),
            },
            artifacts={"model_path": str(model_path)},
        )

    def run_backtest(self) -> Dict[str, Any]:
        logger.info("Starting professional backtest...")
        self.service._reload_runtime_from_disk()
        runtime_feature_columns = self.service.get_runtime_prediction_features()

        if not self.service.ai_model_manager.models:
            logger.error("Please train or load the models first")
            return self.service._response(False, "Please train or load the models first")
        if self.service.feature_data is None or not runtime_feature_columns:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        try:
            result = self.service.backtester.run_backtest(
                self.service.feature_data,
                self.service.ai_model_manager,
                runtime_feature_columns,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/backtest_result_{timestamp}.json"
            chart_file = f"reports/backtest_chart_{timestamp}.png"
            summary_file = ""

            self.service.backtester.save_results(result, result_file)
            self.service.backtester.plot_results(result, chart_file)

            trade_summary = self.service.backtester.get_trade_summary()
            if not trade_summary.empty:
                summary_file = f"reports/trade_summary_{timestamp}.csv"
                trade_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
                logger.info(f"Trade summary saved: {summary_file}")

            result_summary = self.service._serialize_backtest_result(result)
            artifacts = {
                "report_file": result_file,
                "chart_file": chart_file,
            }
            if summary_file:
                artifacts["trade_summary_file"] = summary_file

            self.service.latest_backtest_summary = result_summary
            self.service.latest_backtest_artifacts = artifacts

            return self.service._response(
                True,
                "Backtest completed successfully",
                data={
                    "summary": result_summary,
                    "trade_summary_rows": len(trade_summary),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Professional backtest failed: {exc}")
            return self.service._response(False, f"Professional backtest failed: {exc}", errors=[str(exc)])

    def run_model_test(self) -> Dict[str, Any]:
        logger.info("Starting model test...")
        self.service._reload_runtime_from_disk()
        runtime_feature_columns = self.service.get_runtime_prediction_features()

        if not self.service.ai_model_manager.models:
            logger.error("Please train or load the models first")
            return self.service._response(False, "Please train or load the models first")
        if self.service.feature_data is None or not runtime_feature_columns:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        target_column = self.service.ai_model_manager.target_column or self.service.get_target_column()
        if target_column not in self.service.feature_data.columns:
            message = f"Prepared feature data is missing target column: {target_column}"
            logger.error(message)
            return self.service._response(False, message)

        try:
            prediction_frame = self.service.ai_model_manager.predict_ensemble_batch(
                self.service.feature_data,
                feature_columns=runtime_feature_columns,
                method="voting",
            )
            result = self.service.model_tester.evaluate(
                self.service.feature_data,
                prediction_frame,
                self.service.feature_data[target_column],
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/model_test_result_{timestamp}.json"
            rows_file = f"reports/model_test_rows_{timestamp}.csv"

            threshold_rows = result.threshold_performance.to_dict(orient="records")
            confidence_bucket_rows = result.confidence_buckets.to_dict(orient="records")
            result.row_evaluations.to_csv(rows_file, index=False, encoding="utf-8-sig")

            report_payload = {
                "model_test_summary": result.summary,
                "threshold_performance": threshold_rows,
                "confidence_buckets": confidence_bucket_rows,
                "target_column": target_column,
                "selected_features": list(runtime_feature_columns),
                "research_selected_features": list(self.service.selected_features),
                "loaded_model_path": self.service.loaded_model_path,
                "row_evaluations_file": rows_file,
            }
            with open(result_file, "w", encoding="utf-8") as report_file:
                json.dump(report_payload, report_file, ensure_ascii=False, indent=2)

            result_summary = self.service._serialize_model_test_result(result)
            artifacts = {
                "report_file": result_file,
                "evaluation_rows_file": rows_file,
            }
            self.service.latest_model_test_summary = result_summary
            self.service.latest_model_test_artifacts = artifacts

            return self.service._response(
                True,
                "Model test completed successfully",
                data={
                    "summary": result_summary,
                    "threshold_performance": threshold_rows,
                    "confidence_buckets": confidence_bucket_rows,
                    "evaluation_preview": result.row_evaluations.head(200).to_dict(orient="records"),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Model test failed: {exc}")
            return self.service._response(False, f"Model test failed: {exc}", errors=[str(exc)])

    def run_research_experiment(self, experiment_name: str = "diagnostic_single_experiment") -> Dict[str, Any]:
        logger.info("Starting diagnostic single-experiment run...")
        self.service._reload_runtime_from_disk()

        if self.service.feature_data is None or self.service.feature_data.empty:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        target_column = self.service.get_target_column()
        if target_column not in self.service.feature_data.columns:
            message = f"Prepared feature data is missing target column: {target_column}"
            logger.error(message)
            return self.service._response(False, message)

        try:
            feature_set = self._resolve_stage12_feature_set()
        except Exception as exc:
            logger.error(f"Failed to resolve Stage 1 feature set: {exc}")
            return self.service._response(False, f"Diagnostic single-experiment run failed: {exc}", errors=[str(exc)])
        feature_columns = list(feature_set.columns)

        try:
            runtime_horizon = int(str(target_column).rsplit("_", 1)[-1])
        except ValueError:
            runtime_horizon = 1
        runtime_target_spec = LegacyRuntimeDirectionSpec(
            spec_id=f"legacy_{target_column.lower()}",
            display_name=f"Legacy Runtime Target ({target_column})",
            horizon_bars=runtime_horizon,
        )
        target_series = build_binary_target_labels(self.service.feature_data, runtime_target_spec)
        try:
            execution = self._execute_fixed_feature_research_diagnostic(
                run_name=experiment_name,
                target_series=target_series,
                artifact_prefix=f"research_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                target_spec=runtime_target_spec,
                feature_columns=feature_columns,
                feature_set=feature_set,
                feature_selection_note=(
                    f"Stage 1 uses the fixed research feature set '{feature_set.name}' to avoid full-dataset feature-selection leakage."
                ),
            )
            result = execution["result"]
            summary = execution["summary"]
            artifacts = execution["artifacts"]
            self.service.latest_research_experiment_summary = summary
            self.service.latest_research_experiment_artifacts = artifacts

            return self.service._response(
                True,
                "Diagnostic single-experiment run completed successfully",
                data={
                    "summary": summary,
                    "aggregate_metrics": result.aggregate_metrics,
                    "baseline_comparison": result.baseline_comparison,
                    "calibration_summary": result.calibration_summary,
                    "threshold_summary": result.threshold_summary,
                    "folds": [fold.__dict__ for fold in result.folds],
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Diagnostic single-experiment run failed: {exc}")
            return self.service._response(False, f"Diagnostic single-experiment run failed: {exc}", errors=[str(exc)])

    def run_target_study(self, study_name: str = "diagnostic_target_comparison") -> Dict[str, Any]:
        logger.info("Starting diagnostic target-comparison study...")
        self.service._reload_runtime_from_disk()

        if self.service.feature_data is None or self.service.feature_data.empty:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        try:
            feature_set = self._resolve_stage12_feature_set()
        except Exception as exc:
            logger.error(f"Failed to resolve Stage 2 feature set: {exc}")
            return self.service._response(False, f"Diagnostic target-comparison study failed: {exc}", errors=[str(exc)])
        feature_columns = list(feature_set.columns)

        target_specs = self._build_default_target_specs()
        request = self._build_default_target_study_request(
            study_name=study_name,
            feature_columns=feature_columns,
            target_specs=target_specs,
            total_rows=len(self.service.feature_data),
        )
        target_runner = TargetStudyRunner()
        materialized_targets = target_runner.materialize_targets(self.service.feature_data, target_specs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_results = []
        study_slug = self.service._sanitize_model_name(study_name)

        for materialized in materialized_targets:
            target_result = build_target_result_stub(materialized)
            try:
                execution = self._execute_fixed_feature_research_diagnostic(
                    run_name=f"{study_name}_{target_result.target_id}",
                    target_series=materialized.target,
                    artifact_prefix=f"research_experiment_{timestamp}_{study_slug}_{target_result.target_id}",
                    target_spec=materialized.spec,
                    feature_columns=feature_columns,
                    feature_set=feature_set,
                    feature_selection_note=(
                        f"Stage 2 uses the fixed research feature set '{feature_set.name}' to keep target comparisons leakage-safe and apples-to-apples."
                    ),
                )
                target_result.experiment_report_file = execution["artifacts"]["report_file"]
                target_result.experiment_summary = execution["summary"]
                target_result.aggregate_metrics = execution["result"].aggregate_metrics
                target_result.baseline_comparison = execution["result"].baseline_comparison
                target_result.integrity = execution["integrity"]
            except Exception as exc:
                logger.error(f"Target study evaluation failed for {target_result.target_id}: {exc}")
                target_result.error = str(exc)
            target_results.append(target_result)

        comparison_rows = target_runner.build_comparison_rows(target_results)
        comparison_file = self.service.experiment_store.resolve_path(f"target_study_{timestamp}_{study_slug}_comparison.csv")
        pd.DataFrame(comparison_rows).to_csv(comparison_file, index=False, encoding="utf-8-sig")
        integrity_rows = [
            {
                "target_id": result.target_id,
                "display_name": result.display_name,
                "proof_status": (result.integrity or {}).get("proof_status", "missing"),
                "integrity_contract_ok": bool((result.integrity or {}).get("integrity_contract_ok")),
                "invalid_fold_count": ((result.integrity or {}).get("overview") or {}).get("invalid_fold_count"),
                "total_purged_train_rows": ((result.integrity or {}).get("overview") or {}).get("total_purged_train_rows"),
                "total_purged_validation_rows": ((result.integrity or {}).get("overview") or {}).get("total_purged_validation_rows"),
                "failure_reasons": ", ".join(((result.integrity or {}).get("overview") or {}).get("contract_failure_reasons") or []),
            }
            for result in target_results
        ]
        target_integrity = self._aggregate_integrity_payloads(integrity_rows, scope_label="target")
        target_integrity_file = self.service.experiment_store.resolve_path(f"target_study_{timestamp}_{study_slug}_integrity.csv")
        pd.DataFrame(target_integrity.get("fold_rows") or []).to_csv(target_integrity_file, index=False, encoding="utf-8-sig")

        target_study_result = TargetStudyResult(
            study_name=study_name,
            feature_columns=list(feature_columns),
            request={
                "trainer_name": request.trainer_name,
                "baseline_names": list(request.baseline_names),
                "train_size": request.train_size,
                "validation_size": request.validation_size,
                "test_size": request.test_size,
                "step_size": request.step_size,
                "threshold_list": list(request.threshold_list),
                "expanding_window": request.expanding_window,
                "target_specs": list(request.target_specs),
            },
            target_results=target_results,
            comparison_rows=comparison_rows,
            artifact_paths={"comparison_file": str(comparison_file)},
            integrity=target_integrity,
            integrity_artifact_paths={"fold_integrity_file": str(target_integrity_file)},
            metadata={
                "runtime_target_column": self.service.get_target_column(),
                "research_feature_set_name": feature_set.name,
                "research_feature_set_display_name": feature_set.display_name,
                "runtime_target_unchanged": True,
                "deferred_items": ["true_multiclass_support", "triple_barrier_labels"],
                "resolved_research_defaults": self._resolved_research_defaults_snapshot(),
            },
        )
        report_file = self.service.experiment_store.resolve_path(f"target_study_{timestamp}_{study_slug}.json")
        self.service.experiment_store.save_result(target_study_result, report_file)

        successful_rows = [row for row in comparison_rows if not row.get("error")]
        best_row = max(
            successful_rows,
            key=lambda row: row.get("model_mean_test_accuracy") or float("-inf"),
            default=None,
        )
        summary = {
            "study_name": study_name,
            "target_count": len(target_results),
            "successful_targets": len(successful_rows),
            "feature_set_name": feature_set.name,
            "feature_set_display_name": feature_set.display_name,
            "best_target_name": (best_row or {}).get("display_name"),
            "best_mean_test_accuracy": (best_row or {}).get("model_mean_test_accuracy"),
        }
        artifacts = {
            "report_file": str(report_file),
            "comparison_file": str(comparison_file),
            "fold_integrity_file": str(target_integrity_file),
        }
        self.service.latest_target_study_summary = summary
        self.service.latest_target_study_artifacts = artifacts

        if not successful_rows:
            return self.service._response(
                False,
                    "Diagnostic target-comparison study completed with no successful target evaluations",
                data={
                    "summary": summary,
                    "comparison_rows": comparison_rows,
                    "target_results": [asdict(result) for result in target_results],
                },
                artifacts=artifacts,
                errors=[result.error for result in target_results if result.error],
            )

        return self.service._response(
            True,
            "Diagnostic target-comparison study completed successfully",
            data={
                "summary": summary,
                "comparison_rows": comparison_rows,
                "target_results": [asdict(result) for result in target_results],
            },
            artifacts=artifacts,
        )

    def run_feature_study(self, study_name: str = "diagnostic_feature_comparison") -> Dict[str, Any]:
        logger.info("Starting diagnostic feature-comparison study...")
        bootstrap_feature_columns = list(self.service.selected_features)
        self.service._reload_runtime_from_disk()
        if bootstrap_feature_columns:
            self.service.selected_features = list(bootstrap_feature_columns)

        if self.service.feature_data is None or self.service.feature_data.empty:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        target_specs = self._build_default_feature_study_target_specs()
        request = self._build_default_feature_study_request(
            study_name=study_name,
            target_specs=target_specs,
            total_rows=len(self.service.feature_data),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_slug = self.service._sanitize_model_name(study_name)
        runner = FeatureStudyRunner()

        try:
            result = runner.run(
                request=request,
                feature_frame=self.service.feature_data,
                target_specs=target_specs,
                experiment_executor=self._execute_feature_study_experiment,
                artifact_timestamp=timestamp,
            )
            inventory_file = self.service.experiment_store.resolve_path(f"feature_study_{timestamp}_{study_slug}_inventory.csv")
            fold_selection_file = self.service.experiment_store.resolve_path(f"feature_study_{timestamp}_{study_slug}_folds.csv")
            stability_file = self.service.experiment_store.resolve_path(f"feature_study_{timestamp}_{study_slug}_stability.csv")
            comparison_file = self.service.experiment_store.resolve_path(f"feature_study_{timestamp}_{study_slug}_comparison.csv")
            report_file = self.service.experiment_store.resolve_path(f"feature_study_{timestamp}_{study_slug}.json")

            runner.build_inventory_frame(result).to_csv(inventory_file, index=False, encoding="utf-8-sig")
            runner.build_fold_selection_frame(result).to_csv(fold_selection_file, index=False, encoding="utf-8-sig")
            runner.build_stability_frame(result).to_csv(stability_file, index=False, encoding="utf-8-sig")
            pd.DataFrame(result.comparison_rows).to_csv(comparison_file, index=False, encoding="utf-8-sig")

            result.artifact_paths = {
                "inventory_file": str(inventory_file),
                "fold_selection_file": str(fold_selection_file),
                "stability_file": str(stability_file),
                "comparison_file": str(comparison_file),
            }
            result.metadata = {
                **dict(result.metadata or {}),
                "resolved_research_defaults": self._resolved_research_defaults_snapshot(),
            }
            self.service.experiment_store.save_result(result, report_file)

            successful_rows = [row for row in result.comparison_rows if not row.get("error")]
            best_row = max(
                successful_rows,
                key=lambda row: row.get("mean_test_accuracy") or float("-inf"),
                default=None,
            )
            summary = {
                "study_name": study_name,
                "feature_set_count": len(request.feature_set_names),
                "target_count": len(request.target_specs),
                "successful_runs": len(successful_rows),
                "working_target_id": result.working_target_id,
                "best_feature_set_name": (best_row or {}).get("feature_set_display_name"),
                "best_mean_test_accuracy": (best_row or {}).get("mean_test_accuracy"),
            }
            artifacts = {
                "report_file": str(report_file),
                **result.artifact_paths,
            }
            self.service.latest_feature_study_summary = summary
            self.service.latest_feature_study_artifacts = artifacts

            if not successful_rows:
                return self.service._response(
                    False,
                    "Diagnostic feature-comparison study completed with no successful feature-set evaluations",
                    data={
                        "summary": summary,
                        "comparison_rows": result.comparison_rows,
                        "set_results": [asdict(set_result) for set_result in result.set_results],
                        "inventory_rows": [asdict(row) for row in result.inventory_rows],
                    },
                    artifacts=artifacts,
                    errors=[set_result.error for set_result in result.set_results if set_result.error],
                )

            return self.service._response(
                True,
                "Diagnostic feature-comparison study completed successfully",
                data={
                    "summary": summary,
                    "comparison_rows": result.comparison_rows,
                    "set_results": [asdict(set_result) for set_result in result.set_results],
                    "inventory_rows": [asdict(row) for row in result.inventory_rows],
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Diagnostic feature-comparison study failed: {exc}")
            return self.service._response(False, f"Diagnostic feature-comparison study failed: {exc}", errors=[str(exc)])

    def run_training_experiment(self, experiment_name: str = "diagnostic_candidate_training") -> Dict[str, Any]:
        logger.info("Starting diagnostic candidate-training run...")
        self.service._reload_runtime_from_disk()

        if self.service.feature_data is None or self.service.feature_data.empty:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_slug = self.service._sanitize_model_name(experiment_name)
            target_spec = self._build_stage4_working_target_spec()
            request = self._build_default_training_experiment_request(
                experiment_name=experiment_name,
                target_spec=target_spec,
                total_rows=len(self.service.feature_data),
                timestamp=timestamp,
            )
            target_series = build_binary_target_labels(self.service.feature_data, target_spec)
            feature_sets = {
                feature_set.name: feature_set
                for feature_set in resolve_feature_sets(
                    self.service.feature_data.columns,
                    [request.feature_set_name, request.comparison_feature_set_name],
                )
            }
            primary_feature_set = feature_sets[request.feature_set_name]
            comparison_feature_set = feature_sets.get(request.comparison_feature_set_name)

            trainer_registry = self._build_trainer_registry()
            trainer = trainer_registry.build(
                request.trainer_name,
                model_params=dict(request.trainer_params),
            )

            primary_execution = self._run_training_experiment_candidate(
                request=request,
                target_series=target_series,
                target_spec=target_spec,
                feature_set=primary_feature_set,
                trainer=trainer,
                artifact_prefix=f"training_experiment_{timestamp}_{experiment_slug}_{request.feature_set_name}",
            )

            comparison_runs = []
            if comparison_feature_set is not None:
                comparison_execution = self._run_training_experiment_evaluation_only(
                    request=request,
                    target_series=target_series,
                    target_spec=target_spec,
                    feature_set=comparison_feature_set,
                    trainer=trainer,
                )
                comparison_runs.append(comparison_execution)

            finalized = self._finalize_training_experiment_workflow(
                request=request,
                target_series=target_series,
                target_spec=target_spec,
                candidate_execution=primary_execution,
                comparison_runs=comparison_runs,
                artifact_prefix=f"training_experiment_{timestamp}_{experiment_slug}_{request.feature_set_name}",
                metadata={
                    "out_of_sample_note": "Walk-forward metrics come from out-of-sample folds; the candidate artifact was retrained afterward on the full eligible dataset.",
                    "comparison_feature_set_name": request.comparison_feature_set_name,
                    "selected_threshold_summary": primary_execution["selected_threshold_summary"],
                },
            )
            result = finalized["training_result"]
            summary = finalized["summary"]
            artifacts = finalized["artifacts"]
            self.service.latest_training_experiment_summary = summary
            self.service.latest_training_experiment_artifacts = artifacts

            return self.service._response(
                True,
                "Diagnostic candidate-training run completed successfully",
                data={
                    "summary": summary,
                    "aggregate_metrics": result.aggregate_metrics,
                    "baseline_comparison": result.baseline_comparison,
                    "comparison_runs": result.comparison_runs,
                    "selected_threshold": result.selected_threshold,
                    "candidate_artifact": asdict(result.candidate_artifact) if result.candidate_artifact else None,
                    "diagnostics": result.diagnostics,
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Diagnostic candidate-training run failed: {exc}")
            return self.service._response(False, f"Diagnostic candidate-training run failed: {exc}", errors=[str(exc)])

    def promote_training_experiment(self, experiment_path_or_id: str) -> Dict[str, Any]:
        logger.info("Starting Stage 4 promotion...")
        try:
            report_path = self._resolve_training_experiment_report_path(experiment_path_or_id)
            payload = self.service.experiment_store.load_result(report_path)
            integrity = resolve_integrity_payload(
                metadata=dict(payload.get("metadata") or {}),
                stored_integrity=dict(payload.get("integrity") or {}),
                expected_feature_set_name=str(payload.get("feature_set_name") or ""),
                expected_feature_selection_mode="fold_local_selector" if payload.get("selector_name") else "fixed_feature_columns",
                expected_target_spec_id=str((payload.get("target_spec") or {}).get("spec_id") or ""),
                fallback_fold_count=len(
                    {
                        str(row.get("fold_name") or "")
                        for row in (payload.get("folds") or [])
                        if str(row.get("fold_name") or "")
                    }
                ),
            )
            if not bool(integrity.get("integrity_contract_ok")):
                failure_reasons = ", ".join((integrity.get("overview") or {}).get("contract_failure_reasons") or ["integrity_contract_failed"])
                raise ValueError(f"Promotion blocked because the saved integrity proof is missing or failed: {failure_reasons}")
            candidate_payload = payload.get("candidate_artifact") or {}
            candidate_path = Path(str(candidate_payload.get("artifact_path", "")))
            if not candidate_path.exists():
                raise FileNotFoundError(f"Candidate artifact is missing: {candidate_path}")

            promoted_path = get_default_promoted_model_path("models", payload.get("experiment_name", "promoted_model"))
            promoted_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, promoted_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            promotion_slug = self.service._sanitize_model_name(payload.get("experiment_name", "promoted_model"))
            manifest = PromotionManifest(
                experiment_name=payload.get("experiment_name", ""),
                source_model_path=str(candidate_path),
                promoted_model_path=str(promoted_path),
                target_column=self.service.get_target_column(),
                feature_set_name=payload.get("feature_set_name", ""),
                selected_threshold=payload.get("selected_threshold"),
                integrity=integrity,
                integrity_artifact_paths=dict(payload.get("integrity_artifact_paths") or {}),
                metadata={
                    "experiment_id": payload.get("experiment_id"),
                    "report_file": str(report_path),
                    "trainer_name": payload.get("trainer_name"),
                    "target_spec": payload.get("target_spec"),
                },
            )
            manifest_path = self.service.experiment_store.resolve_path(f"promotion_{timestamp}_{promotion_slug}.json")
            self.service.experiment_store.save_result(manifest, manifest_path)

            payload["promotion_status"] = "promoted"
            payload["promotion_manifest_file"] = str(manifest_path)
            self._save_experiment_payload(payload, report_path)

            summary = {
                "experiment_name": payload.get("experiment_name"),
                "experiment_id": payload.get("experiment_id"),
                "promoted_model_path": str(promoted_path),
                "selected_threshold": payload.get("selected_threshold"),
                "feature_set_name": payload.get("feature_set_name"),
            }
            artifacts = {
                "report_file": str(manifest_path),
                "model_path": str(promoted_path),
            }
            self.service.latest_promotion_summary = summary
            self.service.latest_promotion_artifacts = artifacts

            return self.service._response(
                True,
                "Stage 4 promotion completed successfully",
                data={
                    "summary": summary,
                    "manifest": manifest.to_dict(),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Stage 4 promotion failed: {exc}")
            return self.service._response(False, f"Stage 4 promotion failed: {exc}", errors=[str(exc)])

    def run_automated_search(
        self,
        search_name: str = "research_search",
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
        max_workers: int | None = None,
    ) -> Dict[str, Any]:
        logger.info("Starting primary research search...")
        self.service._reload_runtime_from_disk()

        if self.service.feature_data is None or self.service.feature_data.empty:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        started_at = perf_counter()
        try:
            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="setup",
                step_label="Preparing search inputs",
                current=0,
                total=0,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_slug = self.service._sanitize_model_name(search_name)
            target_specs = self._build_stage5_search_target_specs()
            request = self._build_default_search_request(
                search_name=search_name,
                target_specs=target_specs,
                total_rows=len(self.service.feature_data),
                available_feature_columns=list(self.service.feature_data.columns),
                timestamp=timestamp,
                max_workers=max_workers,
            )
            target_specs_by_id = {
                str(serialize_target_spec(target_spec).get("spec_id")): target_spec
                for target_spec in target_specs
            }
            target_series_by_id = {
                target_spec_id: build_binary_target_labels(self.service.feature_data, target_spec)
                for target_spec_id, target_spec in target_specs_by_id.items()
            }
            feature_sets = {
                feature_set.name: feature_set
                for feature_set in resolve_feature_sets(self.service.feature_data.columns, request.feature_set_names)
            }
            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="setup",
                step_label="Resolving search presets and candidate grid",
                current=0,
                total=0,
                details={
                    "target_count": len(request.resolved_target_specs()),
                    "feature_set_count": len(feature_sets),
                    "preset_count": len(request.preset_names),
                },
            )
            preset_definitions = resolve_stage5_preset_definitions(
                self.service.ai_model_manager.model_names,
                request.preset_names,
            )
            trainer_registry = self._build_trainer_registry()
            search_runner = SearchRunner()
            candidate_configs = search_runner.build_candidate_grid(
                request=request,
                preset_definitions=preset_definitions,
            )
            total_candidates = len(candidate_configs)
            resolved_max_workers = self._resolve_search_worker_count(request.max_workers)
            execution_mode = (
                "parallel_candidate_threads"
                if resolved_max_workers > 1 and total_candidates > 1
                else "sequential"
            )
            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="running",
                step_label="Running candidate experiments",
                current=0,
                total=total_candidates,
                details={
                    "execution_mode": execution_mode,
                    "resolved_max_workers": resolved_max_workers,
                    "target_count": len(request.resolved_target_specs()),
                    "feature_set_count": len(feature_sets),
                    "preset_count": len(request.preset_names),
                    "completed_count": 0,
                    "failed_count": 0,
                    "active_count": min(total_candidates, resolved_max_workers if execution_mode != "sequential" else 1),
                },
            )

            candidate_summaries = []
            candidate_rows = []
            completed_count = 0
            failed_count = 0

            def _consume_candidate_result(candidate_result: Dict[str, Any]) -> None:
                nonlocal completed_count, failed_count
                candidate_summary = candidate_result["candidate_summary"]
                candidate_summaries.append(candidate_summary)
                candidate_rows.append(candidate_result["candidate_row"])
                if str(candidate_summary.execution_status or "completed") == "completed":
                    completed_count += 1
                else:
                    failed_count += 1
                processed_count = completed_count + failed_count
                remaining_count = max(total_candidates - processed_count, 0)
                active_count = (
                    min(resolved_max_workers, remaining_count)
                    if execution_mode != "sequential"
                    else (1 if remaining_count else 0)
                )
                self._emit_search_progress(
                    progress_callback,
                    started_at=started_at,
                    phase="running",
                    step_label=(
                        f"Completed {processed_count}/{total_candidates}: "
                        f"{candidate_summary.target_display_name} | "
                        f"{candidate_summary.feature_set_name} | "
                        f"{candidate_summary.preset_name}"
                    ),
                    current=processed_count,
                    total=total_candidates,
                    details={
                        "execution_mode": execution_mode,
                        "resolved_max_workers": resolved_max_workers,
                        "target_count": len(request.resolved_target_specs()),
                        "feature_set_count": len(feature_sets),
                        "preset_count": len(request.preset_names),
                        "completed_count": completed_count,
                        "failed_count": failed_count,
                        "active_count": active_count,
                        "last_completed_candidate_id": candidate_summary.candidate_id,
                        "target_spec_id": candidate_summary.target_spec_id,
                        "target_display_name": candidate_summary.target_display_name,
                        "feature_set_name": candidate_summary.feature_set_name,
                        "preset_name": candidate_summary.preset_name,
                    },
                )

            if execution_mode == "sequential":
                for candidate_config in candidate_configs:
                    _consume_candidate_result(
                        self._execute_search_candidate(
                            request=request,
                            candidate_config=candidate_config,
                            timestamp=timestamp,
                            search_name=search_name,
                            search_slug=search_slug,
                            target_specs_by_id=target_specs_by_id,
                            target_series_by_id=target_series_by_id,
                            feature_sets=feature_sets,
                            trainer_registry=trainer_registry,
                        )
                    )
            else:
                future_to_candidate = {}
                with ThreadPoolExecutor(max_workers=resolved_max_workers) as executor:
                    for candidate_config in candidate_configs:
                        future = executor.submit(
                            self._execute_search_candidate,
                            request=request,
                            candidate_config=candidate_config,
                            timestamp=timestamp,
                            search_name=search_name,
                            search_slug=search_slug,
                            target_specs_by_id=target_specs_by_id,
                            target_series_by_id=target_series_by_id,
                            feature_sets=feature_sets,
                            trainer_registry=trainer_registry,
                        )
                        future_to_candidate[future] = candidate_config

                    for future in as_completed(future_to_candidate):
                        candidate_config = future_to_candidate[future]
                        try:
                            candidate_result = future.result()
                        except Exception as exc:
                            logger.error(f"Stage 5 candidate execution crashed for {candidate_config.candidate_id}: {exc}")
                            candidate_result = self._build_failed_search_candidate_result(
                                candidate_config=candidate_config,
                                experiment_name=(
                                    f"{search_name}_{candidate_config.target_spec_id}_"
                                    f"{candidate_config.feature_set_name}_{candidate_config.preset_name}"
                                ),
                                error_message=str(exc),
                            )
                        _consume_candidate_result(candidate_result)

            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="finalizing",
                step_label="Ranking candidates and writing reports",
                current=total_candidates,
                total=total_candidates,
                details={
                    "execution_mode": execution_mode,
                    "resolved_max_workers": resolved_max_workers,
                    "completed_count": completed_count,
                    "failed_count": failed_count,
                    "active_count": 0,
                },
            )
            ranking = search_runner.rank_candidates(
                candidate_summaries,
                truth_gate_defaults=asdict(self.research_defaults.truth_gate),
            )
            leaderboard_rows = ranking["leaderboard_rows"]
            ranked_candidates = ranking["ordered_candidates"]
            recommended_winner = ranking["recommended_winner"]
            gate_summary = ranking["gate_summary"]
            search_diagnostics = build_search_diagnostics(ranked_candidates)
            search_integrity = self._aggregate_integrity_payloads(
                [
                    {
                        "candidate_id": candidate.candidate_id,
                        "experiment_name": candidate.experiment_name,
                        "target_spec_id": candidate.target_spec_id,
                        "feature_set_name": candidate.feature_set_name,
                        "preset_name": candidate.preset_name,
                        "proof_status": ((candidate.diagnostics or {}).get("integrity") or {}).get("proof_status", "missing"),
                        "integrity_contract_ok": bool((((candidate.diagnostics or {}).get("integrity") or {}).get("integrity_contract_ok"))),
                        "invalid_fold_count": ((((candidate.diagnostics or {}).get("integrity") or {}).get("overview") or {}).get("invalid_fold_count")),
                        "total_purged_train_rows": ((((candidate.diagnostics or {}).get("integrity") or {}).get("overview") or {}).get("total_purged_train_rows")),
                        "total_purged_validation_rows": ((((candidate.diagnostics or {}).get("integrity") or {}).get("overview") or {}).get("total_purged_validation_rows")),
                        "failure_reasons": ", ".join((((candidate.diagnostics or {}).get("integrity") or {}).get("overview") or {}).get("contract_failure_reasons") or []),
                    }
                    for candidate in ranked_candidates
                ],
                scope_label="candidate",
            )
            elapsed_seconds = round(perf_counter() - started_at, 4)

            search_result = SearchResult(
                search_id=request.search_id,
                search_name=request.search_name,
                target_spec=dict(request.target_spec),
                target_specs=request.resolved_target_specs(),
                trainer_name=request.trainer_name,
                feature_set_names=list(request.feature_set_names),
                preset_definitions=preset_definitions,
                request=asdict(request),
                candidate_count=len(ranked_candidates),
                execution_mode=execution_mode,
                resolved_max_workers=resolved_max_workers,
                successful_candidate_count=completed_count,
                failed_candidate_count=failed_count,
                candidates=list(ranked_candidates),
                leaderboard_rows=leaderboard_rows,
                recommended_winner=recommended_winner,
                diagnostics=search_diagnostics,
                integrity=search_integrity,
                metadata={
                    "gate_summary": gate_summary,
                    "execution_mode": execution_mode,
                    "resolved_max_workers": resolved_max_workers,
                    "successful_candidate_count": completed_count,
                    "failed_candidate_count": failed_count,
                    "elapsed_seconds": elapsed_seconds,
                    "resolved_research_defaults": self._resolved_research_defaults_snapshot(),
                },
            )
            report_file = self.service.experiment_store.resolve_path(f"search_run_{timestamp}_{search_slug}.json")
            leaderboard_file = self.service.experiment_store.resolve_path(f"search_run_{timestamp}_{search_slug}_leaderboard.csv")
            candidates_file = self.service.experiment_store.resolve_path(f"search_run_{timestamp}_{search_slug}_candidates.csv")
            integrity_file = self.service.experiment_store.resolve_path(f"search_run_{timestamp}_{search_slug}_integrity.csv")
            pd.DataFrame(leaderboard_rows).to_csv(leaderboard_file, index=False, encoding="utf-8-sig")
            pd.DataFrame(candidate_rows).to_csv(candidates_file, index=False, encoding="utf-8-sig")
            pd.DataFrame(search_integrity.get("fold_rows") or []).to_csv(integrity_file, index=False, encoding="utf-8-sig")
            search_result.artifact_paths = {
                "report_file": str(report_file),
                "leaderboard_file": str(leaderboard_file),
                "candidates_file": str(candidates_file),
            }
            search_result.integrity_artifact_paths = {
                "fold_integrity_file": str(integrity_file),
            }
            self.service.experiment_store.save_result(search_result, report_file)

            summary = self._build_search_summary(search_result)
            artifacts = {
                **dict(search_result.artifact_paths),
                **dict(search_result.integrity_artifact_paths),
            }
            self.service.latest_search_summary = summary
            self.service.latest_search_artifacts = artifacts
            summary["elapsed_seconds"] = elapsed_seconds
            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="complete",
                step_label="Search completed",
                current=total_candidates,
                total=total_candidates,
                details={
                    "execution_mode": execution_mode,
                    "resolved_max_workers": resolved_max_workers,
                    "completed_count": completed_count,
                    "failed_count": failed_count,
                    "active_count": 0,
                    "winner_status": recommended_winner.get("status"),
                    "winner_reason": recommended_winner.get("reason"),
                },
            )

            return self.service._response(
                True,
                "Primary research search completed successfully",
                data={
                    "summary": summary,
                    "leaderboard_rows": leaderboard_rows,
                    "recommended_winner": recommended_winner,
                    "candidate_count": len(ranked_candidates),
                    "gate_summary": gate_summary,
                    "diagnostics": search_diagnostics,
                    "integrity": search_integrity,
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Primary research search failed: {exc}")
            self._emit_search_progress(
                progress_callback,
                started_at=started_at,
                phase="failed",
                step_label="Search failed",
                current=0,
                total=0,
                details={
                    "execution_mode": "failed",
                    "resolved_max_workers": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "active_count": 0,
                    "error": str(exc),
                },
            )
            return self.service._response(False, f"Primary research search failed: {exc}", errors=[str(exc)])

    def get_model_analysis(self) -> Dict[str, Any]:
        if not self.service.ai_model_manager.models:
            return self.service._response(False, "Please train or load the models first")

        summary = self.service.ai_model_manager.get_models_summary()
        feature_importance = {}
        for model_name in self.service.ai_model_manager.models:
            importance = self.service.ai_model_manager.get_feature_importance(model_name)
            if importance:
                feature_importance[model_name] = list(importance.items())[:10]

        self.service.latest_model_analysis = {
            "summary": summary.to_dict(orient="records") if not summary.empty else [],
            "feature_importance": feature_importance,
        }
        return self.service._response(True, "Model analysis loaded", data=self.service.latest_model_analysis)

    def _build_default_experiment_request(
        self,
        *,
        experiment_name: str,
        target_column: str,
        feature_columns: list[str],
        total_rows: int,
    ) -> ResearchExperimentRequest:
        train_size, validation_size, test_size = self._compute_split_sizes(total_rows)
        common_defaults = self.research_defaults.common

        return ResearchExperimentRequest(
            experiment_name=experiment_name,
            target_column=target_column,
            feature_columns=list(feature_columns),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            step_size=max(test_size, 1),
            baseline_names=list(common_defaults.baseline_names),
            threshold_list=list(common_defaults.threshold_list),
            expanding_window=bool(common_defaults.expanding_window),
        )

    def _resolve_stage12_feature_set(self):
        if self.service.feature_data is None or self.service.feature_data.empty:
            raise ValueError("Prepared feature data is required before resolving Stage 1/2 feature sets")

        feature_set_name = self.research_defaults.stage12.fixed_feature_set_name
        feature_set = resolve_feature_sets(self.service.feature_data.columns, [feature_set_name])[0]
        feature_columns = list(feature_set.columns)
        if not feature_columns:
            raise ValueError(
                f"Resolved Stage 1/2 {feature_set_name} feature set is empty for the current prepared matrix"
            )

        target_column = self.service.get_target_column()
        leakage_columns = [
            column for column in feature_columns
            if str(column) == target_column or str(column).startswith("Future_")
        ]
        if leakage_columns:
            raise ValueError(f"Resolved Stage 1/2 feature set includes target/future leakage columns: {leakage_columns}")
        return feature_set

    def _build_default_target_study_request(
        self,
        *,
        study_name: str,
        feature_columns: list[str],
        target_specs: list[object],
        total_rows: int,
    ) -> TargetStudyRequest:
        experiment_request = self._build_default_experiment_request(
            experiment_name=study_name,
            target_column="stage2_target_study",
            feature_columns=feature_columns,
            total_rows=total_rows,
        )
        return TargetStudyRequest(
            study_name=study_name,
            feature_columns=list(feature_columns),
            target_specs=[serialize_target_spec(spec) for spec in target_specs],
            trainer_name=experiment_request.trainer_name,
            baseline_names=list(experiment_request.baseline_names),
            train_size=experiment_request.train_size,
            validation_size=experiment_request.validation_size,
            test_size=experiment_request.test_size,
            step_size=experiment_request.step_size,
            threshold_list=list(experiment_request.threshold_list),
            expanding_window=experiment_request.expanding_window,
        )

    def _execute_fixed_feature_research_diagnostic(
        self,
        *,
        run_name: str,
        artifact_prefix: str,
        feature_columns: list[str],
        target_series: pd.Series,
        target_spec: object,
        feature_set: Any,
        feature_selection_note: str,
    ) -> Dict[str, Any]:
        experiment_request = self._build_default_experiment_request(
            experiment_name=run_name,
            target_column=str(getattr(target_spec, "spec_id", run_name) or run_name),
            feature_columns=list(feature_columns),
            total_rows=len(self.service.feature_data),
        )
        return self._execute_research_experiment(
            request=experiment_request,
            target_series=target_series,
            artifact_prefix=artifact_prefix,
            target_spec=target_spec,
            experiment_metadata={
                "research_feature_set_name": feature_set.name,
                "research_feature_set_display_name": feature_set.display_name,
                "feature_selection_mode": "fixed_feature_columns",
                "feature_selection_note": feature_selection_note,
            },
            summary_updates={
                "feature_set_name": feature_set.name,
                "feature_set_display_name": feature_set.display_name,
            },
        )

    def _build_default_target_specs(self) -> list[object]:
        runtime_target_column = self.service.get_target_column()
        try:
            legacy_horizon = int(str(runtime_target_column).rsplit("_", 1)[-1])
        except ValueError:
            legacy_horizon = 1

        return [
            LegacyRuntimeDirectionSpec(
                spec_id=f"legacy_{runtime_target_column.lower()}",
                display_name=f"Legacy Runtime Target ({runtime_target_column})",
                horizon_bars=legacy_horizon,
            ),
            FixedHorizonDirectionSpec(
                spec_id="direction_h3",
                display_name="Fixed Direction (3 bars)",
                horizon_bars=3,
            ),
            ReturnThresholdLabelSpec(
                spec_id="return_threshold_h3_0_05pct",
                display_name="Return Threshold (3 bars, 0.05%)",
                horizon_bars=3,
                return_threshold_pct=0.05,
            ),
            VolatilityAdjustedMoveSpec(
                spec_id="vol_adjusted_h3_x1_0",
                display_name="Volatility Adjusted (3 bars, 1.0x vol)",
                horizon_bars=3,
                volatility_window=20,
                volatility_multiplier=1.0,
            ),
            NeutralBandLabelSpec(
                spec_id="neutral_band_h3_0_05pct",
                display_name="Neutral Band (3 bars, 0.05%)",
                horizon_bars=3,
                neutral_band_pct=0.05,
            ),
        ]

    def _build_default_feature_study_target_specs(self) -> list[object]:
        default_target_specs = self._build_default_target_specs()
        available_target_specs = {
            str(getattr(spec, "spec_id", "") or "").strip(): spec
            for spec in default_target_specs
        }
        unknown_target_ids = [
            target_id
            for target_id in self.research_defaults.stage3.target_ids
            if target_id not in available_target_specs
        ]
        if unknown_target_ids:
            raise ValueError(
                "Unsupported research.defaults.stage3.target_ids values: "
                + ", ".join(sorted(unknown_target_ids))
            )
        return [
            available_target_specs[target_id]
            for target_id in self.research_defaults.stage3.target_ids
        ]

    def _build_default_feature_study_request(
        self,
        *,
        study_name: str,
        target_specs: list[object],
        total_rows: int,
    ) -> FeatureStudyRequest:
        experiment_request = self._build_default_experiment_request(
            experiment_name=study_name,
            target_column="stage3_feature_study",
            feature_columns=[],
            total_rows=total_rows,
        )
        common_defaults = self.research_defaults.common
        stage3_defaults = self.research_defaults.stage3
        return FeatureStudyRequest(
            study_name=study_name,
            target_specs=[serialize_target_spec(spec) for spec in target_specs],
            feature_set_names=list(stage3_defaults.feature_set_names),
            selector_name=common_defaults.selector_name,
            selector_max_features=common_defaults.selector_max_features,
            trainer_name=experiment_request.trainer_name,
            baseline_names=list(experiment_request.baseline_names),
            train_size=experiment_request.train_size,
            validation_size=experiment_request.validation_size,
            test_size=experiment_request.test_size,
            step_size=experiment_request.step_size,
            threshold_list=list(experiment_request.threshold_list),
            expanding_window=experiment_request.expanding_window,
            compare_legacy_target=bool(stage3_defaults.compare_legacy_target),
        )

    def _build_stage4_working_target_spec(self) -> object:
        available_target_specs = {
            str(getattr(spec, "spec_id", "") or "").strip(): spec
            for spec in self._build_default_target_specs()
        }
        target_id = self.research_defaults.stage4.working_target_id
        if target_id not in available_target_specs:
            raise ValueError(f"Unsupported research.defaults.stage4.working_target_id value: {target_id}")
        return available_target_specs[target_id]

    def _build_default_training_experiment_request(
        self,
        *,
        experiment_name: str,
        target_spec: object,
        total_rows: int,
        timestamp: str,
    ) -> TrainingExperimentRequest:
        experiment_request = self._build_default_experiment_request(
            experiment_name=experiment_name,
            target_column="stage4_training_experiment",
            feature_columns=[],
            total_rows=total_rows,
        )
        common_defaults = self.research_defaults.common
        stage4_defaults = self.research_defaults.stage4
        return TrainingExperimentRequest(
            experiment_id=f"training_experiment_{timestamp}",
            experiment_name=experiment_name,
            target_spec=serialize_target_spec(target_spec),
            feature_set_name=stage4_defaults.feature_set_name,
            comparison_feature_set_name=stage4_defaults.comparison_feature_set_name,
            selector_name=common_defaults.selector_name,
            selector_max_features=common_defaults.selector_max_features,
            trainer_name=experiment_request.trainer_name,
            trainer_params={},
            baseline_names=list(experiment_request.baseline_names),
            train_size=experiment_request.train_size,
            validation_size=experiment_request.validation_size,
            test_size=experiment_request.test_size,
            step_size=experiment_request.step_size,
            threshold_list=list(experiment_request.threshold_list),
            expanding_window=experiment_request.expanding_window,
        )

    def _build_default_search_request(
        self,
        *,
        search_name: str,
        target_specs: list[object],
        total_rows: int,
        available_feature_columns: list[str],
        timestamp: str,
        max_workers: int | None = None,
    ) -> SearchRequest:
        experiment_request = self._build_default_experiment_request(
            experiment_name=search_name,
            target_column="stage5_search",
            feature_columns=[],
            total_rows=total_rows,
        )
        serialized_target_specs = [serialize_target_spec(target_spec) for target_spec in target_specs]
        serialized_target_specs = self._resolve_stage5_target_specs(serialized_target_specs)
        feature_set_names = self._resolve_stage5_feature_set_names(available_feature_columns)
        preset_names = self._resolve_stage5_preset_names()
        common_defaults = self.research_defaults.common
        return SearchRequest(
            search_id=f"search_run_{timestamp}",
            search_name=search_name,
            target_spec=dict(serialized_target_specs[0]),
            target_specs=serialized_target_specs,
            feature_set_names=feature_set_names,
            trainer_name=experiment_request.trainer_name,
            preset_names=preset_names,
            baseline_names=list(experiment_request.baseline_names),
            selector_name=common_defaults.selector_name,
            selector_max_features=common_defaults.selector_max_features,
            train_size=experiment_request.train_size,
            validation_size=experiment_request.validation_size,
            test_size=experiment_request.test_size,
            step_size=experiment_request.step_size,
            threshold_list=list(experiment_request.threshold_list),
            expanding_window=experiment_request.expanding_window,
            max_workers=max_workers,
            execution_mode="parallel_candidate_threads",
        )

    def _resolve_stage5_target_specs(self, serialized_target_specs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return self._resolve_target_specs_by_ids(
            serialized_target_specs,
            self.research_defaults.stage5.target_ids,
            config_key="research.defaults.stage5.target_ids",
        )

    def _resolve_stage5_feature_set_names(self, available_feature_columns: list[str]) -> list[str]:
        return self._resolve_feature_set_names(
            available_feature_columns,
            self.research_defaults.stage5.feature_set_names,
            config_key="research.defaults.stage5.feature_set_names",
        )

    def _resolve_stage5_preset_names(self) -> list[str]:
        configured_preset_names = list(self.research_defaults.stage5.preset_names)
        configured_models = list((self.service.config.get("ai_model", {}) or {}).get("models") or [])
        resolve_stage5_preset_definitions(configured_models, configured_preset_names)
        return configured_preset_names

    def _resolve_search_worker_count(self, requested_max_workers: int | None) -> int:
        requested = int(requested_max_workers or 0)
        if requested > 0:
            return max(1, requested)
        cpu_count = os.cpu_count() or 2
        stage5_defaults = self.research_defaults.stage5
        return min(stage5_defaults.max_worker_cap, max(stage5_defaults.min_auto_workers, cpu_count // 2))

    def _execute_search_candidate(
        self,
        *,
        request: SearchRequest,
        candidate_config: Any,
        timestamp: str,
        search_name: str,
        search_slug: str,
        target_specs_by_id: Dict[str, object],
        target_series_by_id: Dict[str, pd.Series],
        feature_sets: Dict[str, Any],
        trainer_registry: TrainerRegistry,
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        experiment_name = (
            f"{search_name}_{candidate_config.target_spec_id}_"
            f"{candidate_config.feature_set_name}_{candidate_config.preset_name}"
        )
        try:
            target_spec = target_specs_by_id[candidate_config.target_spec_id]
            target_series = target_series_by_id[candidate_config.target_spec_id]
            feature_set = feature_sets[candidate_config.feature_set_name]
            trainer = trainer_registry.build(
                candidate_config.trainer_name,
                model_params=candidate_config.trainer_params,
            )
            experiment_request = TrainingExperimentRequest(
                experiment_id=candidate_config.candidate_id,
                experiment_name=experiment_name,
                target_spec=dict(candidate_config.target_spec),
                feature_set_name=candidate_config.feature_set_name,
                comparison_feature_set_name="",
                selector_name=request.selector_name,
                selector_max_features=request.selector_max_features,
                trainer_name=request.trainer_name,
                trainer_params=dict(candidate_config.trainer_params),
                baseline_names=list(request.baseline_names),
                train_size=request.train_size,
                validation_size=request.validation_size,
                test_size=request.test_size,
                step_size=request.step_size,
                threshold_list=list(request.threshold_list),
                expanding_window=request.expanding_window,
            )
            artifact_prefix = (
                f"training_experiment_{timestamp}_{search_slug}_"
                f"{candidate_config.target_spec_id}_{candidate_config.feature_set_name}_{candidate_config.preset_name}"
            )
            candidate_execution = self._run_training_experiment_candidate(
                request=experiment_request,
                target_series=target_series,
                target_spec=target_spec,
                feature_set=feature_set,
                trainer=trainer,
                artifact_prefix=artifact_prefix,
            )
            finalized = self._finalize_training_experiment_workflow(
                request=experiment_request,
                target_series=target_series,
                target_spec=target_spec,
                candidate_execution=candidate_execution,
                comparison_runs=[],
                artifact_prefix=artifact_prefix,
                metadata={
                    "out_of_sample_note": "Walk-forward metrics come from out-of-sample folds; the candidate artifact was retrained afterward on the full eligible dataset.",
                    "selected_threshold_summary": candidate_execution["selected_threshold_summary"],
                    "search_id": request.search_id,
                    "preset_name": candidate_config.preset_name,
                    "execution_mode": request.execution_mode,
                },
            )
            training_result = finalized["training_result"]
            report_file = finalized["report_file"]
            candidate_summary = self._build_search_candidate_summary(
                training_result=training_result,
                report_file=report_file,
                preset_name=candidate_config.preset_name,
            )
            candidate_summary.elapsed_seconds = round(perf_counter() - started_at, 4)
            return {
                "candidate_summary": candidate_summary,
                "candidate_row": self._build_search_candidate_row(candidate_summary),
            }
        except Exception as exc:
            logger.error(f"Stage 5 candidate failed for {candidate_config.candidate_id}: {exc}")
            return self._build_failed_search_candidate_result(
                candidate_config=candidate_config,
                experiment_name=experiment_name,
                error_message=str(exc),
                elapsed_seconds=round(perf_counter() - started_at, 4),
            )

    def _build_failed_search_candidate_result(
        self,
        *,
        candidate_config: Any,
        experiment_name: str,
        error_message: str,
        elapsed_seconds: float | None = None,
    ) -> Dict[str, Any]:
        candidate_summary = SearchCandidateSummary(
            candidate_id=str(candidate_config.candidate_id),
            experiment_id=str(candidate_config.candidate_id),
            experiment_name=experiment_name,
            trainer_name=str(candidate_config.trainer_name),
            feature_set_name=str(candidate_config.feature_set_name),
            preset_name=str(candidate_config.preset_name),
            target_spec_id=str(candidate_config.target_spec_id),
            target_display_name=str(candidate_config.target_display_name),
            trainer_params=dict(candidate_config.trainer_params or {}),
            execution_status="failed",
            error_message=str(error_message),
            elapsed_seconds=elapsed_seconds,
            diagnostics={
                "warnings": [
                    {
                        "code": "candidate_execution_failed",
                        "severity": "critical",
                        "message": f"Candidate execution failed: {error_message}",
                    }
                ]
            },
        )
        return {
            "candidate_summary": candidate_summary,
            "candidate_row": self._build_search_candidate_row(candidate_summary),
        }

    def _build_search_candidate_row(self, candidate_summary: SearchCandidateSummary) -> Dict[str, Any]:
        integrity = dict((candidate_summary.diagnostics or {}).get("integrity") or {})
        return {
            "candidate_id": candidate_summary.candidate_id,
            "experiment_id": candidate_summary.experiment_id,
            "experiment_name": candidate_summary.experiment_name,
            "target_spec_id": candidate_summary.target_spec_id,
            "target_display_name": candidate_summary.target_display_name,
            "feature_set_name": candidate_summary.feature_set_name,
            "preset_name": candidate_summary.preset_name,
            "selected_threshold": candidate_summary.selected_threshold,
            "validation_mean_f1": (candidate_summary.validation_summary or {}).get("mean_f1"),
            "validation_mean_coverage": (candidate_summary.validation_summary or {}).get("mean_coverage"),
            "test_mean_f1": (candidate_summary.test_summary or {}).get("mean_f1"),
            "test_mean_coverage": (candidate_summary.test_summary or {}).get("mean_coverage"),
            "overall_mean_test_accuracy": candidate_summary.overall_mean_test_accuracy,
            "majority_baseline_mean_test_accuracy": candidate_summary.majority_baseline_mean_test_accuracy,
            "passed_test_guardrail": candidate_summary.passed_test_guardrail,
            "passed_truth_gate": candidate_summary.passed_truth_gate,
            "truth_gate_failures": list(candidate_summary.truth_gate_failures),
            "proof_status": integrity.get("proof_status"),
            "integrity_contract_ok": bool(integrity.get("integrity_contract_ok")),
            "diagnostics": dict(candidate_summary.diagnostics),
            "report_file": candidate_summary.report_file,
            "candidate_artifact_path": candidate_summary.candidate_artifact_path,
            "execution_status": candidate_summary.execution_status,
            "error_message": candidate_summary.error_message,
            "elapsed_seconds": candidate_summary.elapsed_seconds,
        }

    def _build_stage5_search_target_specs(self) -> list[object]:
        runtime_target_column = self.service.get_target_column()
        try:
            legacy_horizon = int(str(runtime_target_column).rsplit("_", 1)[-1])
        except ValueError:
            legacy_horizon = 1

        return [
            self._build_stage4_working_target_spec(),
            LegacyRuntimeDirectionSpec(
                spec_id=f"legacy_{runtime_target_column.lower()}",
                display_name=f"Legacy Runtime Target ({runtime_target_column})",
                horizon_bars=legacy_horizon,
            ),
            VolatilityAdjustedMoveSpec(
                spec_id="vol_adjusted_h3_x1_0",
                display_name="Volatility Adjusted (3 bars, 1.0x vol)",
                horizon_bars=3,
                volatility_window=20,
                volatility_multiplier=1.0,
            ),
        ]

    def _build_trainer_registry(self) -> TrainerRegistry:
        registry = TrainerRegistry()
        registry.register(
            "current_ensemble",
            CurrentEnsembleTrainer(
                self.service.config,
                target_column=self.service.get_target_column(),
            ),
        )
        return registry

    def _run_research_experiment_core(
        self,
        *,
        request: ResearchExperimentRequest,
        target_series: pd.Series,
        feature_selector: Any = None,
        trainer: Any = None,
        target_spec: object | None = None,
    ) -> Dict[str, Any]:
        splitter = WalkForwardSplitter(
            train_size=request.train_size,
            validation_size=request.validation_size,
            test_size=request.test_size,
            step_size=request.step_size,
            expanding_window=request.expanding_window,
        )
        splits = splitter.split(len(self.service.feature_data))
        if not splits:
            raise ValueError("Prepared data does not contain enough rows for walk-forward evaluation")

        resolved_trainer = trainer or CurrentEnsembleTrainer(self.service.config, target_column=request.target_column)
        runner = ExperimentRunner(
            training_pipeline=TrainingPipeline(resolved_trainer.fit_predict),
            evaluation_pipeline=EvaluationPipeline(request.normalized_thresholds()),
        )
        result = runner.run(
            request=request,
            feature_frame=self.service.feature_data,
            target=target_series,
            splits=splits,
            feature_selector=feature_selector,
            target_spec=target_spec,
        )

        artifact_metadata = dict((result.prediction_artifacts[0].metadata if result.prediction_artifacts else {}) or {})
        prediction_rows = self._normalize_records_for_json(list(artifact_metadata.get("prediction_rows", [])))
        threshold_rows = self._normalize_records_for_json(list(artifact_metadata.get("threshold_rows", [])))
        calibration_rows = self._normalize_records_for_json(list(artifact_metadata.get("calibration_rows", [])))
        if result.prediction_artifacts:
            result.prediction_artifacts[0].metadata = {
                **{
                    key: value
                    for key, value in artifact_metadata.items()
                    if key not in {"prediction_rows", "threshold_rows", "calibration_rows"}
                },
                "prediction_rows": prediction_rows,
                "threshold_rows": threshold_rows,
                "calibration_rows": calibration_rows,
            }
        return {
            "result": result,
            "splits": splits,
            "prediction_rows": prediction_rows,
            "threshold_rows": threshold_rows,
            "calibration_rows": calibration_rows,
        }

    def _execute_research_experiment(
        self,
        *,
        request: ResearchExperimentRequest,
        target_series: pd.Series,
        artifact_prefix: str,
        feature_selector: Any = None,
        trainer: Any = None,
        target_spec: object | None = None,
        experiment_metadata: Dict[str, Any] | None = None,
        summary_updates: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        execution = self._run_research_experiment_core(
            request=request,
            target_series=target_series,
            feature_selector=feature_selector,
            trainer=trainer,
            target_spec=target_spec,
        )
        result = execution["result"]
        splits = execution["splits"]
        prediction_rows = execution["prediction_rows"]
        threshold_rows = execution["threshold_rows"]
        calibration_rows = execution["calibration_rows"]

        prediction_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_predictions.csv")
        threshold_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_thresholds.csv")
        calibration_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_calibration.csv")
        report_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}.json")

        pd.DataFrame(prediction_rows).to_csv(prediction_file, index=False, encoding="utf-8-sig")
        pd.DataFrame(threshold_rows).to_csv(threshold_file, index=False, encoding="utf-8-sig")
        pd.DataFrame(calibration_rows).to_csv(calibration_file, index=False, encoding="utf-8-sig")

        result.metadata = {
            **dict(result.metadata or {}),
            **dict(experiment_metadata or {}),
            "resolved_research_defaults": self._resolved_research_defaults_snapshot(),
        }

        integrity_bundle = self._build_experiment_integrity_bundle(
            artifact_prefix=artifact_prefix,
            experiment_result=result,
            expected_feature_set_name=(experiment_metadata or {}).get("research_feature_set_name"),
            expected_feature_selection_mode=(experiment_metadata or {}).get("feature_selection_mode"),
            expected_target_spec_id=str(getattr(target_spec, "spec_id", request.target_column) or request.target_column),
        )
        result.integrity = integrity_bundle["integrity"]
        result.integrity_artifact_paths = integrity_bundle["artifact_paths"]

        if result.prediction_artifacts:
            result.prediction_artifacts[0].predictions_file = str(prediction_file)
            result.prediction_artifacts[0].threshold_metrics_file = str(threshold_file)
            result.prediction_artifacts[0].calibration_file = str(calibration_file)

        self.service.experiment_store.save_result(result, report_file)

        summary = {
            "experiment_name": result.experiment_name,
            "target_column": result.target_column,
            "trainer_name": request.trainer_name,
            "feature_count": len(result.feature_columns),
            "fold_count": len(splits),
            "mean_test_accuracy": result.aggregate_metrics.get("mean_test_accuracy"),
            "baseline_comparison": result.baseline_comparison,
        }
        if summary_updates:
            summary.update(dict(summary_updates))
        artifacts = {
            "report_file": str(report_file),
            "prediction_rows_file": str(prediction_file),
            "threshold_metrics_file": str(threshold_file),
            "calibration_file": str(calibration_file),
            **dict(result.integrity_artifact_paths or {}),
        }
        return {
            "result": result,
            "summary": summary,
            "artifacts": artifacts,
            "folds": [fold.__dict__ for fold in result.folds],
            "integrity": result.integrity,
        }

    def _run_training_experiment_candidate(
        self,
        *,
        request: TrainingExperimentRequest,
        target_series: pd.Series,
        target_spec: object,
        feature_set: Any,
        trainer: Any,
        artifact_prefix: str,
    ) -> Dict[str, Any]:
        selector = build_feature_selector(
            request.selector_name,
            max_features=min(int(request.selector_max_features), max(len(feature_set.columns), 1)),
        )
        experiment_request = ResearchExperimentRequest(
            experiment_name=request.experiment_name,
            target_column=str(request.target_spec.get("spec_id")),
            feature_columns=list(feature_set.columns),
            trainer_name=request.trainer_name,
            baseline_names=list(request.baseline_names),
            train_size=request.train_size,
            validation_size=request.validation_size,
            test_size=request.test_size,
            step_size=request.step_size,
            threshold_list=list(request.threshold_list),
            expanding_window=request.expanding_window,
        )
        execution = self._run_research_experiment_core(
            request=experiment_request,
            target_series=target_series,
            feature_selector=selector,
            trainer=trainer,
            target_spec=target_spec,
        )
        result = execution["result"]
        threshold_rows = execution["threshold_rows"]
        selected_threshold = select_best_threshold(threshold_rows, model_name=request.trainer_name)
        if selected_threshold is None:
            selected_threshold = float(request.threshold_list[0])
        selected_threshold_summary = summarize_selected_threshold_metrics(
            threshold_rows,
            selected_threshold=selected_threshold,
            model_name=request.trainer_name,
            baseline_names=request.baseline_names,
        )

        prediction_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_predictions.csv")
        threshold_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_thresholds.csv")
        calibration_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_calibration.csv")
        resolved_features_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_resolved_features.csv")
        pd.DataFrame(execution["prediction_rows"]).to_csv(prediction_file, index=False, encoding="utf-8-sig")
        pd.DataFrame(threshold_rows).to_csv(threshold_file, index=False, encoding="utf-8-sig")
        pd.DataFrame(execution["calibration_rows"]).to_csv(calibration_file, index=False, encoding="utf-8-sig")

        train_mask = target_series.notna()
        full_selection = selector.select(
            self.service.feature_data.loc[train_mask, feature_set.columns],
            target_series.loc[train_mask],
        )
        selected_feature_columns = list(full_selection.selected_columns) or list(feature_set.columns)

        train_features = self.service.feature_data.loc[train_mask, selected_feature_columns]
        train_target = target_series.loc[train_mask]

        candidate_directory = Path(self.service.config.get("research", {}).get("candidate_models_directory", "models/candidates"))
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_model_path = candidate_directory / f"{artifact_prefix}.joblib"
        candidate_artifact = trainer.fit_candidate_artifact(
            train_features,
            train_target,
            str(candidate_model_path),
            metadata={
                "experiment_id": request.experiment_id,
                "target_spec": serialize_target_spec(target_spec),
                "feature_set_name": feature_set.name,
                "selected_threshold": selected_threshold,
                "trainer_params": dict(request.trainer_params),
            },
        )

        pd.DataFrame(
            [
                {
                    "selected_feature": column,
                    "rank": ranking_row.get("rank"),
                    "score": ranking_row.get("score"),
                }
                for column in selected_feature_columns
                for ranking_row in [
                    next(
                        (row for row in full_selection.ranking_rows if row.get("column") == column),
                        {},
                    )
                ]
            ]
        ).to_csv(
            resolved_features_file,
            index=False,
            encoding="utf-8-sig",
        )

        if result.prediction_artifacts:
            result.prediction_artifacts[0].predictions_file = str(prediction_file)
            result.prediction_artifacts[0].threshold_metrics_file = str(threshold_file)
            result.prediction_artifacts[0].calibration_file = str(calibration_file)

        integrity_bundle = self._build_experiment_integrity_bundle(
            artifact_prefix=artifact_prefix,
            experiment_result=result,
            expected_feature_set_name=feature_set.name,
            expected_feature_selection_mode="fold_local_selector" if request.selector_name else "fixed_feature_columns",
            expected_target_spec_id=str((request.target_spec or {}).get("spec_id") or ""),
        )
        result.integrity = integrity_bundle["integrity"]
        result.integrity_artifact_paths = integrity_bundle["artifact_paths"]

        return {
            "result": result,
            "splits": execution["splits"],
            "selected_threshold": float(selected_threshold),
            "selected_threshold_summary": selected_threshold_summary,
            "selected_feature_columns": selected_feature_columns,
            "candidate_artifact": candidate_artifact,
            "artifacts": {
                "prediction_rows_file": str(prediction_file),
                "threshold_metrics_file": str(threshold_file),
                "calibration_file": str(calibration_file),
                "resolved_features_file": str(resolved_features_file),
                **dict(result.integrity_artifact_paths or {}),
            },
        }

    def _run_training_experiment_evaluation_only(
        self,
        *,
        request: TrainingExperimentRequest,
        target_series: pd.Series,
        target_spec: object,
        feature_set: Any,
        trainer: Any,
    ) -> Dict[str, Any]:
        selector = build_feature_selector(
            request.selector_name,
            max_features=min(int(request.selector_max_features), max(len(feature_set.columns), 1)),
        )
        experiment_request = ResearchExperimentRequest(
            experiment_name=f"{request.experiment_name}_{feature_set.name}",
            target_column=str(request.target_spec.get("spec_id")),
            feature_columns=list(feature_set.columns),
            trainer_name=request.trainer_name,
            baseline_names=list(request.baseline_names),
            train_size=request.train_size,
            validation_size=request.validation_size,
            test_size=request.test_size,
            step_size=request.step_size,
            threshold_list=list(request.threshold_list),
            expanding_window=request.expanding_window,
        )
        execution = self._run_research_experiment_core(
            request=experiment_request,
            target_series=target_series,
            feature_selector=selector,
            trainer=trainer,
            target_spec=target_spec,
        )
        result = execution["result"]
        threshold_rows = build_threshold_summary_rows(result.threshold_summary)
        return {
            "feature_set_name": feature_set.name,
            "display_name": getattr(feature_set, "display_name", feature_set.name),
            "aggregate_metrics": result.aggregate_metrics,
            "baseline_comparison": result.baseline_comparison,
            "selected_threshold": select_best_threshold(threshold_rows, model_name=request.trainer_name),
        }

    def _build_training_experiment_result(
        self,
        *,
        request: TrainingExperimentRequest,
        target_series: pd.Series,
        target_spec: object,
        candidate_execution: Dict[str, Any],
        diagnostics_bundle: Dict[str, Any],
        comparison_runs: list[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> TrainingExperimentResult:
        return TrainingExperimentResult(
            experiment_id=request.experiment_id,
            experiment_name=request.experiment_name,
            dataset_metadata=self._build_dataset_metadata(target_series),
            target_spec=serialize_target_spec(target_spec),
            feature_set_name=request.feature_set_name,
            comparison_feature_set_name=request.comparison_feature_set_name,
            resolved_feature_columns=list(candidate_execution["selected_feature_columns"]),
            selector_name=request.selector_name,
            selector_settings={"max_features": request.selector_max_features},
            trainer_name=request.trainer_name,
            trainer_params=dict(request.trainer_params),
            split_settings={
                "train_size": request.train_size,
                "validation_size": request.validation_size,
                "test_size": request.test_size,
                "step_size": request.step_size,
                "expanding_window": request.expanding_window,
            },
            threshold_list=list(request.threshold_list),
            selected_threshold=candidate_execution["selected_threshold"],
            aggregate_metrics=candidate_execution["result"].aggregate_metrics,
            baseline_comparison=candidate_execution["result"].baseline_comparison,
            folds=list(candidate_execution["result"].folds),
            prediction_artifacts=list(candidate_execution["result"].prediction_artifacts),
            candidate_artifact=candidate_execution["candidate_artifact"],
            comparison_runs=comparison_runs,
            promotion_status="not_promoted",
            diagnostics=diagnostics_bundle["diagnostics"],
            diagnostics_artifact_paths=diagnostics_bundle["artifact_paths"],
            integrity=candidate_execution["result"].integrity,
            integrity_artifact_paths=candidate_execution["result"].integrity_artifact_paths,
            metadata={
                **dict(metadata or {}),
                "resolved_research_defaults": self._resolved_research_defaults_snapshot(),
            },
        )

    def _finalize_training_experiment_workflow(
        self,
        *,
        request: TrainingExperimentRequest,
        target_series: pd.Series,
        target_spec: object,
        candidate_execution: Dict[str, Any],
        comparison_runs: list[Dict[str, Any]],
        artifact_prefix: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnostics_bundle = self._build_training_diagnostics_bundle(
            artifact_prefix=artifact_prefix,
            target_series=target_series,
            experiment_result=candidate_execution["result"],
            splits=list(candidate_execution["splits"]),
            trainer_name=request.trainer_name,
            selected_threshold=candidate_execution["selected_threshold"],
            selected_threshold_summary=candidate_execution["selected_threshold_summary"],
        )
        training_result = self._build_training_experiment_result(
            request=request,
            target_series=target_series,
            target_spec=target_spec,
            candidate_execution=candidate_execution,
            diagnostics_bundle=diagnostics_bundle,
            comparison_runs=comparison_runs,
            metadata=metadata,
        )
        report_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}.json")
        self.service.experiment_store.save_result(training_result, report_file)
        artifacts = {
            "report_file": str(report_file),
            "prediction_rows_file": candidate_execution["artifacts"]["prediction_rows_file"],
            "threshold_metrics_file": candidate_execution["artifacts"]["threshold_metrics_file"],
            "calibration_file": candidate_execution["artifacts"]["calibration_file"],
            "resolved_features_file": candidate_execution["artifacts"]["resolved_features_file"],
            "model_path": candidate_execution["candidate_artifact"].artifact_path,
            **training_result.diagnostics_artifact_paths,
            **training_result.integrity_artifact_paths,
        }
        return {
            "training_result": training_result,
            "summary": self._build_training_experiment_summary(training_result),
            "artifacts": artifacts,
            "report_file": report_file,
        }

    def _build_dataset_metadata(self, target_series: pd.Series) -> Dict[str, Any]:
        index = self.service.feature_data.index if self.service.feature_data is not None else pd.Index([])
        return {
            "source_path": self.service.last_import_summary.get("path"),
            "row_count": len(self.service.feature_data) if self.service.feature_data is not None else 0,
            "timestamp_start": index.min().isoformat() if len(index) else None,
            "timestamp_end": index.max().isoformat() if len(index) else None,
            "target_non_null_rows": int(pd.to_numeric(target_series, errors="coerce").notna().sum()),
        }

    def _build_training_diagnostics_bundle(
        self,
        *,
        artifact_prefix: str,
        target_series: pd.Series,
        experiment_result: Any,
        splits: list[Any],
        trainer_name: str,
        selected_threshold: float | None,
        selected_threshold_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        diagnostics = build_training_experiment_diagnostics(
            feature_frame=self.service.feature_data,
            target_series=target_series,
            splits=splits,
            experiment_result=experiment_result,
            trainer_name=trainer_name,
            selected_threshold=selected_threshold,
            selected_threshold_summary=selected_threshold_summary,
        )
        fold_diagnostics_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_fold_diagnostics.csv")
        threshold_coverage_file = self.service.experiment_store.resolve_path(
            f"{artifact_prefix}_threshold_coverage_diagnostics.csv"
        )
        feature_health_file = self.service.experiment_store.resolve_path(
            f"{artifact_prefix}_feature_health_diagnostics.csv"
        )

        pd.DataFrame((diagnostics.get("target_balance") or {}).get("fold_rows") or []).to_csv(
            fold_diagnostics_file,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(diagnostics.get("prediction_health_rows") or []).to_csv(
            threshold_coverage_file,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(diagnostics.get("feature_health_rows") or []).to_csv(
            feature_health_file,
            index=False,
            encoding="utf-8-sig",
        )

        return {
            "diagnostics": diagnostics,
            "artifact_paths": {
                "fold_diagnostics_file": str(fold_diagnostics_file),
                "threshold_coverage_diagnostics_file": str(threshold_coverage_file),
                "feature_health_diagnostics_file": str(feature_health_file),
            },
        }

    def _build_experiment_integrity_bundle(
        self,
        *,
        artifact_prefix: str,
        experiment_result: Any,
        expected_feature_set_name: str | None = None,
        expected_feature_selection_mode: str | None = None,
        expected_target_spec_id: str | None = None,
    ) -> Dict[str, Any]:
        integrity = build_experiment_integrity_proof(
            experiment_result=experiment_result,
            expected_feature_set_name=expected_feature_set_name,
            expected_feature_selection_mode=expected_feature_selection_mode,
            expected_target_spec_id=expected_target_spec_id,
        )
        fold_integrity_file = self.service.experiment_store.resolve_path(f"{artifact_prefix}_integrity.csv")
        pd.DataFrame(integrity.get("fold_rows") or []).to_csv(
            fold_integrity_file,
            index=False,
            encoding="utf-8-sig",
        )
        return {
            "integrity": integrity,
            "artifact_paths": {
                "fold_integrity_file": str(fold_integrity_file),
            },
        }

    def _aggregate_integrity_payloads(
        self,
        rows: list[Dict[str, Any]],
        *,
        scope_label: str,
    ) -> Dict[str, Any]:
        normalized_rows = [dict(row or {}) for row in rows]
        passed_count = sum(1 for row in normalized_rows if bool(row.get("integrity_contract_ok")))
        missing_count = sum(1 for row in normalized_rows if str(row.get("proof_status") or "") == "missing")
        failed_count = sum(
            1
            for row in normalized_rows
            if str(row.get("proof_status") or "") != "missing" and not bool(row.get("integrity_contract_ok"))
        )
        invalid_fold_count = sum(int(row.get("invalid_fold_count") or 0) for row in normalized_rows)
        total_purged_train_rows = sum(int(row.get("total_purged_train_rows") or 0) for row in normalized_rows)
        total_purged_validation_rows = sum(int(row.get("total_purged_validation_rows") or 0) for row in normalized_rows)
        warnings: list[Dict[str, Any]] = []
        if missing_count:
            warnings.append(
                {
                    "code": "missing_integrity_proof",
                    "severity": "critical",
                    "message": f"Some {scope_label} results do not have saved integrity proof.",
                }
            )
        if failed_count:
            warnings.append(
                {
                    "code": "integrity_contract_failures",
                    "severity": "critical",
                    "message": f"Some {scope_label} results failed the integrity contract.",
                }
            )
        proof_status = "passed"
        if missing_count and not failed_count:
            proof_status = "missing"
        if failed_count:
            proof_status = "failed"
        return {
            "overview": {
                f"{scope_label}_count": len(normalized_rows),
                f"passed_{scope_label}_count": passed_count,
                f"missing_{scope_label}_count": missing_count,
                f"failed_{scope_label}_count": failed_count,
                "invalid_fold_count": invalid_fold_count,
                "total_purged_train_rows": total_purged_train_rows,
                "total_purged_validation_rows": total_purged_validation_rows,
            },
            "warnings": warnings,
            "fold_rows": normalized_rows,
            "proof_status": proof_status if normalized_rows else "missing",
            "integrity_contract_ok": bool(normalized_rows) and missing_count == 0 and failed_count == 0,
        }

    def _build_training_experiment_summary(self, result: TrainingExperimentResult) -> Dict[str, Any]:
        selected_threshold_summary = (result.metadata or {}).get("selected_threshold_summary") or {}
        validation_summary = selected_threshold_summary.get("validation") or {}
        test_summary = selected_threshold_summary.get("test") or {}
        diagnostics_highlights = (result.diagnostics or {}).get("highlights") or {}
        integrity_overview = (result.integrity or {}).get("overview") or {}
        return {
            "experiment_id": result.experiment_id,
            "experiment_name": result.experiment_name,
            "target_spec_id": result.target_spec.get("spec_id"),
            "feature_set_name": result.feature_set_name,
            "comparison_feature_set_name": result.comparison_feature_set_name,
            "trainer_name": result.trainer_name,
            "selected_threshold": result.selected_threshold,
            "feature_count": len(result.resolved_feature_columns),
            "fold_count": len({fold.fold_name for fold in result.folds if getattr(fold, "model_name", "") == result.trainer_name}),
            "mean_test_accuracy": result.aggregate_metrics.get("mean_test_accuracy"),
            "majority_baseline_mean_test_accuracy": ((result.baseline_comparison or {}).get("baselines", {}).get("majority_class") or {}).get("mean_test_accuracy"),
            "selected_threshold_validation_mean_f1": validation_summary.get("mean_f1"),
            "selected_threshold_validation_mean_coverage": validation_summary.get("mean_coverage"),
            "selected_threshold_test_mean_f1": test_summary.get("mean_f1"),
            "selected_threshold_test_mean_coverage": test_summary.get("mean_coverage"),
            "candidate_model_path": (result.candidate_artifact.artifact_path if result.candidate_artifact else ""),
            "one_class_fold_count": diagnostics_highlights.get("one_class_fold_count"),
            "constant_feature_fold_count": diagnostics_highlights.get("constant_feature_fold_count"),
            "near_constant_feature_fold_count": diagnostics_highlights.get("near_constant_feature_fold_count"),
            "integrity_contract_ok": (result.integrity or {}).get("integrity_contract_ok"),
            "proof_status": (result.integrity or {}).get("proof_status"),
            "horizon_bars": integrity_overview.get("horizon_bars"),
            "total_purged_train_rows": integrity_overview.get("total_purged_train_rows"),
            "total_purged_validation_rows": integrity_overview.get("total_purged_validation_rows"),
            "invalid_fold_count": integrity_overview.get("invalid_fold_count"),
        }

    def _build_search_candidate_summary(
        self,
        *,
        training_result: TrainingExperimentResult,
        report_file: Path,
        preset_name: str,
    ) -> SearchCandidateSummary:
        selected_threshold_summary = (training_result.metadata or {}).get("selected_threshold_summary") or {}
        validation_summary = dict(selected_threshold_summary.get("validation") or {})
        test_summary = dict(selected_threshold_summary.get("test") or {})
        passed_test_guardrail = float(test_summary.get("mean_f1") or 0.0) > float(test_summary.get("best_baseline_mean_f1") or 0.0)
        majority_baseline_mean_test_accuracy = (
            ((training_result.baseline_comparison or {}).get("baselines", {}).get("majority_class") or {}).get("mean_test_accuracy")
        )
        runtime_feature_contract_ok = (
            list(training_result.resolved_feature_columns) ==
            list((training_result.candidate_artifact.selected_features if training_result.candidate_artifact else training_result.resolved_feature_columns) or [])
        )
        diagnostics = dict(training_result.diagnostics or {})
        integrity = resolve_integrity_payload(
            metadata=dict(training_result.metadata or {}),
            stored_integrity=dict(training_result.integrity or {}),
            expected_feature_set_name=str(training_result.feature_set_name or ""),
            expected_feature_selection_mode="fold_local_selector" if training_result.selector_name else "fixed_feature_columns",
            expected_target_spec_id=str((training_result.target_spec or {}).get("spec_id") or ""),
            fallback_fold_count=len({fold.fold_name for fold in training_result.folds if getattr(fold, "fold_name", "")}),
        )
        diagnostics["integrity"] = integrity
        return SearchCandidateSummary(
            candidate_id=training_result.experiment_id,
            experiment_id=training_result.experiment_id,
            experiment_name=training_result.experiment_name,
            trainer_name=training_result.trainer_name,
            target_spec_id=str((training_result.target_spec or {}).get("spec_id") or ""),
            target_display_name=str((training_result.target_spec or {}).get("display_name") or ""),
            feature_set_name=training_result.feature_set_name,
            preset_name=preset_name,
            trainer_params=dict(training_result.trainer_params),
            selected_threshold=training_result.selected_threshold,
            report_file=str(report_file),
            candidate_artifact_path=(training_result.candidate_artifact.artifact_path if training_result.candidate_artifact else ""),
            validation_summary=validation_summary,
            test_summary=test_summary,
            overall_mean_test_accuracy=(training_result.aggregate_metrics or {}).get("mean_test_accuracy"),
            majority_baseline_mean_test_accuracy=majority_baseline_mean_test_accuracy,
            expected_fold_count=len({fold.fold_name for fold in training_result.folds if getattr(fold, "model_name", "") == training_result.trainer_name}),
            runtime_feature_contract_ok=runtime_feature_contract_ok,
            rank_tuple=[
                float(validation_summary.get("beat_rate") or 0.0),
                float(validation_summary.get("f1_std") or 0.0),
                float(validation_summary.get("mean_f1") or 0.0),
                float(validation_summary.get("mean_coverage") or 0.0),
                float(training_result.selected_threshold or 0.0),
            ],
            passed_test_guardrail=passed_test_guardrail,
            diagnostics=diagnostics,
            execution_status="completed",
            error_message="",
        )

    def _build_search_summary(self, result: SearchResult) -> Dict[str, Any]:
        recommended = result.recommended_winner or {}
        gate_summary = (result.metadata or {}).get("gate_summary") or {}
        diagnostics_summary = (result.diagnostics or {}).get("summary") or {}
        integrity_overview = (result.integrity or {}).get("overview") or {}
        return {
            "search_id": result.search_id,
            "search_name": result.search_name,
            "target_spec_id": (result.target_spec or {}).get("spec_id"),
            "target_count": len(result.target_specs or ([result.target_spec] if result.target_spec else [])),
            "searched_target_ids": [spec.get("spec_id") for spec in (result.target_specs or ([result.target_spec] if result.target_spec else []))],
            "searched_target_display_names": [
                spec.get("display_name") or spec.get("spec_id")
                for spec in (result.target_specs or ([result.target_spec] if result.target_spec else []))
            ],
            "trainer_name": result.trainer_name,
            "feature_set_count": len(result.feature_set_names),
            "candidate_count": result.candidate_count,
            "successful_candidate_count": result.successful_candidate_count,
            "failed_candidate_count": result.failed_candidate_count,
            "preset_count": len(result.preset_definitions),
            "recommended_experiment_id": recommended.get("experiment_id"),
            "recommended_report_file": recommended.get("report_file"),
            "recommended_target_spec_id": recommended.get("target_spec_id"),
            "recommended_target_display_name": recommended.get("target_display_name"),
            "recommended_feature_set_name": recommended.get("feature_set_name"),
            "recommended_preset_name": recommended.get("preset_name"),
            "recommended_selected_threshold": recommended.get("selected_threshold"),
            "winner_status": recommended.get("status"),
            "winner_reason": recommended.get("reason"),
            "truth_gate_pass_count": gate_summary.get("passed_truth_gate_count"),
            "test_guardrail_pass_count": gate_summary.get("passed_test_guardrail_count"),
            "truth_gate_failures": gate_summary.get("failure_counts"),
            "low_coverage_candidate_count": diagnostics_summary.get("low_coverage_candidate_count"),
            "majority_dominance_candidate_count": diagnostics_summary.get("majority_dominance_candidate_count"),
            "one_class_candidate_count": diagnostics_summary.get("one_class_candidate_count"),
            "integrity_failure_candidate_count": diagnostics_summary.get("integrity_failure_candidate_count"),
            "integrity_contract_ok": (result.integrity or {}).get("integrity_contract_ok"),
            "proof_status": (result.integrity or {}).get("proof_status"),
            "total_purged_train_rows": integrity_overview.get("total_purged_train_rows"),
            "total_purged_validation_rows": integrity_overview.get("total_purged_validation_rows"),
            "invalid_fold_count": integrity_overview.get("invalid_fold_count"),
            "execution_mode": result.execution_mode,
            "resolved_max_workers": result.resolved_max_workers,
            "elapsed_seconds": (result.metadata or {}).get("elapsed_seconds"),
        }

    def _emit_search_progress(
        self,
        callback: Callable[[Dict[str, Any]], None] | None,
        *,
        started_at: float,
        phase: str,
        step_label: str,
        current: int,
        total: int,
        details: Dict[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return
        payload = {
            "phase": phase,
            "step_label": step_label,
            "current": int(current),
            "total": int(total),
            "progress_ratio": (float(current) / float(total)) if total else 0.0,
            "elapsed_seconds": round(perf_counter() - started_at, 2),
            "details": dict(details or {}),
        }
        callback(payload)

    def _resolve_training_experiment_report_path(self, experiment_path_or_id: str) -> str:
        candidate = Path(str(experiment_path_or_id))
        if candidate.exists():
            return str(candidate)

        for path in self.service.experiment_store.list_results(limit=100, prefix="training_experiment_"):
            payload = self.service.experiment_store.load_result(path)
            if payload.get("experiment_id") == experiment_path_or_id or payload.get("experiment_name") == experiment_path_or_id:
                return str(path)
        raise FileNotFoundError(f"Training experiment report not found: {experiment_path_or_id}")

    def _save_experiment_payload(self, payload: Dict[str, Any], report_path: str) -> None:
        target_path = self.service.experiment_store.resolve_path(report_path)
        with target_path.open("w", encoding="utf-8") as report_file:
            json.dump(payload, report_file, ensure_ascii=False, indent=2)

    def _normalize_records_for_json(self, rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        normalized_rows: list[Dict[str, Any]] = []
        for row in rows:
            normalized_row: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    normalized_row[key] = value.isoformat()
                else:
                    normalized_row[key] = value
            normalized_rows.append(normalized_row)
        return normalized_rows

    def _execute_feature_study_experiment(
        self,
        *,
        target_id: str,
        target_series: pd.Series,
        target_spec: object,
        feature_set: Any,
        artifact_prefix: str,
        selector_name: str,
        selector_max_features: int,
        request: FeatureStudyRequest,
    ) -> Dict[str, Any]:
        experiment_request = ResearchExperimentRequest(
            experiment_name=f"{request.study_name}_{target_id}_{feature_set.name}",
            target_column=target_id,
            feature_columns=list(feature_set.columns),
            trainer_name=request.trainer_name,
            baseline_names=list(request.baseline_names),
            train_size=request.train_size,
            validation_size=request.validation_size,
            test_size=request.test_size,
            step_size=request.step_size,
            threshold_list=list(request.threshold_list),
            expanding_window=request.expanding_window,
        )
        selector = build_feature_selector(
            selector_name,
            max_features=min(int(selector_max_features), max(len(feature_set.columns), 1)),
        )
        return self._execute_research_experiment(
            request=experiment_request,
            target_series=target_series,
            artifact_prefix=artifact_prefix,
            feature_selector=selector,
            target_spec=target_spec,
        )
