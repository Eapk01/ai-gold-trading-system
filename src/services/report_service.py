"""Report listing/loading workflow helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from loguru import logger

from src.research import (
    ExperimentStore,
    get_feature_set_description,
    get_feature_set_display_name,
    get_stage5_preset_description,
    get_stage5_preset_display_name,
    resolve_integrity_payload,
)
from src.report_store import ReportDefinition, ReportStore

if TYPE_CHECKING:
    from src.app_service import ResearchAppService


BACKTEST_REPORT = ReportDefinition("backtest", "backtest_result", "backtest_summary")
MODEL_TEST_REPORT = ReportDefinition("model_test", "model_test_result", "model_test_summary")


def _format_report_timestamp(raw_timestamp: str) -> str:
    candidate = str(raw_timestamp or "").strip()
    for pattern in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(candidate, pattern).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return candidate.replace("_", " ")


def _merge_integrity_summary(summary: Dict[str, Any], integrity: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(summary or {})
    overview = dict((integrity or {}).get("overview") or {})
    merged.update(
        {
            "integrity_contract_ok": (integrity or {}).get("integrity_contract_ok"),
            "proof_status": (integrity or {}).get("proof_status"),
            "horizon_bars": overview.get("horizon_bars"),
            "total_purged_train_rows": overview.get("total_purged_train_rows"),
            "total_purged_validation_rows": overview.get("total_purged_validation_rows"),
            "invalid_fold_count": overview.get("invalid_fold_count"),
        }
    )
    return merged


class ReportWorkflowService:
    """Shared report listing/loading helpers."""

    def __init__(self, service: "ResearchAppService", store: ReportStore, experiment_store: ExperimentStore) -> None:
        self.service = service
        self.store = store
        self.experiment_store = experiment_store

    def list_backtest_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = self.store.list_reports(BACKTEST_REPORT, limit=limit)
        message = "Backtest reports loaded" if reports else "No backtest report files found"
        return self.service._response(True, message, data=reports)

    def get_backtest_report(self, report_path: str) -> Dict[str, Any]:
        try:
            report_data = self.store.load_report(BACKTEST_REPORT, report_path)
            payload = report_data["report_payload"]
            return self.service._response(
                True,
                f"Loaded backtest report: {report_data['name']}",
                data={
                    "report_type": report_data["report_type"],
                    "path": report_path,
                    "summary": report_data["summary"],
                    "trade_count": len(payload.get("trades", [])),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load backtest report: {exc}")
            return self.service._response(False, f"Failed to load backtest report: {exc}", errors=[str(exc)])

    def list_model_test_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = self.store.list_reports(MODEL_TEST_REPORT, limit=limit)
        message = "Model test reports loaded" if reports else "No model test report files found"
        return self.service._response(True, message, data=reports)

    def get_model_test_report(self, report_path: str) -> Dict[str, Any]:
        try:
            report_data = self.store.load_report(MODEL_TEST_REPORT, report_path)
            payload = report_data["report_payload"]
            return self.service._response(
                True,
                f"Loaded model test report: {report_data['name']}",
                data={
                    "report_type": report_data["report_type"],
                    "path": report_path,
                    "summary": report_data["summary"],
                    "threshold_performance": payload.get("threshold_performance", []),
                    "confidence_buckets": payload.get("confidence_buckets", []),
                    "row_evaluations_file": payload.get("row_evaluations_file"),
                    "target_column": payload.get("target_column"),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load model test report: {exc}")
            return self.service._response(False, f"Failed to load model test report: {exc}", errors=[str(exc)])

    def list_experiment_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "research_experiment")
            for path in self.experiment_store.list_results(limit=limit, prefix="research_experiment_")
        ]
        message = "Experiment reports loaded" if reports else "No experiment report files found"
        return self.service._response(True, message, data=reports)

    def get_experiment_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            aggregate_metrics = payload.get("aggregate_metrics", {})
            metadata = payload.get("metadata", {})
            integrity = resolve_integrity_payload(
                metadata=dict(metadata or {}),
                stored_integrity=dict(payload.get("integrity") or {}),
                expected_feature_set_name=str(metadata.get("research_feature_set_name") or ""),
                expected_feature_selection_mode=str(metadata.get("feature_selection_mode") or ""),
                expected_target_spec_id=str(metadata.get("target_spec_id") or payload.get("target_column") or ""),
                fallback_fold_count=int(aggregate_metrics.get("fold_count") or 0),
            )
            summary = {
                "experiment_name": payload.get("experiment_name"),
                "target_column": payload.get("target_column"),
                "trainer_name": payload.get("request", {}).get("trainer_name"),
                "feature_count": len(payload.get("feature_columns", [])),
                "fold_count": aggregate_metrics.get("fold_count", 0),
                "mean_test_accuracy": aggregate_metrics.get("mean_test_accuracy"),
                "feature_set_name": metadata.get("research_feature_set_name"),
                "feature_set_display_name": metadata.get("research_feature_set_display_name"),
            }
            summary = _merge_integrity_summary(summary, integrity)
            artifacts = (payload.get("prediction_artifacts") or [{}])[0]
            return self.service._response(
                True,
                f"Loaded experiment report: {payload.get('experiment_name', report_path)}",
                data={
                    "report_type": "research_experiment",
                    "path": str(report_path),
                    "summary": summary,
                    "aggregate_metrics": aggregate_metrics,
                    "baseline_comparison": payload.get("baseline_comparison", {}),
                    "calibration_summary": payload.get("calibration_summary", {}),
                    "threshold_summary": payload.get("threshold_summary", {}),
                    "folds": payload.get("folds", []),
                    "metadata": metadata,
                    "resolved_research_defaults": metadata.get("resolved_research_defaults", {}),
                    "integrity": integrity,
                    "integrity_artifact_paths": payload.get("integrity_artifact_paths", {}),
                    "artifact_paths": {
                        "report_file": str(report_path),
                        "prediction_rows_file": artifacts.get("predictions_file"),
                        "threshold_metrics_file": artifacts.get("threshold_metrics_file"),
                        "calibration_file": artifacts.get("calibration_file"),
                        **(payload.get("integrity_artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load experiment report: {exc}")
            return self.service._response(False, f"Failed to load experiment report: {exc}", errors=[str(exc)])

    def list_target_study_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "target_study")
            for path in self.experiment_store.list_results(limit=limit, prefix="target_study_")
        ]
        message = "Target study reports loaded" if reports else "No target study report files found"
        return self.service._response(True, message, data=reports)

    def get_target_study_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            target_results = payload.get("target_results", [])
            comparison_rows = payload.get("comparison_rows", [])
            metadata = payload.get("metadata", {})
            integrity = resolve_integrity_payload(
                metadata=dict(metadata or {}),
                stored_integrity=dict(payload.get("integrity") or {}),
                expected_feature_set_name=str(metadata.get("research_feature_set_name") or ""),
                fallback_fold_count=len(target_results),
            )
            normalized_target_results = []
            for row in target_results:
                target_integrity = resolve_integrity_payload(
                    stored_integrity=dict((row or {}).get("integrity") or {}),
                    fallback_fold_count=0,
                )
                normalized_target_results.append(
                    {
                        **row,
                        "integrity": target_integrity,
                    }
                )
            best_row = max(
                (row for row in comparison_rows if row.get("model_mean_test_accuracy") is not None and not row.get("error")),
                key=lambda row: row.get("model_mean_test_accuracy") or float("-inf"),
                default=None,
            )
            summary = {
                "study_name": payload.get("study_name"),
                "target_count": len(normalized_target_results),
                "successful_targets": sum(1 for row in normalized_target_results if not row.get("error")),
                "feature_set_name": metadata.get("research_feature_set_name"),
                "feature_set_display_name": metadata.get("research_feature_set_display_name"),
                "best_target_name": (best_row or {}).get("display_name"),
                "best_mean_test_accuracy": (best_row or {}).get("model_mean_test_accuracy"),
            }
            summary = _merge_integrity_summary(summary, integrity)
            return self.service._response(
                True,
                f"Loaded target study report: {payload.get('study_name', report_path)}",
                data={
                    "report_type": "target_study",
                    "path": str(report_path),
                    "summary": summary,
                    "target_results": normalized_target_results,
                    "comparison_rows": comparison_rows,
                    "metadata": metadata,
                    "resolved_research_defaults": metadata.get("resolved_research_defaults", {}),
                    "integrity": integrity,
                    "integrity_artifact_paths": payload.get("integrity_artifact_paths", {}),
                    "artifact_paths": {
                        "report_file": str(report_path),
                        **(payload.get("artifact_paths") or {}),
                        **(payload.get("integrity_artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load target study report: {exc}")
            return self.service._response(False, f"Failed to load target study report: {exc}", errors=[str(exc)])

    def list_feature_study_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "feature_study")
            for path in self.experiment_store.list_results(limit=limit, prefix="feature_study_")
        ]
        message = "Feature study reports loaded" if reports else "No feature study report files found"
        return self.service._response(True, message, data=reports)

    def get_feature_study_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            set_results = payload.get("set_results", [])
            comparison_rows = payload.get("comparison_rows", [])
            best_row = max(
                (row for row in comparison_rows if row.get("mean_test_accuracy") is not None and not row.get("error")),
                key=lambda row: row.get("mean_test_accuracy") or float("-inf"),
                default=None,
            )
            summary = {
                "study_name": payload.get("study_name"),
                "feature_set_count": len({row.get("feature_set_name") for row in comparison_rows}),
                "target_count": len({row.get("target_id") for row in comparison_rows}),
                "successful_runs": sum(1 for row in comparison_rows if not row.get("error")),
                "working_target_id": payload.get("working_target_id"),
                "best_feature_set_name": (best_row or {}).get("feature_set_display_name"),
                "best_mean_test_accuracy": (best_row or {}).get("mean_test_accuracy"),
            }
            enriched_comparison_rows = [
                {
                    **row,
                    "feature_set_display_name": get_feature_set_display_name(str(row.get("feature_set_name") or "")),
                    "feature_set_description": get_feature_set_description(str(row.get("feature_set_name") or "")),
                }
                for row in comparison_rows
            ]
            enriched_set_results = [
                {
                    **row,
                    "feature_set_display_name": row.get("feature_set_display_name") or get_feature_set_display_name(str(row.get("feature_set_name") or "")),
                    "feature_set_description": get_feature_set_description(str(row.get("feature_set_name") or "")),
                }
                for row in set_results
            ]
            return self.service._response(
                True,
                f"Loaded feature study report: {payload.get('study_name', report_path)}",
                data={
                    "report_type": "feature_study",
                    "path": str(report_path),
                    "summary": summary,
                    "inventory_rows": payload.get("inventory_rows", []),
                    "set_results": enriched_set_results,
                    "comparison_rows": enriched_comparison_rows,
                    "artifact_paths": {
                        "report_file": str(report_path),
                        **(payload.get("artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load feature study report: {exc}")
            return self.service._response(False, f"Failed to load feature study report: {exc}", errors=[str(exc)])

    def list_training_experiment_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "training_experiment")
            for path in self.experiment_store.list_results(limit=limit, prefix="training_experiment_")
        ]
        message = "Training experiment reports loaded" if reports else "No training experiment report files found"
        return self.service._response(True, message, data=reports)

    def get_training_experiment_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            selected_threshold_summary = (payload.get("metadata") or {}).get("selected_threshold_summary") or {}
            integrity = resolve_integrity_payload(
                metadata=dict(payload.get("metadata") or {}),
                stored_integrity=dict(payload.get("integrity") or {}),
                expected_feature_set_name=str(payload.get("feature_set_name") or ""),
                expected_feature_selection_mode="fold_local_selector" if payload.get("selector_name") else "fixed_feature_columns",
                expected_target_spec_id=str((payload.get("target_spec") or {}).get("spec_id") or ""),
                fallback_fold_count=len({row.get("fold_name") for row in payload.get("folds", []) if row.get("fold_name")}),
            )
            summary = {
                "experiment_id": payload.get("experiment_id"),
                "experiment_name": payload.get("experiment_name"),
                "target_spec_id": (payload.get("target_spec") or {}).get("spec_id"),
                "feature_set_name": payload.get("feature_set_name"),
                "feature_set_display_name": get_feature_set_display_name(str(payload.get("feature_set_name") or "")),
                "feature_set_description": get_feature_set_description(str(payload.get("feature_set_name") or "")),
                "comparison_feature_set_name": payload.get("comparison_feature_set_name"),
                "comparison_feature_set_display_name": get_feature_set_display_name(str(payload.get("comparison_feature_set_name") or "")) if payload.get("comparison_feature_set_name") else "",
                "trainer_name": payload.get("trainer_name"),
                "selected_threshold": payload.get("selected_threshold"),
                "feature_count": len(payload.get("resolved_feature_columns", [])),
                "fold_count": len({row.get("fold_name") for row in payload.get("folds", []) if row.get("model_name") == payload.get("trainer_name")}),
                "mean_test_accuracy": (payload.get("aggregate_metrics") or {}).get("mean_test_accuracy"),
                "majority_baseline_mean_test_accuracy": ((payload.get("baseline_comparison") or {}).get("baselines", {}).get("majority_class") or {}).get("mean_test_accuracy"),
                "selected_threshold_validation_mean_f1": (selected_threshold_summary.get("validation") or {}).get("mean_f1"),
                "selected_threshold_validation_mean_coverage": (selected_threshold_summary.get("validation") or {}).get("mean_coverage"),
                "selected_threshold_test_mean_f1": (selected_threshold_summary.get("test") or {}).get("mean_f1"),
                "selected_threshold_test_mean_coverage": (selected_threshold_summary.get("test") or {}).get("mean_coverage"),
                "candidate_model_path": (payload.get("candidate_artifact") or {}).get("artifact_path"),
                "one_class_fold_count": ((payload.get("diagnostics") or {}).get("highlights") or {}).get("one_class_fold_count"),
                "constant_feature_fold_count": ((payload.get("diagnostics") or {}).get("highlights") or {}).get("constant_feature_fold_count"),
                "near_constant_feature_fold_count": ((payload.get("diagnostics") or {}).get("highlights") or {}).get("near_constant_feature_fold_count"),
            }
            summary = _merge_integrity_summary(summary, integrity)
            artifacts = (payload.get("prediction_artifacts") or [{}])[0]
            return self.service._response(
                True,
                f"Loaded training experiment report: {payload.get('experiment_name', report_path)}",
                data={
                    "report_type": "training_experiment",
                    "path": str(report_path),
                    "summary": summary,
                    "aggregate_metrics": payload.get("aggregate_metrics", {}),
                    "baseline_comparison": payload.get("baseline_comparison", {}),
                    "comparison_runs": payload.get("comparison_runs", []),
                    "selected_threshold_summary": selected_threshold_summary,
                    "diagnostics": payload.get("diagnostics", {}),
                    "resolved_research_defaults": (payload.get("metadata") or {}).get("resolved_research_defaults", {}),
                    "integrity": integrity,
                    "candidate_artifact": payload.get("candidate_artifact", {}),
                    "promotion_status": payload.get("promotion_status"),
                    "promotion_manifest_file": payload.get("promotion_manifest_file"),
                    "integrity_artifact_paths": payload.get("integrity_artifact_paths", {}),
                    "artifact_paths": {
                        "report_file": str(report_path),
                        "prediction_rows_file": artifacts.get("predictions_file"),
                        "threshold_metrics_file": artifacts.get("threshold_metrics_file"),
                        "calibration_file": artifacts.get("calibration_file"),
                        "model_path": (payload.get("candidate_artifact") or {}).get("artifact_path"),
                        **(payload.get("diagnostics_artifact_paths") or {}),
                        **(payload.get("integrity_artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load training experiment report: {exc}")
            return self.service._response(False, f"Failed to load training experiment report: {exc}", errors=[str(exc)])

    def list_search_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "search_run")
            for path in self.experiment_store.list_results(limit=limit, prefix="search_run_")
        ]
        message = "Search reports loaded" if reports else "No search report files found"
        return self.service._response(True, message, data=reports)

    def get_search_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            recommended = payload.get("recommended_winner") or {}
            gate_summary = (payload.get("metadata") or {}).get("gate_summary") or {}
            integrity = resolve_integrity_payload(
                metadata=dict(payload.get("metadata") or {}),
                stored_integrity=dict(payload.get("integrity") or {}),
                fallback_fold_count=int(payload.get("candidate_count") or 0),
            )
            summary = {
                "search_id": payload.get("search_id"),
                "search_name": payload.get("search_name"),
                "target_spec_id": (payload.get("target_spec") or {}).get("spec_id"),
                "target_count": len(payload.get("target_specs") or ([payload.get("target_spec")] if payload.get("target_spec") else [])),
                "searched_target_ids": [
                    spec.get("spec_id")
                    for spec in (payload.get("target_specs") or ([payload.get("target_spec")] if payload.get("target_spec") else []))
                ],
                "searched_target_display_names": [
                    spec.get("display_name") or spec.get("spec_id")
                    for spec in (payload.get("target_specs") or ([payload.get("target_spec")] if payload.get("target_spec") else []))
                ],
                "trainer_name": payload.get("trainer_name"),
                "feature_set_count": len(payload.get("feature_set_names") or []),
                "candidate_count": payload.get("candidate_count", 0),
                "successful_candidate_count": payload.get("successful_candidate_count", 0),
                "failed_candidate_count": payload.get("failed_candidate_count", 0),
                "recommended_experiment_id": recommended.get("experiment_id"),
                "recommended_target_spec_id": recommended.get("target_spec_id"),
                "recommended_target_display_name": recommended.get("target_display_name"),
                "recommended_feature_set_name": recommended.get("feature_set_name"),
                "recommended_feature_set_display_name": get_feature_set_display_name(str(recommended.get("feature_set_name") or "")),
                "recommended_feature_set_description": get_feature_set_description(str(recommended.get("feature_set_name") or "")),
                "recommended_preset_name": recommended.get("preset_name"),
                "recommended_preset_display_name": get_stage5_preset_display_name(str(recommended.get("preset_name") or "")),
                "recommended_preset_description": get_stage5_preset_description(str(recommended.get("preset_name") or "")),
                "recommended_selected_threshold": recommended.get("selected_threshold"),
                "winner_status": recommended.get("status"),
                "winner_reason": recommended.get("reason"),
                "truth_gate_pass_count": gate_summary.get("passed_truth_gate_count"),
                "test_guardrail_pass_count": gate_summary.get("passed_test_guardrail_count"),
                "truth_gate_failures": gate_summary.get("failure_counts"),
                "low_coverage_candidate_count": ((payload.get("diagnostics") or {}).get("summary") or {}).get("low_coverage_candidate_count"),
                "majority_dominance_candidate_count": ((payload.get("diagnostics") or {}).get("summary") or {}).get("majority_dominance_candidate_count"),
                "one_class_candidate_count": ((payload.get("diagnostics") or {}).get("summary") or {}).get("one_class_candidate_count"),
                "integrity_failure_candidate_count": ((payload.get("diagnostics") or {}).get("summary") or {}).get("integrity_failure_candidate_count"),
                "execution_mode": payload.get("execution_mode") or (payload.get("metadata") or {}).get("execution_mode"),
                "resolved_max_workers": payload.get("resolved_max_workers") or (payload.get("metadata") or {}).get("resolved_max_workers"),
                "elapsed_seconds": (payload.get("metadata") or {}).get("elapsed_seconds"),
            }
            summary = _merge_integrity_summary(summary, integrity)
            leaderboard_rows = [
                {
                    **row,
                    "target_display_name": row.get("target_display_name") or row.get("target_spec_id"),
                    "feature_set_display_name": get_feature_set_display_name(str(row.get("feature_set_name") or "")),
                    "feature_set_description": get_feature_set_description(str(row.get("feature_set_name") or "")),
                    "preset_display_name": get_stage5_preset_display_name(str(row.get("preset_name") or "")),
                    "preset_description": get_stage5_preset_description(str(row.get("preset_name") or "")),
                }
                for row in payload.get("leaderboard_rows", [])
            ]
            candidates = [
                {
                    **row,
                    "target_display_name": row.get("target_display_name") or row.get("target_spec_id"),
                    "feature_set_display_name": get_feature_set_display_name(str(row.get("feature_set_name") or "")),
                    "feature_set_description": get_feature_set_description(str(row.get("feature_set_name") or "")),
                    "preset_display_name": get_stage5_preset_display_name(str(row.get("preset_name") or "")),
                    "preset_description": get_stage5_preset_description(str(row.get("preset_name") or "")),
                }
                for row in payload.get("candidates", [])
            ]
            recommended = {
                **recommended,
                "target_display_name": recommended.get("target_display_name") or recommended.get("target_spec_id"),
                "feature_set_display_name": get_feature_set_display_name(str(recommended.get("feature_set_name") or "")),
                "feature_set_description": get_feature_set_description(str(recommended.get("feature_set_name") or "")),
                "preset_display_name": get_stage5_preset_display_name(str(recommended.get("preset_name") or "")),
                "preset_description": get_stage5_preset_description(str(recommended.get("preset_name") or "")),
            }
            diagnostics = dict(payload.get("diagnostics") or {})
            candidate_highlights = [
                {
                    **row,
                    "target_display_name": row.get("target_display_name") or row.get("target_spec_id"),
                    "feature_set_display_name": get_feature_set_display_name(str(row.get("feature_set_name") or "")),
                    "feature_set_description": get_feature_set_description(str(row.get("feature_set_name") or "")),
                    "preset_display_name": get_stage5_preset_display_name(str(row.get("preset_name") or "")),
                    "preset_description": get_stage5_preset_description(str(row.get("preset_name") or "")),
                }
                for row in diagnostics.get("candidate_highlights", [])
            ]
            if diagnostics:
                diagnostics["candidate_highlights"] = candidate_highlights
            return self.service._response(
                True,
                f"Loaded search report: {payload.get('search_name', report_path)}",
                data={
                    "report_type": "search_run",
                    "path": str(report_path),
                    "summary": summary,
                    "target_specs": payload.get("target_specs") or ([payload.get("target_spec")] if payload.get("target_spec") else []),
                    "leaderboard_rows": leaderboard_rows,
                    "candidates": candidates,
                    "recommended_winner": recommended,
                    "preset_definitions": payload.get("preset_definitions", {}),
                    "gate_summary": gate_summary,
                    "diagnostics": diagnostics,
                    "resolved_research_defaults": (payload.get("metadata") or {}).get("resolved_research_defaults", {}),
                    "integrity": integrity,
                    "integrity_artifact_paths": payload.get("integrity_artifact_paths", {}),
                    "artifact_paths": {
                        "report_file": str(report_path),
                        **(payload.get("artifact_paths") or {}),
                        **(payload.get("integrity_artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load search report: {exc}")
            return self.service._response(False, f"Failed to load search report: {exc}", errors=[str(exc)])

    def list_promotion_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = [
            self._build_research_report_entry(path, "promotion")
            for path in self.experiment_store.list_results(limit=limit, prefix="promotion_")
        ]
        message = "Promotion reports loaded" if reports else "No promotion report files found"
        return self.service._response(True, message, data=reports)

    def get_promotion_report(self, report_path: str) -> Dict[str, Any]:
        try:
            payload = self.experiment_store.load_result(report_path)
            integrity = resolve_integrity_payload(
                stored_integrity=dict(payload.get("integrity") or {}),
            )
            summary = {
                "experiment_name": payload.get("experiment_name"),
                "source_model_path": payload.get("source_model_path"),
                "promoted_model_path": payload.get("promoted_model_path"),
                "feature_set_name": payload.get("feature_set_name"),
                "selected_threshold": payload.get("selected_threshold"),
            }
            summary = _merge_integrity_summary(summary, integrity)
            return self.service._response(
                True,
                f"Loaded promotion report: {payload.get('experiment_name', report_path)}",
                data={
                    "report_type": "promotion",
                    "path": str(report_path),
                    "summary": summary,
                    "manifest": payload,
                    "resolved_research_defaults": (payload.get("metadata") or {}).get("resolved_research_defaults", {}),
                    "integrity": integrity,
                    "integrity_artifact_paths": payload.get("integrity_artifact_paths", {}),
                    "artifact_paths": {
                        "report_file": str(report_path),
                        "model_path": payload.get("promoted_model_path"),
                        **(payload.get("integrity_artifact_paths") or {}),
                    },
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load promotion report: {exc}")
            return self.service._response(False, f"Failed to load promotion report: {exc}", errors=[str(exc)])

    def _build_research_report_entry(self, path: Path, report_type: str) -> Dict[str, Any]:
        payload = self.experiment_store.load_result(path)
        timestamp = _format_report_timestamp(self._extract_report_timestamp(path))
        return {
            "report_type": report_type,
            "path": str(path),
            "name": self._build_report_display_name(payload, path, report_type, timestamp),
            "timestamp": timestamp,
        }

    def _build_report_display_name(self, payload: Dict[str, Any], path: Path, report_type: str, timestamp: str) -> str:
        if report_type == "target_study":
            return f"{timestamp} | Target Study | {payload.get('study_name', path.stem)}"
        if report_type == "feature_study":
            return f"{timestamp} | Feature Study | {payload.get('study_name', path.stem)}"
        if report_type == "training_experiment":
            feature_set_name = str(payload.get("feature_set_name") or "")
            feature_label = get_feature_set_display_name(feature_set_name)
            experiment_name = payload.get("experiment_name", path.stem)
            preset_name = str((payload.get("metadata") or {}).get("preset_name") or "")
            preset_label = get_stage5_preset_display_name(preset_name) if preset_name else ""
            suffix = f" | {preset_label}" if preset_label else ""
            return f"{timestamp} | Candidate | {experiment_name} | {feature_label}{suffix}"
        if report_type == "search_run":
            return f"{timestamp} | Search | {payload.get('search_name', path.stem)}"
        if report_type == "promotion":
            feature_set_name = str(payload.get("feature_set_name") or "")
            feature_label = get_feature_set_display_name(feature_set_name) if feature_set_name else "Promoted Model"
            return f"{timestamp} | Promotion | {payload.get('experiment_name', path.stem)} | {feature_label}"
        return f"{timestamp} | Experiment | {payload.get('experiment_name', path.stem)}"

    def _extract_report_timestamp(self, path: Path) -> str:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            return f"{parts[-2]}_{parts[-1]}"
        for index in range(len(parts) - 1):
            if len(parts[index]) == 8 and parts[index].isdigit() and len(parts[index + 1]) == 6 and parts[index + 1].isdigit():
                return f"{parts[index]}_{parts[index + 1]}"
        return stem
