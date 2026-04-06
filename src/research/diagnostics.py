"""Structured diagnostics helpers for Stage 5.2 research audits."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from .schemas import ExperimentResult, ResearchSplit, SearchCandidateSummary


LOW_COVERAGE_WARNING_THRESHOLD = 0.20
HIGH_NAN_RATE_WARNING_THRESHOLD = 0.05
NEAR_CONSTANT_FEATURE_RATIO = 0.95


def build_experiment_integrity_proof(
    *,
    experiment_result: ExperimentResult,
    expected_feature_set_name: str | None = None,
    expected_feature_selection_mode: str | None = None,
    expected_target_spec_id: str | None = None,
) -> Dict[str, Any]:
    """Build a normalized integrity proof payload for one experiment result."""
    fallback_fold_count = len(
        {
            getattr(fold, "fold_name", "")
            for fold in (experiment_result.folds or [])
            if getattr(fold, "fold_name", "")
        }
    ) or len(experiment_result.fold_boundaries or [])
    return resolve_integrity_payload(
        metadata=dict(experiment_result.metadata or {}),
        stored_integrity=dict(experiment_result.integrity or {}),
        expected_feature_set_name=expected_feature_set_name,
        expected_feature_selection_mode=expected_feature_selection_mode,
        expected_target_spec_id=expected_target_spec_id,
        fallback_fold_count=fallback_fold_count,
    )


def resolve_integrity_payload(
    *,
    metadata: Dict[str, Any] | None = None,
    stored_integrity: Dict[str, Any] | None = None,
    expected_feature_set_name: str | None = None,
    expected_feature_selection_mode: str | None = None,
    expected_target_spec_id: str | None = None,
    fallback_fold_count: int = 0,
) -> Dict[str, Any]:
    """Resolve integrity proof from metadata or a previously stored payload."""
    metadata = dict(metadata or {})
    stored_integrity = dict(stored_integrity or {})
    fold_rows = metadata.get("integrity_fold_rows") or []
    if fold_rows:
        return _build_integrity_from_metadata(
            metadata=metadata,
            expected_feature_set_name=expected_feature_set_name,
            expected_feature_selection_mode=expected_feature_selection_mode,
            expected_target_spec_id=expected_target_spec_id,
            fallback_fold_count=fallback_fold_count,
        )
    if stored_integrity:
        return _normalize_stored_integrity(
            stored_integrity=stored_integrity,
            expected_feature_set_name=expected_feature_set_name,
            expected_feature_selection_mode=expected_feature_selection_mode,
            expected_target_spec_id=expected_target_spec_id,
            fallback_fold_count=fallback_fold_count,
        )
    return _build_missing_integrity(
        expected_feature_set_name=expected_feature_set_name,
        expected_feature_selection_mode=expected_feature_selection_mode,
        expected_target_spec_id=expected_target_spec_id,
        fallback_fold_count=fallback_fold_count,
    )


def build_training_experiment_diagnostics(
    *,
    feature_frame: pd.DataFrame,
    target_series: pd.Series,
    splits: List[ResearchSplit],
    experiment_result: ExperimentResult,
    trainer_name: str,
    selected_threshold: float | None,
    selected_threshold_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build compact diagnostics for one canonical training experiment."""
    numeric_target = pd.to_numeric(target_series, errors="coerce")
    prediction_rows = _frame_from_artifact_rows(experiment_result, "prediction_rows")
    threshold_rows = _frame_from_artifact_rows(experiment_result, "threshold_rows")

    target_balance_overall = _summarize_target_balance(numeric_target)
    fold_target_balance_rows = _build_fold_target_balance_rows(splits, numeric_target)
    feature_health_rows = _build_feature_health_rows(
        feature_frame=feature_frame,
        splits=splits,
        experiment_result=experiment_result,
    )
    prediction_health_rows = _build_prediction_health_rows(
        prediction_rows=prediction_rows,
        threshold_rows=threshold_rows,
        trainer_name=trainer_name,
        selected_threshold=selected_threshold,
    )
    baseline_context = _build_baseline_context(experiment_result)
    summary_payload = dict(selected_threshold_summary or {})
    validation_threshold_summary = dict(summary_payload.get("validation") or {})
    test_threshold_summary = dict(summary_payload.get("test") or {})

    one_class_fold_count = sum(1 for row in fold_target_balance_rows if bool(row.get("one_class_flag")))
    undefined_threshold_metric_count = sum(
        1
        for row in prediction_health_rows
        if bool(row.get("undefined_selected_threshold_metrics"))
    )
    constant_feature_fold_count = sum(
        1 for row in feature_health_rows
        if int(row.get("constant_selected_feature_count") or 0) > 0
    )
    near_constant_feature_fold_count = sum(
        1 for row in feature_health_rows
        if int(row.get("near_constant_selected_feature_count") or 0) > 0
    )
    high_nan_feature_fold_count = sum(
        1 for row in feature_health_rows
        if float(row.get("selected_nan_rate_max") or 0.0) >= HIGH_NAN_RATE_WARNING_THRESHOLD
    )

    model_mean_test_accuracy = _safe_float((experiment_result.aggregate_metrics or {}).get("mean_test_accuracy"))
    majority_baseline_accuracy = _safe_float(baseline_context.get("majority_mean_test_accuracy"))
    coverage_value = _safe_float(test_threshold_summary.get("mean_coverage"))
    validation_mean_f1 = _safe_float(validation_threshold_summary.get("mean_f1"))
    test_mean_f1 = _safe_float(test_threshold_summary.get("mean_f1"))
    validation_test_drift = (
        float(validation_mean_f1 - test_mean_f1)
        if validation_mean_f1 is not None and test_mean_f1 is not None
        else None
    )

    highlights = {
        "selected_threshold": _safe_float(selected_threshold),
        "selected_threshold_test_mean_coverage": coverage_value,
        "selected_threshold_test_mean_f1": test_mean_f1,
        "selected_threshold_validation_mean_f1": validation_mean_f1,
        "validation_test_f1_drift": validation_test_drift,
        "overall_mean_test_accuracy": model_mean_test_accuracy,
        "majority_baseline_mean_test_accuracy": majority_baseline_accuracy,
        "low_selected_threshold_test_coverage": (
            coverage_value is None or coverage_value < LOW_COVERAGE_WARNING_THRESHOLD
        ),
        "broad_metric_under_majority": (
            model_mean_test_accuracy is not None
            and majority_baseline_accuracy is not None
            and model_mean_test_accuracy < majority_baseline_accuracy
        ),
        "one_class_fold_count": one_class_fold_count,
        "undefined_selected_threshold_metric_count": undefined_threshold_metric_count,
        "constant_feature_fold_count": constant_feature_fold_count,
        "near_constant_feature_fold_count": near_constant_feature_fold_count,
        "high_nan_feature_fold_count": high_nan_feature_fold_count,
    }

    warnings = _build_training_warnings(highlights, baseline_context)
    return {
        "overview": {
            "selected_threshold": _safe_float(selected_threshold),
            "scored_target_rows": int(target_balance_overall.get("scored_rows") or 0),
            "target_positive_rate": target_balance_overall.get("positive_rate"),
            "target_missing_rate": target_balance_overall.get("missing_rate"),
            "overall_mean_test_accuracy": model_mean_test_accuracy,
            "majority_baseline_mean_test_accuracy": majority_baseline_accuracy,
            "selected_threshold_test_mean_f1": test_mean_f1,
            "selected_threshold_test_mean_coverage": coverage_value,
        },
        "warnings": warnings,
        "highlights": highlights,
        "target_balance": {
            "overall": target_balance_overall,
            "fold_rows": fold_target_balance_rows,
        },
        "feature_health_rows": feature_health_rows,
        "prediction_health_rows": prediction_health_rows,
        "baseline_context": baseline_context,
    }


def build_search_diagnostics(candidates: Iterable[SearchCandidateSummary]) -> Dict[str, Any]:
    """Build aggregate diagnostics for one bounded search run."""
    candidate_list = list(candidates)
    candidate_highlight_rows: List[Dict[str, Any]] = []
    low_coverage_count = 0
    majority_dominance_count = 0
    one_class_candidate_count = 0
    undefined_metric_candidate_count = 0
    runtime_mismatch_count = 0
    integrity_failure_count = 0
    failed_candidate_count = 0

    for candidate in candidate_list:
        execution_status = str(candidate.execution_status or "completed")
        if execution_status != "completed":
            failed_candidate_count += 1
        candidate_diagnostics = dict(candidate.diagnostics or {})
        highlights = dict(candidate_diagnostics.get("highlights") or {})
        integrity = dict(candidate_diagnostics.get("integrity") or {})
        if execution_status == "completed" and bool(highlights.get("low_selected_threshold_test_coverage")):
            low_coverage_count += 1
        if execution_status == "completed" and bool(highlights.get("broad_metric_under_majority")):
            majority_dominance_count += 1
        if execution_status == "completed" and int(highlights.get("one_class_fold_count") or 0) > 0:
            one_class_candidate_count += 1
        if execution_status == "completed" and int(highlights.get("undefined_selected_threshold_metric_count") or 0) > 0:
            undefined_metric_candidate_count += 1
        if execution_status == "completed" and not bool(candidate.runtime_feature_contract_ok):
            runtime_mismatch_count += 1
        if execution_status == "completed" and not bool(integrity.get("integrity_contract_ok")):
            integrity_failure_count += 1

        candidate_highlight_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "experiment_name": candidate.experiment_name,
                "target_spec_id": candidate.target_spec_id,
                "target_display_name": candidate.target_display_name,
                "feature_set_name": candidate.feature_set_name,
                "preset_name": candidate.preset_name,
                "execution_status": execution_status,
                "error_message": candidate.error_message,
                "elapsed_seconds": candidate.elapsed_seconds,
                "proof_status": integrity.get("proof_status"),
                "integrity_contract_ok": bool(integrity.get("integrity_contract_ok")),
                "integrity_failure_reasons": list((integrity.get("overview") or {}).get("contract_failure_reasons") or []),
                "passed_truth_gate": bool(candidate.passed_truth_gate),
                "truth_gate_failures": list(candidate.truth_gate_failures),
                "overall_mean_test_accuracy": candidate.overall_mean_test_accuracy,
                "majority_baseline_mean_test_accuracy": candidate.majority_baseline_mean_test_accuracy,
                "selected_threshold_test_mean_f1": (candidate.test_summary or {}).get("mean_f1"),
                "selected_threshold_test_mean_coverage": (candidate.test_summary or {}).get("mean_coverage"),
                "one_class_fold_count": highlights.get("one_class_fold_count"),
                "undefined_selected_threshold_metric_count": highlights.get("undefined_selected_threshold_metric_count"),
                "constant_feature_fold_count": highlights.get("constant_feature_fold_count"),
                "near_constant_feature_fold_count": highlights.get("near_constant_feature_fold_count"),
            }
        )

    warnings: List[Dict[str, Any]] = []
    candidate_count = len(candidate_list)
    if candidate_count and majority_dominance_count == candidate_count:
        warnings.append(
            {
                "code": "majority_dominates_all_candidates",
                "severity": "critical",
                "message": "Every candidate is still below the majority baseline on broad test accuracy.",
            }
        )
    if candidate_count and low_coverage_count >= max(1, candidate_count // 2):
        warnings.append(
            {
                "code": "low_coverage_widespread",
                "severity": "warning",
                "message": "Low selected-threshold test coverage is widespread across this bounded search.",
            }
        )
    if one_class_candidate_count:
        warnings.append(
            {
                "code": "one_class_folds_detected",
                "severity": "warning",
                "message": "Some candidates were evaluated on one-class or degenerate folds.",
            }
        )
    if undefined_metric_candidate_count:
        warnings.append(
            {
                "code": "undefined_threshold_metrics_present",
                "severity": "warning",
                "message": "Some candidates have undefined selected-threshold metrics in at least one fold/segment.",
            }
        )
    if runtime_mismatch_count:
        warnings.append(
            {
                "code": "runtime_feature_contract_mismatch",
                "severity": "critical",
                "message": "Some candidates have runtime/artifact feature mismatches and are unsafe to load.",
            }
        )
    if integrity_failure_count:
        warnings.append(
            {
                "code": "integrity_failures_present",
                "severity": "critical",
                "message": "Some candidates failed the integrity contract and were excluded from recommendation.",
            }
        )
    if failed_candidate_count:
        warnings.append(
            {
                "code": "candidate_execution_failures",
                "severity": "warning",
                "message": "Some candidates failed during execution and were excluded from ranking.",
            }
        )

    return {
        "summary": {
            "candidate_count": candidate_count,
            "successful_candidate_count": candidate_count - failed_candidate_count,
            "failed_candidate_count": failed_candidate_count,
            "low_coverage_candidate_count": low_coverage_count,
            "majority_dominance_candidate_count": majority_dominance_count,
            "one_class_candidate_count": one_class_candidate_count,
            "undefined_metric_candidate_count": undefined_metric_candidate_count,
            "runtime_feature_mismatch_candidate_count": runtime_mismatch_count,
            "integrity_failure_candidate_count": integrity_failure_count,
        },
        "warnings": warnings,
        "candidate_highlights": candidate_highlight_rows,
    }


def _build_integrity_from_metadata(
    *,
    metadata: Dict[str, Any],
    expected_feature_set_name: str | None = None,
    expected_feature_selection_mode: str | None = None,
    expected_target_spec_id: str | None = None,
    fallback_fold_count: int = 0,
) -> Dict[str, Any]:
    raw_fold_rows = metadata.get("integrity_fold_rows") or []
    if not raw_fold_rows:
        return _build_missing_integrity(
            expected_feature_set_name=expected_feature_set_name,
            expected_feature_selection_mode=expected_feature_selection_mode,
            expected_target_spec_id=expected_target_spec_id,
            fallback_fold_count=fallback_fold_count,
        )

    fold_rows: List[Dict[str, Any]] = []
    invalid_fold_count = 0
    total_purged_train_rows = 0
    total_purged_validation_rows = 0
    for raw_row in raw_fold_rows:
        row = dict(raw_row or {})
        status = str(row.get("status") or "passed")
        failure_reason = str(row.get("failure_reason") or "").strip()
        if status != "passed" and not failure_reason:
            failure_reason = "invalid_fold"
        if status != "passed":
            invalid_fold_count += 1
        row["status"] = status
        row["failure_reason"] = failure_reason
        total_purged_train_rows += _safe_int(row.get("purged_train_rows"))
        total_purged_validation_rows += _safe_int(row.get("purged_validation_rows"))
        fold_rows.append(row)

    horizon_bars = _safe_int(metadata.get("horizon_bars"))
    purge_required = bool(metadata.get("purge_required")) or horizon_bars > 0
    feature_selection_mode = str(metadata.get("feature_selection_mode") or "")
    research_feature_set_name = str(
        metadata.get("research_feature_set_name")
        or metadata.get("feature_set_name")
        or ""
    )
    target_spec_id = str(metadata.get("target_spec_id") or metadata.get("target_column") or "")
    contract_failures: List[str] = []
    warnings: List[Dict[str, Any]] = []

    if invalid_fold_count > 0:
        warnings.append(
            {
                "code": "invalid_integrity_folds",
                "severity": "critical",
                "message": "One or more folds failed the integrity contract.",
            }
        )
        contract_failures.append("invalid_folds_present")

    if purge_required and total_purged_train_rows <= 0:
        warnings.append(
            {
                "code": "missing_required_train_purge",
                "severity": "critical",
                "message": "Future-looking labels require purging train rows, but no purged train rows were recorded.",
            }
        )
        contract_failures.append("missing_required_train_purge")

    if purge_required and total_purged_validation_rows <= 0:
        warnings.append(
            {
                "code": "missing_required_validation_purge",
                "severity": "critical",
                "message": "Future-looking labels require purging validation rows, but no purged validation rows were recorded.",
            }
        )
        contract_failures.append("missing_required_validation_purge")

    if expected_feature_set_name and research_feature_set_name and research_feature_set_name != expected_feature_set_name:
        warnings.append(
            {
                "code": "feature_set_contract_mismatch",
                "severity": "critical",
                "message": "The saved experiment feature set does not match the expected integrity contract.",
            }
        )
        contract_failures.append("feature_set_contract_mismatch")

    if (
        expected_feature_selection_mode
        and feature_selection_mode
        and feature_selection_mode != expected_feature_selection_mode
    ):
        warnings.append(
            {
                "code": "feature_selection_mode_mismatch",
                "severity": "critical",
                "message": "The saved feature-selection mode does not match the expected integrity contract.",
            }
        )
        contract_failures.append("feature_selection_mode_mismatch")

    if expected_target_spec_id and target_spec_id and target_spec_id != expected_target_spec_id:
        warnings.append(
            {
                "code": "target_spec_contract_mismatch",
                "severity": "critical",
                "message": "The saved target specification does not match the expected integrity contract.",
            }
        )
        contract_failures.append("target_spec_contract_mismatch")

    fold_count = len({str(row.get("fold_name") or "") for row in fold_rows if str(row.get("fold_name") or "")}) or fallback_fold_count
    integrity_contract_ok = not contract_failures
    proof_status = "passed" if integrity_contract_ok else "failed"
    overview = {
        "target_spec_id": target_spec_id,
        "horizon_bars": horizon_bars,
        "purge_required": bool(purge_required),
        "feature_selection_mode": feature_selection_mode,
        "research_feature_set_name": research_feature_set_name,
        "fold_count": int(fold_count),
        "invalid_fold_count": int(invalid_fold_count),
        "total_purged_train_rows": int(total_purged_train_rows),
        "total_purged_validation_rows": int(total_purged_validation_rows),
        "contract_failure_reasons": list(dict.fromkeys(contract_failures)),
    }
    return {
        "overview": overview,
        "warnings": warnings,
        "fold_rows": fold_rows,
        "proof_status": proof_status,
        "integrity_contract_ok": bool(integrity_contract_ok),
    }


def _normalize_stored_integrity(
    *,
    stored_integrity: Dict[str, Any],
    expected_feature_set_name: str | None = None,
    expected_feature_selection_mode: str | None = None,
    expected_target_spec_id: str | None = None,
    fallback_fold_count: int = 0,
) -> Dict[str, Any]:
    overview = dict(stored_integrity.get("overview") or {})
    warnings = list(stored_integrity.get("warnings") or [])
    fold_rows = list(stored_integrity.get("fold_rows") or [])
    proof_status = str(stored_integrity.get("proof_status") or "").strip().lower()
    integrity_contract_ok = bool(stored_integrity.get("integrity_contract_ok"))

    if expected_feature_set_name and overview.get("research_feature_set_name") not in (None, "", expected_feature_set_name):
        warnings.append(
            {
                "code": "feature_set_contract_mismatch",
                "severity": "critical",
                "message": "The saved experiment feature set does not match the expected integrity contract.",
            }
        )
        integrity_contract_ok = False
    if (
        expected_feature_selection_mode
        and overview.get("feature_selection_mode") not in (None, "", expected_feature_selection_mode)
    ):
        warnings.append(
            {
                "code": "feature_selection_mode_mismatch",
                "severity": "critical",
                "message": "The saved feature-selection mode does not match the expected integrity contract.",
            }
        )
        integrity_contract_ok = False
    if expected_target_spec_id and overview.get("target_spec_id") not in (None, "", expected_target_spec_id):
        warnings.append(
            {
                "code": "target_spec_contract_mismatch",
                "severity": "critical",
                "message": "The saved target specification does not match the expected integrity contract.",
            }
        )
        integrity_contract_ok = False

    overview.setdefault("fold_count", int(fallback_fold_count))
    overview.setdefault("invalid_fold_count", 0)
    overview.setdefault("horizon_bars", 0)
    overview.setdefault("purge_required", False)
    overview.setdefault("total_purged_train_rows", 0)
    overview.setdefault("total_purged_validation_rows", 0)
    overview.setdefault("contract_failure_reasons", [])
    if proof_status not in {"passed", "failed", "missing"}:
        proof_status = "passed" if integrity_contract_ok else "failed"
    return {
        "overview": overview,
        "warnings": warnings,
        "fold_rows": fold_rows,
        "proof_status": proof_status,
        "integrity_contract_ok": bool(integrity_contract_ok),
    }


def _build_missing_integrity(
    *,
    expected_feature_set_name: str | None = None,
    expected_feature_selection_mode: str | None = None,
    expected_target_spec_id: str | None = None,
    fallback_fold_count: int = 0,
) -> Dict[str, Any]:
    return {
        "overview": {
            "target_spec_id": str(expected_target_spec_id or ""),
            "horizon_bars": 0,
            "purge_required": False,
            "feature_selection_mode": str(expected_feature_selection_mode or ""),
            "research_feature_set_name": str(expected_feature_set_name or ""),
            "fold_count": int(fallback_fold_count),
            "invalid_fold_count": int(fallback_fold_count or 0),
            "total_purged_train_rows": 0,
            "total_purged_validation_rows": 0,
            "contract_failure_reasons": ["missing_integrity_proof"],
        },
        "warnings": [
            {
                "code": "missing_integrity_proof",
                "severity": "critical",
                "message": "No saved integrity proof is available for this report.",
            }
        ],
        "fold_rows": [],
        "proof_status": "missing",
        "integrity_contract_ok": False,
    }


def _build_fold_target_balance_rows(
    splits: List[ResearchSplit],
    target_series: pd.Series,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in splits:
        for split_segment, start, end in (
            ("train", split.train_start, split.train_end),
            ("validation", split.validation_start, split.validation_end),
            ("test", split.test_start, split.test_end),
        ):
            segment_target = target_series.iloc[start:end]
            summary = _summarize_target_balance(segment_target)
            summary.update(
                {
                    "fold_name": split.name,
                    "split_segment": split_segment,
                    "total_rows": int(len(segment_target)),
                }
            )
            rows.append(summary)
    return rows


def _build_feature_health_rows(
    *,
    feature_frame: pd.DataFrame,
    splits: List[ResearchSplit],
    experiment_result: ExperimentResult,
) -> List[Dict[str, Any]]:
    candidate_columns = list(experiment_result.feature_columns)
    selection_by_fold = {
        str(row.get("fold_name")): row
        for row in (experiment_result.metadata or {}).get("fold_feature_selections", [])
    }
    rows: List[Dict[str, Any]] = []
    for split in splits:
        train_features = feature_frame.iloc[split.train_start:split.train_end].loc[:, candidate_columns]
        selection_row = selection_by_fold.get(split.name, {})
        selected_columns = [
            str(column)
            for column in (selection_row.get("selected_columns") or candidate_columns)
            if str(column) in train_features.columns
        ]
        selected_frame = train_features.loc[:, selected_columns] if selected_columns else pd.DataFrame(index=train_features.index)
        nan_rates = selected_frame.isna().mean() if not selected_frame.empty else pd.Series(dtype="float64")
        rows.append(
            {
                "fold_name": split.name,
                "selector_name": selection_row.get("selector_name"),
                "candidate_feature_count": int(len(candidate_columns)),
                "selected_feature_count": int(len(selected_columns)),
                "constant_selected_feature_count": int(_count_constant_features(selected_frame)),
                "near_constant_selected_feature_count": int(_count_near_constant_features(selected_frame)),
                "selected_nan_rate_mean": float(nan_rates.mean()) if not nan_rates.empty else 0.0,
                "selected_nan_rate_max": float(nan_rates.max()) if not nan_rates.empty else 0.0,
                "selected_columns": list(selected_columns),
            }
        )
    return rows


def _build_prediction_health_rows(
    *,
    prediction_rows: pd.DataFrame,
    threshold_rows: pd.DataFrame,
    trainer_name: str,
    selected_threshold: float | None,
) -> List[Dict[str, Any]]:
    if prediction_rows.empty:
        return []

    model_prediction_rows = prediction_rows.loc[prediction_rows.get("model_name") == trainer_name].copy()
    model_threshold_rows = threshold_rows.loc[threshold_rows.get("model_name") == trainer_name].copy() if not threshold_rows.empty else pd.DataFrame()
    if model_prediction_rows.empty:
        return []

    rows: List[Dict[str, Any]] = []
    grouped = model_prediction_rows.groupby(["fold_name", "split_segment"], dropna=False)
    for (fold_name, split_segment), group in grouped:
        threshold_match = pd.DataFrame()
        if selected_threshold is not None and not model_threshold_rows.empty:
            threshold_match = model_threshold_rows.loc[
                (model_threshold_rows["fold_name"] == fold_name)
                & (model_threshold_rows["split_segment"] == split_segment)
                & (pd.to_numeric(model_threshold_rows["threshold"], errors="coerce") == float(selected_threshold))
            ]
        threshold_row = threshold_match.iloc[0].to_dict() if not threshold_match.empty else {}
        probability = pd.to_numeric(group["probability"], errors="coerce")
        confidence = pd.to_numeric(group["confidence"], errors="coerce")
        rows.append(
            {
                "fold_name": str(fold_name),
                "split_segment": str(split_segment),
                "rows_scored": int(len(group)),
                "probability_mean": float(probability.mean()) if not probability.empty else 0.0,
                "probability_std": float(probability.std(ddof=0)) if len(probability) > 1 else 0.0,
                "probability_min": float(probability.min()) if not probability.empty else 0.0,
                "probability_max": float(probability.max()) if not probability.empty else 0.0,
                "confidence_mean": float(confidence.mean()) if not confidence.empty else 0.0,
                "confidence_std": float(confidence.std(ddof=0)) if len(confidence) > 1 else 0.0,
                "positive_prediction_rate": float(pd.to_numeric(group["prediction"], errors="coerce").mean()),
                "selected_threshold": _safe_float(selected_threshold),
                "selected_threshold_rows_kept": _safe_int(threshold_row.get("rows_kept")),
                "selected_threshold_coverage": _safe_float(threshold_row.get("coverage_rate")),
                "selected_threshold_f1": _safe_float(threshold_row.get("f1")),
                "undefined_selected_threshold_metrics": (
                    not threshold_row
                    or threshold_row.get("f1") in (None, "")
                ),
            }
        )
    return rows


def _build_baseline_context(experiment_result: ExperimentResult) -> Dict[str, Any]:
    baseline_comparison = experiment_result.baseline_comparison or {}
    baselines = baseline_comparison.get("baselines") or {}
    model_mean_test_accuracy = _safe_float(baseline_comparison.get("model_mean_test_accuracy"))
    majority_mean_test_accuracy = _safe_float((baselines.get("majority_class") or {}).get("mean_test_accuracy"))
    persistence_mean_test_accuracy = _safe_float((baselines.get("persistence") or {}).get("mean_test_accuracy"))
    return {
        "model_mean_test_accuracy": model_mean_test_accuracy,
        "majority_mean_test_accuracy": majority_mean_test_accuracy,
        "persistence_mean_test_accuracy": persistence_mean_test_accuracy,
        "majority_dominates": (
            model_mean_test_accuracy is not None
            and majority_mean_test_accuracy is not None
            and model_mean_test_accuracy < majority_mean_test_accuracy
        ),
        "majority_accuracy_gap": (
            float(majority_mean_test_accuracy - model_mean_test_accuracy)
            if model_mean_test_accuracy is not None and majority_mean_test_accuracy is not None
            else None
        ),
    }


def _build_training_warnings(highlights: Dict[str, Any], baseline_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    if bool(highlights.get("low_selected_threshold_test_coverage")):
        warnings.append(
            {
                "code": "low_selected_threshold_test_coverage",
                "severity": "warning",
                "message": "The selected threshold keeps too few test rows for a robust candidate claim.",
            }
        )
    if bool(highlights.get("broad_metric_under_majority")):
        warnings.append(
            {
                "code": "broad_metric_under_majority",
                "severity": "critical",
                "message": "Broad mean test accuracy is still below the majority baseline.",
            }
        )
    if int(highlights.get("one_class_fold_count") or 0) > 0:
        warnings.append(
            {
                "code": "one_class_folds_present",
                "severity": "warning",
                "message": "At least one train/validation/test fold segment is effectively one-class.",
            }
        )
    if int(highlights.get("undefined_selected_threshold_metric_count") or 0) > 0:
        warnings.append(
            {
                "code": "undefined_selected_threshold_metrics",
                "severity": "warning",
                "message": "Some selected-threshold metrics are undefined in the saved fold outputs.",
            }
        )
    if int(highlights.get("constant_feature_fold_count") or 0) > 0:
        warnings.append(
            {
                "code": "constant_selected_features_present",
                "severity": "warning",
                "message": "Some selected features are constant on at least one train fold.",
            }
        )
    if int(highlights.get("high_nan_feature_fold_count") or 0) > 0:
        warnings.append(
            {
                "code": "high_nan_rate_selected_features",
                "severity": "warning",
                "message": "Some selected features still have notable NaN rates on train folds before preprocessing.",
            }
        )
    if bool(baseline_context.get("majority_dominates")):
        warnings.append(
            {
                "code": "majority_baseline_dominance",
                "severity": "critical",
                "message": "The majority baseline remains stronger than the model on broad test accuracy.",
            }
        )
    return warnings


def _summarize_target_balance(target: pd.Series) -> Dict[str, Any]:
    numeric_target = pd.to_numeric(target, errors="coerce")
    scored = numeric_target.dropna()
    positive_count = int((scored == 1).sum())
    negative_count = int((scored == 0).sum())
    scored_rows = int(len(scored))
    missing_count = int(numeric_target.isna().sum())
    one_class_flag = scored_rows > 0 and int(scored.nunique()) <= 1
    return {
        "scored_rows": scored_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "missing_count": missing_count,
        "positive_rate": float(positive_count / scored_rows) if scored_rows else 0.0,
        "negative_rate": float(negative_count / scored_rows) if scored_rows else 0.0,
        "missing_rate": float(missing_count / len(numeric_target)) if len(numeric_target) else 0.0,
        "one_class_flag": bool(one_class_flag),
    }


def _frame_from_artifact_rows(experiment_result: ExperimentResult, key: str) -> pd.DataFrame:
    artifact_metadata = ((experiment_result.prediction_artifacts or [None])[0] or PredictionArtifactProxy()).metadata
    rows = artifact_metadata.get(key) or []
    return pd.DataFrame(rows)


def _count_constant_features(feature_frame: pd.DataFrame) -> int:
    if feature_frame.empty:
        return 0
    constant_count = 0
    for column in feature_frame.columns:
        series = feature_frame[column]
        if int(series.nunique(dropna=False)) <= 1:
            constant_count += 1
    return constant_count


def _count_near_constant_features(feature_frame: pd.DataFrame) -> int:
    if feature_frame.empty:
        return 0
    near_constant_count = 0
    for column in feature_frame.columns:
        series = feature_frame[column]
        non_null = series.dropna()
        if non_null.empty:
            near_constant_count += 1
            continue
        value_frequencies = non_null.value_counts(normalize=True, dropna=False)
        max_frequency = float(value_frequencies.max()) if not value_frequencies.empty else 0.0
        unique_count = int(non_null.nunique())
        if unique_count > 1 and max_frequency >= NEAR_CONSTANT_FEATURE_RATIO:
            near_constant_count += 1
    return near_constant_count


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class PredictionArtifactProxy:
    """Minimal proxy used when prediction artifacts are absent."""

    metadata: Dict[str, Any] = {}
