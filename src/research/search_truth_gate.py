"""Truth-gate helpers for bounded Stage 5 search recommendations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable

from .defaults import get_builtin_research_defaults
from .schemas import SearchCandidateSummary

MAJORITY_BASELINE_NAME = "majority_class"


def apply_truth_gate(
    candidate: SearchCandidateSummary,
    truth_gate_defaults: Dict[str, Any] | None = None,
) -> SearchCandidateSummary:
    """Annotate a Stage 5 candidate with diagnostics and truth-gate status."""
    resolved_defaults = _resolve_truth_gate_defaults(truth_gate_defaults)
    minimum_test_coverage = float(resolved_defaults.get("minimum_test_coverage") or 0.0)
    max_validation_test_f1_drift = float(resolved_defaults.get("max_validation_test_f1_drift") or 0.0)
    validation_summary = dict(candidate.validation_summary or {})
    test_summary = dict(candidate.test_summary or {})
    diagnostics: Dict[str, Any] = dict(candidate.diagnostics or {})
    integrity = dict(diagnostics.get("integrity") or {})
    truth_gate_diagnostics: Dict[str, Any] = {}
    failures: list[str] = []
    integrity_failures: list[str] = []

    validation_mean_f1 = _to_float_or_none(validation_summary.get("mean_f1"))
    test_mean_f1 = _to_float_or_none(test_summary.get("mean_f1"))
    test_mean_coverage = _to_float_or_none(test_summary.get("mean_coverage"))
    overall_mean_test_accuracy = _to_float_or_none(candidate.overall_mean_test_accuracy)
    majority_baseline_mean_test_accuracy = _to_float_or_none(candidate.majority_baseline_mean_test_accuracy)
    validation_fold_count = int(float(validation_summary.get("fold_count") or 0.0))
    test_fold_count = int(float(test_summary.get("fold_count") or 0.0))
    expected_fold_count = int(candidate.expected_fold_count or 0)

    if not integrity:
        integrity_failures.append("missing_integrity_proof")
    else:
        if not bool(integrity.get("integrity_contract_ok")):
            integrity_failures.extend(
                list((integrity.get("overview") or {}).get("contract_failure_reasons") or [])
                or ["integrity_contract_failed"]
            )
        if str(integrity.get("proof_status") or "") == "missing":
            integrity_failures.append("missing_integrity_proof")
        if int(((integrity.get("overview") or {}).get("invalid_fold_count") or 0)) > 0:
            integrity_failures.append("invalid_integrity_folds")
        if bool((integrity.get("overview") or {}).get("purge_required")):
            if int(((integrity.get("overview") or {}).get("total_purged_train_rows") or 0)) <= 0:
                integrity_failures.append("missing_required_train_purge")
            if int(((integrity.get("overview") or {}).get("total_purged_validation_rows") or 0)) <= 0:
                integrity_failures.append("missing_required_validation_purge")
    if integrity_failures:
        truth_gate_diagnostics["integrity"] = {
            "proof_status": integrity.get("proof_status") if integrity else "missing",
            "failure_reasons": list(dict.fromkeys(integrity_failures)),
        }
        failures.extend(integrity_failures)

    if not candidate.passed_test_guardrail:
        failures.append("failed_test_guardrail")

    if validation_mean_f1 is None or test_mean_f1 is None:
        truth_gate_diagnostics["undefined_selected_threshold_f1"] = True
        failures.append("undefined_selected_threshold_metrics")

    if test_mean_coverage is None:
        truth_gate_diagnostics["missing_test_coverage"] = True
        failures.append("missing_test_coverage")
    elif test_mean_coverage < minimum_test_coverage:
        truth_gate_diagnostics["low_test_coverage"] = {
            "observed": float(test_mean_coverage),
            "minimum_required": minimum_test_coverage,
        }
        failures.append("low_test_coverage")

    if overall_mean_test_accuracy is None or majority_baseline_mean_test_accuracy is None:
        truth_gate_diagnostics["missing_broad_metric_baseline"] = True
        failures.append("missing_broad_metric_baseline")
    elif overall_mean_test_accuracy < majority_baseline_mean_test_accuracy:
        truth_gate_diagnostics["broad_metric_under_majority"] = {
            "candidate_mean_test_accuracy": float(overall_mean_test_accuracy),
            "majority_baseline_mean_test_accuracy": float(majority_baseline_mean_test_accuracy),
        }
        failures.append("broad_metric_under_majority")

    if validation_mean_f1 is not None and test_mean_f1 is not None:
        drift = float(validation_mean_f1 - test_mean_f1)
        truth_gate_diagnostics["validation_test_f1_drift"] = drift
        if drift > max_validation_test_f1_drift:
            failures.append("validation_test_drift")

    if expected_fold_count and (validation_fold_count < expected_fold_count or test_fold_count < expected_fold_count):
        truth_gate_diagnostics["degenerate_fold_warning"] = {
            "expected_fold_count": expected_fold_count,
            "validation_selected_threshold_fold_count": validation_fold_count,
            "test_selected_threshold_fold_count": test_fold_count,
        }

    if not candidate.runtime_feature_contract_ok:
        truth_gate_diagnostics["runtime_feature_contract_mismatch"] = True
        failures.append("runtime_feature_contract_mismatch")

    diagnostics["truth_gate"] = truth_gate_diagnostics
    candidate.diagnostics = diagnostics
    candidate.truth_gate_failures = list(dict.fromkeys(failures))
    candidate.passed_truth_gate = not candidate.truth_gate_failures
    return candidate


def build_truth_gate_summary(
    candidates: Iterable[SearchCandidateSummary],
    truth_gate_defaults: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build aggregate gate statistics for a search run."""
    resolved_defaults = _resolve_truth_gate_defaults(truth_gate_defaults)
    candidate_list = list(candidates)
    failure_counts: Dict[str, int] = {}
    for candidate in candidate_list:
        for failure in candidate.truth_gate_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1

    return {
        "candidate_count": len(candidate_list),
        "passed_test_guardrail_count": sum(1 for candidate in candidate_list if candidate.passed_test_guardrail),
        "passed_truth_gate_count": sum(1 for candidate in candidate_list if candidate.passed_truth_gate),
        "failure_counts": failure_counts,
        "minimum_test_coverage": float(resolved_defaults.get("minimum_test_coverage") or 0.0),
        "max_validation_test_f1_drift": float(resolved_defaults.get("max_validation_test_f1_drift") or 0.0),
        "broad_metric_baseline_name": MAJORITY_BASELINE_NAME,
    }


def _resolve_truth_gate_defaults(truth_gate_defaults: Dict[str, Any] | None) -> Dict[str, Any]:
    if truth_gate_defaults is not None:
        return dict(truth_gate_defaults)
    return asdict(get_builtin_research_defaults().truth_gate)


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
